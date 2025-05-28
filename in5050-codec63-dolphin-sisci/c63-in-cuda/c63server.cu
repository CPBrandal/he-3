#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <stdbool.h>
#include <unistd.h>
#include <time.h>

#include "c63.h"
#include "quantdct.h"
#include "common.h"
#include "me.h"
#include "tables.h"

#include <cuda_runtime.h>
#include <sisci_error.h>
#include <sisci_api.h>

static uint32_t width = 0;
static uint32_t height = 0;

// Frame processing structures with improved pipeline management
typedef struct {
    yuv_t image;
    int frame_number;
    bool valid;
    bool processed;
    bool sent;
    size_t encoded_size;
    char *encoded_data;
    time_t receive_time;
    time_t process_start_time;
    time_t process_end_time;
} frame_slot_t;

#define MAX_PENDING_FRAMES 3
#define ENCODED_DATA_BUFFER_SIZE (8*1024 * 1024)

typedef struct {
    frame_slot_t frames[MAX_PENDING_FRAMES];
    int next_receive_index;
    int next_process_index;
    int next_send_index;
    int frames_received;
    int frames_processed;
    int frames_sent;
    pthread_mutex_t mutex;
    pthread_cond_t frame_received;
    pthread_cond_t frame_processed;
    bool quit_requested;
} frame_manager_t;

// SISCI resources for 4-segment design
typedef struct {
    // Server's own segments
    volatile struct recv_segment *recv_seg;      // Server receives YUV data here
    volatile struct send_segment *send_seg;      // Server sends encoded data from here
    
    // Client's segments (mapped remotely)
    volatile struct send_segment *client_send;   // Client sends YUV data (server reads)
    volatile struct recv_segment *client_recv;   // Client receives encoded data (server writes)
    
    sci_dma_queue_t dma_queue;
    sci_local_segment_t recv_segment;
    sci_local_segment_t send_segment;
    sci_remote_segment_t remote_client_send;
    sci_remote_segment_t remote_client_recv;
} sisci_resources_t;

// Global variables
static frame_manager_t frame_mgr;
static struct c63_common *g_cm = NULL;
static sisci_resources_t g_sisci;

/* getopt */
extern int optind;
extern char *optarg;

struct c63_common *init_c63_enc(int width, int height)
{
    int i;
    c63_common *cm = (c63_common *)calloc(1, sizeof(struct c63_common));

    cm->width = width;
    cm->height = height;

    cm->padw[Y_COMPONENT] = cm->ypw = (uint32_t)(ceil(width / 16.0f) * 16);
    cm->padh[Y_COMPONENT] = cm->yph = (uint32_t)(ceil(height / 16.0f) * 16);
    cm->padw[U_COMPONENT] = cm->upw = (uint32_t)(ceil(width * UX / (YX * 8.0f)) * 8);
    cm->padh[U_COMPONENT] = cm->uph = (uint32_t)(ceil(height * UY / (YY * 8.0f)) * 8);
    cm->padw[V_COMPONENT] = cm->vpw = (uint32_t)(ceil(width * VX / (YX * 8.0f)) * 8);
    cm->padh[V_COMPONENT] = cm->vph = (uint32_t)(ceil(height * VY / (YY * 8.0f)) * 8);

    cm->mb_cols = cm->ypw / 8;
    cm->mb_rows = cm->yph / 8;

    cm->qp = 25;
    cm->me_search_range = 16;
    cm->keyframe_interval = 100;

    for (i = 0; i < 64; ++i)
    {
        cm->quanttbl[Y_COMPONENT][i] = yquanttbl_def[i] / (cm->qp / 10.0);
        cm->quanttbl[U_COMPONENT][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
        cm->quanttbl[V_COMPONENT][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
    }

    return cm;
}

void free_c63_enc(struct c63_common *cm)
{
    destroy_frame(cm->curframe);
    free(cm);
}

// Enhanced frame manager functions
void frame_manager_init(frame_manager_t *mgr) {
    memset(mgr, 0, sizeof(frame_manager_t));
    
    for (int i = 0; i < MAX_PENDING_FRAMES; i++) {
        mgr->frames[i].encoded_data = (char*)malloc(ENCODED_DATA_BUFFER_SIZE);
        mgr->frames[i].valid = false;
        mgr->frames[i].processed = false;
        mgr->frames[i].sent = false;
    }
    
    mgr->next_receive_index = 0;
    mgr->next_process_index = 0;
    mgr->next_send_index = 0;
    
    pthread_mutex_init(&mgr->mutex, NULL);
    pthread_cond_init(&mgr->frame_received, NULL);
    pthread_cond_init(&mgr->frame_processed, NULL);
}

void frame_manager_destroy(frame_manager_t *mgr) {
    for (int i = 0; i < MAX_PENDING_FRAMES; i++) {
        if (mgr->frames[i].encoded_data) {
            printf("a");
            free(mgr->frames[i].encoded_data);
            printf("b");
            mgr->frames[i].encoded_data = NULL;
        }
        
        // Safe CUDA memory cleanup
        if (mgr->frames[i].image.Y) {
            cudaError_t err = cudaFree(mgr->frames[i].image.Y);
            if (err != cudaSuccess) {
                printf("Warning: cudaFree Y failed for slot %d: %s\n", i, cudaGetErrorString(err));
            }
            mgr->frames[i].image.Y = NULL;
        }
        if (mgr->frames[i].image.U) {
            cudaError_t err = cudaFree(mgr->frames[i].image.U);  
            if (err != cudaSuccess) {
                printf("Warning: cudaFree U failed for slot %d: %s\n", i, cudaGetErrorString(err));
            }
            mgr->frames[i].image.U = NULL;
        }
        if (mgr->frames[i].image.V) {
            cudaError_t err = cudaFree(mgr->frames[i].image.V);
            if (err != cudaSuccess) {
                printf("Warning: cudaFree V failed for slot %d: %s\n", i, cudaGetErrorString(err));
            }
            mgr->frames[i].image.V = NULL;
        }
        
        mgr->frames[i].valid = false;
        mgr->frames[i].processed = false;
        mgr->frames[i].sent = false;
    }
    
    pthread_mutex_destroy(&mgr->mutex);
    pthread_cond_destroy(&mgr->frame_received);
    pthread_cond_destroy(&mgr->frame_processed);
}

// Add frame to next available slot
bool frame_manager_add_frame(frame_manager_t *mgr, yuv_t *image) {
    pthread_mutex_lock(&mgr->mutex);
    
    int index = mgr->next_receive_index;
    frame_slot_t *slot = &mgr->frames[index];
    
    // Check if this slot is still busy (not sent yet)
    if (slot->valid && !slot->sent) {
        printf("Server: Warning - Receive slot %d still busy, dropping frame\n", index);
        pthread_mutex_unlock(&mgr->mutex);
        return false;
    }
    
    // Initialize CUDA memory if needed
    if (!slot->image.Y) {
        cudaError_t err1 = cudaMallocManaged((void**)&slot->image.Y, g_cm->padw[Y_COMPONENT] * g_cm->padh[Y_COMPONENT] * sizeof(uint8_t));
        cudaError_t err2 = cudaMallocManaged((void**)&slot->image.U, g_cm->padw[U_COMPONENT] * g_cm->padh[U_COMPONENT] * sizeof(uint8_t));
        cudaError_t err3 = cudaMallocManaged((void**)&slot->image.V, g_cm->padw[V_COMPONENT] * g_cm->padh[V_COMPONENT] * sizeof(uint8_t));
        
        if (err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess) {
            printf("Server: Error allocating CUDA memory for slot %d\n", index);
            if (slot->image.Y) { cudaFree(slot->image.Y); slot->image.Y = NULL; }
            if (slot->image.U) { cudaFree(slot->image.U); slot->image.U = NULL; }
            if (slot->image.V) { cudaFree(slot->image.V); slot->image.V = NULL; }
            pthread_mutex_unlock(&mgr->mutex);
            return false;
        }
    }
    
    size_t y_size = width * height;
    size_t u_size = (width * height) / 4;
    size_t v_size = (width * height) / 4;
    
    // Copy frame data
    memcpy(slot->image.Y, image->Y, y_size);
    memcpy(slot->image.U, image->U, u_size);
    memcpy(slot->image.V, image->V, v_size);
    
    // Set frame metadata
    slot->frame_number = mgr->frames_received;
    slot->valid = true;
    slot->processed = false;
    slot->sent = false;
    slot->receive_time = time(NULL);
    
    mgr->frames_received++;
    mgr->next_receive_index = (mgr->next_receive_index + 1) % MAX_PENDING_FRAMES;
    
    printf("Server: Stored frame %d in slot %d (pipeline: recv=%d, proc=%d, sent=%d)\n", 
           slot->frame_number, index, mgr->frames_received, mgr->frames_processed, mgr->frames_sent);
    
    pthread_cond_signal(&mgr->frame_received);
    pthread_mutex_unlock(&mgr->mutex);
    
    return true;
}

// Get next frame to process (blocks until available)
frame_slot_t *frame_manager_get_next_to_process(frame_manager_t *mgr) {
    pthread_mutex_lock(&mgr->mutex);
    
    while (true) {
        // Search through ALL slots for the next unprocessed frame
        frame_slot_t *oldest_frame = NULL;
        int oldest_frame_number = INT_MAX;
        
        for (int i = 0; i < MAX_PENDING_FRAMES; i++) {
            frame_slot_t *slot = &mgr->frames[i];
            if (slot->valid && !slot->processed) {
                if (slot->frame_number < oldest_frame_number) {
                    oldest_frame = slot;
                    oldest_frame_number = slot->frame_number;
                }
            }
        }
        
        if (oldest_frame) {
            oldest_frame->process_start_time = time(NULL);
            printf("Server: Processing frame %d from slot %d\n", 
                   oldest_frame->frame_number, 
                   oldest_frame - mgr->frames);
            pthread_mutex_unlock(&mgr->mutex);
            return oldest_frame;
        }
        
        if (mgr->quit_requested) {
            pthread_mutex_unlock(&mgr->mutex);
            return NULL;
        }
        
        printf("Server: No frames ready for processing, waiting...\n");
        pthread_cond_wait(&mgr->frame_received, &mgr->mutex);
    }
}

// Mark frame as processed
void frame_manager_mark_processed(frame_manager_t *mgr, frame_slot_t *slot) {
    pthread_mutex_lock(&mgr->mutex);
    
    slot->processed = true;
    slot->process_end_time = time(NULL);
    mgr->frames_processed++;
    mgr->next_process_index = (mgr->next_process_index + 1) % MAX_PENDING_FRAMES;
    
    printf("Server: Marked frame %d as processed (processing time: %ld seconds)\n", 
           slot->frame_number, slot->process_end_time - slot->process_start_time);
    
    pthread_cond_signal(&mgr->frame_processed);
    pthread_mutex_unlock(&mgr->mutex);
}

// Get next frame to send (blocks until available)
frame_slot_t *frame_manager_get_next_to_send(frame_manager_t *mgr) {
    pthread_mutex_lock(&mgr->mutex);
    
    while (true) {
        int index = mgr->next_send_index;
        frame_slot_t *slot = &mgr->frames[index];
        
        if (slot->valid && slot->processed && !slot->sent) {
            printf("Server: Sending frame %d from slot %d\n", slot->frame_number, index);
            pthread_mutex_unlock(&mgr->mutex);
            return slot;
        }
        
        if (mgr->quit_requested && mgr->frames_sent >= mgr->frames_processed) {
            pthread_mutex_unlock(&mgr->mutex);
            return NULL;
        }
        
        pthread_cond_wait(&mgr->frame_processed, &mgr->mutex);
    }
}

// Mark frame as sent and free the slot
void frame_manager_mark_sent(frame_manager_t *mgr, frame_slot_t *slot) {
    pthread_mutex_lock(&mgr->mutex);
    
    slot->sent = true;
    slot->valid = false;  // Free the slot for reuse
    mgr->frames_sent++;
    mgr->next_send_index = (mgr->next_send_index + 1) % MAX_PENDING_FRAMES;
    
    time_t total_time = time(NULL) - slot->receive_time;
    printf("Server: Marked frame %d as sent (total pipeline time: %ld seconds)\n", 
           slot->frame_number, total_time);
    
    pthread_mutex_unlock(&mgr->mutex);
}

// Request shutdown
void frame_manager_request_quit(frame_manager_t *mgr) {
    pthread_mutex_lock(&mgr->mutex);
    mgr->quit_requested = true;
    pthread_cond_broadcast(&mgr->frame_received);
    pthread_cond_broadcast(&mgr->frame_processed);
    pthread_mutex_unlock(&mgr->mutex);
}

// Encode frame into slot buffer
void encode_frame_to_buffer(struct c63_common *cm, frame_slot_t *slot) {
    printf("Server: Encoding frame %d\n", slot->frame_number);
    
    destroy_frame(cm->refframe);
    cm->refframe = cm->curframe;
    cm->curframe = create_frame(cm, &slot->image);
    
    if (cm->framenum == 0 || cm->frames_since_keyframe == cm->keyframe_interval) {
        cm->curframe->keyframe = 1;
        cm->frames_since_keyframe = 0;
        printf("Server: Frame %d is a keyframe\n", cm->framenum);
    } else {
        cm->curframe->keyframe = 0;
        printf("Server: Frame %d is not a keyframe\n", cm->framenum);
    }
    
    if (!cm->curframe->keyframe) {
        printf("Server: Motion estimation for frame %d\n", slot->frame_number);
        c63_motion_estimate(cm);
        c63_motion_compensate(cm);
        printf("Server: Motion estimation complete for frame %d\n", slot->frame_number);
    }
    
    printf("Server: DCT/quantization for frame %d\n", slot->frame_number);
    dct_quantize(slot->image.Y, cm->curframe->predicted->Y, cm->padw[Y_COMPONENT],
               cm->padh[Y_COMPONENT], cm->curframe->residuals->Ydct, cm->quanttbl[Y_COMPONENT]);
    dct_quantize(slot->image.U, cm->curframe->predicted->U, cm->padw[U_COMPONENT],
               cm->padh[U_COMPONENT], cm->curframe->residuals->Udct, cm->quanttbl[U_COMPONENT]);
    dct_quantize(slot->image.V, cm->curframe->predicted->V, cm->padw[V_COMPONENT],
               cm->padh[V_COMPONENT], cm->curframe->residuals->Vdct, cm->quanttbl[V_COMPONENT]);
    
    printf("Server: Dequantization/IDCT for frame %d\n", slot->frame_number);
    dequantize_idct(cm->curframe->residuals->Ydct, cm->curframe->predicted->Y, cm->ypw, cm->yph, cm->curframe->recons->Y, cm->quanttbl[Y_COMPONENT]);
    dequantize_idct(cm->curframe->residuals->Udct, cm->curframe->predicted->U, cm->upw, cm->uph, cm->curframe->recons->U, cm->quanttbl[U_COMPONENT]);
    dequantize_idct(cm->curframe->residuals->Vdct, cm->curframe->predicted->V, cm->vpw, cm->vph, cm->curframe->recons->V, cm->quanttbl[V_COMPONENT]);
    
    // Pack encoded data
    char* ptr = slot->encoded_data;
    
    *((int*)ptr) = cm->curframe->keyframe;
    ptr += sizeof(int);
    
    memcpy(ptr, cm->curframe->residuals->Ydct, cm->ypw * cm->yph * sizeof(int16_t));
    ptr += cm->ypw * cm->yph * sizeof(int16_t);
    
    memcpy(ptr, cm->curframe->residuals->Udct, cm->upw * cm->uph * sizeof(int16_t));
    ptr += cm->upw * cm->uph * sizeof(int16_t);
    
    memcpy(ptr, cm->curframe->residuals->Vdct, cm->vpw * cm->vph * sizeof(int16_t));
    ptr += cm->vpw * cm->vph * sizeof(int16_t);
    
    memcpy(ptr, cm->curframe->mbs[Y_COMPONENT], cm->mb_rows * cm->mb_cols * sizeof(struct macroblock));
    ptr += cm->mb_rows * cm->mb_cols * sizeof(struct macroblock);
    
    memcpy(ptr, cm->curframe->mbs[U_COMPONENT], (cm->mb_rows/2) * (cm->mb_cols/2) * sizeof(struct macroblock));
    ptr += (cm->mb_rows/2) * (cm->mb_cols/2) * sizeof(struct macroblock);
    
    memcpy(ptr, cm->curframe->mbs[V_COMPONENT], (cm->mb_rows/2) * (cm->mb_cols/2) * sizeof(struct macroblock));
    
    slot->encoded_size = sizeof(int) + 
                        (cm->ypw * cm->yph * sizeof(int16_t)) + 
                        (cm->upw * cm->uph * sizeof(int16_t)) + 
                        (cm->vpw * cm->vph * sizeof(int16_t)) + 
                        (cm->mb_rows * cm->mb_cols * sizeof(struct macroblock)) + 
                        ((cm->mb_rows/2) * (cm->mb_cols/2) * sizeof(struct macroblock)) + 
                        ((cm->mb_rows/2) * (cm->mb_cols/2) * sizeof(struct macroblock));
    
    printf("Server: Frame %d encoded, size: %zu bytes\n", slot->frame_number, slot->encoded_size);
    
    cm->framenum++;
    cm->frames_since_keyframe++;
    if (cm->curframe->keyframe) {
        cm->frames_since_keyframe = 0;
    }
}

// Processing thread
void *processing_thread(void *arg) {
    printf("Server: Processing thread started\n");
    
    while (true) {
        frame_slot_t *slot = frame_manager_get_next_to_process(&frame_mgr);
        if (!slot) {
            printf("Server: Processing thread exiting\n");
            break;
        }
        
        encode_frame_to_buffer(g_cm, slot);
        frame_manager_mark_processed(&frame_mgr, slot);
    }
    
    return NULL;
}

// Sending thread
void *sending_thread(void *arg) {
    printf("Server: Sending thread started\n");
    sci_error_t error;
    
    while (true) {
        frame_slot_t *slot = frame_manager_get_next_to_send(&frame_mgr);
        if (!slot) {
            printf("Server: Sending thread exiting\n");
            break;
        }
        
        printf("Server: Sending encoded data for frame %d\n", slot->frame_number);
        
        if (slot->encoded_size > MESSAGE_SIZE) {
            fprintf(stderr, "Server: ERROR - Encoded data size (%zu) exceeds buffer size\n", slot->encoded_size);
            frame_manager_mark_sent(&frame_mgr, slot);
            continue;
        }
        
        // Copy to send segment buffer
        memcpy((void*)g_sisci.send_seg->message_buffer, slot->encoded_data, slot->encoded_size);
        
        // DMA transfer to client's receive segment
        SCIStartDmaTransfer(g_sisci.dma_queue, 
                           g_sisci.send_segment,
                           g_sisci.remote_client_recv,
                           offsetof(struct send_segment, message_buffer),
                           slot->encoded_size,
                           offsetof(struct recv_segment, message_buffer),
                           NO_CALLBACK, NULL, NO_FLAGS, &error);
        
        if (error != SCI_ERR_OK) {
            fprintf(stderr, "Server: Error in encoded data DMA transfer: 0x%x\n", error);
            frame_manager_mark_sent(&frame_mgr, slot);
            continue;
        }
        
        SCIWaitForDMAQueue(g_sisci.dma_queue, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
        
        // Signal client
        SCIFlush(NULL, NO_FLAGS);
        g_sisci.client_recv->packet.data_size = slot->encoded_size;
        g_sisci.client_recv->packet.cmd = CMD_ENCODED_DATA;
        SCIFlush(NULL, NO_FLAGS);
        
        printf("Server: Signaled client about encoded data for frame %d\n", slot->frame_number);
        
        // Wait for ACK
        time_t start_time = time(NULL);
        bool timeout = false;
        
        while (g_sisci.send_seg->packet.cmd != CMD_ENCODED_DATA_ACK && !timeout) {
            if (time(NULL) - start_time > 30) {
                timeout = true;
                fprintf(stderr, "Server: Timeout waiting for encoded data ACK\n");
            }
            usleep(1000);
        }
        
        if (!timeout) {
            printf("Server: Client acknowledged frame %d\n", slot->frame_number);
            g_sisci.send_seg->packet.cmd = CMD_INVALID;
        }
        
        frame_manager_mark_sent(&frame_mgr, slot);
    }
    
    return NULL;
}

// Main loop
int main_loop(sci_desc_t sd)
{
    int running = 1;
    uint32_t cmd;
    sci_error_t error;
    pthread_t processing_tid, sending_tid;
    yuv_t temp_image = {0};
    bool threads_started = false;
    
    printf("Server: Waiting for commands...\n");

    while(running)
    {
        // Wait for command from client on receive segment
        while(g_sisci.recv_seg->packet.cmd == CMD_INVALID) {
            // Small delay to prevent busy waiting
            usleep(1000);
        }

        cmd = g_sisci.recv_seg->packet.cmd;
        g_sisci.recv_seg->packet.cmd = CMD_INVALID;
        
        switch(cmd) {
            case CMD_DIMENSIONS:
            {
                struct dimensions_data dim_data;
                memcpy(&dim_data, (const void*)g_sisci.recv_seg->message_buffer, sizeof(struct dimensions_data));
                
                printf("Server: Received dimensions (width=%u, height=%u)\n", 
                       dim_data.width, dim_data.height);
                
                width = dim_data.width;
                height = dim_data.height;
                
                g_cm = init_c63_enc(width, height);
                frame_manager_init(&frame_mgr);
                
                thread_pool_init();
                task_pool_init(g_cm->padh[Y_COMPONENT]);
                
                cudaError_t err1 = cudaMallocManaged((void**)&temp_image.Y, g_cm->padw[Y_COMPONENT] * g_cm->padh[Y_COMPONENT] * sizeof(uint8_t));
                cudaError_t err2 = cudaMallocManaged((void**)&temp_image.U, g_cm->padw[U_COMPONENT] * g_cm->padh[U_COMPONENT] * sizeof(uint8_t));
                cudaError_t err3 = cudaMallocManaged((void**)&temp_image.V, g_cm->padw[V_COMPONENT] * g_cm->padh[V_COMPONENT] * sizeof(uint8_t));

                if (err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess) {
                    fprintf(stderr, "Server: Failed to allocate CUDA memory for temp image\n");
                    if (temp_image.Y) cudaFree(temp_image.Y);
                    if (temp_image.U) cudaFree(temp_image.U);
                    if (temp_image.V) cudaFree(temp_image.V);
                    temp_image.Y = temp_image.U = temp_image.V = NULL;
                    break;
                }
                
                // Start processing threads
                pthread_create(&processing_tid, NULL, processing_thread, NULL);
                pthread_create(&sending_tid, NULL, sending_thread, NULL);
                threads_started = true;
                
                // Send ACK through send segment to client's receive segment
                memcpy((void*)g_sisci.send_seg->message_buffer, &dim_data, sizeof(struct dimensions_data));
                SCIStartDmaTransfer(g_sisci.dma_queue, g_sisci.send_segment, g_sisci.remote_client_recv,
                                   offsetof(struct send_segment, message_buffer),
                                   sizeof(struct dimensions_data),
                                   offsetof(struct recv_segment, message_buffer),
                                   NO_CALLBACK, NULL, NO_FLAGS, &error);
                SCIWaitForDMAQueue(g_sisci.dma_queue, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
                
                SCIFlush(NULL, NO_FLAGS);
                g_sisci.client_recv->packet.cmd = CMD_DIMENSIONS_ACK;
                SCIFlush(NULL, NO_FLAGS);
                
                printf("Server: Dimensions acknowledged, pipeline ready with %d frame slots\n", MAX_PENDING_FRAMES);
                break;
            }
                
            case CMD_YUV_DATA:
            {
                if (g_cm == NULL) {
                    fprintf(stderr, "Server: ERROR - Received YUV data before dimensions!\n");
                    break;
                }
                
                size_t data_size = g_sisci.recv_seg->packet.data_size;
                size_t expected_size = (width * height) + ((width * height) / 2);
                
                if (data_size == expected_size) {
                    printf("Server: Received YUV frame with size %zu bytes\n", data_size);
                    
                    size_t y_size = width * height;
                    size_t u_size = (width * height) / 4;
                    size_t v_size = (width * height) / 4;
                    
                    memcpy(temp_image.Y, (const void*)g_sisci.recv_seg->message_buffer, y_size);
                    memcpy(temp_image.U, (const void*)(g_sisci.recv_seg->message_buffer + y_size), u_size);
                    memcpy(temp_image.V, (const void*)(g_sisci.recv_seg->message_buffer + y_size + u_size), v_size);
                    
                    if (frame_manager_add_frame(&frame_mgr, &temp_image)) {
                        printf("Server: Frame added to processing pipeline\n");
                    } else {
                        printf("Server: Warning - Pipeline full, dropping frame\n");
                    }
                    
                    // Send ACK back through send segment
                    SCIFlush(NULL, NO_FLAGS);
                    g_sisci.client_recv->packet.cmd = CMD_YUV_DATA_ACK;
                    SCIFlush(NULL, NO_FLAGS);
                } else {
                    fprintf(stderr, "Server: ERROR - Received unexpected data size %zu, expected %zu\n", 
                           data_size, expected_size);
                    
                    SCIFlush(NULL, NO_FLAGS);
                    g_sisci.client_recv->packet.cmd = CMD_YUV_DATA_ACK; 
                    SCIFlush(NULL, NO_FLAGS);
                }
                break;
            }
            
            case CMD_QUIT:
                printf("Server: Received quit command\n");
                running = 0;
                
                if (threads_started) {
                    printf("Server: Waiting for pipeline to finish processing remaining frames...\n");
                    
                    // Don't immediately signal quit - let threads finish current work
                    // Wait a reasonable time for in-flight frames to complete
                    int wait_count = 0;
                    while (frame_mgr.frames_processed < frame_mgr.frames_received && wait_count < 100) {
                        printf("Server: Still processing - Received: %d, Processed: %d, Sent: %d\n",
                               frame_mgr.frames_received, frame_mgr.frames_processed, frame_mgr.frames_sent);
                        usleep(100000); // Wait 100ms
                        wait_count++;
                    }
                    
                    // Now signal threads to quit
                    frame_manager_request_quit(&frame_mgr);
                    
                    pthread_join(processing_tid, NULL);
                    pthread_join(sending_tid, NULL);
                    
                    printf("Server: Final pipeline statistics:\n");
                    printf("  Frames received: %d\n", frame_mgr.frames_received);
                    printf("  Frames processed: %d\n", frame_mgr.frames_processed);
                    printf("  Frames sent: %d\n", frame_mgr.frames_sent);
                    
                    if (frame_mgr.frames_sent != frame_mgr.frames_received) {
                        printf("  WARNING: Not all received frames were sent back!\n");
                    }
                }
                
                if (g_cm) {
                        // Safer CUDA cleanup
                        printf("aaaaaa");
                        if (temp_image.Y) {
                            cudaError_t err = cudaFree(temp_image.Y);
                            if (err != cudaSuccess) {
                                printf("Warning: cudaFree temp_image.Y failed: %s\n", cudaGetErrorString(err));
                            }
                        }
                        if (temp_image.U) {
                            cudaError_t err = cudaFree(temp_image.U);
                            if (err != cudaSuccess) {
                                printf("Warning: cudaFree temp_image.U failed: %s\n", cudaGetErrorString(err));
                            }
                        }
                        if (temp_image.V) {
                            cudaError_t err = cudaFree(temp_image.V);
                            if (err != cudaSuccess) {
                                printf("Warning: cudaFree temp_image.V failed: %s\n", cudaGetErrorString(err));
                            }
                        }
                        printf("1");
                        frame_manager_destroy(&frame_mgr);
                        printf("2");
                        free_c63_enc(g_cm);
                        printf("3");
                        task_pool_destroy();
                        thread_pool_destroy();
                    }
                break;
                
            default:
                printf("Server: Unknown command: %d\n", cmd);
                break;
        }
    }

    return frame_mgr.frames_processed;
}

int main(int argc, char **argv)
{
    unsigned int localAdapterNo = 0;
    unsigned int remoteNodeId = 0;
    sci_error_t error;
    sci_desc_t sd;
    sci_map_t recv_map, send_map, client_send_map, client_recv_map;
    int c;

    while ((c = getopt(argc, argv, "r:")) != -1)
    {
        switch (c)
        {
            case 'r':
                remoteNodeId = atoi(optarg);
                break;
            default:
                break;
        }
    }

    if (remoteNodeId == 0) {
        fprintf(stderr, "Remote node-id is not specified. Use -r <remote node-id>\n");
        exit(EXIT_FAILURE);
    }

    // Initialize SISCI
    SCIInitialize(NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCIInitialize failed - Error code: 0x%x\n", error);
        exit(EXIT_FAILURE);
    }

    SCIOpen(&sd, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCIOpen failed - Error code 0x%x\n", error);
        SCITerminate();
        exit(EXIT_FAILURE);
    }

    // Create server's receive segment (for YUV data from client)
    SCICreateSegment(sd, &g_sisci.recv_segment, SEGMENT_SERVER_RECV, sizeof(struct recv_segment),
                     NO_CALLBACK, NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCICreateSegment (recv) failed - Error code 0x%x\n", error);
        exit(EXIT_FAILURE);
    }

    // Create server's send segment (for encoded data to client)
    SCICreateSegment(sd, &g_sisci.send_segment, SEGMENT_SERVER_SEND, sizeof(struct send_segment),
                     NO_CALLBACK, NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCICreateSegment (send) failed - Error code 0x%x\n", error);
        exit(EXIT_FAILURE);
    }

    // Prepare segments
    SCIPrepareSegment(g_sisci.recv_segment, localAdapterNo, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCIPrepareSegment (recv) failed - Error code 0x%x\n", error);
        exit(EXIT_FAILURE);
    }

    SCIPrepareSegment(g_sisci.send_segment, localAdapterNo, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCIPrepareSegment (send) failed - Error code 0x%x\n", error);
        exit(EXIT_FAILURE);
    }

    // Make segments available
    SCISetSegmentAvailable(g_sisci.recv_segment, localAdapterNo, NO_FLAGS, &error);
    SCISetSegmentAvailable(g_sisci.send_segment, localAdapterNo, NO_FLAGS, &error);

    // Create DMA queue
    SCICreateDMAQueue(sd, &g_sisci.dma_queue, localAdapterNo, 1, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCICreateDMAQueue failed - Error code 0x%x\n", error);
        exit(EXIT_FAILURE);
    }

    // Map local segments
    g_sisci.recv_seg = (volatile struct recv_segment *)SCIMapLocalSegment(
        g_sisci.recv_segment, &recv_map, 0, sizeof(struct recv_segment), NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCIMapLocalSegment (recv) failed - Error code 0x%x\n", error);
        exit(EXIT_FAILURE);
    }

    g_sisci.send_seg = (volatile struct send_segment *)SCIMapLocalSegment(
        g_sisci.send_segment, &send_map, 0, sizeof(struct send_segment), NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCIMapLocalSegment (send) failed - Error code 0x%x\n", error);
        exit(EXIT_FAILURE);
    }

    g_sisci.recv_seg->packet.cmd = CMD_INVALID;
    g_sisci.send_seg->packet.cmd = CMD_INVALID;

    printf("Server: Waiting for client connection...\n");

    // Connect to client's segments
    do {
        SCIConnectSegment(sd, &g_sisci.remote_client_send, remoteNodeId, SEGMENT_CLIENT_SEND, localAdapterNo,
                         NO_CALLBACK, NULL, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
    } while (error != SCI_ERR_OK);

    do {
        SCIConnectSegment(sd, &g_sisci.remote_client_recv, remoteNodeId, SEGMENT_CLIENT_RECV, localAdapterNo,
                         NO_CALLBACK, NULL, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
    } while (error != SCI_ERR_OK);

    printf("Server: Connected to client segments\n");

    // Map client's segments
    g_sisci.client_send = (volatile struct send_segment *)SCIMapRemoteSegment(
        g_sisci.remote_client_send, &client_send_map, 0, sizeof(struct send_segment), NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCIMapRemoteSegment (client send) failed - Error code 0x%x\n", error);
        exit(EXIT_FAILURE);
    }

    g_sisci.client_recv = (volatile struct recv_segment *)SCIMapRemoteSegment(
        g_sisci.remote_client_recv, &client_recv_map, 0, sizeof(struct recv_segment), NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCIMapRemoteSegment (client recv) failed - Error code 0x%x\n", error);
        exit(EXIT_FAILURE);
    }

    printf("Server: Ready for 3-frame pipeline processing\n");
    
    // Enter main loop
    main_loop(sd);

    printf("Server: Exiting\n");
    
    // Cleanup
    SCIUnmapSegment(client_recv_map, NO_FLAGS, &error);
    SCIUnmapSegment(client_send_map, NO_FLAGS, &error);
    SCIDisconnectSegment(g_sisci.remote_client_recv, NO_FLAGS, &error);
    SCIDisconnectSegment(g_sisci.remote_client_send, NO_FLAGS, &error);
    SCIUnmapSegment(send_map, NO_FLAGS, &error);
    SCIUnmapSegment(recv_map, NO_FLAGS, &error);
    SCIRemoveDMAQueue(g_sisci.dma_queue, NO_FLAGS, &error);
    SCISetSegmentUnavailable(g_sisci.send_segment, localAdapterNo, NO_FLAGS, &error);
    SCISetSegmentUnavailable(g_sisci.recv_segment, localAdapterNo, NO_FLAGS, &error);
    SCIRemoveSegment(g_sisci.send_segment, NO_FLAGS, &error);
    SCIRemoveSegment(g_sisci.recv_segment, NO_FLAGS, &error);
    SCIClose(sd, NO_FLAGS, &error);
    SCITerminate();

    return 0;
}