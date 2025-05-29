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

#include "c63.h"
#include "quantdct.h"
#include "common.h"
#include "me.h"
#include "tables.h"

#include <cuda_runtime.h>
#include <sisci_error.h>
#include <sisci_api.h>

// Simplified frame slot - single state enum instead of multiple booleans
typedef enum {
    SLOT_EMPTY = 0,
    SLOT_RECEIVED,      // Has YUV data, ready to process
    SLOT_PROCESSED,     // Encoded, ready to send
    SLOT_SENT          // Sent to client, ready for reuse
} slot_state_t;

typedef struct {
    yuv_t image;
    int frame_number;
    slot_state_t state;
    char *encoded_data;
    size_t encoded_size;
} frame_slot_t;

// Simplified frame manager - much cleaner than original
typedef struct {
    frame_slot_t slots[MAX_FRAMES];
    int head, tail;          // Simple circular buffer
    int count;               // Number of frames in pipeline
    int next_frame_number;   // Frame counter
    bool quit_requested;
    pthread_mutex_t mutex;
    pthread_cond_t not_empty, not_full;
} frame_manager_t;

// Global state
static struct {
    uint32_t width, height;
    uint32_t remote_node;
    struct c63_common *cm;
    frame_manager_t frame_mgr;
    
    // SISCI resources
    volatile struct recv_segment *recv_seg;
    volatile struct send_segment *send_seg;
    volatile struct send_segment *client_send;
    volatile struct recv_segment *client_recv;
    sci_dma_queue_t dma_queue;
    sci_local_segment_t recv_segment, send_segment;
    sci_remote_segment_t remote_client_send, remote_client_recv;
} g;

// Initialize frame manager
void frame_manager_init(frame_manager_t *mgr) {
    memset(mgr, 0, sizeof(frame_manager_t));
    
    // Allocate encoded data buffers
    for (int i = 0; i < MAX_FRAMES; i++) {
        mgr->slots[i].encoded_data = (char*)malloc(MESSAGE_SIZE);
        mgr->slots[i].state = SLOT_EMPTY;
    }
    
    mgr->head = 0;
    mgr->tail = 0;
    mgr->count = 0;
    mgr->next_frame_number = 0;
    mgr->quit_requested = false;
    
    pthread_mutex_init(&mgr->mutex, NULL);
    pthread_cond_init(&mgr->not_empty, NULL);
    pthread_cond_init(&mgr->not_full, NULL);
}

// Cleanup frame manager
void frame_manager_destroy(frame_manager_t *mgr) {
    pthread_mutex_lock(&mgr->mutex);
    
    for (int i = 0; i < MAX_FRAMES; i++) {
        free(mgr->slots[i].encoded_data);
        
        // Clean up CUDA memory
        if (mgr->slots[i].image.Y) cudaFree(mgr->slots[i].image.Y);
        if (mgr->slots[i].image.U) cudaFree(mgr->slots[i].image.U);
        if (mgr->slots[i].image.V) cudaFree(mgr->slots[i].image.V);
    }
    
    pthread_mutex_unlock(&mgr->mutex);
    
    pthread_mutex_destroy(&mgr->mutex);
    pthread_cond_destroy(&mgr->not_empty);
    pthread_cond_destroy(&mgr->not_full);
}

// Add frame data - simplified
bool frame_manager_add_frame(frame_manager_t *mgr, yuv_t *image) {
    pthread_mutex_lock(&mgr->mutex);
    
    // Wait for space if full
    while (mgr->count == MAX_FRAMES && !mgr->quit_requested) {
        pthread_cond_wait(&mgr->not_full, &mgr->mutex);
    }
    
    if (mgr->quit_requested) {
        pthread_mutex_unlock(&mgr->mutex);
        return false;
    }
    
    frame_slot_t *slot = &mgr->slots[mgr->tail];
    
    // Allocate CUDA memory if needed
    if (!slot->image.Y) {
        cudaMallocManaged((void**)&slot->image.Y, g.cm->padw[Y_COMPONENT] * g.cm->padh[Y_COMPONENT]);
        cudaMallocManaged((void**)&slot->image.U, g.cm->padw[U_COMPONENT] * g.cm->padh[U_COMPONENT]);
        cudaMallocManaged((void**)&slot->image.V, g.cm->padw[V_COMPONENT] * g.cm->padh[V_COMPONENT]);
    }
    
    // Copy frame data
    size_t y_size = g.width * g.height;
    size_t u_size = (g.width * g.height) / 4;
    
    memcpy(slot->image.Y, image->Y, y_size);
    memcpy(slot->image.U, image->U, u_size);
    memcpy(slot->image.V, image->V, u_size);
    
    // Set metadata
    slot->frame_number = mgr->next_frame_number++;
    slot->state = SLOT_RECEIVED;
    
    mgr->tail = (mgr->tail + 1) % MAX_FRAMES;
    mgr->count++;
    
    printf("Server: Received frame %d in slot %d\n", slot->frame_number, 
           (mgr->tail - 1 + MAX_FRAMES) % MAX_FRAMES);
    
    pthread_cond_signal(&mgr->not_empty);
    pthread_mutex_unlock(&mgr->mutex);
    return true;
}

// Get next frame to work on - simplified interface
frame_slot_t* frame_manager_get_next_frame(frame_manager_t *mgr) {
    pthread_mutex_lock(&mgr->mutex);
    
    while (mgr->count == 0 && !mgr->quit_requested) {
        pthread_cond_wait(&mgr->not_empty, &mgr->mutex);
    }
    
    if (mgr->count == 0) {
        pthread_mutex_unlock(&mgr->mutex);
        return NULL;
    }
    
    frame_slot_t *slot = &mgr->slots[mgr->head];
    pthread_mutex_unlock(&mgr->mutex);
    return slot;
}

// Mark frame as completed and remove from pipeline
void frame_manager_frame_done(frame_manager_t *mgr) {
    pthread_mutex_lock(&mgr->mutex);
    
    if (mgr->count > 0) {
        frame_slot_t *slot = &mgr->slots[mgr->head];
        slot->state = SLOT_EMPTY;
        
        printf("Server: Frame %d completed\n", slot->frame_number);
        
        mgr->head = (mgr->head + 1) % MAX_FRAMES;
        mgr->count--;
        
        pthread_cond_signal(&mgr->not_full);
    }
    
    pthread_mutex_unlock(&mgr->mutex);
}

// Request quit
void frame_manager_request_quit(frame_manager_t *mgr) {
    pthread_mutex_lock(&mgr->mutex);
    mgr->quit_requested = true;
    pthread_cond_broadcast(&mgr->not_empty);
    pthread_cond_broadcast(&mgr->not_full);
    pthread_mutex_unlock(&mgr->mutex);
}

// Initialize encoder
struct c63_common *init_encoder(int width, int height) {
    struct c63_common *cm = (struct c63_common*)calloc(1, sizeof(struct c63_common));
    
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
    
    for (int i = 0; i < 64; ++i) {
        cm->quanttbl[Y_COMPONENT][i] = yquanttbl_def[i] / (cm->qp / 10.0);
        cm->quanttbl[U_COMPONENT][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
        cm->quanttbl[V_COMPONENT][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
    }
    
    return cm;
}

// Encode frame - same logic but cleaner
void encode_frame(struct c63_common *cm, frame_slot_t *slot) {
    printf("Server: Encoding frame %d\n", slot->frame_number);
    
    // Update frame references
    destroy_frame(cm->refframe);
    cm->refframe = cm->curframe;
    cm->curframe = create_frame(cm, &slot->image);
    
    // Determine if keyframe
    if (cm->framenum == 0 || cm->frames_since_keyframe == cm->keyframe_interval) {
        cm->curframe->keyframe = 1;
        cm->frames_since_keyframe = 0;
    } else {
        cm->curframe->keyframe = 0;
    }
    
    // Motion estimation for non-keyframes
    if (!cm->curframe->keyframe) {
        c63_motion_estimate(cm);
        c63_motion_compensate(cm);
    }
    
    // DCT and quantization
    dct_quantize(slot->image.Y, cm->curframe->predicted->Y, cm->padw[Y_COMPONENT],
                 cm->padh[Y_COMPONENT], cm->curframe->residuals->Ydct, cm->quanttbl[Y_COMPONENT]);
    dct_quantize(slot->image.U, cm->curframe->predicted->U, cm->padw[U_COMPONENT],
                 cm->padh[U_COMPONENT], cm->curframe->residuals->Udct, cm->quanttbl[U_COMPONENT]);
    dct_quantize(slot->image.V, cm->curframe->predicted->V, cm->padw[V_COMPONENT],
                 cm->padh[V_COMPONENT], cm->curframe->residuals->Vdct, cm->quanttbl[V_COMPONENT]);
    
    // Dequantization and IDCT for reconstruction
    dequantize_idct(cm->curframe->residuals->Ydct, cm->curframe->predicted->Y, cm->ypw, cm->yph, 
                    cm->curframe->recons->Y, cm->quanttbl[Y_COMPONENT]);
    dequantize_idct(cm->curframe->residuals->Udct, cm->curframe->predicted->U, cm->upw, cm->uph, 
                    cm->curframe->recons->U, cm->quanttbl[U_COMPONENT]);
    dequantize_idct(cm->curframe->residuals->Vdct, cm->curframe->predicted->V, cm->vpw, cm->vph, 
                    cm->curframe->recons->V, cm->quanttbl[V_COMPONENT]);
    
    // Pack encoded data
    char *ptr = slot->encoded_data;
    
    // Keyframe flag
    *((int*)ptr) = cm->curframe->keyframe;
    ptr += sizeof(int);
    
    // DCT coefficients
    size_t ydct_size = cm->ypw * cm->yph * sizeof(int16_t);
    memcpy(ptr, cm->curframe->residuals->Ydct, ydct_size);
    ptr += ydct_size;
    
    size_t udct_size = cm->upw * cm->uph * sizeof(int16_t);
    memcpy(ptr, cm->curframe->residuals->Udct, udct_size);
    ptr += udct_size;
    
    size_t vdct_size = cm->vpw * cm->vph * sizeof(int16_t);
    memcpy(ptr, cm->curframe->residuals->Vdct, vdct_size);
    ptr += vdct_size;
    
    // Macroblock data  
    size_t mby_size = cm->mb_rows * cm->mb_cols * sizeof(struct macroblock);
    memcpy(ptr, cm->curframe->mbs[Y_COMPONENT], mby_size);
    ptr += mby_size;
    
    size_t mbu_size = (cm->mb_rows/2) * (cm->mb_cols/2) * sizeof(struct macroblock);
    memcpy(ptr, cm->curframe->mbs[U_COMPONENT], mbu_size);
    ptr += mbu_size;
    
    size_t mbv_size = (cm->mb_rows/2) * (cm->mb_cols/2) * sizeof(struct macroblock);
    memcpy(ptr, cm->curframe->mbs[V_COMPONENT], mbv_size);
    
    slot->encoded_size = sizeof(int) + ydct_size + udct_size + vdct_size + 
                        mby_size + mbu_size + mbv_size;
    
    // Update frame counters
    cm->framenum++;
    cm->frames_since_keyframe++;
    if (cm->curframe->keyframe) {
        cm->frames_since_keyframe = 0;
    }
    
    printf("Server: Frame %d encoded (%zu bytes, %s)\n", 
           slot->frame_number, slot->encoded_size, 
           cm->curframe->keyframe ? "keyframe" : "non-keyframe");
}

// Send encoded frame to client
bool send_encoded_frame(frame_slot_t *slot) {
    sci_error_t error;
    
    printf("Server: Sending frame %d (%zu bytes)\n", 
           slot->frame_number, slot->encoded_size);
    
    // Copy to send buffer
    memcpy((void*)g.send_seg->message_buffer, slot->encoded_data, slot->encoded_size);
    
    // DMA transfer to client
    SCIStartDmaTransfer(g.dma_queue, g.send_segment, g.remote_client_recv,
                       offsetof(struct send_segment, message_buffer),
                       slot->encoded_size,
                       offsetof(struct recv_segment, message_buffer),
                       NO_CALLBACK, NULL, NO_FLAGS, &error);
    
    if (error == SCI_ERR_OK) {
        SCIWaitForDMAQueue(g.dma_queue, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
    }
    
    if (error != SCI_ERR_OK) {
        printf("Server: DMA transfer failed for frame %d\n", slot->frame_number);
        return false;
    }
    
    // Signal client
    SCIFlush(NULL, NO_FLAGS);
    g.client_recv->packet.data_size = slot->encoded_size;
    g.client_recv->packet.cmd = CMD_ENCODED_DATA;
    SCIFlush(NULL, NO_FLAGS);
    
    // Wait for acknowledgment
    time_t start = time(NULL);
    while (g.send_seg->packet.cmd != CMD_ENCODED_DATA_ACK) {
        if (time(NULL) - start > TIMEOUT_SECONDS) {
            printf("Server: Timeout waiting for ACK for frame %d\n", slot->frame_number);
            return false;
        }
        usleep(1000);
    }
    
    printf("Server: Frame %d acknowledged\n", slot->frame_number);
    g.send_seg->packet.cmd = CMD_INVALID;
    return true;
}

// Simplified single worker thread - processes and sends frames
void *worker_thread(void *arg) {
    printf("Server: Worker thread started\n");
    
    while (true) {
        frame_slot_t *slot = frame_manager_get_next_frame(&g.frame_mgr);
        if (!slot) break;
        
        // Process frame based on current state
        switch (slot->state) {
            case SLOT_RECEIVED:
                // Encode the frame
                encode_frame(g.cm, slot);
                slot->state = SLOT_PROCESSED;
                // Don't advance pipeline yet - still need to send
                break;
                
            case SLOT_PROCESSED:
                // Send encoded frame
                if (send_encoded_frame(slot)) {
                    slot->state = SLOT_SENT;
                }
                // Complete the frame (advance pipeline)
                frame_manager_frame_done(&g.frame_mgr);
                break;
                
            default:
                // Shouldn't happen, but handle gracefully
                frame_manager_frame_done(&g.frame_mgr);
                break;
        }
    }
    
    printf("Server: Worker thread finished\n");
    return NULL;
}

// Main processing loop - simplified
int main_loop() {
    yuv_t temp_image = {0};
    pthread_t worker_tid;
    bool thread_started = false;
    int running = 1;
    
    printf("Server: Waiting for commands...\n");
    
    while (running) {
        // Wait for command
        while (g.recv_seg->packet.cmd == CMD_INVALID) {
            usleep(1000);
        }
        
        uint32_t cmd = g.recv_seg->packet.cmd;
        g.recv_seg->packet.cmd = CMD_INVALID;
        
        switch (cmd) {
            case CMD_DIMENSIONS: {
                struct dimensions_data dim_data;
                memcpy(&dim_data, (const void*)g.recv_seg->message_buffer, sizeof(dim_data));
                
                printf("Server: Received dimensions %ux%u\n", dim_data.width, dim_data.height);
                
                g.width = dim_data.width;
                g.height = dim_data.height;
                
                // Initialize encoder and frame manager
                g.cm = init_encoder(g.width, g.height);
                frame_manager_init(&g.frame_mgr);
                
                // Initialize thread pools
                thread_pool_init();
                task_pool_init(g.cm->padh[Y_COMPONENT]);
                
                // Allocate temporary image
                cudaMallocManaged((void**)&temp_image.Y, g.cm->padw[Y_COMPONENT] * g.cm->padh[Y_COMPONENT]);
                cudaMallocManaged((void**)&temp_image.U, g.cm->padw[U_COMPONENT] * g.cm->padh[U_COMPONENT]);
                cudaMallocManaged((void**)&temp_image.V, g.cm->padw[V_COMPONENT] * g.cm->padh[V_COMPONENT]);
                
                // Start single worker thread
                pthread_create(&worker_tid, NULL, worker_thread, NULL);
                thread_started = true;
                
                // Send acknowledgment
                sci_error_t error;
                memcpy((void*)g.send_seg->message_buffer, &dim_data, sizeof(dim_data));
                SCIStartDmaTransfer(g.dma_queue, g.send_segment, g.remote_client_recv,
                                   offsetof(struct send_segment, message_buffer),
                                   sizeof(dim_data),
                                   offsetof(struct recv_segment, message_buffer),
                                   NO_CALLBACK, NULL, NO_FLAGS, &error);
                SCIWaitForDMAQueue(g.dma_queue, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
                
                SCIFlush(NULL, NO_FLAGS);
                g.client_recv->packet.cmd = CMD_DIMENSIONS_ACK;
                SCIFlush(NULL, NO_FLAGS);
                
                printf("Server: Dimensions acknowledged, ready for frames\n");
                break;
            }
            
            case CMD_YUV_DATA: {
                if (!g.cm) {
                    printf("Server: ERROR - YUV data before dimensions\n");
                    break;
                }
                
                size_t data_size = g.recv_seg->packet.data_size;
                size_t expected_size = g.width * g.height * 1.5;
                
                if (data_size == expected_size) {
                    // Copy YUV data
                    size_t y_size = g.width * g.height;
                    size_t u_size = (g.width * g.height) / 4;
                    
                    memcpy(temp_image.Y, (const void*)g.recv_seg->message_buffer, y_size);
                    memcpy(temp_image.U, (const void*)(g.recv_seg->message_buffer + y_size), u_size);
                    memcpy(temp_image.V, (const void*)(g.recv_seg->message_buffer + y_size + u_size), u_size);
                    
                    frame_manager_add_frame(&g.frame_mgr, &temp_image);
                } else {
                    printf("Server: Invalid YUV data size: %zu (expected %zu)\n", 
                           data_size, expected_size);
                }
                
                // Send acknowledgment
                SCIFlush(NULL, NO_FLAGS);
                g.client_recv->packet.cmd = CMD_YUV_DATA_ACK;
                SCIFlush(NULL, NO_FLAGS);
                break;
            }
            
            case CMD_QUIT:
                printf("Server: Quit command received\n");
                running = 0;
                
                if (thread_started) {
                    frame_manager_request_quit(&g.frame_mgr);
                    pthread_join(worker_tid, NULL);
                    
                    printf("Server: Final statistics:\n");
                    printf("  Frames processed: %d\n", g.frame_mgr.next_frame_number);
                }
                
                // Cleanup
                if (g.cm) {
                    if (temp_image.Y) cudaFree(temp_image.Y);
                    if (temp_image.U) cudaFree(temp_image.U);
                    if (temp_image.V) cudaFree(temp_image.V);
                    
                    frame_manager_destroy(&g.frame_mgr);
                    free(g.cm);
                    task_pool_destroy();
                    thread_pool_destroy();
                }
                break;
                
            default:
                printf("Server: Unknown command: %d\n", cmd);
                break;
        }
    }
    
    return g.frame_mgr.next_frame_number;
}

// Initialize SISCI - same as before
bool init_sisci() {
    sci_error_t error;
    unsigned int localAdapterNo = 0;
    
    SCIInitialize(NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    sci_desc_t sd;
    SCIOpen(&sd, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    // Create server segments
    SCICreateSegment(sd, &g.recv_segment, SEGMENT_SERVER_RECV, sizeof(struct recv_segment),
                     NO_CALLBACK, NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    SCICreateSegment(sd, &g.send_segment, SEGMENT_SERVER_SEND, sizeof(struct send_segment),
                     NO_CALLBACK, NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    // Prepare segments
    SCIPrepareSegment(g.recv_segment, localAdapterNo, NO_FLAGS, &error);
    SCIPrepareSegment(g.send_segment, localAdapterNo, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    // Make segments available
    SCISetSegmentAvailable(g.recv_segment, localAdapterNo, NO_FLAGS, &error);
    SCISetSegmentAvailable(g.send_segment, localAdapterNo, NO_FLAGS, &error);
    
    // Create DMA queue
    SCICreateDMAQueue(sd, &g.dma_queue, localAdapterNo, 1, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    // Map local segments
    sci_map_t recv_map, send_map;
    g.recv_seg = (volatile struct recv_segment *)SCIMapLocalSegment(
        g.recv_segment, &recv_map, 0, sizeof(struct recv_segment), NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    g.send_seg = (volatile struct send_segment *)SCIMapLocalSegment(
        g.send_segment, &send_map, 0, sizeof(struct send_segment), NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    g.recv_seg->packet.cmd = CMD_INVALID;
    g.send_seg->packet.cmd = CMD_INVALID;
    
    printf("Server: Waiting for client connection...\n");
    
    // Connect to client segments
    do {
        SCIConnectSegment(sd, &g.remote_client_send, g.remote_node, SEGMENT_CLIENT_SEND,
                         localAdapterNo, NO_CALLBACK, NULL, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
    } while (error != SCI_ERR_OK);
    
    do {
        SCIConnectSegment(sd, &g.remote_client_recv, g.remote_node, SEGMENT_CLIENT_RECV,
                         localAdapterNo, NO_CALLBACK, NULL, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
    } while (error != SCI_ERR_OK);
    
    // Map client segments
    sci_map_t client_send_map, client_recv_map;
    g.client_send = (volatile struct send_segment *)SCIMapRemoteSegment(
        g.remote_client_send, &client_send_map, 0, sizeof(struct send_segment), NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    g.client_recv = (volatile struct recv_segment *)SCIMapRemoteSegment(
        g.remote_client_recv, &client_recv_map, 0, sizeof(struct recv_segment), NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    printf("Server: Connected to client, ready for processing\n");
    return true;
}

int main(int argc, char **argv) {
    int c;
    
    while ((c = getopt(argc, argv, "r:")) != -1) {
        switch (c) {
            case 'r':
                g.remote_node = atoi(optarg);
                break;
            default:
                break;
        }
    }
    
    if (g.remote_node == 0) {
        fprintf(stderr, "Remote node-id not specified. Use -r <remote node-id>\n");
        exit(EXIT_FAILURE);
    }
    
    if (!init_sisci()) {
        fprintf(stderr, "SISCI initialization failed\n");
        exit(EXIT_FAILURE);
    }
    
    int frames_processed = main_loop();
    
    printf("Server: Processed %d frames, exiting\n", frames_processed);
    
    return 0;
}