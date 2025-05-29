#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>
#include <stdbool.h>

#include <sisci_error.h>
#include <sisci_api.h>

#include "c63.h"
#include "c63_write.h"
#include "common.h"
#include "tables.h"

// Simplified frame buffer for better CPU utilization
typedef struct {
    yuv_t *yuv_data;
    char *encoded_data;
    size_t encoded_size;
    int frame_number;
    bool ready_to_send;
    bool result_received;
    bool keyframe;
} frame_buffer_t;

// Optimized pipeline with separate queues
typedef struct {
    frame_buffer_t frames[MAX_FRAMES];
    
    // Queue indices
    int send_idx;           // Next frame to send
    int result_idx;         // Next result to receive
    int write_idx;          // Next frame to write
    
    int frames_read;        // Total frames read from file
    int frames_sent;        // Total frames sent to server
    int frames_received;    // Total results received
    int frames_written;     // Total frames written to output
    
    bool finished_reading;
    
    pthread_mutex_t mutex;
    pthread_cond_t frames_available, results_available;
} pipeline_t;

// Global state
static struct {
    char *input_file, *output_file;
    uint32_t remote_node, width, height;
    int limit_frames;
    FILE *outfile;
    pipeline_t pipeline;
    struct c63_common *cm;
    
    // SISCI resources
    volatile struct send_segment *send_seg;
    volatile struct recv_segment *recv_seg;
    volatile struct recv_segment *server_recv;
    volatile struct send_segment *server_send;
    sci_dma_queue_t dma_queue;
    sci_local_segment_t send_segment, recv_segment;
    sci_remote_segment_t remote_server_recv, remote_server_send;
} g;

// Initialize pipeline
void pipeline_init(pipeline_t *p) {
    memset(p, 0, sizeof(pipeline_t));
    
    for (int i = 0; i < MAX_FRAMES; i++) {
        p->frames[i].encoded_data = (char*)malloc(MESSAGE_SIZE);
        p->frames[i].yuv_data = NULL;
        p->frames[i].ready_to_send = false;
        p->frames[i].result_received = false;
    }
    
    pthread_mutex_init(&p->mutex, NULL);
    pthread_cond_init(&p->frames_available, NULL);
    pthread_cond_init(&p->results_available, NULL);
}

// Cleanup pipeline
void pipeline_destroy(pipeline_t *p) {
    for (int i = 0; i < MAX_FRAMES; i++) {
        if (p->frames[i].yuv_data) {
            free(p->frames[i].yuv_data->Y);
            free(p->frames[i].yuv_data->U);
            free(p->frames[i].yuv_data->V);
            free(p->frames[i].yuv_data);
        }
        free(p->frames[i].encoded_data);
    }
    
    pthread_mutex_destroy(&p->mutex);
    pthread_cond_destroy(&p->frames_available);
    pthread_cond_destroy(&p->results_available);
}

// Read YUV frame
static yuv_t *read_yuv(FILE *file) {
    yuv_t *image = (yuv_t*)malloc(sizeof(*image));
    
    image->Y = (uint8_t*)calloc(1, g.cm->padw[Y_COMPONENT] * g.cm->padh[Y_COMPONENT]);
    image->U = (uint8_t*)calloc(1, g.cm->padw[U_COMPONENT] * g.cm->padh[U_COMPONENT]);
    image->V = (uint8_t*)calloc(1, g.cm->padw[V_COMPONENT] * g.cm->padh[V_COMPONENT]);
    
    size_t len = 0;
    len += fread(image->Y, 1, g.width * g.height, file);
    len += fread(image->U, 1, (g.width * g.height) / 4, file);
    len += fread(image->V, 1, (g.width * g.height) / 4, file);
    
    if (ferror(file) || feof(file) || len != g.width * g.height * 1.5) {
        free(image->Y);
        free(image->U);
        free(image->V);
        free(image);
        return NULL;
    }
    
    return image;
}

// Producer thread - reads frames and buffers them
void *producer_thread(void *arg) {
    FILE *infile = (FILE *)arg;
    
    printf("Producer: Starting to read frames\n");
    
    pthread_mutex_lock(&g.pipeline.mutex);
    
    while (true) {
        if (g.limit_frames && g.pipeline.frames_read >= g.limit_frames) {
            printf("Producer: Frame limit reached (%d frames)\n", g.pipeline.frames_read);
            break;
        }
        
        // Find next available slot
        int slot_idx = -1;
        for (int i = 0; i < MAX_FRAMES; i++) {
            if (!g.pipeline.frames[i].ready_to_send && !g.pipeline.frames[i].yuv_data) {
                slot_idx = i;
                break;
            }
        }
        
        if (slot_idx == -1) {
            // Wait for a slot to become available
            pthread_cond_wait(&g.pipeline.frames_available, &g.pipeline.mutex);
            continue;
        }
        
        pthread_mutex_unlock(&g.pipeline.mutex);
        
        // Read frame outside of lock
        yuv_t *image = read_yuv(infile);
        if (!image) {
            printf("Producer: End of file reached\n");
            pthread_mutex_lock(&g.pipeline.mutex);
            break;
        }
        
        pthread_mutex_lock(&g.pipeline.mutex);
        
        // Add frame to pipeline
        frame_buffer_t *frame = &g.pipeline.frames[slot_idx];
        frame->yuv_data = image;
        frame->frame_number = g.pipeline.frames_read++;
        frame->ready_to_send = true;
        frame->result_received = false;
        
        printf("Producer: Read frame %d into slot %d\n", frame->frame_number, slot_idx);
        
        pthread_cond_signal(&g.pipeline.frames_available);
    }
    
    g.pipeline.finished_reading = true;
    pthread_cond_broadcast(&g.pipeline.frames_available);
    pthread_cond_broadcast(&g.pipeline.results_available);
    pthread_mutex_unlock(&g.pipeline.mutex);
    
    printf("Producer: Finished reading %d frames\n", g.pipeline.frames_read);
    return NULL;
}

// Wait with timeout
bool wait_for_command(uint32_t expected_cmd, volatile struct recv_segment *seg, int timeout) {
    time_t start = time(NULL);
    while (seg->packet.cmd != expected_cmd) {
        if (time(NULL) - start > timeout) return false;
        usleep(1000);
    }
    return true;
}

// Send DMA data
bool send_dma_data(void *data, size_t size, size_t offset) {
    sci_error_t error;
    
    memcpy((void*)g.send_seg->message_buffer, data, size);
    
    SCIStartDmaTransfer(g.dma_queue, g.send_segment, g.remote_server_recv,
                       offsetof(struct send_segment, message_buffer),
                       size, offsetof(struct recv_segment, message_buffer) + offset,
                       NO_CALLBACK, NULL, NO_FLAGS, &error);
    
    if (error != SCI_ERR_OK) return false;
    
    SCIWaitForDMAQueue(g.dma_queue, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
    return error == SCI_ERR_OK;
}

// Send/Receive thread - handles communication with server
void *communication_thread(void *arg) {
    printf("Communication: Starting\n");
    
    while (true) {
        pthread_mutex_lock(&g.pipeline.mutex);
        
        // Look for frames ready to send
        bool found_frame = false;
        frame_buffer_t *frame_to_send = NULL;
        
        for (int i = 0; i < MAX_FRAMES; i++) {
            if (g.pipeline.frames[i].ready_to_send && 
                !g.pipeline.frames[i].result_received &&
                g.pipeline.frames[i].frame_number == g.pipeline.frames_sent) {
                frame_to_send = &g.pipeline.frames[i];
                found_frame = true;
                break;
            }
        }
        
        if (!found_frame) {
            if (g.pipeline.finished_reading && g.pipeline.frames_sent == g.pipeline.frames_read) {
                pthread_mutex_unlock(&g.pipeline.mutex);
                printf("Communication: All frames sent\n");
                break;
            }
            
            // Wait for frames to become available
            pthread_cond_wait(&g.pipeline.frames_available, &g.pipeline.mutex);
            pthread_mutex_unlock(&g.pipeline.mutex);
            continue;
        }
        
        pthread_mutex_unlock(&g.pipeline.mutex);
        
        // Send frame
        yuv_t *image = frame_to_send->yuv_data;
        int frame_number = frame_to_send->frame_number;
        
        printf("Communication: Sending frame %d\n", frame_number);
        
        // Pack YUV data
        size_t y_size = g.width * g.height;
        size_t u_size = (g.width * g.height) / 4;
        size_t v_size = (g.width * g.height) / 4;
        size_t total_size = y_size + u_size + v_size;
        
        memcpy((void*)g.send_seg->message_buffer, image->Y, y_size);
        memcpy((void*)(g.send_seg->message_buffer + y_size), image->U, u_size);
        memcpy((void*)(g.send_seg->message_buffer + y_size + u_size), image->V, v_size);
        
        // Send YUV data
        if (!send_dma_data((void*)g.send_seg->message_buffer, total_size, 0)) {
            printf("Communication: DMA transfer failed for frame %d\n", frame_number);
            continue;
        }
        
        // Signal server
        SCIFlush(NULL, NO_FLAGS);
        g.server_recv->packet.cmd = CMD_YUV_DATA;
        g.server_recv->packet.data_size = total_size;
        SCIFlush(NULL, NO_FLAGS);
        
        // Wait for YUV acknowledgment
        if (!wait_for_command(CMD_YUV_DATA_ACK, g.recv_seg, TIMEOUT_SECONDS)) {
            printf("Communication: Timeout waiting for YUV ACK for frame %d\n", frame_number);
            continue;
        }
        g.recv_seg->packet.cmd = CMD_INVALID;
        
        pthread_mutex_lock(&g.pipeline.mutex);
        g.pipeline.frames_sent++;
        pthread_mutex_unlock(&g.pipeline.mutex);
        
        printf("Communication: Frame %d sent, waiting for result\n", frame_number);
        
        // Wait for encoded data (can take longer)
        if (!wait_for_command(CMD_ENCODED_DATA, g.recv_seg, TIMEOUT_SECONDS * 4)) {
            printf("Communication: Timeout waiting for encoded data for frame %d\n", frame_number);
            continue;
        }
        
        // Process encoded data
        size_t data_size = g.recv_seg->packet.data_size;
        char *encoded_data = (char*)g.recv_seg->message_buffer;
        
        // Copy keyframe flag
        frame_to_send->keyframe = *((int*)encoded_data);
        encoded_data += sizeof(int);
        
        // Copy encoded data
        memcpy(frame_to_send->encoded_data, encoded_data, data_size - sizeof(int));
        frame_to_send->encoded_size = data_size - sizeof(int);
        
        // Acknowledge encoded data
        g.recv_seg->packet.cmd = CMD_INVALID;
        SCIFlush(NULL, NO_FLAGS);
        g.server_send->packet.cmd = CMD_ENCODED_DATA_ACK;
        SCIFlush(NULL, NO_FLAGS);
        
        pthread_mutex_lock(&g.pipeline.mutex);
        frame_to_send->result_received = true;
        g.pipeline.frames_received++;
        pthread_cond_signal(&g.pipeline.results_available);
        pthread_mutex_unlock(&g.pipeline.mutex);
        
        printf("Communication: Frame %d result received\n", frame_number);
    }
    
    printf("Communication: Finished\n");
    return NULL;
}

// Writer thread - writes frames to output in correct order
void *writer_thread(void *arg) {
    printf("Writer: Starting\n");
    
    while (true) {
        pthread_mutex_lock(&g.pipeline.mutex);
        
        // Look for the next frame to write
        frame_buffer_t *frame_to_write = NULL;
        int expected_frame = g.pipeline.frames_written;
        
        for (int i = 0; i < MAX_FRAMES; i++) {
            if (g.pipeline.frames[i].result_received && 
                g.pipeline.frames[i].frame_number == expected_frame) {
                frame_to_write = &g.pipeline.frames[i];
                break;
            }
        }
        
        if (!frame_to_write) {
            if (g.pipeline.finished_reading && 
                g.pipeline.frames_written == g.pipeline.frames_read) {
                pthread_mutex_unlock(&g.pipeline.mutex);
                printf("Writer: All frames written\n");
                break;
            }
            
            printf("Writer: Waiting for frame %d\n", expected_frame);
            pthread_cond_wait(&g.pipeline.results_available, &g.pipeline.mutex);
            pthread_mutex_unlock(&g.pipeline.mutex);
            continue;
        }
        
        pthread_mutex_unlock(&g.pipeline.mutex);
        
        printf("Writer: Writing frame %d\n", frame_to_write->frame_number);
        
        // Reconstruct frame data from encoded data
        char *ptr = frame_to_write->encoded_data;
        
        // Copy DCT data
        size_t ydct_size = g.cm->ypw * g.cm->yph * sizeof(int16_t);
        memcpy(g.cm->curframe->residuals->Ydct, ptr, ydct_size);
        ptr += ydct_size;
        
        size_t udct_size = g.cm->upw * g.cm->uph * sizeof(int16_t);
        memcpy(g.cm->curframe->residuals->Udct, ptr, udct_size);
        ptr += udct_size;
        
        size_t vdct_size = g.cm->vpw * g.cm->vph * sizeof(int16_t);
        memcpy(g.cm->curframe->residuals->Vdct, ptr, vdct_size);
        ptr += vdct_size;
        
        // Copy macroblock data
        size_t mby_size = g.cm->mb_rows * g.cm->mb_cols * sizeof(struct macroblock);
        memcpy(g.cm->curframe->mbs[Y_COMPONENT], ptr, mby_size);
        ptr += mby_size;
        
        size_t mbu_size = (g.cm->mb_rows/2) * (g.cm->mb_cols/2) * sizeof(struct macroblock);
        memcpy(g.cm->curframe->mbs[U_COMPONENT], ptr, mbu_size);
        ptr += mbu_size;
        
        size_t mbv_size = (g.cm->mb_rows/2) * (g.cm->mb_cols/2) * sizeof(struct macroblock);
        memcpy(g.cm->curframe->mbs[V_COMPONENT], ptr, mbv_size);
        
        // Set keyframe flag
        g.cm->curframe->keyframe = frame_to_write->keyframe;
        
        // Write frame
        write_frame(g.cm);
        g.cm->framenum++;
        g.cm->frames_since_keyframe++;
        if (g.cm->curframe->keyframe) {
            g.cm->frames_since_keyframe = 0;
        }
        
        printf("Writer: Frame %d written successfully\n", frame_to_write->frame_number);
        
        // Cleanup frame
        pthread_mutex_lock(&g.pipeline.mutex);
        if (frame_to_write->yuv_data) {
            free(frame_to_write->yuv_data->Y);
            free(frame_to_write->yuv_data->U);
            free(frame_to_write->yuv_data->V);
            free(frame_to_write->yuv_data);
            frame_to_write->yuv_data = NULL;
        }
        frame_to_write->ready_to_send = false;
        frame_to_write->result_received = false;
        g.pipeline.frames_written++;
        
        pthread_cond_signal(&g.pipeline.frames_available);
        pthread_mutex_unlock(&g.pipeline.mutex);
    }
    
    printf("Writer: Finished\n");
    return NULL;
}

// Send dimensions to server
bool send_dimensions() {
    struct dimensions_data dim_data = {g.width, g.height};
    
    if (!send_dma_data(&dim_data, sizeof(dim_data), 0)) {
        printf("Failed to send dimensions\n");
        return false;
    }
    
    SCIFlush(NULL, NO_FLAGS);
    g.server_recv->packet.cmd = CMD_DIMENSIONS;
    SCIFlush(NULL, NO_FLAGS);
    
    if (!wait_for_command(CMD_DIMENSIONS_ACK, g.recv_seg, TIMEOUT_SECONDS)) {
        printf("Timeout waiting for dimensions ACK\n");
        return false;
    }
    
    g.recv_seg->packet.cmd = CMD_INVALID;
    printf("Dimensions acknowledged\n");
    return true;
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

// SISCI initialization
bool init_sisci() {
    sci_error_t error;
    unsigned int localAdapterNo = 0;
    
    SCIInitialize(NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    sci_desc_t sd;
    SCIOpen(&sd, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    // Create segments
    SCICreateSegment(sd, &g.send_segment, SEGMENT_CLIENT_SEND, sizeof(struct send_segment),
                     NO_CALLBACK, NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    SCICreateSegment(sd, &g.recv_segment, SEGMENT_CLIENT_RECV, sizeof(struct recv_segment),
                     NO_CALLBACK, NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    // Prepare segments
    SCIPrepareSegment(g.send_segment, localAdapterNo, NO_FLAGS, &error);
    SCIPrepareSegment(g.recv_segment, localAdapterNo, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    // Create DMA queue
    SCICreateDMAQueue(sd, &g.dma_queue, localAdapterNo, 1, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    // Map local segments
    sci_map_t send_map, recv_map;
    g.send_seg = (volatile struct send_segment *)SCIMapLocalSegment(
        g.send_segment, &send_map, 0, sizeof(struct send_segment), NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    g.recv_seg = (volatile struct recv_segment *)SCIMapLocalSegment(
        g.recv_segment, &recv_map, 0, sizeof(struct recv_segment), NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    g.send_seg->packet.cmd = CMD_INVALID;
    g.recv_seg->packet.cmd = CMD_INVALID;
    
    // Make segments available
    SCISetSegmentAvailable(g.send_segment, localAdapterNo, NO_FLAGS, &error);
    SCISetSegmentAvailable(g.recv_segment, localAdapterNo, NO_FLAGS, &error);
    
    // Connect to server segments
    do {
        SCIConnectSegment(sd, &g.remote_server_recv, g.remote_node, SEGMENT_SERVER_RECV, 
                         localAdapterNo, NO_CALLBACK, NULL, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
    } while (error != SCI_ERR_OK);
    
    do {
        SCIConnectSegment(sd, &g.remote_server_send, g.remote_node, SEGMENT_SERVER_SEND,
                         localAdapterNo, NO_CALLBACK, NULL, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
    } while (error != SCI_ERR_OK);
    
    // Map server segments
    sci_map_t server_recv_map, server_send_map;
    g.server_recv = (volatile struct recv_segment *)SCIMapRemoteSegment(
        g.remote_server_recv, &server_recv_map, 0, sizeof(struct recv_segment), NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    g.server_send = (volatile struct send_segment *)SCIMapRemoteSegment(
        g.remote_server_send, &server_send_map, 0, sizeof(struct send_segment), NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    printf("SISCI initialized and connected\n");
    return true;
}

// Print help
static void print_help() {
    printf("Usage: ./c63client -r nodeid [options] input_file\n");
    printf("Options:\n");
    printf("  -r   Node id of server\n");
    printf("  -h   Height of images\n");
    printf("  -w   Width of images\n");
    printf("  -o   Output file (.c63)\n");
    printf("  -f   Limit number of frames\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char **argv) {
    int c;
    
    if (argc == 1) print_help();
    
    while ((c = getopt(argc, argv, "r:h:w:o:f:")) != -1) {
        switch (c) {
            case 'r': g.remote_node = atoi(optarg); break;
            case 'h': g.height = atoi(optarg); break;
            case 'w': g.width = atoi(optarg); break;
            case 'o': g.output_file = optarg; break;
            case 'f': g.limit_frames = atoi(optarg); break;
            default: print_help(); break;
        }
    }
    
    if (optind >= argc || g.remote_node == 0) {
        fprintf(stderr, "Missing input file or remote node\n");
        exit(EXIT_FAILURE);
    }
    
    g.input_file = argv[optind];
    
    // Open files
    g.outfile = fopen(g.output_file, "wb");
    if (!g.outfile) {
        perror("Output file");
        exit(EXIT_FAILURE);
    }
    
    FILE *infile = fopen(g.input_file, "rb");
    if (!infile) {
        perror("Input file");
        exit(EXIT_FAILURE);
    }
    
    // Initialize
    g.cm = init_encoder(g.width, g.height);
    g.cm->e_ctx.fp = g.outfile;
    g.cm->curframe = create_frame(g.cm, NULL);
    g.cm->refframe = create_frame(g.cm, NULL);
    
    pipeline_init(&g.pipeline);
    
    if (!init_sisci()) {
        fprintf(stderr, "SISCI initialization failed\n");
        exit(EXIT_FAILURE);
    }
    
    if (!send_dimensions()) {
        fprintf(stderr, "Failed to send dimensions\n");
        exit(EXIT_FAILURE);
    }
    
    // Start optimized threads
    pthread_t producer_tid, communication_tid, writer_tid;
    
    pthread_create(&producer_tid, NULL, producer_thread, infile);
    pthread_create(&communication_tid, NULL, communication_thread, NULL);
    pthread_create(&writer_tid, NULL, writer_thread, NULL);
    
    // Wait for all threads
    pthread_join(producer_tid, NULL);
    pthread_join(communication_tid, NULL);
    pthread_join(writer_tid, NULL);
    
    // Send quit command
    SCIFlush(NULL, NO_FLAGS);
    g.server_recv->packet.cmd = CMD_QUIT;
    SCIFlush(NULL, NO_FLAGS);
    
    // Print statistics
    printf("Client finished:\n");
    printf("  Frames read: %d\n", g.pipeline.frames_read);
    printf("  Frames sent: %d\n", g.pipeline.frames_sent);
    printf("  Frames received: %d\n", g.pipeline.frames_received);
    printf("  Frames written: %d\n", g.pipeline.frames_written);
    
    // Cleanup
    pipeline_destroy(&g.pipeline);
    if (g.cm) {
        destroy_frame(g.cm->refframe);
        destroy_frame(g.cm->curframe);
        free(g.cm);
        g.cm = NULL;
    }
    fclose(g.outfile);
    fclose(infile);
    
    printf("Client finished successfully\n");
    return 0;
}