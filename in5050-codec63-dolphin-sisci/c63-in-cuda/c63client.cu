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

// Simplified frame slot
typedef struct {
    yuv_t *image;
    int frame_number;
    bool occupied;  // Single state flag instead of multiple
} frame_slot_t;

// Simplified pipeline manager
typedef struct {
    frame_slot_t slots[MAX_FRAMES];
    int head, tail;          // Just two pointers instead of three
    int count;               // Number of frames currently in pipeline
    int next_frame_number;   // Counter for frame numbering
    bool finished;
    pthread_mutex_t mutex;
    pthread_cond_t not_full, not_empty;
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
    p->head = 0;
    p->tail = 0;
    p->count = 0;
    p->next_frame_number = 0;
    p->finished = false;
    
    pthread_mutex_init(&p->mutex, NULL);
    pthread_cond_init(&p->not_full, NULL);
    pthread_cond_init(&p->not_empty, NULL);
}

// Cleanup pipeline
void pipeline_destroy(pipeline_t *p) {
    pthread_mutex_lock(&p->mutex);
    
    // Free any remaining frames
    for (int i = 0; i < MAX_FRAMES; i++) {
        if (p->slots[i].occupied && p->slots[i].image) {
            free(p->slots[i].image->Y);
            free(p->slots[i].image->U);
            free(p->slots[i].image->V);
            free(p->slots[i].image);
            p->slots[i].image = NULL;
            p->slots[i].occupied = false;
        }
    }
    
    pthread_mutex_unlock(&p->mutex);
    
    pthread_mutex_destroy(&p->mutex);
    pthread_cond_destroy(&p->not_full);
    pthread_cond_destroy(&p->not_empty);
}

// Add frame to pipeline (blocks if full)
void pipeline_add_frame(pipeline_t *p, yuv_t *image) {
    pthread_mutex_lock(&p->mutex);
    
    // Wait for space if pipeline is full
    while (p->count == MAX_FRAMES && !p->finished) {
        pthread_cond_wait(&p->not_full, &p->mutex);
    }
    
    if (!p->finished) {
        // Add frame to tail
        frame_slot_t *slot = &p->slots[p->tail];
        slot->image = image;
        slot->frame_number = p->next_frame_number++;
        slot->occupied = true;
        
        printf("Added frame %d to slot %d\n", slot->frame_number, p->tail);
        
        // Update tail and count
        p->tail = (p->tail + 1) % MAX_FRAMES;
        p->count++;
        
        pthread_cond_signal(&p->not_empty);
    } else {
        // Pipeline finished, free the image we can't use
        free(image->Y);
        free(image->U);
        free(image->V);
        free(image);
    }
    
    pthread_mutex_unlock(&p->mutex);
}

// Get next frame to process
frame_slot_t* pipeline_get_next_frame(pipeline_t *p) {
    pthread_mutex_lock(&p->mutex);
    
    // Wait for frame if pipeline is empty
    while (p->count == 0 && !p->finished) {
        pthread_cond_wait(&p->not_empty, &p->mutex);
    }
    
    if (p->count == 0) {
        // No more frames
        pthread_mutex_unlock(&p->mutex);
        return NULL;
    }
    
    // Return frame at head (don't remove yet)
    frame_slot_t *slot = &p->slots[p->head];
    pthread_mutex_unlock(&p->mutex);
    return slot;
}

// Mark current frame as completed and remove it
void pipeline_frame_done(pipeline_t *p) {
    pthread_mutex_lock(&p->mutex);
    
    if (p->count > 0) {
        frame_slot_t *slot = &p->slots[p->head];
        
        if (slot->occupied) {
            // Free frame memory
            if (slot->image) {
                free(slot->image->Y);
                free(slot->image->U);
                free(slot->image->V);
                free(slot->image);
                slot->image = NULL;
            }
            
            slot->occupied = false;
            printf("Frame %d completed\n", slot->frame_number);
            
            // Update head and count
            p->head = (p->head + 1) % MAX_FRAMES;
            p->count--;
            
            pthread_cond_signal(&p->not_full);
        }
    }
    
    pthread_mutex_unlock(&p->mutex);
}

// Mark pipeline as finished
void pipeline_finish(pipeline_t *p) {
    pthread_mutex_lock(&p->mutex);
    p->finished = true;
    pthread_cond_broadcast(&p->not_full);
    pthread_cond_broadcast(&p->not_empty);
    pthread_mutex_unlock(&p->mutex);
}

// Get pipeline statistics (for debugging/monitoring)
void pipeline_get_stats(pipeline_t *p, int *frames_in_pipeline, int *total_processed) {
    pthread_mutex_lock(&p->mutex);
    *frames_in_pipeline = p->count;
    *total_processed = p->next_frame_number - p->count;
    pthread_mutex_unlock(&p->mutex);
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

// Producer thread - reads frames
void *producer_thread(void *arg) {
    FILE *infile = (FILE *)arg;
    int frame_count = 0;
    
    printf("Producer: Starting\n");
    
    while (true) {
        if (g.limit_frames && frame_count >= g.limit_frames) {
            printf("Producer: Frame limit reached\n");
            break;
        }
        
        yuv_t *image = read_yuv(infile);
        if (!image) {
            printf("Producer: End of file\n");
            break;
        }
        
        pipeline_add_frame(&g.pipeline, image);
        frame_count++;
    }
    
    pipeline_finish(&g.pipeline);
    printf("Producer: Finished reading %d frames\n", frame_count);
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

// Consumer thread - sends frames and receives results
void *consumer_thread(void *arg) {
    printf("Consumer: Starting\n");
    
    while (true) {
        // Get next frame to process
        frame_slot_t *slot = pipeline_get_next_frame(&g.pipeline);
        if (!slot) {
            printf("Consumer: No more frames\n");
            break;
        }
        
        yuv_t *image = slot->image;
        int frame_number = slot->frame_number;
        
        printf("Consumer: Processing frame %d\n", frame_number);
        
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
            printf("Consumer: DMA transfer failed for frame %d\n", frame_number);
            pipeline_frame_done(&g.pipeline);  // Mark as done even if failed
            continue;
        }
        
        // Signal server
        SCIFlush(NULL, NO_FLAGS);
        g.server_recv->packet.cmd = CMD_YUV_DATA;
        g.server_recv->packet.data_size = total_size;
        SCIFlush(NULL, NO_FLAGS);
        
        // Wait for YUV acknowledgment
        if (!wait_for_command(CMD_YUV_DATA_ACK, g.recv_seg, TIMEOUT_SECONDS)) {
            printf("Consumer: Timeout waiting for YUV ACK for frame %d\n", frame_number);
            pipeline_frame_done(&g.pipeline);
            continue;
        }
        g.recv_seg->packet.cmd = CMD_INVALID;
        
        // Wait for encoded data
        if (!wait_for_command(CMD_ENCODED_DATA, g.recv_seg, TIMEOUT_SECONDS * 4)) {
            printf("Consumer: Timeout waiting for encoded data for frame %d\n", frame_number);
            pipeline_frame_done(&g.pipeline);
            continue;
        }
        
        // Process encoded data
        size_t data_size = g.recv_seg->packet.data_size;
        char *encoded_data = (char*)g.recv_seg->message_buffer;
        
        // Copy keyframe flag
        g.cm->curframe->keyframe = *((int*)encoded_data);
        encoded_data += sizeof(int);
        
        // Copy DCT data
        size_t ydct_size = g.cm->ypw * g.cm->yph * sizeof(int16_t);
        memcpy(g.cm->curframe->residuals->Ydct, encoded_data, ydct_size);
        encoded_data += ydct_size;
        
        size_t udct_size = g.cm->upw * g.cm->uph * sizeof(int16_t);
        memcpy(g.cm->curframe->residuals->Udct, encoded_data, udct_size);
        encoded_data += udct_size;
        
        size_t vdct_size = g.cm->vpw * g.cm->vph * sizeof(int16_t);
        memcpy(g.cm->curframe->residuals->Vdct, encoded_data, vdct_size);
        encoded_data += vdct_size;
        
        // Copy macroblock data
        size_t mby_size = g.cm->mb_rows * g.cm->mb_cols * sizeof(struct macroblock);
        memcpy(g.cm->curframe->mbs[Y_COMPONENT], encoded_data, mby_size);
        encoded_data += mby_size;
        
        size_t mbu_size = (g.cm->mb_rows/2) * (g.cm->mb_cols/2) * sizeof(struct macroblock);
        memcpy(g.cm->curframe->mbs[U_COMPONENT], encoded_data, mbu_size);
        encoded_data += mbu_size;
        
        size_t mbv_size = (g.cm->mb_rows/2) * (g.cm->mb_cols/2) * sizeof(struct macroblock);
        memcpy(g.cm->curframe->mbs[V_COMPONENT], encoded_data, mbv_size);
        
        // Acknowledge encoded data
        g.recv_seg->packet.cmd = CMD_INVALID;
        SCIFlush(NULL, NO_FLAGS);
        g.server_send->packet.cmd = CMD_ENCODED_DATA_ACK;
        SCIFlush(NULL, NO_FLAGS);
        
        // Write frame
        write_frame(g.cm);
        g.cm->framenum++;
        g.cm->frames_since_keyframe++;
        if (g.cm->curframe->keyframe) {
            g.cm->frames_since_keyframe = 0;
        }
        
        // Mark frame as completed (this will free the memory and update pipeline)
        pipeline_frame_done(&g.pipeline);
        printf("Consumer: Frame %d complete\n", frame_number);
    }
    
    printf("Consumer: Finished\n");
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
    
    // Start threads
    pthread_t producer_tid, consumer_tid;
    pthread_create(&producer_tid, NULL, producer_thread, infile);
    pthread_create(&consumer_tid, NULL, consumer_thread, NULL);
    
    // Wait for threads
    pthread_join(producer_tid, NULL);
    pthread_join(consumer_tid, NULL);
    
    // Send quit command
    SCIFlush(NULL, NO_FLAGS);
    g.server_recv->packet.cmd = CMD_QUIT;
    SCIFlush(NULL, NO_FLAGS);
    
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