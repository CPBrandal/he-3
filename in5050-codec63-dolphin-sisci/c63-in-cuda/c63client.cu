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

#define MAX_PIPELINE_FRAMES 3
#define PIPELINE_TIMEOUT_SECONDS 30
#define ENCODE_TIMEOUT_SECONDS 120
#define DMA_WAIT_MICROSECONDS 1000

// Pipeline management structures
typedef struct {
    yuv_t *image;
    int frame_number;
    bool valid;
    bool sent;
    time_t send_time;
} pipeline_slot_t;

typedef struct {
    pipeline_slot_t slots[MAX_PIPELINE_FRAMES];
    int frames_in_pipeline;
    int next_send_slot;
    int next_process_slot;      // NEW: Which slot to process next
    pthread_mutex_t mutex;
    pthread_cond_t slot_available;
    pthread_cond_t frame_ready;
    bool finished;
    int total_frames_read;
    int total_frames_sent;
    int total_frames_received;
} pipeline_manager_t;

// Global variables
static char *output_file, *input_file;
static uint32_t remote_node = 0;
static int limit_numframes = 0;
static uint32_t width;
static uint32_t height;
static FILE *outfile;

// SISCI resources
typedef struct {
    volatile struct send_segment *send_seg;
    volatile struct recv_segment *recv_seg;
    volatile struct recv_segment *server_recv;
    volatile struct send_segment *server_send;
    
    sci_dma_queue_t dma_queue;
    sci_local_segment_t send_segment;
    sci_local_segment_t recv_segment;
    sci_remote_segment_t remote_server_recv;
    sci_remote_segment_t remote_server_send;
    struct c63_common *cm;
} sisci_resources_t;

static sisci_resources_t g_sisci;
static pipeline_manager_t pipeline_mgr;

// Pipeline management functions
void pipeline_manager_init(pipeline_manager_t *mgr) {
    memset(mgr, 0, sizeof(pipeline_manager_t));
    mgr->finished = false;
    mgr->frames_in_pipeline = 0;
    mgr->next_send_slot = 0;
    mgr->next_process_slot = 0;  // Initialize process pointer
    
    for (int i = 0; i < MAX_PIPELINE_FRAMES; i++) {
        mgr->slots[i].valid = false;
        mgr->slots[i].image = NULL;
        mgr->slots[i].sent = false; 
    }
    
    pthread_mutex_init(&mgr->mutex, NULL);
    pthread_cond_init(&mgr->slot_available, NULL);
    pthread_cond_init(&mgr->frame_ready, NULL);
}

void pipeline_manager_destroy(pipeline_manager_t *mgr) {
    for (int i = 0; i < MAX_PIPELINE_FRAMES; i++) {
        if (mgr->slots[i].image) {
            free(mgr->slots[i].image->Y);
            free(mgr->slots[i].image->U);
            free(mgr->slots[i].image->V);
            free(mgr->slots[i].image);
        }
    }
    
    pthread_mutex_destroy(&mgr->mutex);
    pthread_cond_destroy(&mgr->slot_available);
    pthread_cond_destroy(&mgr->frame_ready);
}

// Get an available slot for a new frame (blocks if pipeline is full)
int pipeline_manager_get_send_slot(pipeline_manager_t *mgr) {
    pthread_mutex_lock(&mgr->mutex);
    
    while (mgr->frames_in_pipeline >= MAX_PIPELINE_FRAMES && !mgr->finished) {
        printf("Pipeline full, waiting for slot...\n");
        pthread_cond_wait(&mgr->slot_available, &mgr->mutex);
    }
    
    if (mgr->finished) {
        pthread_mutex_unlock(&mgr->mutex);
        return -1;
    }
    
    int slot = mgr->next_send_slot;
    mgr->next_send_slot = (mgr->next_send_slot + 1) % MAX_PIPELINE_FRAMES;
    
    pthread_mutex_unlock(&mgr->mutex);
    return slot;
}

// Add a frame to the pipeline
bool pipeline_manager_add_frame(pipeline_manager_t *mgr, int slot, yuv_t *image, int frame_number) {
    pthread_mutex_lock(&mgr->mutex);
    
    if (mgr->finished || slot < 0 || slot >= MAX_PIPELINE_FRAMES) {
        pthread_mutex_unlock(&mgr->mutex);
        return false;
    }
    
    mgr->slots[slot].image = image;
    mgr->slots[slot].frame_number = frame_number;
    mgr->slots[slot].valid = true;
    mgr->slots[slot].send_time = time(NULL);
    mgr->slots[slot].sent = false; 
    mgr->frames_in_pipeline++;
    mgr->total_frames_read++;
    
    printf("Added frame %d to pipeline slot %d (pipeline: %d/%d)\n", 
           frame_number, slot, mgr->frames_in_pipeline, MAX_PIPELINE_FRAMES);
    
    pthread_cond_signal(&mgr->frame_ready);
    pthread_mutex_unlock(&mgr->mutex);
    return true;
}

// SIMPLIFIED: Get next frame to send - no more complex search!
pipeline_slot_t* pipeline_manager_get_frame_to_send(pipeline_manager_t *mgr) {
    pthread_mutex_lock(&mgr->mutex);
    
    while (true) {
        // Simply check the next slot in sequence
        pipeline_slot_t *slot = &mgr->slots[mgr->next_process_slot];
        
        // If this slot has a valid, unsent frame - return it
        if (slot->valid && !slot->sent) {
            printf("Consumer: Processing frame %d from slot %d\n", 
                   slot->frame_number, mgr->next_process_slot);
            pthread_mutex_unlock(&mgr->mutex);
            return slot;
        }
        
        // Check if we're completely done
        if (mgr->finished && mgr->total_frames_received >= mgr->total_frames_read) {
            printf("Consumer: All frames processed (read: %d, sent: %d, received: %d)\n",
                   mgr->total_frames_read, mgr->total_frames_sent, mgr->total_frames_received);
            pthread_mutex_unlock(&mgr->mutex);
            return NULL;
        }
        
        // Wait for the next frame to be ready
        printf("Consumer: Waiting for frame in slot %d (Read: %d, Sent: %d, Received: %d)\n", 
               mgr->next_process_slot, mgr->total_frames_read, mgr->total_frames_sent, mgr->total_frames_received);
        
        pthread_cond_wait(&mgr->frame_ready, &mgr->mutex);
    }
}

// SIMPLIFIED: Mark frame as sent and advance process pointer
void pipeline_manager_mark_sent(pipeline_manager_t *mgr, pipeline_slot_t *slot) {
    pthread_mutex_lock(&mgr->mutex);
    
    slot->sent = true;
    mgr->total_frames_sent++;
    
    // Always advance the process pointer since we process in order
    mgr->next_process_slot = (mgr->next_process_slot + 1) % MAX_PIPELINE_FRAMES;
    
    printf("Frame %d sent to server (sent: %d, in pipeline: %d)\n", 
           slot->frame_number, mgr->total_frames_sent, mgr->frames_in_pipeline);
           
    pthread_mutex_unlock(&mgr->mutex);
}

// Remove frame from pipeline (when result received)
void pipeline_manager_frame_completed(pipeline_manager_t *mgr, int frame_number) {
    pthread_mutex_lock(&mgr->mutex);
    
    // Find and clear the slot
    for (int i = 0; i < MAX_PIPELINE_FRAMES; i++) {
        if (mgr->slots[i].valid && mgr->slots[i].frame_number == frame_number) {
            if (mgr->slots[i].image) {
                free(mgr->slots[i].image->Y);
                free(mgr->slots[i].image->U);
                free(mgr->slots[i].image->V);
                free(mgr->slots[i].image);
                mgr->slots[i].sent = false;
                mgr->slots[i].image = NULL;
            }
            mgr->slots[i].valid = false;
            mgr->frames_in_pipeline--;
            mgr->total_frames_received++;
            
            printf("Frame %d completed, removed from pipeline (received: %d, pipeline: %d/%d)\n", 
                   frame_number, mgr->total_frames_received, mgr->frames_in_pipeline, MAX_PIPELINE_FRAMES);
            
            pthread_cond_signal(&mgr->slot_available);
            
            // Signal consumer to check exit condition
            if (mgr->finished && mgr->frames_in_pipeline == 0) {
                printf("Pipeline empty and finished - signaling consumer to exit\n");
                pthread_cond_signal(&mgr->frame_ready);
            }
            
            break;
        }
    }
    
    pthread_mutex_unlock(&mgr->mutex);
}

void pipeline_manager_finish(pipeline_manager_t *mgr) {
    pthread_mutex_lock(&mgr->mutex);
    mgr->finished = true;
    pthread_cond_broadcast(&mgr->slot_available);
    pthread_cond_broadcast(&mgr->frame_ready);
    pthread_mutex_unlock(&mgr->mutex);
}

/* Read planar YUV frames with 4:2:0 chroma sub-sampling */
static yuv_t *read_yuv(FILE *file, struct c63_common *cm)
{
    size_t len = 0;
    yuv_t *image = (yuv_t *)malloc(sizeof(*image));

    image->Y = (uint8_t *)calloc(1, cm->padw[Y_COMPONENT] * cm->padh[Y_COMPONENT]);
    len += fread(image->Y, 1, width * height, file);

    image->U = (uint8_t *)calloc(1, cm->padw[U_COMPONENT] * cm->padh[U_COMPONENT]);
    len += fread(image->U, 1, (width * height) / 4, file);

    image->V = (uint8_t *)calloc(1, cm->padw[V_COMPONENT] * cm->padh[V_COMPONENT]);
    len += fread(image->V, 1, (width * height) / 4, file);

    if (ferror(file))
    {
        perror("ferror");
        free(image->Y);
        free(image->U);
        free(image->V);
        free(image);
        exit(EXIT_FAILURE);
    }

    if (feof(file))
    {
        free(image->Y);
        free(image->U);
        free(image->V);
        free(image);
        return NULL;
    }
    else if (len != width * height * 1.5)
    {
        fprintf(stderr, "Reached end of file, but incorrect bytes read.\n");
        fprintf(stderr, "Wrong input? (height: %d width: %d)\n", height, width);
        free(image->Y);
        free(image->U);
        free(image->V);
        free(image);
        return NULL;
    }

    return image;
}

// Producer thread - reads frames and adds them to pipeline (with flow control)
void *producer_thread(void *arg) {
    FILE *infile = (FILE *)arg;
    yuv_t *image;
    int numframes = 0;
    
    printf("Producer: Starting to read frames\n");
    
    while (1) {
        // Check termination conditions BEFORE getting a slot
        if (limit_numframes && numframes >= limit_numframes) {
            printf("Producer: Reached frame limit (%d frames), stopping\n", limit_numframes);
            break;
        }
        
        // Try to read next frame first
        image = read_yuv(infile, g_sisci.cm);
        if (!image) {
            printf("Producer: End of input file reached\n");
            break;
        }
        
        // Only get pipeline slot after we have a valid frame
        int slot = pipeline_manager_get_send_slot(&pipeline_mgr);
        if (slot < 0) {
            printf("Producer: Pipeline finished, stopping\n");
            // Free the frame we just read since we can't process it
            free(image->Y);
            free(image->U);
            free(image->V);
            free(image);
            break;
        }

        printf("Producer: Read frame %d from disk\n", numframes);

        if (!pipeline_manager_add_frame(&pipeline_mgr, slot, image, numframes)) {
            printf("Producer: Failed to add frame to pipeline\n");
            // Free the image since it wasn't added
            free(image->Y);
            free(image->U);
            free(image->V);
            free(image);
            break;
        }
        
        ++numframes;
    }
    
    pipeline_manager_finish(&pipeline_mgr);
    printf("Producer: Finished reading %d frames\n", numframes);
    return NULL;
}

// Consumer thread - sends frames to server and receives/writes results
void *consumer_thread(void *arg) {
    sci_error_t error;
    
    printf("Consumer: Starting to process frames\n");
    
    while (true) {
        pipeline_slot_t *slot = pipeline_manager_get_frame_to_send(&pipeline_mgr);
        if (!slot) {
            printf("Consumer: No more frames to process\n");
            break;
        }
        
        yuv_t *image = slot->image;
        int frame_number = slot->frame_number;
        
        printf("Consumer: Processing frame %d\n", frame_number);

        size_t y_size = width * height;
        size_t u_size = (width * height) / 4;
        size_t v_size = (width * height) / 4;
        size_t total_yuv_size = y_size + u_size + v_size;

        if (total_yuv_size > MESSAGE_SIZE) {
            fprintf(stderr, "Consumer: ERROR - Total YUV frame size (%zu) exceeds message buffer size (%d)\n", 
                   total_yuv_size, MESSAGE_SIZE);
            pipeline_manager_frame_completed(&pipeline_mgr, frame_number);
            continue;
        }

        // Pack YUV frames in send segment buffer
        memcpy((void*)g_sisci.send_seg->message_buffer, image->Y, y_size);
        memcpy((void*)(g_sisci.send_seg->message_buffer + y_size), image->U, u_size);
        memcpy((void*)(g_sisci.send_seg->message_buffer + y_size + u_size), image->V, v_size);

        // DMA transfer to server's receive segment
        SCIStartDmaTransfer(g_sisci.dma_queue, 
                        g_sisci.send_segment,
                        g_sisci.remote_server_recv,
                        offsetof(struct send_segment, message_buffer),
                        total_yuv_size,
                        offsetof(struct recv_segment, message_buffer),
                        NO_CALLBACK, NULL, NO_FLAGS, &error);

        if (error != SCI_ERR_OK) {
            fprintf(stderr, "Consumer: YUV frame DMA transfer failed - Error code 0x%x\n", error);
            pipeline_manager_frame_completed(&pipeline_mgr, frame_number);
            continue;
        }

        SCIWaitForDMAQueue(g_sisci.dma_queue, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);

        // Signal server that YUV frame is ready
        SCIFlush(NULL, NO_FLAGS);
        g_sisci.server_recv->packet.cmd = CMD_YUV_DATA;
        g_sisci.server_recv->packet.data_size = total_yuv_size;
        g_sisci.server_recv->packet.y_size = y_size;
        g_sisci.server_recv->packet.u_size = u_size;
        g_sisci.server_recv->packet.v_size = v_size;
        SCIFlush(NULL, NO_FLAGS);

        printf("Consumer: Sent frame %d to server, waiting for acknowledgment\n", frame_number);
        pipeline_manager_mark_sent(&pipeline_mgr, slot);

        // Wait for frame acknowledgment
        time_t frame_start = time(NULL);
        bool frame_timeout = false;

        while (g_sisci.recv_seg->packet.cmd != CMD_YUV_DATA_ACK && !frame_timeout) {
            if (time(NULL) - frame_start > PIPELINE_TIMEOUT_SECONDS) {
                frame_timeout = true;
                fprintf(stderr, "Consumer: Timeout waiting for YUV frame acknowledgment\n");
            }
            usleep(DMA_WAIT_MICROSECONDS);
        }

        if (frame_timeout) {
            fprintf(stderr, "Consumer: Failed to get YUV frame acknowledgment, skipping frame %d\n", frame_number);
            pipeline_manager_frame_completed(&pipeline_mgr, frame_number);
            continue;
        }

        printf("Consumer: Frame %d acknowledged by server\n", frame_number);
        g_sisci.recv_seg->packet.cmd = CMD_INVALID;

        printf("Consumer: Waiting for encoded data for frame %d\n", frame_number);
        time_t encode_start = time(NULL);
        bool encode_timeout = false;
        
        while (g_sisci.recv_seg->packet.cmd != CMD_ENCODED_DATA && !encode_timeout) {
            if (time(NULL) - encode_start > ENCODE_TIMEOUT_SECONDS) {
                encode_timeout = true;
                fprintf(stderr, "Consumer: Timeout waiting for encoded data for frame %d\n", frame_number);
            }
            usleep(DMA_WAIT_MICROSECONDS);
        }
        
        if (encode_timeout) {
            fprintf(stderr, "Consumer: Failed to receive encoded data, marking frame %d as completed anyway\n", frame_number);
            pipeline_manager_frame_completed(&pipeline_mgr, frame_number);
            continue;
        }

        printf("Consumer: Received encoded data for frame %d\n", frame_number);
        
        size_t data_size = g_sisci.recv_seg->packet.data_size;
        
        // Get keyframe flag
        int keyframe = *((int*)g_sisci.recv_seg->message_buffer);
        g_sisci.cm->curframe->keyframe = keyframe;
        
        // Get encoded data pointer
        char* encoded_data = (char*)g_sisci.recv_seg->message_buffer + sizeof(int);
        
        // Copy encoded data to curframe structure
        size_t ydct_size = g_sisci.cm->ypw * g_sisci.cm->yph * sizeof(int16_t);
        memcpy(g_sisci.cm->curframe->residuals->Ydct, encoded_data, ydct_size);
        encoded_data += ydct_size;
        
        size_t udct_size = g_sisci.cm->upw * g_sisci.cm->uph * sizeof(int16_t);
        memcpy(g_sisci.cm->curframe->residuals->Udct, encoded_data, udct_size);
        encoded_data += udct_size;
        
        size_t vdct_size = g_sisci.cm->vpw * g_sisci.cm->vph * sizeof(int16_t);
        memcpy(g_sisci.cm->curframe->residuals->Vdct, encoded_data, vdct_size);
        encoded_data += vdct_size;
        
        size_t mby_size = g_sisci.cm->mb_rows * g_sisci.cm->mb_cols * sizeof(struct macroblock);
        memcpy(g_sisci.cm->curframe->mbs[Y_COMPONENT], encoded_data, mby_size);
        encoded_data += mby_size;
        
        size_t mbu_size = (g_sisci.cm->mb_rows/2) * (g_sisci.cm->mb_cols/2) * sizeof(struct macroblock);
        memcpy(g_sisci.cm->curframe->mbs[U_COMPONENT], encoded_data, mbu_size);
        encoded_data += mbu_size;
        
        size_t mbv_size = (g_sisci.cm->mb_rows/2) * (g_sisci.cm->mb_cols/2) * sizeof(struct macroblock);
        memcpy(g_sisci.cm->curframe->mbs[V_COMPONENT], encoded_data, mbv_size);
        
        // Acknowledge receipt of encoded data
        g_sisci.recv_seg->packet.cmd = CMD_INVALID;
        SCIFlush(NULL, NO_FLAGS);
        g_sisci.server_send->packet.cmd = CMD_ENCODED_DATA_ACK;
        SCIFlush(NULL, NO_FLAGS);
        
        // Write frame to disk
        printf("Consumer: Writing frame %d to output file\n", frame_number);
        write_frame(g_sisci.cm);
        
        printf("Consumer: Frame %d complete!\n", frame_number);
        g_sisci.cm->framenum++;
        g_sisci.cm->frames_since_keyframe++;
        if (g_sisci.cm->curframe->keyframe) {
            g_sisci.cm->frames_since_keyframe = 0;
        }
        
        // Mark frame as completed (frees pipeline slot)
        pipeline_manager_frame_completed(&pipeline_mgr, frame_number);
    }
    
    printf("Consumer: Finished processing frames\n");
    return NULL;
}

// Send dimensions to server
int send_dimensions_to_server() {
    sci_error_t error;
    
    struct dimensions_data dim_data;
    dim_data.width = width;
    dim_data.height = height;
    
    memcpy((void*)g_sisci.send_seg->message_buffer, &dim_data, sizeof(struct dimensions_data));
    
    // DMA transfer to server's receive segment
    SCIStartDmaTransfer(g_sisci.dma_queue, 
        g_sisci.send_segment,
        g_sisci.remote_server_recv,
        offsetof(struct send_segment, message_buffer),
        sizeof(struct dimensions_data),
        offsetof(struct recv_segment, message_buffer),
        NO_CALLBACK, NULL, NO_FLAGS, &error);
                       
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "Client: SCIStartDmaTransfer for dimensions failed - Error code 0x%x\n", error);
        return -1;
    }
    
    SCIWaitForDMAQueue(g_sisci.dma_queue, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
    
    // Signal server
    SCIFlush(NULL, NO_FLAGS);
    g_sisci.server_recv->packet.cmd = CMD_DIMENSIONS;
    SCIFlush(NULL, NO_FLAGS);
    
    // Wait for acknowledgment
    printf("Client: Waiting for server to acknowledge dimensions\n");
    time_t dim_start = time(NULL);
    bool dim_timeout = false;
    
    while (g_sisci.recv_seg->packet.cmd != CMD_DIMENSIONS_ACK && !dim_timeout) {
        if (time(NULL) - dim_start > PIPELINE_TIMEOUT_SECONDS) {
            dim_timeout = true;
            fprintf(stderr, "Client: Timeout waiting for dimensions acknowledgment\n");
        }
        usleep(DMA_WAIT_MICROSECONDS);
    }
    
    if (dim_timeout) {
        fprintf(stderr, "Client: Failed to receive dimensions acknowledgment, exiting\n");
        return -1;
    }
    
    printf("Client: Dimensions acknowledged by server\n");
    g_sisci.recv_seg->packet.cmd = CMD_INVALID;
    
    return 0;
}

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

static void print_help()
{
    printf("Usage: ./c63client -r nodeid [options] input_file\n");
    printf("Commandline options:\n");
    printf("  -r                             Node id of server\n");
    printf("  -h                             Height of images to compress\n");
    printf("  -w                             Width of images to compress\n");
    printf("  -o                             Output file (.c63)\n");
    printf("  [-f]                           Limit number of frames to encode\n");
    printf("\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char **argv)
{
    unsigned int localAdapterNo = 0;
    int c;
    sci_error_t error;
    
    if (argc == 1) {
        print_help();
    }

    while ((c = getopt(argc, argv, "r:h:w:o:f:i:")) != -1)
    {
        switch (c)
        {
            case 'r':
                remote_node = atoi(optarg);
                break;
            case 'h':
                height = atoi(optarg);
                break;
            case 'w':
                width = atoi(optarg);
                break;
            case 'o':
                output_file = optarg;
                break;
            case 'f':
                limit_numframes = atoi(optarg);
                break;
            default:
                print_help();
                break;
        }
    }

    if (optind >= argc)
    {
        fprintf(stderr, "Error getting program options, try --help.\n");
        exit(EXIT_FAILURE);
    }

    input_file = argv[optind];

    if (remote_node == 0) {
        fprintf(stderr, "Remote node-id is not specified. Use -r <remote node-id>\n");
        exit(EXIT_FAILURE);
    }

    outfile = fopen(output_file, "wb");
    if (outfile == NULL)
    {
        perror("fopen");
        exit(EXIT_FAILURE);
    }

    g_sisci.cm = init_c63_enc(width, height);
    g_sisci.cm->e_ctx.fp = outfile;
    g_sisci.cm->curframe = create_frame(g_sisci.cm, NULL);
    g_sisci.cm->refframe = create_frame(g_sisci.cm, NULL);

    FILE *infile = fopen(input_file, "rb");
    if (infile == NULL)
    {
        perror("fopen");
        exit(EXIT_FAILURE); 
    }

    pipeline_manager_init(&pipeline_mgr);

    // Initialize SISCI
    SCIInitialize(NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCIInitialize failed: %s\n", SCIGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    sci_desc_t sd;
    sci_map_t send_map, recv_map, server_recv_map, server_send_map;

    SCIOpen(&sd, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCIOpen failed - Error code 0x%x\n", error);
        SCITerminate();
        exit(EXIT_FAILURE);
    }
    
    // Create client's send segment (for YUV data to server)
    SCICreateSegment(sd, &g_sisci.send_segment, SEGMENT_CLIENT_SEND, sizeof(struct send_segment),
                     NO_CALLBACK, NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCICreateSegment (send) failed - Error code 0x%x\n", error);
        exit(EXIT_FAILURE);
    }
    
    // Create client's receive segment (for encoded data from server)
    SCICreateSegment(sd, &g_sisci.recv_segment, SEGMENT_CLIENT_RECV, sizeof(struct recv_segment),
                     NO_CALLBACK, NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCICreateSegment (recv) failed - Error code 0x%x\n", error);
        exit(EXIT_FAILURE);
    }
    
    // Prepare segments
    SCIPrepareSegment(g_sisci.send_segment, localAdapterNo, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCIPrepareSegment (send) failed - Error code 0x%x\n", error);
        exit(EXIT_FAILURE);
    }
    
    SCIPrepareSegment(g_sisci.recv_segment, localAdapterNo, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCIPrepareSegment (recv) failed - Error code 0x%x\n", error);
        exit(EXIT_FAILURE);
    }
    
    // Create DMA queue
    SCICreateDMAQueue(sd, &g_sisci.dma_queue, localAdapterNo, 1, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCICreateDMAQueue failed - Error code 0x%x\n", error);
        exit(EXIT_FAILURE);
    }
    
    // Map local segments
    g_sisci.send_seg = (volatile struct send_segment *)SCIMapLocalSegment(
        g_sisci.send_segment, &send_map, 0, sizeof(struct send_segment), NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCIMapLocalSegment (send) failed - Error code 0x%x\n", error);
        exit(EXIT_FAILURE);
    }
    
    g_sisci.recv_seg = (volatile struct recv_segment *)SCIMapLocalSegment(
        g_sisci.recv_segment, &recv_map, 0, sizeof(struct recv_segment), NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCIMapLocalSegment (recv) failed - Error code 0x%x\n", error);
        exit(EXIT_FAILURE);
    }
    
    g_sisci.send_seg->packet.cmd = CMD_INVALID;
    g_sisci.recv_seg->packet.cmd = CMD_INVALID;
    
    // Make segments available
    SCISetSegmentAvailable(g_sisci.send_segment, localAdapterNo, NO_FLAGS, &error);
    SCISetSegmentAvailable(g_sisci.recv_segment, localAdapterNo, NO_FLAGS, &error);
    
    printf("Client: Connecting to server segments...\n");
    
    // Connect to server's segments
    do {
        SCIConnectSegment(sd, &g_sisci.remote_server_recv, remote_node, SEGMENT_SERVER_RECV, localAdapterNo,
                          NO_CALLBACK, NULL, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
    } while (error != SCI_ERR_OK);
    
    do {
        SCIConnectSegment(sd, &g_sisci.remote_server_send, remote_node, SEGMENT_SERVER_SEND, localAdapterNo,
                          NO_CALLBACK, NULL, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
    } while (error != SCI_ERR_OK);
    
    printf("Client: Connected to server segments\n");
    
    // Map server's segments
    g_sisci.server_recv = (volatile struct recv_segment *)SCIMapRemoteSegment(
        g_sisci.remote_server_recv, &server_recv_map, 0, sizeof(struct recv_segment), NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCIMapRemoteSegment (server recv) failed - Error code 0x%x\n", error);
        exit(EXIT_FAILURE);
    }

    g_sisci.server_send = (volatile struct send_segment *)SCIMapRemoteSegment(
        g_sisci.remote_server_send, &server_send_map, 0, sizeof(struct send_segment), NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCIMapRemoteSegment (server send) failed - Error code 0x%x\n", error);
        exit(EXIT_FAILURE);
    }

    // Send dimensions to server
    if (send_dimensions_to_server() != 0) {
        fprintf(stderr, "Failed to send dimensions to server\n");
        exit(EXIT_FAILURE);
    }

    // Create and start threads
    pthread_t producer_tid, consumer_tid;
    
    printf("Client: Starting producer and consumer threads\n");
    
    if (pthread_create(&producer_tid, NULL, producer_thread, infile) != 0) {
        fprintf(stderr, "Failed to create producer thread\n");
        exit(EXIT_FAILURE);
    }
    
    if (pthread_create(&consumer_tid, NULL, consumer_thread, NULL) != 0) {
        fprintf(stderr, "Failed to create consumer thread\n");
        exit(EXIT_FAILURE);
    }
    
    // Wait for producer to finish reading
    pthread_join(producer_tid, NULL);
    printf("Client: Producer finished, waiting for consumer to process all frames\n");
    
    // Wait for consumer to finish processing all frames
    pthread_join(consumer_tid, NULL);

    // Verify all frames processed before sending QUIT
    if (pipeline_mgr.total_frames_received != pipeline_mgr.total_frames_read) {
        printf("Client: WARNING - Waiting for remaining frames...\n");
        sleep(2);  // Give time for last frames to complete
    }

    // Send quit command
    printf("Client: Sending quit command to server\n");
    SCIFlush(NULL, NO_FLAGS);
    g_sisci.server_recv->packet.cmd = CMD_QUIT;
    SCIFlush(NULL, NO_FLAGS);
    
    // Cleanup
    pipeline_manager_destroy(&pipeline_mgr);
    destroy_frame(g_sisci.cm->refframe);
    fclose(outfile);
    fclose(infile);
    free_c63_enc(g_sisci.cm);
    
    SCIUnmapSegment(server_send_map, NO_FLAGS, &error);
    SCIUnmapSegment(server_recv_map, NO_FLAGS, &error);
    SCIDisconnectSegment(g_sisci.remote_server_send, NO_FLAGS, &error);
    SCIDisconnectSegment(g_sisci.remote_server_recv, NO_FLAGS, &error);
    SCISetSegmentUnavailable(g_sisci.recv_segment, localAdapterNo, NO_FLAGS, &error);
    SCISetSegmentUnavailable(g_sisci.send_segment, localAdapterNo, NO_FLAGS, &error);
    SCIUnmapSegment(recv_map, NO_FLAGS, &error);
    SCIUnmapSegment(send_map, NO_FLAGS, &error);
    SCIRemoveDMAQueue(g_sisci.dma_queue, NO_FLAGS, &error);
    SCIRemoveSegment(g_sisci.recv_segment, NO_FLAGS, &error);
    SCIRemoveSegment(g_sisci.send_segment, NO_FLAGS, &error);
    SCIClose(sd, NO_FLAGS, &error);
    SCITerminate();
    
    return 0;
}