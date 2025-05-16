#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "c63.h"
#include "quantdct.h"
#include "common.h"
#include "me.h"
#include "tables.h"

#include <sisci_error.h>
#include <sisci_api.h>

static uint32_t width = 0;
static uint32_t height = 0;

/* getopt */
extern int optind;
extern char *optarg;

struct c63_common *
init_c63_enc( int width, int height )
{
    int i;

    /* calloc() sets allocated memory to zero */
    c63_common *cm =
        ( c63_common * ) calloc( 1, sizeof( struct c63_common ) );

    cm->width = width;
    cm->height = height;

    cm->padw[Y_COMPONENT] = cm->ypw =
        ( uint32_t ) ( ceil( width / 16.0f ) * 16 );
    cm->padh[Y_COMPONENT] = cm->yph =
        ( uint32_t ) ( ceil( height / 16.0f ) * 16 );
    cm->padw[U_COMPONENT] = cm->upw =
        ( uint32_t ) ( ceil( width * UX / ( YX * 8.0f ) ) * 8 );
    cm->padh[U_COMPONENT] = cm->uph =
        ( uint32_t ) ( ceil( height * UY / ( YY * 8.0f ) ) * 8 );
    cm->padw[V_COMPONENT] = cm->vpw =
        ( uint32_t ) ( ceil( width * VX / ( YX * 8.0f ) ) * 8 );
    cm->padh[V_COMPONENT] = cm->vph =
        ( uint32_t ) ( ceil( height * VY / ( YY * 8.0f ) ) * 8 );

    cm->mb_cols = cm->ypw / 8;
    cm->mb_rows = cm->yph / 8;

    /* Quality parameters -- Home exam deliveries should have original values,
       i.e., quantization factor should be 25, search range should be 16, and the
       keyframe interval should be 100. */
    cm->qp = 25;                // Constant quantization factor. Range: [1..50]
    cm->me_search_range = 16;   // Pixels in every direction
    cm->keyframe_interval = 100;        // Distance between keyframes

    /* Initialize quantization tables */
    for ( i = 0; i < 64; ++i )
    {
        cm->quanttbl[Y_COMPONENT][i] = yquanttbl_def[i] / ( cm->qp / 10.0 );
        cm->quanttbl[U_COMPONENT][i] = uvquanttbl_def[i] / ( cm->qp / 10.0 );
        cm->quanttbl[V_COMPONENT][i] = uvquanttbl_def[i] / ( cm->qp / 10.0 );
    }

    return cm;
}

static yuv_t *yuv_server(struct c63_common *cm)
{
    if(!cm){
        printf("CM IS NULL");
        return NULL;
    }
    yuv_t *image = (yuv_t *)malloc(sizeof(*image));

    /* Read Y. The size of Y is the same as the size of the image */
    cudaMallocManaged((void**)&image->Y, cm->padw[Y_COMPONENT] * cm->padh[Y_COMPONENT] * sizeof(uint8_t));

    /* Read U. Given 4:2:0 chroma sub-sampling, the size is 1/4 of Y */
    cudaMallocManaged((void**)&image->U, cm->padw[U_COMPONENT] * cm->padh[U_COMPONENT] * sizeof(uint8_t));

    /* Read V. Given 4:2:0 chroma sub-sampling, the size is 1/4 of Y. */
    cudaMallocManaged((void**)&image->V, cm->padw[V_COMPONENT] * cm->padh[V_COMPONENT] * sizeof(uint8_t));

    return image;
}

static void
c63_encode_image_server( struct c63_common *cm, yuv_t *image )
{
    /* Advance to next frame */
    printf("1\n");
    destroy_frame( cm->refframe );
    cm->refframe = cm->curframe;
    cm->curframe = create_frame( cm, image );
    /* Check if keyframe */
    printf("2\n");

    if ( cm->framenum == 0
         || cm->frames_since_keyframe == cm->keyframe_interval )
    {
        cm->curframe->keyframe = 1;
        cm->frames_since_keyframe = 0;

        fprintf( stderr, " (keyframe) " );
    }
    else
    {
        cm->curframe->keyframe = 0;
    }
    printf("3\n");

    if ( !cm->curframe->keyframe )
    {
        /* Motion Estimation */
        c63_motion_estimate( cm );

        /* Motion Compensation */
        c63_motion_compensate( cm );
    }
    printf("4\n");

    /* DCT and Quantization */
    dct_quantize( image->Y, cm->curframe->predicted->Y, cm->padw[Y_COMPONENT],
                  cm->padh[Y_COMPONENT], cm->curframe->residuals->Ydct,
                  cm->quanttbl[Y_COMPONENT] );

    dct_quantize( image->U, cm->curframe->predicted->U, cm->padw[U_COMPONENT],
                  cm->padh[U_COMPONENT], cm->curframe->residuals->Udct,
                  cm->quanttbl[U_COMPONENT] );

    dct_quantize( image->V, cm->curframe->predicted->V, cm->padw[V_COMPONENT],
                  cm->padh[V_COMPONENT], cm->curframe->residuals->Vdct,
                  cm->quanttbl[V_COMPONENT] );
    printf("5\n");

    /* Reconstruct frame for inter-prediction */
    dequantize_idct( cm->curframe->residuals->Ydct,
                     cm->curframe->predicted->Y, cm->ypw, cm->yph,
                     cm->curframe->recons->Y, cm->quanttbl[Y_COMPONENT] );
    dequantize_idct( cm->curframe->residuals->Udct,
                     cm->curframe->predicted->U, cm->upw, cm->uph,
                     cm->curframe->recons->U, cm->quanttbl[U_COMPONENT] );
    dequantize_idct( cm->curframe->residuals->Vdct,
                     cm->curframe->predicted->V, cm->vpw, cm->vph,
                     cm->curframe->recons->V, cm->quanttbl[V_COMPONENT] );

    /* Function dump_image(), found in common.c, can be used here to check if the
       prediction is correct */
    printf("6\n");

    ++cm->framenum;
    ++cm->frames_since_keyframe;
}

void
free_c63_enc( struct c63_common *cm )
{
    destroy_frame( cm->curframe );
    free( cm );
}

int process_frame(struct c63_common *cm,
    volatile struct server_segment *local_seg,
    volatile struct client_segment *remote_seg,
    sci_dma_queue_t dma_queue,
    sci_local_segment_t local_segment,
    sci_remote_segment_t remote_segment) 
{
    sci_error_t error;

    // 1. Wait for and process frame header
    while (local_seg->packet.cmd != CMD_FRAME_HEADER) {
    // Wait for frame header
    }

    struct frame_header header;
    memcpy(&header, (const void*)local_seg->message_buffer, sizeof(struct frame_header));

    int frame_number = header.frame_number;
    int is_last_frame = header.is_last_frame;

    printf("Server: Received frame header for frame %d\n", frame_number);

    // Acknowledge frame header
    local_seg->packet.cmd = CMD_INVALID;
    SCIFlush(NULL, NO_FLAGS);
    remote_seg->packet.cmd = CMD_FRAME_HEADER_ACK;
    SCIFlush(NULL, NO_FLAGS);

    // 2. Wait for Y component
    while (local_seg->packet.cmd != CMD_Y_DATA_READY) {
        // Wait for Y data
    }
    printf("Server: Received Y component for frame %d\n", frame_number);

    // Acknowledge Y component
    local_seg->packet.cmd = CMD_INVALID;
    SCIFlush(NULL, NO_FLAGS);
    remote_seg->packet.cmd = CMD_Y_DATA_ACK;
    SCIFlush(NULL, NO_FLAGS);

    // 3. Wait for U component
    while (local_seg->packet.cmd != CMD_U_DATA_READY) {
        // Wait for U data
    }
    printf("Server: Received U component for frame %d\n", frame_number);

    // Acknowledge U component
    local_seg->packet.cmd = CMD_INVALID;
    SCIFlush(NULL, NO_FLAGS);
    remote_seg->packet.cmd = CMD_U_DATA_ACK;
    SCIFlush(NULL, NO_FLAGS);

    // 4. Wait for V component
    while (local_seg->packet.cmd != CMD_V_DATA_READY) {
        // Wait for V data
    }
    printf("Server: Received V component for frame %d\n", frame_number);

    // Acknowledge V component
    local_seg->packet.cmd = CMD_INVALID;
    SCIFlush(NULL, NO_FLAGS);
    remote_seg->packet.cmd = CMD_V_DATA_ACK;
    SCIFlush(NULL, NO_FLAGS);

    // 5. Create yuv_t structure pointing to received data
    yuv_t *image = yuv_server(cm);
    
    // Allocate memory for the Y, U, V planes
    int y_size = cm->padw[Y_COMPONENT] * cm->padh[Y_COMPONENT];
    int u_size = cm->padw[U_COMPONENT] * cm->padh[U_COMPONENT];
    int v_size = cm->padw[V_COMPONENT] * cm->padh[V_COMPONENT];
    
    if (!image->Y || !image->U || !image->V) {
        fprintf(stderr, "Failed to allocate image buffers\n");
        // Free any allocated memory
        if (image->Y) cudaFree(image->Y);
        if (image->U) cudaFree(image->U);
        if (image->V) cudaFree(image->V);
        free(image);
        return -1;
    }

    // Copy data from shared memory buffers
    memcpy(image->Y, (const void*)local_seg->y_buffer, y_size);
    memcpy(image->U, (const void*)local_seg->u_buffer, u_size);
    memcpy(image->V, (const void*)local_seg->v_buffer, v_size);

    // 6. Process the frame with encoding functions
    printf("Server: Processing frame %d\n", frame_number);

    // Perform encoding as in your provided code
    c63_encode_image_server(cm, image);

    printf("Server: Sending encoded data for frame %d\n", frame_number);

    // First, send the encoded frame header
    struct encoded_frame_header enc_header;
    enc_header.keyframe = cm->curframe->keyframe;
    enc_header.y_size = cm->padw[Y_COMPONENT] * cm->padh[Y_COMPONENT] * sizeof(int16_t);
    enc_header.u_size = cm->padw[U_COMPONENT] * cm->padh[U_COMPONENT] * sizeof(int16_t);
    enc_header.v_size = cm->padw[V_COMPONENT] * cm->padh[V_COMPONENT] * sizeof(int16_t);
    enc_header.mv_y_size = (cm->mb_rows * cm->mb_cols) * sizeof(struct macroblock);
    enc_header.mv_u_size = ((cm->mb_rows/2) * (cm->mb_cols/2)) * sizeof(struct macroblock);
    enc_header.mv_v_size = ((cm->mb_rows/2) * (cm->mb_cols/2)) * sizeof(struct macroblock);
    
    memcpy((void*)local_seg->message_buffer, &enc_header, sizeof(struct encoded_frame_header));
    
    // Use DMA to transfer the header
    SCIStartDmaTransfer(dma_queue, 
        local_segment,
        remote_segment,
        offsetof(struct server_segment, message_buffer),
        sizeof(struct encoded_frame_header),
        offsetof(struct client_segment, message_buffer),
        NO_CALLBACK,
        NULL,
        NO_FLAGS,
        &error);
        
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "Server: SCIStartDmaTransfer for encoded header failed - Error code 0x%x\n", error);
        return -1;
    }
    
    SCIWaitForDMAQueue(dma_queue, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "Server: SCIWaitForDMAQueue for encoded header failed - Error code 0x%x\n", error);
        return -1;
    }
    
    // Notify client that encoded header is ready
    SCIFlush(NULL, NO_FLAGS);
    remote_seg->packet.cmd = CMD_ENCODED_DATA_HEADER;
    SCIFlush(NULL, NO_FLAGS);
    
    // Wait for client to acknowledge
    local_seg->packet.cmd = CMD_INVALID;
    while (local_seg->packet.cmd != CMD_ENCODED_DATA_HEADER_ACK) {
        // Wait for acknowledgment
    }
    
    // Send Y residuals
    memcpy((void*)local_seg->y_buffer, cm->curframe->residuals->Ydct, enc_header.y_size);
    
    SCIStartDmaTransfer(dma_queue, 
                        local_segment,
                        remote_segment,
                        offsetof(struct server_segment, y_buffer),
                        enc_header.y_size,
                        offsetof(struct client_segment, y_buffer),
                        NO_CALLBACK,
                        NULL,
                        NO_FLAGS,
                        &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "Server: SCIStartDmaTransfer for Y residuals failed - Error code 0x%x\n", error);
        return -1;
    }
    
    SCIWaitForDMAQueue(dma_queue, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "Server: SCIWaitForDMAQueue for Y residuals failed - Error code 0x%x\n", error);
        return -1;
    }
    
    // Notify client that Y residuals data is ready
    SCIFlush(NULL, NO_FLAGS);
    remote_seg->packet.cmd = CMD_RESIDUALS_Y_READY;
    SCIFlush(NULL, NO_FLAGS);
    
    // Wait for acknowledgment
    local_seg->packet.cmd = CMD_INVALID;
    while (local_seg->packet.cmd != CMD_RESIDUALS_Y_ACK) {
        // Wait for acknowledgment
    }
    
    // Send U residuals
    memcpy((void*)local_seg->u_buffer, cm->curframe->residuals->Udct, enc_header.u_size);
    
    SCIStartDmaTransfer(dma_queue, 
                        local_segment,
                        remote_segment,
                        offsetof(struct server_segment, u_buffer),
                        enc_header.u_size,
                        offsetof(struct client_segment, u_buffer),
                        NO_CALLBACK,
                        NULL,
                        NO_FLAGS,
                        &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "Server: SCIStartDmaTransfer for U residuals failed - Error code 0x%x\n", error);
        return -1;
    }
    
    SCIWaitForDMAQueue(dma_queue, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "Server: SCIWaitForDMAQueue for U residuals failed - Error code 0x%x\n", error);
        return -1;
    }
    
    // Notify client that U residuals data is ready
    SCIFlush(NULL, NO_FLAGS);
    remote_seg->packet.cmd = CMD_RESIDUALS_U_READY;
    SCIFlush(NULL, NO_FLAGS);
    
    // Wait for acknowledgment
    local_seg->packet.cmd = CMD_INVALID;
    while (local_seg->packet.cmd != CMD_RESIDUALS_U_ACK) {
        // Wait for acknowledgment
    }
    
    // Send V residuals
    memcpy((void*)local_seg->v_buffer, cm->curframe->residuals->Vdct, enc_header.v_size);
    
    SCIStartDmaTransfer(dma_queue, 
                        local_segment,
                        remote_segment,
                        offsetof(struct server_segment, v_buffer),
                        enc_header.v_size,
                        offsetof(struct client_segment, v_buffer),
                        NO_CALLBACK,
                        NULL,
                        NO_FLAGS,
                        &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "Server: SCIStartDmaTransfer for V residuals failed - Error code 0x%x\n", error);
        return -1;
    }
    
    SCIWaitForDMAQueue(dma_queue, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "Server: SCIWaitForDMAQueue for V residuals failed - Error code 0x%x\n", error);
        return -1;
    }
    
    // Notify client that V residuals data is ready
    SCIFlush(NULL, NO_FLAGS);
    remote_seg->packet.cmd = CMD_RESIDUALS_V_READY;
    SCIFlush(NULL, NO_FLAGS);
    
    // Wait for acknowledgment
    local_seg->packet.cmd = CMD_INVALID;
    while (local_seg->packet.cmd != CMD_RESIDUALS_V_ACK) {
        // Wait for acknowledgment
    }
    
    // Send motion vectors for all components
    memcpy((void*)local_seg->mv_y_buffer, cm->curframe->mbs[Y_COMPONENT], enc_header.mv_y_size);
    memcpy((void*)local_seg->mv_u_buffer, cm->curframe->mbs[U_COMPONENT], enc_header.mv_u_size);
    memcpy((void*)local_seg->mv_v_buffer, cm->curframe->mbs[V_COMPONENT], enc_header.mv_v_size);
    
    // Transfer Y motion vectors
    SCIStartDmaTransfer(dma_queue, 
        local_segment,
        remote_segment,
        offsetof(struct server_segment, mv_y_buffer),
        enc_header.mv_y_size,
        offsetof(struct client_segment, mv_y_buffer),
        NO_CALLBACK,
        NULL,
        NO_FLAGS,
        &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "Server: SCIStartDmaTransfer for Y motion vectors failed - Error code 0x%x\n", error);
        return -1;
    }
    
    SCIWaitForDMAQueue(dma_queue, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "Server: SCIWaitForDMAQueue for Y motion vectors failed - Error code 0x%x\n", error);
        return -1;
    }
    
    // Transfer U motion vectors
    SCIStartDmaTransfer(dma_queue, 
        local_segment,
        remote_segment,
        offsetof(struct server_segment, mv_u_buffer),
        enc_header.mv_u_size,
        offsetof(struct client_segment, mv_u_buffer),
        NO_CALLBACK,
        NULL,
        NO_FLAGS,
        &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "Server: SCIStartDmaTransfer for U motion vectors failed - Error code 0x%x\n", error);
        return -1;
    }
    
    SCIWaitForDMAQueue(dma_queue, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "Server: SCIWaitForDMAQueue for U motion vectors failed - Error code 0x%x\n", error);
        return -1;
    }
    
    // Transfer V motion vectors
    SCIStartDmaTransfer(dma_queue, 
        local_segment,
        remote_segment,
        offsetof(struct server_segment, mv_v_buffer),
        enc_header.mv_v_size,
        offsetof(struct client_segment, mv_v_buffer),
        NO_CALLBACK,
        NULL,
        NO_FLAGS,
        &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "Server: SCIStartDmaTransfer for V motion vectors failed - Error code 0x%x\n", error);
        return -1;
    }
    
    SCIWaitForDMAQueue(dma_queue, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "Server: SCIWaitForDMAQueue for V motion vectors failed - Error code 0x%x\n", error);
        return -1;
    }
    
    // Notify client that motion vectors are ready
    SCIFlush(NULL, NO_FLAGS);
    remote_seg->packet.cmd = CMD_MOTION_VECTORS_READY;
    SCIFlush(NULL, NO_FLAGS);
    
    // Wait for acknowledgment
    local_seg->packet.cmd = CMD_INVALID;
    while (local_seg->packet.cmd != CMD_MOTION_VECTORS_ACK) {
        // Wait for acknowledgment
    }
    
    // Notify client that all encoded data has been transferred
    printf("Server: Completed sending encoded data for frame %d\n", frame_number);
    SCIFlush(NULL, NO_FLAGS);
    remote_seg->packet.cmd = CMD_FRAME_ENCODED;
    SCIFlush(NULL, NO_FLAGS);

    return is_last_frame;
}

int main_loop(sci_desc_t sd,
    volatile struct server_segment *local_seg,
    volatile struct client_segment *remote_seg,
    sci_dma_queue_t dma_queue,
    sci_local_segment_t local_segment,
    sci_remote_segment_t remote_segment)
{
    int running = 1;
    uint32_t cmd;
    sci_error_t error;
    int frame_count = 0;
    struct c63_common *cm = NULL;

    printf("Server: Waiting for commands...\n");

    while(running) {
        // Wait for command from client
        if (local_seg->packet.cmd == CMD_INVALID) {
            continue; // Keep waiting
        }

        // Process command
        cmd = local_seg->packet.cmd;

        switch(cmd) {
            case CMD_DIMENSIONS:
                // Extract the dimensions from the message buffer
                struct dimensions_data dim_data;
                memcpy(&dim_data, (const void*)local_seg->message_buffer, sizeof(struct dimensions_data));
                
                printf("Server: Received dimensions (width=%u, height=%u)\n", 
                        dim_data.width, dim_data.height);
                
                // Store dimensions for later use
                width = dim_data.width;
                height = dim_data.height;
                
                // Initialize encoder
                cm = init_c63_enc(width, height);
                
                // Initialize threading resources
                thread_pool_init();
                task_pool_init(cm->padh[Y_COMPONENT]);
                
                // Acknowledge dimensions
                local_seg->packet.cmd = CMD_INVALID;
                SCIFlush(NULL, NO_FLAGS);
                
                // Send acknowledgment
                memcpy((void*)local_seg->message_buffer, &dim_data, sizeof(struct dimensions_data));
                local_seg->packet.data_size = sizeof(struct dimensions_data);
                
                // Use DMA to transfer the acknowledgment
                SCIStartDmaTransfer(dma_queue, 
                                    local_segment,
                                    remote_segment,
                                    offsetof(struct server_segment, message_buffer),
                                    sizeof(struct dimensions_data),
                                    offsetof(struct client_segment, message_buffer),
                                    NO_CALLBACK,
                                    NULL,
                                    NO_FLAGS,
                                    &error);
                                    
                // [Error handling...]
                
                remote_seg->packet.cmd = CMD_DIMENSIONS_ACK;
                SCIFlush(NULL, NO_FLAGS);
                
                printf("Server: Dimensions acknowledged\n");
                printf("Server: Waiting for frames...\n");
                break;
                
            // Process frame components (handled by process_frame function)
            case CMD_FRAME_HEADER:
            case CMD_Y_DATA_READY:
            case CMD_U_DATA_READY:
            case CMD_V_DATA_READY:
                {
                    int is_last = process_frame(cm, local_seg, remote_seg, dma_queue, local_segment, remote_segment);
                    frame_count++;
                    
                    if (is_last) {
                        printf("Server: Last frame processed. Total frames: %d\n", frame_count);
                        running = 0;
                    }
                }
                break;
                
            case CMD_QUIT:
                printf("Server: Received quit command after processing %d frames\n", frame_count);
                running = 0;
                break;
                
            // Handle acknowledgment commands from the new protocol
            case CMD_ENCODED_DATA_HEADER_ACK:
            case CMD_RESIDUALS_Y_ACK:
            case CMD_RESIDUALS_U_ACK:
            case CMD_RESIDUALS_V_ACK:
            case CMD_MOTION_VECTORS_ACK:
                // These are acknowledgments from the client during the data transfer
                // We don't need to do anything special here, just clear the command
                local_seg->packet.cmd = CMD_INVALID;
                SCIFlush(NULL, NO_FLAGS);
                break;
                
            default:
                printf("Server: Unknown command: %d\n", cmd);
                local_seg->packet.cmd = CMD_INVALID;
                SCIFlush(NULL, NO_FLAGS);
                break;
        }
    }

    // Clean up resources
    if (cm) {
        free_c63_enc(cm);
    }

    task_pool_destroy();
    thread_pool_destroy();

    return 0;
}

static void print_help()
{
    printf("Usage: ./c63server -r nodeid\n");
    printf("Commandline options:\n");
    printf("  -r                             Node id of client\n");
    printf("\n");

    exit(EXIT_FAILURE);
}

int main(int argc, char **argv)
{
    unsigned int localAdapterNo = 0;
    unsigned int remoteNodeId = 0;

    sci_error_t error;
    sci_desc_t sd;
    sci_remote_segment_t remoteSegment;
    sci_local_segment_t localSegment;
    sci_dma_queue_t dmaQueue;
    sci_map_t localMap, remoteMap;

    volatile struct server_segment *server_segment;
    volatile struct client_segment *client_segment;

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

    // Open virtual device
    SCIOpen(&sd, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCIOpen failed - Error code 0x%x\n", error);
        SCITerminate();
        exit(EXIT_FAILURE);
    }

    // Create local segment
    SCICreateSegment(sd,
                &localSegment,
                SEGMENT_SERVER,
                sizeof(struct server_segment),
                NO_CALLBACK,
                NULL,
                NO_FLAGS,
                &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCICreateSegment failed - Error code 0x%x\n", error);
        SCIClose(sd, NO_FLAGS, &error);
        SCITerminate();
        exit(EXIT_FAILURE);
    }

    // Prepare segment
    SCIPrepareSegment(localSegment, localAdapterNo, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCIPrepareSegment failed - Error code 0x%x\n", error);
        SCIRemoveSegment(localSegment, NO_FLAGS, &error);
        SCIClose(sd, NO_FLAGS, &error);
        SCITerminate();
        exit(EXIT_FAILURE);
    }

    // Make segment available
    SCISetSegmentAvailable(localSegment, localAdapterNo, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCISetSegmentAvailable failed - Error code 0x%x\n", error);
        SCIRemoveSegment(localSegment, NO_FLAGS, &error);
        SCIClose(sd, NO_FLAGS, &error);
        SCITerminate();
        exit(EXIT_FAILURE);
    }

    // Create DMA queue
    SCICreateDMAQueue(sd, &dmaQueue, localAdapterNo, 1, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCICreateDMAQueue failed - Error code 0x%x\n", error);
        SCISetSegmentUnavailable(localSegment, localAdapterNo, NO_FLAGS, &error);
        SCIRemoveSegment(localSegment, NO_FLAGS, &error);
        SCIClose(sd, NO_FLAGS, &error);
        SCITerminate();
        exit(EXIT_FAILURE);
    }

    // Map local segment
    server_segment = (volatile struct server_segment *)SCIMapLocalSegment(
        localSegment, 
        &localMap, 
        0, 
        sizeof(struct server_segment), 
        NULL, 
        NO_FLAGS, 
        &error);

    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCIMapLocalSegment failed - Error code 0x%x\n", error);
        SCIRemoveDMAQueue(dmaQueue, NO_FLAGS, &error);
        SCISetSegmentUnavailable(localSegment, localAdapterNo, NO_FLAGS, &error);
        SCIRemoveSegment(localSegment, NO_FLAGS, &error);
        SCIClose(sd, NO_FLAGS, &error);
        SCITerminate();
        exit(EXIT_FAILURE);
    }

    // Initialize control packet
    server_segment->packet.cmd = CMD_INVALID;

    printf("Server: Waiting for client connection...\n");

    // Connect to client segment
    do {
        SCIConnectSegment(sd,
                        &remoteSegment,
                        remoteNodeId,
                        SEGMENT_CLIENT,
                        localAdapterNo,
                        NO_CALLBACK,
                        NULL,
                        SCI_INFINITE_TIMEOUT,
                        NO_FLAGS,
                        &error);
    } while (error != SCI_ERR_OK);

    printf("Server: Connected to client segment\n");

    // Map remote segment
    client_segment = (volatile struct client_segment *)SCIMapRemoteSegment(
        remoteSegment, 
        &remoteMap, 
        0,
        sizeof(struct client_segment),
        NULL, 
        NO_FLAGS, 
        &error);

    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCIMapRemoteSegment failed - Error code 0x%x\n", error);
        SCIDisconnectSegment(remoteSegment, NO_FLAGS, &error);
        SCIUnmapSegment(localMap, NO_FLAGS, &error);
        SCIRemoveDMAQueue(dmaQueue, NO_FLAGS, &error);
        SCISetSegmentUnavailable(localSegment, localAdapterNo, NO_FLAGS, &error);
        SCIRemoveSegment(localSegment, NO_FLAGS, &error);
        SCIClose(sd, NO_FLAGS, &error);
        SCITerminate();
        exit(EXIT_FAILURE);
    }

    // Enter main loop
    main_loop(sd, server_segment, client_segment, dmaQueue, localSegment, remoteSegment);

    printf("Server: Exiting\n");
    
    // Clean up resources
    SCIDisconnectSegment(remoteSegment, NO_FLAGS, &error);
    SCIUnmapSegment(remoteMap, NO_FLAGS, &error);
    SCIUnmapSegment(localMap, NO_FLAGS, &error);
    SCIRemoveDMAQueue(dmaQueue, NO_FLAGS, &error);
    SCISetSegmentUnavailable(localSegment, localAdapterNo, NO_FLAGS, &error);
    SCIRemoveSegment(localSegment, NO_FLAGS, &error);
    SCIClose(sd, NO_FLAGS, &error);
    SCITerminate();

    return 0;
}