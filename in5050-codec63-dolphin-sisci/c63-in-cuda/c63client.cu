#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sisci_error.h>
#include <sisci_api.h>

#include "c63.h"
#include "c63_write.h"
#include "common.h"
#include "tables.h"

static char *output_file, *input_file;
FILE *outfile;

static uint32_t remote_node = 0;
static int limit_numframes = 0;

static uint32_t width;
static uint32_t height;

/* getopt */
extern int optind;
extern char *optarg;

/* Read planar YUV frames with 4:2:0 chroma sub-sampling */
static yuv_t *read_yuv(FILE *file, struct c63_common *cm)
{
    size_t len = 0;
    yuv_t *image = (yuv_t *)malloc(sizeof(*image));

    /* Read Y. The size of Y is the same as the size of the image. */
    image->Y = (uint8_t *)calloc(1, cm->padw[Y_COMPONENT] * cm->padh[Y_COMPONENT]);
    len += fread(image->Y, 1, width * height, file);

    /* Read U. Given 4:2:0 chroma sub-sampling, the size is 1/4 of Y */
    image->U = (uint8_t *)calloc(1, cm->padw[U_COMPONENT] * cm->padh[U_COMPONENT]);
    len += fread(image->U, 1, (width * height) / 4, file);

    /* Read V. Given 4:2:0 chroma sub-sampling, the size is 1/4 of Y. */
    image->V = (uint8_t *)calloc(1, cm->padw[V_COMPONENT] * cm->padh[V_COMPONENT]);
    len += fread(image->V, 1, (width * height) / 4, file);

    if (ferror(file))
    {
        perror("ferror");
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

void
free_c63_enc( struct c63_common *cm )
{
    destroy_frame( cm->curframe );
    free( cm );
}

int send_frame_to_server(int frame_number, yuv_t *image, 
    volatile struct client_segment *local_seg,
    volatile struct server_segment *remote_seg,
    sci_dma_queue_t dma_queue,
    sci_local_segment_t local_segment,
    sci_remote_segment_t remote_segment,
    uint32_t width, uint32_t height,
    int is_last_frame,
    struct c63_common *cm) 
{
    sci_error_t error;

    // Calculate component sizes based on width & height
    uint32_t y_size = width * height;
    uint32_t u_size = y_size / 4;
    uint32_t v_size = y_size / 4;

    // 1. Send frame header with frame number
    struct frame_header header;
    header.frame_number = frame_number;
    header.is_last_frame = is_last_frame;

    memcpy((void*)local_seg->message_buffer, &header, sizeof(struct frame_header));
    local_seg->packet.data_size = sizeof(struct frame_header);

    // Start DMA transfer for header
    SCIStartDmaTransfer(dma_queue, 
        local_segment,
        remote_segment,
        offsetof(struct client_segment, message_buffer),
        sizeof(struct frame_header),
        offsetof(struct server_segment, message_buffer),
        NO_CALLBACK,
        NULL,
        NO_FLAGS,
        &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "Client: SCIStartDmaTransfer for frame header failed - Error code 0x%x\n", error);
        return -1;
    }

    // Wait for DMA transfer to complete
    SCIWaitForDMAQueue(dma_queue, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "Client: SCIWaitForDMAQueue for frame header failed - Error code 0x%x\n", error);
        return -1;
    }

    // Notify server that frame header is ready
    SCIFlush(NULL, NO_FLAGS);
    remote_seg->packet.cmd = CMD_FRAME_HEADER;
    SCIFlush(NULL, NO_FLAGS);

    // Wait for server to acknowledge
    local_seg->packet.cmd = CMD_INVALID;
    while (local_seg->packet.cmd != CMD_FRAME_HEADER_ACK) {
        // Wait for acknowledgment
    }

    printf("Frame %d header sent and acknowledged\n", frame_number);

    // 2. Transfer Y component
    memcpy((void*)local_seg->y_buffer, image->Y, y_size);

    SCIStartDmaTransfer(dma_queue, 
                        local_segment,
                        remote_segment,
                        offsetof(struct client_segment, y_buffer),
                        y_size,
                        offsetof(struct server_segment, y_buffer),
                        NO_CALLBACK,
                        NULL,
                        NO_FLAGS,
                        &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "Client: SCIStartDmaTransfer for Y component failed - Error code 0x%x\n", error);
        return -1;
    }

    SCIWaitForDMAQueue(dma_queue, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "Client: SCIWaitForDMAQueue for Y component failed - Error code 0x%x\n", error);
        return -1;
    }

    // Notify server that Y data is ready
    SCIFlush(NULL, NO_FLAGS);
    remote_seg->packet.cmd = CMD_Y_DATA_READY;
    SCIFlush(NULL, NO_FLAGS);

    // Wait for acknowledgment
    local_seg->packet.cmd = CMD_INVALID;
    while (local_seg->packet.cmd != CMD_Y_DATA_ACK) {
        // Wait for acknowledgment
    }

    printf("Frame %d Y component sent and acknowledged\n", frame_number);

    // 3. Transfer U component
    memcpy((void*)local_seg->u_buffer, image->U, u_size);

    SCIStartDmaTransfer(dma_queue, 
                        local_segment,
                        remote_segment,
                        offsetof(struct client_segment, u_buffer),
                        u_size,
                        offsetof(struct server_segment, u_buffer),
                        NO_CALLBACK,
                        NULL,
                        NO_FLAGS,
                        &error);
                        if (error != SCI_ERR_OK) {
        fprintf(stderr, "Client: SCIStartDmaTransfer for U component failed - Error code 0x%x\n", error);
        return -1;
    }

    SCIWaitForDMAQueue(dma_queue, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "Client: SCIWaitForDMAQueue for U component failed - Error code 0x%x\n", error);
        return -1;
    }

    // Notify server that U data is ready
    SCIFlush(NULL, NO_FLAGS);
    remote_seg->packet.cmd = CMD_U_DATA_READY;
    SCIFlush(NULL, NO_FLAGS);

    // Wait for acknowledgment
    local_seg->packet.cmd = CMD_INVALID;
    while (local_seg->packet.cmd != CMD_U_DATA_ACK) {
        // Wait for acknowledgment
    }

    printf("Frame %d U component sent and acknowledged\n", frame_number);

    // 4. Transfer V component
    memcpy((void*)local_seg->v_buffer, image->V, v_size);

    SCIStartDmaTransfer(dma_queue, 
                        local_segment,
                        remote_segment,
                        offsetof(struct client_segment, v_buffer),
                        v_size,
                        offsetof(struct server_segment, v_buffer),
                        NO_CALLBACK,
                        NULL,
                        NO_FLAGS,
                        &error);
                        if (error != SCI_ERR_OK) {
    fprintf(stderr, "Client: SCIStartDmaTransfer for V component failed - Error code 0x%x\n", error);
        return -1;
    }

    SCIWaitForDMAQueue(dma_queue, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "Client: SCIWaitForDMAQueue for V component failed - Error code 0x%x\n", error);
        return -1;
    }

    // Notify server that V data is ready
    SCIFlush(NULL, NO_FLAGS);
    remote_seg->packet.cmd = CMD_V_DATA_READY;
    SCIFlush(NULL, NO_FLAGS);

    // Wait for acknowledgment
    local_seg->packet.cmd = CMD_INVALID;
    while (local_seg->packet.cmd != CMD_V_DATA_ACK) {
        // Wait for acknowledgment
    }

    printf("Frame %d V component sent and acknowledged\n", frame_number);

    // 5. Wait for frame encoding completion
   // 5. Receive encoded data from server
    printf("Frame %d waiting for encoded data...\n", frame_number);

    // Wait for encoded data header
    while (local_seg->packet.cmd != CMD_ENCODED_DATA_HEADER) {
        // Wait for header
    }

    // Get the encoded header
    struct encoded_frame_header enc_header;
    memcpy(&enc_header, (const void*)local_seg->message_buffer, sizeof(struct encoded_frame_header));

    // Acknowledge encoded header
    local_seg->packet.cmd = CMD_INVALID;
    SCIFlush(NULL, NO_FLAGS);
    remote_seg->packet.cmd = CMD_ENCODED_DATA_HEADER_ACK;
    SCIFlush(NULL, NO_FLAGS);

    printf("Frame %d received encoded header (keyframe=%d)\n", frame_number, enc_header.keyframe);

    // Prepare frame structure in client's cm
    if (cm->curframe == NULL) {
        cm->curframe = (struct frame*)calloc(1, sizeof(struct frame));    
    }

    // Allocate residuals if needed
    if (cm->curframe->residuals == NULL) {
        cm->curframe->residuals = (dct_t*)calloc(1, sizeof(dct_t));
    }

    // Allocate or reallocate DCT buffers as needed
    cm->curframe->residuals->Ydct = (int16_t*)realloc(cm->curframe->residuals->Ydct, enc_header.y_size);
    cm->curframe->residuals->Udct = (int16_t*)realloc(cm->curframe->residuals->Udct, enc_header.u_size);
    cm->curframe->residuals->Vdct = (int16_t*)realloc(cm->curframe->residuals->Vdct, enc_header.v_size);

    // Receive Y residuals
    local_seg->packet.cmd = CMD_INVALID;
    while (local_seg->packet.cmd != CMD_RESIDUALS_Y_READY) {
        // Wait for Y residuals
    }

    // Copy Y residuals
    memcpy(cm->curframe->residuals->Ydct, (const void*)local_seg->y_buffer, enc_header.y_size);

    // Acknowledge Y residuals
    local_seg->packet.cmd = CMD_INVALID;
    SCIFlush(NULL, NO_FLAGS);
    remote_seg->packet.cmd = CMD_RESIDUALS_Y_ACK;
    SCIFlush(NULL, NO_FLAGS);

    printf("Frame %d received Y residuals\n", frame_number);

    // Receive U residuals
    local_seg->packet.cmd = CMD_INVALID;
    while (local_seg->packet.cmd != CMD_RESIDUALS_U_READY) {
        // Wait for U residuals
    }

    // Copy U residuals
    memcpy(cm->curframe->residuals->Udct, (const void*)local_seg->u_buffer, enc_header.u_size);

    // Acknowledge U residuals
    local_seg->packet.cmd = CMD_INVALID;
    SCIFlush(NULL, NO_FLAGS);
    remote_seg->packet.cmd = CMD_RESIDUALS_U_ACK;
    SCIFlush(NULL, NO_FLAGS);

    printf("Frame %d received U residuals\n", frame_number);

    // Receive V residuals
    local_seg->packet.cmd = CMD_INVALID;
    while (local_seg->packet.cmd != CMD_RESIDUALS_V_READY) {
        // Wait for V residuals
    }

    // Copy V residuals
    memcpy(cm->curframe->residuals->Vdct, (const void*)local_seg->v_buffer, enc_header.v_size);

    // Acknowledge V residuals
    local_seg->packet.cmd = CMD_INVALID;
    SCIFlush(NULL, NO_FLAGS);
    remote_seg->packet.cmd = CMD_RESIDUALS_V_ACK;
    SCIFlush(NULL, NO_FLAGS);

    printf("Frame %d received V residuals\n", frame_number);

    // Receive motion vectors
    local_seg->packet.cmd = CMD_INVALID;
    while (local_seg->packet.cmd != CMD_MOTION_VECTORS_READY) {
        // Wait for motion vectors
    }

    // Allocate memory for motion vectors if needed
    if (cm->curframe->mbs[Y_COMPONENT] == NULL) {
        cm->curframe->mbs[Y_COMPONENT] = (struct macroblock*)calloc(cm->mb_rows * cm->mb_cols, sizeof(struct macroblock));
    }
    if (cm->curframe->mbs[U_COMPONENT] == NULL) {
        cm->curframe->mbs[U_COMPONENT] = (struct macroblock*)calloc((cm->mb_rows/2) * (cm->mb_cols/2), sizeof(struct macroblock));
    }
    if (cm->curframe->mbs[V_COMPONENT] == NULL) {
        cm->curframe->mbs[V_COMPONENT] = (struct macroblock*)calloc((cm->mb_rows/2) * (cm->mb_cols/2), sizeof(struct macroblock));
    }

    // Copy motion vectors
    memcpy(cm->curframe->mbs[Y_COMPONENT], (const void*)local_seg->mv_y_buffer, enc_header.mv_y_size);
    memcpy(cm->curframe->mbs[U_COMPONENT], (const void*)local_seg->mv_u_buffer, enc_header.mv_u_size);
    memcpy(cm->curframe->mbs[V_COMPONENT], (const void*)local_seg->mv_v_buffer, enc_header.mv_v_size);

    // Acknowledge motion vectors
    local_seg->packet.cmd = CMD_INVALID;
    SCIFlush(NULL, NO_FLAGS);
    remote_seg->packet.cmd = CMD_MOTION_VECTORS_ACK;
    SCIFlush(NULL, NO_FLAGS);

    printf("Frame %d received motion vectors\n", frame_number);

    // Set keyframe flag
    cm->curframe->keyframe = enc_header.keyframe;

    // Wait for final acknowledgment that all data is transferred
    local_seg->packet.cmd = CMD_INVALID;
    while (local_seg->packet.cmd != CMD_FRAME_ENCODED) {
        // Wait for encoding completion
    }

    printf("Frame %d processing and data transfer completed\n", frame_number);

    // Write the encoded frame to file
    write_frame(cm);

    return 0;
}

/* Main client processing loop */
int main_client_loop(struct c63_common *cm, FILE *infile, int limit_numframes,
    volatile struct client_segment *local_seg,
    volatile struct server_segment *remote_seg,
    sci_dma_queue_t dma_queue,
    sci_local_segment_t local_segment,
    sci_remote_segment_t remote_segment) 
{
    yuv_t *image;
    int numframes = 0;
    sci_error_t error;

    printf("Client: Sending dimensions (width=%u, height=%u)\n", width, height);

    // Send dimensions first
    struct dimensions_data dim_data;
    dim_data.width = width;
    dim_data.height = height;

    memcpy((void*)local_seg->message_buffer, &dim_data, sizeof(struct dimensions_data));
    local_seg->packet.data_size = sizeof(struct dimensions_data);

    // Use DMA to transfer the dimensions
    SCIStartDmaTransfer(dma_queue, 
        local_segment,
        remote_segment,
        offsetof(struct client_segment, message_buffer),
        sizeof(struct dimensions_data),
        offsetof(struct server_segment, message_buffer),
        NO_CALLBACK,
        NULL,
        NO_FLAGS,
        &error);
        
    SCIWaitForDMAQueue(dma_queue, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "Client: SCIWaitForDMAQueue failed - Error code 0x%x\n", error);
        return -1;
    }
    
    // Notify server that dimensions are ready
    SCIFlush(NULL, NO_FLAGS);
    remote_seg->packet.cmd = CMD_DIMENSIONS;
    SCIFlush(NULL, NO_FLAGS);
    
    // Wait for server to acknowledge dimensions
    printf("Client: Waiting for dimensions acknowledgment...\n");
    local_seg->packet.cmd = CMD_INVALID;
    while (local_seg->packet.cmd != CMD_DIMENSIONS_ACK) {
        // Wait for acknowledgment
    }
    printf("Client: Dimensions verified, starting video encoding\n");

    // Process each frame
    while (1) {
    // Read YUV frame
        image = read_yuv(infile, cm);
        if (!image) break;

        // Check if this is the last frame
        int is_last = (limit_numframes && numframes == limit_numframes - 1);

        // Send frame to server
        printf("Processing frame %d, ", numframes);

        int result = send_frame_to_server(
            numframes, 
            image, 
            local_seg, 
            remote_seg, 
            dma_queue, 
            local_segment, 
            remote_segment, 
            width, 
            height, 
            is_last || !image,
            cm
            );

        if (result < 0) {
        // Handle error
        fprintf(stderr, "Error sending frame %d\n", numframes);
            break;
        }

        // Clean up the image
        free(image->Y);
        free(image->U);
        free(image->V);
        free(image);

        printf("Done!\n");

        ++numframes;

        if (limit_numframes && numframes >= limit_numframes) {
            break;
        }
    }

    // Signal server to quit if we exited due to other reasons
    remote_seg->packet.cmd = CMD_QUIT;
    SCIFlush(NULL, NO_FLAGS);

    printf("Client: Finished processing %d frames\n", numframes);
    return numframes;
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
    yuv_t *image;
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

    // Open output file
    outfile = fopen(output_file, "wb");
    if (outfile == NULL)
    {
        perror("fopen");
        exit(EXIT_FAILURE);
    }

    // Initialize encoder
    struct c63_common *cm = init_c63_enc(width, height);
    cm->e_ctx.fp = outfile;

    if (limit_numframes)
    {
        printf("Limited to %d frames.\n", limit_numframes);
    }

    // Open input file
    FILE *infile = fopen(input_file, "rb");
    if (infile == NULL)
    {
        perror("fopen");
        exit(EXIT_FAILURE);
    }

    // Initialize SISCI
    SCIInitialize(NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCIInitialize failed: %s\n", SCIGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Set up SISCI resources
    sci_desc_t sd;
    sci_local_segment_t localSegment;
    sci_remote_segment_t remoteSegment;
    sci_map_t localMap, remoteMap;
    sci_dma_queue_t dmaQueue;
    volatile struct client_segment *client_segment;
    volatile struct server_segment *server_segment;

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
                     SEGMENT_CLIENT,
                     sizeof(struct client_segment),
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
    
    // Create DMA queue
    SCICreateDMAQueue(sd, &dmaQueue, localAdapterNo, 1, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCICreateDMAQueue failed - Error code 0x%x\n", error);
        SCIRemoveSegment(localSegment, NO_FLAGS, &error);
        SCIClose(sd, NO_FLAGS, &error);
        SCITerminate();
        exit(EXIT_FAILURE);
    }
    
    // Map local segment
    client_segment = (volatile struct client_segment *)SCIMapLocalSegment(
        localSegment, 
        &localMap, 
        0, 
        sizeof(struct client_segment), 
        NULL, 
        NO_FLAGS, 
        &error);
    
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCIMapLocalSegment failed - Error code 0x%x\n", error);
        SCIRemoveDMAQueue(dmaQueue, NO_FLAGS, &error);
        SCIRemoveSegment(localSegment, NO_FLAGS, &error);
        SCIClose(sd, NO_FLAGS, &error);
        SCITerminate();
        exit(EXIT_FAILURE);
    }
    
    // Initialize control packet
    client_segment->packet.cmd = CMD_INVALID;
    
    // Make segment available
    SCISetSegmentAvailable(localSegment, localAdapterNo, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCISetSegmentAvailable failed - Error code 0x%x\n", error);
        SCIUnmapSegment(localMap, NO_FLAGS, &error);
        SCIRemoveDMAQueue(dmaQueue, NO_FLAGS, &error);
        SCIRemoveSegment(localSegment, NO_FLAGS, &error);
        SCIClose(sd, NO_FLAGS, &error);
        SCITerminate();
        exit(EXIT_FAILURE);
    }
    
    printf("Client: Connecting to server segment...\n");
    
    // Connect to server segment
    do {
        SCIConnectSegment(sd,
                          &remoteSegment,
                          remote_node,
                          SEGMENT_SERVER,
                          localAdapterNo,
                          NO_CALLBACK,
                          NULL,
                          SCI_INFINITE_TIMEOUT,
                          NO_FLAGS,
                          &error);
    } while (error != SCI_ERR_OK);
    
    printf("Client: Connected to server segment\n");
    
    // Map remote segment
    server_segment = (volatile struct server_segment *)SCIMapRemoteSegment(
        remoteSegment, 
        &remoteMap, 
        0,
        sizeof(struct server_segment),
        NULL, 
        NO_FLAGS, 
        &error);
    
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCIMapRemoteSegment failed - Error code 0x%x\n", error);
        SCIDisconnectSegment(remoteSegment, NO_FLAGS, &error);
        SCISetSegmentUnavailable(localSegment, localAdapterNo, NO_FLAGS, &error);
        SCIUnmapSegment(localMap, NO_FLAGS, &error);
        SCIRemoveDMAQueue(dmaQueue, NO_FLAGS, &error);
        SCIRemoveSegment(localSegment, NO_FLAGS, &error);
        SCIClose(sd, NO_FLAGS, &error);
        SCITerminate();
        exit(EXIT_FAILURE);
    }
    
    // Enter main processing loop
    main_client_loop(cm, infile, limit_numframes, client_segment, server_segment, 
                     dmaQueue, localSegment, remoteSegment);
    
    // Clean up resources
    fclose(outfile);
    fclose(infile);
    free_c63_enc(cm);
    
    SCIDisconnectSegment(remoteSegment, NO_FLAGS, &error);
    SCIUnmapSegment(remoteMap, NO_FLAGS, &error);
    SCISetSegmentUnavailable(localSegment, localAdapterNo, NO_FLAGS, &error);
    SCIUnmapSegment(localMap, NO_FLAGS, &error);
    SCIRemoveDMAQueue(dmaQueue, NO_FLAGS, &error);
    SCIRemoveSegment(localSegment, NO_FLAGS, &error);
    SCIClose(sd, NO_FLAGS, &error);
    SCITerminate();
    
    return 0;
}