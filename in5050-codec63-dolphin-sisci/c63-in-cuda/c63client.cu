#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

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
    
    printf("Client: Starting video encoding\n");
    
    // Send dimensions to server
    struct dimensions_data dim_data;
    dim_data.width = width;
    dim_data.height = height;
    
    // Place dimensions in local buffer
    memcpy((void*)local_seg->message_buffer, &dim_data, sizeof(struct dimensions_data));
    
    // Transfer dimensions to server's segment
    SCIStartDmaTransfer(dma_queue, 
                       local_segment,
                       remote_segment,
                       offsetof(struct client_segment, message_buffer),  // Source offset
                       sizeof(struct dimensions_data),                  // Size to transfer
                       offsetof(struct server_segment, message_buffer),  // Destination offset
                       NO_CALLBACK,
                       NULL,
                       NO_FLAGS,
                       &error);
                       
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "Client: SCIStartDmaTransfer for dimensions failed - Error code 0x%x\n", error);
        return -1;
    }
    
    // Wait for transfer to complete
    SCIWaitForDMAQueue(dma_queue, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
    
    // Signal server that dimensions are ready
    SCIFlush(NULL, NO_FLAGS);
    remote_seg->packet.cmd = CMD_DIMENSIONS;
    SCIFlush(NULL, NO_FLAGS);
    
    // Wait for server to acknowledge dimensions
    printf("Client: Waiting for server to acknowledge dimensions\n");
    time_t dim_start = time(NULL);
    bool dim_timeout = false;
    
    while (local_seg->packet.cmd != CMD_DIMENSIONS_ACK && !dim_timeout) {
        if (time(NULL) - dim_start > 30) {  // 30 second timeout
            dim_timeout = true;
            fprintf(stderr, "Client: Timeout waiting for dimensions acknowledgment\n");
        }
    }
    
    if (dim_timeout) {
        fprintf(stderr, "Client: Failed to receive dimensions acknowledgment, exiting\n");
        return -1;
    }
    
    printf("Client: Dimensions acknowledged by server\n");
    local_seg->packet.cmd = CMD_INVALID;  // Reset command
    
    // Frame processing loop
    while (1) {
        // Read YUV frame
        image = read_yuv(infile, cm);
        if (!image) {
            printf("Client: End of input file reached\n");
            break;
        }
        
        printf("Processing frame %d, ", numframes);
        
        // Calculate sizes of YUV components
        size_t y_size = width * height;               // Y plane
        size_t u_size = (width * height) / 4;         // U plane
        size_t v_size = (width * height) / 4;         // V plane
        
        printf("Client: YUV plane sizes - Y: %zu bytes, U: %zu bytes, V: %zu bytes\n", 
               y_size, u_size, v_size);
        
        // Verify buffer size is adequate
        if (y_size > MESSAGE_SIZE || u_size > MESSAGE_SIZE || v_size > MESSAGE_SIZE) {
            fprintf(stderr, "Client: ERROR - YUV plane size exceeds message buffer size (%d)\n", 
                    MESSAGE_SIZE);
            free(image->Y);
            free(image->U);
            free(image->V);
            free(image);
            return -1;
        }
        
        //
        // TRANSFER Y PLANE
        //
        printf("Client: Transferring Y plane (%zu bytes)\n", y_size);
        
        // Copy Y data to message buffer
        memcpy((void*)local_seg->message_buffer, image->Y, y_size);
        
        // Start DMA transfer to server
        SCIStartDmaTransfer(dma_queue, 
                           local_segment,
                           remote_segment,
                           offsetof(struct client_segment, message_buffer),
                           y_size,
                           offsetof(struct server_segment, message_buffer),
                           NO_CALLBACK,
                           NULL,
                           NO_FLAGS,
                           &error);
        
        if (error != SCI_ERR_OK) {
            fprintf(stderr, "Client: Y plane DMA transfer failed - Error code 0x%x\n", error);
            free(image->Y);
            free(image->U);
            free(image->V);
            free(image);
            continue;  // Try next frame
        }
        
        // Wait for Y plane transfer to complete
        SCIWaitForDMAQueue(dma_queue, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
        printf("Client: Y plane DMA transfer complete\n");
        
        // Signal server that Y plane is ready
        SCIFlush(NULL, NO_FLAGS);
        remote_seg->packet.cmd = CMD_YUV_DATA;
        remote_seg->packet.data_size = y_size;
        SCIFlush(NULL, NO_FLAGS);
        printf("Client: Y plane sent, waiting for acknowledgment\n");
        
        // Wait for Y plane acknowledgment
        time_t y_start = time(NULL);
        bool y_timeout = false;
        
        while (local_seg->packet.cmd != CMD_YUV_DATA_ACK && !y_timeout) {
            if (time(NULL) - y_start > 30) {  // 30 second timeout
                y_timeout = true;
                fprintf(stderr, "Client: Timeout waiting for Y plane acknowledgment\n");
            }
        }
        
        if (y_timeout) {
            fprintf(stderr, "Client: Failed to get Y plane acknowledgment, skipping frame\n");
            free(image->Y);
            free(image->U);
            free(image->V);
            free(image);
            continue;  // Try next frame
        }
        
        printf("Client: Y plane acknowledgment received from server\n");
        local_seg->packet.cmd = CMD_INVALID;  // Reset command
        printf("Client: Reset command flag. Current command value: %d\n", local_seg->packet.cmd);
        
        //
        // TRANSFER U PLANE
        //
        printf("Client: Starting U plane transfer process\n");
        printf("Client: U plane size: %zu bytes\n", u_size);
        
        // Copy U data to message buffer
        printf("Client: Copying U plane to message buffer\n");
        memcpy((void*)local_seg->message_buffer, image->U, u_size);
        
        // Start DMA transfer to server
        printf("Client: Initiating DMA transfer for U plane\n");
        SCIStartDmaTransfer(dma_queue, 
                           local_segment,
                           remote_segment,
                           offsetof(struct client_segment, message_buffer),
                           u_size,
                           offsetof(struct server_segment, message_buffer),
                           NO_CALLBACK,
                           NULL,
                           NO_FLAGS,
                           &error);
        
        if (error != SCI_ERR_OK) {
            fprintf(stderr, "Client: U plane DMA transfer failed - Error code 0x%x\n", error);
            free(image->Y);
            free(image->U);
            free(image->V);
            free(image);
            continue;  // Try next frame
        }
        
        // Wait for U plane transfer to complete
        printf("Client: Waiting for U plane DMA transfer to complete\n");
        SCIWaitForDMAQueue(dma_queue, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
        printf("Client: U plane DMA transfer completed\n");
        
        // Signal server that U plane is ready
        printf("Client: Sending U plane ready signal to server\n");
        SCIFlush(NULL, NO_FLAGS);
        remote_seg->packet.cmd = CMD_YUV_DATA;
        remote_seg->packet.data_size = u_size;
        SCIFlush(NULL, NO_FLAGS);
        printf("Client: U plane ready signal sent. Command value: %d, Data size: %u\n", 
               remote_seg->packet.cmd, remote_seg->packet.data_size);
        
        // Wait for U plane acknowledgment
        printf("Client: Waiting for U plane acknowledgment\n");
        time_t u_start = time(NULL);
        bool u_timeout = false;
        
        while (local_seg->packet.cmd != CMD_YUV_DATA_ACK && !u_timeout) {
            if (time(NULL) - u_start > 30) {  // 30 second timeout
                u_timeout = true;
                fprintf(stderr, "Client: Timeout waiting for U plane acknowledgment\n");
            }
        }
        
        if (u_timeout) {
            fprintf(stderr, "Client: Failed to get U plane acknowledgment, skipping frame\n");
            free(image->Y);
            free(image->U);
            free(image->V);
            free(image);
            continue;  // Try next frame
        }
        
        printf("Client: U plane acknowledgment received from server\n");
        local_seg->packet.cmd = CMD_INVALID;  // Reset command
        
        //
        // TRANSFER V PLANE
        //
        printf("Client: Starting V plane transfer process\n");
        printf("Client: V plane size: %zu bytes\n", v_size);
        
        // Copy V data to message buffer
        memcpy((void*)local_seg->message_buffer, image->V, v_size);
        
        // Start DMA transfer to server
        SCIStartDmaTransfer(dma_queue, 
                           local_segment,
                           remote_segment,
                           offsetof(struct client_segment, message_buffer),
                           v_size,
                           offsetof(struct server_segment, message_buffer),
                           NO_CALLBACK,
                           NULL,
                           NO_FLAGS,
                           &error);
        
        if (error != SCI_ERR_OK) {
            fprintf(stderr, "Client: V plane DMA transfer failed - Error code 0x%x\n", error);
            free(image->Y);
            free(image->U);
            free(image->V);
            free(image);
            continue;  // Try next frame
        }
        
        // Wait for V plane transfer to complete
        SCIWaitForDMAQueue(dma_queue, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
        printf("Client: V plane DMA transfer completed\n");
        
        // Signal server that V plane is ready
        SCIFlush(NULL, NO_FLAGS);
        remote_seg->packet.cmd = CMD_YUV_DATA;
        remote_seg->packet.data_size = v_size;
        SCIFlush(NULL, NO_FLAGS);
        printf("Client: V plane ready signal sent\n");
        
        // Wait for V plane acknowledgment
        printf("Client: Waiting for V plane acknowledgment\n");
        time_t v_start = time(NULL);
        bool v_timeout = false;
        
        while (local_seg->packet.cmd != CMD_YUV_DATA_ACK && !v_timeout) {
            if (time(NULL) - v_start > 30) {  // 30 second timeout
                v_timeout = true;
                fprintf(stderr, "Client: Timeout waiting for V plane acknowledgment\n");
            }
        }
        
        if (v_timeout) {
            fprintf(stderr, "Client: Failed to get V plane acknowledgment, skipping frame\n");
            free(image->Y);
            free(image->U);
            free(image->V);
            free(image);
            continue;  // Try next frame
        }
        
        printf("Client: V plane acknowledgment received from server\n");
        local_seg->packet.cmd = CMD_INVALID;  // Reset command
        
        // Free original image buffers as they're no longer needed
        printf("Client: All YUV planes transferred successfully, freeing original image buffers\n");
        free(image->Y);
        free(image->U);
        free(image->V);
        free(image);
        
        //
        // WAIT FOR ENCODED DATA FROM SERVER
        //
        printf("Client: Waiting for server to process and return encoded data\n");
        time_t encode_start = time(NULL);
        bool encode_timeout = false;
        
        while (local_seg->packet.cmd != CMD_ENCODED_DATA && !encode_timeout) {
            if (time(NULL) - encode_start > 120) {  // 2 minute timeout - encoding can take time
                encode_timeout = true;
                fprintf(stderr, "Client: Timeout waiting for encoded data\n");
            }
            
            // Print status every 10 seconds
            if (!encode_timeout && (time(NULL) - encode_start) % 10 == 0 && time(NULL) > encode_start) {
                printf("Client: Still waiting for encoded data... (%ld seconds)\n", 
                       time(NULL) - encode_start);
            }
            
        }
        
        if (encode_timeout) {
            fprintf(stderr, "Client: Failed to receive encoded data, skipping frame\n");
            continue;  // Try next frame
        }
        
        //
        // PROCESS ENCODED DATA
        //
        printf("Client: Received encoded data from server\n");
        
        // Get total data size
        size_t data_size = local_seg->packet.data_size;
        printf("Client: Encoded data size: %zu bytes\n", data_size);
        
        // Get keyframe flag
        int keyframe = *((int*)local_seg->message_buffer);
        cm->curframe->keyframe = keyframe;
        printf("Client: Frame is %s\n", keyframe ? "a keyframe" : "not a keyframe");
        
        // Get a pointer to the encoded data (after the keyframe flag)
        char* encoded_data = (char*)local_seg->message_buffer + sizeof(int);
        
        // Copy the encoded data to our curframe structure
        // Ydct
        size_t ydct_size = cm->ypw * cm->yph * sizeof(int16_t);
        memcpy(cm->curframe->residuals->Ydct, encoded_data, ydct_size);
        encoded_data += ydct_size;
        
        // Udct
        size_t udct_size = cm->upw * cm->uph * sizeof(int16_t);
        memcpy(cm->curframe->residuals->Udct, encoded_data, udct_size);
        encoded_data += udct_size;
        
        // Vdct
        size_t vdct_size = cm->vpw * cm->vph * sizeof(int16_t);
        memcpy(cm->curframe->residuals->Vdct, encoded_data, vdct_size);
        encoded_data += vdct_size;
        
        // Macroblocks - Y component
        size_t mby_size = cm->mb_rows * cm->mb_cols * sizeof(struct macroblock);
        memcpy(cm->curframe->mbs[Y_COMPONENT], encoded_data, mby_size);
        encoded_data += mby_size;
        
        // Macroblocks - U component
        size_t mbu_size = (cm->mb_rows/2) * (cm->mb_cols/2) * sizeof(struct macroblock);
        memcpy(cm->curframe->mbs[U_COMPONENT], encoded_data, mbu_size);
        encoded_data += mbu_size;
        
        // Macroblocks - V component
        size_t mbv_size = (cm->mb_rows/2) * (cm->mb_cols/2) * sizeof(struct macroblock);
        memcpy(cm->curframe->mbs[V_COMPONENT], encoded_data, mbv_size);
        
        // Acknowledge receipt of encoded data
        printf("Client: Sending acknowledgment of encoded data receipt\n");
        local_seg->packet.cmd = CMD_INVALID;  // Reset command first
        SCIFlush(NULL, NO_FLAGS);
        remote_seg->packet.cmd = CMD_ENCODED_DATA_ACK;
        SCIFlush(NULL, NO_FLAGS);
        
        // Write the encoded frame to disk
        printf("Client: Writing encoded frame to output file\n");
        write_frame(cm);
        
        printf("Done!\n");
        cm->framenum++;
        cm->frames_since_keyframe++;
        if (cm->curframe->keyframe) {
            cm->frames_since_keyframe = 0;
        }
        
        ++numframes;
        
        if (limit_numframes && numframes >= limit_numframes) {
            printf("Client: Reached frame limit (%d frames), stopping\n", limit_numframes);
            break;
        }
    }
    
    // Signal server to quit
    printf("Client: Sending quit command to server\n");
    SCIFlush(NULL, NO_FLAGS);
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
    cm->curframe = create_frame(cm, NULL);  // Create with NULL for a placeholder
    cm->refframe = create_frame(cm, NULL); 
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
    destroy_frame(cm->refframe);
    destroy_frame(cm->curframe);
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