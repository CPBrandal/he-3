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
#include "quantdct.h"
#include "common.h"
#include "me.h"
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
static yuv_t *read_yuv_frame(FILE *file, yuv_t *image, uint32_t width, uint32_t height)
{
    size_t len = 0;
    size_t y_size = width * height;
    size_t uv_size = y_size / 4;

    // Read Y
    len += fread(image->Y, 1, y_size, file);
    
    // Read U
    len += fread(image->U, 1, uv_size, file);
    
    // Read V
    len += fread(image->V, 1, uv_size, file);

    if (ferror(file))
    {
        perror("ferror");
        return NULL;
    }

    if (feof(file) || len != y_size + 2 * uv_size)
    {
        if (feof(file) && len == 0) {
            // Clean EOF
            return NULL;
        }
        
        fprintf(stderr, "Reached end of file or incorrect bytes read: %zu\n", len);
        fprintf(stderr, "Expected: %zu\n", y_size + 2 * uv_size);
        return NULL;
    }

    return image;
}

struct c63_common *init_c63_enc(int width, int height)
{
    int i;

    /* calloc() sets allocated memory to zero */
    struct c63_common *cm = (struct c63_common *)calloc(1, sizeof(struct c63_common));

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

    /* Quality parameters */
    cm->qp = 25;                // Constant quantization factor. Range: [1..50]
    cm->me_search_range = 16;   // Pixels in every direction
    cm->keyframe_interval = 100; // Distance between keyframes

    /* Initialize quantization tables */
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
    destroy_frame(cm->refframe);
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

// Function to allocate YUV structure with CUDA managed memory
static yuv_t* allocate_yuv_buffer(struct c63_common *cm)
{
    yuv_t *image = (yuv_t *)malloc(sizeof(yuv_t));
    
    cudaMallocManaged((void**)&image->Y, cm->padw[Y_COMPONENT] * cm->padh[Y_COMPONENT] * sizeof(uint8_t));
    cudaMallocManaged((void**)&image->U, cm->padw[U_COMPONENT] * cm->padh[U_COMPONENT] * sizeof(uint8_t));
    cudaMallocManaged((void**)&image->V, cm->padw[V_COMPONENT] * cm->padh[V_COMPONENT] * sizeof(uint8_t));
    
    return image;
}

// Function to free YUV buffer
static void free_yuv_buffer(yuv_t *image)
{
    if (image) {
        cudaFree(image->Y);
        cudaFree(image->U);
        cudaFree(image->V);
        free(image);
    }
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

    // Open output file
    outfile = fopen(output_file, "wb");
    if (outfile == NULL)
    {
        perror("fopen");
        exit(EXIT_FAILURE);
    }

    // Open input file
    FILE *infile = fopen(input_file, "rb");
    if (infile == NULL)
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

    // Initialize SISCI
    SCIInitialize(NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCIInitialize failed: %s\n", SCIGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Set up SISCI resources
    sci_desc_t sd;
    sci_local_segment_t clientSegment;    // For client -> server data
    sci_local_segment_t resultSegment;    // For server -> client results
    sci_local_segment_t clientCtrlSegment; // For client control
    sci_remote_segment_t serverSegment;
    sci_remote_segment_t serverResultSegment;
    sci_remote_segment_t serverCtrlSegment;
    sci_map_t clientMap, resultMap, clientCtrlMap;
    sci_map_t serverMap, serverResultMap, serverCtrlMap;
    sci_dma_queue_t dmaQueue;

    // Open virtual device
    SCIOpen(&sd, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCIOpen failed - Error code 0x%x\n", error);
        SCITerminate();
        exit(EXIT_FAILURE);
    }
    
    // Calculate required segment sizes
    size_t y_size = width * height;
    size_t uv_size = y_size / 4;
    size_t client_segment_size = sizeof(client_to_server_t) + y_size + 2 * uv_size;
    
    // Calculate result segment size
    size_t ydct_size = cm->padw[Y_COMPONENT] * cm->padh[Y_COMPONENT] * sizeof(int16_t);
    size_t udct_size = cm->padw[U_COMPONENT] * cm->padh[U_COMPONENT] * sizeof(int16_t);
    size_t vdct_size = cm->padw[V_COMPONENT] * cm->padh[V_COMPONENT] * sizeof(int16_t);
    size_t mb_y_size = cm->mb_rows * cm->mb_cols * sizeof(struct macroblock);
    size_t mb_u_size = (cm->mb_rows/2) * (cm->mb_cols/2) * sizeof(struct macroblock);
    size_t mb_v_size = mb_u_size;
    size_t result_segment_size = sizeof(processed_frame_t) + ydct_size + udct_size + vdct_size + mb_y_size + mb_u_size + mb_v_size;
    
    // Create data segment
    SCICreateSegment(sd, &clientSegment, SEGMENT_CLIENT, client_segment_size, 
                    NO_CALLBACK, NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCICreateSegment failed - Error code 0x%x\n", error);
        SCIClose(sd, NO_FLAGS, &error);
        SCITerminate();
        exit(EXIT_FAILURE);
    }
    
    // Create result segment
    SCICreateSegment(sd, &resultSegment, SEGMENT_CLIENT_RESULT, result_segment_size, 
                    NO_CALLBACK, NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCICreateSegment (result) failed - Error code 0x%x\n", error);
        SCIRemoveSegment(clientSegment, NO_FLAGS, &error);
        SCIClose(sd, NO_FLAGS, &error);
        SCITerminate();
        exit(EXIT_FAILURE);
    }
    
    // Create control segment
    SCICreateSegment(sd, &clientCtrlSegment, SEGMENT_CLIENT_CONTROL, sizeof(control_message_t), 
                    NO_CALLBACK, NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCICreateSegment (control) failed - Error code 0x%x\n", error);
        SCIRemoveSegment(resultSegment, NO_FLAGS, &error);
        SCIRemoveSegment(clientSegment, NO_FLAGS, &error);
        SCIClose(sd, NO_FLAGS, &error);
        SCITerminate();
        exit(EXIT_FAILURE);
    }
    
    // Prepare segments
    SCIPrepareSegment(clientSegment, localAdapterNo, NO_FLAGS, &error);
    SCIPrepareSegment(resultSegment, localAdapterNo, NO_FLAGS, &error);
    SCIPrepareSegment(clientCtrlSegment, localAdapterNo, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCIPrepareSegment failed - Error code 0x%x\n", error);
        SCIRemoveSegment(clientCtrlSegment, NO_FLAGS, &error);
        SCIRemoveSegment(resultSegment, NO_FLAGS, &error);
        SCIRemoveSegment(clientSegment, NO_FLAGS, &error);
        SCIClose(sd, NO_FLAGS, &error);
        SCITerminate();
        exit(EXIT_FAILURE);
    }
    
    // Create DMA queue
    SCICreateDMAQueue(sd, &dmaQueue, localAdapterNo, 1, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCICreateDMAQueue failed - Error code 0x%x\n", error);
        SCIRemoveSegment(clientCtrlSegment, NO_FLAGS, &error);
        SCIRemoveSegment(resultSegment, NO_FLAGS, &error);
        SCIRemoveSegment(clientSegment, NO_FLAGS, &error);
        SCIClose(sd, NO_FLAGS, &error);
        SCITerminate();
        exit(EXIT_FAILURE);
    }
    
    // Map segments
    client_to_server_t *client_msg = (client_to_server_t*)SCIMapLocalSegment(
        clientSegment, &clientMap, 0, client_segment_size, NULL, NO_FLAGS, &error);
    
    processed_frame_t *result_msg = (processed_frame_t*)SCIMapLocalSegment(
        resultSegment, &resultMap, 0, result_segment_size, NULL, NO_FLAGS, &error);
    
    control_message_t *client_ctrl = (control_message_t*)SCIMapLocalSegment(
        clientCtrlSegment, &clientCtrlMap, 0, sizeof(control_message_t), NULL, NO_FLAGS, &error);
    
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCIMapLocalSegment failed - Error code 0x%x\n", error);
        SCIRemoveDMAQueue(dmaQueue, NO_FLAGS, &error);
        SCIRemoveSegment(clientCtrlSegment, NO_FLAGS, &error);
        SCIRemoveSegment(resultSegment, NO_FLAGS, &error);
        SCIRemoveSegment(clientSegment, NO_FLAGS, &error);
        SCIClose(sd, NO_FLAGS, &error);
        SCITerminate();
        exit(EXIT_FAILURE);
    }
    
    // Initialize control message
    client_ctrl->command = CMD_INVALID;
    
    // Make segments available
    SCISetSegmentAvailable(clientSegment, localAdapterNo, NO_FLAGS, &error);
    SCISetSegmentAvailable(resultSegment, localAdapterNo, NO_FLAGS, &error);
    SCISetSegmentAvailable(clientCtrlSegment, localAdapterNo, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCISetSegmentAvailable failed - Error code 0x%x\n", error);
        SCIUnmapSegment(clientCtrlMap, NO_FLAGS, &error);
        SCIUnmapSegment(resultMap, NO_FLAGS, &error);
        SCIUnmapSegment(clientMap, NO_FLAGS, &error);
        SCIRemoveDMAQueue(dmaQueue, NO_FLAGS, &error);
        SCIRemoveSegment(clientCtrlSegment, NO_FLAGS, &error);
        SCIRemoveSegment(resultSegment, NO_FLAGS, &error);
        SCIRemoveSegment(clientSegment, NO_FLAGS, &error);
        SCIClose(sd, NO_FLAGS, &error);
        SCITerminate();
        exit(EXIT_FAILURE);
    }
    
    printf("Client: Connecting to server segment...\n");
    
    // Connect to server segments
    do {
        SCIConnectSegment(sd, &serverSegment, remote_node, SEGMENT_SERVER,
                        localAdapterNo, NO_CALLBACK, NULL, SCI_INFINITE_TIMEOUT,
                        NO_FLAGS, &error);
    } while (error != SCI_ERR_OK);
    
    do {
        SCIConnectSegment(sd, &serverResultSegment, remote_node, SEGMENT_SERVER_CONTROL,
                        localAdapterNo, NO_CALLBACK, NULL, SCI_INFINITE_TIMEOUT,
                        NO_FLAGS, &error);
    } while (error != SCI_ERR_OK);
    
    printf("Client: Connected to server segment\n");
    
    // Map remote segments
    volatile struct server_segment *server_seg = (volatile struct server_segment *)SCIMapRemoteSegment(
        serverSegment, &serverMap, 0, sizeof(struct server_segment), NULL, NO_FLAGS, &error);
    
    control_message_t *server_ctrl = (control_message_t *)SCIMapRemoteSegment(
        serverResultSegment, &serverCtrlMap, 0, sizeof(control_message_t), NULL, NO_FLAGS, &error);
    
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCIMapRemoteSegment failed - Error code 0x%x\n", error);
        SCIDisconnectSegment(serverResultSegment, NO_FLAGS, &error);
        SCIDisconnectSegment(serverSegment, NO_FLAGS, &error);
        SCISetSegmentUnavailable(clientCtrlSegment, localAdapterNo, NO_FLAGS, &error);
        SCISetSegmentUnavailable(resultSegment, localAdapterNo, NO_FLAGS, &error);
        SCISetSegmentUnavailable(clientSegment, localAdapterNo, NO_FLAGS, &error);
        SCIUnmapSegment(clientCtrlMap, NO_FLAGS, &error);
        SCIUnmapSegment(resultMap, NO_FLAGS, &error);
        SCIUnmapSegment(clientMap, NO_FLAGS, &error);
        SCIRemoveDMAQueue(dmaQueue, NO_FLAGS, &error);
        SCIRemoveSegment(clientCtrlSegment, NO_FLAGS, &error);
        SCIRemoveSegment(resultSegment, NO_FLAGS, &error);
        SCIRemoveSegment(clientSegment, NO_FLAGS, &error);
        SCIClose(sd, NO_FLAGS, &error);
        SCITerminate();
        exit(EXIT_FAILURE);
    }
    
    // Establish communication with server
    printf("Client: Starting communication\n");
    strcpy(client_msg->header.message_buffer, "hello from client");
    client_msg->header.command = CMD_HELLO;
    client_msg->header.data_size = strlen("hello from client") + 1;
    
    // DMA transfer the hello message
    SCIStartDmaTransfer(dmaQueue, clientSegment, serverSegment, 
                       offsetof(client_to_server_t, header.message_buffer),
                       client_msg->header.data_size,
                       offsetof(struct server_segment, message_buffer),
                       NO_CALLBACK, NULL, NO_FLAGS, &error);
    
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCIStartDmaTransfer failed - Error code 0x%x\n", error);
        exit(EXIT_FAILURE);
    }
    
    // Wait for transfer to complete
    SCIWaitForDMAQueue(dmaQueue, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
    
    // Signal that message is ready
    SCIFlush(NULL, NO_FLAGS);
    server_seg->packet.cmd = CMD_HELLO;
    SCIFlush(NULL, NO_FLAGS);
    
    printf("Client: Sent hello message\n");
    
    // Wait for server response
    client_ctrl->command = CMD_WAITING;
    
    while (client_ctrl->command != CMD_HELLO_ACK) {
        // Just wait
    }
    
    printf("Client: Received response: \"%s\"\n", result_msg->header.message_buffer);
    
    // Allocate reusable YUV buffer for frame data
    yuv_t *reusable_image = allocate_yuv_buffer(cm);
    
    // Main encoding loop
    printf("Client: Starting video encoding\n");
    int numframes = 0;
    
    // Set initial frame state
    cm->framenum = 0;
    cm->frames_since_keyframe = 0;
    
    while (1) {
        // Read next frame into reusable buffer
        if (!read_yuv_frame(infile, reusable_image, width, height)) {
            break;  // End of file or error
        }
        
        printf("Processing frame %d, ", numframes);
        
        // Prepare data for server
        client_msg->header.command = CMD_FRAME_DATA;
        client_msg->header.frame_number = numframes;
        client_msg->header.frames_since_keyframe = cm->frames_since_keyframe;
        client_msg->header.width = width;
        client_msg->header.height = height;
        client_msg->header.y_size = y_size;
        client_msg->header.u_size = uv_size;
        client_msg->header.v_size = uv_size;
        
        // Copy YUV data
        uint8_t *data_ptr = (uint8_t*)(client_msg + 1);
        memcpy(data_ptr, reusable_image->Y, y_size);
        memcpy(data_ptr + y_size, reusable_image->U, uv_size);
        memcpy(data_ptr + y_size + uv_size, reusable_image->V, uv_size);
        
        // Transfer frame data to server
        SCIStartDmaTransfer(dmaQueue, clientSegment, serverSegment, 
                           0, sizeof(client_to_server_t) + y_size + 2*uv_size,
                           0, NO_CALLBACK, NULL, NO_FLAGS, &error);
        
        if (error != SCI_ERR_OK) {
            fprintf(stderr, "Frame data transfer failed - Error code 0x%x\n", error);
            break;
        }
        
        // Wait for DMA to complete
        SCIWaitForDMAQueue(dmaQueue, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
        
        // Signal server to process frame
        SCIFlush(NULL, NO_FLAGS);
        server_ctrl->command = CMD_PROCESS_FRAME;
        server_ctrl->frame_number = numframes;
        SCIFlush(NULL, NO_FLAGS);
        
        // Wait for processing completion
        client_ctrl->command = CMD_WAITING;
        
        while (client_ctrl->command != CMD_PROCESSED_FRAME) {
            // Just wait
        }
        
        // Update frame information from result
        cm->frames_since_keyframe = result_msg->header.frames_since_keyframe;
        
        // Create frame structures for write_frame
        if (cm->refframe) {
            destroy_frame(cm->refframe);
        }
        cm->refframe = cm->curframe;
        cm->curframe = create_frame(cm, reusable_image);
        
        // Set keyframe flag
        cm->curframe->keyframe = result_msg->header.is_keyframe;
        
        // Copy DCT coefficients and macroblock data from result
        uint8_t *result_data = (uint8_t*)(result_msg + 1);
        
        // Copy DCT coefficients
        memcpy(cm->curframe->residuals->Ydct, result_data, ydct_size);
        memcpy(cm->curframe->residuals->Udct, result_data + ydct_size, udct_size);
        memcpy(cm->curframe->residuals->Vdct, result_data + ydct_size + udct_size, vdct_size);
        
        // Copy macroblock data
        uint8_t *mb_data = result_data + ydct_size + udct_size + vdct_size;
        
        memcpy(cm->curframe->mbs[Y_COMPONENT], mb_data, 
               result_msg->header.mb_y_count * sizeof(struct macroblock));
        
        memcpy(cm->curframe->mbs[U_COMPONENT], 
               mb_data + result_msg->header.mb_y_count * sizeof(struct macroblock),
               result_msg->header.mb_u_count * sizeof(struct macroblock));
        
        memcpy(cm->curframe->mbs[V_COMPONENT], 
               mb_data + (result_msg->header.mb_y_count + result_msg->header.mb_u_count) * sizeof(struct macroblock),
               result_msg->header.mb_v_count * sizeof(struct macroblock));
        
        // Write frame to output file
        write_frame(cm);
        
        printf("Done!\n");
        
        // Increment frame counters
        ++numframes;
        ++cm->framenum;
        
        if (limit_numframes && numframes >= limit_numframes) {
            break;
        }
    }
    
    printf("Client: Finished processing %d frames\n", numframes);
    
    // Signal server to quit
    SCIFlush(NULL, NO_FLAGS);
    server_ctrl->command = CMD_QUIT;
    SCIFlush(NULL, NO_FLAGS);
    
    printf("Client: Exiting\n");
    
    // Clean up resources
    free_yuv_buffer(reusable_image);
    free_c63_enc(cm);
    fclose(outfile);
    fclose(infile);
    
    // Clean up SISCI resources
    SCIDisconnectSegment(serverResultSegment, NO_FLAGS, &error);
    SCIDisconnectSegment(serverSegment, NO_FLAGS, &error);
    SCIUnmapSegment(serverResultMap, NO_FLAGS, &error);
    SCIUnmapSegment(serverMap, NO_FLAGS, &error);
    SCISetSegmentUnavailable(clientCtrlSegment, localAdapterNo, NO_FLAGS, &error);
    SCISetSegmentUnavailable(resultSegment, localAdapterNo, NO_FLAGS, &error);
    SCISetSegmentUnavailable(clientSegment, localAdapterNo, NO_FLAGS, &error);
    SCIUnmapSegment(clientCtrlMap, NO_FLAGS, &error);
    SCIUnmapSegment(resultMap, NO_FLAGS, &error);
    SCIUnmapSegment(clientMap, NO_FLAGS, &error);
    SCIRemoveDMAQueue(dmaQueue, NO_FLAGS, &error);
    SCIRemoveSegment(clientCtrlSegment, NO_FLAGS, &error);
    SCIRemoveSegment(resultSegment, NO_FLAGS, &error);
    SCIRemoveSegment(clientSegment, NO_FLAGS, &error);
    SCIClose(sd, NO_FLAGS, &error);
        SCITerminate();

    return EXIT_SUCCESS;
}