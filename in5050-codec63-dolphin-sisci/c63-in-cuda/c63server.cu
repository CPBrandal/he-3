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
#include "quantdct.h"
#include "common.h"
#include "me.h"
#include "tables.h"

/* getopt */
extern int optind;
extern char *optarg;

// Function to process a frame without calling write_frame
static void c63_encode_image_server(struct c63_common *cm, yuv_t *image)
{
    /* Advance to next frame */
    destroy_frame(cm->refframe);
    cm->refframe = cm->curframe;
    cm->curframe = create_frame(cm, image);

    /* Check if keyframe */
    if (cm->framenum == 0 || cm->frames_since_keyframe == cm->keyframe_interval)
    {
        cm->curframe->keyframe = 1;
        cm->frames_since_keyframe = 0;
        printf(" (keyframe) ");
    }
    else
    {
        cm->curframe->keyframe = 0;
    }

    if (!cm->curframe->keyframe)
    {
        /* Motion Estimation */
        c63_motion_estimate(cm);

        /* Motion Compensation */
        c63_motion_compensate(cm);
    }

    /* DCT and Quantization */
    dct_quantize(image->Y, cm->curframe->predicted->Y, cm->padw[Y_COMPONENT],
                cm->padh[Y_COMPONENT], cm->curframe->residuals->Ydct,
                cm->quanttbl[Y_COMPONENT]);

    dct_quantize(image->U, cm->curframe->predicted->U, cm->padw[U_COMPONENT],
                cm->padh[U_COMPONENT], cm->curframe->residuals->Udct,
                cm->quanttbl[U_COMPONENT]);

    dct_quantize(image->V, cm->curframe->predicted->V, cm->padw[V_COMPONENT],
                cm->padh[V_COMPONENT], cm->curframe->residuals->Vdct,
                cm->quanttbl[V_COMPONENT]);

    /* Reconstruct frame for inter-prediction */
    dequantize_idct(cm->curframe->residuals->Ydct, cm->curframe->predicted->Y,
                   cm->ypw, cm->yph, cm->curframe->recons->Y, cm->quanttbl[Y_COMPONENT]);
                   
    dequantize_idct(cm->curframe->residuals->Udct, cm->curframe->predicted->U,
                   cm->upw, cm->uph, cm->curframe->recons->U, cm->quanttbl[U_COMPONENT]);
                   
    dequantize_idct(cm->curframe->residuals->Vdct, cm->curframe->predicted->V,
                   cm->vpw, cm->vph, cm->curframe->recons->V, cm->quanttbl[V_COMPONENT]);

    // Skip write_frame() - client will handle this
    ++cm->framenum;
    ++cm->frames_since_keyframe;
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

    // Set up SISCI resources
    sci_desc_t sd;
    sci_local_segment_t serverSegment;
    sci_local_segment_t serverCtrlSegment;
    sci_remote_segment_t clientSegment;
    sci_remote_segment_t clientResultSegment;
    sci_remote_segment_t clientCtrlSegment;
    sci_map_t serverMap, serverCtrlMap;
    sci_map_t clientMap, clientResultMap, clientCtrlMap;
    sci_dma_queue_t dmaQueue;

    // Open virtual device
    SCIOpen(&sd, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCIOpen failed - Error code 0x%x\n", error);
        SCITerminate();
        exit(EXIT_FAILURE);
    }

    // Create local segments
    SCICreateSegment(sd, &serverSegment, SEGMENT_SERVER, sizeof(struct server_segment),
                    NO_CALLBACK, NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCICreateSegment failed - Error code 0x%x\n", error);
        SCIClose(sd, NO_FLAGS, &error);
        SCITerminate();
        exit(EXIT_FAILURE);
    }
    
    SCICreateSegment(sd, &serverCtrlSegment, SEGMENT_SERVER_CONTROL, sizeof(control_message_t),
                    NO_CALLBACK, NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCICreateSegment (control) failed - Error code 0x%x\n", error);
        SCIRemoveSegment(serverSegment, NO_FLAGS, &error);
        SCIClose(sd, NO_FLAGS, &error);
        SCITerminate();
        exit(EXIT_FAILURE);
    }

    // Prepare segments
    SCIPrepareSegment(serverSegment, localAdapterNo, NO_FLAGS, &error);
    SCIPrepareSegment(serverCtrlSegment, localAdapterNo, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCIPrepareSegment failed - Error code 0x%x\n", error);
        SCIRemoveSegment(serverCtrlSegment, NO_FLAGS, &error);
        SCIRemoveSegment(serverSegment, NO_FLAGS, &error);
        SCIClose(sd, NO_FLAGS, &error);
        SCITerminate();
        exit(EXIT_FAILURE);
    }

    // Create DMA queue
    SCICreateDMAQueue(sd, &dmaQueue, localAdapterNo, 1, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCICreateDMAQueue failed - Error code 0x%x\n", error);
        SCIRemoveSegment(serverCtrlSegment, NO_FLAGS, &error);
        SCIRemoveSegment(serverSegment, NO_FLAGS, &error);
        SCIClose(sd, NO_FLAGS, &error);
        SCITerminate();
        exit(EXIT_FAILURE);
    }

    // Map local segments
    volatile struct server_segment *server_segment = (volatile struct server_segment *)SCIMapLocalSegment(
        serverSegment, &serverMap, 0, sizeof(struct server_segment), NULL, NO_FLAGS, &error);

    control_message_t *server_ctrl = (control_message_t *)SCIMapLocalSegment(
        serverCtrlSegment, &serverCtrlMap, 0, sizeof(control_message_t), NULL, NO_FLAGS, &error);

    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCIMapLocalSegment failed - Error code 0x%x\n", error);
        SCIRemoveDMAQueue(dmaQueue, NO_FLAGS, &error);
        SCIRemoveSegment(serverCtrlSegment, NO_FLAGS, &error);
        SCIRemoveSegment(serverSegment, NO_FLAGS, &error);
        SCIClose(sd, NO_FLAGS, &error);
        SCITerminate();
        exit(EXIT_FAILURE);
    }

    // Initialize control packet
    server_segment->packet.cmd = CMD_INVALID;
    server_ctrl->command = CMD_INVALID;

    // Make segments available
    SCISetSegmentAvailable(serverSegment, localAdapterNo, NO_FLAGS, &error);
    SCISetSegmentAvailable(serverCtrlSegment, localAdapterNo, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCISetSegmentAvailable failed - Error code 0x%x\n", error);
        SCIUnmapSegment(serverCtrlMap, NO_FLAGS, &error);
        SCIUnmapSegment(serverMap, NO_FLAGS, &error);
        SCIRemoveDMAQueue(dmaQueue, NO_FLAGS, &error);
        SCIRemoveSegment(serverCtrlSegment, NO_FLAGS, &error);
        SCIRemoveSegment(serverSegment, NO_FLAGS, &error);
        SCIClose(sd, NO_FLAGS, &error);
        SCITerminate();
        exit(EXIT_FAILURE);
    }

    printf("Server: Waiting for client connection...\n");

    // Wait for initial hello message
    while (server_segment->packet.cmd != CMD_HELLO) {
        // Just wait
    }
    
    printf("Server: Connected to client segment\n");
    
    // Connect to client segments
    do {
        SCIConnectSegment(sd,
                          &clientSegment,
                          remoteNodeId,
                          SEGMENT_CLIENT,
                          localAdapterNo,
                          NO_CALLBACK,
                          NULL,
                          SCI_INFINITE_TIMEOUT,
                          NO_FLAGS,
                          &error);
    } while (error != SCI_ERR_OK);
    
    do {
        SCIConnectSegment(sd,
                          &clientResultSegment,
                          remoteNodeId,
                          SEGMENT_CLIENT_RESULT,
                          localAdapterNo,
                          NO_CALLBACK,
                          NULL,
                          SCI_INFINITE_TIMEOUT,
                          NO_FLAGS,
                          &error);
    } while (error != SCI_ERR_OK);
    
    do {
        SCIConnectSegment(sd,
                          &clientCtrlSegment,
                          remoteNodeId,
                          SEGMENT_CLIENT_CONTROL,
                          localAdapterNo,
                          NO_CALLBACK,
                          NULL,
                          SCI_INFINITE_TIMEOUT,
                          NO_FLAGS,
                          &error);
    } while (error != SCI_ERR_OK);

    // Map client segments
    client_to_server_t *client_msg = (client_to_server_t*)SCIMapRemoteSegment(
        clientSegment, &clientMap, 0, sizeof(client_to_server_t) + MAX_Y_SIZE + 2*MAX_UV_SIZE,
        NULL, NO_FLAGS, &error);
    
    processed_frame_t *result_msg = (processed_frame_t*)SCIMapRemoteSegment(
        clientResultSegment, &clientResultMap, 0, sizeof(processed_frame_t) + MAX_Y_SIZE*4,
        NULL, NO_FLAGS, &error);
    
    control_message_t *client_ctrl = (control_message_t*)SCIMapRemoteSegment(
        clientCtrlSegment, &clientCtrlMap, 0, sizeof(control_message_t),
        NULL, NO_FLAGS, &error);
    
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCIMapRemoteSegment failed - Error code 0x%x\n", error);
        SCIDisconnectSegment(clientCtrlSegment, NO_FLAGS, &error);
        SCIDisconnectSegment(clientResultSegment, NO_FLAGS, &error);
        SCIDisconnectSegment(clientSegment, NO_FLAGS, &error);
        SCISetSegmentUnavailable(serverCtrlSegment, localAdapterNo, NO_FLAGS, &error);
        SCISetSegmentUnavailable(serverSegment, localAdapterNo, NO_FLAGS, &error);
        SCIUnmapSegment(serverCtrlMap, NO_FLAGS, &error);
        SCIUnmapSegment(serverMap, NO_FLAGS, &error);
        SCIRemoveDMAQueue(dmaQueue, NO_FLAGS, &error);
        SCIRemoveSegment(serverCtrlSegment, NO_FLAGS, &error);
        SCIRemoveSegment(serverSegment, NO_FLAGS, &error);
        SCIClose(sd, NO_FLAGS, &error);
        SCITerminate();
        exit(EXIT_FAILURE);
    }

    // Process hello message
    printf("Server: Received hello message: \"%s\"\n", server_segment->message_buffer);
    
    // Send response
    strcpy(result_msg->header.message_buffer, "HELLO FROM SERVER");
    
    // Signal client
    client_ctrl->command = CMD_HELLO_ACK;
    SCIFlush(NULL, NO_FLAGS);
    
    printf("Server: Sent response message\n");
    printf("Server: Waiting for commands...\n");
    
    // Encoder and frame processing setup
    struct c63_common *cm = NULL;
    yuv_t *server_image = NULL;
    int deviceId;
    cudaGetDevice(&deviceId);
    
    int frame_count = 0;
    bool running = true;
    
    while(running) {
        // Wait for command from client
        while(server_ctrl->command != CMD_PROCESS_FRAME && server_ctrl->command != CMD_QUIT) {
            // Just wait
        }
        
        // Check for quit command
        if (server_ctrl->command == CMD_QUIT) {
            printf("Server: Received quit command\n");
            running = false;
            break;
        }
        
        // Get frame data from client
        uint32_t frame_number = server_ctrl->frame_number;
        uint32_t width = client_msg->header.width;
        uint32_t height = client_msg->header.height;
        uint32_t y_size = client_msg->header.y_size;
        uint32_t uv_size = client_msg->header.u_size;
        
        // Initialize encoder if this is the first frame
        if (!cm) {
            cm = init_c63_enc(width, height);
            
            // Allocate reusable image buffer
            server_image = allocate_yuv_buffer(cm);
        }
        
        // Copy frame data to server buffers
        uint8_t *data_ptr = (uint8_t*)(client_msg + 1);
        
        memcpy(server_image->Y, data_ptr, y_size);
        memcpy(server_image->U, data_ptr + y_size, uv_size);
        memcpy(server_image->V, data_ptr + y_size + uv_size, uv_size);
        
        // Prefetch data to GPU
        cudaMemPrefetchAsync(server_image->Y, y_size, deviceId);
        cudaMemPrefetchAsync(server_image->U, uv_size, deviceId);
        cudaMemPrefetchAsync(server_image->V, uv_size, deviceId);
        
        // Update frame sequence information from client
        cm->framenum = frame_number;
        cm->frames_since_keyframe = client_msg->header.frames_since_keyframe;
        
        // Process the frame
        printf("Server: Processing frame %d\n", frame_number);
        c63_encode_image_server(cm, server_image);
        
        // Prepare processed data to send back to client
        result_msg->header.command = CMD_PROCESSED_FRAME;
        result_msg->header.frame_number = frame_number;
        result_msg->header.frames_since_keyframe = cm->frames_since_keyframe;
        result_msg->header.is_keyframe = cm->curframe->keyframe;
        
        // Calculate data sizes
        size_t ydct_size = cm->padw[Y_COMPONENT] * cm->padh[Y_COMPONENT] * sizeof(int16_t);
        size_t udct_size = cm->padw[U_COMPONENT] * cm->padh[U_COMPONENT] * sizeof(int16_t);
        size_t vdct_size = cm->padw[V_COMPONENT] * cm->padh[V_COMPONENT] * sizeof(int16_t);
        
        size_t mb_y_count = cm->mb_rows * cm->mb_cols;
        size_t mb_u_count = (cm->mb_rows/2) * (cm->mb_cols/2);
        size_t mb_v_count = mb_u_count;
        
        result_msg->header.ydct_size = ydct_size;
        result_msg->header.udct_size = udct_size;
        result_msg->header.vdct_size = vdct_size;
        result_msg->header.mb_y_count = mb_y_count;
        result_msg->header.mb_u_count = mb_u_count;
        result_msg->header.mb_v_count = mb_v_count;
        
        // Copy DCT coefficients and macroblock data
        uint8_t *result_data = (uint8_t*)(result_msg + 1);
        
        // Copy DCT coefficients
        memcpy(result_data, cm->curframe->residuals->Ydct, ydct_size);
        memcpy(result_data + ydct_size, cm->curframe->residuals->Udct, udct_size);
        memcpy(result_data + ydct_size + udct_size, cm->curframe->residuals->Vdct, vdct_size);
        
        // Copy macroblock data
        uint8_t *mb_data = result_data + ydct_size + udct_size + vdct_size;
        
        memcpy(mb_data, cm->curframe->mbs[Y_COMPONENT], mb_y_count * sizeof(struct macroblock));
        memcpy(mb_data + mb_y_count * sizeof(struct macroblock),
               cm->curframe->mbs[U_COMPONENT], mb_u_count * sizeof(struct macroblock));
        memcpy(mb_data + (mb_y_count + mb_u_count) * sizeof(struct macroblock),
               cm->curframe->mbs[V_COMPONENT], mb_v_count * sizeof(struct macroblock));
        
        // Signal that processing is complete
        SCIFlush(NULL, NO_FLAGS);
        client_ctrl->command = CMD_PROCESSED_FRAME;
        SCIFlush(NULL, NO_FLAGS);
        
        // Reset command for next frame
        server_ctrl->command = CMD_INVALID;
        
        frame_count++;
    }
    
    printf("Server: Processed %d frames\n", frame_count);
    printf("Server: Exiting\n");
    
    // Clean up resources
    if (server_image) {
        free_yuv_buffer(server_image);
    }
    
    if (cm) {
        destroy_frame(cm->refframe);
        destroy_frame(cm->curframe);
        free(cm);
    }
    
    // Clean up SISCI resources
    SCIDisconnectSegment(clientCtrlSegment, NO_FLAGS, &error);
    SCIDisconnectSegment(clientResultSegment, NO_FLAGS, &error);
    SCIDisconnectSegment(clientSegment, NO_FLAGS, &error);
    SCIUnmapSegment(clientCtrlMap, NO_FLAGS, &error);
    SCIUnmapSegment(clientResultMap, NO_FLAGS, &error);
    SCIUnmapSegment(clientMap, NO_FLAGS, &error);
    SCISetSegmentUnavailable(serverCtrlSegment, localAdapterNo, NO_FLAGS, &error);
    SCISetSegmentUnavailable(serverSegment, localAdapterNo, NO_FLAGS, &error);
    SCIUnmapSegment(serverCtrlMap, NO_FLAGS, &error);
    SCIUnmapSegment(serverMap, NO_FLAGS, &error);
    SCIRemoveDMAQueue(dmaQueue, NO_FLAGS, &error);
    SCIRemoveSegment(serverCtrlSegment, NO_FLAGS, &error);
    SCIRemoveSegment(serverSegment, NO_FLAGS, &error);
    SCIClose(sd, NO_FLAGS, &error);
    SCITerminate();
    
    return EXIT_SUCCESS;
}