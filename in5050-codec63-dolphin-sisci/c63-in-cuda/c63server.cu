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

#include <sisci_error.h>
#include <sisci_api.h>

/* getopt */
extern int optind;
extern char *optarg;

// Modified main loop to echo frame numbers
// Modified main loop to echo frame numbers
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
    int frame_number;  // Moved declaration outside of the switch

    printf("Server: Waiting for frames...\n");

    while(running)
    {
        // Wait for command from client
        while(local_seg->packet.cmd == CMD_INVALID) {
            // Just wait
        }

        // Process command
        cmd = local_seg->packet.cmd;
        
        // Reset command field
        local_seg->packet.cmd = CMD_INVALID;

        switch(cmd) {
            case CMD_DATA_READY:
                // Get frame number from client - REMOVED "int" DECLARATION HERE
                frame_number = *((int*)local_seg->message_buffer);
                printf("Server: Received frame number %d\n", frame_number);
                frame_count++;
                
                // Echo the frame number back to client 
                *((int*)local_seg->message_buffer) = frame_number;
                local_seg->packet.data_size = sizeof(int);
                
                // Use DMA to transfer the response 
                SCIStartDmaTransfer(dma_queue, 
                                   local_segment,  // Source segment
                                   remote_segment, // Destination segment
                                   offsetof(struct server_segment, message_buffer),  // Source offset
                                   local_seg->packet.data_size, // Size to transfer
                                   offsetof(struct client_segment, message_buffer),  // Destination offset
                                   NO_CALLBACK, 
                                   NULL, 
                                   NO_FLAGS, 
                                   &error);
                if (error != SCI_ERR_OK) {
                    fprintf(stderr, "Server: SCIStartDmaTransfer failed - Error code 0x%x\n", error);
                    running = 0;
                    break;
                }
                
                // Wait for DMA transfer to complete
                SCIWaitForDMAQueue(dma_queue, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
                if (error != SCI_ERR_OK) {
                    fprintf(stderr, "Server: SCIWaitForDMAQueue failed - Error code 0x%x\n", error);
                    running = 0;
                    break;
                }
                
                // Signal that data is ready
                SCIFlush(NULL, NO_FLAGS);
                remote_seg->packet.cmd = CMD_DATA_READY;
                SCIFlush(NULL, NO_FLAGS);
                
                printf("Server: Echoed frame number %d\n", frame_number);
                break;
                
            case CMD_QUIT:
                printf("Server: Received quit command after processing %d frames\n", frame_count);
                running = 0;
                break;
                
            default:
                printf("Server: Unknown command: %d\n", cmd);
                break;
        }
    }

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