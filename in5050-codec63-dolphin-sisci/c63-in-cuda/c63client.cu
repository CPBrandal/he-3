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

#define SERVER_SEG_ID 1
#define SEGMENT_SIZE 1024
#define ADAPTER_NO 0
#define NO_FLAGS 0

/* Read planar YUV frames with 4:2:0 chroma sub-sampling */
static yuv_t *
read_yuv( FILE *file, struct c63_common *cm )
{
    size_t len = 0;
    yuv_t *image = ( yuv_t * ) malloc( sizeof( *image ) );

    /* Read Y. The size of Y is the same as the size of the image. The indices
       represents the color component (0 is Y, 1 is U, and 2 is V) */
    image->Y =
        ( uint8_t * ) calloc( 1,
                              cm->padw[Y_COMPONENT] * cm->padh[Y_COMPONENT] );
    len += fread( image->Y, 1, width * height, file );

    /* Read U. Given 4:2:0 chroma sub-sampling, the size is 1/4 of Y
       because (height/2)*(width/2) = (height*width)/4. */
    image->U =
        ( uint8_t * ) calloc( 1,
                              cm->padw[U_COMPONENT] * cm->padh[U_COMPONENT] );
    len += fread( image->U, 1, ( width * height ) / 4, file );

    /* Read V. Given 4:2:0 chroma sub-sampling, the size is 1/4 of Y. */
    image->V =
        ( uint8_t * ) calloc( 1,
                              cm->padw[V_COMPONENT] * cm->padh[V_COMPONENT] );
    len += fread( image->V, 1, ( width * height ) / 4, file );

    if ( ferror( file ) )
    {
        perror( "ferror" );
        exit( EXIT_FAILURE );
    }

    if ( feof( file ) )
    {
        free( image->Y );
        free( image->U );
        free( image->V );
        free( image );

        return NULL;
    }
    else if ( len != width * height * 1.5 )
    {
        fprintf( stderr, "Reached end of file, but incorrect bytes read.\n" );
        fprintf( stderr, "Wrong input? (height: %d width: %d)\n", height,
                 width );

        free( image->Y );
        free( image->U );
        free( image->V );
        free( image );

        return NULL;
    }

    return image;
}

static void
c63_encode_image( struct c63_common *cm, yuv_t *image )
{
    /* Advance to next frame */
    destroy_frame( cm->refframe );
    cm->refframe = cm->curframe;
    cm->curframe = create_frame( cm, image );

    /* Check if keyframe */
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

    if ( !cm->curframe->keyframe )
    {
        /* Motion Estimation */
        c63_motion_estimate( cm );

        /* Motion Compensation */
        c63_motion_compensate( cm );
    }

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

    write_frame( cm );

    ++cm->framenum;
    ++cm->frames_since_keyframe;
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

static void
print_help(  )
{
    printf( "Usage: ./c63client -r nodeid [options] input_file\n" );
    printf( "Commandline options:\n" );
  printf("  -r                             Node id of client\n");
    printf
        ( "  -h                             Height of images to compress\n" );
    printf
        ( "  -w                             Width of images to compress\n" );
    printf( "  -o                             Output file (.c63)\n" );
    printf
        ( "  [-f]                           Limit number of frames to encode\n" );
    printf( "\n" );

    exit( EXIT_FAILURE );
}
/* int
main( int argc, char **argv )
{
    int c;
    yuv_t *image;
    sci_error_t error;

    if ( argc == 1 )
    {
        print_help(  );
    }

    while ( ( c = getopt( argc, argv, "r:h:w:o:f:i:" ) ) != -1 )
    {
        switch ( c )
        {
            case 'r':
                remote_node = atoi(optarg);
                break;
            case 'h':
                height = atoi( optarg );
                break;
            case 'w':
                width = atoi( optarg );
                break;
            case 'o':
                output_file = optarg;
                break;
            case 'f':
                limit_numframes = atoi( optarg );
                break;
            default:
                print_help(  );
                break;
        }
    }

    if ( optind >= argc )
    {
        fprintf( stderr, "Error getting program options, try --help.\n" );
        exit( EXIT_FAILURE );
    }

    SCIInitialize(0, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr,"SCIInitialize failed: %s\n", SCIGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    outfile = fopen( output_file, "wb" );

    if ( outfile == NULL )
    {
        perror( "fopen" );
        exit( EXIT_FAILURE );
    }

    struct c63_common *cm = init_c63_enc( width, height );
    cm->e_ctx.fp = outfile;

    input_file = argv[optind];

    if ( limit_numframes )
    {
        printf( "Limited to %d frames.\n", limit_numframes );
    }

    FILE *infile = fopen( input_file, "rb" );

    if ( infile == NULL )
    {
        perror( "fopen" );
        exit( EXIT_FAILURE );
    }

    int numframes = 0;

    

    while ( 1 )
    {
        image = read_yuv( infile, cm );

        if ( !image )
        {
            break;
        }

        printf( "Encoding frame %d, ", numframes );
        c63_encode_image( cm, image );

        free( image->Y );
        free( image->U );
        free( image->V );
        free( image );

        printf( "Done!\n" );

        ++numframes;

        if ( limit_numframes && numframes >= limit_numframes )
        {
            break;
        }
    }

    free_c63_enc( cm );
    fclose( outfile );
    fclose( infile );
  
    SCITerminate();

    return EXIT_SUCCESS;
} */

int main_loop(sci_desc_t sd,
    volatile struct client_segment *local_seg,
    volatile struct server_segment *remote_seg,
    sci_dma_queue_t dma_queue,
    sci_local_segment_t local_segment,
    sci_remote_segment_t remote_segment)
{
    sci_error_t error;

    printf("Client: Starting communication\n");

    // Prepare hello message
    strcpy((char*)local_seg->message_buffer, "hello from client");
    local_seg->packet.data_size = strlen("hello from client") + 1;

    // Use DMA to transfer the message
    SCIStartDmaTransfer(dma_queue, 
                local_segment, 
                remote_segment,
                offsetof(struct client_segment, message_buffer), 
                local_seg->packet.data_size,
                offsetof(struct server_segment, message_buffer), 
                NO_CALLBACK, 
                NULL, 
                NO_FLAGS, 
                &error);
    if (error != SCI_ERR_OK) {
    fprintf(stderr, "Client: SCIStartDmaTransfer failed - Error code 0x%x\n", error);
    return -1;
    }

    // Wait for DMA transfer to complete
    SCIWaitForDMAQueue(dma_queue, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
    fprintf(stderr, "Client: SCIWaitForDMAQueue failed - Error code 0x%x\n", error);
    return -1;
    }

    // Signal that data is ready
    SCIFlush(NULL, NO_FLAGS);
    remote_seg->packet.cmd = CMD_HELLO;
    SCIFlush(NULL, NO_FLAGS);

    printf("Client: Sent hello message\n");

    // Wait for server acknowledgment
    while (local_seg->packet.cmd != CMD_HELLO_ACK) {
    // Just wait
    }

    // Print the response
    printf("Client: Received response: \"%s\"\n", local_seg->message_buffer);

    // Reset command
    local_seg->packet.cmd = CMD_INVALID;

    // Tell server we're quitting
    remote_seg->packet.cmd = CMD_QUIT;
    SCIFlush(NULL, NO_FLAGS);

    return 0;
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
    
    volatile struct client_segment *client_segment;
    volatile struct server_segment *server_segment;
    
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
// Map remote segment
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
                          remoteNodeId,
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
    
    // Enter main loop
    main_loop(sd, client_segment, server_segment, dmaQueue, localSegment, remoteSegment);
    
    printf("Client: Exiting\n");
    
    // Clean up resources
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