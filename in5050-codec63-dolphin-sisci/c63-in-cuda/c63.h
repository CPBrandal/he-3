#ifndef C63_C63_H_
#define C63_C63_H_

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#define GROUP 9

#ifndef GROUP
#error Fill in group number
#endif

#define NO_FLAGS 0
#define NO_CALLBACK NULL
#define NO_ARG NULL

#define MAX_WIDTH 1280
#define MAX_HEIGHT 720
#define MAX_Y_SIZE ((((MAX_WIDTH+15)/16)*16) * (((MAX_HEIGHT+15)/16)*16))
#define MAX_U_SIZE (MAX_Y_SIZE/4)
#define MAX_V_SIZE (MAX_Y_SIZE/4)

/* GET_SEGMENTID(2) gives you segmentid 2 at your groups offset */
#define GET_SEGMENTID(id) ( GROUP << 16 | id )
#define SEGMENT_CLIENT GET_SEGMENTID(1)
#define SEGMENT_SERVER GET_SEGMENTID(2)

// Message sizes
#define MESSAGE_SIZE 256   // Size for hello message

enum cmd {
    CMD_INVALID = 0,
    CMD_HELLO,
    CMD_HELLO_ACK,
    CMD_DIMENSIONS,
    CMD_DIMENSIONS_ACK,
    CMD_FRAME_HEADER,     // New command for frame header
    CMD_FRAME_HEADER_ACK, // Server acknowledges header
    CMD_Y_DATA_READY,     // Y data is ready
    CMD_Y_DATA_ACK,       // Y data acknowledged
    CMD_U_DATA_READY,     // U data is ready
    CMD_U_DATA_ACK,       // U data acknowledged
    CMD_V_DATA_READY,     // V data is ready
    CMD_V_DATA_ACK,       // V data acknowledged
    CMD_FRAME_ENCODED,    // Server has encoded the frame
    CMD_ENCODED_DATA_HEADER,     // Metadata about the encoded frame
    CMD_ENCODED_DATA_HEADER_ACK, // Client acknowledges header
    CMD_RESIDUALS_Y_READY,       // Y residuals data is ready
    CMD_RESIDUALS_Y_ACK,         // Y residuals acknowledged
    CMD_RESIDUALS_U_READY,       // U residuals data is ready
    CMD_RESIDUALS_U_ACK,         // U residuals acknowledged
    CMD_RESIDUALS_V_READY,       // V residuals data is ready
    CMD_RESIDUALS_V_ACK,         // V residuals acknowledged
    CMD_MOTION_VECTORS_READY,    // Motion vectors data is ready
    CMD_MOTION_VECTORS_ACK,      // Motion vectors acknowledged
    CMD_QUIT     
};

#define MAX_FILELENGTH 200
#define DEFAULT_OUTPUT_FILE "a.mjpg"

#define PI 3.14159265358979
#define ILOG2 1.442695040888963 // 1/log(2);

#define COLOR_COMPONENTS 3

#define Y_COMPONENT 0
#define U_COMPONENT 1
#define V_COMPONENT 2

#define YX 2
#define YY 2
#define UX 1
#define UY 1
#define VX 1
#define VY 1

/* The JPEG file format defines several parts and each part is defined by a
 marker. A file always starts with 0xFF and is then followed by a magic number,
 e.g., like 0xD8 in the SOI marker below. Some markers have a payload, and if
 so, the size of the payload is written before the payload itself. */

#define JPEG_DEF_MARKER 0xFF
#define JPEG_SOI_MARKER 0xD8
#define JPEG_DQT_MARKER 0xDB
#define JPEG_SOF_MARKER 0xC0
#define JPEG_DHT_MARKER 0xC4
#define JPEG_SOS_MARKER 0xDA
#define JPEG_EOI_MARKER 0xD9

#define HUFF_AC_ZERO 16
#define HUFF_AC_SIZE 11

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))

struct yuv
{
    uint8_t *Y;
    uint8_t *U;
    uint8_t *V;
};

struct dct
{
    int16_t *Ydct;
    int16_t *Udct;
    int16_t *Vdct;
};

typedef struct yuv yuv_t;
typedef struct dct dct_t;

struct entropy_ctx
{
    FILE *fp;
    unsigned int bit_buffer;
    unsigned int bit_buffer_width;
};

struct macroblock
{
    int use_mv;
    int8_t mv_x, mv_y;
};

struct frame
{
    yuv_t *orig;        // Original input image
    yuv_t *recons;      // Reconstructed image
    yuv_t *predicted;   // Predicted frame from intra-prediction

    dct_t *residuals;   // Difference between original image and predicted frame

    struct macroblock *mbs[COLOR_COMPONENTS];
    int keyframe;
};

struct c63_common
{
    int width, height;
    int ypw, yph, upw, uph, vpw, vph;

    int padw[COLOR_COMPONENTS], padh[COLOR_COMPONENTS];

    int mb_cols, mb_rows;

    uint8_t qp;         // Quality parameter

    int me_search_range;

    uint8_t quanttbl[COLOR_COMPONENTS][64];

    struct frame *refframe;
    struct frame *curframe;

    int framenum;

    int keyframe_interval;
    int frames_since_keyframe;

    struct entropy_ctx e_ctx;
};

struct dimensions_data {
    uint32_t width;
    uint32_t height;
};

struct packet
{
    union {
        struct {
            uint32_t cmd;          // Command type
            uint32_t data_size;    // Size of data in buffer
        };
        uint8_t padding[64];       // Align to cache line
    } __attribute__((aligned(64)));
};

struct frame_header {
    uint32_t frame_number;       // Current frame number
    uint8_t is_last_frame;       // Flag to indicate if this is the last frame
};

struct server_segment {
    struct packet packet __attribute__((aligned(64)));
    char message_buffer[MESSAGE_SIZE] __attribute__((aligned(64))); 
    uint8_t y_buffer[MAX_Y_SIZE] __attribute__((aligned(64)));
    uint8_t u_buffer[MAX_U_SIZE] __attribute__((aligned(64)));
    uint8_t v_buffer[MAX_V_SIZE] __attribute__((aligned(64)));
    // Add new buffers for motion vectors
    uint8_t mv_y_buffer[MAX_Y_SIZE/64] __attribute__((aligned(64))); // Motion vectors are much smaller
    uint8_t mv_u_buffer[MAX_U_SIZE/64] __attribute__((aligned(64)));
    uint8_t mv_v_buffer[MAX_V_SIZE/64] __attribute__((aligned(64)));
};

struct client_segment {
    struct packet packet __attribute__((aligned(64)));
    char message_buffer[MESSAGE_SIZE] __attribute__((aligned(64)));
    uint8_t y_buffer[MAX_Y_SIZE] __attribute__((aligned(64)));
    uint8_t u_buffer[MAX_U_SIZE] __attribute__((aligned(64)));
    uint8_t v_buffer[MAX_V_SIZE] __attribute__((aligned(64)));
    // Add new buffers for motion vectors
    uint8_t mv_y_buffer[MAX_Y_SIZE/64] __attribute__((aligned(64)));
    uint8_t mv_u_buffer[MAX_U_SIZE/64] __attribute__((aligned(64)));
    uint8_t mv_v_buffer[MAX_V_SIZE/64] __attribute__((aligned(64)));
};

struct encoded_frame_header {
    int keyframe;            // Is this a keyframe?
    uint32_t y_size;         // Size of Y residuals data
    uint32_t u_size;         // Size of U residuals data
    uint32_t v_size;         // Size of V residuals data
    uint32_t mv_y_size;      // Size of Y motion vectors
    uint32_t mv_u_size;      // Size of U motion vectors
    uint32_t mv_v_size;      // Size of V motion vectors
};

#endif  /* C63_C63_H_ */