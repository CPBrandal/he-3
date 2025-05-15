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

#define MAX_WIDTH 1920
#define MAX_HEIGHT 1080
#define MAX_Y_SIZE (MAX_WIDTH * MAX_HEIGHT)
#define MAX_UV_SIZE (MAX_Y_SIZE / 4)

/* GET_SEGMENTID(2) gives you segmentid 2 at your groups offset */
#define GET_SEGMENTID(id) ( GROUP << 16 | id )
#define SEGMENT_CLIENT_RESULT GET_SEGMENTID(3)
#define SEGMENT_CLIENT_CONTROL GET_SEGMENTID(4)
#define SEGMENT_SERVER_CONTROL GET_SEGMENTID(5)

// Message sizes
#define MESSAGE_SIZE 256   // Size for hello message

// Command definitions for signaling
enum cmd
{
    CMD_INVALID = 0,   // No command/initial state
    CMD_HELLO,         // Client sending hello
    CMD_HELLO_ACK,     // Server acknowledging hello
    CMD_QUIT,          // Signal to terminate
    CMD_DATA_READY     // Signal that data is ready to be read
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
    yuv_t *orig;                // Original input image
    yuv_t *recons;              // Reconstructed image
    yuv_t *predicted;           // Predicted frame from intra-prediction

    dct_t *residuals;           // Difference between original image and predicted frame

    struct macroblock *mbs[COLOR_COMPONENTS];
    int keyframe;
};

struct c63_common
{
    int width, height;
    int ypw, yph, upw, uph, vpw, vph;

    int padw[COLOR_COMPONENTS], padh[COLOR_COMPONENTS];

    int mb_cols, mb_rows;

    uint8_t qp;                 // Quality parameter

    int me_search_range;

    uint8_t quanttbl[COLOR_COMPONENTS][64];

    struct frame *refframe;
    struct frame *curframe;

    int framenum;

    int keyframe_interval;
    int frames_since_keyframe;

    struct entropy_ctx e_ctx;
};

enum frame_cmd
{
    CMD_FRAME_DATA = 10,      // Client sending frame data
    CMD_PROCESS_FRAME = 11,   // Request to process frame
    CMD_PROCESSED_FRAME = 12, // Server sending processed frame data
    CMD_FRAME_DONE = 13,      // Frame processing complete
    CMD_PROCESSING = 14,      // Currently processing frame
    CMD_WAITING = 15          // Waiting for processing
};

// Frame data message from client to server
typedef struct {
    struct {
        uint32_t command;           // Command type (CMD_FRAME_DATA)
        uint32_t frame_number;      // Current frame number
        uint32_t frames_since_keyframe;
        uint32_t width;
        uint32_t height;
        uint32_t y_size;
        uint32_t u_size;
        uint32_t v_size;
        uint8_t padding[36];        // Pad to 64 bytes
    } __attribute__((packed, aligned(64))) header;
    
    // YUV data follows in the segment (after header)
} client_to_server_t;

// Processed frame data from server to client
typedef struct {
    struct {
        uint32_t command;           // Command type (CMD_PROCESSED_FRAME)
        uint32_t frame_number;      // Frame that was processed
        uint32_t frames_since_keyframe;  // Updated count
        uint32_t is_keyframe;       // Was this processed as a keyframe?
        uint32_t ydct_size;         // Size of Y DCT coefficients
        uint32_t udct_size;         // Size of U DCT coefficients
        uint32_t vdct_size;         // Size of V DCT coefficients
        uint32_t mb_y_count;        // Number of Y macroblocks
        uint32_t mb_u_count;        // Number of U macroblocks
        uint32_t mb_v_count;        // Number of V macroblocks
        uint8_t padding[24];        // Pad to 64 bytes
    } __attribute__((packed, aligned(64))) header;
    
    // After the header follows:
    // 1. Ydct coefficients
    // 2. Udct coefficients
    // 3. Vdct coefficients
    // 4. Y macroblocks
    // 5. U macroblocks
    // 6. V macroblocks
} processed_frame_t;

// Control message structure
typedef struct {
    uint32_t command;        // Command type
    uint32_t frame_number;   // Current frame
    uint32_t status;         // Status code
    uint8_t padding[52];     // Pad to 64 bytes
} __attribute__((aligned(64))) control_message_t;

#endif /* C63_C63_H_ */
