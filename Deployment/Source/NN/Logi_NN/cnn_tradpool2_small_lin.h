/* 
Implementation for ARM Cortex M7 of the keyword spotting neural 
network "cnn-tradpool2" as defined in the Honk project
https://github.com/castorini/honk
*/



#ifndef __CNN_TRADPOOL2_SMALL_LIN_H__
#define __CNN_TRADPOOL2_SMALL_LIN_H__

#include "nn.h"
#include "cnn-tradpool2_small_lin_weights.h"
#include "local_NN.h"
#include "arm_math.h"


#define SAMP_FREQ 16000
#define SAMP_LENGTH 1.0

// Number of decimal bits for output of MFCC (8 bits quantization)
#define MFCC_DEC_BITS 9999 // TODO

// Honk cnn-tradpool2 features
#define FRAME_LEN_MS 30
#define FRAME_LEN ((int16_t)(SAMP_FREQ * 0.001 * FRAME_LEN_MS))
#define FRAME_SHIFT_MS 10
#define FRAME_SHIFT ((int16_t)(SAMP_FREQ * 0.001 * FRAME_SHIFT_MS))
#define NUM_FRAMES ((int16_t)(SAMP_LENGTH * SAMP_FREQ / FRAME_SHIFT) + 1)
#define NUM_MFCC_COEFFS 40
#define MFCC_BUFFER_SIZE (NUM_FRAMES*NUM_MFCC_COEFFS)

#define IN_DIM (NUM_FRAMES*NUM_MFCC_COEFFS)
#define OUT_DIM 12

#define CONV1_OUT_CH 64
#define CONV1_IN_X NUM_MFCC_COEFFS
#define CONV1_IN_Y NUM_FRAMES
#define CONV1_KX 8
#define CONV1_KY 20
#define CONV1_SX 1
#define CONV1_SY 1
#define CONV1_PX 0
#define CONV1_PY 0
#define CONV1_OUT_X ((int16_t)((CONV1_IN_X + 2*CONV1_PX - (CONV1_KX - 1) -1)/CONV1_SX +1))
#define CONV1_OUT_Y ((int16_t)((CONV1_IN_Y + 2*CONV1_PY - (CONV1_KY - 1) -1)/CONV1_SY +1))
#define CONV1_BIAS_LSHIFT 9999 // TODO
#define CONV1_OUT_RSHIFT 9999 // TODO


#define MAX_POOL1_IN_X CONV1_OUT_X
#define MAX_POOL1_IN_Y CONV1_OUT_Y
#define MAX_POOL1_IN_CH CONV1_OUT_CH
#define MAX_POOL1_KX 2
#define MAX_POOL1_KY 2
#define MAX_POOL1_PX 0
#define MAX_POOL1_PY 0
#define MAX_POOL1_SX 2  // stride: in pytorch, set to size of kernel in this dim unless specified otherwise
#define MAX_POOL1_SY 2
#define MAX_POOL1_OUT_X ((int16_t)((MAX_POOL1_IN_X + 2*MAX_POOL1_PX - (MAX_POOL1_KX - 1) -1)/MAX_POOL1_SX +1))
#define MAX_POOL1_OUT_Y ((int16_t)((MAX_POOL1_IN_Y + 2*MAX_POOL1_PY - (MAX_POOL1_KY - 1) -1)/MAX_POOL1_SY +1))
#define MAX_POOL1_OUT_LSHIFT 9999 // TODO


#define CONV2_IN_X MAX_POOL1_OUT_X
#define CONV2_IN_Y MAX_POOL1_OUT_Y
#define CONV2_IN_CH MAX_POOL1_IN_CH
#define CONV2_KX 4
#define CONV2_KY 10
#define CONV2_SX 1
#define CONV2_SY 1
#define CONV2_PX 0
#define CONV2_PY 0
#define CONV2_OUT_X ((int16_t)((CONV2_IN_X + 2*CONV2_PX - (CONV2_KX - 1) -1)/CONV2_SX +1))
#define CONV2_OUT_Y ((int16_t)((CONV2_IN_Y + 2*CONV2_PY - (CONV2_KY - 1) -1)/CONV2_SY +1))
#define CONV2_OUT_CH 64
#define CONV2_BIAS_LSHIFT 9999 // TODO
#define CONV2_OUT_RSHIFT 9999 // TODO

#define MAX_POOL2_IN_X CONV2_OUT_X
#define MAX_POOL2_IN_Y CONV2_OUT_Y
#define MAX_POOL2_IN_CH CONV2_OUT_CH
#define MAX_POOL2_KX MAX_POOL2_IN_X
#define MAX_POOL2_KY MAX_POOL2_IN_Y
#define MAX_POOL2_PX 0
#define MAX_POOL2_PY 0
#define MAX_POOL2_SX 1
#define MAX_POOL2_SY 1
#define MAX_POOL2_OUT_X ((int16_t)((MAX_POOL2_IN_X + 2*MAX_POOL2_PX - (MAX_POOL2_KX - 1) -1)/MAX_POOL2_SX +1)) // 1
#define MAX_POOL2_OUT_Y ((int16_t)((MAX_POOL2_IN_Y + 2*MAX_POOL2_PY - (MAX_POOL2_KY - 1) -1)/MAX_POOL2_SY +1)) // 1
#define MAX_POOL2_OUT_LSHIFT 9999 // TODO


#define FINAL_FC_IN_DIM (MAX_POOL2_IN_CH * MAX_POOL2_OUT_X * MAX_POOL2_OUT_Y)
#define FINAL_FC_OUT_DIM OUT_DIM
#define FINAL_FC_BIAS_LSHIFT 9999 // TODO
#define FINAL_FC_OUT_RSHIFT 9999 // TODO


#define IMCOL_BUFFER_SIZE (2*CONV2_IN_CH*CONV1_KX*CONV1_KY)
#define IM_BUFFER_SIZE (CONV1_OUT_CH*CONV1_OUT_X*CONV1_OUT_Y) // size after conv1, maximal size layer activation for the rest of the net


class CNN_TRADPOOL2_SMALL_LIN : public NN {

  public:
    CNN_TRADPOOL2_SMALL_LIN();
    ~CNN_TRADPOOL2_SMALL_LIN();
    void run_nn(q7_t* in_data, q7_t* out_data);

  private:
    q7_t* scratch_pad;
    q7_t* col_buffer; // Buffer used for convolution computation, for imcol.
    q7_t* buffer1;    // Buffer used to store input or output of convolution layer
    q7_t* buffer2;    // Buffer used to store input or output of convolution layer
    static q7_t const conv1_wt[CONV1_KX*CONV1_KY*CONV1_OUT_CH];
    static q7_t const conv1_bias[CONV1_OUT_CH];
    static q7_t const conv2_wt[CONV2_IN_CH*CONV2_KX*CONV2_KY*CONV2_OUT_CH];
    static q7_t const conv2_bias[CONV2_OUT_CH];
    static q7_t const final_fc_wt[FINAL_FC_IN_DIM*OUT_DIM];
    static q7_t const final_fc_bias[OUT_DIM];

};


#endif