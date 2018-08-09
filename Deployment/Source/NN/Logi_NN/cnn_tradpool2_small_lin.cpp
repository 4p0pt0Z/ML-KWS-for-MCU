#include "cnn_tradpool2_small_lin.h"

const q7_t CNN_TRADPOOL2_SMALL_LIN::conv1_wt[CONV1_KX*CONV1_KY*CONV1_OUT_CH]=CNN_TRADPOOL2_SMALL_LIN_CONV1_WEIGHT;
const q7_t CNN_TRADPOOL2_SMALL_LIN::conv1_bias[CONV1_OUT_CH]=CNN_TRADPOOL2_SMALL_LIN_CONV1_BIAS;
const q7_t CNN_TRADPOOL2_SMALL_LIN::conv2_wt[CONV2_IN_CH*CONV2_KX*CONV2_KY*CONV2_OUT_CH]=CNN_TRADPOOL2_SMALL_LIN_CONV2_WEIGHT;
const q7_t CNN_TRADPOOL2_SMALL_LIN::conv2_bias[CONV2_OUT_CH]=CNN_TRADPOOL2_SMALL_LIN_CONV2_BIAS;
const q7_t CNN_TRADPOOL2_SMALL_LIN::final_fc_wt[FINAL_FC_IN_DIM*OUT_DIM]=CNN_TRADPOOL2_SMALL_LIN_OUTPUT_WEIGHT;
const q7_t CNN_TRADPOOL2_SMALL_LIN::final_fc_bias[OUT_DIM]=CNN_TRADPOOL2_SMALL_LIN_OUTPUT_BIAS;

CNN_TRADPOOL2_SMALL_LIN::CNN_TRADPOOL2_SMALL_LIN()
{
  scratch_pad = new q7_t[IMCOL_BUFFER_SIZE+2*IM_BUFFER_SIZE];
  buffer1 = scratch_pad;
  buffer2 = buffer1 + IM_BUFFER_SIZE;
  col_buffer = buffer2 + IM_BUFFER_SIZE;
  frame_len = FRAME_LEN;
  frame_shift = FRAME_SHIFT;
  num_mfcc_features = NUM_MFCC_COEFFS;
  num_frames = NUM_FRAMES;
  num_out_classes = OUT_DIM;
  in_dec_bits = MFCC_DEC_BITS;
}

CNN_TRADPOOL2_SMALL_LIN::~CNN_TRADPOOL2_SMALL_LIN()
{
  delete scratch_pad;
}

void CNN_TRADPOOL2_SMALL_LIN::run_nn(q7_t* in_data, q7_t* out_data)
{

    arm_convolve_HWC_q7_basic_nonsquare(in_data,
                                        CONV1_IN_X, CONV1_IN_Y, 1,
                                        conv1_wt,
                                        CONV1_OUT_CH, CONV1_KX, CONV1_KY, CONV1_PX, CONV1_PY, CONV1_SX, CONV1_SY,
                                        conv1_bias,
                                        CONV1_BIAS_LSHIFT, CONV1_OUT_RSHIFT,
                                        buffer1,
                                        CONV1_OUT_X, CONV1_OUT_Y,
                                        (q15_t*)col_buffer, NULL);

    maxpool_q7_HWC_nonsquare (buffer1,
                              MAX_POOL1_IN_X, MAX_POOL1_IN_Y, MAX_POOL1_IN_CH, 
                              MAX_POOL1_KX, MAX_POOL1_KY, 
                              0, 0,
                              MAX_POOL1_KX, MAX_POOL1_KY, 
                              MAX_POOL1_OUT_X, MAX_POOL1_OUT_Y, 
                              NULL, buffer2,
                              MAX_POOL1_OUT_LSHIFT);

    arm_convolve_HWC_q7_fast_nonsquare(buffer2,
                                       CONV2_IN_X, CONV2_IN_Y, CONV2_IN_CH,
                                       conv2_wt,
                                       CONV2_OUT_CH, CONV2_KX, CONV2_KY, CONV2_PX, CONV2_PY, CONV2_SX, CONV2_SY,
                                       conv2_bias,
                                       CONV2_BIAS_LSHIFT, CONV2_OUT_RSHIFT,
                                       buffer1,
                                       CONV2_OUT_X, CONV2_OUT_Y,
                                       (q15_t*)col_buffer, NULL);
    
    maxpool_q7_HWC_nonsquare (buffer1,
                              MAX_POOL2_IN_X, MAX_POOL2_IN_Y, MAX_POOL2_IN_CH, 
                              MAX_POOL2_KX, MAX_POOL2_KY, 
                              0, 0,
                              MAX_POOL2_KX, MAX_POOL2_KY, 
                              MAX_POOL2_OUT_X, MAX_POOL2_OUT_Y, 
                              NULL, buffer2,
                              MAX_POOL2_OUT_LSHIFT);
    
    arm_fully_connected_q7(buffer2,
                           final_fc_wt,
                           FINAL_FC_IN_DIM, OUT_DIM,
                           FINAL_FC_BIAS_LSHIFT, FINAL_FC_OUT_RSHIFT,
                           final_fc_bias,
                           out_data,
                           (q15_t*)col_buffer);
}