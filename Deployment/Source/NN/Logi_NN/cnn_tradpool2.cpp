#include "cnn_tradpool2.h"

const q7_t CNN_TRADPOOL2::conv1_wt[CONV1_KX*CONV1_KY*CONV1_OUT_CH]=CONV1_WEIGHT;
const q7_t CNN_TRADPOOL2::conv1_bias[CONV1_OUT_CH]=CONV1_BIAS;
const q7_t CNN_TRADPOOL2::conv2_wt[CONV2_IN_CH*CONV2_KX*CONV2_KY*CONV2_OUT_CH]=CONV2_WEIGHT;
const q7_t CNN_TRADPOOL2::conv2_bias[CONV2_OUT_CH]=CONV2_BIAS;
const q7_t CNN_TRADPOOL2::final_fc_wt[FINAL_FC_IN_DIM*OUT_DIM]=OUTPUT_WEIGHT;
const q7_t CNN_TRADPOOL2::final_fc_bias[OUT_DIM]=OUTPUT_BIAS;

CNN_TRADPOOL2::CNN_TRADPOOL2()
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

CNN_TRADPOOL2::~CNN_TRADPOOL2()
{
  delete scratch_pad;
}

void CNN_TRADPOOL2::run_nn(q7_t* in_data, q7_t* out_data)
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
                              MAX_POOL1_PX, MAX_POOL1_PY,
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
                              MAX_POOL2_PX, MAX_POOL2_PY,
                              MAX_POOL2_KX, MAX_POOL2_KY, 
                              MAX_POOL2_OUT_X, MAX_POOL2_OUT_Y, 
                              NULL, buffer2,
                              MAX_POOL2_OUT_LSHIFT);
    
    basic_fully_connected_q7(buffer2,
                           final_fc_wt,
                           FINAL_FC_IN_DIM, OUT_DIM,
                           FINAL_FC_BIAS_LSHIFT, FINAL_FC_OUT_RSHIFT,
                           final_fc_bias,
                           out_data);
}


arm_status
basic_fully_connected_q7(const q7_t * pV,
                         const q7_t * pM,
                         const uint16_t dim_vec,
                         const uint16_t num_of_rows,
                         const uint16_t bias_shift,
                         const uint16_t out_shift, const q7_t * bias, q7_t * pOut)
{
  int       i, j;

  /* Run the following code as reference implementation for Cortex-M0 and Cortex-M3 */
  for (i = 0; i < num_of_rows; i++)
  {
    int       ip_out = ((q31_t)(bias[i]) << bias_shift) + NN_ROUND(out_shift);
    for (j = 0; j < dim_vec; j++)
    {
      ip_out += pV[j] * pM[i * dim_vec + j];
    }
    pOut[i] = (q7_t) __SSAT((ip_out >> out_shift), 8);
  }
  return (ARM_MATH_SUCCESS);
}