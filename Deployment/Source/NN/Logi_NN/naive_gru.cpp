#include "naive_gru.h"

const q7_t NAIVE_GRU::FC_IR_wt[FC_IR_IN_DIM*FC_IR_OUT_DIM]=NAIVE_GRU_EMBEDDED_FEATURES_H_256_FC_IR_WEIGHT; // TODO !!
const q7_t NAIVE_GRU::FC_IR_bias[FC_IR_OUT_DIM]=NAIVE_GRU_EMBEDDED_FEATURES_H_256_FC_IR_BIAS;
const q7_t NAIVE_GRU::FC_HR_wt[FC_HR_IN_DIM*FC_HR_OUT_DIM]=NAIVE_GRU_EMBEDDED_FEATURES_H_256_FC_HR_WEIGHT;
const q7_t NAIVE_GRU::FC_HR_bias[FC_HR_OUT_DIM]=NAIVE_GRU_EMBEDDED_FEATURES_H_256_FC_HR_BIAS;
const q7_t NAIVE_GRU::FC_IZ_wt[FC_IZ_IN_DIM*FC_IZ_OUT_DIM]=NAIVE_GRU_EMBEDDED_FEATURES_H_256_FC_IZ_WEIGHT;
const q7_t NAIVE_GRU::FC_IZ_bias[FC_IZ_OUT_DIM]=NAIVE_GRU_EMBEDDED_FEATURES_H_256_FC_IZ_BIAS;
const q7_t NAIVE_GRU::FC_HZ_wt[FC_HZ_IN_DIM*FC_HZ_OUT_DIM]=NAIVE_GRU_EMBEDDED_FEATURES_H_256_FC_HZ_WEIGHT;
const q7_t NAIVE_GRU::FC_HZ_bias[FC_HZ_OUT_DIM]=NAIVE_GRU_EMBEDDED_FEATURES_H_256_FC_HZ_BIAS;
const q7_t NAIVE_GRU::FC_IN_wt[FC_IN_IN_DIM*FC_IN_OUT_DIM]=NAIVE_GRU_EMBEDDED_FEATURES_H_256_FC_IN_WEIGHT;
const q7_t NAIVE_GRU::FC_IN_bias[FC_IN_OUT_DIM]=NAIVE_GRU_EMBEDDED_FEATURES_H_256_FC_IN_BIAS;
const q7_t NAIVE_GRU::FC_HN_wt[FC_HN_IN_DIM*FC_HN_OUT_DIM]=NAIVE_GRU_EMBEDDED_FEATURES_H_256_FC_HN_WEIGHT;
const q7_t NAIVE_GRU::FC_HN_bias[FC_HN_OUT_DIM]=NAIVE_GRU_EMBEDDED_FEATURES_H_256_FC_HN_BIAS;
const q7_t NAIVE_GRU::FC_OUTPUT_wt[FC_OUTPUT_IN_DIM*FC_OUTPUT_OUT_DIM] = NAIVE_GRU_EMBEDDED_FEATURES_H_256_FC_FINAL_WEIGHT;
const q7_t NAIVE_GRU::FC_OUTPUT_bias[FC_OUTPUT_OUT_DIM] = NAIVE_GRU_EMBEDDED_FEATURES_H_256_FC_FINAL_BIAS;

NAIVE_GRU::NAIVE_GRU()
{
  scratch_buffer = new q7_t[SCRATCH_BUFFER_SIZE];
  frame_len = FRAME_LEN;
  frame_shift = FRAME_SHIFT;
  num_mfcc_features = NUM_MEL_FILTERS;
  num_frames = NUM_FRAMES;
  num_out_classes = OUT_DIM;
  in_dec_bits = MFCC_DEC_BITS;
}

NAIVE_GRU::~NAIVE_GRU()
{
  delete scratch_buffer;
}

void NAIVE_GRU::run_nn(q7_t *in_data, q7_t *out_data)
{
    /*
        in_data stores the features for each frames, one frame after another, eg:
        FRAME[0]MFCC[0], FRAME[0]MFCC[1], .... , FRAME[1]MFCC[0] ...
    */

    // Initialize hidden state to 0
    q7_t h[HIDDEN_DIM] = { 0 };

    // Loop over the time frames:
    int32_t in_data_head = 0; // Indicate the start of the frame in in_data
    for(int16_t i = 0; i < SEQ_LENGTH; ++i)
    {
        GRU_layer_q7_q7(&in_data[in_data_head], INPUT_DIM, HIDDEN_DIM,
                        FC_IR_wt, FC_IR_bias, FC_HR_wt, FC_HR_bias,
                        FC_IR_BIAS_LSHIFT,  FC_IR_OUT_RSHIFT,
                        FC_HR_BIAS_LSHIFT, FC_HR_OUT_RSHIFT,
                        RESET_INT_BIT_WIDTH,
                        FC_IZ_wt,  FC_IZ_bias,  FC_HZ_wt, FC_HZ_bias,
                        FC_IZ_BIAS_LSHIFT, FC_IZ_OUT_RSHIFT,
                        FC_HZ_BIAS_LSHIFT, FC_HZ_OUT_RSHIFT,
                        UPDATE_INT_BIT_WIDTH,
                        FC_IN_wt, FC_IN_bias, FC_HN_wt, FC_HN_bias,
                        FC_IN_BIAS_LSHIFT, FC_IN_OUT_RSHIFT,
                        FC_HN_BIAS_LSHIFT, FC_HN_OUT_RSHIFT,
                        NEW_INT_BIT_WIDTH,
                        h,
                        scratch_buffer);
        in_data_head += NUM_MEL_FILTERS;
    }
    
    arm_fully_connected_q7(h,
                           FC_OUTPUT_wt,
                           FC_OUTPUT_IN_DIM, FC_OUTPUT_OUT_DIM,
                           FC_OUTPUT_BIAS_LSHIFT, FC_OUTPUT_OUT_RSHIFT,
                           FC_OUTPUT_bias,
                           out_data,
                           (q15_t*)scratch_buffer);
}