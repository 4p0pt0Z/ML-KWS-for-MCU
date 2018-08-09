#ifndef __NAIVE_GRU__
#define __NAIVE_GRU__

#include "nn.h"
#include "naive_gru_embedded_features_h_256_q7_t.h"
#include "local_NN.h"
#include "arm_math.h"

#define SAMP_FREQ 16000
#define SAMP_LENGTH 1.0

// Number of decimal bits for output of MFCC (8 bits quantization)
#define MFCC_DEC_BITS 0 // TODO

// Model features
#define FRAME_LEN_MS 10 
#define FRAME_LEN ((int16_t)(SAMP_FREQ * 0.001 * FRAME_LEN_MS))
#define FRAME_SHIFT_MS 5 
#define FRAME_SHIFT ((int16_t)(SAMP_FREQ * 0.001 * FRAME_SHIFT_MS))
#define NUM_FRAMES ((int16_t)(SAMP_LENGTH * SAMP_FREQ / FRAME_SHIFT) + 1)
#define NUM_MEL_FILTERS 40 
#define MFCC_BUFFER_SIZE (NUM_FRAMES*NUM_MEL_FILTERS)

#define INPUT_DIM NUM_MEL_FILTERS
#define HIDDEN_DIM 256 // TODO
#define SEQ_LENGTH NUM_FRAMES

#define OUT_DIM 12

// Reset gate: input matrix
#define FC_IR_IN_DIM INPUT_DIM
#define FC_IR_OUT_DIM HIDDEN_DIM
#define FC_IR_BIAS_LSHIFT 0 // TODO
#define FC_IR_OUT_RSHIFT 0 // TODO
// Reset gate: hidden state matrix
#define FC_HR_IN_DIM HIDDEN_DIM
#define FC_HR_OUT_DIM HIDDEN_DIM
#define FC_HR_BIAS_LSHIFT 0 // TODO
#define FC_HR_OUT_RSHIFT 0 // TODO

#define RESET_INT_BIT_WIDTH 0 // TODO

// Update gate: input matrix
#define FC_IZ_IN_DIM INPUT_DIM
#define FC_IZ_OUT_DIM HIDDEN_DIM
#define FC_IZ_BIAS_LSHIFT 0 // TODO
#define FC_IZ_OUT_RSHIFT 0 // TODO
// Update gate: hidden state matrix
#define FC_HZ_IN_DIM HIDDEN_DIM
#define FC_HZ_OUT_DIM HIDDEN_DIM
#define FC_HZ_BIAS_LSHIFT 0 // TODO
#define FC_HZ_OUT_RSHIFT 0 // TODO

#define UPDATE_INT_BIT_WIDTH 0 // TODO

// New gate: input matrix
#define FC_IN_IN_DIM INPUT_DIM
#define FC_IN_OUT_DIM HIDDEN_DIM
#define FC_IN_BIAS_LSHIFT 0 // TODO
#define FC_IN_OUT_RSHIFT 0 // TODO
// New gate: hidden state matrix
#define FC_HN_IN_DIM HIDDEN_DIM
#define FC_HN_OUT_DIM HIDDEN_DIM
#define FC_HN_BIAS_LSHIFT 0 // TODO
#define FC_HN_OUT_RSHIFT 0 // TODO

#define NEW_INT_BIT_WIDTH 0 // TODO

#define FC_OUTPUT_IN_DIM HIDDEN_DIM
#define FC_OUTPUT_OUT_DIM OUT_DIM
#define FC_OUTPUT_BIAS_LSHIFT 0 // TODO
#define FC_OUTPUT_OUT_RSHIFT 0 // TODO

#define max(a,b) ({ __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); _a > _b ? _a : _b; })
#define SCRATCH_BUFFER_SIZE 3 * HIDDEN_DIM + 2 * max(INPUT_DIM, HIDDEN_DIM) + 2 * HIDDEN_DIM

class NAIVE_GRU : public NN {

    public:
        NAIVE_GRU();
        ~NAIVE_GRU();
        void run_nn(q7_t * in_data, q7_t * output);

    private:
        q7_t * scratch_buffer; // Buffer for GRU function computations

        static q7_t const FC_IR_wt[FC_IR_IN_DIM*FC_IR_OUT_DIM];
        static q7_t const FC_IR_bias[FC_IR_OUT_DIM];
        static q7_t const FC_HR_wt[FC_HR_IN_DIM*FC_HR_OUT_DIM];
        static q7_t const FC_HR_bias[FC_HR_OUT_DIM];
        static q7_t const FC_IZ_wt[FC_IZ_IN_DIM*FC_IZ_OUT_DIM];
        static q7_t const FC_IZ_bias[FC_IZ_OUT_DIM];
        static q7_t const FC_HZ_wt[FC_HZ_IN_DIM*FC_HZ_OUT_DIM];
        static q7_t const FC_HZ_bias[FC_HZ_OUT_DIM];
        static q7_t const FC_IN_wt[FC_IN_IN_DIM*FC_IN_OUT_DIM];
        static q7_t const FC_IN_bias[FC_IN_OUT_DIM];
        static q7_t const FC_HN_wt[FC_HN_IN_DIM*FC_HN_OUT_DIM];
        static q7_t const FC_HN_bias[FC_HN_OUT_DIM];
        static q7_t const FC_OUTPUT_wt[FC_OUTPUT_IN_DIM*FC_OUTPUT_OUT_DIM];
        static q7_t const FC_OUTPUT_bias[FC_OUTPUT_OUT_DIM];
};


#endif