#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "arm_math.h"
#include "arm_nnfunctions.h"
#include "local_NN.h"

/*
    Implementation of pytorch GRU layer, which is described by the equations:
    r_t = sigmoid [ W_ir @ x_t + b_ir + W_hr @ h_{t-1} + b_hr ]
    z_t = sigmoid [ W_iz @ x_t + b_iz + W_hz @ h_{t-1} + b_hz ]
    n_t = tanh  [W_in @ x_t + b_in + r_t*(W_hn @ h_{t-1} + b_hn)]
    h_t = (1 - z_t)*n_t + z_t*h_{t-1}

    Where * denotes element wise multiplication (gating mechanism) and @ denotes matrix - vector multiplication
    i stands for "input" (input is x_t), and h stands for the hidden state (h_t)
    r_t, z_t, n_t are respectively the reset, update and "new" vectors. (see pytorch doc for GRU)

    scratch buffer size: 3 * hidden_size + max(input_size, hidden_size) x2 + 2 * hidden_size
        We need: 
            - buffer to store reset vector (r), update vector (z), new vector (n)
            - buffer to store results of matrix multiplication (hidden_size x2 for input and hidden)
            - buffer for matrix multiplication, which is size of the input to the layer, x2 for q15_t internal computation

    hidden: [in, out]: user should pass pointer to h_{t-1} (pointer to previous hidden state)
                       this function will replace the values with h_t (new hidden state)
*/
void GRU_layer_q7_q7(q7_t * input, uint16_t input_size, uint16_t hidden_size,
                     const q7_t * weights_input_reset, const q7_t * bias_input_reset, const q7_t * weights_hidden_reset, const q7_t * bias_hidden_reset,
                     const uint16_t bias_lshift_input_reset, const uint16_t out_rshift_input_reset,
                     const uint16_t bias_lshift_hidden_reset, const uint16_t out_rshift_hidden_reset,
                     const uint16_t reset_int_bit_width,
                     const q7_t * weights_input_update, const q7_t * bias_input_update, const q7_t * weights_hidden_update, const q7_t * bias_hidden_update,
                     const uint16_t bias_lshift_input_update, const uint16_t out_rshift_input_update,
                     const uint16_t bias_lshift_hidden_update, const uint16_t out_rshift_hidden_update,
                     const uint16_t update_int_bit_width,
                     const q7_t * weights_input_new, const q7_t * bias_input_new, const q7_t * weights_hidden_new, const q7_t * bias_hidden_new,
                     const uint16_t bias_lshift_input_new, const uint16_t out_rshift_input_new,
                     const uint16_t bias_lshift_hidden_new, const uint16_t out_rshift_hidden_new,
                     const uint16_t new_int_bit_width,
                     q7_t * hidden,
                     q7_t * scratch_buffer)
{
    q7_t * r = scratch_buffer;                                // r_t, size: hidden_size
    q7_t * z = scratch_buffer + hidden_size;                  // z_t, size: hidden_size
    q7_t * n = z + hidden_size;                               // n_t, size: hidden_size
    q7_t * buffer_input = n + hidden_size;                    // size: hidden_size
    q7_t * buffer_hidden = buffer_input + hidden_size;        // size: hidden_size
    q15_t * mat_mult_buffer = (q15_t *) (buffer_hidden + hidden_size);    // size: max(input_size, hidden_size) (in q15_t, so x2 compared to q7_t)

    // ------ Reset computation
    // W_ir @ x_t + b_ir
    arm_fully_connected_q7_opt(input, weights_input_reset,
                               input_size, hidden_size,
                               bias_lshift_input_reset, out_rshift_input_reset,
                               bias_input_reset,
                               buffer_input,
                               mat_mult_buffer);
    // W_hr @ h_{t-1} + b_hr
    arm_fully_connected_q7_opt(hidden, weights_hidden_reset,
                               hidden_size, hidden_size,
                               bias_lshift_hidden_reset, out_rshift_hidden_reset,
                               bias_hidden_reset,
                               buffer_hidden,
                               mat_mult_buffer);
    // (W_ir @ x_t + b_ir) + (W_hr @ h_{t-1} + b_hr)
    arm_add_q7(buffer_input, buffer_hidden, r, hidden_size);
    // r = sigmoid [ (W_ir @ x_t + b_ir) + (W_hr @ h_{t-1} + b_hr) ]
    arm_nn_activations_direct_q7(r, hidden_size, reset_int_bit_width, ARM_SIGMOID);

    // ------ Update computation
    // W_iz @ x_t + b_iz
    arm_fully_connected_q7_opt(input, weights_input_update,
                               input_size, hidden_size,
                               bias_lshift_input_update, out_rshift_input_update,
                               bias_input_update,
                               buffer_input,
                               mat_mult_buffer);
    // W_hz @ h_{t-1} + b_hz
    arm_fully_connected_q7_opt(hidden, weights_hidden_update,
                               hidden_size, hidden_size,
                               bias_lshift_hidden_update, out_rshift_hidden_update,
                               bias_hidden_update,
                               buffer_hidden,
                               mat_mult_buffer);
    // (W_iz @ x_t + b_ir) + (W_hz @ h_{t-1} + b_hz)
    arm_add_q7(buffer_input, buffer_hidden, z, hidden_size);
    // r = sigmoid [ (W_iz @ x_t + b_ir) + (W_hz @ h_{t-1} + b_hz) ]
    arm_nn_activations_direct_q7(z, hidden_size, update_int_bit_width, ARM_SIGMOID);

    // ------ New computation
    // W_in @ x_t + b_in
    arm_fully_connected_q7_opt(input, weights_input_new,
                               input_size, hidden_size,
                               bias_lshift_input_new, out_rshift_input_new,
                               bias_input_new,
                               buffer_input,
                               mat_mult_buffer);
    // W_hn @ h_{t-1} + b_hn
    arm_fully_connected_q7_opt(hidden, weights_hidden_new,
                               hidden_size, hidden_size,
                               bias_lshift_hidden_new, out_rshift_hidden_new,
                               bias_hidden_new,
                               buffer_hidden,
                               mat_mult_buffer);
    // r_t * W_hn @ h_{t-1} + b_hn
    arm_mult_q7(r, buffer_hidden, buffer_hidden, hidden_size);
    // (W_in @ x_t + b_in) + (r_t * W_hn @ h_{t-1} + b_hn)
    arm_add_q7(buffer_input, buffer_hidden, n, hidden_size);
    // n = tanh [ (W_in @ x_t + b_in) + (r_t * W_hn @ h_{t-1} + b_hn) ]
    arm_nn_activations_direct_q7(n, hidden_size, new_int_bit_width, ARM_TANH);

    // ------ Hidden state computation
    // z_t * h_{t-1}
    arm_mult_q7(z, hidden, buffer_hidden, hidden_size);
    // -z_t     From this point on, z = -z_t
    arm_negate_q7(z, z, hidden_size);
    // 1 - z_t  From this point on, z = 1 - z_t
    arm_offset_q7(z, (q7_t) 0x0001, z, hidden_size);
    // (1 - z_t) * n_t
    arm_mult_q7(z, n, buffer_input, hidden_size);
    // h = (1 - z_t) * n_t + z_t * h_{t-1}
    arm_add_q7(buffer_input, buffer_hidden, hidden, hidden_size);
}