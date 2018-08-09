#include "arm_nnsupportfunctions.h"
#include "arm_nn_tables.h"

#define USE_INTRINSIC

#ifdef __cplusplus
extern    "C"
{
#endif

arm_status arm_convolve_HWC_q7_basic_nonsquare(const q7_t * Im_in,
        const uint16_t dim_im_in_X,
        const uint16_t dim_im_in_Y,
        const uint16_t ch_im_in,
        const q7_t * wt,
        const uint16_t ch_im_out,
        const uint16_t dim_kernel_X,
        const uint16_t dim_kernel_Y,
        const uint16_t padding_X,
        const uint16_t padding_Y,
        const uint16_t stride_X,
        const uint16_t stride_Y,
        const q7_t * bias,
        const uint16_t bias_shift,
        const uint16_t out_shift,
        q7_t * Im_out, 
        const uint16_t dim_im_out_X, 
        const uint16_t dim_im_out_Y, 
        q15_t * bufferA, 
        q7_t * bufferB);

void arm_avepool_q7_HWC_nonsquare (
        const q7_t * Im_in,         
        const uint16_t dim_im_in_x,   
        const uint16_t dim_im_in_y,   
        const uint16_t ch_im_in,    
        const uint16_t dim_kernel_x,  
        const uint16_t dim_kernel_y,  
        const uint16_t padding_x,     
        const uint16_t padding_y,     
        const uint16_t stride_x,      
        const uint16_t stride_y,      
        const uint16_t dim_im_out_x,  
        const uint16_t dim_im_out_y,  
        q7_t * bufferA,             
        q7_t * Im_out,
        const uint16_t out_lshift);

void maxpool_q7_HWC_nonsquare (
        const q7_t * Im_in,
        const uint16_t dim_im_in_x,
        const uint16_t dim_im_in_y,
        const uint16_t ch_im_in,
        const uint16_t dim_kernel_x,
        const uint16_t dim_kernel_y,
        const uint16_t padding_x,
        const uint16_t padding_y,
        const uint16_t stride_x,
        const uint16_t stride_y,
        const uint16_t dim_im_out_x,
        const uint16_t dim_im_out_y,
        q7_t * bufferA,
        q7_t * Im_out,
        const uint16_t out_lshift);

void GRU_layer_q7_q7(
        q7_t * input, uint16_t input_size, uint16_t hidden_size,
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
        q7_t * scratch_buffer);

#ifdef __cplusplus
}
#endif


