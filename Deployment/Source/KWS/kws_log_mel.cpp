#include "kws_log_mel.h"

#include <stdio.h>
#include <inttypes.h>

KWS_LOG_MEL::KWS_LOG_MEL(int16_t* audio_data_buffer)
{
    nn = new NAIVE_GRU();
    // nn = new CNN_TRADPOOL2();
    // nn = new CNN_TRADPOOL2_SMALL_LIN();
    audio_buffer = audio_data_buffer;
    recording_win = nn->get_num_frames();
    sliding_window_len = 1;
    init_kws();
}

KWS_LOG_MEL::~KWS_LOG_MEL()
{
  delete nn;
  delete feature_extractor;
  delete feature_buffer;
  delete output;
  delete predictions;
  delete averaged_output;
}

void KWS_LOG_MEL::init_kws()
{
  num_mel_filters = nn->get_num_mfcc_features();
  num_frames = nn->get_num_frames();
  frame_len = nn->get_frame_len();
  frame_shift = nn->get_frame_shift();
  int features_dec_bits = nn->get_in_dec_bits();
  num_out_classes = nn->get_num_out_classes();
  feature_extractor = new FeatureExtractor(num_mel_filters, frame_len, features_dec_bits, 20, 8000);
  feature_buffer = new q7_t[num_frames*num_mel_filters];
  output = new q7_t[num_out_classes];
  averaged_output = new q7_t[num_out_classes];
  predictions = new q7_t[sliding_window_len*num_out_classes];
  audio_block_size = recording_win*frame_shift;
  audio_buffer_size = audio_block_size + frame_len - frame_shift;
}

void KWS_LOG_MEL::extract_features()
{
  if(num_frames>recording_win) {
    //move old features left 
    memmove(feature_buffer,feature_buffer+(recording_win*num_mel_filters),(num_frames-recording_win)*num_mel_filters);
  }
  //compute features only for the newly recorded audio
  int32_t mfcc_buffer_head = (num_frames-recording_win)*num_mel_filters; 
  for (uint16_t f = 0; f < recording_win; f++) {
    // TODO !! Overflow if frame_shift is not divisor of audio_buffer length.
    feature_extractor->extract_log_mel(audio_buffer+(f*frame_shift),&feature_buffer[mfcc_buffer_head]);
    mfcc_buffer_head += num_mel_filters;
  }

  /*printf("Audio buffer: ...\r\n");
  for (uint16_t i = 0; i < audio_buffer_size; ++i)
    printf("%"PRId16" ", audio_buffer[i]);
  printf("\r\n");*/

  /*printf("Audio features: ...\r\n");
  for (uint16_t f = 0; f < num_frames; ++f)
  {
    for (uint16_t bin = 0; bin < num_mel_filters; ++bin)
    {
      printf("%" PRId8 " ", feature_buffer[num_mel_filters*f + bin]);
    }
    printf("\r\n");
  }*/
}

void KWS_LOG_MEL::classify()
{
  nn->run_nn(feature_buffer, output);
  // Softmax
  arm_softmax_q7(output,num_out_classes,output);
}

int KWS_LOG_MEL::get_top_class(q7_t* prediction)
{
  int max_ind=0;
  int max_val=-128;
  for(int i=0;i<num_out_classes;i++) {
    if(max_val<prediction[i]) {
      max_val = prediction[i];
      max_ind = i;
    }    
  }
  return max_ind;
}

void KWS_LOG_MEL::average_predictions()
{
  //shift right old predictions 
  arm_copy_q7((q7_t *)predictions, (q7_t *)(predictions+num_out_classes), (sliding_window_len-1)*num_out_classes);
  //add new predictions
  arm_copy_q7((q7_t *)output, (q7_t *)predictions, num_out_classes);
  //compute averages
  int sum;
  for(int j=0;j<num_out_classes;j++) {
    sum=0;
    for(int i=0;i<sliding_window_len;i++) 
      sum += predictions[i*num_out_classes+j];
    averaged_output[j] = (q7_t)(sum/sliding_window_len);
  }   
}
  
