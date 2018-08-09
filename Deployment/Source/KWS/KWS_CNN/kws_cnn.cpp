#include "kws_cnn.h"

KWS_CNN::KWS_CNN(int16_t* audio_data_buffer)
{
  nn = new NAIVE_GRU();
  // nn = new CNN_TRADPOOL2();
  // nn = new CNN_TRADPOOL2_SMALL_LIN();
  audio_buffer = audio_data_buffer;
  recording_win = nn->get_num_frames();
  sliding_window_len = 1;
  init_kws();
}