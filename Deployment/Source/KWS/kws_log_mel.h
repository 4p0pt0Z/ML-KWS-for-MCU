#ifndef __KWS_LOG_MEL_H__
#define __KWS_LOG_MEL_H__

#include "mbed.h"
#include "nn.h"
#include "feature_extractor.h"
#include "naive_gru.h"

class KWS_LOG_MEL{

public:
    KWS_LOG_MEL(int16_t* audio_data_buffer);
    ~KWS_LOG_MEL();
    void extract_features();
    void classify();
    void average_predictions();
    int get_top_class(q7_t* prediction);
    int16_t* audio_buffer;
    q7_t *feature_buffer;
    q7_t *output;
    q7_t *predictions;
    q7_t *averaged_output;
    int num_frames;
    int num_mel_filters;
    int frame_len;
    int frame_shift;
    int num_out_classes;
    int audio_block_size;
    int audio_buffer_size;

protected:
    virtual void init_kws();
    FeatureExtractor *feature_extractor;
    NN *nn;
    int feature_buffer_size;
    int recording_win;
    int sliding_window_len;
  
};

#endif
