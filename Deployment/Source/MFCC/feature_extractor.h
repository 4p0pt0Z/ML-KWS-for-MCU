#ifndef __FEATURE_EXTRACTOR_H__
#define __FEATURE_EXTRACTOR_H__


#include "float.h"
#include "arm_math.h"


#define M_2PI 6.283185307179586476925286766559005
#define SAMP_FREQ 16000 // audio sampling rate

class FeatureExtractor{
    private:
        int num_fbank_bins;  // number of filter in the mel filter bank.
        int frame_length;    // length of the audio frames
        int frame_length_padded;  // audio frames will be padded as fft takes array which length is a power of 2
        int feature_dec_bits;  // number of decimal bits to use when quantizing the features
        float min_freq;  // smallest frequency to consider
        float max_freq;  // highest frequency to consider
        float* frame;  // array for audio frame (size: length_padded)
        float* buffer;  // (size: length_padded)
        float* mel_energies;  // (size: num_fbank_bins)
        float* window_func;  // array with values of the window function (size: frame_len)
        int32_t* fbank_filter_first_non_zero_bin;  // Array with the index of the first non-zero coefficient in each mel filter (size: num_fbank_bins)
        int32_t* fbank_filter_last_non_zero_bin;   // Array with the index of the last non-zero coefficient in each mel filter (size: num_fbank_bins)
        float** mel_fbank;  // (size: num_fbank_bins * x where x is the number of non-zero coefficient for the particular filter)
        arm_rfft_fast_instance_f32 * rfft;  // real fast fourier transform calculator
        
        float** create_mel_fbank();  // allocate and compute the filters in the mel filterbank

        static inline float InverseMelScale(float mel_freq) {
            return 700.0f * (expf (mel_freq / 1127.0f) - 1.0f);
        }

        static inline float MelScale(float freq) {
            return 1127.0f * logf (1.0f + freq / 700.0f);
        }

    public:
        FeatureExtractor(int num_fbank_bins, int frame_length, int feature_dec_bits,
                         float min_freq = 20, float max_freq = 4000);
        ~FeatureExtractor();
        void extract_log_mel(const int16_t* data, q7_t* log_mel_features);
};

#endif