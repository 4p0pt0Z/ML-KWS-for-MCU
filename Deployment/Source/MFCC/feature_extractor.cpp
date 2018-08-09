#include "feature_extractor.h"

FeatureExtractor::FeatureExtractor(int num_fbank_bins, int frame_length, int feature_dec_bits,
                                   float min_freq, float max_freq)
: num_fbank_bins(num_fbank_bins), frame_length(frame_length), feature_dec_bits(feature_dec_bits),
  min_freq(min_freq), max_freq(max_freq)
{
    // Round up to the frame length to the nearest power of 2 (for fft)
    frame_length_padded = pow(2, ceil((log(frame_length) / log(2))));

    frame = new float[frame_length_padded];
    buffer = new float[frame_length_padded];
    mel_energies = new float[num_fbank_bins];

    // create array storing Hanning window function values at each frame point
    window_func = new float[frame_length];
    for (int i = 0; i < frame_length; ++i)
        window_func[i] = 0.5 - 0.5*cos(M_2PI * ((float)i) / frame_length);

    // create mel filterbank
    fbank_filter_first_non_zero_bin = new int32_t[num_fbank_bins];
    fbank_filter_last_non_zero_bin =  new int32_t[num_fbank_bins];
    mel_fbank = create_mel_fbank();  // allocation in function

    // initialize FFT
    rfft = new arm_rfft_fast_instance_f32;
    arm_rfft_fast_init_f32(rfft, frame_length_padded);
}

FeatureExtractor::~FeatureExtractor()
{
    delete[] frame;
    delete[] buffer;
    delete[] mel_energies;
    delete[] window_func;
    delete[] fbank_filter_first_non_zero_bin;
    delete[] fbank_filter_last_non_zero_bin;
    delete rfft;
    for (int i = 0; i < num_fbank_bins; ++i)
        delete[] mel_fbank[i];
    delete[] mel_fbank;
}

float** FeatureExtractor::create_mel_fbank()
{    
    int32_t num_fft_bins = frame_length_padded / 2;
    float fft_bin_width = ((float)SAMP_FREQ) / frame_length_padded;
    float mel_low_freq = MelScale(min_freq);
    float mel_high_freq = MelScale(max_freq);
    float mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_fbank_bins + 1);

    float* this_bin = new float[num_fft_bins];

    float** mel_fbank = new float*[num_fbank_bins];

    for (int32_t bin = 0; bin < num_fbank_bins; ++bin)
    {
        float left_mel = mel_low_freq + bin * mel_freq_delta;
        float center_mel = mel_low_freq + (bin + 1) * mel_freq_delta;
        float right_mel = mel_low_freq + (bin + 2) * mel_freq_delta;

        int32_t first_index = -1, last_index = -1;

        for (int32_t i = 0; i < num_fft_bins; ++i)
        {
            float freq = (fft_bin_width * i);  // center frequency for this bin
            float mel_freq = MelScale(freq);
            this_bin[i] = 0.0;

            if (mel_freq > left_mel && mel_freq < right_mel) {
                if (mel_freq <= center_mel)
                    this_bin[i] = (mel_freq - left_mel) / (center_mel - left_mel);
                else
                    this_bin[i] = (right_mel - mel_freq) / (right_mel - center_mel);
                if (first_index == -1)
                    first_index = i;
                last_index = i;
            }
        }

        fbank_filter_first_non_zero_bin[bin] = first_index;
        fbank_filter_last_non_zero_bin[bin] = last_index;
        mel_fbank[bin] = new float[last_index - first_index + 1];

        for (int32_t i = first_index, j = 0; i <= last_index; ++i, ++j)
            mel_fbank[bin][j] = this_bin[i];
    }

    delete[] this_bin;

    return mel_fbank;
}


void FeatureExtractor::extract_log_mel(const int16_t* audio_data, q7_t* log_mel_features)
{

    // TODO: Is this required at each extraction ? (check if frame gets overwritten by fft ?)
    // Pad the audio frame with zeros to get a length which is a power of 2
    memset(&frame[frame_length], 0, sizeof(float) * (frame_length_padded - frame_length));

    // Multiply by the window function
    for (int32_t i = 0; i < frame_length; ++i)
        frame[i] = audio_data[i] * window_func[i];

    // Compute FFT
    arm_rfft_fast_f32(rfft, frame, buffer, 0);

    // Convert results to power spectrum
    // fft of real is symmetric, so even if result is complex (twice as many parameters), size of array is same
    // Result of fft is stored as [real0, realN/2-1, real1, im1, real2, im2, ...]
    int32_t half_dim = frame_length_padded / 2;
    // Handle special case of first and last bins
    float first_energy = buffer[0] * buffer[0];
    float last_energy = buffer[1] * buffer[1];
    // Compute squared magnitude of the fft coeffs (continue using buffer array to save memory)
    for (int i = 1; i < half_dim; ++i)
        buffer[i] = buffer[i * 2] * buffer[i * 2] + buffer[i * 2 + 1] * buffer[i * 2 + 1];
    buffer[0] = first_energy;
    buffer[half_dim] = last_energy;
    // now buffer contains squared modulus of fft in "normal" order.

    // Apply mel filterbanks

    float sqrt_data;
    for (int32_t bin = 0; bin < num_fbank_bins; ++bin)
    {
        mel_energies[bin] = 0;
        for (int32_t i = fbank_filter_first_non_zero_bin[bin], j = 0; i <= fbank_filter_last_non_zero_bin[bin]; ++i, ++j)
        {
            arm_sqrt_f32(buffer[i], &sqrt_data);
            mel_energies[bin] += (sqrt_data) * mel_fbank[bin][j];
        }
        // Take the log (avoid log(0.0) by clipping to minimal value)
        mel_energies[bin] = mel_energies[bin] == 0.0 ? logf(FLT_MIN) : logf(mel_energies[bin]);

        // Quantization
        // Format is Qx.feature_dec_bits (determine this number with quant_test_pytorch.py script)
        mel_energies[bin] *= round((0x1<<feature_dec_bits));
        if (mel_energies[bin] >= 127)
            log_mel_features[bin] = 127;
        else if (mel_energies[bin] <= -128)
            log_mel_features[bin] = -128;
        else
            log_mel_features[bin] = mel_energies[bin];
    }
}