#ifndef __KWS_CNN_H__
#define __KWS_CNN_H__

#include "kws.h"
#include "feature_extractor.h"
// #include "cnn_tradpool2.h"
// #include "cnn_tradpool2_small_lin.h"
#include "naive_gru.h"


class KWS_CNN : public KWS {
public:
  KWS_CNN(int16_t* audio_buffer);
  virtual void init_kws();
};

#endif
