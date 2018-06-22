#ifndef __KWS_CNN_H__
#define __KWS_CNN_H__

#include "kws.h"
#include "cnn_tradpool2.h"

class KWS_CNN : public KWS {
public:
  KWS_CNN(int16_t* audio_buffer);
};

#endif
