#ifndef ATTENTION_SOFTMAX_H
#define ATTENTION_SOFTMAX_H

#include "common.h"

void attention_softmax_forward(float* out, float* inp, int B, int NH, int T);

#endif // ATTENTION_SOFTMAX_H
