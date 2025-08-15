#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "common.h"

void softmax_forward(float* out, float* inp, int B, int T, int V);

#endif // SOFTMAX_H
