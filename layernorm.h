#ifndef LAYERNORM_H
#define LAYERNORM_H

#include "common.h"

void layernorm_forward(float* out, float* inp, float* weight, float* bias, int B, int T, int C);

#endif // LAYERNORM_H
