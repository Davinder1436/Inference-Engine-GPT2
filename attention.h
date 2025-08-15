#ifndef ATTENTION_H
#define ATTENTION_H

#include "common.h"
#include <cublas_v2.h>

void attention_forward(float* out, float* inp, cublasHandle_t handle,
                       float* d_qkv_weight, float* d_qkv_bias,
                       float* d_proj_weight, float* d_proj_bias,
                       int B, int T, int C, int NH);

#endif // ATTENTION_H
