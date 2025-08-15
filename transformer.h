#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "common.h"
#include <cublas_v2.h>

void transformer_block_forward(float* out, float* inp, cublasHandle_t handle,
                               float* ln1_weight, float* ln1_bias,
                               float* qkv_weight, float* qkv_bias,
                               float* proj_weight, float* proj_bias,
                               float* ln2_weight, float* ln2_bias,
                               float* fc_weight, float* fc_bias,
                               float* fc_proj_weight, float* fc_proj_bias,
                               int B, int T, int C, int NH);

#endif // TRANSFORMER_H
