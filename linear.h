#ifndef LINEAR_H
#define LINEAR_H

#include "common.h"
#include <cublas_v2.h>

void linear_forward(float* out, float* inp, float* weight, cublasHandle_t handle, int B, int T, int C, int OC);

#endif // LINEAR_H
