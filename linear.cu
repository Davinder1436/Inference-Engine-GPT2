#include "linear.h"

void linear_forward(float* out, float* inp, float* weight, cublasHandle_t handle, int B, int T, int C, int OC) {
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, OC, B * T, C, &alpha, weight, C, inp, C, &beta, out, OC);
}
