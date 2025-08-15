#include "layernorm.h"

__global__ void layernorm_kernel(float* out, float* inp, float* weight, float* bias, int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * T) return;

    int C_ = C;
    float* x = inp + idx * C_;
    float* out_x = out + idx * C_;

    // calculate mean
    float mean = 0.0f;
    for (int i = 0; i < C_; i++) {
        mean += x[i];
    }
    mean /= C_;

    // calculate variance
    float variance = 0.0f;
    for (int i = 0; i < C_; i++) {
        float diff = x[i] - mean;
        variance += diff * diff;
    }
    variance /= C_;

    // normalize
    float inv_std = rsqrtf(variance + 1e-5f);
    for (int i = 0; i < C_; i++) {
        out_x[i] = weight[i] * (x[i] - mean) * inv_std + bias[i];
    }
}

void layernorm_forward(float* out, float* inp, float* weight, float* bias, int B, int T, int C) {
    int block_size = 256;
    int grid_size = (B * T + block_size - 1) / block_size;
    layernorm_kernel<<<grid_size, block_size>>>(out, inp, weight, bias, B, T, C);
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaGetLastError());
}
