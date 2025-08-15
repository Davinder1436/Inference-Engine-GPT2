#include "softmax.h"

__global__ void softmax_kernel(float* out, float* inp, int B, int T, int V) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * T) return;

    float* x = inp + idx * V;
    float* out_x = out + idx * V;

    // find max value (to prevent overflow)
    float max_val = -1e10f;
    for (int i = 0; i < V; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    // compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < V; i++) {
        float exp_val = expf(x[i] - max_val);
        out_x[i] = exp_val;
        sum += exp_val;
    }

    // normalize
    float inv_sum = 1.0f / (sum + 1e-8f); // Add small epsilon to prevent division by zero
    for (int i = 0; i < V; i++) {
        out_x[i] *= inv_sum;
    }
}

void softmax_forward(float* out, float* inp, int B, int T, int V) {
    int block_size = 128; // Reduced block size for large V
    int grid_size = (B * T + block_size - 1) / block_size;
    softmax_kernel<<<grid_size, block_size>>>(out, inp, B, T, V);
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaGetLastError());
}
