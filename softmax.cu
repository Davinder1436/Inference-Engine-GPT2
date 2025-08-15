#include "softmax.h"

__global__ void softmax_kernel(float* out, float* inp, int B, int T, int V) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * T) return;

    int V_ = V;
    float* x = inp + idx * V_;
    float* out_x = out + idx * V_;

    // find max value
    float max_val = -1e10;
    for (int i = 0; i < V_; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < V_; i++) {
        out_x[i] = expf(x[i] - max_val);
        sum += out_x[i];
    }

    // normalize
    for (int i = 0; i < V_; i++) {
        out_x[i] /= sum;
    }
}

void softmax_forward(float* out, float* inp, int B, int T, int V) {
    int block_size = 256;
    int grid_size = (B * T + block_size - 1) / block_size;
    softmax_kernel<<<grid_size, block_size>>>(out, inp, B, T, V);
    checkCuda(cudaGetLastError());
}
