#include "gelu.h"

__global__ void gelu_kernel(float* out, float* inp, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float x = inp[idx];
    out[idx] = 0.5f * x * (1.0f + tanhf(sqrtf(2.0f / 3.1415926535f) * (x + 0.044715f * x * x * x)));
}

void gelu_forward(float* out, float* inp, int size) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    gelu_kernel<<<grid_size, block_size>>>(out, inp, size);
    checkCuda(cudaGetLastError());
}
