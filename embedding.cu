#include "embedding.h"

__global__ void embedding_kernel(float* out, const int* tokens, const float* wte, const float* wpe, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;

    if (idx < T) {
        int token_idx = tokens[b * T + idx];
        int out_offset = b * T * C + idx * C;
        const float* wte_row = wte + token_idx * C;
        const float* wpe_row = wpe + idx * C; // Position embedding for current position

        for (int i = 0; i < C; i++) {
            out[out_offset + i] = wte_row[i] + wpe_row[i];
        }
    }
}

void embedding_forward(float* out, int* tokens, float* wte, float* wpe, int B, int T, int C) {
    dim3 block_size(256);
    dim3 grid_size((T + block_size.x - 1) / block_size.x, B);
    embedding_kernel<<<grid_size, block_size>>>(out, tokens, wte, wpe, T, C);
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaGetLastError());
}
