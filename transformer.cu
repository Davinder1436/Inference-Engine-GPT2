#include "transformer.h"
#include "layernorm.h"
#include "attention.h"
#include "gelu.h"
#include <cublas_v2.h>

__global__ void add_kernel(float* out, float* a, float* b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

__global__ void add_bias_kernel_transformer(float* output, float* bias, int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * T) {
        int offset = idx * C;
        for (int i = 0; i < C; i++) {
            output[offset + i] += bias[i];
        }
    }
}


void transformer_block_forward(float* out, float* inp, cublasHandle_t handle,
                               float* ln1_weight, float* ln1_bias,
                               float* qkv_weight, float* qkv_bias,
                               float* proj_weight, float* proj_bias,
                               float* ln2_weight, float* ln2_bias,
                               float* fc_weight, float* fc_bias,
                               float* fc_proj_weight, float* fc_proj_bias,
                               int B, int T, int C, int NH) {

    float *ln1_out, *attn_out, *residual1, *ln2_out, *fc_out, *gelu_out, *fc_proj_out;
    checkCuda(cudaMalloc(&ln1_out, B * T * C * sizeof(float)));
    checkCuda(cudaMalloc(&attn_out, B * T * C * sizeof(float)));
    checkCuda(cudaMalloc(&residual1, B * T * C * sizeof(float)));
    checkCuda(cudaMalloc(&ln2_out, B * T * C * sizeof(float)));
    checkCuda(cudaMalloc(&fc_out, B * T * 4 * C * sizeof(float)));
    checkCuda(cudaMalloc(&gelu_out, B * T * 4 * C * sizeof(float)));
    checkCuda(cudaMalloc(&fc_proj_out, B * T * C * sizeof(float)));

    // --- 1. LayerNorm 1 ---
    layernorm_forward(ln1_out, inp, ln1_weight, ln1_bias, B, T, C);

    // --- 2. Multi-Head Attention ---
    attention_forward(attn_out, ln1_out, handle, qkv_weight, qkv_bias, proj_weight, proj_bias, B, T, C, NH);

    // --- 3. Residual Connection 1 ---
    add_kernel<<<(B * T * C + 255) / 256, 256>>>(residual1, inp, attn_out, B * T * C);

    // --- 4. LayerNorm 2 ---
    layernorm_forward(ln2_out, residual1, ln2_weight, ln2_bias, B, T, C);

    // --- 5. Feed-Forward Network ---
    float alpha = 1.0f, beta = 0.0f;
    // fc_out = ln2_out @ fc_weight^T + fc_bias
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 4 * C, B * T, C, &alpha, fc_weight, C, ln2_out, C, &beta, fc_out, 4 * C);
    add_bias_kernel_transformer<<<(B * T + 255) / 256, 256>>>(fc_out, fc_bias, B, T, 4*C);

    // gelu_out = gelu(fc_out)
    gelu_forward(gelu_out, fc_out, B * T * 4 * C);

    // out = gelu_out @ fc_proj_weight^T + fc_proj_bias
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, C, B * T, 4 * C, &alpha, fc_proj_weight, 4*C, gelu_out, 4*C, &beta, fc_proj_out, C);
    add_bias_kernel_transformer<<<(B * T + 255) / 256, 256>>>(fc_proj_out, fc_proj_bias, B, T, C);

    // --- 6. Residual Connection 2 ---
    add_kernel<<<(B * T * C + 255) / 256, 256>>>(out, residual1, fc_proj_out, B * T * C);

    // Free memory
    checkCuda(cudaFree(ln1_out));
    checkCuda(cudaFree(attn_out));
    checkCuda(cudaFree(residual1));
    checkCuda(cudaFree(ln2_out));
    checkCuda(cudaFree(fc_out));
    checkCuda(cudaFree(gelu_out));
    checkCuda(cudaFree(fc_proj_out));
}
