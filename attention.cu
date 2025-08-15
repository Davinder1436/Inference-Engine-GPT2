#include "attention.h"
#include "softmax.h"
#include <cublas_v2.h>

// Kernel to add bias
__global__ void add_bias_kernel(float* output, float* bias, int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * T) {
        int offset = idx * C;
        for (int i = 0; i < C; i++) {
            output[offset + i] += bias[i];
        }
    }
}

// Kernel for splitting Q, K, V and rearranging for multi-head attention
__global__ void split_qkv_kernel(float* q, float* k, float* v, float* qkv, int B, int T, int C, int NH) {
    int HS = C / NH;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * T * C) {
        int b = idx / (T * C);
        int t = (idx / C) % T;
        int c = idx % C;

        int head = c / HS;
        int h_offset = c % HS;

        int qkv_offset = b * T * 3 * C + t * 3 * C;
        int q_offset = b * NH * T * HS + head * T * HS + t * HS + h_offset;
        int k_offset = q_offset;
        int v_offset = q_offset;

        q[q_offset] = qkv[qkv_offset + c];
        k[k_offset] = qkv[qkv_offset + C + c];
        v[v_offset] = qkv[qkv_offset + 2 * C + c];
    }
}

// Causal masking kernel
__global__ void causal_mask_kernel(float* att, int T) {
    int t1 = blockIdx.x * blockDim.x + threadIdx.x;
    int t2 = blockIdx.y * blockDim.y + threadIdx.y;

    if (t1 < T && t2 < T) {
        if (t2 > t1) {
            att[t1 * T + t2] = -1e10; // Set to a large negative number
        }
    }
}


void attention_forward(float* out, float* inp, cublasHandle_t handle,
                       float* d_qkv_weight, float* d_qkv_bias,
                       float* d_proj_weight, float* d_proj_bias,
                       int B, int T, int C, int NH) {

    int HS = C / NH;

    float *d_qkv, *d_q, *d_k, *d_v, *d_att, *d_att_softmax, *d_y;
    checkCuda(cudaMalloc(&d_qkv, B * T * 3 * C * sizeof(float)));
    checkCuda(cudaMalloc(&d_q, B * NH * T * HS * sizeof(float)));
    checkCuda(cudaMalloc(&d_k, B * NH * T * HS * sizeof(float)));
    checkCuda(cudaMalloc(&d_v, B * NH * T * HS * sizeof(float)));
    checkCuda(cudaMalloc(&d_att, B * NH * T * T * sizeof(float)));
    checkCuda(cudaMalloc(&d_att_softmax, B * NH * T * T * sizeof(float)));
    checkCuda(cudaMalloc(&d_y, B * T * C * sizeof(float)));

    // --- 1. QKV projection ---
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 3 * C, B * T, C, &alpha, d_qkv_weight, C, inp, C, &beta, d_qkv, 3 * C);
    add_bias_kernel<<<(B * T + 255) / 256, 256>>>(d_qkv, d_qkv_bias, B, T, 3 * C);

    // --- 2. Split Q, K, V and rearrange ---
    split_qkv_kernel<<<(B * T * C + 255) / 256, 256>>>(d_q, d_k, d_v, d_qkv, B, T, C, NH);

    // --- 3. Scaled Dot-Product Attention ---
    // att = (q @ k.transpose(-2, -1)) / sqrt(HS)
    const float** q_ptr_array = new const float*[B * NH];
    const float** k_ptr_array = new const float*[B * NH];
    float** att_ptr_array = new float*[B * NH];

    for(int i = 0; i < B * NH; ++i) {
        q_ptr_array[i] = d_q + i * T * HS;
        k_ptr_array[i] = d_k + i * T * HS;
        att_ptr_array[i] = d_att + i * T * T;
    }

    cublasSgemmBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, T, T, HS, &alpha, k_ptr_array, HS, q_ptr_array, HS, &beta, att_ptr_array, T, B * NH);
    
    delete[] q_ptr_array;
    delete[] k_ptr_array;
    delete[] att_ptr_array;

    // Scale and causal mask
    // (This could be fused into a single kernel)
    // scale_kernel<<<...>>>(d_att, 1.0f / sqrtf(HS));
    causal_mask_kernel<<<dim3((T+15)/16, (T+15)/16), dim3(16,16)>>>(d_att, T);

    // Softmax
    softmax_forward(d_att_softmax, d_att, B * NH, T, T);

    // --- 4. Attention output ---
    // y = att_softmax @ v
    const float** att_softmax_ptr_array = new const float*[B * NH];
    const float** v_ptr_array = new const float*[B * NH];
    float** y_ptr_array = new float*[B * NH];

    for(int i = 0; i < B * NH; ++i) {
        att_softmax_ptr_array[i] = d_att_softmax + i * T * T;
        v_ptr_array[i] = d_v + i * T * HS;
        y_ptr_array[i] = d_y + i * T * HS; // Note: this is a temporary layout
    }

    cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, HS, T, T, &alpha, v_ptr_array, HS, att_softmax_ptr_array, T, &beta, y_ptr_array, HS, B * NH);

    delete[] att_softmax_ptr_array;
    delete[] v_ptr_array;
    delete[] y_ptr_array;

    // Rearrange y back to (B, T, C)
    // transpose_and_rearrange_kernel<<<...>>>(out, d_y, B, T, C, NH);

    // --- 5. Output projection ---
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, C, B * T, C, &alpha, d_proj_weight, C, d_y, C, &beta, out, C);
    add_bias_kernel<<<(B * T + 255) / 256, 256>>>(out, d_proj_bias, B, T, C);

    // Free memory
    checkCuda(cudaFree(d_qkv));
    checkCuda(cudaFree(d_q));
    checkCuda(cudaFree(d_k));
    checkCuda(cudaFree(d_v));
    checkCuda(cudaFree(d_att));
    checkCuda(cudaFree(d_att_softmax));
    checkCuda(cudaFree(d_y));
}
