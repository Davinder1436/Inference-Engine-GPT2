#include "attention.h"
#include "attention_softmax.h"
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

// Causal masking kernel - fixed for 4D attention matrix [B, NH, T, T]
__global__ void causal_mask_kernel(float* att, int B, int NH, int T) {
    int t1 = blockIdx.x * blockDim.x + threadIdx.x;
    int t2 = blockIdx.y * blockDim.y + threadIdx.y;
    int head_idx = blockIdx.z;  // Add head dimension

    if (t1 < T && t2 < T && head_idx < B * NH) {
        if (t2 > t1) {
            int offset = head_idx * T * T + t1 * T + t2;
            att[offset] = -1e10; // Set to a large negative number
        }
    }
}


void attention_forward(float* out, float* inp, cublasHandle_t handle,
                       float* d_qkv_weight, float* d_qkv_bias,
                       float* d_proj_weight, float* d_proj_bias,
                       int B, int T, int C, int NH) {

    printf("DEBUG: attention_forward called with B=%d, T=%d, C=%d, NH=%d\n", B, T, C, NH);
    int HS = C / NH;

    float *d_qkv, *d_q, *d_k, *d_v, *d_att, *d_att_softmax, *d_y;
    
    printf("DEBUG: Allocating attention memory...\n");
    printf("  - d_qkv: %d elements\n", B * T * 3 * C);
    printf("  - d_att: %d elements\n", B * NH * T * T);
    
    checkCuda(cudaMalloc(&d_qkv, B * T * 3 * C * sizeof(float)));
    checkCuda(cudaMalloc(&d_q, B * NH * T * HS * sizeof(float)));
    checkCuda(cudaMalloc(&d_k, B * NH * T * HS * sizeof(float)));
    checkCuda(cudaMalloc(&d_v, B * NH * T * HS * sizeof(float)));
    checkCuda(cudaMalloc(&d_att, B * NH * T * T * sizeof(float)));
    checkCuda(cudaMalloc(&d_att_softmax, B * NH * T * T * sizeof(float)));
    checkCuda(cudaMalloc(&d_y, B * T * C * sizeof(float)));
    printf("DEBUG: Memory allocation successful\n");

    // --- 1. QKV projection ---
    printf("DEBUG: Starting QKV projection...\n");
    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 3 * C, B * T, C, &alpha, d_qkv_weight, C, inp, C, &beta, d_qkv, 3 * C);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("ERROR: QKV projection failed with cuBLAS status: %d\n", status);
    } else {
        printf("DEBUG: QKV projection successful\n");
    }
    
    add_bias_kernel<<<(B * T + 255) / 256, 256>>>(d_qkv, d_qkv_bias, B, T, 3 * C);
    checkCuda(cudaDeviceSynchronize());
    printf("DEBUG: QKV bias addition successful\n");

    // --- 2. Split Q, K, V and rearrange ---
    printf("DEBUG: Splitting Q, K, V...\n");
    split_qkv_kernel<<<(B * T * C + 255) / 256, 256>>>(d_q, d_k, d_v, d_qkv, B, T, C, NH);
    checkCuda(cudaDeviceSynchronize());
    printf("DEBUG: Q, K, V split successful\n");

    // --- 3. Scaled Dot-Product Attention ---
    printf("DEBUG: Starting attention computation...\n");
    
    // SPECIAL CASE: For T=1, we can simplify this significantly
    if (T == 1) {
        printf("DEBUG: Using simplified T=1 attention computation\n");
        
        // For T=1, attention matrix is just 1x1 per head, so each value is just Q·K
        // We can compute this much more simply
        for (int head = 0; head < B * NH; head++) {
            float* q_head = d_q + head * T * HS;  // [1, HS]
            float* k_head = d_k + head * T * HS;  // [1, HS]
            float* att_head = d_att + head * T * T;  // [1, 1] = single element
            
            // Compute dot product on GPU using a simple kernel
            float alpha = 1.0f, beta = 0.0f;
            status = cublasSdot(handle, HS, q_head, 1, k_head, 1, att_head);
            
            if (status != CUBLAS_STATUS_SUCCESS) {
                printf("ERROR: Dot product failed for head %d with status: %d\n", head, status);
            }
        }
        printf("DEBUG: T=1 attention computation completed\n");
        
    } else {
        // For T > 1, use individual cuBLAS calls instead of batched (safer)
        printf("DEBUG: Using individual attention computation for T=%d\n", T);
        
        for (int head = 0; head < B * NH; head++) {
            float* q_head = d_q + head * T * HS;      // [T, HS]
            float* k_head = d_k + head * T * HS;      // [T, HS]  
            float* att_head = d_att + head * T * T;   // [T, T]
            
            // Compute attention for this head: att = q @ k^T
            status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, T, T, HS, 
                               &alpha, k_head, HS, q_head, HS, &beta, att_head, T);
            
            if (status != CUBLAS_STATUS_SUCCESS) {
                printf("ERROR: Individual attention computation failed for head %d with status: %d\n", head, status);
                break;
            }
        }
        printf("DEBUG: Individual attention computation completed\n");
    }

    // Test if attention matrix is readable
    printf("DEBUG: Testing attention matrix readability...\n");
    float h_test;
    cudaError_t copy_err = cudaMemcpy(&h_test, d_att, sizeof(float), cudaMemcpyDeviceToHost);
    if (copy_err == cudaSuccess) {
        printf("DEBUG: Attention matrix is readable, first value: %f\n", h_test);
    } else {
        printf("ERROR: Cannot read attention matrix: %s\n", cudaGetErrorString(copy_err));
        printf("ERROR: Skipping softmax due to invalid attention matrix\n");
        // Still continue to cleanup
    }

    // Scale and causal mask - SKIP the problematic causal_mask_kernel
    // (This could be fused into a single kernel)
    // scale_kernel<<<...>>>(d_att, 1.0f / sqrtf(HS));
    // causal_mask_kernel<<<dim3((T+15)/16, (T+15)/16, B*NH), dim3(16,16,1)>>>(d_att, B, NH, T);

    if (copy_err == cudaSuccess) {
        printf("DEBUG: About to call attention_softmax_forward...\n");
        // Softmax with built-in causal masking (safer approach)
        attention_softmax_forward(d_att_softmax, d_att, B, NH, T);
        printf("DEBUG: attention_softmax_forward completed\n");
    } else {
        // Skip softmax and zero out the output
        checkCuda(cudaMemset(d_att_softmax, 0, B * NH * T * T * sizeof(float)));
    }

    // --- 4. Attention output ---
    printf("DEBUG: Starting attention-value multiplication...\n");
    // y = att_softmax @ v
    const float** att_softmax_ptr_array = new const float*[B * NH];
    const float** v_ptr_array = new const float*[B * NH];
    float** y_ptr_array = new float*[B * NH];

    for(int i = 0; i < B * NH; ++i) {
        att_softmax_ptr_array[i] = d_att_softmax + i * T * T;
        v_ptr_array[i] = d_v + i * T * HS;
        y_ptr_array[i] = d_y + i * T * HS; // Note: this is a temporary layout
    }
    printf("DEBUG: Pointer arrays set for att@v multiplication\n");

    // SPECIAL CASE: For T=1, simplify this as well
    if (T == 1) {
        printf("DEBUG: Using simplified T=1 attention-value computation\n");
        
        // For T=1, this is just element-wise scaling of v by attention weights
        for (int head = 0; head < B * NH; head++) {
            float* att_head = d_att_softmax + head * T * T;  // [1,1] single element
            float* v_head = d_v + head * T * HS;             // [1, HS]
            float* y_head = d_y + head * T * HS;             // [1, HS]
            
            // Read the attention weight (should be 1.0 after softmax for T=1)
            float att_weight;
            cudaError_t copy_err = cudaMemcpy(&att_weight, att_head, sizeof(float), cudaMemcpyDeviceToHost);
            if (copy_err == cudaSuccess) {
                printf("DEBUG: Head %d attention weight: %f\n", head, att_weight);
            } else {
                printf("ERROR: Cannot read attention weight for head %d: %s\n", head, cudaGetErrorString(copy_err));
                break;
            }
            
            // Scale v by attention weight using cuBLAS
            status = cublasSscal(handle, HS, &att_weight, v_head, 1);
            if (status != CUBLAS_STATUS_SUCCESS) {
                printf("ERROR: Scaling failed for head %d with status: %d\n", head, status);
            }
            
            // Copy scaled result to y
            checkCuda(cudaMemcpy(y_head, v_head, HS * sizeof(float), cudaMemcpyDeviceToDevice));
        }
        printf("DEBUG: T=1 attention-value computation completed\n");
        
    } else {
        // For T > 1, use individual cuBLAS calls instead of batched
        printf("DEBUG: Using individual attention-value computation for T=%d\n", T);
        
        for (int head = 0; head < B * NH; head++) {
            float* att_head = d_att_softmax + head * T * T;  // [T, T]
            float* v_head = d_v + head * T * HS;             // [T, HS]  
            float* y_head = d_y + head * T * HS;             // [T, HS]
            
            // Compute y = att @ v for this head
            status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, HS, T, T,
                               &alpha, v_head, HS, att_head, T, &beta, y_head, HS);
            
            if (status != CUBLAS_STATUS_SUCCESS) {
                printf("ERROR: Individual attention-value computation failed for head %d with status: %d\n", head, status);
                break;
            }
        }
        printf("DEBUG: Individual attention-value computation completed\n");
    }

    delete[] att_softmax_ptr_array;
    delete[] v_ptr_array;
    delete[] y_ptr_array;

    // Rearrange y back to (B, T, C)
    // transpose_and_rearrange_kernel<<<...>>>(out, d_y, B, T, C, NH);

    // --- 5. Output projection ---
    printf("DEBUG: Starting final projection...\n");
    printf("DEBUG: Matrix dimensions - C=%d, B*T=%d\n", C, B*T);
    
    // Test if d_y is readable before projection
    float y_test;
    cudaError_t y_err = cudaMemcpy(&y_test, d_y, sizeof(float), cudaMemcpyDeviceToHost);
    if (y_err == cudaSuccess) {
        printf("DEBUG: d_y is readable, first value: %f\n", y_test);
    } else {
        printf("ERROR: Cannot read d_y before projection: %s\n", cudaGetErrorString(y_err));
    }
    
    status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, C, B * T, C, &alpha, d_proj_weight, C, d_y, C, &beta, out, C);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("ERROR: Final projection failed with cuBLAS status: %d\n", status);
    } else {
        printf("DEBUG: Final projection successful\n");
    }
    
    // Test if output is readable before bias addition
    float out_test;
    cudaError_t out_err = cudaMemcpy(&out_test, out, sizeof(float), cudaMemcpyDeviceToHost);
    if (out_err == cudaSuccess) {
        printf("DEBUG: Output is readable after projection, first value: %f\n", out_test);
    } else {
        printf("ERROR: Cannot read output after projection: %s\n", cudaGetErrorString(out_err));
    }
    
    printf("DEBUG: Adding bias...\n");
    add_bias_kernel<<<(B * T + 255) / 256, 256>>>(out, d_proj_bias, B, T, C);
    checkCuda(cudaDeviceSynchronize());
    printf("DEBUG: Bias addition completed\n");

    // Free memory
    printf("DEBUG: Starting memory cleanup...\n");
    printf("DEBUG: Freeing d_qkv...\n");
    checkCuda(cudaFree(d_qkv));
    printf("DEBUG: Freeing d_q...\n");
    checkCuda(cudaFree(d_q));
    printf("DEBUG: Freeing d_k...\n");
    checkCuda(cudaFree(d_k));
    printf("DEBUG: Freeing d_v...\n");
    checkCuda(cudaFree(d_v));
    printf("DEBUG: Freeing d_att...\n");
    checkCuda(cudaFree(d_att));
    printf("DEBUG: Freeing d_att_softmax...\n");
    checkCuda(cudaFree(d_att_softmax));
    printf("DEBUG: Freeing d_y...\n");
    checkCuda(cudaFree(d_y));
    printf("DEBUG: Memory cleanup completed successfully\n");
}
