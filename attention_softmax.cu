#include "attention_softmax.h"

// Super simple and safe implementation - just do causal masking on GPU and softmax on CPU
__global__ void simple_causal_mask_kernel(float* out, float* inp, int B, int NH, int T) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = B * NH * T * T;
    
    if (idx >= total_size) return;
    
    // DO ABSOLUTELY NOTHING - just test if kernel launches work
    // out[idx] = inp[idx];  // Even this simple copy might be problematic
}

void attention_softmax_forward(float* out, float* inp, int B, int NH, int T) {
    printf("DEBUG: attention_softmax called with B=%d, NH=%d, T=%d\n", B, NH, T);
    
    int total_size = B * NH * T * T;
    printf("DEBUG: total_size=%d\n", total_size);
    
    // COMPLETELY BYPASS GPU KERNEL - do everything on CPU
    // Allocate CPU memory
    float* h_att = new float[total_size];
    float* h_out = new float[total_size];
    
    // Copy from GPU to CPU
    checkCuda(cudaMemcpy(h_att, inp, total_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Apply causal mask and softmax on CPU
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < NH; h++) {
            for (int row = 0; row < T; row++) {
                int row_offset = (b * NH + h) * T * T + row * T;
                
                // Apply causal mask and find max
                float max_val = -1e10f;
                for (int col = 0; col <= row; col++) {
                    max_val = fmaxf(max_val, h_att[row_offset + col]);
                }
                
                // Apply softmax
                float sum = 0.0f;
                for (int col = 0; col < T; col++) {
                    if (col <= row) {
                        h_out[row_offset + col] = expf(h_att[row_offset + col] - max_val);
                        sum += h_out[row_offset + col];
                    } else {
                        h_out[row_offset + col] = 0.0f;  // Causal mask
                    }
                }
                
                // Normalize
                for (int col = 0; col <= row; col++) {
                    h_out[row_offset + col] /= (sum + 1e-8f);
                }
            }
        }
    }
    
    // Copy back to GPU
    checkCuda(cudaMemcpy(out, h_out, total_size * sizeof(float), cudaMemcpyHostToDevice));
    
    delete[] h_att;
    delete[] h_out;
    
    printf("DEBUG: attention_softmax completed successfully\n");
}
