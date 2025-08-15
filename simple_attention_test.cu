#include <iostream>
#include <cuda_runtime.h>
#include "common.h"

// Ultra-simple test to isolate the attention memory issue
__global__ void test_simple_copy(float* out, float* inp, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = inp[idx];
    }
}

int main() {
    const int B = 1, NH = 12, T = 1;  // Start with the simplest case
    const int size = B * NH * T * T;
    
    float *d_in, *d_out;
    cudaMalloc(&d_in, size * sizeof(float));
    cudaMalloc(&d_out, size * sizeof(float));
    
    // Initialize with simple values
    float init_val = 1.0f;
    cudaMemset(d_in, 0, size * sizeof(float));
    cudaMemcpy(d_in, &init_val, sizeof(float), cudaMemcpyHostToDevice);
    
    // Test simple copy
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    test_simple_copy<<<grid_size, block_size>>>(d_out, d_in, size);
    
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cout << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    std::cout << "Simple copy test passed!" << std::endl;
    
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
