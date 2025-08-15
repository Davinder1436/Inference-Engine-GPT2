#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define checkCuda(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

int main() {
    std::cout << "=== ATTENTION STEP-BY-STEP DEBUG ===" << std::endl;
    
    // Test parameters (same as in main)
    int B = 1, T = 1, C = 768, NH = 12;
    int HS = C / NH;
    
    std::cout << "Parameters: B=" << B << ", T=" << T << ", C=" << C << ", NH=" << NH << ", HS=" << HS << std::endl;
    
    // Test 1: Basic memory allocation
    std::cout << "\n1. Testing basic memory allocation..." << std::endl;
    float *d_att, *d_att_softmax;
    int att_size = B * NH * T * T;
    
    checkCuda(cudaMalloc(&d_att, att_size * sizeof(float)));
    checkCuda(cudaMalloc(&d_att_softmax, att_size * sizeof(float)));
    std::cout << "   ✓ Memory allocated successfully (" << att_size << " elements)" << std::endl;
    
    // Test 2: Memory initialization
    std::cout << "\n2. Testing memory initialization..." << std::endl;
    checkCuda(cudaMemset(d_att, 0, att_size * sizeof(float)));
    
    float test_val = 1.0f;
    checkCuda(cudaMemcpy(d_att, &test_val, sizeof(float), cudaMemcpyHostToDevice));
    std::cout << "   ✓ Memory initialized successfully" << std::endl;
    
    // Test 3: Memory read back
    std::cout << "\n3. Testing memory read back..." << std::endl;
    float* h_test = new float[att_size];
    checkCuda(cudaMemcpy(h_test, d_att, att_size * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "   ✓ Memory read back successful, first value: " << h_test[0] << std::endl;
    
    // Test 4: cuBLAS setup (this might be the culprit)
    std::cout << "\n4. Testing cuBLAS operations..." << std::endl;
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status == CUBLAS_STATUS_SUCCESS) {
        std::cout << "   ✓ cuBLAS handle created successfully" << std::endl;
        
        // Test a simple cuBLAS operation
        float *d_a, *d_b, *d_c;
        int n = 64;  // HS size
        checkCuda(cudaMalloc(&d_a, n * n * sizeof(float)));
        checkCuda(cudaMalloc(&d_b, n * n * sizeof(float)));
        checkCuda(cudaMalloc(&d_c, n * n * sizeof(float)));
        
        // Initialize with small values
        checkCuda(cudaMemset(d_a, 0, n * n * sizeof(float)));
        checkCuda(cudaMemset(d_b, 0, n * n * sizeof(float)));
        test_val = 0.1f;
        checkCuda(cudaMemcpy(d_a, &test_val, sizeof(float), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(d_b, &test_val, sizeof(float), cudaMemcpyHostToDevice));
        
        // Simple matrix multiplication
        float alpha = 1.0f, beta = 0.0f;
        status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_a, n, d_b, n, &beta, d_c, n);
        
        if (status == CUBLAS_STATUS_SUCCESS) {
            std::cout << "   ✓ Basic cuBLAS operation successful" << std::endl;
        } else {
            std::cout << "   ✗ Basic cuBLAS operation failed: " << status << std::endl;
        }
        
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        cublasDestroy(handle);
    } else {
        std::cout << "   ✗ cuBLAS handle creation failed: " << status << std::endl;
    }
    
    delete[] h_test;
    cudaFree(d_att);
    cudaFree(d_att_softmax);
    
    std::cout << "\n=== All basic tests passed! The issue is likely in the attention computation logic ===" << std::endl;
    return 0;
}
