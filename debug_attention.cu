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

// Debug function to test each step of attention computation
void debug_attention_step_by_step() {
    std::cout << "=== ATTENTION DEBUG MODE ===" << std::endl;
    
    // Test parameters (same as in main)
    int B = 1, T = 1, C = 768, NH = 12;
    int HS = C / NH;  // 64
    
    std::cout << "Parameters: B=" << B << ", T=" << T << ", C=" << C << ", NH=" << NH << ", HS=" << HS << std::endl;
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // === STEP 1: Test memory allocations ===
    std::cout << "\n1. Testing memory allocations..." << std::endl;
    float *d_qkv, *d_q, *d_k, *d_v, *d_att, *d_att_softmax, *d_y;
    float *d_input;
    
    try {
        checkCuda(cudaMalloc(&d_input, B * T * C * sizeof(float)));
        checkCuda(cudaMalloc(&d_qkv, B * T * 3 * C * sizeof(float)));
        checkCuda(cudaMalloc(&d_q, B * NH * T * HS * sizeof(float)));
        checkCuda(cudaMalloc(&d_k, B * NH * T * HS * sizeof(float)));
        checkCuda(cudaMalloc(&d_v, B * NH * T * HS * sizeof(float)));
        checkCuda(cudaMalloc(&d_att, B * NH * T * T * sizeof(float)));
        checkCuda(cudaMalloc(&d_att_softmax, B * NH * T * T * sizeof(float)));
        checkCuda(cudaMalloc(&d_y, B * T * C * sizeof(float)));
        std::cout << "   ✓ All memory allocations successful" << std::endl;
    } catch (...) {
        std::cout << "   ✗ Memory allocation failed!" << std::endl;
        return;
    }
    
    // === STEP 2: Initialize with safe values ===
    std::cout << "\n2. Initializing with test data..." << std::endl;
    checkCuda(cudaMemset(d_input, 0, B * T * C * sizeof(float)));
    checkCuda(cudaMemset(d_qkv, 0, B * T * 3 * C * sizeof(float)));
    checkCuda(cudaMemset(d_att, 0, B * NH * T * T * sizeof(float)));
    
    // Set some test values
    float test_val = 1.0f;
    checkCuda(cudaMemcpy(d_input, &test_val, sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_att, &test_val, sizeof(float), cudaMemcpyHostToDevice));
    std::cout << "   ✓ Test data initialized" << std::endl;
    
    // === STEP 3: Test simple memory copy ===
    std::cout << "\n3. Testing simple memory operations..." << std::endl;
    checkCuda(cudaMemcpy(d_att_softmax, d_att, B * NH * T * T * sizeof(float), cudaMemcpyDeviceToDevice));
    std::cout << "   ✓ Device-to-device copy successful" << std::endl;
    
    // Test CPU-GPU copy
    float* h_test = new float[B * NH * T * T];
    checkCuda(cudaMemcpy(h_test, d_att, B * NH * T * T * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "   ✓ GPU-to-CPU copy successful, first value: " << h_test[0] << std::endl;
    delete[] h_test;
    
    // === STEP 4: Test each attention component individually ===
    std::cout << "\n4. Testing attention components..." << std::endl;
    
    // Test QKV computation (this might be where cuBLAS fails)
    std::cout << "   4a. Testing QKV projection..." << std::endl;
    // We'll skip this since we don't have weights loaded
    
    // Test attention matrix computation
    std::cout << "   4b. Testing attention matrix..." << std::endl;
    // Initialize Q, K with simple values
    checkCuda(cudaMemset(d_q, 0, B * NH * T * HS * sizeof(float)));
    checkCuda(cudaMemset(d_k, 0, B * NH * T * HS * sizeof(float)));
    
    test_val = 0.1f;
    checkCuda(cudaMemcpy(d_q, &test_val, sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_k, &test_val, sizeof(float), cudaMemcpyHostToDevice));
    
    // Test cuBLAS batched matrix multiplication
    const float** q_ptr_array = new const float*[B * NH];
    const float** k_ptr_array = new const float*[B * NH];
    float** att_ptr_array = new float*[B * NH];
    
    for(int i = 0; i < B * NH; ++i) {
        q_ptr_array[i] = d_q + i * T * HS;
        k_ptr_array[i] = d_k + i * T * HS;
        att_ptr_array[i] = d_att + i * T * T;
    }
    
    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t cublas_status = cublasSgemmBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, T, T, HS, 
                                                      &alpha, k_ptr_array, HS, q_ptr_array, HS, 
                                                      &beta, att_ptr_array, T, B * NH);
    
    if (cublas_status == CUBLAS_STATUS_SUCCESS) {
        std::cout << "   ✓ cuBLAS batched operation successful" << std::endl;
    } else {
        std::cout << "   ✗ cuBLAS batched operation failed with status: " << cublas_status << std::endl;
    }
    
    delete[] q_ptr_array;
    delete[] k_ptr_array;
    delete[] att_ptr_array;
    
    // === STEP 5: Test the problematic softmax step ===
    std::cout << "\n5. Testing softmax step (the failing part)..." << std::endl;
    
    // First, verify the attention matrix is valid
    float* h_att_check = new float[B * NH * T * T];
    cudaError_t copy_err = cudaMemcpy(h_att_check, d_att, B * NH * T * T * sizeof(float), cudaMemcpyDeviceToHost);
    
    if (copy_err == cudaSuccess) {
        std::cout << "   ✓ Attention matrix is readable from GPU" << std::endl;
        std::cout << "   First few values: ";
        for (int i = 0; i < std::min(5, B * NH * T * T); i++) {
            std::cout << h_att_check[i] << " ";
        }
        std::cout << std::endl;
        
        // Now test our attention_softmax function with valid data
        std::cout << "   Testing attention_softmax with valid data..." << std::endl;
        // This is where we'd call attention_softmax_forward, but we know it fails
        
    } else {
        std::cout << "   ✗ Cannot read attention matrix from GPU: " << cudaGetErrorString(copy_err) << std::endl;
    }
    
    delete[] h_att_check;
    
    // === Cleanup ===
    cudaFree(d_input);
    cudaFree(d_qkv);
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_att);
    cudaFree(d_att_softmax);
    cudaFree(d_y);
    cublasDestroy(handle);
    
    std::cout << "\n=== DEBUG COMPLETE ===" << std::endl;
}

int main() {
    debug_attention_step_by_step();
    return 0;
}
