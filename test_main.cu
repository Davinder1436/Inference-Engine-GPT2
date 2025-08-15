#include "common.h"
#include "utils.h"
#include "embedding.h"
#include "layernorm.h"

#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// GPT-2 small parameters
#define NUM_LAYERS 12
#define NUM_HEADS 12
#define EMBED_DIM 768
#define VOCAB_SIZE 50257
#define MAX_SEQ_LEN 1024

// Function to upload weights to the GPU
void upload_weights_to_gpu(std::map<std::string, Tensor>& gpu_weights, const std::map<std::string, Tensor>& cpu_weights) {
    for (auto const& [name, cpu_tensor] : cpu_weights) {
        Tensor gpu_tensor;
        gpu_tensor.shape = cpu_tensor.shape;
        gpu_tensor.size = cpu_tensor.size;
        checkCuda(cudaMalloc(&gpu_tensor.data, gpu_tensor.size * sizeof(float)));
        checkCuda(cudaMemcpy(gpu_tensor.data, cpu_tensor.data, gpu_tensor.size * sizeof(float), cudaMemcpyHostToDevice));
        gpu_weights[name] = gpu_tensor;
    }
}

int main() {
    // --- Initialization ---
    std::cout << "Initializing CUDA..." << std::endl;
    
    // --- Load Weights ---
    std::cout << "Loading weights..." << std::endl;
    std::map<std::string, Tensor> cpu_weights = load_weights("gpt2_small_weights.bin");
    std::cout << "CPU weights loaded. Found " << cpu_weights.size() << " tensors." << std::endl;
    
    // Print some weight info for debugging
    for (auto const& [name, tensor] : cpu_weights) {
        std::cout << "Tensor: " << name << ", shape: [";
        for (size_t i = 0; i < tensor.shape.size(); ++i) {
            std::cout << tensor.shape[i];
            if (i < tensor.shape.size() - 1) std::cout << ", ";
        }
        std::cout << "], size: " << tensor.size << std::endl;
        
        // Only show first few tensors
        if (cpu_weights.size() > 10) {
            std::cout << "... (showing only first few)" << std::endl;
            break;
        }
    }
    
    std::cout << "Uploading weights to GPU..." << std::endl;
    std::map<std::string, Tensor> gpu_weights;
    upload_weights_to_gpu(gpu_weights, cpu_weights);
    std::cout << "Weights uploaded to GPU successfully." << std::endl;

    // --- Simple test: Try embedding lookup ---
    std::cout << "Testing embedding lookup..." << std::endl;
    
    float *d_input_embeds;
    int *d_input_tokens;
    int *h_input_tokens;
    
    checkCuda(cudaMalloc(&d_input_embeds, EMBED_DIM * sizeof(float)));
    checkCuda(cudaMalloc(&d_input_tokens, sizeof(int)));
    checkCuda(cudaHostAlloc(&h_input_tokens, sizeof(int), cudaHostAllocDefault));
    
    h_input_tokens[0] = 50256; // <|endoftext|> token
    checkCuda(cudaMemcpy(d_input_tokens, h_input_tokens, sizeof(int), cudaMemcpyHostToDevice));
    
    // Test embedding forward
    if (gpu_weights.find("transformer.wte.weight") != gpu_weights.end() && 
        gpu_weights.find("transformer.wpe.weight") != gpu_weights.end()) {
        
        std::cout << "Found embedding weights. Testing..." << std::endl;
        embedding_forward(d_input_embeds, d_input_tokens, 
                         gpu_weights["transformer.wte.weight"].data, 
                         gpu_weights["transformer.wpe.weight"].data, 
                         1, 1, EMBED_DIM);
        checkCuda(cudaDeviceSynchronize());
        std::cout << "Embedding test passed!" << std::endl;
        
    } else {
        std::cout << "ERROR: Could not find embedding weights!" << std::endl;
    }

    // --- Test LayerNorm ---
    std::cout << "Testing LayerNorm..." << std::endl;
    float *d_ln_out;
    checkCuda(cudaMalloc(&d_ln_out, EMBED_DIM * sizeof(float)));
    
    if (gpu_weights.find("transformer.ln_f.weight") != gpu_weights.end() && 
        gpu_weights.find("transformer.ln_f.bias") != gpu_weights.end()) {
        
        std::cout << "Found LayerNorm weights. Testing..." << std::endl;
        layernorm_forward(d_ln_out, d_input_embeds,
                         gpu_weights["transformer.ln_f.weight"].data,
                         gpu_weights["transformer.ln_f.bias"].data,
                         1, 1, EMBED_DIM);
        checkCuda(cudaDeviceSynchronize());
        std::cout << "LayerNorm test passed!" << std::endl;
        
    } else {
        std::cout << "ERROR: Could not find LayerNorm weights!" << std::endl;
    }

    // --- Cleanup ---
    std::cout << "Cleaning up..." << std::endl;
    for (auto const& [key, val] : gpu_weights) {
        checkCuda(cudaFree(val.data));
    }
    for (auto const& [key, val] : cpu_weights) {
        delete[] val.data;
    }
    checkCuda(cudaFree(d_input_embeds));
    checkCuda(cudaFree(d_input_tokens));
    checkCuda(cudaFree(d_ln_out));
    checkCuda(cudaFreeHost(h_input_tokens));

    std::cout << "Test completed successfully!" << std::endl;
    return 0;
}
