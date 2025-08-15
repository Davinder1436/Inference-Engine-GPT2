#include "common.h"
#include "utils.h"
#include "embedding.h"
#include "transformer.h"
#include "layernorm.h"
#include "linear.h"
#include "softmax.h"

#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>

#include <cublas_v2.h>
#include <cuda_runtime.h>

// GPT-2 small parameters
#define NUM_LAYERS 12
#define NUM_HEADS 12
#define EMBED_DIM 768
#define VOCAB_SIZE 50257
#define MAX_SEQ_LEN 1024

// --- Sampling Functions ---
void top_k_sampling(float* logits, int vocab_size, int k, std::vector<int>& top_indices, std::vector<float>& top_probs) {
    std::vector<std::pair<float, int>> logit_pairs;
    for (int i = 0; i < vocab_size; ++i) {
        logit_pairs.push_back({logits[i], i});
    }
    std::partial_sort(logit_pairs.begin(), logit_pairs.begin() + k, logit_pairs.end(), std::greater<std::pair<float, int>>());

    float sum_probs = 0.0f;
    for (int i = 0; i < k; ++i) {
        top_probs.push_back(logit_pairs[i].first);
        sum_probs += top_probs.back();
    }
    for (int i = 0; i < k; ++i) {
        top_probs[i] /= sum_probs;
        top_indices.push_back(logit_pairs[i].second);
    }
}

int sample_from_distribution(const std::vector<float>& probs) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> dist(probs.begin(), probs.end());
    return dist(gen);
}


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
    cublasHandle_t handle;
    cublasCreate(&handle);

    // --- Load Weights ---
    std::map<std::string, Tensor> cpu_weights = load_weights("gpt2_small_weights.bin");
    std::map<std::string, Tensor> gpu_weights;
    upload_weights_to_gpu(gpu_weights, cpu_weights);
    std::cout << "Weights loaded to GPU successfully." << std::endl;

    // --- Allocate GPU Memory ---
    float *d_transformer_out, *d_input_embeds, *d_final_norm_out, *d_logits;
    int *d_input_tokens;
    checkCuda(cudaMalloc(&d_input_embeds, MAX_SEQ_LEN * EMBED_DIM * sizeof(float)));
    checkCuda(cudaMalloc(&d_transformer_out, MAX_SEQ_LEN * EMBED_DIM * sizeof(float)));
    checkCuda(cudaMalloc(&d_final_norm_out, MAX_SEQ_LEN * EMBED_DIM * sizeof(float)));
    checkCuda(cudaMalloc(&d_logits, MAX_SEQ_LEN * VOCAB_SIZE * sizeof(float)));
    checkCuda(cudaMalloc(&d_input_tokens, MAX_SEQ_LEN * sizeof(int)));

    // --- Pinned Host Memory for Token Transfer ---
    int* h_input_tokens;
    float* h_logits;
    checkCuda(cudaHostAlloc(&h_input_tokens, MAX_SEQ_LEN * sizeof(int), cudaHostAllocDefault));
    checkCuda(cudaHostAlloc(&h_logits, VOCAB_SIZE * sizeof(float), cudaHostAllocDefault));


    // --- Tokenizer (Simplified) ---
    h_input_tokens[0] = 50256; // <|endoftext|> token

    // --- Inference Loop ---
    std::cout << "Starting inference..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    int B = 1;
    int T = 1;

    for (int step = 0; step < 100; ++step) { // Generate 100 tokens
        checkCuda(cudaMemcpy(d_input_tokens, h_input_tokens, T * sizeof(int), cudaMemcpyHostToDevice));

        // 1. Embedding Lookup
        embedding_forward(d_input_embeds, d_input_tokens, gpu_weights["wte.weight"].data, gpu_weights["wpe.weight"].data, B, T, EMBED_DIM);

        // 2. Transformer Blocks
        float* current_input = d_input_embeds;
        for (int i = 0; i < NUM_LAYERS; ++i) {
            std::string l = "h." + std::to_string(i) + ".";
            transformer_block_forward(d_transformer_out, current_input, handle,
                                      gpu_weights[l + "ln_1.weight"].data, gpu_weights[l + "ln_1.bias"].data,
                                      gpu_weights[l + "attn.c_attn.weight"].data, gpu_weights[l + "attn.c_attn.bias"].data,
                                      gpu_weights[l + "attn.c_proj.weight"].data, gpu_weights[l + "attn.c_proj.bias"].data,
                                      gpu_weights[l + "ln_2.weight"].data, gpu_weights[l + "ln_2.bias"].data,
                                      gpu_weights[l + "mlp.c_fc.weight"].data, gpu_weights[l + "mlp.c_fc.bias"].data,
                                      gpu_weights[l + "mlp.c_proj.weight"].data, gpu_weights[l + "mlp.c_proj.bias"].data,
                                      B, T, EMBED_DIM, NUM_HEADS);
            current_input = d_transformer_out;
        }

        // 3. Final LayerNorm
        layernorm_forward(d_final_norm_out, d_transformer_out, gpu_weights["ln_f.weight"].data, gpu_weights["ln_f.bias"].data, B, T, EMBED_DIM);

        // 4. LM Head (logits)
        linear_forward(d_logits, d_final_norm_out, gpu_weights["wte.weight"].data, handle, B, T, EMBED_DIM, VOCAB_SIZE);

        // 5. Sample next token
        checkCuda(cudaMemcpy(h_logits, d_logits + (T - 1) * VOCAB_SIZE, VOCAB_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
        
        std::vector<int> top_indices;
        std::vector<float> top_probs;
        top_k_sampling(h_logits, VOCAB_SIZE, 50, top_indices, top_probs);
        int next_token = top_indices[sample_from_distribution(top_probs)];

        if (T < MAX_SEQ_LEN) {
            h_input_tokens[T] = next_token;
            T++;
        } else {
            // Shift context window
            for(int i = 0; i < MAX_SEQ_LEN - 1; ++i) h_input_tokens[i] = h_input_tokens[i+1];
            h_input_tokens[MAX_SEQ_LEN - 1] = next_token;
        }
        
        std::cout << "Token: " << next_token << std::endl;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Generated 100 tokens in " << duration.count() << " ms" << std::endl;
    std::cout << "Tokens/sec: " << 100.0 / (duration.count() / 1000.0) << std::endl;


    // --- Cleanup ---
    for (auto const& [key, val] : gpu_weights) {
        checkCuda(cudaFree(val.data));
    }
    for (auto const& [key, val] : cpu_weights) {
        delete[] val.data;
    }
    checkCuda(cudaFree(d_input_embeds));
    checkCuda(cudaFree(d_transformer_out));
    checkCuda(cudaFree(d_final_norm_out));
    checkCuda(cudaFree(d_logits));
    checkCuda(cudaFree(d_input_tokens));
    checkCuda(cudaFreeHost(h_input_tokens));
    checkCuda(cudaFreeHost(h_logits));
    cublasDestroy(handle);

    return 0;
}
