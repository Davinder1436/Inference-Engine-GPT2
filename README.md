# GPT-2 Small Inference Engine

This project implements a minimal inference engine for GPT-2 small using CUDA.

## Goal

- Implement a working LLM forward pass loop for GPT-2 small.
- Implement the transformer block forward pass in CUDA.
- Implement self-attention with causal masking.
- Implement a sampling loop (top-k, top-p).
- Optimize memory transfer (pinned host memory, avoid CPU-GPU ping-pong).
- Measure token generation speed (tokens/sec).

## Steps

1.  **Download GPT-2 Small Weights:** A Python script will be provided to download the model weights from Hugging Face and convert them to a binary format suitable for our CUDA application.
2.  **Implement CUDA Kernels:**
    *   Transformer Block
    *   Self-Attention with Causal Masking
    *   Layer Normalization
    *   GELU Activation
    *   Softmax
3.  **Implement Inference Loop:**
    *   Load model weights.
    *   Implement the main forward pass loop.
    *   Implement top-k/top-p sampling.
4.  **Build and Run:**
    *   Compile the CUDA code using `nvcc`.
    *   Run the inference engine and measure performance.
# Inference-Engine-GPT2
