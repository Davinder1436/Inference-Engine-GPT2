import torch
from transformers import GPT2LMHeadModel
import numpy as np

def download_and_convert_gpt2_small():
    """
    Downloads the GPT-2 small model from Hugging Face, converts the weights to a
    simple binary format, and saves them to disk.
    """
    print("Downloading GPT-2 small model...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    print("Model downloaded successfully.")

    # Create a dictionary to hold the weights
    weights = {}
    for name, param in model.named_parameters():
        weights[name] = param.detach().numpy()
        print(f"Weight: {name}, Shape: {param.shape}")

    # Save the weights to a binary file
    print("\nSaving weights to binary file...")
    with open("gpt2_small_weights.bin", "wb") as f:
        for name, arr in weights.items():
            # Write the name of the tensor
            f.write(f"{name}\n".encode('utf-8'))
            # Write the shape of the tensor
            shape_str = ",".join(map(str, arr.shape))
            f.write(f"{shape_str}\n".encode('utf-8'))
            # Write the tensor data
            f.write(arr.tobytes())
    
    print("Weights saved to gpt2_small_weights.bin")

if __name__ == "__main__":
    download_and_convert_gpt2_small()
