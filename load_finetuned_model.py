import os
import json
import torch
from transformers import GPT2Tokenizer
from safetensors.torch import load_file
from finetune_gpt2 import GPT, GPTConfig

def load_model_and_tokenizer(model_path: str, config_path: str):
    """
    Load and return the tokenizer and model with weights.

    Parameters:
    - model_path (str): Path to the model weights.
    - config_path (str): Path to the configuration file.

    Returns:
    - tokenizer (GPT2Tokenizer): The tokenizer instance.
    - model (GPT): The model instance.
    """
    
    # Select the device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    # Load the configuration
    with open(config_path, "r", encoding='utf-8') as f:
        config_data = json.load(f)

    config = GPTConfig(**config_data)
    model = GPT(config)

    # Load the weights
    state_dict = load_file(os.path.join(model_path, "model.safetensors"))

    # Weight sharing
    if 'lm_head.weight' in state_dict:
        state_dict['transformer.wte.weight'] = state_dict['lm_head.weight']
    
    # Move state_dict tensors to the correct device
    state_dict = {k: v.to(device) for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(device)  # Move the model to the correct device

    # Initialize the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    return tokenizer, model, device, config
