import os
from transformers import GPT2Tokenizer
import torch
from torch.nn import functional as F
import json
from safetensors.torch import load_file
from finetune_gpt2 import GPT, GPTConfig
from modeling_value_head import AutoModelForCausalLMWithValueHead


"""
This script is for verifying the  finetuned model. Generate coherent text
and also the added ValueHead to generate the estimated returns for tokens. 

"""

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

# load the final checkpoint
model_path = "checkpoints/checkpoint_57.pt"
with open("checkpoints/config.json", "r", encoding='utf-8') as f:
    config_data = json.load(f)

config = GPTConfig(**config_data)
model = GPT(config)

# Load the weights
state_dict = load_file(os.path.join(model_path, "model.safetensors"))

# weight sharing
if 'lm_head.weight' in state_dict:
    state_dict['transformer.wte.weight'] = state_dict['lm_head.weight']

state_dict = {k: v.to(device) for k, v in state_dict.items()}

model.load_state_dict(state_dict, strict=False)

model.to(device)  
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def generate_text(model, tokenizer, prompt, num_return_sequences=5, max_length=30, temperature=0.7, repetition_penalty=1.2):
    model.eval()
    encoded_prompt = tokenizer.encode(prompt, return_tensors='pt').to(device)
    input_ids = encoded_prompt.repeat(num_return_sequences, 1)

    generated = [[] for _ in range(num_return_sequences)]

    for _ in range(max_length - encoded_prompt.size(1)):
        with torch.no_grad():
            logits, loss, hidden_states = model(input_ids)
            logits = logits[:, -1, :] / temperature
            
            for i in range(num_return_sequences):
                for token in set(generated[i]):
                    logits[i, token] /= repetition_penalty
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            for i in range(num_return_sequences):
                generated[i].append(next_token[i].item())
            
            input_ids = torch.cat([input_ids, next_token], dim=-1)

    return [tokenizer.decode(input_ids[i], skip_special_tokens=True) for i in range(num_return_sequences)]


prompt = "Ooh, that movie was "
generated_texts = generate_text(model, tokenizer, prompt)
for text in generated_texts:
    print(text)


model = AutoModelForCausalLMWithValueHead(model, config, device).to(device)

input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

lm_logits, loss, value = model(input_ids)
print("head_value", value.shape)
print(value)