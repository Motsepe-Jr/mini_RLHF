import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

from huggingface_hub import PyTorchModelHubMixin

class ValueHead(nn.Module, PyTorchModelHubMixin):
    def __init__(self, hidden_size, dropout_prob=0.1, device='cuda'):
        super().__init__()
        self.device = device
        self.dropout = nn.Dropout(dropout_prob).to(self.device)
        self.summary = nn.Linear(hidden_size, 1).to(self.device)

    def forward(self, hidden_states):
        output = self.dropout(hidden_states)
        output = self.summary(output)
        return output
    
class AutoModelForCausalLMWithValueHead(nn.Module, PyTorchModelHubMixin): 
    def __init__(self, pretrained_model, config, device):
        super().__init__() 
        self.config = config
        self.pretrained_model = pretrained_model

        self.v_head = ValueHead(self.config.n_embd, device=device)

        self._init_weights()

    def forward(
        self,
        input # B, T
    ):
    
        lm_logits, loss, last_hidden_state = self.pretrained_model(input) # last_hidden_state (B, T, H)

        value = self.v_head(last_hidden_state).squeeze(-1)

        return (lm_logits, loss, value)
    
    def _init_weights(self):
        initializer_range = 0.2  
        init_strategy = "normal" 
        if init_strategy == "normal":
            self.v_head.summary.weight.data.normal_(mean=0.0, std=initializer_range)
            self.v_head.summary.bias.data.zero_()