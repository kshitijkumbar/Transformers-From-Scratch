import torch
import torch.nn as nn
import math
class InputEmbedding(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, d_model)
    
    def forward(self, x):
        # Returns vector of shape(d_model)
        return self.embedding(x) * math.sqrt(self.d_model)