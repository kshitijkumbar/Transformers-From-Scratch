import torch
import torch.nn as nn

from MultiHeadAttn import MultiHeadAttention
from Feedforward import FeedForward
from ResidualConnection import ResidualConnection

class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttention, feedforward_block: FeedForward, dropout_p: float):
        super().__init__()

        self.self_attention_block = self_attention_block
        self.feedforward_block = feedforward_block
        self.dropout = nn.Dropout(dropout_p)
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout_p) for _ in range(2)])
    
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feedforward_block)

class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        
        return x