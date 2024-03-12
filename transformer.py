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
    

class DecoderBlock(nn.Module):

    def __init__(self, self_attn_block: MultiHeadAttention, cross_attn_block: MultiHeadAttention, feedforward_block: FeedForward, dropout_p: float) -> None:
        super().__init__()

        self.self_attention_block = self_attn_block
        self.cross_attention_block = cross_attn_block
        self.feedforward_block = feedforward_block
        self.dropout = nn.Dropout(dropout_p)
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout_p) for _ in range(3)])
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feedforward_block)

        return x

class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        return x
    