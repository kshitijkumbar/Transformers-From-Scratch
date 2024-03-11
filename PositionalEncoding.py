import torch 
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout_p: float):
        super().__init__()
        
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout_p)

        # Postional Encoding matrix (seq_len, d_model)
        self.pos_encoding = torch.zeros(self.seq_len, self.d_model)

        # Create position vector
        pos_vec = torch.arange(0, self.seq_len).unsqueeze(1)

        # Using numerically stable version for exp term
        div_term = torch.exp(torch.arange(0,self.d_model, 2))

        # Apply sin/cos to even positions
        self.pos_encoding[:, 0::2] = torch.sin(pos_vec * div_term)
        self.pos_encoding[:, 1::2] = torch.cos(pos_vec * div_term)

        # Expand positional encoding matrix to accomodate batch dim
        self.pos_encoding = self.pos_encoding.unsqueeze(0)

        # Save non parametric val in model
        self.register_buffer('self.pos_encoding', self.pos_encoding)


    def forward(self, x):

        x = x + (self.pos_encoding[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)