import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout_p: float) -> None:

        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        assert self.d_model % self.n_heads == 0, "Model dimension not divisible by 0, can't use multi head attention"

        # Define split dimension values
        self.d_k = self.d_model // self.n_heads

        # Define Query, Key and Value weights
        self.w_q = nn.Linear(self.d_model, self.d_model)
        self.w_k = nn.Linear(self.d_model, self.d_model)
        self.w_v = nn.Linear(self.d_model, self.d_model)

        # Define Output Weights
        self.w_o = nn.Linear(self.d_model, self.d_model)

        self.dropout = nn.Dropout(dropout_p)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # (batch, h, seq_len, d_k) -> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill_(mask==0, -1e9)
        
        attention_scores = attention_scores.softmax(dim = -1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores


    def forward(self, q, k, v, mask):
        
        query = self.w_q(q) # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) -> (batch, seq_len, d_model)

        # Split above vars for multi head attn and swap seq and n_heads dims
        query = query.view(query.shape[0], query.shape[1], self.n_heads, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.n_heads, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.n_heads, self.d_k).transpose(1,2)


        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # (batch, h, seq_len, d_k) -> (batch, seq_len, h, d_k) -> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.n_heads * self.d_k)

        # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        return self.w_o(x)