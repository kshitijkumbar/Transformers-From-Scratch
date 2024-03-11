import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout_p: float) -> None:
        super().__init__()

        self.nn_seq = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.Dropout(dropout_p),
            nn.Linear(d_ff, d_model),
        )
    
    def forward(self, x):

        return self.nn_seq(x)