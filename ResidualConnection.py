import torch
import torch.nn as nn

class ResidualConnection(nn.Module):

    def __init__(self, dropout_p: float) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout_p)
        self.norm = nn.LayerNorm(eps=1e-9)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))