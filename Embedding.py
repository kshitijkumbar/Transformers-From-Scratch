import torch
import torch.nn as nn
import math
class InputEmbedding(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        print(type(self.d_model), type(self.vocab_size))
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
    
    def forward(self, x):
        # Returns vector of shape(d_model)
        print()
        embedded_input = self.embedding(x)
        scaled_embedding = embedded_input * math.sqrt(self.d_model)
        return scaled_embedding

if __name__ == "__main__":
    d_model = 512
    vocab_size = 10
    seq_len = 12
    input = torch.randint(0, vocab_size-1, (seq_len, vocab_size))

    emb = InputEmbedding(d_model, vocab_size)
    print(emb(input)) 