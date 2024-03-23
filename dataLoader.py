import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

class GPTDataset(Dataset):
    
    def __init__(self, data, tokenizer, max_len, stride) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []
        
        token_ids = tokenizer.encode(data, allowed_special ={'<|endoftext|>'})
        
        # Get training data
        for i in range(0,len(token_ids) - max_len, stride):
            input_chunk = token_ids[i:i + max_len]
            output_chunk = token_ids[i+1:i + max_len + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(output_chunk))
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    
    

def create_dataloader(data, batch_sz=4, max_len=256, stride=128, shuffle=True, drop_last=True):
    
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(data, tokenizer, max_len, stride)
    dataloader = DataLoader(dataset, batch_size=batch_sz, shuffle=shuffle, drop_last=drop_last)
    
    return dataloader
    
    

if __name__=="__main__":
    
    with open("verdict.txt", "r", encoding="utf-8") as f:
        data = f.read()
    
    vocab_sz = 50257 # GPT2 spec
    output_dim = 256
    block_sz = 1024
    
    token_embedding_layer = nn.Embedding(vocab_sz, output_dim)
    pos_embedding_layer = nn.Embedding(block_sz, output_dim)
    
    max_len = 4
    dataloader = create_dataloader(data, batch_sz=8, max_len=max_len, stride=max_len)
    
    for batch in dataloader:
        x, y = batch
        token_embeds = token_embedding_layer(x)
        pos_embeds = pos_embedding_layer(torch.arange(max_len))
        
        input_embeds = token_embeds + pos_embeds
        
        print(input_embeds.shape)
        
    
