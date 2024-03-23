import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


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
            self.output_ids.append(torch.tensor(output_chunk))
        
        def __len__(self):
            return len(self.input_ids)
        
        def __getitem__(self, idx):
            return self.input_ids[idx], self.output_ids[idx]
        
    

# def create_dataloader(data, batch_sz,4, max_len=256, stride=128, shuffle=True, drop_last=True):
    
#     tokenizer = 