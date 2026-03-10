import torch
from torch import nn
import torch.nn.functional as F


class Bigram(nn.Module):
    def __init__(self, vocab_size: int):
        self.vocab_table = nn.Embedding(vocab_size, vocab_size) # 输出只关心一个token，而不是多个token
        
    def _init_weights(self):
        nn.init.normal_(self.vocab_table, mean=0, std=0.02)
    
    def forward(self, x:torch.tensor):  # x[B, T]
        return self.vocab_table(x)    # logits: [B, T, vocab_size]
    
    def generate(self, idx:torch.tensor,    # [B, T]
                 max_new_tokens):      # the length of tokens next to generate
        """
        idx: array of indices in the 'current context'
        """
        for _ in range(max_new_tokens):
            logits = self(idx)  # get the predictions   [B, T, vocab_size]
            # get the last time step predictions [B, T]
            logits = logits[:, -1, :]   # Bigram只关心前一个词 -> 每次只用最后一个位置做预测
            probs = F.softmax(logits)   # compute probabilities
            idx_next = torch.multinomial(probs, num_samples=1)  # [B, 1]
            idx = torch.cat([idx, idx_next], dim=1) # [B, 1]
        
        return idx
    

