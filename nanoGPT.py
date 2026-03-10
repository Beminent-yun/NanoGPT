import torch
from torch import nn
import torch.nn.functional as F
from attention import MultiHead_Attention


class FeedForward(nn.Module):
    """
    A simple linear layer followed by a non-linearity
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4*embed_dim),
            nn.ReLU(),
            nn.Linear(4*embed_dim, embed_dim)
        )
    
    def forward(self, x):
        out = self.net(x)
        # print(f"ffn output shape: {out.shape}")
        return out
    

class Block(nn.Module):
    """
    Transformer Block: communication followed by computation
    """
    
    def __init__(self, embed_dim, n_head):
        # embed_dim: embedding dimension, n_head: number of heads
        
        super().__init__()
        self.self_attention = MultiHead_Attention(embed_dim, embed_dim, n_head)
        self.ffn = FeedForward(embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        x = x + self.self_attention(self.ln1(x))    # residual connection + layernorm(pre-ln)
        x = x + self.ffn(self.ln2(x))
        return x
    


class NanoGPT(nn.Module):
    def __init__(self, vocab_size: int, block_size:int, embed_dim:int, n_head:int):
        super().__init__()
        
        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim) # 输出只关心一个token，而不是多个token
        self.position_embedding_table = nn.Embedding(block_size, embed_dim)
        self.block = Block(embed_dim, n_head)
        self.ln = nn.LayerNorm(embed_dim)   # final layer norm
        self.proj_head = nn.Linear(embed_dim, vocab_size)
        
        self.block_size = block_size
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, x:torch.tensor):
        # x: [B, T] -> input tokens
        B, T = x.shape
        
        token_emb = self.token_embedding_table(x)   # [B, T, embed_dim]
        # Use self.token_embedding_table.weight.device to ensure pos_emb is on the right device
        device = self.token_embedding_table.weight.device
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))   # [B, T, embed_dim]
        x = token_emb + pos_emb # [B, T, embed_dim]
        x = self.block(x)   # [B, T, embed_dim]
        x = self.ln(x)  # [B, T, embed_dim]
        logits = self.proj_head(x)
        
        return logits
    
    def generate(self, idx: torch.tensor, max_new_tokens):
        # idx is [B, T] array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]    # [B, block_size]
            # get the predictions
            logits = self(idx_cond) # [B, block_size, embed_dim]
            # focus only on the last time step
            logits = logits[:, -1, :]   # [B, embed_dim]
            probs = F.softmax(logits, dim=-1)   # [B, embed_dim]
            next_idx = torch.multinomial(probs, num_samples=1)  # [B, 1]
            # append sampled index to the running sequence
            idx = torch.cat([idx, next_idx], dim=-1)    # [B, T+1]
        
        return idx
    

        
        
        
    
    

