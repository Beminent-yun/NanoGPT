import torch
from torch import nn
import torch.nn.functional as F

class Self_Attention(nn.Module):
    """
    Attention(Q, K, V) = softmax(Q@K.T/sqrt(d_k))@V
    """
    
    def __init__(self, input_dim:int, head_size:int):
        super().__init__()
        
        self.Wq = nn.Linear(input_dim, head_size, bias=False)
        self.Wk = nn.Linear(input_dim, head_size, bias=False)
        self.Wv = nn.Linear(input_dim, head_size, bias=False)
        
    def forward(self, x:torch.tensor):  
        """
        x: [B, T, d_model]
        d_model: model的最大输入
        """
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        
        scale = K.shape[-1] ** -0.5
        # softmax normalization
        scores = F.softmax(Q @ K.transpose(-2, -1) * scale, dim=-1) # [B, T, T]
        
        out = scores @ V    # [B, T, T] @ [B, T, C] -> [B, T, C]
        
        return out
    

class Causal_Attention(nn.Module):
    """
    (Single Head) Attention(Q, K, V) = softmax(Q@K.T/sqrt(d_k) + M)@V
    
    M: mask matrix:
            [[1, 0, ... 0],
            [1, 1, 0 ... 0],
            ...
        第t步[1, 1, 1, ... 0... 0], <- t个1， T-t个0
            ...
            [1, 1, 1, ... 1, 1, 1]]
    when M == 0 -> fill in '-inf' -> softmax(-inf) = 0 -> no attention

    x: [B, T, d_model]
    block_size/T: the maximum context length for predictions
    
    >>> torch.tril(torch.ones(3,3))
        tensor([[1., 0., 0.],
                [1., 1., 0.],
                [1., 1., 1.]])
    """
    def __init__(self, input_dim:int, head_size:int, max_seq_len=1024):
        super().__init__()
        
        self.Wq = nn.Linear(input_dim, head_size, bias=False)
        self.Wk = nn.Linear(input_dim, head_size, bias=False)
        self.Wv = nn.Linear(input_dim, head_size, bias=False)
        
    
        self.register_buffer(
            'mask',
            torch.tril(torch.ones(max_seq_len, max_seq_len))
        )
    
    def forward(self, x: torch.tensor):
        B, T, C = x.shape
        
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        
        mask = self.mask[:T, :T]
        
        scale = K.shape[-1]**(-0.5)
        scores = Q@K.transpose(-2, -1) * scale
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        scores = F.softmax(scores, dim=-1)
        
        out = scores@V
        # print(f"Single Head output shape: {out.shape}")
        
        return out
    


class MultiHead_Attention(nn.Module):
    def __init__(self, input_dim, d_model, num_head):
        super().__init__()
        assert d_model % num_head == 0
        
        head_size = d_model // num_head
        
        # 多个头“并行”(这里实际上是串行 -> 硬件优化 -> 一次矩阵乘法同时算完)
        self.heads = nn.ModuleList([
            Causal_Attention(input_dim, head_size)
            for _ in range(num_head)
        ])
        
        # proj head (将所有头的输出拼接后投影)
        self.proj = nn.Linear(d_model, d_model)
    
    def forward(self, x:torch.tensor):
        # print(f"MHA input shape: {x.shape}")
        out = torch.cat(
            [head(x) for head in self.heads],
        dim=-1)
        # print(f"Concatenated output shape: {out.shape}")
        out = self.proj(out)
        # print(f"attention output shape: {out.shape}")
        return out
    

