import torch
from torch import nn
import numpy as np

# States used in the model
effective_dim_start = 3
effective_dim_end = 18

# Control constraints
f_bound = 5.0
saturation_factor = 0.3

class LocalAttention(nn.Module):
    """Local self-attention module inspired by FieldFormer"""
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super(LocalAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_per_head = dim // num_heads
        self.scale = self.dim_per_head ** -0.5
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        bs, n, d = x.shape
        q = self.query(x).view(bs, n, self.num_heads, self.dim_per_head).transpose(1, 2)
        k = self.key(x).view(bs, n, self.num_heads, self.dim_per_head).transpose(1, 2)
        v = self.value(x).view(bs, n, self.num_heads, self.dim_per_head).transpose(1, 2)
        
        # Local attention: scale by state differences (e.g., drone distances)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bs, n, d)
        return self.out(out)

class FieldFormerBlock(nn.Module):
    """FieldFormer block with local attention and feedforward"""
    def __init__(self, dim, num_heads=4, ff_dim=128, dropout=0.1):
        super(FieldFormerBlock, self).__init__()
        self.attn = LocalAttention(dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ff(x))
        return x

class FieldFormer(nn.Module):
    """Main FieldFormer model for CCM metric and controller"""
    def __init__(self, input_dim, output_dim, num_layers=2, dim=64, num_heads=4, ff_dim=128, dropout=0.1):
        super(FieldFormer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embedding = nn.Linear(input_dim, dim)
        self.blocks = nn.ModuleList([
            FieldFormerBlock(dim, num_heads, ff_dim, dropout) for _ in range(num_layers)
        ])
        self.output = nn.Linear(dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        return self.output(x)

class U_FUNC(nn.Module):
    """Controller using FieldFormer"""
    def __init__(self, model_u, num_dim_x, num_dim_control):
        super(U_FUNC, self).__init__()
        self.model_u = model_u
        self.num_dim_x = num_dim_x
        self.num_dim_control = num_dim_control

    def forward(self, x, xe, uref):
        bs = x.shape[0]
        # Input: concatenate effective states and errors
        input_x = torch.cat([
            x[:, effective_dim_start:effective_dim_end, :],
            (x - xe)[:, effective_dim_start:effective_dim_end, :]
        ], dim=1).squeeze(-1)  # (bs, 2*(end-start), 1)
        
        # Reshape for Transformer: treat each dimension as a token
        input_x = input_x.view(bs, -1, 1)  # (bs, 2*dim, 1)
        u_raw = self.model_u(input_x).view(bs, self.num_dim_control, 1)
        
        # Apply control bounds with soft saturation
        bounds = torch.tensor([f_bound] * self.num_dim_control, dtype=x.dtype, device=x.device).view(1, -1, 1).expand(bs, -1, -1)
        u = torch.tanh(u_raw * saturation_factor) * bounds + uref
        return u

def get_model(num_dim_x, num_dim_control, w_lb, use_cuda=False):
    dim = effective_dim_end - effective_dim_start  # 15
    
    # FieldFormer for Wbot (null space metric)
    model_Wbot = FieldFormer(
        input_dim=dim - num_dim_control,  # 6
        output_dim=(num_dim_x - num_dim_control) ** 2,  # (18-9)^2 = 81
        num_layers=2,
        dim=64,
        num_heads=4,
        ff_dim=128,
        dropout=0.1
    )
    
    # FieldFormer for W (full metric)
    model_W = FieldFormer(
        input_dim=dim,  # 15
        output_dim=num_dim_x * num_dim_x,  # 18*18 = 324
        num_layers=2,
        dim=64,
        num_heads=4,
        ff_dim=128,
        dropout=0.1
    )
    
    # FieldFormer for controller (u)
    model_u = FieldFormer(
        input_dim=2 * dim,  # 30 (x and x-xe)
        output_dim=num_dim_control,  # 9
        num_layers=2,
        dim=64,
        num_heads=4,
        ff_dim=128,
        dropout=0.1
    )
    
    if use_cuda:
        model_W = model_W.cuda()
        model_Wbot = model_Wbot.cuda()
        model_u = model_u.cuda()

    def W_func(x):
        bs = x.shape[0]
        x = x.squeeze(-1)
        W = model_W(x[:, effective_dim_start:effective_dim_end]).view(bs, num_dim_x, num_dim_x)
        W = W.transpose(1, 2).matmul(W) + w_lb * torch.eye(num_dim_x).view(1, num_dim_x, num_dim_x).type(x.type())
        return W

    u_func = U_FUNC(model_u, num_dim_x, num_dim_control)
    return model_W, model_Wbot, None, None, W_func, u_func