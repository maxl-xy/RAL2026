import torch
from torch import nn
import numpy as np

effective_dim_start = 3
effective_dim_end = 18

class U_FUNC(nn.Module):
    """Control correction function: u = w2 @ tanh(w1 @ xe) + uref"""

    def __init__(self, model_u_w1, model_u_w2, num_dim_x, num_dim_control):
        super(U_FUNC, self).__init__()
        self.model_u_w1 = model_u_w1
        self.model_u_w2 = model_u_w2
        self.num_dim_x = num_dim_x
        self.num_dim_control = num_dim_control

    def forward(self, x, xe, uref):
        # x: B x n x 1
        # xe: B x n x 1
        # uref: B x m x 1
        bs = x.shape[0]

        # Condition on effective subspace of current state and error
        cond = torch.cat([
            x[:, effective_dim_start:effective_dim_end, :],
            (x - xe)[:, effective_dim_start:effective_dim_end, :]
        ], dim=1).squeeze(-1)  # B x 2*dim

        w1 = self.model_u_w1(cond).reshape(bs, -1, self.num_dim_x)          # B x c x n
        w2 = self.model_u_w2(cond).reshape(bs, self.num_dim_control, -1)    # B x m x c

        u = w2.matmul(torch.tanh(w1.matmul(xe))) + uref
        return u


class ResidualBlock(nn.Module):
    """Simple residual block with pre-activation, LayerNorm, and optional gating."""
    
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        
        # Optional adaptive gating
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.norm1(x)
        out = torch.tanh(out)
        out = self.dropout1(self.linear1(out))
        
        out = self.norm2(out)
        out = self.dropout2(self.linear2(out))
        
        gate = self.gate(x)
        return gate * out + (1 - gate) * x


class ResNetFeatureExtractor(nn.Module):
    """Deep ResNet-style feature extractor with consistent dimensions."""
    
    def __init__(self, input_size, output_size, hidden_size=128, num_blocks=6, dropout=0.1, bias=True):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh()
        )
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_size, dropout) for _ in range(num_blocks)
        ])
        
        # Simple channel attention (Squeeze-and-Excitation style)
        reduction = max(8, hidden_size // 16)
        self.se = nn.Sequential(
            nn.Linear(hidden_size, reduction),
            nn.ReLU(),
            nn.Linear(reduction, hidden_size),
            nn.Sigmoid()
        )
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size, bias=bias)  # bias=False for matrix layers like original
        )

    def forward(self, x):
        x = self.input_proj(x)
        
        for block in self.res_blocks:
            x = block(x)
        
        # SE attention
        se_weight = self.se(x.mean(dim=0, keepdim=True))  # global avg over batch
        x = x * se_weight
        
        return self.output_proj(x)


def get_model(num_dim_x, num_dim_control, w_lb, use_cuda=False):
    dim = effective_dim_end - effective_dim_start  # 15
    
    # ResNet for full W matrix (no bias in final layer)
    model_W = ResNetFeatureExtractor(
        input_size=dim,
        output_size=num_dim_x * num_dim_x,
        hidden_size=128,
        num_blocks=8,
        dropout=0.1,
        bias=False
    )
    
    # ResNet for bottom block (smaller input, no bias in final layer)
    model_Wbot = ResNetFeatureExtractor(
        input_size=dim - num_dim_control,
        output_size=(num_dim_x - num_dim_control) ** 2,
        hidden_size=64,
        num_blocks=6,
        dropout=0.1,
        bias=False
    )
    
    # Control correction networks
    c = 3 * num_dim_x  # low-rank factor (same as original)
    model_u_w1 = ResNetFeatureExtractor(
        input_size=2 * dim,
        output_size=c * num_dim_x,
        hidden_size=128,
        num_blocks=8,
        dropout=0.1,
        bias=True
    )
    model_u_w2 = ResNetFeatureExtractor(
        input_size=2 * dim,
        output_size=num_dim_control * c,
        hidden_size=128,
        num_blocks=8,
        dropout=0.1,
        bias=True
    )
    
    if use_cuda:
        model_W = model_W.cuda()
        model_Wbot = model_Wbot.cuda()
        model_u_w1 = model_u_w1.cuda()
        model_u_w2 = model_u_w2.cuda()

    def W_func(x):
        """
        Compute structured positive definite matrix W(x).
        """
        bs = x.shape[0]
        x = x.squeeze(-1)  # B x n

        # Full matrix prediction
        W_full = model_W(x[:, effective_dim_start:effective_dim_end]) \
                 .view(bs, num_dim_x, num_dim_x)
        
        # Bottom block prediction
        W_bot = model_Wbot(x[:, effective_dim_start:effective_dim_end - num_dim_control]) \
                .view(bs, num_dim_x - num_dim_control, num_dim_x - num_dim_control)
        
        # Assemble structured W
        W = W_full.clone()
        W[:, :num_dim_x - num_dim_control, :num_dim_x - num_dim_control] = W_bot
        W[:, num_dim_x - num_dim_control:, :num_dim_x - num_dim_control] = 0  # zero top-right block
        
        # Ensure positive definiteness: W <- W^T W + w_lb I
        W = W.transpose(1, 2).matmul(W)
        W = W + w_lb * torch.eye(num_dim_x, device=W.device, dtype=W.dtype) \
                     .unsqueeze(0).expand(bs, -1, -1)
        
        return W

    u_func = U_FUNC(model_u_w1, model_u_w2, num_dim_x, num_dim_control)

    return model_W, model_Wbot, model_u_w1, model_u_w2, W_func, u_func