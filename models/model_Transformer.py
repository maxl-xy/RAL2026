import torch
from torch import nn
import math

# Correct effective subspace: 3 to 18 → 15 dimensions
effective_dim_start = 3
effective_dim_end = 18

class U_FUNC(nn.Module):
    """Low-rank control correction: u = w2 @ tanh(w1 @ xe) + uref"""
    
    def __init__(self, model_u_w1, model_u_w2, num_dim_x, num_dim_control):
        super().__init__()
        self.model_u_w1 = model_u_w1
        self.model_u_w2 = model_u_w2
        self.num_dim_x = num_dim_x
        self.num_dim_control = num_dim_control

    def forward(self, x, xe, uref):
        bs = x.shape[0]
        
        # Condition on effective subspace [3:18] + error
        cond = torch.cat([
            x[:, effective_dim_start:effective_dim_end, :],
            (x - xe)[:, effective_dim_start:effective_dim_end, :]
        ], dim=1).squeeze(-1)  # B x 30 (15 + 15)

        w1 = self.model_u_w1(cond).reshape(bs, -1, self.num_dim_x)
        w2 = self.model_u_w2(cond).reshape(bs, self.num_dim_control, -1)

        u = w2.matmul(torch.tanh(w1.matmul(xe))) + uref
        return u


class SetTransformerExtractor(nn.Module):
    """
    Permutation-invariant Set Transformer for static state vectors.
    Treats each dimension in the input as a separate token → fully invariant to ordering.
    Uses a learnable class token that attends to all input tokens for global pooling.
    """
    
    def __init__(
        self,
        input_size,           # e.g., 15 for W, 12 for Wbot, 30 for u networks
        output_size,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        num_cls_tokens=1
    ):
        super().__init__()
        self.d_model = d_model
        self.input_size = input_size
        self.num_cls_tokens = num_cls_tokens
        
        # Project each scalar input dimension to d_model
        self.input_projection = nn.Linear(1, d_model)
        
        # Learnable class/query token(s)
        self.cls_token = nn.Parameter(torch.randn(1, num_cls_tokens, d_model))
        
        # Learnable positional embeddings for input tokens (helps if subtle order matters)
        self.pos_emb = nn.Parameter(torch.randn(1, input_size, d_model) * 0.02)
        
        # Transformer encoder with pre-norm for stability
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final head
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_size)
        )

    def forward(self, x):
        """
        x: bs x input_size
        """
        bs = x.shape[0]
        device = x.device
        
        # Treat each dimension as a token: bs x input_size x 1
        x_tokens = x.unsqueeze(-1)  # bs x L x 1
        x_tokens = self.input_projection(x_tokens)  # bs x L x d_model
        
        # Add positional embeddings
        x_tokens = x_tokens + self.pos_emb.expand(bs, -1, -1)
        
        # Prepend class token(s)
        cls_tokens = self.cls_token.expand(bs, -1, -1)  # bs x num_cls x d_model
        seq = torch.cat([cls_tokens, x_tokens], dim=1)  # bs x (num_cls + L) x d_model
        
        # Full self-attention
        seq = self.transformer(seq)
        
        # Use class token(s) as pooled representation
        pooled = seq[:, :self.num_cls_tokens].mean(dim=1)  # bs x d_model
        
        # Final projection
        pooled = self.norm(pooled)
        output = self.head(pooled)
        
        return output


def get_model(num_dim_x=24, num_dim_control=3, w_lb=1e-4, use_cuda=True):
    """
    Returns the Set Transformer-based model with correct 15D effective subspace (3:18).
    """
    dim = effective_dim_end - effective_dim_start  # 15
    
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    
    # Full W matrix (input: 15 dims)
    model_W = SetTransformerExtractor(
        input_size=dim,
        output_size=num_dim_x * num_dim_x,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1
    ).to(device)
    
    # Bottom block (input: 15 - 3 = 12 dims)
    model_Wbot = SetTransformerExtractor(
        input_size=dim - num_dim_control,
        output_size=(num_dim_x - num_dim_control) ** 2,
        d_model=96,
        nhead=6,
        num_layers=3,
        dim_feedforward=384,
        dropout=0.1
    ).to(device)
    
    # Control networks (input: state + error → 30 dims)
    c = 3 * num_dim_x
    model_u_w1 = SetTransformerExtractor(
        input_size=2 * dim,
        output_size=c * num_dim_x,
        d_model=160,
        nhead=8,
        num_layers=5,
        dim_feedforward=640,
        dropout=0.1
    ).to(device)
    
    model_u_w2 = SetTransformerExtractor(
        input_size=2 * dim,
        output_size=num_dim_control * c,
        d_model=160,
        nhead=8,
        num_layers=5,
        dim_feedforward=640,
        dropout=0.1
    ).to(device)

    def W_func(x):
        bs = x.shape[0]
        x = x.squeeze(-1)

        W_full = model_W(x[:, effective_dim_start:effective_dim_end]) \
                 .view(bs, num_dim_x, num_dim_x)
        
        W_bot = model_Wbot(x[:, effective_dim_start:effective_dim_end - num_dim_control]) \
                .view(bs, num_dim_x - num_dim_control, num_dim_x - num_dim_control)
        
        W = W_full.clone()
        W[:, :num_dim_x - num_dim_control, :num_dim_x - num_dim_control] = W_bot
        W[:, num_dim_x - num_dim_control:, :num_dim_x - num_dim_control] = 0.0
        
        # Ensure positive definite
        W = W.transpose(1, 2) @ W
        W = W + w_lb * torch.eye(num_dim_x, device=device).unsqueeze(0)
        
        return W

    u_func = U_FUNC(model_u_w1, model_u_w2, num_dim_x, num_dim_control).to(device)

    return model_W, model_Wbot, model_u_w1, model_u_w2, W_func, u_func