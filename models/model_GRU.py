import torch
from torch import nn

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
        
        # Condition on effective subspace [3:18] + error → 30 dims
        cond = torch.cat([
            x[:, effective_dim_start:effective_dim_end, :],
            (x - xe)[:, effective_dim_start:effective_dim_end, :]
        ], dim=1).squeeze(-1)  # B x 30

        w1 = self.model_u_w1(cond).reshape(bs, -1, self.num_dim_x)
        w2 = self.model_u_w2(cond).reshape(bs, self.num_dim_control, -1)

        u = w2.matmul(torch.tanh(w1.matmul(xe))) + uref
        return u


class GRUExtractor(nn.Module):
    """
    GRU feature extractor with chunking.
    Splits the input vector into fixed-size chunks to form a short sequence.
    Example: 15D → 5 chunks of 3 dims; 30D → 10 chunks of 3 dims.
    """
    
    def __init__(
        self,
        input_size,
        output_size,
        chunk_size=3,
        hidden_size=128,
        num_layers=3,
        dropout=0.1,
        bidirectional=True
    ):
        super().__init__()
        assert input_size % chunk_size == 0, f"input_size ({input_size}) must be divisible by chunk_size ({chunk_size})"
        
        self.input_size = input_size
        self.chunk_size = chunk_size
        self.seq_len = input_size // chunk_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.dir_factor = 2 if bidirectional else 1
        
        # Project each chunk
        self.chunk_proj = nn.Linear(chunk_size, hidden_size)
        
        # GRU
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        
        # Normalization and dropout
        self.norm = nn.LayerNorm(hidden_size * self.dir_factor)
        self.dropout = nn.Dropout(dropout)
        
        # Self-attention over sequence
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * self.dir_factor,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Final head
        self.head = nn.Sequential(
            nn.Linear(hidden_size * self.dir_factor, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
        """
        x: bs x input_size
        """
        bs = x.shape[0]
        
        # Reshape into chunks: bs x seq_len x chunk_size
        x_chunked = x.view(bs, self.seq_len, self.chunk_size)
        
        # Project chunks
        x_seq = self.chunk_proj(x_chunked)  # bs x seq_len x hidden_size
        
        # GRU
        gru_out, _ = self.gru(x_seq)  # bs x seq_len x (hidden * dir)
        
        # Self-attention
        attn_out, _ = self.attention(gru_out, gru_out, gru_out)
        
        # Residual + norm
        seq_out = self.norm(gru_out + attn_out)
        seq_out = self.dropout(seq_out)
        
        # Global mean pooling
        pooled = seq_out.mean(dim=1)  # bs x (hidden * dir)
        
        # Output
        output = self.head(pooled)
        
        return output


def get_model(num_dim_x=24, num_dim_control=3, w_lb=1e-4, use_cuda=True):
    dim = effective_dim_end - effective_dim_start  # 15
    chunk_size = 3  # Works perfectly: 15→5, 12→4, 30→10
    
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    
    model_W = GRUExtractor(
        input_size=dim,
        output_size=num_dim_x * num_dim_x,
        chunk_size=chunk_size,
        hidden_size=128,
        num_layers=3,
        dropout=0.1
    ).to(device)
    
    model_Wbot = GRUExtractor(
        input_size=dim - num_dim_control,  # 12
        output_size=(num_dim_x - num_dim_control) ** 2,
        chunk_size=chunk_size,
        hidden_size=96,
        num_layers=3,
        dropout=0.1
    ).to(device)
    
    c = 3 * num_dim_x
    model_u_w1 = GRUExtractor(
        input_size=2 * dim,  # 30
        output_size=c * num_dim_x,
        chunk_size=chunk_size,
        hidden_size=160,
        num_layers=4,
        dropout=0.1
    ).to(device)
    
    model_u_w2 = GRUExtractor(
        input_size=2 * dim,
        output_size=num_dim_control * c,
        chunk_size=chunk_size,
        hidden_size=160,
        num_layers=4,
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
        
        W = W.transpose(1, 2) @ W
        W = W + w_lb * torch.eye(num_dim_x, device=device).unsqueeze(0)
        
        return W

    u_func = U_FUNC(model_u_w1, model_u_w2, num_dim_x, num_dim_control).to(device)

    return model_W, model_Wbot, model_u_w1, model_u_w2, W_func, u_func