import torch
from torch import nn
from torch.autograd import grad
import numpy as np

effective_dim_start = 3
effective_dim_end = 9

class U_FUNC(nn.Module):
    """Control function using GRU-based neural networks."""

    def __init__(self, model_u_w1, model_u_w2, num_dim_x, num_dim_control):
        super(U_FUNC, self).__init__()
        self.model_u_w1 = model_u_w1
        self.model_u_w2 = model_u_w2
        self.num_dim_x = num_dim_x
        self.num_dim_control = num_dim_control

    def forward(self, x, xe, uref):
        # x: B x n x 1
        # u: B x m x 1
        bs = x.shape[0]

        w1 = self.model_u_w1(torch.cat([x[:,effective_dim_start:effective_dim_end,:],(x-xe)[:,effective_dim_start:effective_dim_end,:]],dim=1).squeeze(-1)).reshape(bs, -1, self.num_dim_x)
        w2 = self.model_u_w2(torch.cat([x[:,effective_dim_start:effective_dim_end,:],(x-xe)[:,effective_dim_start:effective_dim_end,:]],dim=1).squeeze(-1)).reshape(bs, self.num_dim_control, -1)
        u = w2.matmul(torch.tanh(w1.matmul(xe))) + uref

        return u

class GRUFeatureExtractor(nn.Module):
    """GRU-based feature extractor for sequential data processing."""
    
    def __init__(self, input_size, output_size, hidden_size=128, num_layers=2):
        super(GRUFeatureExtractor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input projection to create sequence dimension
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # GRU layers with CuDNN disabled to support double backwards
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Store original setting
        self.original_cudnn_enabled = torch.backends.cudnn.enabled
        
        # Multi-head attention for better sequence aggregation
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # *2 for bidirectional
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Final output projection with residual connection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(128, output_size)
        )
    
    def forward(self, x):
        bs = x.shape[0]
        
        # Create artificial sequence by expanding and varying the input
        seq_len = 10  # Create a sequence of length 10
        x_proj = self.input_projection(x)  # bs x hidden_size
        
        # Create sequence with different transformations and noise
        sequence = []
        for i in range(seq_len):
            # Add structured variations to create meaningful sequence
            phase = 2 * np.pi * i / seq_len
            cos_encoding = torch.cos(torch.tensor(phase, dtype=x.dtype, device=x.device))
            sin_encoding = torch.sin(torch.tensor(phase, dtype=x.dtype, device=x.device))
            
            # Apply rotation-like transformation
            x_var = x_proj * (1 + 0.1 * cos_encoding) + 0.05 * sin_encoding * torch.roll(x_proj, shifts=1, dims=-1)
            sequence.append(x_var)
        
        x_seq = torch.stack(sequence, dim=1)  # bs x seq_len x hidden_size
        
        # Apply GRU with CuDNN disabled to support double backwards
        with torch.backends.cudnn.flags(enabled=False):
            gru_out, h_n = self.gru(x_seq)  # bs x seq_len x (hidden_size*2)
        
        # Apply multi-head self-attention
        attn_out, _ = self.multihead_attention(gru_out, gru_out, gru_out)
        
        # Combine GRU output and attention output
        combined = gru_out + attn_out  # Residual connection
        
        # Global average pooling across sequence dimension
        aggregated = torch.mean(combined, dim=1)  # bs x (hidden_size*2)
        
        # Final projection
        output = self.output_projection(aggregated)
        
        return output

def get_model(num_dim_x, num_dim_control, w_lb, use_cuda = False):
    # GRU-based models for W computation
    model_Wbot = GRUFeatureExtractor(
        effective_dim_end-effective_dim_start-num_dim_control, 
        (num_dim_x-num_dim_control) ** 2,
        hidden_size=64,
        num_layers=2
    )

    dim = effective_dim_end - effective_dim_start
    model_W = GRUFeatureExtractor(
        dim, 
        num_dim_x * num_dim_x,
        hidden_size=128,
        num_layers=2
    )

    # GRU-based models for control computation
    c = 3 * num_dim_x
    model_u_w1 = GRUFeatureExtractor(
        2*dim, 
        c*num_dim_x,
        hidden_size=128,
        num_layers=2
    )
    model_u_w2 = GRUFeatureExtractor(
        2*dim, 
        num_dim_control*c,
        hidden_size=128,
        num_layers=2
    )

    if use_cuda:
        model_W = model_W.cuda()
        model_Wbot = model_Wbot.cuda()
        model_u_w1 = model_u_w1.cuda()
        model_u_w2 = model_u_w2.cuda()

    def W_func(x):
        bs = x.shape[0]
        x = x.squeeze(-1)

        W = model_W(x[:, effective_dim_start:effective_dim_end]).view(bs, num_dim_x, num_dim_x)
        Wbot = model_Wbot(x[:, effective_dim_start:effective_dim_end-num_dim_control]).view(bs, num_dim_x-num_dim_control, num_dim_x-num_dim_control)
        W[:, 0:num_dim_x-num_dim_control, 0:num_dim_x-num_dim_control] = Wbot
        W[:, num_dim_x-num_dim_control::, 0:num_dim_x-num_dim_control] = 0

        W = W.transpose(1,2).matmul(W)
        W = W + w_lb * torch.eye(num_dim_x).view(1, num_dim_x, num_dim_x).type(x.type())
        return W

    u_func = U_FUNC(model_u_w1, model_u_w2, num_dim_x, num_dim_control)

    return model_W, model_Wbot, model_u_w1, model_u_w2, W_func, u_func