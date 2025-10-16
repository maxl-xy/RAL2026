import torch
from torch import nn
from torch.autograd import grad
import numpy as np
import math

effective_dim_start = 3
effective_dim_end = 9

class U_FUNC(nn.Module):
    """Control function using Transformer-based neural networks."""

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

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer input."""
    
    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerFeatureExtractor(nn.Module):
    """Transformer-based feature extractor for complex pattern recognition."""
    
    def __init__(self, input_size, output_size, d_model=128, nhead=8, num_layers=4, dim_feedforward=512):
        super(TransformerFeatureExtractor, self).__init__()
        self.input_size = input_size
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Class token for global representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Output projection with layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim_feedforward, output_size)
        )
    
    def forward(self, x):
        bs = x.shape[0]
        
        # Create sequence by applying different mathematical transformations
        seq_len = 12  # Create a sequence of length 12
        x_proj = self.input_projection(x)  # bs x d_model
        
        # Create meaningful sequence variations using mathematical transforms
        sequence = []
        
        # Add class token
        cls_tokens = self.cls_token.expand(bs, -1, -1)  # bs x 1 x d_model
        sequence.append(cls_tokens.squeeze(1))
        
        for i in range(seq_len - 1):  # -1 because we added cls token
            # Different mathematical transformations
            if i % 4 == 0:
                # Original projection
                x_var = x_proj
            elif i % 4 == 1:
                # Rotate features
                x_var = torch.roll(x_proj, shifts=self.d_model//4, dims=-1)
            elif i % 4 == 2:
                # Scale with sinusoidal pattern
                scale = torch.sin(torch.tensor(i * np.pi / seq_len, dtype=x.dtype, device=x.device))
                x_var = x_proj * (1 + 0.2 * scale)
            else:
                # Permute and mix features
                perm = torch.randperm(self.d_model, device=x.device)[:self.d_model//2]
                x_var = x_proj.clone()
                x_var[:, perm] = x_var[:, perm] * 0.8 + x_var[:, torch.roll(perm, 1)] * 0.2
            
            sequence.append(x_var)
        
        x_seq = torch.stack(sequence, dim=1)  # bs x seq_len x d_model
        
        # Add positional encoding
        x_seq = x_seq.transpose(0, 1)  # seq_len x bs x d_model
        x_seq = self.pos_encoder(x_seq)
        x_seq = x_seq.transpose(0, 1)  # bs x seq_len x d_model
        
        # Apply transformer
        transformer_out = self.transformer_encoder(x_seq)  # bs x seq_len x d_model
        
        # Use class token representation (first token)
        cls_output = transformer_out[:, 0, :]  # bs x d_model
        
        # Apply layer normalization and output projection
        cls_output = self.layer_norm(cls_output)
        output = self.output_projection(cls_output)
        
        return output

def get_model(num_dim_x, num_dim_control, w_lb, use_cuda = False):
    # Transformer-based models for W computation
    model_Wbot = TransformerFeatureExtractor(
        effective_dim_end-effective_dim_start-num_dim_control, 
        (num_dim_x-num_dim_control) ** 2,
        d_model=64,
        nhead=4,
        num_layers=3,
        dim_feedforward=256
    )

    dim = effective_dim_end - effective_dim_start
    model_W = TransformerFeatureExtractor(
        dim, 
        num_dim_x * num_dim_x,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=512
    )

    # Transformer-based models for control computation
    c = 3 * num_dim_x
    model_u_w1 = TransformerFeatureExtractor(
        2*dim, 
        c*num_dim_x,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=512
    )
    model_u_w2 = TransformerFeatureExtractor(
        2*dim, 
        num_dim_control*c,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=512
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