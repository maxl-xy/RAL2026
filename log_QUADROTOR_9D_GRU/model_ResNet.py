import torch
from torch import nn
from torch.autograd import grad
import numpy as np

effective_dim_start = 3
effective_dim_end = 9

class U_FUNC(nn.Module):
    """Control function using ResNet-based neural networks."""

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

class ResidualBlock(nn.Module):
    """Residual block with skip connections."""
    
    def __init__(self, input_size, hidden_size=None, dropout=0.1):
        super(ResidualBlock, self).__init__()
        
        if hidden_size is None:
            hidden_size = input_size
        
        # Main path
        self.main_path = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, input_size),
            nn.LayerNorm(input_size),
            nn.Dropout(dropout)
        )
        
        # Skip connection projection (if needed)
        self.skip_projection = nn.Identity() if input_size == input_size else nn.Linear(input_size, input_size)
        
        # Gating mechanism for adaptive skip connections
        self.gate = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        residual = self.skip_projection(x)
        main_out = self.main_path(x)
        
        # Adaptive gating
        gate_weights = self.gate(x)
        
        # Gated residual connection
        output = gate_weights * main_out + (1 - gate_weights) * residual
        return output

class ResNetFeatureExtractor(nn.Module):
    """ResNet-based feature extractor with deep residual connections."""
    
    def __init__(self, input_size, output_size, hidden_size=128, num_blocks=6):
        super(ResNetFeatureExtractor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh()
        )
        
        # Stack of residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_size, hidden_size * 2, dropout=0.1)
            for _ in range(num_blocks)
        ])
        
        # Squeeze-and-Excitation block for channel attention
        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, hidden_size),
            nn.Sigmoid()
        )
        
        # Multi-scale feature fusion
        self.multi_scale_conv = nn.ModuleList([
            nn.Conv1d(1, 1, kernel_size=k, padding=k//2)
            for k in [3, 5, 7]
        ])
        
        # Output projection with bottleneck
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        bs = x.shape[0]
        
        # Input projection
        x = self.input_projection(x)  # bs x hidden_size
        
        # Store intermediate features for multi-scale fusion
        features = []
        
        # Apply residual blocks
        for i, block in enumerate(self.residual_blocks):
            x = block(x)
            
            # Store features at different depths
            if i % 2 == 1:  # Every other block
                features.append(x)
        
        # Multi-scale feature fusion using convolutions
        x_expanded = x.unsqueeze(1)  # bs x 1 x hidden_size
        multi_scale_features = []
        
        for conv in self.multi_scale_conv:
            ms_feat = conv(x_expanded)  # bs x 1 x hidden_size
            multi_scale_features.append(ms_feat.squeeze(1))
        
        # Combine multi-scale features
        x_ms = torch.stack(multi_scale_features, dim=0).mean(dim=0)  # bs x hidden_size
        
        # Combine with residual connection
        x = x + 0.3 * x_ms
        
        # Apply Squeeze-and-Excitation attention
        se_weights = self.se_block(x.unsqueeze(-1))  # bs x hidden_size
        x = x * se_weights
        
        # Aggregate features from different depths
        if features:
            aggregated_features = torch.stack(features, dim=0).mean(dim=0)  # bs x hidden_size
            x = x + 0.2 * aggregated_features
        
        # Output projection
        output = self.output_projection(x)
        
        return output

def get_model(num_dim_x, num_dim_control, w_lb, use_cuda = False):
    # ResNet-based models for W computation
    model_Wbot = ResNetFeatureExtractor(
        effective_dim_end-effective_dim_start-num_dim_control, 
        (num_dim_x-num_dim_control) ** 2,
        hidden_size=64,
        num_blocks=4
    )

    dim = effective_dim_end - effective_dim_start
    model_W = ResNetFeatureExtractor(
        dim, 
        num_dim_x * num_dim_x,
        hidden_size=128,
        num_blocks=6
    )

    # ResNet-based models for control computation
    c = 3 * num_dim_x
    model_u_w1 = ResNetFeatureExtractor(
        2*dim, 
        c*num_dim_x,
        hidden_size=128,
        num_blocks=6
    )
    model_u_w2 = ResNetFeatureExtractor(
        2*dim, 
        num_dim_control*c,
        hidden_size=128,
        num_blocks=6
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