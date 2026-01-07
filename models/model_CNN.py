import torch
from torch import nn

effective_dim_start = 3
effective_dim_end = 18

class U_FUNC(nn.Module):
    """Control correction: u = w2 @ tanh(w1 @ xe) + uref"""
    
    def __init__(self, model_u_w1, model_u_w2, num_dim_x, num_dim_control):
        super(U_FUNC, self).__init__()
        self.model_u_w1 = model_u_w1
        self.model_u_w2 = model_u_w2
        self.num_dim_x = num_dim_x
        self.num_dim_control = num_dim_control

    def forward(self, x, xe, uref):
        bs = x.shape[0]
        
        # Concatenate current effective state and error
        cond = torch.cat([
            x[:, effective_dim_start:effective_dim_end, :],
            (x - xe)[:, effective_dim_start:effective_dim_end, :]
        ], dim=1).squeeze(-1)  # B x 2*dim

        w1 = self.model_u_w1(cond).reshape(bs, -1, self.num_dim_x)           # B x c x n
        w2 = self.model_u_w2(cond).reshape(bs, self.num_dim_control, -1)     # B x m x c

        u = w2.matmul(torch.tanh(w1.matmul(xe))) + uref
        return u


class ResidualCNNBlock(nn.Module):
    """1D Convolutional block with residual connection."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.Tanh()
        
        # 1x1 conv projection if dimensions don't match
        self.proj = None
        if in_channels != out_channels or dilation != 1:
            self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        if self.proj is not None:
            residual = self.proj(residual)
        
        return out + residual


class CNNFeatureExtractor(nn.Module):
    """
    Improved 1D CNN feature extractor.
    Input: flat vector → treated as sequence of length = input_size with 1 channel.
    """
    
    def __init__(self, input_size, output_size, bias=True):
        super().__init__()
        
        # Stem: expand channels
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.Tanh(),
            nn.Dropout(0.1)
        )
        
        # Deep residual blocks with dilation for larger receptive field
        self.blocks = nn.Sequential(
            ResidualCNNBlock(32, 64, kernel_size=3, dilation=1),
            ResidualCNNBlock(64, 64, kernel_size=3, dilation=2),
            ResidualCNNBlock(64, 96, kernel_size=3, dilation=4),
            ResidualCNNBlock(96, 96, kernel_size=3, dilation=8),
            ResidualCNNBlock(96, 128, kernel_size=3, dilation=1),
        )
        
        # Head: global pooling + small MLP
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(128, output_size, bias=bias)
        )

    def forward(self, x):
        # x: bs x input_size
        x = x.unsqueeze(1)  # bs x 1 x input_size
        
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        
        return x


def get_model(num_dim_x, num_dim_control, w_lb, use_cuda=True):
    """
    Returns the model components for single-GPU (or CPU) training.
    
    Args:
        num_dim_x: state dimension (24)
        num_dim_control: control dimension (3)
        w_lb: lower bound for positive definiteness
        use_cuda: whether to move models to GPU (default: True)
    """
    dim = effective_dim_end - effective_dim_start  # 15
    
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    
    # CNN models for structured W matrix (no bias in final layer)
    model_W = CNNFeatureExtractor(
        input_size=dim,
        output_size=num_dim_x * num_dim_x,
        bias=False
    ).to(device)
    
    model_Wbot = CNNFeatureExtractor(
        input_size=dim - num_dim_control,
        output_size=(num_dim_x - num_dim_control) ** 2,
        bias=False
    ).to(device)
    
    # Control correction networks
    c = 3 * num_dim_x
    model_u_w1 = CNNFeatureExtractor(
        input_size=2 * dim,
        output_size=c * num_dim_x,
        bias=True
    ).to(device)
    
    model_u_w2 = CNNFeatureExtractor(
        input_size=2 * dim,
        output_size=num_dim_control * c,
        bias=True
    ).to(device)

    def W_func(x):
        """
        Compute structured positive definite W(x).
        x: B x n x 1
        """
        bs = x.shape[0]
        x = x.squeeze(-1)  # B x n

        W_full = model_W(x[:, effective_dim_start:effective_dim_end]) \
                 .view(bs, num_dim_x, num_dim_x)
        
        W_bot = model_Wbot(x[:, effective_dim_start:effective_dim_end - num_dim_control]) \
                .view(bs, num_dim_x - num_dim_control, num_dim_x - num_dim_control)
        
        W = W_full.clone()
        W[:, :num_dim_x - num_dim_control, :num_dim_x - num_dim_control] = W_bot
        W[:, num_dim_x - num_dim_control:, :num_dim_x - num_dim_control] = 0.0
        
        # Positive definiteness: W = W^T W + w_lb * I
        W = W.transpose(1, 2) @ W
        W = W + w_lb * torch.eye(num_dim_x, device=device).unsqueeze(0)
        
        return W

    u_func = U_FUNC(model_u_w1, model_u_w2, num_dim_x, num_dim_control).to(device)

    return model_W, model_Wbot, model_u_w1, model_u_w2, W_func, u_func