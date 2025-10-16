import torch
from torch import nn
from torch.autograd import grad
import numpy as np

effective_dim_start = 3
effective_dim_end = 9

class U_FUNC(nn.Module):
    """Control function using CNN-based neural networks."""

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

class CNNBlock(nn.Module):
    """1D CNN block for processing sequential features."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

class CNNFeatureExtractor(nn.Module):
    """CNN-based feature extractor for input processing."""
    
    def __init__(self, input_size, output_size):
        super(CNNFeatureExtractor, self).__init__()
        self.input_size = input_size
        
        # Reshape input to have a channel dimension for 1D conv
        self.input_reshape = nn.Linear(input_size, input_size * 4)  # Create multiple channels
        
        # CNN layers
        self.cnn_layers = nn.Sequential(
            CNNBlock(4, 16, kernel_size=3, padding=1),
            CNNBlock(16, 32, kernel_size=3, padding=1),
            CNNBlock(32, 64, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(1)  # Global average pooling
        )
        
        # Final projection
        self.final_projection = nn.Sequential(
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, output_size)
        )
    
    def forward(self, x):
        bs = x.shape[0]
        
        # Reshape input to create channel dimension
        x = self.input_reshape(x)  # bs x (input_size * 4)
        x = x.view(bs, 4, -1)  # bs x 4 x input_size
        
        # Apply CNN layers
        x = self.cnn_layers(x)  # bs x 64 x 1
        x = x.squeeze(-1)  # bs x 64
        
        # Final projection
        x = self.final_projection(x)
        
        return x

def get_model(num_dim_x, num_dim_control, w_lb, use_cuda = False):
    # CNN-based models for W computation
    model_Wbot = CNNFeatureExtractor(
        effective_dim_end-effective_dim_start-num_dim_control, 
        (num_dim_x-num_dim_control) ** 2
    )

    dim = effective_dim_end - effective_dim_start
    model_W = CNNFeatureExtractor(dim, num_dim_x * num_dim_x)

    # CNN-based models for control computation
    c = 3 * num_dim_x
    model_u_w1 = CNNFeatureExtractor(2*dim, c*num_dim_x)
    model_u_w2 = CNNFeatureExtractor(2*dim, num_dim_control*c)

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
