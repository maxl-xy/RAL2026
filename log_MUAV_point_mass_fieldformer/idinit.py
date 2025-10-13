import torch
import torch.nn as nn
import numpy as np

def idinit_linear(layer):
    """Initialize a linear layer with identity initialization."""
    input_dim, output_dim = layer.weight.shape[1], layer.weight.shape[0]
    nn.init.zeros_(layer.weight)
    min_dim = min(input_dim, output_dim)
    eye = torch.eye(min_dim, device=layer.weight.device)
    layer.weight.data[:min_dim, :min_dim] = eye
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(layer.weight)
    scale = torch.sqrt(torch.tensor(2.0 / (fan_in + fan_out)))
    layer.weight.data *= scale
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)