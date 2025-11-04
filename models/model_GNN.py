import torch
from torch import nn
from torch.autograd import grad
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, GATConv, SAGEConv, TransformerConv, 
    global_mean_pool, global_max_pool, global_add_pool,
    BatchNorm, LayerNorm, GraphNorm
)
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch, dense_to_sparse
import math

effective_dim_start = 3
effective_dim_end = 9

class U_FUNC(nn.Module):
    """Control function using GNN-based neural networks."""

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

class MultiHeadGATLayer(nn.Module):
    """Advanced Multi-Head Graph Attention Layer with residual connections."""
    
    def __init__(self, in_channels, out_channels, heads=8, dropout=0.1, edge_dim=None):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = heads
        self.out_channels = out_channels
        
        # Multi-head attention
        self.gat = GATConv(
            in_channels, 
            out_channels, 
            heads=heads, 
            dropout=dropout, 
            concat=False,
            edge_dim=edge_dim,
            add_self_loops=True
        )
        
        # Normalization and activation
        self.norm = GraphNorm(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection projection if needed
        self.residual_proj = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x, edge_index, batch=None, edge_attr=None):
        residual = self.residual_proj(x)
        
        # Apply GAT
        x = self.gat(x, edge_index, edge_attr=edge_attr)
        
        # Add residual connection
        x = x + residual
        
        # Normalize and activate
        x = self.norm(x, batch)
        x = self.activation(x)
        x = self.dropout(x)
        
        return x

class GraphTransformerLayer(nn.Module):
    """Graph Transformer layer for long-range dependencies."""
    
    def __init__(self, in_channels, out_channels, heads=8, dropout=0.1):
        super(GraphTransformerLayer, self).__init__()
        
        self.transformer = TransformerConv(
            in_channels,
            out_channels,
            heads=heads,
            dropout=dropout,
            concat=False,
            beta=True,
            edge_dim=None
        )
        
        self.norm = GraphNorm(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # Residual projection
        self.residual_proj = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x, edge_index, batch=None):
        residual = self.residual_proj(x)
        
        x = self.transformer(x, edge_index)
        x = x + residual
        x = self.norm(x, batch)
        x = self.activation(x)
        x = self.dropout(x)
        
        return x

class AdaptivePooling(nn.Module):
    """Adaptive pooling that combines multiple pooling strategies."""
    
    def __init__(self, hidden_dim):
        super(AdaptivePooling, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Attention weights for combining different pooling methods
        self.pool_attention = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x, batch):
        # Different pooling strategies
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        add_pool = global_add_pool(x, batch)
        
        # Stack pooled features
        pooled_features = torch.stack([mean_pool, max_pool, add_pool], dim=-1)  # bs x hidden_dim x 3
        
        # Compute attention weights
        combined = torch.cat([mean_pool, max_pool, add_pool], dim=-1)  # bs x (hidden_dim * 3)
        attention_weights = self.pool_attention(combined)  # bs x 3
        
        # Weighted combination
        output = torch.sum(pooled_features * attention_weights.unsqueeze(1), dim=-1)  # bs x hidden_dim
        
        return output

class PositionalEncoding(nn.Module):
    """Positional encoding for graph nodes."""
    
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x, node_indices):
        # x: num_nodes x d_model
        # node_indices: num_nodes (indices for each node)
        return x + self.pe[node_indices % self.pe.size(0)]

class GNNFeatureExtractor(nn.Module):
    """Advanced GNN-based feature extractor with multiple graph layers."""
    
    def __init__(self, input_size, output_size, num_nodes=None, hidden_dim=256):
        super(GNNFeatureExtractor, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        
        # Adaptive number of nodes based on input size
        if num_nodes is None:
            self.num_nodes = max(8, min(32, input_size * 2))  # Adaptive sizing
        else:
            self.num_nodes = num_nodes
            
        # Initial node embedding with positional encoding
        self.node_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=self.num_nodes)
        
        # Graph neural network layers
        self.gnn_layers = nn.ModuleList([
            # Initial GAT layers for local attention
            MultiHeadGATLayer(hidden_dim, hidden_dim, heads=8, dropout=0.1),
            MultiHeadGATLayer(hidden_dim, hidden_dim, heads=8, dropout=0.1),
            
            # Graph Transformer for global attention
            GraphTransformerLayer(hidden_dim, hidden_dim, heads=8, dropout=0.1),
            
            # Another GAT layer
            MultiHeadGATLayer(hidden_dim, hidden_dim, heads=4, dropout=0.1),
            
            # Final compression layer
            MultiHeadGATLayer(hidden_dim, hidden_dim // 2, heads=4, dropout=0.1),
        ])
        
        # Adaptive pooling
        self.adaptive_pool = AdaptivePooling(hidden_dim // 2)
        
        # Final output projection with skip connection from input
        self.input_projection = nn.Linear(input_size, hidden_dim // 4)
        self.final_projection = nn.Sequential(
            nn.Linear(hidden_dim // 2 + hidden_dim // 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_size)
        )
        
        # Register graph structure
        self._register_graph_structure()
    
    def _register_graph_structure(self):
        """Create an adaptive graph structure based on the number of nodes."""
        edge_indices = []
        
        # Create a more sophisticated graph topology
        for i in range(self.num_nodes):
            # Local connections (nearest neighbors in a ring)
            next_node = (i + 1) % self.num_nodes
            prev_node = (i - 1) % self.num_nodes
            edge_indices.extend([[i, next_node], [i, prev_node]])
            
            # Long-range connections (every k-th node)
            for k in [2, 3, 5]:  # Different stride patterns
                if k < self.num_nodes:
                    target = (i + k) % self.num_nodes
                    edge_indices.append([i, target])
            
            # Hub connections (connect to central nodes)
            if self.num_nodes > 4:
                hub_nodes = [self.num_nodes // 4, self.num_nodes // 2, 3 * self.num_nodes // 4]
                for hub in hub_nodes:
                    if hub != i and hub < self.num_nodes:
                        edge_indices.extend([[i, hub], [hub, i]])
        
        # Remove duplicates and self-loops
        edge_indices = list(set([tuple(edge) for edge in edge_indices if edge[0] != edge[1]]))
        
        if len(edge_indices) == 0:  # Fallback for single node
            edge_indices = [[0, 0]]
            
        self.edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    
    def _create_graph_from_vector(self, x):
        """Convert input vector to sophisticated graph representation."""
        bs = x.shape[0]
        device = x.device
        
        # Create node features with multiple strategies
        node_features = torch.zeros(bs, self.num_nodes, 1, device=device)
        
        if self.input_size <= self.num_nodes:
            # Direct mapping with padding
            node_features[:, :self.input_size, 0] = x
            
            # Add contextual features to remaining nodes
            if self.input_size < self.num_nodes:
                # Statistical features
                x_mean = x.mean(dim=1, keepdim=True)
                x_std = x.std(dim=1, keepdim=True)
                x_max = x.max(dim=1, keepdim=True)[0]
                x_min = x.min(dim=1, keepdim=True)[0]
                
                extra_features = [x_mean, x_std, x_max, x_min]
                for i, feat in enumerate(extra_features):
                    if self.input_size + i < self.num_nodes:
                        node_features[:, self.input_size + i, 0] = feat.squeeze()
        else:
            # Intelligent downsampling for large inputs
            # Use learned projections to map to nodes
            step = self.input_size / self.num_nodes
            for i in range(self.num_nodes):
                start_idx = int(i * step)
                end_idx = int((i + 1) * step)
                if end_idx <= self.input_size:
                    # Average pooling over the segment
                    node_features[:, i, 0] = x[:, start_idx:end_idx].mean(dim=1)
                else:
                    node_features[:, i, 0] = x[:, start_idx:].mean(dim=1)
        
        return node_features
    
    def forward(self, x):
        bs = x.shape[0]
        device = x.device
        
        # Store original input for skip connection
        input_proj = self.input_projection(x)
        
        # Convert input to graph representation
        node_features = self._create_graph_from_vector(x)  # bs x num_nodes x 1
        
        # Embed node features
        node_features = node_features.view(-1, 1)  # (bs * num_nodes) x 1
        node_features = self.node_embedding(node_features)  # (bs * num_nodes) x hidden_dim
        
        # Add positional encoding
        node_indices = torch.arange(self.num_nodes, device=device).repeat(bs)
        node_features = self.pos_encoding(node_features, node_indices)
        
        # Create batch indices
        batch = torch.arange(bs, device=device).repeat_interleave(self.num_nodes)
        
        # Expand edge indices for batched processing
        edge_index = self.edge_index.to(device)
        batch_edge_index = []
        for b in range(bs):
            batch_edge_index.append(edge_index + b * self.num_nodes)
        batch_edge_index = torch.cat(batch_edge_index, dim=1)
        
        # Apply GNN layers
        for gnn_layer in self.gnn_layers:
            node_features = gnn_layer(node_features, batch_edge_index, batch)
        
        # Adaptive pooling to get graph-level representation
        graph_features = self.adaptive_pool(node_features, batch)  # bs x (hidden_dim // 2)
        
        # Combine with input skip connection
        combined_features = torch.cat([graph_features, input_proj], dim=1)
        
        # Final projection
        output = self.final_projection(combined_features)
        
        return output

def get_model(num_dim_x, num_dim_control, w_lb, use_cuda=False):
    """
    Create GNN-based models for the control system.
    
    Args:
        num_dim_x: State dimension
        num_dim_control: Control dimension  
        w_lb: Lower bound for W matrix regularization
        use_cuda: Whether to use CUDA
        
    Returns:
        Tuple of models: (model_W, model_Wbot, model_u_w1, model_u_w2, W_func, u_func)
    """
    
    # Advanced GNN-based models for W computation
    model_Wbot = GNNFeatureExtractor(
        input_size=effective_dim_end - effective_dim_start - num_dim_control,
        output_size=(num_dim_x - num_dim_control) ** 2,
        num_nodes=max(8, effective_dim_end - effective_dim_start - num_dim_control),
        hidden_dim=256
    )

    dim = effective_dim_end - effective_dim_start
    model_W = GNNFeatureExtractor(
        input_size=dim,
        output_size=num_dim_x * num_dim_x,
        num_nodes=max(8, dim),
        hidden_dim=256
    )

    # Advanced GNN-based models for control computation
    c = 3 * num_dim_x
    model_u_w1 = GNNFeatureExtractor(
        input_size=2 * dim,
        output_size=c * num_dim_x,
        num_nodes=max(8, 2 * dim),
        hidden_dim=512  # Larger for control computation
    )
    
    model_u_w2 = GNNFeatureExtractor(
        input_size=2 * dim,
        output_size=num_dim_control * c,
        num_nodes=max(8, 2 * dim),
        hidden_dim=512
    )

    if use_cuda:
        model_W = model_W.cuda()
        model_Wbot = model_Wbot.cuda()
        model_u_w1 = model_u_w1.cuda()
        model_u_w2 = model_u_w2.cuda()

    def W_func(x):
        """Compute the W matrix using GNN models."""
        bs = x.shape[0]
        x = x.squeeze(-1)

        # Generate W matrix using GNN
        W = model_W(x[:, effective_dim_start:effective_dim_end]).view(bs, num_dim_x, num_dim_x)
        
        # Generate bottom part of W matrix
        Wbot = model_Wbot(x[:, effective_dim_start:effective_dim_end-num_dim_control]).view(
            bs, num_dim_x-num_dim_control, num_dim_x-num_dim_control
        )
        
        # Construct structured W matrix
        W[:, 0:num_dim_x-num_dim_control, 0:num_dim_x-num_dim_control] = Wbot
        W[:, num_dim_x-num_dim_control::, 0:num_dim_x-num_dim_control] = 0

        # Make W positive definite
        W = W.transpose(1,2).matmul(W)
        W = W + w_lb * torch.eye(num_dim_x).view(1, num_dim_x, num_dim_x).type(x.type())
        
        return W

    # Create control function
    u_func = U_FUNC(model_u_w1, model_u_w2, num_dim_x, num_dim_control)

    return model_W, model_Wbot, model_u_w1, model_u_w2, W_func, u_func
