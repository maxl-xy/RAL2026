import torch
from torch import nn
from torch_geometric.nn import GATConv, TransformerConv, global_mean_pool
from torch_geometric.utils import dense_to_sparse

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
        
        cond = torch.cat([
            x[:, effective_dim_start:effective_dim_end, :],
            (x - xe)[:, effective_dim_start:effective_dim_end, :]
        ], dim=1).squeeze(-1)  # B x 30

        w1 = self.model_u_w1(cond).reshape(bs, -1, self.num_dim_x)
        w2 = self.model_u_w2(cond).reshape(bs, self.num_dim_control, -1)

        u = w2.matmul(torch.tanh(w1.matmul(xe))) + uref
        return u


class MultiAgentGNN(nn.Module):
    """
    Small GNN over 4 nodes: 3 drones + 1 slung payload.
    Input: flattened effective state subspace (dim=15 or 30)
    Output: global feature vector via mean pooling.
    """
    
    def __init__(self, input_dim, hidden_dim=128, output_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.num_nodes = 4  # 3 drones + 1 payload
        
        # Project input vector to node features (broadcast + learned transformation)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Fully connected graph (undirected)
        # edge_index: 2 x (4*3) = 2 x 12
        rows = []
        cols = []
        for i in range(4):
            for j in range(4):
                if i != j:
                    rows.append(i)
                    cols.append(j)
        self.register_buffer('edge_index', torch.tensor([rows, cols], dtype=torch.long))
        
        # GNN layers
        self.gat1 = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout, add_self_loops=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        self.transf = TransformerConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout, concat=False)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        # Global pooling + output head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: bs x input_dim  (15 or 30)
        """
        bs = x.shape[0]
        device = x.device
        
        # Project to node features and broadcast to 4 nodes
        h = self.input_proj(x)  # bs x hidden_dim
        h = h.unsqueeze(1).expand(-1, self.num_nodes, -1)  # bs x 4 x hidden_dim
        h = h.reshape(bs * self.num_nodes, -1)  # (bs*4) x hidden_dim
        
        # Batch indices for PyG
        batch = torch.arange(bs, device=device).repeat_interleave(self.num_nodes)
        
        # Repeat edge_index for batch
        edge_index = self.edge_index.to(device)
        edge_index = edge_index + torch.arange(bs, device=device).unsqueeze(1) * self.num_nodes
        edge_index = edge_index.view(2, -1)
        
        # GNN message passing
        h = self.gat1(h, edge_index).relu()
        h = self.norm1(h)
        h = self.dropout(h)
        
        h = self.transf(h, edge_index)
        h = self.norm2(h)
        h = self.dropout(h)
        
        h = self.gat2(h, edge_index)
        h = self.norm3(h)
        
        # Global mean pooling over the 4 agents
        h_pooled = global_mean_pool(h, batch)  # bs x hidden_dim
        
        # Final projection
        out = self.head(h_pooled)
        
        return out


def get_model(num_dim_x=24, num_dim_control=3, w_lb=1e-4, use_cuda=True):
    """
    Returns a meaningful multi-agent GNN-based model for the 3-drone + payload system.
    """
    dim = effective_dim_end - effective_dim_start  # 15
    
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    
    # GNN for full W matrix
    model_W = MultiAgentGNN(
        input_dim=dim,
        hidden_dim=128,
        output_dim=num_dim_x * num_dim_x,
        num_heads=8
    ).to(device)
    
    # GNN for bottom block
    model_Wbot = MultiAgentGNN(
        input_dim=dim - num_dim_control,
        hidden_dim=96,
        output_dim=(num_dim_x - num_dim_control) ** 2,
        num_heads=6
    ).to(device)
    
    # Control networks (larger capacity)
    c = 3 * num_dim_x
    model_u_w1 = MultiAgentGNN(
        input_dim=2 * dim,      # state + error
        hidden_dim=256,
        output_dim=c * num_dim_x,
        num_heads=8
    ).to(device)
    
    model_u_w2 = MultiAgentGNN(
        input_dim=2 * dim,
        hidden_dim=256,
        output_dim=num_dim_control * c,
        num_heads=8
    ).to(device)

    def W_func(x):
        bs = x.shape[0]
        x = x.squeeze(-1)  # B x n

        W_full = model_W(x[:, effective_dim_start:effective_dim_end]).view(bs, num_dim_x, num_dim_x)
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