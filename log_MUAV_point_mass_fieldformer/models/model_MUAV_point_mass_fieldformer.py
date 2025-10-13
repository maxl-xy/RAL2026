"""
FieldFormer-based Control Model for MUAV Point Mass System

This module implements a FieldFormer architecture for learning control policies
for the Multi-UAV (MUAV) point mass system. The FieldFormer processes local
neighborhoods of state-error information using transformer attention mechanisms
to generate adaptive control actions.

Key Components:
- StateFieldFormer: Processes local state neighborhoods using transformers
- U_FUNC_FieldFormer: Control function with matrix decomposition approach
- get_model: Factory function for creating all model components

The architecture maintains the Lyapunov-based stability framework while using
FieldFormer networks to learn the control policy, providing better adaptability
to complex dynamics and spatial relationships in the state space.

Author: Generated for RAL 2026 submission
"""

import torch
from torch import nn
from torch.autograd import grad
import torch.nn.functional as F
import numpy as np

# States that are used in the model (effective dimensions from MUAV system)
effective_dim_start = 0  # Include all states starting from payload position
effective_dim_end = 18   # End of all state dimensions

# Control constraints (should match config_MUAV_point_mass.py)
f_bound = 5.0  # Force bounds for each quadrotor

# Control saturation factor - tuned to avoid premature saturation
# while maintaining smooth control response
saturation_factor = 0.3

# Mathematical constant for potential numerical computations
torch.pi = torch.acos(torch.zeros(1)).item() * 2


# ----------------------
# FieldFormer Components (based on original implementation)
# ----------------------

def lin_to_ijk(lin, Nx, Ny, Nt):
    """Convert linear index to (i, j, k) coordinates"""
    i = lin // (Ny * Nt)
    r = lin % (Ny * Nt)
    j = r // Nt
    k = r % Nt
    return i, j, k

def ijk_to_lin(i, j, k, Nx, Ny, Nt):
    """Convert (i, j, k) coordinates to linear index with periodic wrapping"""
    return (i % Nx) * (Ny * Nt) + (j % Ny) * Nt + (k % Nt)

def build_offset_table(k, gammas, dx, dy, dt, max_rad=None):
    """
    Returns integer offsets [(di,dj,dk)] of length >= k, sorted by scaled distance.
    """
    gx, gy, gt = [float(x) for x in gammas]
    # Heuristic radius so we have more than k candidates
    c = 4.0
    base = (c * k) ** (1/3)
    Rx = max(1, int(base * (1.0/max(gx,1e-8))))
    Ry = max(1, int(base * (1.0/max(gy,1e-8))))
    Rt = max(1, int(base * (1.0/max(gt,1e-8))))
    if max_rad is not None:
        Rx = min(Rx, max_rad); Ry = min(Ry, max_rad); Rt = min(Rt, max_rad)

    # Enumerate offsets and distances
    offs = []
    for di in range(-Rx, Rx+1):
        for dj in range(-Ry, Ry+1):
            for dk in range(-Rt, Rt+1):
                # physical deltas
                dxp = di * dx; dyp = dj * dy; dtp = dk * dt
                d2 = (gx*dxp)**2 + (gy*dyp)**2 + (gt*dtp)**2
                offs.append((d2, di, dj, dk))
    offs.sort(key=lambda z: z[0])  # increasing distance
    # drop the center if present
    offs = [(di,dj,dk) for (d2,di,dj,dk) in offs if not (di==0 and dj==0 and dk==0)]
    return offs

def gather_neighbors_periodic(q_lin_idx, k, offsets_ijk, Nx, Ny, Nt):
    """
    Gather k nearest neighbors using pre-sorted offsets with periodic wrapping.
    """
    # unpack to i, j, k0
    i, j, k0 = lin_to_ijk(q_lin_idx, Nx, Ny, Nt)  # each (B,)

    sel = offsets_ijk[:k]
    di = torch.tensor([o[0] for o in sel], device=q_lin_idx.device, dtype=i.dtype)  # (k,)
    dj = torch.tensor([o[1] for o in sel], device=q_lin_idx.device, dtype=i.dtype)
    dk = torch.tensor([o[2] for o in sel], device=q_lin_idx.device, dtype=i.dtype)

    I = i[:, None] + di[None, :]   # (B,k)
    J = j[:, None] + dj[None, :]
    K = k0[:, None] + dk[None, :]

    nb_lin = ijk_to_lin(I, J, K, Nx, Ny, Nt)   # (B,k) with periodic wrap
    return nb_lin, di, dj, dk


class FieldFormer(nn.Module):
    """
    FieldFormer for control problems - closely following the original implementation
    """
    def __init__(self, d_in, d_model=64, nhead=4, num_layers=2, k_neighbors=32, d_ff=128):
        super().__init__()
        self.k = k_neighbors
        # learnable log-gammas -> positive gammas for x,y,t (used when building offsets_ijk outside)
        self.log_gammas = nn.Parameter(torch.zeros(3))
        self.input_proj = nn.Linear(d_in, d_model)
        # Configure TransformerEncoderLayer with explicit settings to avoid flash attention issues
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_ff, 
            dropout=0.1,
            activation='relu',  # Use ReLU instead of GELU for better compatibility
            batch_first=True,
            norm_first=False  # Use post-norm (default) for better stability
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, q_lin_idx, offsets_ijk, coords, vals, dx, dy, dt, Nx, Ny, Nt):
        """
        Forward pass following original FieldFormer structure
        q_lin_idx: (B,)
        offsets_ijk: list[(di,dj,dk)] sorted for current gammas; will take first self.k
        coords: (N, 3) coordinates [x, y, t] 
        vals: (N,) values at coordinates
        dx, dy, dt: grid spacing
        Nx, Ny, Nt: grid dimensions
        Returns:
          pred: (B,)
          extra: dict
        """
        # neighbors via periodic offsets (no BxN distances)
        nb_idx, di, dj, dk = gather_neighbors_periodic(q_lin_idx, self.k, offsets_ijk, Nx, Ny, Nt)  # (B,k)
        device_ = q_lin_idx.device
        
        # Ensure proper tensor types and device placement
        rel_ijk = torch.stack([di, dj, dk], dim=1).to(device=device_, dtype=torch.float32)       # (k,3)
        rel = rel_ijk[None, :, :].expand(q_lin_idx.shape[0], -1, -1)       # (B,k,3)
        
        # scale to physical deltas
        grid_scales = torch.tensor([dx, dy, dt], device=device_, dtype=torch.float32)
        rel = rel * grid_scales[None, None, :]  # (B,k,3)
        
        # Apply learnable scaling with proper gradient handling
        scale = torch.exp(self.log_gammas).to(device_)[None, None, :]      # (1,1,3)
        rel = rel * scale
        
        # neighbor values with proper indexing
        nb_vals = vals[nb_idx]  # (B,k)
        nb_vals = nb_vals.to(device=device_, dtype=torch.float32)[..., None]  # (B,k,1)
        
        # Create tokens ensuring all tensors are on the same device
        tokens = torch.cat([rel, nb_vals], dim=-1)  # (B,k,4)
        
        # Project and process through encoder
        h = self.input_proj(tokens)  # (B,k,d_model)
        
        # Process through transformer encoder with flash attention disabled
        try:
            # Use the new API to disable both flash and efficient attention
            with torch.nn.attention.sdpa_kernel(backends=[torch.nn.attention.SDPBackend.MATH]):
                h = self.encoder(h)  # (B,k,d_model)
        except (AttributeError, ImportError):
            try:
                # Fallback to older API, disable both flash and efficient attention
                with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                    h = self.encoder(h)  # (B,k,d_model)
            except (AttributeError, RuntimeError):
                # Final fallback - use math backend only
                h = self.encoder(h)  # (B,k,d_model)
        
        h_mean = h.mean(dim=1)
        out = self.head(h_mean).squeeze(-1)
        return out, {"gammas": torch.exp(self.log_gammas).detach()}


class U_FUNC_FieldFormer(nn.Module):
    """
    Control function using FieldFormer architecture - following original approach
    """
    def __init__(self, num_dim_x, num_dim_control):
        super(U_FUNC_FieldFormer, self).__init__()
        self.num_dim_x = num_dim_x
        self.num_dim_control = num_dim_control
        
        # Grid parameters for creating coordinate system (following original)
        self.grid_size = 16  # Small grid for efficiency
        self.dx = 0.2  # Grid spacing in state space
        self.dy = 0.2
        self.dt = 0.1
        
        # Control matrix dimensions
        c = 3 * num_dim_x
        
        # Two FieldFormers for the control matrix decomposition (following original structure)
        self.fieldformer_w1 = FieldFormer(d_in=4, d_model=64, nhead=4, num_layers=2, k_neighbors=32, d_ff=128)
        self.fieldformer_w2 = FieldFormer(d_in=4, d_model=64, nhead=4, num_layers=2, k_neighbors=32, d_ff=128)
        
        # Additional MLPs to get the right output dimensions from FieldFormer (since FieldFormer outputs scalars)
        self.w1_head = nn.Linear(1, c * num_dim_x)
        self.w2_head = nn.Linear(1, num_dim_control * c)

    def create_grid_data(self, x, xe):
        """
        Create grid-based representation from state-error information (following original approach)
        """
        bs = x.shape[0]
        device = x.device
        
        # Create coordinate grid (simplified version)
        Nx = Ny = Nt = self.grid_size
        
        # Simple grid coordinates
        x_vals = torch.linspace(-1, 1, Nx, device=device)
        y_vals = torch.linspace(-1, 1, Ny, device=device)
        t_vals = torch.linspace(0, 1, Nt, device=device)
        
        XX, YY, TT = torch.meshgrid(x_vals, y_vals, t_vals, indexing="ij")
        coords = torch.stack([XX.ravel(), YY.ravel(), TT.ravel()], dim=1)  # (N, 3)
        
        # Create values based on state-error information
        vals = torch.zeros(coords.shape[0], device=device)
        
        # Map state information to grid values (simplified)
        for i in range(bs):
            state_norm = torch.norm(x[i, :, 0])
            error_norm = torch.norm(xe[i, :, 0])
            # Use combination of state and error norms to create field values
            vals += (state_norm + error_norm) / bs
        
        return coords, vals, Nx, Ny, Nt

    def forward(self, x, xe, uref):
        bs = x.shape[0]
        device = x.device
        
        # Create grid representation
        coords, vals, Nx, Ny, Nt = self.create_grid_data(x, xe)
        
        # Build offset tables for both FieldFormers
        with torch.no_grad():
            gam1 = torch.exp(self.fieldformer_w1.log_gammas).detach().cpu().numpy()
            gam2 = torch.exp(self.fieldformer_w2.log_gammas).detach().cpu().numpy()
        
        offsets_w1 = build_offset_table(
            k=self.fieldformer_w1.k, gammas=gam1,
            dx=self.dx, dy=self.dy, dt=self.dt
        )
        offsets_w2 = build_offset_table(
            k=self.fieldformer_w2.k, gammas=gam2,
            dx=self.dx, dy=self.dy, dt=self.dt
        )
        
        # Query indices for FieldFormer (sample from grid)
        n_queries = min(bs, coords.shape[0])
        query_indices = torch.randperm(coords.shape[0], device=device)[:n_queries]
        
        # Get predictions from both FieldFormers
        w1_pred, _ = self.fieldformer_w1(query_indices, offsets_w1, coords, vals,
                                        self.dx, self.dy, self.dt, Nx, Ny, Nt)
        w2_pred, _ = self.fieldformer_w2(query_indices, offsets_w2, coords, vals,
                                        self.dx, self.dy, self.dt, Nx, Ny, Nt)
        
        # Expand to batch size and get weight matrices
        w1_pred = w1_pred.mean().unsqueeze(0).expand(bs, 1)  # (bs, 1)
        w2_pred = w2_pred.mean().unsqueeze(0).expand(bs, 1)  # (bs, 1)
        
        w1_flat = self.w1_head(w1_pred)  # (bs, c * num_dim_x)
        w2_flat = self.w2_head(w2_pred)  # (bs, num_dim_control * c)
        
        # Reshape to matrix form
        c = 3 * self.num_dim_x
        w1 = w1_flat.view(bs, c, self.num_dim_x)
        w2 = w2_flat.view(bs, self.num_dim_control, c)
        
        # Extract effective error state
        xe_eff = xe[:, effective_dim_start:effective_dim_end, :]
        
        # Compute control action
        u_raw = torch.bmm(w2, torch.tanh(torch.bmm(w1, xe_eff)))
        
        # Apply control bounds
        bounds = torch.tensor([f_bound] * self.num_dim_control, 
                            dtype=x.dtype, device=device).view(1, -1, 1).expand(bs, -1, -1)
        
        u = torch.tanh(u_raw * saturation_factor) * bounds + uref
        
        return u


def get_model(num_dim_x, num_dim_control, w_lb, use_cuda=False):
    """
    Create FieldFormer-based model components
    """
    dim = effective_dim_end - effective_dim_start
    
    # Traditional MLP components (kept for Lyapunov function for stability)
    model_Wbot = torch.nn.Sequential(
        torch.nn.Linear(dim-num_dim_control, 128, bias=True),
        torch.nn.Tanh(),
        torch.nn.Linear(128, (num_dim_x-num_dim_control) ** 2, bias=False))

    model_W = torch.nn.Sequential(
        torch.nn.Linear(dim, 128, bias=True),
        torch.nn.Tanh(),
        torch.nn.Linear(128, num_dim_x * num_dim_x, bias=False))

    # FieldFormer-based control function
    u_func = U_FUNC_FieldFormer(num_dim_x, num_dim_control)

    if use_cuda:
        model_W = model_W.cuda()
        model_Wbot = model_Wbot.cuda()
        u_func = u_func.cuda()

    def W_func(x):
        """Lyapunov function derivative matrix (kept as MLP for stability)"""
        bs = x.shape[0]
        x = x.squeeze(-1)

        W = model_W(x[:, effective_dim_start:effective_dim_end]).view(bs, num_dim_x, num_dim_x)
        W = W.transpose(1,2).matmul(W)
        W = W + w_lb * torch.eye(num_dim_x).view(1, num_dim_x, num_dim_x).type(x.type())
        
        return W

    # Return components - note that we return the FieldFormer components directly
    # instead of separate fieldformer_w1 and fieldformer_w2
    return model_W, model_Wbot, u_func.fieldformer_w1, u_func.fieldformer_w2, W_func, u_func
