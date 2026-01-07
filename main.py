import torch
from torch.autograd import grad
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

import importlib
import numpy as np
import time
from tqdm import tqdm
import os
import sys

sys.path.append('systems')
sys.path.append('configs')
sys.path.append('models')

# ==================== HYPERPARAMETERS ====================
bs = 256                    # Per-GPU batch size (total batch = bs * 4)
num_train = bs * 32         # Total training samples
num_test = bs * 8           # Total testing samples
learning_rate = 0.001
epochs = 30
lr_step = 10
_lambda = 0.5
w_ub = 10
w_lb = 0.1

task = 'MUAV'
structure = 'CNN'
log = 'log_MUAV_' + structure

np.random.seed(1024)
# ========================================================

def main_worker(local_rank, world_size):
    # ------------------ DDP Initialization ------------------
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)

    is_main_process = (local_rank == 0)

    # Create log directory and copy source files (only rank 0)
    if is_main_process and log is not None:
        os.makedirs(log, exist_ok=True)
        os.system(f'cp *.py {log}/')
        os.system(f'cp -r models/ {log}/')
        os.system(f'cp -r configs/ {log}/')
        os.system(f'cp -r systems/ {log}/')

    # ------------------ Config & System ------------------
    config = importlib.import_module('config_' + task)
    X_MIN = config.X_MIN
    X_MAX = config.X_MAX
    U_MIN = config.UREF_MIN
    U_MAX = config.UREF_MAX
    XE_MIN = config.XE_MIN
    XE_MAX = config.XE_MAX

    system = importlib.import_module('system_' + task)
    f_func = system.f_func
    B_func = system.B_func
    num_dim_x = system.num_dim_x
    num_dim_control = system.num_dim_control
    if hasattr(system, 'Bbot_func'):
        Bbot_func = system.Bbot_func

    # ------------------ Model (with DDP) ------------------
    model_mod = importlib.import_module('model_' + structure)
    model_W, model_Wbot, model_u_w1, model_u_w2, W_func, u_func = model_mod.get_model(
        num_dim_x, num_dim_control, w_lb=w_lb,
        local_rank=local_rank,
        world_size=world_size,
        use_ddp=True
    )

    # Optimizer on the underlying modules
    optimizer = torch.optim.Adam([
        {'params': model_W.module.parameters()},
        {'params': model_Wbot.module.parameters()},
        {'params': model_u_w1.module.parameters()},
        {'params': model_u_w2.module.parameters()},
    ], lr=learning_rate)

    epsilon = _lambda * 0.1

    # ------------------ Dataset Sampling ------------------
    def sample_full():
        xref = (X_MAX - X_MIN) * np.random.rand(num_dim_x, 1) + X_MIN
        uref = (U_MAX - U_MIN) * np.random.rand(num_dim_control, 1) + U_MIN
        xe = (XE_MAX - XE_MIN) * np.random.rand(num_dim_x, 1) + XE_MIN
        x = xref + xe
        x = np.clip(x, X_MIN, X_MAX)
        return x.astype(np.float32), xref.astype(np.float32), uref.astype(np.float32)

    train_samples = [sample_full() for _ in range(num_train)]
    test_samples  = [sample_full() for _ in range(num_test)]

    x_tr, xref_tr, uref_tr = zip(*train_samples)
    x_te, xref_te, uref_te = zip(*test_samples)

    train_tensors = (
        torch.from_numpy(np.stack(x_tr)),
        torch.from_numpy(np.stack(xref_tr)),
        torch.from_numpy(np.stack(uref_tr))
    )
    test_tensors = (
        torch.from_numpy(np.stack(x_te)),
        torch.from_numpy(np.stack(xref_te)),
        torch.from_numpy(np.stack(uref_te))
    )

    train_dataset = TensorDataset(*train_tensors)
    test_dataset  = TensorDataset(*test_tensors)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
    test_sampler  = DistributedSampler(test_dataset,  num_replicas=world_size, rank=local_rank)

    train_loader = DataLoader(train_dataset, batch_size=bs, sampler=train_sampler,
                              num_workers=8, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=bs, sampler=test_sampler,
                              num_workers=8, pin_memory=True)

    # ------------------ Helper Functions ------------------
    if 'Bbot_func' not in locals():
        def Bbot_func(B):
            bs = B.shape[0]
            Bbot = torch.cat((torch.eye(num_dim_x - num_dim_control, num_dim_x - num_dim_control),
                              torch.zeros(num_dim_control, num_dim_x - num_dim_control)), dim=0)
            Bbot = Bbot.to(local_rank)
            return Bbot.unsqueeze(0).repeat(bs, 1, 1)

    def Jacobian_Matrix(M, x):
        bs = x.shape[0]
        m = M.size(-1)
        n = x.size(1)
        J = torch.zeros(bs, m, m, n, device=x.device, dtype=x.dtype)
        for i in range(m):
            for j in range(m):
                J[:, i, j, :] = grad(M[:, i, j].sum(), x, create_graph=True)[0].squeeze(-1)
        return J

    def Jacobian(f, x):
        f = f + 0. * x.sum()
        bs = x.shape[0]
        m = f.size(1)
        n = x.size(1)
        J = torch.zeros(bs, m, n, device=x.device, dtype=x.dtype)
        for i in range(m):
            J[:, i, :] = grad(f[:, i, 0].sum(), x, create_graph=True)[0].squeeze(-1)
        return J

    def weighted_gradients(W, v, x, detach=False):
        assert v.size() == x.size()
        bs = x.shape[0]
        if detach:
            return (Jacobian_Matrix(W, x).detach() * v.view(bs, 1, 1, -1)).sum(dim=3)
        else:
            return (Jacobian_Matrix(W, x) * v.view(bs, 1, 1, -1)).sum(dim=3)

    K = 1024
    def loss_pos_matrix_random_sampling(A):
        z = torch.randn(K, A.size(-1), device=A.device)
        z = z / z.norm(dim=1, keepdim=True)
        zTAz = (z.matmul(A) * z.unsqueeze(1)).sum(dim=2).view(-1)
        negative = zTAz < 0
        if negative.any():
            return -zTAz[negative].mean()
        return torch.tensor(0., device=A.device)

    def loss_pos_matrix_eigen_values(A):
        eigv = torch.linalg.eigvals(A).real
        negative = eigv < 0
        return negative.float().sum()  # simple alternative

    def forward(x, xref, uref, _lambda, acc=False, detach=False):
        bs = x.shape[0]
        W = W_func(x)
        M = torch.inverse(W)
        f = f_func(x)
        B = B_func(x)
        DfDx = Jacobian(f, x)

        DBDx = torch.zeros(bs, num_dim_x, num_dim_x, num_dim_control, device=x.device)
        for i in range(num_dim_control):
            DBDx[:, :, :, i] = Jacobian(B[:, :, i].unsqueeze(-1), x)

        _Bbot = Bbot_func(B)
        u = u_func(x, x - xref, uref)
        K = Jacobian(u, x)

        A = DfDx + sum([u[:, i, 0].unsqueeze(-1).unsqueeze(-1) * DBDx[:, :, :, i] for i in range(num_dim_control)])
        dot_x = f + B.matmul(u)

        dot_M = weighted_gradients(M, dot_x, x, detach=detach)
        dot_W = weighted_gradients(W, dot_x, x, detach=detach)

        if detach:
            Contraction = dot_M + (A + B.matmul(K)).transpose(1,2).matmul(M.detach()) \
                        + M.detach().matmul(A + B.matmul(K)) + 2 * _lambda * M.detach()
        else:
            Contraction = dot_M + (A + B.matmul(K)).transpose(1,2).matmul(M) \
                        + M.matmul(A + B.matmul(K)) + 2 * _lambda * M

        C1_inner = -weighted_gradients(W, f, x) + DfDx.matmul(W) + W.matmul(DfDx.transpose(1,2)) + 2 * _lambda * W
        C1_LHS_1 = _Bbot.transpose(1,2).matmul(C1_inner).matmul(_Bbot)

        C2s = []
        for j in range(num_dim_control):
            C2_inner = weighted_gradients(W, B[:,:,j].unsqueeze(-1), x) \
                     - (DBDx[:,:,:,j].matmul(W) + W.matmul(DBDx[:,:,:,j].transpose(1,2)))
            C2 = _Bbot.transpose(1,2).matmul(C2_inner).matmul(_Bbot)
            C2s.append(C2)

        loss = 0.0
        loss += loss_pos_matrix_random_sampling(-Contraction - epsilon * torch.eye(num_dim_x, device=x.device).unsqueeze(0))
        loss += loss_pos_matrix_random_sampling(-C1_LHS_1 - epsilon * torch.eye(C1_LHS_1.shape[-1], device=x.device).unsqueeze(0))
        loss += loss_pos_matrix_random_sampling(w_ub * torch.eye(num_dim_x, device=x.device).unsqueeze(0) - W)
        loss += sum((C2**2).reshape(bs, -1).sum(dim=1).mean() for C2 in C2s)

        control_penalty = torch.relu(u.abs() - 10).sum(dim=1).mean()
        loss += control_penalty

        if acc:
            contr_ok = (torch.linalg.eigvalsh(Contraction) >= -epsilon).all(dim=1)
            c1_ok    = (torch.linalg.eigvalsh(C1_LHS_1) >= -epsilon).all(dim=1)
            return loss, contr_ok.float().cpu().numpy(), c1_ok.float().cpu().numpy(), \
                   sum((C2**2).reshape(bs,-1).sum(dim=1).mean().item() for C2 in C2s), control_penalty.item()
        return loss, None, None, None, None

    # ------------------ LR Scheduler ------------------
    def adjust_learning_rate(optimizer, epoch):
        lr = learning_rate * (0.1 ** (epoch // lr_step))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # ------------------ Training Loop ------------------
    best_acc = 0.0

    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)

        # Training
        model_W.train(); model_Wbot.train(); model_u_w1.train(); model_u_w2.train()
        u_func.train()

        train_loss = 0.0
        for x_batch, xref_batch, uref_batch in train_loader:
            x    = x_batch.to(local_rank).unsqueeze(-1).requires_grad_()
            xref = xref_batch.to(local_rank).unsqueeze(-1)
            uref = uref_batch.to(local_rank).unsqueeze(-1)

            loss, _, _, _, _ = forward(x, xref, uref, _lambda=_lambda,
                                       detach=(epoch < lr_step))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_loader.dataset)

        if is_main_process:
            print(f"Epoch {epoch:02d} [Train] Loss: {train_loss:.6f}")

        # Validation
        model_W.eval(); model_Wbot.eval(); model_u_w1.eval(); model_u_w2.eval()
        u_func.eval()

        val_loss = 0.0
        val_p1 = val_p2 = val_l3 = val_c4 = 0.0
        total_samples = 0

        with torch.no_grad():
            for x_batch, xref_batch, uref_batch in test_loader:
                x    = x_batch.to(local_rank).unsqueeze(-1)
                xref = xref_batch.to(local_rank).unsqueeze(-1)
                uref = uref_batch.to(local_rank).unsqueeze(-1)

                loss, p1, p2, l3, c4 = forward(x, xref, uref, _lambda=0.0, acc=True, detach=False)

                bs_batch = x.size(0)
                val_loss += loss.item() * bs_batch
                val_p1   += p1.sum()
                val_p2   += p2.sum()
                val_l3   += l3 * bs_batch
                val_c4   += c4 * bs_batch
                total_samples += bs_batch

        val_loss /= total_samples
        val_p1   /= total_samples
        val_p2   /= total_samples
        val_l3   /= total_samples
        val_c4   /= total_samples

        if is_main_process:
            print(f"Epoch {epoch:02d} [Test]  Loss: {val_loss:.6f} | "
                  f"p1: {val_p1:.4f} | p2: {val_p2:.4f} | l3: {val_l3:.6f} | c4: {val_c4:.6f}")

            if val_p1 + val_p2 > best_acc:
                best_acc = val_p1 + val_p2
                torch.save({
                    'precs': (val_loss, val_p1, val_p2, val_l3, val_c4),
                    'model_W': model_W.module.state_dict(),
                    'model_Wbot': model_Wbot.module.state_dict(),
                    'model_u_w1': model_u_w1.module.state_dict(),
                    'model_u_w2': model_u_w2.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch
                }, f'{log}/model_best.pth.tar')
                torch.save(u_func.module, f'{log}/controller_best.pth.tar')

    dist.destroy_process_group()


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)