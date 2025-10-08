import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
from torch.autograd import grad
import torch.nn.functional as F
import importlib
import numpy as np
import time
from tqdm import tqdm
import os
import sys

sys.path.append('systems')
sys.path.append('configs')
sys.path.append('models')

# Clear GPU cache
torch.cuda.empty_cache()

# Hyperparameters
bs = 16  # Reduced to avoid OOM
accumulation_steps = 16  # Maintain effective bs=256
num_train = 16 * 64
num_test = 16 * 16
learning_rate = 0.001
epochs = 30
lr_step = 10
_lambda = 0.5
w_ub = 10
w_lb = 0.1

# Configuration variables
task = 'MUAV_point_mass'
log = 'log_MUAV_point_mass_fieldformer'
use_cuda = True

np.random.seed(1024)

# Copy files to log directory
if log is not None:
    os.system('cp *.py ' + log)
    os.system('cp -r models/ ' + log)
    os.system('cp -r configs/ ' + log)
    os.system('cp -r systems/ ' + log)

epsilon = _lambda * 0.1

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

model = importlib.import_module('model_' + task + '_fieldformer')
get_model = model.get_model

model_W, model_Wbot, _, _, W_func, u_func = get_model(num_dim_x, num_dim_control, w_lb=w_lb, use_cuda=use_cuda)

# Datasets
def sample_xef():
    return (X_MAX - X_MIN) * np.random.rand(num_dim_x, 1) + X_MIN

def sample_x(xref):
    xe = (XE_MAX - XE_MIN) * np.random.rand(num_dim_x, 1) + XE_MIN
    x = xref + xe
    x[x > X_MAX] = X_MAX[x > X_MAX]
    x[x < X_MIN] = X_MIN[x < X_MIN]
    return x

def sample_uref():
    return (U_MAX - U_MIN) * np.random.rand(num_dim_control, 1) + U_MIN

def sample_full():
    xref = sample_xef()
    uref = sample_uref()
    x = sample_x(xref)
    return (x, xref, uref)

X_tr = [sample_full() for _ in range(num_train)]
X_te = [sample_full() for _ in range(num_test)]

if 'Bbot_func' not in locals():
    def Bbot_func(B):
        bs = B.shape[0]
        Bbot = torch.cat((
            torch.eye(num_dim_x - num_dim_control, num_dim_x - num_dim_control),
            torch.zeros(num_dim_control, num_dim_x - num_dim_control)
        ), dim=0)
        if use_cuda:
            Bbot = Bbot.cuda()
        Bbot = Bbot.unsqueeze(0)
        return Bbot.repeat(bs, 1, 1)

def Jacobian_Matrix(M, x):
    # M: bs x m x m, x: bs x n x 1
    bs = x.shape[0]
    m = M.size(-1)
    n = x.size(1)
    J = torch.zeros(bs, m, m, n).type(x.type())
    batch_size_indices = 2  # Process 2 i,j pairs at a time
    for i_start in range(0, m, batch_size_indices):
        for j_start in range(0, m, batch_size_indices):
            i_end = min(i_start + batch_size_indices, m)
            j_end = min(j_start + batch_size_indices, m)
            for i in range(i_start, i_end):
                for j in range(j_start, j_end):
                    J[:, i, j, :] = grad(M[:, i, j].sum(), x, create_graph=True, retain_graph=True)[0].squeeze(-1)
            torch.cuda.empty_cache()
    return J

def Jacobian(f, x):
    f = f + 0. * x.sum()
    bs = x.shape[0]
    m = f.size(1)
    n = x.size(1)
    J = torch.zeros(bs, m, n).type(x.type())
    for i in range(m):
        J[:, i, :] = grad(f[:, i, 0].sum(), x, create_graph=True, retain_graph=True)[0].squeeze(-1)
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
    z = torch.randn(K, A.size(-1)).cuda()
    z = z / z.norm(dim=1, keepdim=True)
    zTAz = (z.matmul(A) * z.view(1, K, -1)).sum(dim=2).view(-1)
    negative_index = zTAz.detach().cpu().numpy() < 0
    if negative_index.sum() > 0:
        negative_zTAz = zTAz[negative_index]
        return -1.0 * (negative_zTAz.mean())
    else:
        return torch.tensor(0.).type(z.type()).requires_grad_()

def loss_pos_matrix_eigen_values(A):
    eigv = torch.symeig(A, eigenvectors=True)[0].view(-1)
    negative_index = eigv.detach().cpu().numpy() < 0
    negative_eigv = eigv[negative_index]
    return negative_eigv.norm()

def forward(x, xref, uref, _lambda, verbose=False, acc=False, detach=False):
    bs = x.shape[0]
    with torch.amp.autocast('cuda'):
        W = W_func(x)
        M = torch.inverse(W)
        f = f_func(x)
        B = B_func(x)
        DfDx = Jacobian(f, x)
        DBDx = torch.zeros(bs, num_dim_x, num_dim_x, num_dim_control).type(x.type())
        for i in range(num_dim_control):
            DBDx[:, :, :, i] = Jacobian(B[:, :, i].unsqueeze(-1), x)
        del f
        _Bbot = Bbot_func(B)
        u = u_func(x, x - xref, uref)
        K = Jacobian(u, x)
        del u
        A = DfDx + sum([u_func(x, x - xref, uref)[:, i, 0].unsqueeze(-1).unsqueeze(-1) * DBDx[:, :, :, i] for i in range(num_dim_control)])
        del DfDx
        dot_x = f_func(x) + B.matmul(u_func(x, x - xref, uref))
        dot_M = weighted_gradients(M, dot_x, x, detach=detach)
        dot_W = weighted_gradients(W, dot_x, x, detach=detach)
        del dot_x
        if detach:
            Contraction = dot_M + (A + B.matmul(K)).transpose(1, 2).matmul(M.detach()) + M.detach().matmul(A + B.matmul(K)) + 2 * _lambda * M.detach()
        else:
            Contraction = dot_M + (A + B.matmul(K)).transpose(1, 2).matmul(M) + M.matmul(A + B.matmul(K)) + 2 * _lambda * M
        del dot_M, A, K
        C1_inner = -weighted_gradients(W, f_func(x), x) + Jacobian(f_func(x), x).matmul(W) + W.matmul(Jacobian(f_func(x), x).transpose(1, 2)) + 2 * _lambda * W
        C1_LHS_1 = _Bbot.transpose(1, 2).matmul(C1_inner).matmul(_Bbot)
        del C1_inner
        C2_inners = []
        C2s = []
        for j in range(num_dim_control):
            C2_inner = weighted_gradients(W, B[:, :, j].unsqueeze(-1), x) - (DBDx[:, :, :, j].matmul(W) + W.matmul(DBDx[:, :, :, j].transpose(1, 2)))
            C2 = _Bbot.transpose(1, 2).matmul(C2_inner).matmul(_Bbot)
            C2_inners.append(C2_inner)
            C2s.append(C2)
        del DBDx, _Bbot
        loss = 0
        loss += loss_pos_matrix_random_sampling(-Contraction - epsilon * torch.eye(Contraction.shape[-1]).unsqueeze(0).type(x.type()))
        loss += loss_pos_matrix_random_sampling(-C1_LHS_1 - epsilon * torch.eye(C1_LHS_1.shape[-1]).unsqueeze(0).type(x.type()))
        loss += loss_pos_matrix_random_sampling(w_ub * torch.eye(W.shape[-1]).unsqueeze(0).type(x.type()) - W)
        loss += 1. * sum([1. * (C2 ** 2).reshape(bs, -1).sum(dim=1).mean() for C2 in C2s])
        control_penalty = torch.relu(u_func(x, x - xref, uref).abs() - 5).sum(dim=1).mean()
        if verbose:
            print(torch.symeig(Contraction)[0].min(dim=1)[0].mean(), torch.symeig(Contraction)[0].max(dim=1)[0].mean(), torch.symeig(Contraction)[0].mean())
        if acc:
            Contraction_cpu = Contraction.cpu().float()
            C1_LHS_1_cpu = C1_LHS_1.cpu().float()
            p1 = ((torch.linalg.eigvalsh(Contraction_cpu, UPLO='L') >= 0).sum(dim=1) == 0).numpy()
            p2 = ((torch.linalg.eigvalsh(C1_LHS_1_cpu, UPLO='L') >= 0).sum(dim=1) == 0).numpy()
            del Contraction_cpu, C1_LHS_1_cpu
            return loss, p1, p2, sum([1. * (C2 ** 2).reshape(bs, -1).sum(dim=1).mean() for C2 in C2s]).item(), control_penalty.item()
        else:
            return loss, None, None, None, None

optimizer = torch.optim.Adam(
    list(model_W.parameters()) + list(model_Wbot.parameters()) + list(u_func.model_u.parameters()),
    lr=learning_rate
)

scaler = torch.amp.GradScaler('cuda')

def trainval(X, bs=bs, train=True, _lambda=_lambda, acc=False, detach=False):
    if train:
        indices = np.random.permutation(len(X))
    else:
        indices = np.array(list(range(len(X))))
    total_loss = 0
    total_p1 = 0
    total_p2 = 0
    total_l3 = 0
    total_c4 = 0
    if train:
        _iter = tqdm(range(len(X) // bs))
    else:
        _iter = range(len(X) // bs)
    for b in _iter:
        x = []; xref = []; uref = []
        for id in indices[b * bs:(b + 1) * bs]:
            if use_cuda:
                x.append(torch.from_numpy(X[id][0]).float().cuda())
                xref.append(torch.from_numpy(X[id][1]).float().cuda())
                uref.append(torch.from_numpy(X[id][2]).float().cuda())
            else:
                x.append(torch.from_numpy(X[id][0]).float())
                xref.append(torch.from_numpy(X[id][1]).float())
                uref.append(torch.from_numpy(X[id][2]).float())
        x, xref, uref = (torch.stack(d).detach() for d in (x, xref, uref))
        x = x.requires_grad_()
        with torch.amp.autocast('cuda'):
            loss, p1, p2, l3, c4 = forward(x, xref, uref, _lambda=_lambda, verbose=False, acc=acc, detach=detach)
        if train:
            scaler.scale(loss / accumulation_steps).backward()
            if (b + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            torch.cuda.empty_cache()
        total_loss += loss.item() * x.shape[0]
        if acc:
            total_p1 += p1.sum()
            total_p2 += p2.sum()
            total_l3 += l3 * x.shape[0]
            total_c4 += c4 * x.shape[0]
    return total_loss / len(X), total_p1 / len(X), total_p2 / len(X), total_l3 / len(X), total_c4 / len(X)

best_acc = 0

def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate * (0.1 ** (epoch // lr_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

for epoch in range(0, epochs):
    adjust_learning_rate(optimizer, epoch)
    loss, _, _, _, _ = trainval(X_tr, train=True, _lambda=_lambda, acc=False, detach=True if epoch < lr_step else False)
    print("Training loss: ", loss)
    loss, p1, p2, l3, c4 = trainval(X_te, train=False, _lambda=0., acc=True, detach=False)
    print("Epoch %d: Testing loss/p1/p2/l3/c4: " % epoch, loss, p1, p2, l3, c4)
    if p1 + p2 >= best_acc:
        best_acc = p1 + p2
        filename = log + '/model_best.pth.tar'
        filename_controller = log + '/controller_best.pth.tar'
        torch.save({
            'precs': (loss, p1, p2, l3, c4),
            'model_W': model_W.state_dict(),
            'model_Wbot': model_Wbot.state_dict(),
            'model_u': u_func.model_u.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }, filename)
        torch.save(u_func, filename_controller)