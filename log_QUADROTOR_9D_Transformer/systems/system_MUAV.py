import torch
import math
import numpy as np

num_dim_x = 24
num_dim_control = 9


# Physical parameters
g = 9.81
m_p = 1.3  # mass of the payload
m_q = [1.5, 1.5, 1.5]  # mass of the quadrotors
J_p = torch.diag(torch.tensor([0.5, 0.5, 0.5]))  # payload inertia
l = [0.98, 0.98, 0.98]  # cable lengths

# Tether point vectors in body frame (3x3 matrix) - equilateral triangle
# Centered at payload center of mass
t = torch.tensor([
    [1.085, 0, 0],
    [-0.5425, -0.9396, 0],
    [-0.5425, 0.9396, 0]
]).T


def get_R(phi, theta, psi):
    """Rotation matrix from Euler angles (NED frame: North-East-Down)
    Standard aerospace convention (3-2-1 Euler sequence: Yaw-Pitch-Roll)
    phi: roll about North axis
    theta: pitch about East axis
    psi: yaw about Down axis
    """
    bs = phi.shape[0]
    phi = phi.view(-1, 1, 1)
    theta = theta.view(-1, 1, 1)
    psi = psi.view(-1, 1, 1)
    
    c_phi = torch.cos(phi)
    s_phi = torch.sin(phi)
    c_theta = torch.cos(theta)
    s_theta = torch.sin(theta)
    c_psi = torch.cos(psi)
    s_psi = torch.sin(psi)
    
    R11 = c_psi * c_theta
    R12 = c_psi * s_theta * s_phi - s_psi * c_phi
    R13 = c_psi * s_theta * c_phi + s_psi * s_phi
    R21 = s_psi * c_theta
    R22 = s_psi * s_theta * s_phi + c_psi * c_phi
    R23 = s_psi * s_theta * c_phi - c_psi * s_phi
    R31 = -s_theta
    R32 = c_theta * s_phi
    R33 = c_theta * c_phi
    
    return torch.cat([
        torch.cat([R11, R12, R13], dim=2), 
        torch.cat([R21, R22, R23], dim=2), 
        torch.cat([R31, R32, R33], dim=2)
    ], dim=1)


def get_Z(r, l):
    r_dot_product = torch.bmm(r.transpose(1, 2), r)
    Z = torch.sqrt(l**2 - r_dot_product)
    return Z


def get_B(r, l):
    bs = r.shape[0]
    Z = get_Z(r, l)
    I2 = torch.eye(2).repeat(bs, 1, 1).type(r.type())
    B = torch.cat([I2, -r.transpose(1, 2)/Z], dim=1)
    return B


def get_B_dot(r, v, l):
    bs = r.shape[0]
    O2 = torch.zeros(bs, 2, 2).type(r.type())
    Z = get_Z(r, l)
    r_T = r.transpose(1, 2)
    v_T = v.transpose(1, 2)
    B_dot = torch.cat([O2, -(Z ** 2 * v_T + torch.bmm(r_T, v) * r_T)/(Z ** 3)], dim=1)
    return B_dot


def skew_symmetric(v):
    """Skew symmetric matrix from vector - handles batch operations"""
    bs = v.shape[0]
    zeros = torch.zeros(bs, 1, 1).type(v.type())
    return torch.cat([
        torch.cat([zeros, -v[:, 2:3, :], v[:, 1:2, :]], dim=2), 
        torch.cat([v[:, 2:3, :], zeros, -v[:, 0:1, :]], dim=2), 
        torch.cat([-v[:, 1:2, :], v[:, 0:1, :], zeros], dim=2)
    ], dim=1)


def f_func(x):
    """System dynamics"""
    bs = x.shape[0]
    
    # Extract states
    x_p_x, x_p_y, x_p_z, phi, theta, psi = [x[:, i, 0] for i in range(6)]
    r_1_x, r_1_y, r_2_x, r_2_y, r_3_x, r_3_y = [x[:, i, 0] for i in range(6, 12)]
    v_p_x, v_p_y, v_p_z, omega_x, omega_y, omega_z = [x[:, i, 0] for i in range(12, 18)]
    v_q1_x, v_q1_y, v_q2_x, v_q2_y, v_q3_x, v_q3_y = [x[:, i, 0] for i in range(18, 24)]

    phi = phi.view(-1, 1, 1)
    theta = theta.view(-1, 1, 1)
    psi = psi.view(-1, 1, 1)

    # Compute Euler angle rates
    S_B_inv = torch.cat([
        torch.cat([
            torch.ones_like(phi),
            torch.sin(phi) * torch.tan(theta),
            torch.cos(phi) * torch.tan(theta)
        ], dim=2),
        torch.cat([
            torch.zeros_like(phi),
            torch.cos(phi),
            -torch.sin(phi)
        ], dim=2),
        torch.cat([
            torch.zeros_like(phi),
            torch.sin(phi) / torch.cos(theta),
            torch.cos(phi) / torch.cos(theta)
        ], dim=2)
    ], dim=1)  # (bs, 3, 3)

    # State vectors
    x_p = torch.stack([x_p_x, x_p_y, x_p_z], dim=1).unsqueeze(-1)  # (bs, 3, 1)
    omega_p = torch.stack([omega_x, omega_y, omega_z], dim=1).unsqueeze(-1)  # (bs, 3, 1)
    euler_dot_p = torch.bmm(S_B_inv, omega_p)  # (bs, 3, 1)
    v_p = torch.stack([v_p_x, v_p_y, v_p_z], dim=1).unsqueeze(-1)  # (bs, 3, 1)
    
    # Cable vectors
    r_1 = torch.stack([r_1_x, r_1_y], dim=1).unsqueeze(-1)  # (bs, 2, 1)
    r_2 = torch.stack([r_2_x, r_2_y], dim=1).unsqueeze(-1)  # (bs, 2, 1)
    r_3 = torch.stack([r_3_x, r_3_y], dim=1).unsqueeze(-1)  # (bs, 2, 1)

    v_q1 = torch.stack([v_q1_x, v_q1_y], dim=1).unsqueeze(-1)  # (bs, 2, 1)
    v_q2 = torch.stack([v_q2_x, v_q2_y], dim=1).unsqueeze(-1)  # (bs, 2, 1)
    v_q3 = torch.stack([v_q3_x, v_q3_y], dim=1).unsqueeze(-1)  # (bs, 2, 1)
    
    R = get_R(phi, theta, psi)  # (bs, 3, 3)
    R_T = R.transpose(1, 2)  # (bs, 3, 3)

    # Cable matrices
    B_1 = get_B(r_1, l[0])  # (bs, 3, 2)
    B_2 = get_B(r_2, l[1])  # (bs, 3, 2)
    B_3 = get_B(r_3, l[2])  # (bs, 3, 2)
    
    B_1_T = B_1.transpose(1, 2)  # (bs, 2, 3)
    B_2_T = B_2.transpose(1, 2)  # (bs, 2, 3)
    B_3_T = B_3.transpose(1, 2)  # (bs, 2, 3)
    
    B_1_dot = get_B_dot(r_1, v_q1, l[0])
    B_2_dot = get_B_dot(r_2, v_q2, l[1])
    B_3_dot = get_B_dot(r_3, v_q3, l[2])

    # Skew symmetric matrices for tether points
    t_expanded = t.unsqueeze(0).expand(bs, -1, -1).unsqueeze(-1).type(x.type())  # (bs, 3, 3, 1)
    t_1_skew = skew_symmetric(t_expanded[:, :, 0, :])  # (bs, 3, 3)
    t_2_skew = skew_symmetric(t_expanded[:, :, 1, :])  # (bs, 3, 3)
    t_3_skew = skew_symmetric(t_expanded[:, :, 2, :])  # (bs, 3, 3)
    
    omega_p_skew = skew_symmetric(omega_p)
    
    # Inertia matrices
    J_p_expanded = J_p.unsqueeze(0).expand(bs, -1, -1).type(x.type())  # (bs, 3, 3)
    
    # A matrix and J matrix  
    A = m_q[0] * t_1_skew + m_q[1] * t_2_skew + m_q[2] * t_3_skew  # (bs, 3, 3)
    A_T = A.transpose(1, 2)
    
    J_q = -m_q[0] * torch.bmm(t_1_skew, t_1_skew) - m_q[1] * torch.bmm(t_2_skew, t_2_skew) - m_q[2] * torch.bmm(t_3_skew, t_3_skew)
    
    # Mass matrix M (12x12)
    M = torch.zeros(bs, 12, 12).type(x.type())
    
    # M11: (3x3) - translational mass
    M11 = (m_p + sum(m_q)) * torch.eye(3).repeat(bs, 1, 1).type(x.type())
    
    # M12: (3x3) - coupling between translation and rotation
    M12 = torch.bmm(R, A_T)
    
    # M13, M14, M15: (3x2) - coupling with cable constraints
    M13 = m_q[0] * B_1
    M14 = m_q[1] * B_2
    M15 = m_q[2] * B_3
    
    # M22: (3x3) - rotational inertia
    M22 = J_p_expanded + J_q
    
    # M23, M24, M25: (3x2) - rotational coupling with cables
    M23 = m_q[0] * torch.bmm(t_1_skew, torch.bmm(R_T, B_1))
    M24 = m_q[1] * torch.bmm(t_2_skew, torch.bmm(R_T, B_2))
    M25 = m_q[2] * torch.bmm(t_3_skew, torch.bmm(R_T, B_3))

    # M33, M44, M55: (2x2) - cable mass matrices
    M33 = m_q[0] * torch.bmm(B_1_T, B_1)
    M44 = m_q[1] * torch.bmm(B_2_T, B_2)
    M55 = m_q[2] * torch.bmm(B_3_T, B_3)
    
    # Assemble mass matrix
    zeros_22 = torch.zeros(bs, 2, 2).type(x.type())
    
    M = torch.cat([
        torch.cat([M11, M12, M13, M14, M15], dim=2),
        torch.cat([M12.transpose(1, 2), M22, M23, M24, M25], dim=2),
        torch.cat([M13.transpose(1, 2), M23.transpose(1, 2), M33, zeros_22, zeros_22], dim=2),
        torch.cat([M14.transpose(1, 2), M24.transpose(1, 2), zeros_22, M44, zeros_22], dim=2),
        torch.cat([M15.transpose(1, 2), M25.transpose(1, 2), zeros_22, zeros_22, M55], dim=2)
    ], dim=1)

    # Coriolis matrix C
    C = torch.zeros(bs, 12, 12).type(x.type())
    zeros_33 = torch.zeros(bs, 3, 3).type(x.type())
    zeros_23 = torch.zeros(bs, 2, 3).type(x.type())
    
    # C matrix components
    C12 = torch.bmm(R, torch.bmm(omega_p_skew, A_T))
    C13 = m_q[0] * B_1_dot
    C14 = m_q[1] * B_2_dot
    C15 = m_q[2] * B_3_dot
    
    C22 = -skew_symmetric(torch.bmm(J_p_expanded, omega_p)) - (
        m_q[0] * torch.bmm(t_1_skew, torch.bmm(omega_p_skew, t_1_skew)) +
        m_q[1] * torch.bmm(t_2_skew, torch.bmm(omega_p_skew, t_2_skew)) +
        m_q[2] * torch.bmm(t_3_skew, torch.bmm(omega_p_skew, t_3_skew))
    )

    C23 = m_q[0] * torch.bmm(t_1_skew, torch.bmm(R_T, B_1_dot))
    C24 = m_q[1] * torch.bmm(t_2_skew, torch.bmm(R_T, B_2_dot))
    C25 = m_q[2] * torch.bmm(t_3_skew, torch.bmm(R_T, B_3_dot))

    C32 = -m_q[0] * torch.bmm(B_1_T, torch.bmm(R, torch.bmm(omega_p_skew, t_1_skew)))
    C42 = -m_q[1] * torch.bmm(B_2_T, torch.bmm(R, torch.bmm(omega_p_skew, t_2_skew)))
    C52 = -m_q[2] * torch.bmm(B_3_T, torch.bmm(R, torch.bmm(omega_p_skew, t_3_skew)))
    
    C33 = m_q[0] * torch.bmm(B_1_T, B_1_dot)
    C44 = m_q[1] * torch.bmm(B_2_T, B_2_dot)
    C55 = m_q[2] * torch.bmm(B_3_T, B_3_dot)
    
    C = torch.cat([
        torch.cat([zeros_33, C12, C13, C14, C15], dim=2),
        torch.cat([zeros_33, C22, C23, C24, C25], dim=2),
        torch.cat([zeros_23, C32, C33, zeros_22, zeros_22], dim=2),
        torch.cat([zeros_23, C42, zeros_22, C44, zeros_22], dim=2),
        torch.cat([zeros_23, C52, zeros_22, zeros_22, C55], dim=2)
    ], dim=1)
    
    # Gravity forces - NED convention: gravity is positive in Down direction
    g_I = torch.tensor([0, 0, -g]).view(1, 3, 1).expand(bs, 3, 1).type(x.type())
    
    # Upper part: no direct gravity effect on kinematic states
    zeros_61 = torch.zeros(bs, 6, 1).type(x.type())
    
    # Lower part: gravity effects on dynamic states
    # Each cable constraint gets contribution from payload gravity distributed equally
    # Plus each quadrotor's own gravity through the cable constraint
    G1 = torch.bmm(B_1_T, -m_p/3 * g_I)  # Payload portion + Quadrotor 1 mass
    G2 = torch.bmm(B_2_T, -m_p/3 * g_I)  # Payload portion + Quadrotor 2 mass
    G3 = torch.bmm(B_3_T, -m_p/3 * g_I)  # Payload portion + Quadrotor 3 mass

    F_g = torch.cat([zeros_61, G1, G2, G3], dim=1)
    
    # Dynamics
    f = torch.zeros(bs, num_dim_x, 1).type(x.type())
    
    # Kinematic equations (upper 12 states)
    motion_vel = torch.cat([v_p, omega_p, v_q1, v_q2, v_q3], dim=1)  # (bs, 12, 1)
    vel = torch.cat([v_p, euler_dot_p, v_q1, v_q2, v_q3], dim=1)  # (bs, 12, 1)
    f[:, 0:12, 0] = vel.squeeze(-1)
    
    # Dynamic equations (lower 12 states)
    dynamics_term = torch.linalg.solve(M, F_g - torch.bmm(C, motion_vel))
    f[:, 12:24, 0] = dynamics_term.squeeze(-1)
    return f


def DfDx_func(x):
    raise NotImplemented('NotImplemented')


def B_func(x):
    """Control input matrix (equivalent to gx from original class)"""
    bs = x.shape[0]
    
    # Extract states needed for B matrices
    phi, theta, psi = [x[:, i, 0] for i in range(3, 6)]
    r_1_x, r_1_y, r_2_x, r_2_y, r_3_x, r_3_y = [x[:, i, 0] for i in range(6, 12)]

    r_1 = torch.stack([r_1_x, r_1_y], dim=1).unsqueeze(-1)
    r_2 = torch.stack([r_2_x, r_2_y], dim=1).unsqueeze(-1)
    r_3 = torch.stack([r_3_x, r_3_y], dim=1).unsqueeze(-1)

    B_1_T = get_B(r_1, l[0]).transpose(1, 2)
    B_2_T = get_B(r_2, l[1]).transpose(1, 2)
    B_3_T = get_B(r_3, l[2]).transpose(1, 2)
    
    R = get_R(phi, theta, psi)
    R_T = R.transpose(1, 2)
    
    
    # Recompute mass matrix for control allocation
    t_expanded = t.unsqueeze(0).expand(bs, -1, -1).unsqueeze(-1).type(x.type())
    t_1_skew = skew_symmetric(t_expanded[:, :, 0, :])
    t_2_skew = skew_symmetric(t_expanded[:, :, 1, :])
    t_3_skew = skew_symmetric(t_expanded[:, :, 2, :])
    
    A = m_q[0] * t_1_skew + m_q[1] * t_2_skew + m_q[2] * t_3_skew
    A_T = A.transpose(1, 2)
    J_p_expanded = J_p.unsqueeze(0).expand(bs, -1, -1).type(x.type())
    J_q = -m_q[0] * torch.bmm(t_1_skew, t_1_skew) - m_q[1] * torch.bmm(t_2_skew, t_2_skew) - m_q[2] * torch.bmm(t_3_skew, t_3_skew)
    
    B_1 = get_B(r_1, l[0])
    B_2 = get_B(r_2, l[1])
    B_3 = get_B(r_3, l[2])
    
    # Rebuild mass matrix
    M = torch.zeros(bs, 12, 12).type(x.type())
    M11 = (m_p + sum(m_q)) * torch.eye(3).repeat(bs, 1, 1).type(x.type())
    M12 = torch.bmm(R, A_T)
    M13 = m_q[0] * B_1
    M14 = m_q[1] * B_2
    M15 = m_q[2] * B_3
    M22 = J_p_expanded + J_q
    M23 = m_q[0] * torch.bmm(t_1_skew, torch.bmm(R_T, B_1))
    M24 = m_q[1] * torch.bmm(t_2_skew, torch.bmm(R_T, B_2))
    M25 = m_q[2] * torch.bmm(t_3_skew, torch.bmm(R_T, B_3))
    M33 = m_q[0] * torch.bmm(B_1_T, B_1)
    M44 = m_q[1] * torch.bmm(B_2_T, B_2)
    M55 = m_q[2] * torch.bmm(B_3_T, B_3)
    
    zeros_22 = torch.zeros(bs, 2, 2).type(x.type())
    M = torch.cat([
        torch.cat([M11, M12, M13, M14, M15], dim=2),
        torch.cat([M12.transpose(1, 2), M22, M23, M24, M25], dim=2),
        torch.cat([M13.transpose(1, 2), M23.transpose(1, 2), M33, zeros_22, zeros_22], dim=2),
        torch.cat([M14.transpose(1, 2), M24.transpose(1, 2), zeros_22, M44, zeros_22], dim=2),
        torch.cat([M15.transpose(1, 2), M25.transpose(1, 2), zeros_22, zeros_22, M55], dim=2)
    ], dim=1)
    
    # Control input matrix H (12 x 9)
    # Create block diagonal matrix for control inputs
    H_upper = torch.zeros(bs, 3, 9).type(x.type())
    H_upper[:, 0:3, 0:3] = torch.eye(3).unsqueeze(0).expand(bs, -1, -1).type(x.type())  # Quad 1 forces
    H_upper[:, 0:3, 3:6] = torch.eye(3).unsqueeze(0).expand(bs, -1, -1).type(x.type())  # Quad 2 forces
    H_upper[:, 0:3, 6:9] = torch.eye(3).unsqueeze(0).expand(bs, -1, -1).type(x.type())  # Quad 3 forces

    H_lower = torch.zeros(bs, 9, 9).type(x.type())
    H_lower[:, 0:3, 0:3] = torch.bmm(t_1_skew, R_T)
    H_lower[:, 0:3, 3:6] = torch.bmm(t_2_skew, R_T)
    H_lower[:, 0:3, 6:9] = torch.bmm(t_3_skew, R_T)
    H_lower[:, 3:5, 0:3] = B_1_T
    H_lower[:, 5:7, 3:6] = B_2_T
    H_lower[:, 7:9, 6:9] = B_3_T
    
    H = torch.cat([H_upper, H_lower], dim=1)  # (bs, 12, 9)

    # B matrix: control input effect on full state
    control_effect = torch.linalg.solve(M, H)
    B = torch.cat([torch.zeros(bs, 12, num_dim_control).type(x.type()), control_effect], dim=1)
    return B


def DBDx_func(x):
    raise NotImplemented('NotImplemented')


def Bbot_func(B):
    # Compute Bbot
    Bbot = []
    for i in range(B.shape[0]):
        gi = B[i]  # num_dim_x x num_dim_control
        # SVD: Bi = U S Vh
        U, S, Vh = torch.linalg.svd(gi, full_matrices=True)
        # Null space: columns of U corresponding to zero singular values
        # For numerical stability, use a tolerance
        tol = 1e-7
        null_mask = S < tol
        if null_mask.sum() == 0:
            # If no exact zeros, take the last (n-m) columns of U
            Bbot_i = U[:, num_dim_control:]
        else:
            Bbot_i = U[:, null_mask]
        Bbot.append(Bbot_i)
    # Stack to shape bs x n x (n-m)
    Bbot = torch.stack(Bbot, dim=0).type(B.type())
    return Bbot