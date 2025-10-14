import numpy as np
from utils import temp_seed

m_p = 1.3  # payload mass
m_q = [1.5, 1.5, 1.5]
g = 9.81  # gravity

# for training
X_REF = np.array([0., 0., 0.,  # payload position
                  0., 0., 0.,  # payload Euler angles
                  0., 0.,  # cable 1 position
                  0., 0.,  # cable 2 position
                  0., 0.,  # cable 3 position
                  0., 0., 0.,  # payload velocity
                  0., 0., 0.,  # payload angular velocity
                  0., 0.,  # cable 1 velocity
                  0., 0.,  # cable 2 velocity
                  0., 0.  # cable 3 velocity
                  ]).reshape(-1,1)

r_j_proj_x = 0.5
r_j_proj_y = 0.5
euler_bound = np.radians(30)  # Euler angle bounds
yaw_bound = np.radians(30)  # Yaw angle bounds
X_MIN = np.array([
                -10., -10., -10.,  # payload position bounds
                -euler_bound, -euler_bound, -yaw_bound,  # payload Euler angles bounds
                -r_j_proj_x, -r_j_proj_y,
                -r_j_proj_x, -r_j_proj_y,
                -r_j_proj_x, -r_j_proj_y,  # cable projection bounds
                -2.5, -2.5, -2.5,  # payload velocity bounds
                -0.5, -0.5, -0.5,  # payload angular velocity bounds
                -0.5, -0.5, 
                -0.5, -0.5, 
                -0.5, -0.5  # cable velocity bounds
            ]).reshape(-1,1)

X_MAX = np.array([
                10., 10., 10.,  # payload position bounds
                euler_bound, euler_bound, yaw_bound,  # payload Euler angles bounds
                r_j_proj_x, r_j_proj_y,
                r_j_proj_x, r_j_proj_y,
                r_j_proj_x, r_j_proj_y,  # cable projection bounds
                2.5, 2.5, 2.5,  # payload velocity bounds
                0.5, 0.5, 0.5,  # payload angular velocity bounds
                0.5, 0.5, 
                0.5, 0.5, 
                0.5, 0.5  # cable velocity bounds
            ]).reshape(-1,1)

f_bound = 2.
UREF_MIN = np.array([-f_bound, -f_bound, -f_bound, -f_bound, -f_bound, -f_bound, -f_bound, -f_bound, -f_bound]).reshape(-1,1)
UREF_MAX = np.array([f_bound, f_bound, f_bound, f_bound, f_bound, f_bound, f_bound, f_bound, f_bound]).reshape(-1,1)

lim = 1

XE_MIN = np.array([-lim, -lim, -lim, 
                   -lim/20, -lim/20, -lim/20,
                   -lim/10, -lim/10, 
                   -lim/10, -lim/10, 
                   -lim/10, -lim/10, 
                   -lim/5, -lim/5, -lim/5, 
                   -lim/20, -lim/20, -lim/20,
                   -lim/10, -lim/10,
                   -lim/10, -lim/10, 
                   -lim/10, -lim/10]).reshape(-1,1)

XE_MAX = np.array([lim, lim, lim, 
                   lim/20, lim/20, lim/20,
                   lim/10, lim/10, 
                   lim/10, lim/10, 
                   lim/10, lim/10, 
                   lim/5, lim/5, lim/5, 
                   lim/20, lim/20, lim/20,
                   lim/10, lim/10,
                   lim/10, lim/10, 
                   lim/10, lim/10]).reshape(-1,1)


# for sampling ref
X_INIT_MIN = X_REF.flatten()
X_INIT_MAX = X_REF.flatten()

XE_INIT_MIN = 0.5*XE_MIN.flatten()
XE_INIT_MAX = 0.5*XE_MAX.flatten()

time_bound = 36.
time_step = 0.03
t = np.arange(0, time_bound, time_step)

def system_reset(seed):
    SEED_MAX = 10000000
    with temp_seed(int(seed * SEED_MAX)):
        xref_0 = X_INIT_MIN + np.random.rand(len(X_INIT_MIN)) * (X_INIT_MAX - X_INIT_MIN)
        xe_0 = XE_INIT_MIN + np.random.rand(len(XE_INIT_MIN)) * (XE_INIT_MAX - XE_INIT_MIN)
        x_0 = xref_0 + xe_0

        freqs = list(range(1, 11))
        # freqs = []
        weights = np.random.randn(len(freqs), len(UREF_MIN))
        weights = (2. * weights / np.sqrt((weights**2).sum(axis=0, keepdims=True))).tolist()
        uref = []
        for _t in t:
            u = np.array([0, 0, 0, 
                          0, 0, 0, 
                          0, 0, 0])
            # for freq, weight in zip(freqs, weights):
            #     u += np.array([0.1*weight[0] * np.sin(freq * _t/time_bound * 2*np.pi), 0.1*weight[1] * np.sin(freq * _t/time_bound * 2*np.pi), 
            #                    0.1*weight[2] * np.sin(freq * _t/time_bound * 2*np.pi), 0.1*weight[3] * np.sin(freq * _t/time_bound * 2*np.pi),
            #                    0.1*weight[4] * np.sin(freq * _t/time_bound * 2*np.pi), 0.1*weight[5] * np.sin(freq * _t/time_bound * 2*np.pi)])
            # u += 0.01*np.random.randn(6)
            uref.append(u)

    return x_0, xref_0, uref