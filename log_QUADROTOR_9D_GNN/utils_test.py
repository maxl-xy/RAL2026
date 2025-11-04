import contextlib
import numpy as np

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

def EulerIntegrate(controller, f, B, xstar, ustar, xinit, t_max = 10, dt = 0.05, with_tracking = False, sigma = 0., noise_bound = None):
    t = np.arange(0, t_max, dt)

    trace = []
    u = []

    xcurr = xinit
    trace.append(xcurr)

    for i in range(len(t)):
        if with_tracking:
            xe = xcurr - xstar[i]
        ui = controller(xcurr, xe, ustar[i]) if with_tracking else ustar[i]
        if with_tracking:
            # print(xcurr.reshape(-1), xstar[i].reshape(-1), ui.reshape(-1))
            pass

        if not noise_bound:
            noise_bound = 3 * sigma
        noise = np.random.randn(*xcurr.shape) * sigma
        noise[noise>noise_bound] = noise_bound
        noise[noise<-noise_bound] = -noise_bound

        dx = f(xcurr) + B(xcurr).dot(ui) + noise
        xnext =  xcurr + dx*dt
        # xnext[xnext>100] = 100
        # xnext[xnext<-100] = -100

        trace.append(xnext)
        u.append(ui)
        xcurr = xnext
    return trace, u

def RK4(controller, f, B, xstar, ustar, xinit, t_max = 10, dt = 0.05, with_tracking = False, sigma = 0., noise_bound = None):
    t = np.arange(0, t_max, dt)

    trace = []
    u = []

    xcurr = xinit
    trace.append(xcurr)

    for i in range(len(t)):
        if with_tracking:
            xe = xcurr - xstar[i]
        ui = controller(xcurr, xe, ustar[i]) if with_tracking else ustar[i]
        if with_tracking:
            #  print(xcurr.reshape(-1), xstar[i].reshape(-1), ui.reshape(-1))
            pass

        if not noise_bound:
            noise_bound = 3 * sigma
        noise = np.random.randn(*xcurr.shape) * sigma
        noise[noise>noise_bound] = noise_bound
        noise[noise<-noise_bound] = -noise_bound
        
        k1 = f(xcurr) + B(xcurr).dot(ui)
        ui_k1 = controller(xcurr + 0.5*dt*k1, xe, ustar[i]) if with_tracking else ustar[i]
        k2 = f(xcurr + 0.5*dt*k1) + B(xcurr + 0.5*dt*k1).dot(ui_k1)
        ui_k2 = controller(xcurr + 0.5*dt*k2, xe, ustar[i]) if with_tracking else ustar[i]
        k3 = f(xcurr + 0.5*dt*k2) + B(xcurr + 0.5*dt*k2).dot(ui_k2)
        ui_k3 = controller(xcurr + dt*k3, xe, ustar[i]) if with_tracking else ustar[i]
        k4 = f(xcurr + dt*k3) + B(xcurr + dt*k3).dot(ui_k3)

        dx = (k1 + 2*k2 + 2*k3 + k4) / 6
        xnext = xcurr + dx*dt
        # xnext[xnext>100] = 100
        # xnext[xnext<-100] = -100

        trace.append(xnext)
        u.append(ui)
        xcurr = xnext
    return trace, u