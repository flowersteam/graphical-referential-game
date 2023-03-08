import torch
import numpy as np

from utils import device, path

##### Canonical System #########################################################
class CanonicalSystem:
    def __init__(self, a, T, dt):
        self.a = a
        self.T = T
        self.dt = dt

        self.time = torch.arange(0, T, dt)
        self.N = self.time.shape[0]

        self.theta = None
        self.reset()

    def reset(self):
        self.theta = 1.0

    def step(self, tau=1.0):
        self.theta = self.theta - self.a * self.dt * self.theta / tau
        return self.theta

    def all_steps(self, tau=1.0):
        return torch.tensor([self.step(tau) for _ in range(self.N)])

##### DMP ######################################################################
class DMP:
    def __init__(self, T, dt, a=150, b=25, n_bfs=10, w=None, s=None, g=None):
        self.T = T
        self.dt = dt
        self.y0 = s
        self.g  = g
        self.a  = a
        self.b  = b
        self.n_bfs = n_bfs

        # canonical system
        a = 1.0
        self.cs = CanonicalSystem(a, T, dt)

        # initialize basis functions for LWR
        self.w = w
        self.centers = None
        self.widths = None
        self.set_basis_functions()

        # executed trajectory
        self.y = None
        self.yd = None
        self.z = None
        self.zd = None

        # desired path
        self.path = []

        self.reset()


    def reset(self):
        self.y = self.y0  # .copy()
        self.path = [self.y0]

        self.yd = 0.0
        self.z = 0.0
        self.zd = 0.0
        self.cs.reset()

    def fit(self, y_demo, tau=1.0):
        self.path = y_demo
        self.y0 = y_demo[0].copy()
        self.g = y_demo[-1].copy()

        y_demo = interpolate_path(self, y_demo)
        yd_demo, ydd_demo = calc_derivatives(y_demo, self.dt)

        f_target = tau**2 * ydd_demo - self.a * (self.b * (self.g - y_demo) - tau * yd_demo)
        f_target /= (self.g - self.y0)

        theta_seq = self.cs.all_steps()
        psi_funs = self.psi(theta_seq.to(device)).cpu().numpy()

        theta_seq = theta_seq.numpy()


        # Locally Weighted Regression
        aa = np.multiply(theta_seq.reshape((1, theta_seq.shape[0])), psi_funs.T)
        aa = np.multiply(aa, f_target.reshape((1, theta_seq.shape[0])))
        aa = np.sum(aa, axis=1)

        bb = np.multiply(theta_seq.reshape((1, theta_seq.shape[0])) ** 2, psi_funs.T)
        bb = np.sum(bb, axis=1)
        self.w = torch.from_numpy(aa / bb).to(device).float().unsqueeze(0)

        self.reset()

    def set_basis_functions(self):
        time = torch.linspace(0, self.T, self.n_bfs).to(device)
        self.centers = torch.zeros(self.n_bfs).to(device)
        self.centers = torch.exp(-self.cs.a * time).to(device)
        self.widths = torch.ones(self.n_bfs).to(device) * self.n_bfs ** 1.5 / self.centers / self.cs.a

    def psi(self, theta):
        return torch.exp(-self.widths * (theta - self.centers) ** 2)

    def step(self, tau=1.0, k=1.0, start=None, goal=None):
        g=self.g
        y0=self.y0

        theta = self.cs.step(tau)
        psi = self.psi(theta)

        f = torch.matmul(self.w, psi) * theta * k * (g - y0) / torch.sum(psi).item()
        #f = torch.matmul(self.w, psi).dot(g - y0) * theta * k / torch.sum(psi)

        self.zd = self.a * (self.b * (g - self.path[-1]) - self.z) + f  # transformation system
        self.zd /= tau

        self.z += self.zd * self.dt

        self.yd = self.z / tau
        #self.y += self.yd * self.dt

        self.path.append(self.path[-1] + self.yd * self.dt)
        return self.path[-1], self.yd, self.z, self.zd

    def run_sequence(self, tau=1.0, k=1.0, start=None, goal=None):
        y = torch.zeros(self.cs.N).to(device)
        y[0] = self.y0
        for i in range(self.cs.N):
            y[i], _, _, _ = self.step(tau=tau, k=k, start=start, goal=goal)
        return y
