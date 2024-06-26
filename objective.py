from benchopt import BaseObjective
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy import optimize
    from benchmark_utils.shared import huber
    from benchmark_utils.shared import grad_huber
    import torch
    import deepinv as dinv
    from deepinv.optim.data_fidelity import L1


class Objective(BaseObjective):
    name = "TV1D"
    min_benchopt_version = "1.5"

    parameters = {
        'reg': [0.5],
        'delta': [0.9],
        'data_fit': ['quad', 'huber']
    }

    def linop(self, x):
        if torch.cuda.is_available():
            device = dinv.utils.get_freer_gpu()
        else:
            device = 'cpu'

        physics = dinv.physics.Inpainting(
            tensor_size=x.shape[1:],
            mask=0.5,
            device=device
        )
        physics.noise_model = dinv.physics.UniformNoise(a=0)
        return physics(x)

    def set_data(self, A, y, x):
        self.A, self.y, self.x = A, y, x
        if self.A != 0:
            S = self.A @ np.ones(self.A.shape[1])
        else:
            S = self.linop(np.ones(self.x.shape[0]))
        self.c = self.get_c(S, self.delta)
        self.reg_scaled = self.reg*self.get_reg_max(self.c)

    def evaluate_result(self, u):
        if self.A != 0:
            R = self.y - self.A @ u
        else:
            R = self.y - self.linop(u)

        reg_TV = abs(np.diff(u)).sum()
        if self.data_fit == 'quad':
            loss = .5 * R @ R
        elif self.data_fit == 'huber':
            loss = huber(R, self.delta)
        norm_x = np.linalg.norm(u - self.x)

        return dict(value=loss + self.reg_scaled * reg_TV, norm_x=norm_x)

    def get_one_result(self):
        return dict(u=np.zeros(self.x.shape[0]))

    def get_objective(self):
        return dict(A=self.A, reg=self.reg_scaled, y=self.y, c=self.c,
                    delta=self.delta, data_fit=self.data_fit)

    def get_c(self, S, delta):
        if self.data_fit == 'quad':
            return (S @ self.y)/(S @ S)
        else:
            return self.c_huber(S, delta)

    def c_huber(self, S, delta):
        def f(c):
            R = self.y - S * c
            return abs((S * grad_huber(R, delta)).sum())
        yS = self.y / S
        return optimize.golden(f, brack=(min(yS), max(yS)))

    def get_reg_max(self, c):
        L = np.tri(self.x.shape[0])
        if self.A != 0:
            AL = self.A @ L
        else:
            AL = self.linop(L)
        z = np.zeros(self.x.shape[0])
        z[0] = c
        return np.max(abs(self.grad(AL, z)))

    def grad(self, A, u):
        if A == 0:
            if torch.cuda.is_available():
                device = dinv.utils.get_freer_gpu()
            else:
                device = 'cpu'

            physics = dinv.physics.Inpainting(
                tensor_size=u.shape[1:],
                mask=0.5,
                device=device
            )
            physics.noise_model = dinv.physics.UniformNoise(a=0)
            data_fidelity = L1()
            return data_fidelity.grad(self.linop(u), self.y, physics)
        R = self.y - A @ u
        if self.data_fit == 'quad':
            return - A.T @ R
        else:
            return - A.T @ grad_huber(R, self.delta)
