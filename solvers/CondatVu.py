from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientDescentCriterion
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    get_l2norm = import_ctx.import_from('shared', 'get_l2norm')
    st = import_ctx.import_from('shared', 'st')
    grad_huber = import_ctx.import_from('shared', 'grad_huber')


class Solver(BaseSolver):
    """Primal-Dual Splitting Method for analysis formulation."""
    name = 'CondatVu analysis'

    stopping_criterion = SufficientDescentCriterion(
        patience=3, strategy="callback"
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {'ratio': [1.],
                  'eta': [1.]}

    def set_objective(self, A, reg, y, c, delta, data_fit):
        self.reg = reg
        self.A, self.y = A, y
        self.c = c
        self.delta = delta
        self.data_fit = data_fit

    def run(self, callback):
        n, p = self.A.shape
        # Block preconditioning (2x2)
        LD = 2.0  # Lipschitz constant associated to D (only for 1d!!)
        LA = get_l2norm(self.A)
        sigma_v = 1.0 / (self.ratio * LD)
        tau = 1 / (LA ** 2 / 2 + sigma_v * LD ** 2)
        eta = self.eta
        # initialisation
        u = self.c * np.ones(p)
        v = np.zeros(p - 1)

        while callback(u):
            u_tmp = (u - tau * self.grad(self.A, u)
                     - tau * (-np.diff(v, append=0, prepend=0)))
            v_tmp = (v + sigma_v * np.diff(2 * u_tmp - u)
                     - sigma_v * st(v / sigma_v +
                                    np.diff(2 * u_tmp - u),
                                    self.reg / sigma_v))
            u = eta * u_tmp + (1 - eta) * u
            v = eta * v_tmp + (1 - eta) * v
        self.u = u

    def get_result(self):
        return self.u

    def grad(self, A, u):
        R = self.y - A @ u
        if self.data_fit == 'quad':
            return - A.T @ R
        else:
            return - A.T @ grad_huber(R, self.delta)
