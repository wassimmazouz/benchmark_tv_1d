import numpy as np
from benchopt import BaseSolver


class Solver(BaseSolver):
    """sythetic"""
    name = 'ISTA'

    stopping_strategy = "callback"
    # any parameter defined here is accessible as a class attribute

    def set_objective(self, A, reg, reg_max, y):
        self.reg, self.reg_max = reg, reg_max
        self.A, self.y = A, y

    def run(self, callback):
        len_y = len(self.y)
        L = np.tri(len_y)
        AL = self.A @ L
        stepsize = 1 / (np.linalg.norm(AL, ord=2)**2)  # 1/ rho
        reg_tot = self.reg*self.reg_max
        z = np.zeros(len_y)  # initialisation
        while callback(L.dot(z)):
            z = self.st(z - stepsize * AL.T @ (AL @ z - self.y),
                        reg_tot * stepsize)
        self.x = L.dot(z)

    def get_result(self):
        return self.x

    def st(self, w, mu):
        w0 = w[0]
        w -= np.sign(w) * abs(np.clip(w, -mu, mu))
        w[0] = w0
        return w
