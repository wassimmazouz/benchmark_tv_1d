from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import deepinv as dinv
    import torch
    from deepinv.optim.data_fidelity import L2


class Solver(BaseSolver):
    name = 'deepinv'

    parameters = {
        'reg': [0.1],
        'gamma': [1],
        'max_iter': [50]
    }

    def set_objective(self, A, reg, y, c, delta, data_fit):
        self.A, self.reg, self.y, self.c, self.deta, self.data_fit = A, reg, torch.from_numpy(y), c, delta, data_fit

    def run(self, n_iter):
        device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
        x, y = self.x, self.y
        gamma, reg = self.gamma, self.reg
        x2 = x.clone().to(device)
        data_fidelity = L2()
        prior = dinv.optim.TVPrior()

        physics = dinv.physics.Inpainting(
            tensor_size=x.shape[1:],
            mask=0.5,
            device=device
            )
        physics.noise_model = dinv.physics.GaussianNoise(sigma=0.2)
        for _ in range(50):
            x2 = x2 - gamma*data_fidelity.grad(x2, y, physics)
            x2 = prior.prox(x2,  gamma=gamma*reg)
        self.out = x2.clone()

    def get_result(self):
        return dict(x=self.out)
