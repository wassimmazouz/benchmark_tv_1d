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

    def skip(self, A, reg, y, c, delta, data_fit):
        if data_fit == 'huber':
            return True, "solver does not work with huber loss"
        return False, None

    def set_objective(self, A, reg, y, c, delta, data_fit):
        self.A, self.reg, self.y = A, reg, torch.from_numpy(y)
        self.c, self.deta, self.data_fit = c, delta, data_fit

    def run(self, n_iter):
        if torch.cuda.is_available():
            device = dinv.utils.get_freer_gpu()
        else:
            device = 'cpu'

        y = self.y
        gamma, reg = self.gamma, self.reg
        x = y.clone().to(device)
        data_fidelity = L2()
        prior = dinv.optim.TVPrior()

        physics = dinv.physics.Inpainting(
            tensor_size=x.shape[1:],
            mask=0.5,
            device=device
        )
        physics.noise_model = dinv.physics.GaussianNoise(sigma=0.2)
        for _ in range(50):
            x = x - gamma*data_fidelity.grad(x, y, physics)
            x = prior.prox(x,  gamma=gamma*reg)
        self.out = x.clone()

    def get_result(self):
        return dict(x=self.out.numpy())
