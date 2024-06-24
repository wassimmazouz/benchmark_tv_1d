from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import deepinv as dinv
    import torch
    import numpy as np


class Dataset(BaseDataset):

    name = "chateau"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
    }

    def get_data(self):
        device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

        url = (
            "https://culturezvous.com/wp-content/uploads/2017/10/chateau-azay-le-rideau.jpg?download=true"
        )
        x = dinv.utils.load_url_image(url=url, img_size=100).to(device)
        tensor_size = x.shape[1:]
        operator = physics = dinv.physics.Inpainting(
            tensor_size=tensor_size,
            mask=0.5,
            device=device
        )
        physics.noise_model = dinv.physics.GaussianNoise(sigma=0.2)

        A = np.diag(100*[0.5]) #Inpainting Matrix

        y = operator(x)

        return dict(A=A.numpy(), y=y.numpy(), x=x.numpy())
