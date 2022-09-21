import torch.nn as nn
from einops import rearrange
from scipy.sparse import csr_matrix


class SpyritNet(nn.Module):
    def __init__(self, F_O, DC_layer, Denoi):
        super().__init__()
        self.F_O = F_O
        self.DC_layer = DC_layer  # must be Tikhonov solve
        self.Denoi = Denoi

    @staticmethod
    def wrap(x, n: int):
        return ((x + n) % (2 * n)) - n

    def F_Orward(self, x, DPower):
        _, c, h, w = x.shape
        A1, A2 = self.F_O.get_diff_matrix()
        # W =
        # x = A1.transpose().dot(csr_matrix()
        x = self.F_Orward_tikh(x)
        x = self.Denoi(x)  # shape stays the same
        x = rearrange(x, "(b c) () (h w) -> b c h w", c=c, h=h)

        return x

    def F_Orward_tikh(self, x, DPower):
        # x - of shape [b,c,h,w]
        _, c, h, w = x.shape
        x = rearrange(x, "b c h w ->  (b c) (h w)")
        # Acquisition
        x = self.F_O(x)  # shape x = [b*c,h*w]
        x = self.reconstruct_tick(x, h)

        return x

    def reconstruct_tick(self, x, DPower, h):
        # Data consistency layer
        # measurements to the image domain
        x = self.DC_layer(x, torch.zeros_like(x), self.F_O)  # shape x = [b*c, N]

        # Image-to-image mapping via convolutional networks
        # Image domain denoising
        x = rearrange(x, "(b c) (h w) -> (b c) () h w", h=h)

        return x

    def reconstruct(self, x, DPower, c, h):
        x = self.reconstruct_tick(x, h)
        x = self.Denoi(x)  # shape stays the same
        x = rearrange(x, "(b c) () (h w) -> b c h w", c=c, h=h)
        return x


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import pyrootutils
    import torch
    from monai.data import DataLoader
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from torchvision import datasets, transF_Orms

    from covid.models.components.spyrit_related.utils import (
        F_Orward_operator,
        Tikhonov_solve,
        Unet,
    )

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    n_x = 87
    n_y = 87
    bs = 10  # Batch size
    data_dir = root / "data" / "STL_10"

    def imagesc(Img, title="", colormap=plt.cm.gray):
        """imagesc(IMG) Display image Img with scaled colors with greyscale colormap and colorbar
        imagesc(IMG, title=ttl) Display image Img with scaled colors with greyscale colormap and
        colorbar, with the title ttl imagesc(IMG, title=ttl, colormap=cmap) Display image Img with
        scaled colors with colormap and colorbar specified by cmap (choose between 'plasma', 'jet',
        and 'grey'), with the title ttl."""
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.imshow(Img, cmap=colormap)
        plt.title(title)
        divider = make_axes_locatable(ax)
        cax = plt.axes([0.85, 0.1, 0.075, 0.8])
        plt.colorbar(cax=cax)
        plt.show()

    # %% A batch of STL-10 test images
    torch.manual_seed(7)

    transF_Orm = transF_Orms.Compose(
        [
            transF_Orms.functional.to_grayscale,
            transF_Orms.Resize((n_x, n_y)),
            transF_Orms.ToTensor(),
            transF_Orms.Normalize([0.5], [0.5]),
        ]
    )

    testset = datasets.STL10(root=data_dir, split="test", download=False, transF_Orm=transF_Orm)
    testloader = DataLoader(testset, batch_size=bs, shuffle=False)

    # %%
    inputs, _ = next(iter(testloader))
    b, c, h, w = inputs.shape

    z = inputs.view(b * c, w * h)

    x = z[5, :]
    x = x.numpy()
    imagesc(x.reshape((w, h)))

    # %% data
    # Build A
    A1 = -np.eye(h) + np.eye(h, h, 1)
    A1[-1, -1] = 1
    A1 = np.kron(A1, np.eye(w))

    A2 = -np.eye(w) + np.eye(w, w, 1)
    A2[-1, -1] = 1
    A2 = np.kron(np.eye(h), A2)

    W = np.zeros((w * h, w * h))

    np.fill_diagonal(W, np.ones((h, w)).flatten())

    A1 = csr_matrix(A1)
    A2 = csr_matrix(A2)
    W = csr_matrix(W)

    A = (A1.transpose().dot(W).dot(A1) + A2.transpose().dot(W).dot(A2)).toarray()

    # A = np.matmul(np.matmul(A1.transpose(), W), A1) + np.matmul(np.matmul(A2.transpose(), W), A2)

    # A = A1.transpose() @ W @ A1 + A2.transpose() @ W @ A2

    y = A @ x

    imagesc(y.reshape((w, h)))

    # %%
    F_O = F_Orward_operator(A)  # F_Orward operator

    DC_layer = Tikhonov_solve(mu=0.01)
    Denoi = Unet()
    model = SpyritNet(F_O, DC_layer, Denoi)

    # Bruit ???

    # %%
    m = F_O(z)
    imagesc(m[5, :].numpy().reshape((w, h)))

    m_rec = model.F_Orward_tikh(x)
    imagesc(m_rec[5, :].numpy().reshape((w, h)))
