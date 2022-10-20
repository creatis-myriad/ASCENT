import torch.nn as nn
from einops import rearrange
from scipy.sparse import csr_matrix


class SpyritNet(nn.Module):
    def __init__(self, Fwd_OP, DC_layer, Denoi):
        super().__init__()
        self.Fwd_OP = Fwd_OP
        self.DC_layer = DC_layer  # must be Tikhonov solve
        self.Denoi = Denoi

    def forward(self, x, W):
        x = self.foward_tikh(x, W)
        x = self.Denoi(x)
        return x

    def forward_tikh(self, x, W):
        # Acquisition
        x = self.reconstruct_tick(x, W)
        return x

    def reconstruct(self, x, W):
        x = self.reconstruct_tick(x, W)
        # Image-to-image mapping via convolutional networks
        # Image domain denoising
        x = self.Denoi(x)  # shape stays the same
        return x

    def reconstruct_tick(self, x, W):
        # Data consistency layer
        # measurements to the image domain
        x = self.DC_layer(x, torch.zeros_like(x), W, self.Fwd_OP)
        return x


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import pyrootutils
    import torch
    from monai.data import DataLoader
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from torchvision import datasets, transforms

    from ascent.models.components.spyrit_related.utils import (
        Forward_operator,
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

    transform = transforms.Compose(
        [
            transforms.functional.to_grayscale,
            transforms.Resize((n_x, n_y)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    testset = datasets.STL10(root=data_dir, split="test", download=False, transform=transform)
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
    F_O = Forward_operator(A)  # Forward operator

    DC_layer = Tikhonov_solve(mu=0.01)
    Denoi = Unet()
    model = SpyritNet(F_O, DC_layer, Denoi)

    # Bruit ???

    # %%
    m = F_O(z)
    imagesc(m[5, :].numpy().reshape((w, h)))

    m_rec = model.Forward_tikh(x)
    imagesc(m_rec[5, :].numpy().reshape((w, h)))
