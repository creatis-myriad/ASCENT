from typing import Union

import numpy as np
import torch
from einops import rearrange
from monai.data import MetaTensor
from scipy.sparse import csr_matrix, identity, kron
from torch import Tensor, nn

from ascent.utils.tensor_utils import reshape_fortran, round_differentiable


class Robust2DUnwrap(nn.Module):
    """Robust 2D phase unwrapping in PyTorch.

    Instead of using weights like the paper did, the weight is directly multiplied with the input to
    create a weighted input. This avoids the reconstruction of A matrix (which was A1'@W@A1 + A2'@W@A2)
    during the forward pass since the weight varies for each input.

    Reference:
        D. C. Ghiglia and L. A. Romero. "Robust two-dimensional weighted and unweighted
        phase unwrapping that uses fast transforms and iterative methods". Journal of the Optical
        Society of America A 1994.
    """

    def __init__(
        self,
        shape: Union[list[int, int], tuple[int, int]],
        mu: float = 0.1,
        wrap_param: float = 1.0,
        normalize: bool = False,
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        """Define the sparse finite difference matrices (for faster matrix multiplication) and
        fully connected layers.

        Args:
            shape: Shape of the input tensor.
            mu: Initial regularization weight.
            wrap_param: Wrapping parameter.
            normalize: Whether to normalize the wrapped tensor between -1 and 1.
            in_channels: Number of input channels.
        """
        # learnable weight for regularization
        self.mu = nn.Parameter(torch.tensor([float(mu)], requires_grad=True))

        self.M = shape[0]
        self.N = shape[1]

        self.wrap_param = wrap_param
        self.normalize = normalize
        self.in_channels = in_channels

        # create sparse differentiation matrix; A1, A2 = (m*n, m*n) sparse array
        self.A1 = kron(identity(self.N, format="csr"), self.differentiation_matrix(self.M))
        self.A2 = kron(self.differentiation_matrix(self.N), identity(self.M, format="csr"))

        # build A matrix: A = A1' @ A1 + A2' @ A2 + mu * I
        A = torch.from_numpy(
            (self.A1.transpose().dot(self.A1) + self.A2.transpose().dot(self.A2)).toarray()
        ) + self.mu * torch.from_numpy(identity(self.M * self.N, format="csr").toarray())

        # fully connected layers that take the weight equalling to the forward operator A = (m*n, m*n)
        self.A = nn.Linear(self.M * self.N, self.M * self.N, False)
        self.A.weight.requires_grad = False
        self.A.weight.data = A.float()

        # adjoint of A
        self.A_adj = nn.Linear(self.M * self.N, self.M * self.N, False)
        self.A_adj.weight.requires_grad = False
        self.A_adj.weight.data = torch.t(A.float())

        # delete A array to free up space
        del A

    @staticmethod
    def wrap(
        x: Union[Tensor, MetaTensor], wrap_param: float = 1.0, normalize: bool = False
    ) -> Union[Tensor, MetaTensor]:
        """Wrap any element with its absolute value surpassing the wrapping parameter.

        Args:
            x: Input tensor.
            wrap_param: Wrapping parameter.
            normalize: Whether to normalize the wrapped tensor between -1 and 1.

        Returns:
            Wrapped tensor.
        """
        x = (x + wrap_param) % (2 * wrap_param) - wrap_param
        if normalize:
            return x / wrap_param
        else:
            return x

    def preprocess(self, x: Union[Tensor, MetaTensor]) -> Union[Tensor, MetaTensor]:
        """Preprocess the input tensor by calculating its differentiation along the horizontal and
        vertical axes.

        Args:
            x: Input tensor.
            wrap_param: Wrapping parameter.
            normalize: Whether to normalize the wrapped tensor between -1 and 1.

        Returns:
            Preprocessed input tensor.
        """
        b, c, m, n = x.shape

        # finite difference matrix along vertical axis
        d1y = csr_matrix(
            self.wrap(
                self.differentiation_matrix(self.M).dot(
                    rearrange(x.cpu().detach().numpy(), "b c m n -> m (n b c)")
                ),
                self.wrap_param,
                self.normalize,
            )
        )

        # Fortran-style flatten
        d1y = reshape_fortran(d1y.toarray(), (self.M * self.N, b * c))

        # finite difference matrix along horizontal axis
        d1x = csr_matrix(
            self.wrap(
                (
                    csr_matrix(rearrange(x.cpu().detach().numpy(), "b c m n -> (b c m) n")).dot(
                        self.differentiation_matrix(self.N).transpose()
                    )
                ).toarray(),
                self.wrap_param,
                self.normalize,
            )
        )
        d1x = reshape_fortran(d1x.toarray(), (self.M * self.N, b * c))

        # compute A1' @ d1y + A2' @ d1x + mu * x
        return rearrange(
            torch.from_numpy(self.A1.transpose().dot(d1y) + self.A2.transpose().dot(d1x))
            .float()
            .to(x.device.index),
            "(m n) (b c) -> (b c) (m n)",
            b=b,
            c=c,
            m=m,
        ) + self.mu * reshape_fortran(x, (b * c, m * n))

    @staticmethod
    def differentiation_matrix(k: int) -> csr_matrix:
        """Build a sparse finite difference matrix with -1 and 1.

        Args:
            k: Dimension of square matrix to create.

        Returns:
            CSR sparse finite difference matrix.
        """
        m = -np.eye(k) + np.eye(k, k, 1)
        m[-1, -1] = 1
        m[-1, -2] = -1
        return csr_matrix(m)

    def forward(self, x: Union[Tensor, MetaTensor]) -> Union[Tensor, MetaTensor]:
        """Solve the linear inverse problem for phase unwrapping.

        Args:
            x: Weighted input tensor (b, c, m, n).

        Returns:
            Unwrapped tensor (b, c, m, n).
        """
        b, c, m, n = x.shape

        # preprocess x
        x = self.preprocess(x)  # x = (b*c, m*n)

        # Fortran-style reshape to form channel * batch last tensor, x = (m*n, b*c)
        x = reshape_fortran(x, (m * n, b * c))

        # obtain the solution of a square system of linear equations with a unique solution using
        # pytorch.linalg.solve
        x = torch.linalg.solve(self.mat(), x)  # x = (m*n, b*c)

        # Fortran-style reshape x back to its initial dimension
        x = reshape_fortran(x, (b, c, m, n))

        return x

    def mat(self) -> Tensor:
        """Return the weight of the fully connected layers A.

        Returns:
            Weight of the fully connected layers A.
        """
        return self.A.weight.data


if __name__ == "__main__":
    import hydra
    import pyrootutils
    from hydra import compose, initialize_config_dir
    from lightning.pytorch.trainer.states import TrainerFn
    from matplotlib import pyplot as plt
    from monai.transforms import Compose, EnsureChannelFirstd
    from omegaconf import OmegaConf

    from ascent.datamodules.components.transforms import LoadNpyd
    from ascent.utils.visualization import dopplermap, imagesc

    root = pyrootutils.setup_root(
        search_from=__file__,
        indicator=[".git", "pyproject.toml"],
        pythonpath=True,
        dotenv=True,
    )

    # load specific file
    data_path = str(
        root / "data" / "UNWRAPV2" / "preprocessed" / "data_and_properties" / "Dealias_0022.npy"
    )
    transforms = Compose(
        [
            LoadNpyd(keys=["data"], seg_label=False),
            EnsureChannelFirstd(keys=["image", "label"]),
        ]
    )
    batch = transforms({"data": data_path})["image"][:, :, :, 15]
    Vd = rearrange(batch[0:1], "c h w -> () c w h")
    Pd = rearrange(batch[1:2], "c h w -> () c w h")
    Vu = rearrange(batch[-1:], "c h w -> () c w h")
    Vgt = transforms({"data": data_path})["label"][:, :, :, 15]
    Vgt = rearrange(Vgt, "c h w -> () c w h")

    # setup dataloader
    # initialize_config_dir(
    #     config_dir=str(root / "configs" / "datamodule"), job_name="test", version_base="1.2"
    # )
    # cfg = compose(config_name="unwrap_2d.yaml")
    # print(OmegaConf.to_yaml(cfg))

    # cfg.data_dir = str(root / "data")
    # cfg.in_channels = 3
    # cfg.patch_size = [40, 192]
    # cfg.batch_size = 1
    # cfg.fold = 0
    # datamodule = hydra.utils.instantiate(cfg)
    # datamodule.prepare_data()
    # datamodule.setup(stage=TrainerFn.FITTING)
    # train_dl = datamodule.train_dataloader()

    # gen = iter(train_dl)
    # batch = next(gen)
    # Vd = batch["image"][:, :-1]
    # Pd = batch["image"][:, -1:]
    # Vgt = batch["label"]

    unwrap = Robust2DUnwrap(Vd.shape[-2:], 1e-6)
    x = Vd * Pd
    y = unwrap(x)
    n = round_differentiable((y - Pd * Vd) / 2.0)
    n[n > 1] = 0
    n[n < -1] = 0
    y = Vd + 2 * n

    v1 = y[0, 0, :, :].array
    # v2 = Vu[0, 0, :, :].array
    v3 = Vd[0, 0, :, :].array
    v4 = Vgt[0, 0, :, :].array
    p = Pd[0, 0, :, :].array

    plt.figure("image", (18, 6))
    ax = plt.subplot(1, 5, 1)
    imagesc(ax, v1, "pytorch", dopplermap(), list(np.array([-1, 1]) * np.max(np.abs(v1))))
    # ax = plt.subplot(1, 5, 2)
    # imagesc(ax, v2, "matlab", dopplermap(), list(np.array([-1, 1]) * np.max(np.abs(v2))))
    ax = plt.subplot(1, 5, 3)
    imagesc(ax, v3, "ori", dopplermap(), [-1, 1])
    ax = plt.subplot(1, 5, 4)
    imagesc(ax, v4, "gt", dopplermap(), list(np.array([-1, 1]) * np.max(np.abs(v4))))
    ax = plt.subplot(1, 5, 5)
    imagesc(ax, v1 - v4, "residue", dopplermap(), [-2, 2])
    plt.show()
