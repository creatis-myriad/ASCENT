from typing import Union

import numpy as np
import torch
from einops import rearrange
from monai.data import MetaTensor
from scipy.sparse import csr_matrix, identity, kron, lil_matrix
from torch import Tensor, nn


def reshape_fortran(x: Union[MetaTensor, Tensor], shape: Union[tuple, list]):
    """Reshape tensor in Fortran-like style.

    Args:
        x: Tensor to reshape.
        shape: Desired shape for reshapping.
    """

    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


def round_differentiable(x):
    # This is equivalent to replacing round function (non-differentiable) with
    # an identity function (differentiable) only when backward.
    forward_value = torch.round(x)
    out = x.clone()
    out.data = forward_value.data
    return out


class Doppler_operator(nn.Module):
    """Forward operator for phase unwrapping. Instead of solving the linear inverse problem using
    the usual matrix multiplication, the A matrix (Forward Operator) is expressed in the form of
    fully connected layers having a weight that equals to the elements in A.

    Reference: D. C. Ghiglia and L. A. Romero. "Robust two-dimensional weighted and unweighted
    phase unwrapping that uses fast transforms and iterative methods". Journal of the Optical
    Society of America A 1994.
    """

    def __init__(self, M: int, N: int, eta: float = 0.1):
        super().__init__()
        """ Define the sparse finite difference matrices (for faster matrix multiplication) and
        fully connected layers.

            Args:
                M: Height of the input.
                N: Width of the input.
                mu: Initial regularization weight.
        """

        # learnable weight for regularization
        self.eta = nn.Parameter(torch.tensor([float(eta)], requires_grad=True))

        self.M = M
        self.N = N

        # create sparse differentiation matrix; A1, A2 = (m*n, m*n) sparse array
        self.A1 = kron(identity(self.N, format="csr"), self.differentiation_matrix(M))
        self.A2 = kron(self.differentiation_matrix(N), identity(self.M, format="csr"))

        # create empty sparse weight matrix, Ws = (m*n, m*n) sparse array
        self.Ws = lil_matrix((self.M * self.N, self.M * self.N))

        # fully connected layers that take the weight equalling to the forward operator that will be
        # defined in update_weight_matrix()
        # A = (m*n, m*n)
        self.A = nn.Linear(self.M * self.N, self.M * self.N, False)
        self.A.weight.requires_grad = False

        # adjoint of A
        self.A_t = nn.Linear(self.M * self.N, self.M * self.N, False)
        self.A_t.weight.requires_grad = False

    def update_weight_matrix(self, W: MetaTensor):
        """Update the diagonal elements of the sparse weight matrix, Ws, with the flattened given.

        weight and construct the A matrix (A = A1' @ Ws @ A1 + A2' @ Ws @ A2 + I). In Doppler
        dealiasing, W refers to the Doppler power.

        Args:
            W: Weight tensor.
        """

        # update weight matrix
        W = torch.nan_to_num(W)
        W = W / torch.max(W)
        W[W == 0] = 1e-6
        self.Ws.setdiag(W.array.flatten(order="F"))

        # build A matrix: A = A1' @ Ws @ A1 + A2' @ Ws @ A2 + I
        A = torch.from_numpy(
            (
                self.A1.transpose().dot(self.Ws.tocsr()).dot(self.A1)
                + self.A2.transpose().dot(self.Ws.tocsr()).dot(self.A2)
            ).toarray()
        ).to(W.device.index) + self.eta * torch.from_numpy(
            identity(self.M * self.N, format="csr").toarray()
        ).to(
            W.device.index
        )

        # update the weight of the fully connected layers
        self.A.weight.data = A.float()
        self.A_t.weight.data = torch.t(A.float())

    @staticmethod
    def wrap(x: MetaTensor, wrap_param: float = 1.0, normalize: bool = False):
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

    def preprocess(self, x: MetaTensor, wrap_param: float = 1.0, normalize: bool = False):
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
                    rearrange(x.array, "b c m n -> m (n b c)")
                ),
                wrap_param,
                normalize,
            )
        )

        # Fortran-style flatten
        d1y = np.reshape(d1y.toarray(), (self.M * self.N, b * c), order="F")

        # finite difference matrix along horizontal axis
        d1x = csr_matrix(
            self.wrap(
                (
                    csr_matrix(rearrange(x.array, "b c m n -> (b c m) n")).dot(
                        self.differentiation_matrix(self.N).transpose()
                    )
                ).toarray(),
                wrap_param,
                normalize,
            )
        )
        d1x = np.reshape(d1x.toarray(), (self.M * self.N, b * c), order="F")

        return rearrange(
            torch.from_numpy(
                self.A1.transpose().dot(self.Ws.tocsr()).dot(d1y)
                + self.A2.transpose().dot(self.Ws.tocsr()).dot(d1x)
            )
            .float()
            .to(x.device.index),
            "(m n) (b c) -> (b c) (m n)",
            b=b,
            c=c,
            m=m,
        ) + self.eta * reshape_fortran(x, (b * c, m * n))

    @staticmethod
    def differentiation_matrix(k: int):
        """Build a finite difference matrix with -1 and 1.

        Args:
            k: Dimension of square matrix to create.

        Returns:
            Sparse finite difference matrix.
        """

        m = -np.eye(k) + np.eye(k, k, 1)
        m[-1, -1] = 1
        m[-1, -2] = -1
        return csr_matrix(m)

    def forward(
        self, x: MetaTensor, W: MetaTensor, wrap_param: float = 1.0, normalize: bool = False
    ):
        """Forward propagate x through fully connected layer.

        Args:
            x: Input tensor (b, c, m, n).
            W: Weight tensor (b, c, m, n).
            wrap_param: Wrapping parameter.
            normalize: Whether to normalize the wrapped tensor between -1 and 1.

        Returns:
            A @ x.

        Raises:
            ValueError: Error when the shape of the weight tensor does not match the one of input
            tensor.

        Example:
            >>> Input_Matrix = np.array(np.random.random([100,32]))
            >>> Fwd_OP = Doppler_operator(Input_Matrix)
            >>> print('Input Matrix shape:', Input_Matrix.shape)
            >>> print('Forward propagation layer:',  Fwd_OP.A)
            Input Matrix shape: (100, 32)
            Forward propagation layer: Linear(in_features=3200, out_features=3200, bias=False)
        """

        if tuple([self.M, self.N]) != W.shape[2:]:
            raise ValueError(f"Weight must have size ({self.M}, {self.N}), got {W.shape} instead.")
        self.update_weight_matrix(W)
        b, c, m, n = x.shape
        # x = (b*c, m*n)
        x = self.preprocess(x, wrap_param, normalize)
        # x = (b*c, m*n)
        x = self.A(x)
        # x = (b, c, m, n)
        x = reshape_fortran(x, (b, c, m, n))
        return x

    def direct(
        self, x: MetaTensor, W: MetaTensor, wrap_param: float = 1.0, normalize: bool = False
    ):
        if tuple([self.M, self.N]) != W.shape[2:]:
            raise ValueError(f"Weight must have size ({self.M}, {self.N}), got {W.shape} instead.")
        self.update_weight_matrix(W)
        b, c, m, n = x.shape
        x = self.preprocess(x, wrap_param, normalize)
        x = self.A(x)
        x = reshape_fortran(x, (b, c, m, n))
        return x

    def adjoint(
        self, x: MetaTensor, W: MetaTensor, wrap_param: float = 1.0, normalize: bool = False
    ):
        """Back propagate x through fully connected layer.

        Args:
            x: Input tensor (b, c, m, n).
            W: Weight tensor (b, c, m, n).
            wrap_param: Wrapping parameter.
            normalize: Whether to normalize the wrapped tensor between -1 and 1.

        Returns:
            A_t @ x.

        Raises:
            ValueError: Error when the shape of the weight tensor does not match the one of input
            tensor.

        Example:
            >>> Input_Matrix = np.array(np.random.random([100,32]))
            >>> Fwd_OP = Doppler_operator(Input_Matrix)
            >>> print('Input Matrix shape:', Input_Matrix.shape)
            >>> print('Back propagation layer:',  Fwd_OP.A_t)
            Input Matrix shape: (100, 32)
            Back propagation layer: Linear(in_features=3200, out_features=3200, bias=False)
        """

        if tuple([self.M, self.N]) != W.shape[2:]:
            raise ValueError(f"Weight must have size ({self.M}, {self.N}), got {W.shape} instead.")
        self.update_weight_matrix(W)
        b, c, m, n = x.shape
        x = self.preprocess(x, wrap_param, normalize)
        x = self.A_t(x)
        x = reshape_fortran(x, (b, c, m, n))
        return x

    def mat(self):
        """Return the weight of the fully connected layers A.

        Returns:
            Weight of the fully connected layers A.
        """

        return self.A.weight.data


class Doppler_operatorV2(nn.Module):
    """Forward operator for phase unwrapping. Instead of solving the linear inverse problem using
    the usual matrix multiplication, the A matrix (Forward Operator) is expressed in the form of
    fully connected layers having a weight that equals to the elements in A.

    Reference: D. C. Ghiglia and L. A. Romero. "Robust two-dimensional weighted and unweighted
    phase unwrapping that uses fast transforms and iterative methods". Journal of the Optical
    Society of America A 1994.
    """

    def __init__(
        self, M: int, N: int, eta: float = 0.1, wrap_param: float = 1.0, normalize: bool = False
    ):
        super().__init__()
        """ Define the sparse finite difference matrices (for faster matrix multiplication) and
        fully connected layers.

            Args:
                M: Height of the input.
                N: Width of the input.
                mu: Initial regularization weight.
                wrap_param: Wrapping parameter.
                normalize: Whether to normalize the wrapped tensor between -1 and 1.
        """

        # learnable weight for regularization
        self.eta = nn.Parameter(torch.tensor([float(eta)], requires_grad=True))

        self.M = M
        self.N = N

        self.wrap_param = wrap_param
        self.normalize = normalize

        # create sparse differentiation matrix; A1, A2 = (m*n, m*n) sparse array
        self.A1 = kron(identity(self.N, format="csr"), self.differentiation_matrix(M))
        self.A2 = kron(self.differentiation_matrix(N), identity(self.M, format="csr"))

        # build A matrix: A = A1' @ A1 + A2' @ A2 + eta * I
        A = torch.from_numpy(
            (self.A1.transpose().dot(self.A1) + self.A2.transpose().dot(self.A2)).toarray()
        ) + self.eta * torch.from_numpy(identity(self.M * self.N, format="csr").toarray())

        # fully connected layers that take the weight equalling to the forward operator that will be
        # defined in update_weight_matrix()
        # A = (m*n, m*n)
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
    def wrap(x: MetaTensor, wrap_param: float = 1.0, normalize: bool = False):
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

    def preprocess(self, x: MetaTensor):
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
                    rearrange(x.array, "b c m n -> m (n b c)")
                ),
                self.wrap_param,
                self.normalize,
            )
        )

        # Fortran-style flatten
        d1y = np.reshape(d1y.toarray(), (self.M * self.N, b * c), order="F")

        # finite difference matrix along horizontal axis
        d1x = csr_matrix(
            self.wrap(
                (
                    csr_matrix(rearrange(x.array, "b c m n -> (b c m) n")).dot(
                        self.differentiation_matrix(self.N).transpose()
                    )
                ).toarray(),
                self.wrap_param,
                self.normalize,
            )
        )
        d1x = np.reshape(d1x.toarray(), (self.M * self.N, b * c), order="F")

        return rearrange(
            torch.from_numpy(self.A1.transpose().dot(d1y) + self.A2.transpose().dot(d1x))
            .float()
            .to(x.device.index),
            "(m n) (b c) -> (b c) (m n)",
            b=b,
            c=c,
            m=m,
        ) + self.eta * reshape_fortran(x, (b * c, m * n))

    @staticmethod
    def differentiation_matrix(k: int):
        """Build a finite difference matrix with -1 and 1.

        Args:
            k: Dimension of square matrix to create.

        Returns:
            Sparse finite difference matrix.
        """

        m = -np.eye(k) + np.eye(k, k, 1)
        m[-1, -1] = 1
        m[-1, -2] = -1
        return csr_matrix(m)

    def forward(self, x: MetaTensor):
        """Forward propagate x through fully connected layer.

        Args:
            x: Input tensor (b, c, m, n).
            W: Weight tensor (b, c, m, n).
            wrap_param: Wrapping parameter.
            normalize: Whether to normalize the wrapped tensor between -1 and 1.

        Returns:
            A @ x.

        Raises:
            ValueError: Error when the shape of the weight tensor does not match the one of input
            tensor.

        Example:
            >>> Input_Matrix = np.array(np.random.random([100,32]))
            >>> Fwd_OP = Doppler_operator(Input_Matrix)
            >>> print('Input Matrix shape:', Input_Matrix.shape)
            >>> print('Forward propagation layer:',  Fwd_OP.A)
            Input Matrix shape: (100, 32)
            Forward propagation layer: Linear(in_features=3200, out_features=3200, bias=False)
        """

        if tuple([self.M, self.N]) != x.shape[2:]:
            raise ValueError(f"Input must have size ({self.M}, {self.N}), got {x.shape} instead.")
        b, c, _, _ = x.shape
        # x = (b*c, m*n)
        x = self.preprocess(x)
        # x = (b*c, m*n)
        x = self.A(x)
        # x = (b, c, m, n)
        x = reshape_fortran(x, (b, c, self.M, self.N))
        return x

    def direct(self, x: MetaTensor):
        if tuple([self.M, self.N]) != x.shape[2:]:
            raise ValueError(f"Input must have size ({self.M}, {self.N}), got {x.shape} instead.")
        b, c, m, n = x.shape
        x = self.preprocess(x)
        x = self.A(x)
        x = reshape_fortran(x, (b, c, self.M, self.N))
        return x

    def adjoint(self, x: MetaTensor):
        """Back propagate x through fully connected layer.

        Args:
            x: Input tensor (b, c, m, n).
            W: Weight tensor (b, c, m, n).
            wrap_param: Wrapping parameter.
            normalize: Whether to normalize the wrapped tensor between -1 and 1.

        Returns:
            A_t @ x.

        Raises:
            ValueError: Error when the shape of the weight tensor does not match the one of input
            tensor.

        Example:
            >>> Input_Matrix = np.array(np.random.random([100,32]))
            >>> Fwd_OP = Doppler_operator(Input_Matrix)
            >>> print('Input Matrix shape:', Input_Matrix.shape)
            >>> print('Back propagation layer:',  Fwd_OP.A_t)
            Input Matrix shape: (100, 32)
            Back propagation layer: Linear(in_features=3200, out_features=3200, bias=False)
        """

        if tuple([self.M, self.N]) != x.shape[2:]:
            raise ValueError(f"Input must have size ({self.M}, {self.N}), got {x.shape} instead.")
        b, c, m, n = x.shape
        x = self.preprocess(x)
        x = self.A_adj(x)
        x = reshape_fortran(x, (b, c, self.M, self.N))
        return x

    def mat(self):
        """Return the weight of the fully connected layers A.

        Returns:
            Weight of the fully connected layers A.
        """

        return self.A.weight.data

    def mat_adj(self):
        """Return the weight of the fully connected layers A_adj.

        Returns:
            Weight of the fully connected layers A_adj.
        """

        return self.A_adj.weight.data


class Tikhonov_solve(nn.Module):
    """Tikhonov solver for linear inverse problem."""

    def __init__(self, mu: int = 0.1):
        """Initialize the Tikhonov regularization weight, mu.

        Args:
            mu: Weight for the Tikhonov regularization weight.
        """

        super().__init__()
        self.mu = nn.Parameter(torch.tensor([float(mu)], requires_grad=True))

    def solve(self, x: MetaTensor, W: MetaTensor, FwdOperator: nn.Module):
        """Solve linear inverse problem using torch.linalg.solve.

        Args:
            x: Input tensor.
            W: Weight tensor.
            FwdOperator: Forward operator, i.e. A in Ax=b system.
        """

        b, c, m, n = x.shape

        # update weight matrix
        # FwdOperator.update_weight_matrix(W)

        # x = (b*c, m*n)
        # x = FwdOperator.preprocess(x)
        # x = FwdOperator(x, W)

        # Fortran-style flatten
        # x = (m*n, b*c)
        x = reshape_fortran(x, (m * n, b * c))

        # A = (m*n, m*n) @ (m*n, m*n) = (m*n, m*n)
        # A = FwdOperator.mat() @ rearrange(FwdOperator.mat(), 'h w -> w h') + self.mu * torch.eye(FwdOperator.M * FwdOperator.N);

        # x = (m*n, b*c)
        # x = torch.linalg.solve(A, x)
        x = torch.linalg.solve(FwdOperator.mat(), x)
        # x = torch.linalg.lstsq(FwdOperator.mat(), x, rcond=1e-35)
        # x = x[0]

        # Fortran-style reshape
        x = reshape_fortran(x, (b, c, m, n))
        return x

    def forward(self, x, x_0, W, FwdOperator):  # noqa: D102
        # uses torch.linalg.solve [As of Pytorch 1.9 autograd supports solve!!]
        # x = x - FwdOperator.direct(x_0, W)
        x = self.solve(x, W, FwdOperator)
        # x = x_0 + FwdOperator.adjoint(x, W)
        return x


class Tikhonov_solveV2(nn.Module):
    """Tikhonov solver for linear inverse problem."""

    def __init__(self, mu: int = 0.1):
        """Initialize the Tikhonov regularization weight, mu.

        Args:
            mu: Weight for the Tikhonov regularization weight.
        """

        super().__init__()
        self.mu = nn.Parameter(torch.tensor([float(mu)], requires_grad=True))

    def solve(self, x: MetaTensor, FwdOperator: nn.Module):
        """Solve linear inverse problem using torch.linalg.solve.

        Args:
            x: Input tensor.
            W: Weight tensor.
            FwdOperator: Forward operator, i.e. A in Ax=b system.
        """

        b, c, m, n = x.shape

        # x = (b*c, m*n)
        x = FwdOperator.preprocess(x)
        # x = FwdOperator(x)

        # Fortran-style flatten
        # x = (m*n, b*c)
        x = reshape_fortran(x, (m * n, b * c))

        # A = (m*n, m*n) @ (m*n, m*n) = (m*n, m*n)
        # A = FwdOperator.mat() @ FwdOperator.mat_adj() + self.mu * torch.eye(FwdOperator.M * FwdOperator.N);

        # x = (m*n, b*c)
        # x = torch.linalg.solve(A, x)
        x = torch.linalg.solve(FwdOperator.mat(), x)

        # Fortran-style reshape
        x = reshape_fortran(x, (b, c, m, n))
        return x

    def forward(self, x, FwdOperator):  # noqa: D102
        # uses torch.linalg.solve [As of Pytorch 1.9 autograd supports solve!!]
        x = self.solve(x, FwdOperator)
        # x = FwdOperator.adjoint(x)
        return x


if __name__ == "__main__":
    import hydra
    import pyrootutils
    from hydra import compose, initialize_config_dir
    from matplotlib import pyplot as plt
    from monai.transforms import Compose, EnsureChannelFirstd, SpatialCropd, SpatialPadd
    from omegaconf import OmegaConf
    from pytorch_lightning.trainer.states import TrainerFn

    from ascent.datamodules.components.transforms import LoadNpyd, MayBeSqueezed
    from ascent.utils.visualization import dopplermap, imagesc

    root = pyrootutils.setup_root(
        search_from=__file__,
        indicator=[".git", "pyproject.toml"],
        pythonpath=True,
        dotenv=True,
    )

    # load specific file
    # data_path = str(
    #     root / "data" / "UNWRAP" / "preprocessed" / "data_and_properties" / "Dealias_0022.npy"
    # )
    # transforms = Compose(
    #     [
    #         LoadNpyd(keys=["data"], seg_label=False),
    #         EnsureChannelFirstd(keys=["image", "label"]),
    #     ]
    # )
    # batch = transforms({"data": data_path})["image"][:, :, :, 15]
    # Vd = rearrange(batch[0:1], "c h w -> () c w h")
    # Pd = rearrange(batch[1:2], "c h w -> () c w h")
    # Vu = rearrange(batch[-1:], "c h w -> () c w h")
    # Vgt = transforms({"data": data_path})["label"][:, :, :, 15]
    # Vgt = rearrange(Vgt, "c h w -> () c w h")

    initialize_config_dir(
        config_dir=str(root / "configs" / "datamodule"), job_name="test", version_base="1.2"
    )
    cfg = compose(config_name="unwrapV2_2d.yaml")
    print(OmegaConf.to_yaml(cfg))

    cfg.data_dir = str(root / "data")
    cfg.in_channels = 3
    cfg.patch_size = [40, 192]
    cfg.batch_size = 1
    cfg.fold = 0
    datamodule = hydra.utils.instantiate(cfg)
    datamodule.prepare_data()
    datamodule.setup(stage=TrainerFn.FITTING)
    train_dl = datamodule.train_dataloader()

    gen = iter(train_dl)
    batch = next(gen)
    Vd = batch["image"][:, :-1]
    Pd = batch["image"][:, -1:]
    Vgt = batch["label"]
    Vgt = Vgt

    # # %%
    # Fwd_OP = Doppler_operator(Vd.shape[-2], Vd.shape[-1], 1e-6)
    # DC_layer = Tikhonov_solve(mu=0.1)
    # x = Vd
    # y = DC_layer(x, torch.zeros_like(x), Pd, Fwd_OP)
    # n = round_differentiable((y - Vd) / 2.0)
    # y = Vd + 2 * n

    # # %%
    # v1 = y[0, 0, :, :].array
    # v2 = Vu[0, 0, :, :].array
    # v3 = Vd[0, 0, :, :].array
    # v4 = Vgt[0, 0, :, :].array

    # plt.figure("image", (18, 6))
    # ax = plt.subplot(1, 5, 1)
    # imagesc(ax, v1, "pytorch", dopplermap(), list(np.array([-1, 1]) * np.max(np.abs(v1))))
    # ax = plt.subplot(1, 5, 2)
    # imagesc(ax, v2, "matlab", dopplermap(), list(np.array([-1, 1]) * np.max(np.abs(v2))))
    # ax = plt.subplot(1, 5, 3)
    # imagesc(ax, v3, "ori", dopplermap(), [-1, 1])
    # ax = plt.subplot(1, 5, 4)
    # imagesc(ax, v4, "gt", dopplermap(), list(np.array([-1, 1]) * np.max(np.abs(v4))))
    # ax = plt.subplot(1, 5, 5)
    # imagesc(
    #     ax, v1 - v4, "residue", dopplermap(), list(np.array([-1, 1]) * np.max(np.abs(v1 - v4)))
    # )
    # plt.show()

    # %%
    Fwd_OP = Doppler_operatorV2(Vd.shape[-2], Vd.shape[-1], 1e-6)
    DC_layer = Tikhonov_solveV2(mu=0.1)
    x = Vd * Pd
    # x = Fwd_OP(x)
    y = DC_layer(x, Fwd_OP)
    n = round_differentiable((y - Pd * Vd) / 2.0)
    n[n > 1] = 0
    n[n < -1] = 0
    y = Vd + 2 * n

    # %%
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
