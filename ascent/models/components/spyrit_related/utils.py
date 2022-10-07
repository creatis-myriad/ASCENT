import numpy as np
import torch
from einops import rearrange
from scipy.sparse import csr_matrix, diags, identity, kron, lil_matrix
from torch import nn


class Forward_operator(nn.Module):
    def __init__(self, Hsub):
        super().__init__()
        r""" Defines different fully connected layers that take weights from Hsub matrix
            Args:
                Hsub (np.ndarray): M-by-N matrix.
            Returns:
                Pytorch Object of the parent-class nn.Module with two main methods:
                    - Forward: Forward propagation nn.Linear layer that assigns weights from Hsub matrix
                    - adjoint: Back-propagation pytorch nn.Linear layer obtained from Hsub.transpose() as it is orthogonal
        """
        # instancier nn.linear
        # Pmat --> (torch) --> Poids ()
        self.M = Hsub.shape[0]
        self.N = Hsub.shape[1]
        self.Hsub = nn.Linear(self.N, self.M, False)
        self.Hsub.weight.data = torch.from_numpy(Hsub)
        # Data must be of type float (or double) rather than the default float64 when creating torch tensor
        self.Hsub.weight.data = self.Hsub.weight.data.float()
        self.Hsub.weight.requires_grad = False

        # adjoint (Not useful here ??)
        self.Hsub_adjoint = nn.Linear(self.M, self.N, False)
        self.Hsub_adjoint.weight.data = torch.from_numpy(Hsub.transpose())
        self.Hsub_adjoint.weight.data = self.Hsub_adjoint.weight.data.float()
        self.Hsub_adjoint.weight.requires_grad = False

    def forward(self, x):
        r"""Forward propagate x through fully connected layer.
        Args:
            x (np.ndarray): M-by-N matrix.
        Returns:
            nn.Linear Pytorch Fully Connecter Layer that has input shape of N and output shape of M
        Example:
            >>> Input_Matrix = np.array(np.random.random([100,32]))
            >>> Forwad_OP = Forward_operator(Input_Matrix)
            >>> print('Input Matrix shape:', Input_Matrix.shape)
            >>> print('Forward propagation layer:',  Forwad_OP.Hsub)
            Input Matrix shape: (100, 32)
            Forward propagation layer: Linear(in_features=32, out_features=100, bias=False)
        """
        # x.shape[b*c,N]
        x = self.Hsub(x)
        return x

    def direct(self, x):
        # x.shape[b*c,N]
        x = self.Hsub(x)
        return x

    def adjoint(self, x):
        r"""Backpropagate x through fully connected layer.
        Args:
            x (np.ndarray): M-by-N matrix.
        Returns:
            nn.Linear Pytorch Fully Connecter Layer that has input shape of N and output shape of M
        Example:
            >>> Input_Matrix = np.array(np.random.random([100,32]))
            >>> Forwad_OP = Forward_operator(Input_Matrix)
            >>> print('Input Matrix shape:', Input_Matrix.shape)
            >>> print('Backpropagaton layer:', Forwad_OP.Hsub_adjoint)
            Input Matrix shape: (100, 32)
            Backpropagaton layer: Linear(in_features=100, out_features=32, bias=False
        """
        # x.shape[b*c,M]
        # Pmat.transpose()*f
        x = self.Hsub_adjoint(x)
        return x

    def mat(self):  # todo: Remove capital letter
        return self.Hsub.weight.data


class Doppler_operator(nn.Module):
    def __init__(self, M, N, mu):
        super().__init__()
        r""" Defines different fully connected layers that take weights from Hsub matrix
            Args:
                Hsub (np.ndarray): M-by-N matrix.
            Returns:
                Pytorch Object of the parent-class nn.Module with two main methods:
                    - Forward: Forward propagation nn.Linear layer that assigns weights from Hsub matrix
                    - adjoint: Back-propagation pytorch nn.Linear layer obtained from Hsub.transpose() as it is orthogonal
        """
        # instancier nn.linear
        # Pmat --> (torch) --> Poids ()

        self.mu = nn.Parameter(torch.tensor([float(mu)], requires_grad=True))

        self.M = M
        self.N = N

        # create sparse differentiation matrix
        self.A1 = kron(identity(self.N, format="csr"), self.differentiation_matrix(M))
        self.A2 = kron(self.differentiation_matrix(N), identity(self.M, format="csr"))

        # create empty sparse weight matrix
        self.Ws = lil_matrix((self.M * self.N, self.M * self.N))

        self.Hsub = nn.Linear(self.M * self.N, self.M * self.N, False)
        self.Hsub.weight.requires_grad = False

        # adjoint (Not useful here ??)
        self.Hsub_adjoint = nn.Linear(self.M * self.N, self.M * self.N, False)
        self.Hsub_adjoint.weight.requires_grad = False

    def create_forward_operator(self, W):
        # update weight
        self.Ws.setdiag(W.array().flatten())

        # convert Ws to csr matrix for faster matrix operation
        A = self.mu * torch.from_numpy(
            (
                self.A1.transpose().dot(self.Ws.tocsr()).dot(self.A1)
                + self.A2.transpose().dot(self.Ws.tocsr()).dot(self.A2)
            ).toarray()
        ) + torch.from_numpy(identity(self.M * self.N, format="csr").toarray())

        self.Hsub.weight.data = A.float()
        self.Hsub_adjoint.weight.data = torch.from_numpy(A.t()).float()

    def wrap(x, param=1, normalize=False):
        x = (x + param) % (2 * param) - param
        if normalize:
            return x / param
        else:
            return x

    def preprocess(self, x, wrap_param):
        d1x = csr_matrix(
            self.wrap((self.differentiation_matrix(self.M).dot(x.array())).toarray(), wrap_param)
        )
        d1x = rearrange(d1x, "b c h w -> (b c) (h w)")
        d1y = csr_matrix(
            self.wrap(
                (csr_matrix(x.array()).dot(self.differentiation_matrix(self.N))).toarray(),
                wrap_param,
            )
        )
        d1y = rearrange(d1y, "b c h w -> (b c) (h w)")
        return self.mu * torch.from_numpy(
            (
                self.A1.transpose().dot(self.Ws.tocsr()).dot(d1x)
                + self.A2.transpose().dot(self.Ws.tocsr()).dot(d1y)
            ).toarray()
        ) + rearrange(x, "b c h w -> (b c) (h w)")

    def differentiation_matrix(k):
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

    def get_diff_matrix(self):
        return self.A1, self.A2

    def forward(self, x, W):
        r"""Forward propagate x through fully connected layer.
        Args:
            x: Input.
            W: Weight.
        Returns:
            nn.Linear Pytorch Fully Connecter Layer that has input shape of N and output shape of M
        Example:
            >>> Input_Matrix = np.array(np.random.random([100,32]))
            >>> Forwad_OP = Forward_operator(Input_Matrix)
            >>> print('Input Matrix shape:', Input_Matrix.shape)
            >>> print('Forward propagation layer:',  Forwad_OP.Hsub)
            Input Matrix shape: (100, 32)
            Forward propagation layer: Linear(in_features=32, out_features=100, bias=False)
        """

        if tuple([self.M, self.N]) != W.shape:
            raise ValueError(f"Weight must have size ({self.M}, {self.N}), got {W.shape} instead.")
        self.create_forward_operator(self, W.array())
        x = self.Hsub(x)
        return x

    def direct(self, x, W):
        if tuple([self.M, self.N]) != W.shape:
            raise ValueError(f"Weight must have size ({self.M}, {self.N}), got {W.shape} instead.")
        self.create_forward_operator(self, W.array())
        x = self.Hsub(x)
        return x

    def adjoint(self, x, W):
        r"""Backpropagate x through fully connected layer.
        Args:
            x (np.ndarray): M-by-N matrix.
        Returns:
            nn.Linear Pytorch Fully Connecter Layer that has input shape of N and output shape of M
        Example:
            >>> Input_Matrix = np.array(np.random.random([100,32]))
            >>> Forwad_OP = Forward_operator(Input_Matrix)
            >>> print('Input Matrix shape:', Input_Matrix.shape)
            >>> print('Backpropagaton layer:', Forwad_OP.Hsub_adjoint)
            Input Matrix shape: (100, 32)
            Backpropagaton layer: Linear(in_features=100, out_features=32, bias=False
        """

        if tuple([self.M, self.N]) != W.shape:
            raise ValueError(f"Weight must have size ({self.M}, {self.N}), got {W.shape} instead.")
        self.create_forward_operator(self, W.array())
        x = self.Hsub_adjoint(x)
        return x

    def mat(self):
        return self.Hsub.weight.data


class Tikhonov_solve(nn.Module):
    def __init__(self):
        super().__init__()

    # def solve(self, x, W, ForwOperator):
    #     x = ForwOperator.preprocess(x)
    #     ForwOperator.create_forward_operator(W)
    #     A = ForwOperator.mat() @ torch.transpose(ForwOperator.mat(), 0, 1) + self.mu * torch.eye(F_O.M)
    #     # Can precompute H@H.T to save time!
    #     A = A.view(1, F_O.M, F_O.M)
    #     # Instead of reshaping A, reshape x in the batch-final dimension
    #     # A = A.repeat(x.shape[0],1, 1); # Not optimal in terms of memory
    #     A = A.expand(x.shape[0], -1, -1)
    #     # Not optimal in terms of memory
    #     x = torch.linalg.solve(A, x)
    #     return x

    def forward(self, x, x_0, F_O):
        # x - input (b*c, M) - measurement vector
        # x_0 - input (b*c, N) - previous estimate
        # z - output (b*c, N)

        # uses torch.linalg.solve [As of Pytorch 1.9 autograd supports solve!!]
        x = x - F_O.Forward_op(x_0)
        x = self.solve(x, F_O)
        x = x_0 + F_O.adjoint(x)
        return x


class Unet(nn.Module):
    def __init__(self, in_channel=1, out_channel=1):
        super().__init__()
        # Descending branch
        self.conv_encode1 = self.contract(in_channels=in_channel, out_channels=16)
        self.conv_maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode2 = self.contract(16, 32)
        self.conv_maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode3 = self.contract(32, 64)
        self.conv_maxpool3 = torch.nn.MaxPool2d(kernel_size=2)
        # Bottleneck
        self.bottleneck = self.bottle_neck(64)
        # Decode branch
        self.conv_decode4 = self.expans(64, 64, 64)
        self.conv_decode3 = self.expans(128, 64, 32)
        self.conv_decode2 = self.expans(64, 32, 16)
        self.final_layer = self.final_block(32, 16, out_channel)

    def contract(self, in_channels, out_channels, kernel_size=3, padding=1):
        block = torch.nn.Sequential(
            torch.nn.Conv2d(
                kernel_size=kernel_size,
                in_channels=in_channels,
                out_channels=out_channels,
                padding=padding,
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.Conv2d(
                kernel_size=kernel_size,
                in_channels=out_channels,
                out_channels=out_channels,
                padding=padding,
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels),
        )
        return block

    def expans(self, in_channels, mid_channel, out_channels, kernel_size=3, padding=1):
        block = torch.nn.Sequential(
            torch.nn.Conv2d(
                kernel_size=kernel_size,
                in_channels=in_channels,
                out_channels=mid_channel,
                padding=padding,
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.Conv2d(
                kernel_size=kernel_size,
                in_channels=mid_channel,
                out_channels=mid_channel,
                padding=padding,
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.ConvTranspose2d(
                in_channels=mid_channel,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                output_padding=1,
            ),
        )

        return block

    def concat(self, upsampled, bypass):
        out = torch.cat((upsampled, bypass), 1)
        return out

    def bottle_neck(self, in_channels, kernel_size=3, padding=1):
        bottleneck = torch.nn.Sequential(
            torch.nn.Conv2d(
                kernel_size=kernel_size,
                in_channels=in_channels,
                out_channels=2 * in_channels,
                padding=padding,
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                kernel_size=kernel_size,
                in_channels=2 * in_channels,
                out_channels=in_channels,
                padding=padding,
            ),
            torch.nn.ReLU(),
        )
        return bottleneck

    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
            torch.nn.Conv2d(
                kernel_size=kernel_size,
                in_channels=in_channels,
                out_channels=mid_channel,
                padding=1,
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.Conv2d(
                kernel_size=kernel_size,
                in_channels=mid_channel,
                out_channels=mid_channel,
                padding=1,
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.Conv2d(
                kernel_size=kernel_size,
                in_channels=mid_channel,
                out_channels=out_channels,
                padding=1,
            ),
        )
        return block

    def forward(self, x):

        # Encode
        encode_block1 = self.conv_encode1(x)
        x = self.conv_maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(x)
        x = self.conv_maxpool2(encode_block2)
        encode_block3 = self.conv_encode3(x)
        x = self.conv_maxpool3(encode_block3)

        # Bottleneck
        x = self.bottleneck(x)

        # Decode
        x = self.conv_decode4(x)
        x = self.concat(x, encode_block3)
        x = self.conv_decode3(x)
        x = self.concat(x, encode_block2)
        x = self.conv_decode2(x)
        x = self.concat(x, encode_block1)
        x = self.final_layer(x)
        return x
