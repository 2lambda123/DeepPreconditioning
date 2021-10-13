"""Define convolutional neural network architecture for preconditioning.

Classes:
    PrecondNet: Fully convolutional model generating preconditioners.
"""

from typing import TYPE_CHECKING

import torch
from torch import nn

import spconv

if TYPE_CHECKING:
    from torch import Tensor


class PrecondNet(nn.Module):
    """CNN returns preconditioner for conjugate gradient solver."""

    def __init__(self) -> None:
        """Initialize model weights before training."""
        super(PrecondNet, self).__init__()
        self.layers = spconv.SparseSequential(
            spconv.SparseConv2d(1, 64, 1),
            nn.PReLU(),
            spconv.SparseConv2d(64, 256, 2, padding=(1, 0)),
            nn.PReLU(),
            spconv.SparseConv2d(256, 512, 2, padding=(1, 0)),
            nn.PReLU(),
            spconv.SparseConv2d(512, 256, 2, padding=(0, 1)),
            nn.PReLU(),
            spconv.SparseConv2d(256, 64, 2, padding=(0, 1)),
            nn.PReLU(),
            spconv.SparseConv2d(64, 1, 1),
        )

    def forward(self, tensor: "Tensor") -> "Tensor":
        """Passing system matrix through model."""
        tensor = self.layers(tensor).dense().squeeze()
        lower = torch.tril(tensor, diagonal=-1)
        diag = nn.functional.threshold(torch.diag(tensor), 1e-3, 1e-3)
        tensor = lower + torch.diag(diag)
        return tensor.mm(tensor.transpose(-2, -1))
