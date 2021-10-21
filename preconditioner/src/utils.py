"""A collection of miscellaneous helper functions."""

from time import time
from typing import TYPE_CHECKING, Tuple

import numpy as np
import scipy.sparse.linalg as sla
import torch
from scipy.sparse import coo_matrix, diags, tril, triu

if TYPE_CHECKING:
    from numpy import ndarray
    from torch import Tensor


def _time_cg(l_matrix, rhs, preconditioner=None) -> Tuple["ndarray", int, float]:
    """Clock CG solver with and w/o preconditioner."""
    n_iter = 0
    maxiter = 256
    residuals = np.empty((maxiter, 2))

    def callback(xk) -> None:
        """Save residual after CG iteration."""
        nonlocal n_iter
        residuals[n_iter] = [n_iter + 1, np.sum((l_matrix * xk - rhs)**2)]
        n_iter += 1

    if preconditioner is None:
        tic = time()
        _, info = sla.cg(l_matrix, rhs, tol=1e-08, maxiter=maxiter, callback=callback)
        toc = time()
    else:
        tic = time()
        _, info = sla.cg(l_matrix, rhs, tol=1e-08, M=preconditioner, maxiter=maxiter, callback=callback)
        toc = time()

    return residuals, n_iter, toc - tic, info


def evaluate(method: str, l_matrix, rhs, preconditioner=None) -> Tuple[float, int, float, float]:
    """Compute convergence speed, condition number, and density."""
    residuals, n_iter, time, info = _time_cg(l_matrix, rhs, preconditioner)
    np.savetxt(
        f"../assets/csv/residual_{method}.csv", residuals[:n_iter], fmt="%.32f", delimiter=",", header="it,res",
        comments="")

    sigma = np.linalg.svd(l_matrix.dot(preconditioner).toarray(), compute_uv=False)
    kappa = sigma[0] / sigma[-1]
    density = preconditioner.nnz / np.prod(preconditioner.shape) * 100

    return time * 1e3, n_iter, kappa, density, info


def is_positive_definite(l_matrix_csv):
    """Check if matrix is positive definite."""
    data = np.genfromtxt(l_matrix_csv, delimiter=",")
    row = data[:, 0]
    col = data[:, 1]
    val = -data[:, 2]
    n_rows = int(max(row)) + 1
    l_matrix = coo_matrix((val, (row, col)), shape=(n_rows, n_rows))

    if (l_matrix.transpose() != l_matrix).nnz != 0:
        raise Exception("Non-symmetric matrix generated!")

    vals, _ = sla.eigs(l_matrix)
    if not (vals > 0).any():
        raise Exception("Non-positive definite matrix generated!")

    return l_matrix


def power_iteration(matrix: "Tensor", max_iter: int = 8) -> "Tensor":
    """Approximate the greatest (in absolute value) eigenvalue of the matrix."""
    b_0 = torch.rand((matrix.shape[-1], 1), device=matrix.device, requires_grad=True)
    for _ in range(max_iter):
        b_1 = torch.mm(matrix, b_0)
        b_0 = b_1 / torch.norm(b_1)
    # Rayleigh quotient
    return torch.mm(b_0.T, torch.mm(matrix, b_0)) / torch.norm(b_0)


def split(matrix):
    """Split the matrix into D (diagonal), L (strictly lower), and U (strictly upper)."""
    diagval = matrix.diagonal()
    L = tril(matrix, -1)
    D = diags(diagval, 0)
    U = triu(matrix, 1)
    return L, D, U
