"""A collection of miscellaneous helper functions."""

import shutil
from time import time
from typing import TYPE_CHECKING, Tuple

import numpy as np
import scipy.sparse.linalg as sla
from scipy.sparse import coo_matrix
from torch.utils.tensorboard import SummaryWriter

if TYPE_CHECKING:
    from numpy import ndarray


def _time_cg(l_matrix, rhs, preconditioner=None) -> Tuple["ndarray", int, float]:
    """Clock CG solver with and w/o preconditioner."""
    n_iter = 0
    maxiter = 1024
    residuals = np.empty((maxiter, 2))

    def callback(xk):
        nonlocal n_iter
        residuals[n_iter] = [n_iter + 1, np.sum((l_matrix * xk - rhs)**2)]
        n_iter += 1

    if preconditioner is None:
        t0 = time()
        _, _ = sla.cg(l_matrix, rhs, maxiter=maxiter, callback=callback)
        t1 = time()
    else:
        t0 = time()
        _, _ = sla.cg(l_matrix, rhs, M=preconditioner, maxiter=maxiter, callback=callback)
        t1 = time()

    return residuals, n_iter, t1 - t0


def evaluate(method, l_matrix, rhs, preconditioner=None) -> Tuple[float, int, float, float]:
    """Compute convergence speed, condition number, and density."""
    residuals, n_iter, time = _time_cg(l_matrix, rhs, preconditioner)
    np.savetxt(
        "../assets/csv/residual_" + method + ".csv", residuals[:n_iter], fmt="%.32f", header="it,res", delimiter=",")

    sigma = np.linalg.svd(l_matrix.dot(preconditioner).toarray(), compute_uv=False)
    kappa = sigma[0] / sigma[-1]
    density = preconditioner.nnz / np.prod(preconditioner.shape) * 100

    return time, n_iter, kappa, density


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
