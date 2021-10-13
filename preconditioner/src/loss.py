"""Define different loss functions for training PrecondNet.

Further consider the approaches proposed in these papers.
[1] http://www2.cs.cas.cz/semincm/lectures/2010-07-1920-DuintjerTebbens.pdf
[2] https://arxiv.org/pdf/1301.1107v6.pdf
"""

from typing import TYPE_CHECKING

import torch

from src.utils import power_iteration

if TYPE_CHECKING:
    from torch import Tensor


def condition_loss(l_matrix: "Tensor", preconditioner: "Tensor") -> "Tensor":
    """Compute the analytical condition number with SVD."""
    _, sigma, _ = torch.svd(torch.sparse.mm(l_matrix, preconditioner))
    return sigma[0] / sigma[-1]


def power_iteration_loss(l_matrix: "Tensor", preconditioner: "Tensor") -> "Tensor":
    """Replace condition number by equivalent optimization problem.

    Oseledets, Ivan, and Vladimir Fanaskov. "Direct optimization of BPX preconditioners." Journal of Computational and
    Applied Mathematics (2021): 113811.
    """
    preconditioned = torch.sparse.mm(l_matrix, preconditioner).T

    rho = power_iteration(preconditioned)
    eye = torch.eye(preconditioned.shape[0], device=l_matrix.device)
    return power_iteration(eye - preconditioned / rho)
    # c_matrix = torch.eye(l_matrix.shape[-1], device=l_matrix.device) - preconditioned

    # eigv, _ = torch.symeig(preconditioned, eigenvectors=True)
    # print(f"kappa true {max(abs(eigv)) / min(abs(eigv))}")
    # rho = max(abs(eigv))
    # print(f"spectral true {max(abs(eigv))}")
    # print(f"spectral bound {(1 + rho) / (1 - rho)}")
    # quit()

    # lambda_min = power_iteration(
    #     preconditioned - lambda_max * torch.eye(l_matrix.shape[-1], device=l_matrix.device), max_iter=16)
    # lambda_min += lambda_max
    # return lambda_max / lambda_min


def qr_loss(l_matrix: "Tensor", preconditioner: "Tensor", n_iter: int = 8) -> "Tensor":
    """Estimate kappa with QR decomposition."""
    preconditioned = torch.sparse.mm(l_matrix, preconditioner)
    for _ in range(n_iter):
        q_matrix, r_matrix = torch.qr(preconditioned)
        preconditioned = r_matrix.matmul(q_matrix)
    return preconditioned.diag()[0] / preconditioned.diag()[-1]


def cholesky_iteration_loss(l_matrix: "Tensor", preconditioner: "Tensor", n_iter: int = 8) -> "Tensor":
    """Cholesky iterations for positive (semi-)definite matrices.

    Krishnamoorthy, Aravindh, and Kenan Kocagoez. "Singular Values using
    Cholesky Decomposition." arXiv preprint arXiv:1202.1490 (2012).
    """
    # Algorithm 2.
    J_k = torch.sparse.mm(l_matrix, preconditioner)
    for _ in range(n_iter):
        R_k = torch.cholesky(J_k, upper=True)
        J_k = torch.mm(R_k, R_k.t())
    sigma = sorted(J_k.diag())
    return sigma[-1] / sigma[0]


def laguerre_loss(l_matrix: "Tensor", preconditioner: "Tensor") -> "Tensor":
    """Sharp lower bound for smallest eigenvalue to estimate condition number.

    Yamamoto, Y. (2017). On the optimality and sharpness of Laguerre"s lower
    bound on the smallest eigenvalue of a symmetric positive definite matrix.
    Applications of Mathematics, 62(4), 319-331.
    https://doi.org/10.21136/AM.2017.0022-17
    """
    preconditioned = torch.sparse.mm(l_matrix, preconditioner)
    m = preconditioned.shape[-1]
    inverse = preconditioned.inverse()
    tr = inverse.trace()

    # Equation (2.10).
    low_bound = m / tr * (1 + torch.sqrt((m - 1) * (m * torch.mm(inverse, inverse).trace() / tr**2 - 1)))

    # Gershgorin upper bound.
    diag = preconditioned.diagonal(dim1=-2, dim2=-1)
    radii = torch.sum(preconditioned.abs(), dim=-1) - diag.abs()
    up_bound = torch.max(diag + radii)

    return up_bound / low_bound


def trace_loss(l_matrix: "Tensor", preconditioner: "Tensor") -> "Tensor":
    """Bound eigenvalues using traces of matrices.

    Wolkowicz, Henry & P.H. Styan, George (1980). Bounds for eigenvalues using
    traces. Linear Algebra and its Applications. 29. 471-506.
    10.1016/0024-3795(80)90258-X.
    """
    n_size = l_matrix.shape[-1]
    preconditioned = torch.sparse.mm(l_matrix, preconditioner)
    # m_val = preconditioned.trace() / n_size
    # s_val = ((preconditioned.mm(preconditioned)).trace() / n_size - m_val**2)**(1 / 2)
    trace = preconditioned.trace()
    trace_square = preconditioned.mm(preconditioned).trace()

    # Corollary 2.1 (ii).
    # if not trace > 0 or not trace**2 > (n_size - 1) * trace_square:
    #     raise ValueError("Conditions of Corollary 2.1 (ii) not fulfilled.")
    # return 1 + (2 * s_val * (n_size - 1)**(1 / 2)) / (m_val - s_val * (n_size - 1)**(1 / 2))

    p_val = trace**2 / trace_square - (n_size - 1)
    if not trace > 0 or not p_val > 0:
        raise ValueError("Conditions of Theorem 2.6 not fulfilled.")
    return (1 + (1 - p_val**2) * (1 / 2)) / p_val
