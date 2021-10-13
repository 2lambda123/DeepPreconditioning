"""Test preconditioner performance of trained PrecondNet."""

from time import time

import numpy as np
import torch
from pyamg.aggregation import smoothed_aggregation_solver
from scipy.sparse import csc_matrix, csr_matrix, diags, eye, lil_matrix
from scipy.sparse import linalg as sla
from src.data_loader import init_loaders
from src.model import PrecondNet
from src.utils import evaluate, split

from spconv import SparseConvTensor


def _analytical_preconditioner(method: str, matrix):
    """Specify method and return preconditioner."""
    if method == "vanilla":
        return eye(matrix.shape[0])
    elif method == "jacobi":
        return diags(1. / matrix.diagonal())
    elif method == "ic":
        lu = sla.spilu(matrix.tocsc(), fill_factor=1., drop_tol=0.)
        L = lu.L
        D = diags(lu.U.diagonal())  # https://is.gd/5PJcTp
        Pr = np.zeros(matrix.shape)
        Pc = np.zeros(matrix.shape)
        Pr[lu.perm_r, np.arange(matrix.shape[0])] = 1
        Pc[np.arange(matrix.shape[0]), lu.perm_c] = 1
        Pr = lil_matrix(Pr)
        Pc = lil_matrix(Pc)
        return sla.inv((Pr.T * (L * D * L.T) * Pc.T).tocsc())
    elif method == "amg":
        preconditioner = smoothed_aggregation_solver(matrix).aspreconditioner(cycle="V")
        return csr_matrix(preconditioner.matmat(np.eye(matrix.shape[0], dtype=np.float32)))
    elif method == "ssor":
        omega = 1.0
        L, D, _ = split(matrix)
        M = omega / (2 - omega) * (1 / omega * D + L) * sla.inv(D.tocsc()) * (1 / omega * D + L).T
        return sla.inv(M.tocsc())
    elif method == "ilu":
        lu = sla.spilu(matrix.tocsc(), drop_tol=1e-3)
        n = matrix.shape[0]
        Pr = csc_matrix((np.ones(n), (lu.perm_r, np.arange(n))))
        Pc = csc_matrix((np.ones(n), (np.arange(n), lu.perm_c)))
        return csr_matrix(sla.inv(csc_matrix((Pr.T * (lu.L * lu.U) * Pc.T).A)))
    elif method == "polynomial":
        # Neumann series
        M = diags(matrix.diagonal()).toarray()
        N = M - matrix
        G = np.matmul(np.linalg.inv(M), N)
        return csr_matrix(np.matmul(eye(matrix.shape[0]) + G + np.matmul(G, G), np.linalg.inv(M)))


def _test(data_root: str, pc_train: float, pc_val: float, model, device) -> None:
    """Test loop for whole test data set."""
    model.eval()
    _, _, test_loader = init_loaders(data_root, pc_train, pc_val)
    methods = ["vanilla", "jacobi", "ic", "amg", "ssor", "ilu", "polynomial"]
    params = ["tsetup", "tconverge", "niter", "kappa", "density"]
    results = np.zeros((len(test_loader), len(methods) + 1, len(params)))

    for idx, (features, coors, shape, l_matrix) in enumerate(test_loader):
        l_matrix = csr_matrix(l_matrix[0].to_dense().numpy(), dtype=np.float32)
        rhs = np.random.randn(shape[0])

        for idy, method in enumerate(methods):
            t0 = time()
            preconditioner = _analytical_preconditioner(method, l_matrix)
            t1 = time()
            results[idx, idy, 0] = 1e3 * (t1 - t0)
            results[idx, idy, 1:] = evaluate(method, l_matrix, rhs, preconditioner)

        # learned preconditioner
        sp_tensor = SparseConvTensor(features.T.to(device), coors.int().squeeze(), shape, 1)
        t0 = time()
        preconditioner = csr_matrix(model(sp_tensor).detach().cpu().numpy())
        t1 = time()
        results[idx, -1, 0] = 1e3 * (t1 - t0)
        results[idx, -1, 1:] = evaluate("learned", l_matrix, rhs, preconditioner)

    for idx, param in enumerate(params):
        np.savetxt(f"../assets/csv/{param}.csv", results[:, :, idx], fmt="%.4f")
    # Save average values for each method.
    np.savetxt(
        "../assets/csv/averages.csv", results.mean(axis=0), fmt="%f", delimiter=",", header=(",").join(params),
        comments="")


def main(config: dict) -> None:
    """Test model and compare with various analytical preconditioners."""
    torch.manual_seed(config["SEED"])
    torch.set_num_threads(config["N_THREADS"])
    device = torch.device(config["DEVICE"])

    model = PrecondNet().to(device)
    model.load_state_dict(torch.load(config["LOAD_MODEL"]))

    _test(config["DATA_ROOT"], config["PC_TRAIN"], config["PC_VAL"], model, device)
