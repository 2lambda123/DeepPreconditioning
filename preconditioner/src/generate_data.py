"""Generate OpenFOAM system matrices based on sludge patterns or baffles."""

import subprocess
from itertools import product

import numpy as np
import triangle as tr
from scipy.sparse import save_npz
from src.utils import is_positive_definite
from stl import mesh


def _baffle(data_root: str) -> None:
    """Simulate baffles with different heights and positions."""
    parameters = dict(xmin=[2., 2.5, 3., 3.5, 4., 4.5], height=[1., 1.5, 2., 2.5, 3.])
    param_values = [val for val in parameters.values()]

    for idx, (xmin, height) in enumerate(product(*param_values)):
        # open OpenFOAM dictionary template
        with open("../foam/sim/system/snappyHexMeshDict.org", "r") as file_:
            data = file_.readlines()

        # baffle minimum and maximum point
        data[26] = data[26].replace("xx", str(xmin)).replace("yy", "-" + str(height)).replace("zz", "-0.5")
        data[27] = data[27].replace("xx", str(xmin + .5)).replace("yy", "0").replace("zz", "0")

        # write OpenFOAM dictionary file for simulation
        with open("../foam/sim/system/snappyHexMeshDict", "w") as file_:
            file_.writelines(data)

        # run simulation and dump discretization matrix
        subprocess.call(["../foam/sim/Allrun"])
        l_matrix = is_positive_definite("../foam/sim/L.csv")
        save_npz(f"{data_root}/L" + str(idx).zfill(3) + ".npz", l_matrix)


def _sludge_pattern(resolution: int = 128) -> None:
    """Create random sludge pattern at tank bottom."""
    x_pos = np.linspace(1, 25, num=resolution)
    y_pos = .0625 * x_pos - 6.0625
    y_pos[1:-1] += np.random.normal(loc=1.0, scale=.5, size=resolution - 2)

    vertices = np.zeros((2 * resolution, 3))
    vertices[:, 0] = np.concatenate((x_pos, x_pos[::-1]))
    vertices[:, 1] = np.concatenate((y_pos, y_pos[::-1]))
    vertices[resolution:, 2] = resolution * [-.5]

    vert_id = np.array(range(2 * resolution))
    triags = tr.triangulate(
        dict(vertices=vertices[:, [0, 2]], segments=np.stack((vert_id, (vert_id + 1) % len(vert_id))).T), "pq")
    faces = triags["triangles"]

    sludge = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for idx, face in enumerate(faces):
        for idy in range(3):
            sludge.vectors[idx][idy] = vertices[face[idy], :]

    sludge.save("../foam/sim/constant/triSurface/sludge.stl")


def _sludge(seed: int, data_count: int, data_root: str) -> None:
    """Simulate various random sluge patterns."""
    np.random.seed(seed)
    for idx in range(data_count):
        _sludge_pattern()

        # run simulation and dump discretization matrix
        subprocess.call(["../foam/sim/Allrun"])
        l_matrix = is_positive_definite("../foam/sim/L.csv")
        print(l_matrix.shape)
        save_npz(f"{data_root}/L" + str(idx).zfill(3) + ".npz", l_matrix)


def main(config: dict) -> None:
    """Pipeline generating train and test data."""
    # _baffle(config["DATA_ROOT"])
    _sludge(config["SEED"], config["DATA_COUNT"], config["DATA_ROOT"])
