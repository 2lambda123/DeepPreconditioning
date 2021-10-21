"""Generate matrices for benchmark CFD problems."""

import subprocess
from glob import glob

from scipy.sparse import save_npz
from src.utils import is_positive_definite


def main() -> None:
    benchmarks = glob("../benchmarks/waterC*")
    for benchmark in benchmarks:
        name = benchmark.split("/")[-1]
        subprocess.call([f"{benchmark}/Allrun"])
        l_matrix = is_positive_definite(f"{benchmark}/L.csv")
        print(l_matrix.shape)
        save_npz(f"../assets/data/benchmarks/L_{name}.npz", l_matrix)


if __name__ == "__main__":
    main()
