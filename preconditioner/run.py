"""Main pipeline for training and testing PrecondNet."""

from src import generate_data, train
from glob import glob


def main(config: dict) -> None:
    if not glob("../assets/data/*"):
        generate_data.main(config)

    train.main(config)


if __name__ == "__main__":
    config = {
        "SEED": 42,
        "N_THREADS": 20,
        "DEVICE": "cuda:0",
        "DATA_ROOT": "../assets/data/",
        "PC_TRAIN": .50,
        "PC_VAL": .25,
        "VALIDATE": True,
        "N_EPOCHS": 512,
    }

    main(config)
