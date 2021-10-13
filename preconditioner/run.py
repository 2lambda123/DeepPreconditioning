"""Main pipeline for training and testing PrecondNet."""

from src import generate_data, train, test


def main(config: dict) -> None:
    # generate_data.main(config)

    # train.main(config)
    test.main(config)


if __name__ == "__main__":
    config = {
        "SEED": 42,
        "N_THREADS": 20,
        "DEVICE": "cuda:0",
        "DATA_ROOT": "../assets/data/cells_02/",
        "DATA_COUNT": 100,
        "PC_TRAIN": 0.0,
        "PC_VAL": 0.0,
        "VALIDATE": True,
        "N_EPOCHS": 512,
        "LOAD_MODEL": "../assets/runs/2021-10-11_15:42:28/model.pt",
    }

    main(config)
