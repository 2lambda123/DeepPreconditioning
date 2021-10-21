"""Main pipeline for training and testing PrecondNet."""

from src import benchmark, generate_data, test, train


def main(config: dict) -> None:
    # generate_data.main(config)
    # benchmark.main()

    # train.main(config)
    test.main(config)


if __name__ == "__main__":
    config = {
        "SEED": 42,
        "N_THREADS": 20,
        "DEVICE": "cuda:0",
        "DATA_ROOT": "../assets/data/benchmarks/",
        "DATA_COUNT": 1,
        "PC_TRAIN": 0.00,
        "PC_VAL": 0.00,
        "VALIDATE": True,
        "N_EPOCHS": 512,
        "LOAD_MODEL": "../assets/runs/best/model.pt",
    }

    main(config)
