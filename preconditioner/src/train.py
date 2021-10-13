"""Training loop for PrecondNet based on OpenFOAM system matrix data set."""

from datetime import datetime
from typing import TYPE_CHECKING, Callable, Tuple

import torch
from src.data_loader import init_loaders
from src.loss import condition_loss, power_iteration_loss
from src.model import PrecondNet
from torch.utils.tensorboard import SummaryWriter

from spconv import SparseConvTensor

if TYPE_CHECKING:
    from torch import device
    from torch.nn import Module
    from torch.optim import Optimizer
    from torch.utils.data.dataloader import DataLoader


def _train_epoch(loader: "DataLoader", model: "Module", criterion: Callable, optimizer: "Optimizer",
                 device: "device") -> Tuple[float, "Module"]:
    """Train model for a single epoch."""
    model.train()
    epoch_loss = 0.
    for features, coors, shape, l_matrix in loader:
        sp_tensor = SparseConvTensor(features.T.to(device), coors.int().squeeze(), shape, 1)
        l_matrix = l_matrix[0].to(device)

        optimizer.zero_grad()
        loss = criterion(l_matrix, model(sp_tensor))
        loss.backward()
        optimizer.step()

        epoch_loss += loss.data.item()
    return epoch_loss, model


def _validate(loader: "DataLoader", model: "Module", criterion: Callable, device: "device") -> float:
    """Validate model during training."""
    model.eval()
    val_loss = 0.
    for features, coors, shape, l_matrix in loader:
        sp_tensor = SparseConvTensor(features.T.to(device), coors.int().squeeze(), shape, 1)
        l_matrix = l_matrix[0].to(device)

        loss = criterion(l_matrix, model(sp_tensor))
        val_loss += loss.data.item()
    return val_loss


def _train(
        data_root: str, pc_train: float, pc_val: float, model: "Module", criterion: Callable, n_epochs: int,
        optimizer: "Optimizer", validate: bool, writer: "SummaryWriter", device: "device") -> None:
    """Training loop for PrecondNet."""
    train_loader, val_loader, _ = init_loaders(data_root, pc_train, pc_val)
    min_val_loss = 1e9
    for epoch in range(n_epochs):
        epoch_loss, model = _train_epoch(train_loader, model, criterion, optimizer, device)
        writer.add_scalar("loss/train", epoch_loss / len(train_loader), epoch)

        if validate and epoch % 5 == 0:
            val_loss = _validate(val_loader, model, condition_loss, device)
            if min_val_loss > val_loss:
                torch.save(model.state_dict(), f"{writer.log_dir}/model.pt")
                min_val_loss = val_loss
            writer.add_scalar("loss/val", val_loss / len(val_loader), epoch)


def main(config: dict) -> None:
    """Train CNN on linear system matrices."""
    torch.manual_seed(config["SEED"])
    torch.set_num_threads(config["N_THREADS"])
    device = torch.device(config["DEVICE"])

    model = PrecondNet().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    writer = SummaryWriter(f"../assets/runs/{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
    writer.add_hparams(config, metric_dict={})

    _train(
        config["DATA_ROOT"], config["PC_TRAIN"], config["PC_VAL"], model, condition_loss, config["N_EPOCHS"],
        optimizer, config["VALIDATE"], writer, device)

    writer.close()
