"""
train.py
--------
Training loop, early stopping, and checkpoint saving for all models.

Functions:
    train_one_epoch  — Run one training epoch, return loss and accuracy.
    validate         — Run one validation epoch, return loss and accuracy.
    train_model      — Full training loop with early stopping and checkpointing.
    plot_curves      — Plot and save training/validation loss and accuracy curves.
"""

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm


# ── Single epoch helpers ──────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimiser: Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """
    Run one training epoch.

    Returns
    -------
    dict with keys 'loss' and 'acc' for this epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, leave=False, desc="  Train"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimiser.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return {
        "loss": running_loss / total,
        "acc":  correct / total,
    }


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    Run one validation epoch.

    Returns
    -------
    dict with keys 'loss' and 'acc' for this epoch.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, leave=False, desc="  Val  "):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return {
        "loss": running_loss / total,
        "acc":  correct / total,
    }


# ── Full training loop ────────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimiser: Optimizer,
    device: torch.device,
    num_epochs: int = 30,
    patience: int = 10,
    checkpoint_path: Optional[str] = None,
    scheduler: Optional[ReduceLROnPlateau] = None,
) -> Dict:
    """
    Full training loop with early stopping and optional checkpointing.

    Stops training when validation loss has not improved for `patience`
    consecutive epochs. Saves the best model weights to checkpoint_path.

    Parameters
    ----------
    model : nn.Module
        The model to train. Should already be moved to device.
    train_loader, val_loader : DataLoader
        DataLoaders for training and validation sets.
    criterion : nn.Module
        Loss function (CrossEntropyLoss for multi-class).
    optimiser : Optimizer
        Adam or SGD optimiser.
    device : torch.device
        'cuda' or 'cpu'.
    num_epochs : int
        Maximum number of training epochs.
    patience : int
        Early stopping patience in epochs.
    checkpoint_path : str, optional
        If provided, saves the best model weights here (.pth file).
    scheduler : ReduceLROnPlateau, optional
        Learning rate scheduler stepped on validation loss.

    Returns
    -------
    history : dict
        Keys: 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        Each is a list of per-epoch values.
    """
    history = {
        "train_loss": [],
        "val_loss":   [],
        "train_acc":  [],
        "val_acc":    [],
    }

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")

        train_metrics = train_one_epoch(model, train_loader, criterion, optimiser, device)
        val_metrics   = validate(model, val_loader, criterion, device)

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_acc"].append(val_metrics["acc"])

        print(
            f"  Train loss: {train_metrics['loss']:.4f}  acc: {train_metrics['acc']:.4f}"
            f"  |  Val loss: {val_metrics['loss']:.4f}  acc: {val_metrics['acc']:.4f}"
        )

        if scheduler is not None:
            scheduler.step(val_metrics["loss"])

        # Checkpoint on improvement
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0
            if checkpoint_path:
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  Checkpoint saved → {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement}/{patience} epochs")

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    # Reload best weights
    if checkpoint_path and Path(checkpoint_path).exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Best model weights restored.")

    return history


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_curves(
    history: Dict,
    model_name: str,
    save_path: str,
) -> None:
    """
    Plot training and validation loss/accuracy curves and save the figure.

    Parameters
    ----------
    history : dict
        Output of train_model().
    model_name : str
        Used in the figure title, e.g. 'ResNet-50'.
    save_path : str
        Full path where the PNG figure is saved.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{model_name} — Training Curves", fontsize=14)

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   linewidth=2)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], label="Train Acc", linewidth=2)
    axes[1].plot(epochs, history["val_acc"],   label="Val Acc",   linewidth=2)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Training curves saved to {save_path}")
