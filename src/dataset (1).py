"""
dataset.py
----------
Dataset loading, splitting, and DataLoader construction for LC25000.

Classes:
    LC25000Dataset   — PyTorch Dataset wrapping the LC25000 image folders.

Functions:
    get_class_names  — Return the five canonical class names.
    make_splits      — Split image paths into train / val / test sets.
    get_dataloaders  — Return DataLoader objects for all three splits.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ── Class definitions ─────────────────────────────────────────────────────────

CLASS_NAMES: List[str] = [
    "colon_aca",  # Colon adenocarcinoma
    "colon_n",    # Benign colonic tissue
    "lung_aca",   # Lung adenocarcinoma
    "lung_n",     # Benign lung tissue
    "lung_scc",   # Lung squamous cell carcinoma
]

CLASS_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}


def get_class_names() -> List[str]:
    """Return the five canonical LC25000 class names in a fixed order."""
    return CLASS_NAMES


# ── Dataset class ─────────────────────────────────────────────────────────────

class LC25000Dataset(Dataset):
    """
    PyTorch Dataset for the LC25000 histopathology image dataset.

    Parameters
    ----------
    image_paths : list of str
        Full paths to each image file.
    labels : list of int
        Integer class label for each image (0–4).
    transform : torchvision.transforms.Compose, optional
        Transforms applied to each image at load time.
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


# ── Split helper ──────────────────────────────────────────────────────────────

def make_splits(
    dataset_root: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    random_state: int = 42,
    save_path: Optional[str] = None,
) -> Tuple[List[str], List[str], List[str], List[int], List[int], List[int]]:
    """
    Walk the LC25000 directory tree and split into train / val / test.

    The split is stratified — class proportions are preserved in all three
    sets. The split is performed once and can be saved to JSON so every
    notebook uses identical splits.

    Parameters
    ----------
    dataset_root : str
        Path to the folder containing the five class subfolders.
        Expected structure:
            dataset_root/
                colon_aca/  *.jpeg
                colon_n/    *.jpeg
                lung_aca/   *.jpeg
                lung_n/     *.jpeg
                lung_scc/   *.jpeg
    train_ratio : float
        Fraction of data for training (default 0.70).
    val_ratio : float
        Fraction of data for validation (default 0.15).
        Test ratio = 1 - train_ratio - val_ratio.
    random_state : int
        Seed for reproducibility.
    save_path : str, optional
        If provided, saves the split to a JSON file at this path.

    Returns
    -------
    train_paths, val_paths, test_paths : list of str
    train_labels, val_labels, test_labels : list of int
    """
    all_paths: List[str] = []
    all_labels: List[int] = []

    root = Path(dataset_root)
    for class_name in CLASS_NAMES:
        class_dir = root / class_name
        if not class_dir.exists():
            # Try nested structure (colon_image_sets / lung_image_sets)
            for parent in root.iterdir():
                candidate = parent / class_name
                if candidate.exists():
                    class_dir = candidate
                    break

        label = CLASS_TO_IDX[class_name]
        for img_file in sorted(class_dir.glob("*.jpeg")):
            all_paths.append(str(img_file))
            all_labels.append(label)
        for img_file in sorted(class_dir.glob("*.jpg")):
            all_paths.append(str(img_file))
            all_labels.append(label)

    # First split: train vs (val + test)
    test_ratio = 1.0 - train_ratio - val_ratio
    val_test_ratio = val_ratio + test_ratio

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_paths, all_labels,
        test_size=val_test_ratio,
        stratify=all_labels,
        random_state=random_state,
    )

    # Second split: val vs test (from the temp set)
    relative_val = val_ratio / val_test_ratio
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=1.0 - relative_val,
        stratify=temp_labels,
        random_state=random_state,
    )

    if save_path:
        split_data = {
            "train": {"paths": train_paths, "labels": train_labels},
            "val":   {"paths": val_paths,   "labels": val_labels},
            "test":  {"paths": test_paths,  "labels": test_labels},
        }
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(split_data, f)
        print(f"Split saved to {save_path}")

    print(f"Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")
    return train_paths, val_paths, test_paths, train_labels, val_labels, test_labels


def load_splits(split_path: str):
    """Load a previously saved split from JSON."""
    with open(split_path) as f:
        data = json.load(f)
    return (
        data["train"]["paths"], data["val"]["paths"], data["test"]["paths"],
        data["train"]["labels"], data["val"]["labels"], data["test"]["labels"],
    )


# ── DataLoader factory ────────────────────────────────────────────────────────

def get_dataloaders(
    train_paths: List[str],
    train_labels: List[int],
    val_paths: List[str],
    val_labels: List[int],
    test_paths: List[str],
    test_labels: List[int],
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 0,
    augment: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build DataLoaders for train, val, and test splits.

    The train loader applies augmentation when augment=True.
    Val and test loaders only resize and normalise — no augmentation.

    Parameters
    ----------
    image_size : int
        Target side length in pixels (default 224 for ImageNet models).
    batch_size : int
        Images per batch.
    num_workers : int
        Set to 0 on Google Colab with Drive to prevent connection drops.
    augment : bool
        Apply random augmentation to the training set.

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        train_transform = eval_transform

    train_loader = DataLoader(
        LC25000Dataset(train_paths, train_labels, transform=train_transform),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        LC25000Dataset(val_paths, val_labels, transform=eval_transform),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        LC25000Dataset(test_paths, test_labels, transform=eval_transform),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader
