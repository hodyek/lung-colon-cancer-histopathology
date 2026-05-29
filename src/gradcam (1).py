"""
gradcam.py
----------
Grad-CAM implementation for ResNet-50 and other CNN architectures.

Classes:
    GradCAM          — Registers hooks and computes heatmaps.

Functions:
    get_gradcam_layer   — Return the correct target layer for a given model.
    generate_heatmap    — Compute and resize a Grad-CAM heatmap for one image.
    overlay_heatmap     — Overlay a heatmap on an original image.
    plot_gradcam_grid   — Plot original / heatmap / overlay for N images per class.
    plot_gradcam_detail — Plot one image per class with three-column layout.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ── GradCAM class ─────────────────────────────────────────────────────────────

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    Registers forward and backward hooks on a target convolutional layer.
    Call generate() to produce a heatmap for a single input tensor.

    Parameters
    ----------
    model : nn.Module
        The model to explain. Should be in eval mode.
    target_layer : nn.Module
        The convolutional layer to compute gradients for.
        For ResNet-50 use model.layer4[-1].
        For EfficientNet-B0 use model.features[-1].
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap for a single image.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Shape (1, C, H, W), already on the model's device.
        target_class : int, optional
            Class index to explain. If None, uses the predicted class.

        Returns
        -------
        heatmap : np.ndarray
            Shape (H, W), values in [0, 1].
        """
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        score = output[0, target_class]
        score.backward()

        # Global average pool gradients over spatial dimensions
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = torch.relu(cam).squeeze().cpu().numpy()

        # Normalise to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam


# ── Layer selector ────────────────────────────────────────────────────────────

def get_gradcam_layer(model: nn.Module, model_name: str) -> nn.Module:
    """
    Return the appropriate target layer for Grad-CAM.

    Parameters
    ----------
    model : nn.Module
    model_name : str
        One of 'resnet50', 'efficientnet_b0', 'baseline'.
    """
    name = model_name.lower()
    if "resnet" in name:
        return model.layer4[-1]
    elif "efficientnet" in name:
        return model.features[-1]
    elif "baseline" in name:
        # Last conv block in the baseline CNN
        return model.features[-1].block[-3]
    else:
        raise ValueError(f"Unknown model name for Grad-CAM: '{model_name}'.")


# ── Heatmap helpers ───────────────────────────────────────────────────────────

def generate_heatmap(
    gradcam: GradCAM,
    input_tensor: torch.Tensor,
    target_class: Optional[int] = None,
    output_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Compute a Grad-CAM heatmap and resize to output_size.

    Returns
    -------
    heatmap : np.ndarray
        Shape (H, W), values in [0, 1].
    """
    cam = gradcam.generate(input_tensor, target_class)
    heatmap = cv2.resize(cam, (output_size[1], output_size[0]))
    return heatmap


def overlay_heatmap(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on the original image.

    Parameters
    ----------
    original_image : np.ndarray
        Shape (H, W, 3), uint8, RGB.
    heatmap : np.ndarray
        Shape (H, W), values in [0, 1].
    alpha : float
        Heatmap opacity (0 = invisible, 1 = fully opaque).
    colormap : int
        OpenCV colormap for the heatmap (default JET).

    Returns
    -------
    overlay : np.ndarray
        Shape (H, W, 3), uint8, RGB.
    """
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap_rgb, alpha, 0)
    return overlay


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a normalised image tensor back to a uint8 RGB numpy array.

    Reverses ImageNet normalisation.
    """
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    img = tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_gradcam_grid(
    model: nn.Module,
    loader: DataLoader,
    gradcam: GradCAM,
    class_names: List[str],
    device: torch.device,
    n_per_class: int = 3,
    save_path: Optional[str] = None,
    title: str = "Grad-CAM: Correctly Classified Images",
) -> None:
    """
    Plot original and Grad-CAM overlay for n_per_class correctly classified
    images per class.

    Layout: each class gets one row. Columns alternate Original / Grad-CAM.

    Parameters
    ----------
    model : nn.Module
    loader : DataLoader
        Should be the test loader (shuffle=False).
    gradcam : GradCAM
    class_names : list of str
    device : torch.device
    n_per_class : int
        Number of images to show per class.
    save_path : str, optional
        If provided, saves the figure here.
    title : str
        Figure title.
    """
    model.eval()
    collected = {i: [] for i in range(len(class_names))}

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            preds = outputs.argmax(dim=1).cpu()
            for j in range(len(labels)):
                label = labels[j].item()
                pred  = preds[j].item()
                if pred == label and len(collected[label]) < n_per_class:
                    collected[label].append(images[j])
            if all(len(v) >= n_per_class for v in collected.values()):
                break

    num_classes = len(class_names)
    fig, axes = plt.subplots(
        num_classes, n_per_class * 2,
        figsize=(n_per_class * 4, num_classes * 2.5)
    )
    fig.suptitle(title, fontsize=12, y=1.01)

    for row, (class_idx, tensors) in enumerate(collected.items()):
        for col_pair, tensor in enumerate(tensors):
            inp = tensor.unsqueeze(0).to(device)
            inp.requires_grad_(True)

            heatmap = generate_heatmap(gradcam, inp, target_class=class_idx)
            original = tensor_to_image(tensor)
            overlay  = overlay_heatmap(original, heatmap)

            col_orig = col_pair * 2
            col_over = col_pair * 2 + 1

            axes[row, col_orig].imshow(original)
            axes[row, col_orig].axis("off")
            if col_pair == 0:
                axes[row, col_orig].set_title("Original", fontsize=8)

            axes[row, col_over].imshow(overlay)
            axes[row, col_over].axis("off")
            if col_pair == 0:
                axes[row, col_over].set_title("Grad-CAM", fontsize=8)

        axes[row, 0].set_ylabel(class_names[class_idx], fontsize=9, rotation=90, labelpad=4)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_gradcam_detail(
    model: nn.Module,
    loader: DataLoader,
    gradcam: GradCAM,
    class_names: List[str],
    device: torch.device,
    save_path: Optional[str] = None,
    title: str = "Grad-CAM Detail: One Image per Class",
) -> None:
    """
    Plot one correctly classified image per class with three columns:
    Original Image | Grad-CAM Heatmap | Overlay.

    Parameters
    ----------
    Same as plot_gradcam_grid, but shows one image per row in three columns.
    """
    model.eval()
    collected = {i: None for i in range(len(class_names))}

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            preds = outputs.argmax(dim=1).cpu()
            for j in range(len(labels)):
                label = labels[j].item()
                pred  = preds[j].item()
                if pred == label and collected[label] is None:
                    collected[label] = images[j]
            if all(v is not None for v in collected.values()):
                break

    num_classes = len(class_names)
    fig, axes = plt.subplots(num_classes, 3, figsize=(9, num_classes * 2.2))
    fig.suptitle(title, fontsize=12)

    col_titles = ["Original Image", "Grad-CAM Heatmap", "Overlay"]
    for col, ctitle in enumerate(col_titles):
        axes[0, col].set_title(ctitle, fontsize=10)

    for row, class_idx in enumerate(range(num_classes)):
        tensor = collected[class_idx]
        inp = tensor.unsqueeze(0).to(device)
        inp.requires_grad_(True)

        heatmap  = generate_heatmap(gradcam, inp, target_class=class_idx)
        original = tensor_to_image(tensor)
        overlay  = overlay_heatmap(original, heatmap)

        axes[row, 0].imshow(original)
        axes[row, 1].imshow(heatmap, cmap="jet")
        axes[row, 2].imshow(overlay)

        for col in range(3):
            axes[row, col].axis("off")

        axes[row, 0].set_ylabel(class_names[class_idx], fontsize=9, rotation=90, labelpad=4)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()
