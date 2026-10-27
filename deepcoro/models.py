"""Checkpoint-compatible model constructors without implicit weight downloads."""

from pathlib import Path
import re
import importlib

import numpy as np

import torch
from torch import nn


SEGMENTATION_ARCHITECTURES = (
    "Unet", "UnetPlusPlus", "MAnet", "Linknet", "FPN", "PSPNet",
    "DeepLabV3", "DeepLabV3Plus", "PAN",
)


def choose_model(model_name):
    import segmentation_models_pytorch as smp

    if model_name not in SEGMENTATION_ARCHITECTURES:
        raise ValueError(f"Invalid segmentation model name: {model_name}")
    return getattr(smp, model_name)(encoder_weights=None, in_channels=3, classes=26)


def load_state_dict(checkpoint_path, device="cpu"):
    """Read tensor-only checkpoints and normalize DataParallel key prefixes."""
    path = Path(checkpoint_path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        checkpoint = load_file(str(path), device=str(device))
    else:
        # The original published files include NumPy scalar training metrics.
        # Allow only their numeric constructors, never arbitrary checkpoint globals.
        numpy_core = "numpy._core.multiarray" if np.lib.NumpyVersion(np.__version__) >= "2.0.0" else "numpy.core.multiarray"
        scalar = importlib.import_module(numpy_core).scalar
        numeric_globals = [
            (scalar, "numpy.core.multiarray.scalar"),
            (scalar, "numpy._core.multiarray.scalar"),
            np.dtype, type(np.dtype(np.float64)), type(np.dtype(np.float32)),
        ]
        with torch.serialization.safe_globals(numeric_globals):
            checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must contain a state dictionary: {path}")
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError(f"Checkpoint does not contain a nonempty state dict: {path}")
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def load_model(checkpoint_path, device):
    match = re.search(r"_model_([^_]+)_", Path(checkpoint_path).name)
    if match is None:
        raise ValueError("Segmentation checkpoint name must include _model_<architecture>_")
    model = choose_model(match.group(1))
    model.load_state_dict(load_state_dict(checkpoint_path))
    return model.to(device).eval()


class Identity(nn.Module):
    def forward(self, x):
        return x


class Regressor(nn.Module):
    def __init__(self, num_class=1, num_features=12):
        super().__init__()
        from torchvision.models.video import swin3d_b

        model = swin3d_b(weights=None)
        n_inputs = model.head.in_features
        model.head = Identity()
        self.model = model
        self.fc = nn.Linear(n_inputs + num_features, num_class)
        self.layer = nn.Sigmoid()

    def forward(self, x):
        features = self.model(x[0])
        age = (x[1] / 120).to(dtype=features.dtype)
        segment = torch.nn.functional.one_hot(x[2].reshape(-1).long(), num_classes=11)
        segment = segment.to(dtype=features.dtype)
        return self.layer(self.fc(torch.cat([features, segment, age], dim=1)))


def get_model():
    return Regressor()


def load_regression_model(checkpoint_path, device):
    model = get_model().double()
    model.load_state_dict(load_state_dict(checkpoint_path))
    return model.to(device).double().eval()
