"""Preprocessing and ensemble segmentation inference."""

import numpy as np


def histogram_equalization(image):
    import cv2
    import torch

    array = image.detach().cpu().numpy() if isinstance(image, torch.Tensor) else np.asarray(image)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError("Histogram equalization expects a grayscale frame")
    equalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(array.astype(np.uint8))
    return torch.from_numpy(equalized).unsqueeze(0)


def process_image(img):
    import torch

    if np.ndim(img) != 2 or np.size(img) == 0:
        raise ValueError("Segmentation expects a nonempty grayscale frame")
    image = histogram_equalization(np.asarray(img).astype(np.uint8)).to(torch.float32)
    low, high = image.min(), image.max()
    if high == low:
        return torch.zeros_like(image)
    return (image - low) / (high - low)


def perform_segmentation_inference(models, img, device):
    import torch

    if not models:
        raise ValueError("At least one segmentation model is required")
    image = process_image(img).repeat(1, 3, 1, 1).to(device)
    height, width = image.shape[-2:]
    # PAN's feature pyramid needs at least 128 pixels; every ensemble member
    # requires dimensions divisible by 32. Zero is normalized background.
    padded_height = max(128, ((height + 31) // 32) * 32)
    padded_width = max(128, ((width + 31) // 32) * 32)
    if (padded_height, padded_width) != (height, width):
        image = torch.nn.functional.pad(image, (0, padded_width - width, 0, padded_height - height))
    with torch.inference_mode():
        outputs = [model(image) for model in models]
        labels = torch.stack(outputs).mean(dim=0).argmax(1).squeeze(0)
        return labels[:height, :width].cpu().numpy()


# Kept here as a convenience for callers that group all preprocessing together.
from .geometry import create_cropped_registered_video  # noqa: E402,F401
