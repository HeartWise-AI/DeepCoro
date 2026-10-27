"""Physical crop geometry and temporal registration for angiography videos."""

import numpy as np
from scipy.ndimage import shift, uniform_filter1d


def resize_coordinates(x1, y1, x2, y2, pixel_spacing, img_shape, target_size):
    """Return a square physical crop, bounded by (frames, height, width)."""
    if len(img_shape) != 3 or min(img_shape) <= 0:
        raise ValueError("Expected a nonempty video with shape (frames, height, width)")
    if not np.isfinite(pixel_spacing) or pixel_spacing <= 0:
        raise ValueError("Pixel spacing must be positive and finite")
    if not np.isfinite(target_size) or target_size <= 0:
        raise ValueError("Target crop size must be positive and finite")
    if not all(np.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        raise ValueError("Crop coordinates must describe a nonempty finite rectangle")
    height, width = img_shape[-2:]
    xc, yc = np.floor((x2 + x1) / 2), np.floor((y2 + y1) / 2)
    half_size = target_size / pixel_spacing / 2
    # Preserve the original rounding for crops that fit within the image.
    left, right = int(np.floor(xc - half_size)), int(np.floor(xc + half_size))
    side = min(max(1, right - left), height, width)
    left = max(0, min(left, width - side))
    top = max(0, min(int(np.floor(yc - half_size)), height - side))
    return left, top, left + side, top + side


def _create_tracker():
    import cv2

    for namespace in (cv2, getattr(cv2, "legacy", None)):
        if namespace is None:
            continue
        factory = getattr(namespace, "TrackerCSRT_create", None)
        if factory is not None:
            return factory()
        tracker = getattr(namespace, "TrackerCSRT", None)
        if tracker is not None and hasattr(tracker, "create"):
            return tracker.create()
    raise RuntimeError("CSRT tracking requires opencv-contrib-python (headless is supported)")


def register(img, index, x1, y1, x2, y2):
    """Track a crop in both directions and return frame-indexed (dy, dx) shifts.

    Failed tracking raises an error instead of using invalid bounding boxes.
    """
    import cv2

    if img.ndim != 3 or not 0 <= index < len(img):
        raise ValueError("Expected grayscale video and an in-range reference frame")
    if not (0 <= x1 < x2 <= img.shape[2] and 0 <= y1 < y2 <= img.shape[1]):
        raise ValueError("Tracking crop must lie within the video")
    offsets = np.zeros((len(img), 2), dtype=float)

    def tracking_frame(frame):
        return cv2.cvtColor(np.clip(frame, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    for frame_indices in (range(index - 1, -1, -1), range(index + 1, len(img))):
        tracker = _create_tracker()
        initialized = tracker.init(tracking_frame(img[index]), tuple(int(v) for v in (x1, y1, x2 - x1, y2 - y1)))
        if initialized is False:
            raise RuntimeError("Could not initialize CSRT tracker")
        for frame_index in frame_indices:
            success, bbox = tracker.update(tracking_frame(img[frame_index]))
            if not success or not all(np.isfinite(v) for v in bbox):
                raise RuntimeError(f"CSRT tracking failed at frame {frame_index}")
            x, y, width, height = (int(v) for v in bbox)
            offsets[frame_index] = (
                -(int(y + height / 2 - (y2 - y1) / 2) - y1),
                -(int(x + width / 2 - (x2 - x1) / 2) - x1),
            )
    offsets = uniform_filter1d(offsets, size=5, axis=0, output=float)
    return {i: offsets[i].tolist() for i in range(len(img))}


def create_cropped_registered_video(dicom, frame, reg_shifts, x1, y1, x2, y2):
    """Create the original normalized, zero-padded 24-frame regression input."""
    import torch
    from torchvision.transforms import Normalize, Resize

    if dicom.ndim != 3 or not 0 <= frame < len(dicom):
        raise ValueError("Expected grayscale video and an in-range reference frame")
    if not (0 <= x1 < x2 <= dicom.shape[2] and 0 <= y1 < y2 <= dicom.shape[1]):
        raise ValueError("Regression crop must lie within the video")
    crop = np.empty((len(dicom), y2 - y1, x2 - x1), dtype=np.float64)
    for i in range(len(dicom)):
        crop[i] = shift(dicom[i], shift=reg_shifts[i], order=5)[y1:y2, x1:x2]
    images = torch.from_numpy(np.stack((crop,) * 3, axis=1))
    images = Resize((224, 224), antialias=False)(images)
    images = Normalize([144.80116] * 3, [52.669548] * 3)(images)
    padded = torch.cat((torch.zeros_like(images[:1]).repeat(12, 1, 1, 1), images,
                        torch.zeros_like(images[:1]).repeat(12, 1, 1, 1)), dim=0)
    return padded[frame:frame + 24].permute(1, 0, 2, 3).unsqueeze(0)
