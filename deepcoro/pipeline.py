"""Composable inference stages for the original DeepCoro stenosis models."""
from collections import Counter
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import torch

from .dicom import read_dicom
from .geometry import create_cropped_registered_video, register, resize_coordinates
from .hub import DEFAULT_REPO_ID, REGRESSION_FILE, SEGMENTATION_FILES, resolve_models
from .models import load_model, load_regression_model
from .preprocessing import perform_segmentation_inference
from .segments import get_segment_center

logger = logging.getLogger(__name__)
INPUT_COLUMNS = ("dicom_path", "artery_view", "frame", "x1", "y1", "x2", "y2")
OUTPUT_COLUMNS = ("dicom_path", "video_path", "frame", "box", "box_resized",
                  "artery_segment", "percent_stenosis", "severe_stenosis")
SEGMENT_PRIORITY = ("prox_rca", "mid_rca", "dist_rca", "pda", "posterolateral",
                    "leftmain", "lad", "mid_lad", "dist_lad", "lcx", "dist_lcx")
ARTERY_LABELS = {"leftmain": 0, "lad": 1, "lcx": 2, "mid_lad": 3, "dist_lcx": 4,
                 "dist_lad": 5, "prox_rca": 6, "mid_rca": 7, "dist_rca": 8,
                 "pda": 9, "posterolateral": 10}
SEVERE_THRESHOLD = 0.23


def read_annotations(path: str | Path) -> pd.DataFrame:
    """Read manual annotations without silently truncating fractional indices."""
    frame = pd.read_csv(path)
    missing = set(INPUT_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing input CSV columns: {', '.join(sorted(missing))}")
    frame = frame.loc[:, INPUT_COLUMNS].copy()
    if frame.isna().any().any():
        raise ValueError("Input CSV contains missing annotation values")
    frame["dicom_path"] = frame["dicom_path"].astype(str).str.strip()
    if (frame["dicom_path"] == "").any():
        raise ValueError("dicom_path must not be empty")
    frame["artery_view"] = frame["artery_view"].astype(str).str.strip().str.upper()
    if not frame["artery_view"].isin(("RCA", "LCA")).all():
        raise ValueError("artery_view must be RCA or LCA")
    for column in INPUT_COLUMNS[2:]:
        try:
            values = pd.to_numeric(frame[column], errors="raise")
            valid = np.isfinite(values) & (values >= 0) & (values < 2**63) & (values % 1 == 0)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{column} must contain nonnegative integers") from exc
        if not valid.all():
            raise ValueError(f"{column} must contain nonnegative integers")
        frame[column] = values.astype("int64")
    if ((frame.x1 >= frame.x2) | (frame.y1 >= frame.y2)).any():
        raise ValueError("Annotation boxes must have x1 < x2 and y1 < y2")
    if (frame.groupby("dicom_path")["artery_view"].nunique() > 1).any():
        raise ValueError("Each DICOM must have exactly one artery_view")
    return frame


class _ModelBundle:
    """Cache weights once per run, offloading unused models on accelerators."""
    def __init__(self, root: Path, device: str):
        self.root = root
        self.device = device
        self._segmentation = None
        self._regression = None

    def segmentation(self):
        if self._regression is not None and self.device != "cpu":
            self._regression.to("cpu")
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
        if self._segmentation is None:
            self._segmentation = [load_model(self.root / path, self.device) for path in SEGMENTATION_FILES]
        else:
            for model in self._segmentation:
                model.to(self.device)
        return self._segmentation

    def regression(self):
        if self._segmentation is not None and self.device != "cpu":
            for model in self._segmentation:
                model.to("cpu")
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
        if self._regression is None:
            self._regression = load_regression_model(self.root / REGRESSION_FILE, self.device)
        else:
            self._regression.to(self.device)
        return self._regression


def register_annotations(pixels, annotations, pixel_spacing):
    """Associate each box with its own DICOM-local reference frame."""
    _, height, width = pixels.shape
    stenoses = []
    for row in annotations.itertuples(index=False):
        if row.frame >= len(pixels):
            raise ValueError(f"Frame {row.frame} is outside the {len(pixels)}-frame DICOM")
        if row.x2 > width or row.y2 > height:
            raise ValueError(f"Box {(row.x1, row.y1, row.x2, row.y2)} exceeds image dimensions")
        box = (int(row.x1), int(row.y1), int(row.x2), int(row.y2))
        resized = resize_coordinates(*box, pixel_spacing, pixels.shape, 17.5)
        stenoses.append({"frame": int(row.frame), "box": box, "box_resized": resized,
                         "reg_shift": register(pixels, int(row.frame), *resized)})
    return stenoses


def _tracked_region(labels, box, shift):
    """Extract a moving box, using background outside the image boundaries."""
    x1, y1, x2, y2 = box
    start_x, start_y = int(x1 - shift[1]), int(y1 - shift[0])
    height, width = labels.shape
    region = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    src_x1, src_y1 = max(start_x, 0), max(start_y, 0)
    src_x2, src_y2 = min(start_x + region.shape[1], width), min(start_y + region.shape[0], height)
    if src_x2 > src_x1 and src_y2 > src_y1:
        region[src_y1 - start_y:src_y2 - start_y, src_x1 - start_x:src_x2 - start_x] = labels[src_y1:src_y2, src_x1:src_x2]
    return region


def identify_segments(stenoses, labels, artery_view):
    """Vote across frames and keep the middle annotation per artery segment."""
    by_segment = {}
    for stenosis in stenoses:
        votes = [get_segment_center(_tracked_region(label_frame, stenosis["box_resized"],
                                                    stenosis["reg_shift"][index]), artery_view)
                 for index, label_frame in enumerate(labels)]
        counts = Counter(vote for vote in votes if vote not in ("None", "other"))
        if not counts:
            continue
        segment = max(counts, key=lambda key: (counts[key], -SEGMENT_PRIORITY.index(key)))
        stenosis["artery_segment"] = segment
        by_segment.setdefault(segment, []).append(stenosis)
    return [by_segment[key][len(by_segment[key]) // 2] for key in sorted(by_segment)]


def _save_video(path, pixels, stenosis, fps):
    import cv2
    import imageio.v2 as imageio
    from scipy.ndimage import shift

    x1, y1, x2, y2 = stenosis["box_resized"]
    text = f"{stenosis['artery_segment']} - {int(stenosis['percent_stenosis'])}%"
    # Stream frames so output video writing does not duplicate the entire acquisition.
    with imageio.get_writer(path, fps=fps) as writer:
        for index, pixels_frame in enumerate(pixels):
            registered = shift(pixels_frame.astype(float), stenosis["reg_shift"][index], order=5)
            registered = np.clip(registered, 0, 255).astype(np.uint8)
            cv2.rectangle(registered, (x1, y1), (x2, y2), 255, 1)
            cv2.putText(registered, text, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, 255)
            writer.append_data(registered)


def deepcoro(input_file_path, save_dir, models_dir=None, *, repo_id=DEFAULT_REPO_ID,
             revision=None, cache_dir=None, local_files_only=False, device=None,
             save_videos=True) -> pd.DataFrame:
    """Run inference with local weights or a cached Hugging Face model snapshot.

    Relative DICOM paths are resolved from the CSV's parent directory. The returned
    table is also written to ``save_dir/df_stenosis.csv`` without a synthetic index.
    """
    annotations = read_annotations(input_file_path)
    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(columns=OUTPUT_COLUMNS)
    if annotations.empty:
        results.to_csv(output_dir / "df_stenosis.csv", index=False)
        return results
    selected_device = str(torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu")))
    models_root = resolve_models(models_dir, repo_id=repo_id, revision=revision,
                                 cache_dir=cache_dir, local_files_only=local_files_only)
    models = _ModelBundle(models_root, selected_device)
    rows = []
    with torch.inference_mode():
        for dicom_id, (dicom_path, group) in enumerate(annotations.groupby("dicom_path", sort=False), 1):
            path = Path(dicom_path)
            if not path.is_absolute():
                path = Path(input_file_path).resolve().parent / path
            acquisition = read_dicom(path)
            pixels = acquisition.pixels
            stenoses = register_annotations(pixels, group, acquisition.pixel_spacing)
            ensemble = models.segmentation()
            labels = np.empty(pixels.shape, dtype=np.uint8)
            for index, image in enumerate(pixels):
                labels[index] = perform_segmentation_inference(ensemble, image, selected_device)
            selected = identify_segments(stenoses, labels, group["artery_view"].iloc[0])
            del labels
            if selected:
                regressor = models.regression()
            for stenosis in selected:
                segment = stenosis["artery_segment"]
                video = create_cropped_registered_video(pixels, stenosis["frame"], stenosis["reg_shift"], *stenosis["box_resized"])
                age = torch.tensor([[acquisition.patient_age]], dtype=torch.float64, device=selected_device)
                encoded_segment = torch.tensor([[ARTERY_LABELS[segment]]], device=selected_device)
                probability = regressor([video.to(selected_device, dtype=torch.float64), age, encoded_segment]).item()
                if not np.isfinite(probability) or not 0 <= probability <= 1:
                    raise ValueError(f"Model returned invalid stenosis probability: {probability}")
                stenosis["percent_stenosis"] = probability * 100
                stenosis["severe_stenosis"] = probability >= SEVERE_THRESHOLD
                stenosis["dicom_path"] = dicom_path
                stenosis["video_path"] = None
                if save_videos:
                    severity = "severe" if stenosis["severe_stenosis"] else "nonsevere"
                    filename = f"DICOM{dicom_id}_{segment}_{int(probability * 100)}_{severity}_stenosis.mp4"
                    video_path = output_dir / filename
                    _save_video(video_path, pixels, stenosis, acquisition.fps)
                    stenosis["video_path"] = str(video_path)
                rows.append({column: stenosis[column] for column in OUTPUT_COLUMNS})
            results = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
            results.to_csv(output_dir / "df_stenosis.csv", index=False)
            logger.info("Finished DICOM %d (%d artery segments)", dicom_id, len(selected))
    return results


run_inference = deepcoro
