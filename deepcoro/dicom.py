"""DICOM decoding and acquisition metadata validation."""
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pydicom


@dataclass(frozen=True)
class DicomVideo:
    pixels: np.ndarray
    pixel_spacing: float
    fps: float
    patient_age: int


def get_age(dicom_info) -> int:
    """Return completed years at acquisition, using the original date fields."""
    try:
        study = datetime.strptime(str(dicom_info.StudyDate), "%Y%m%d").date()
        birth = datetime.strptime(str(dicom_info.PatientBirthDate), "%Y%m%d").date()
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Valid StudyDate and PatientBirthDate are required") from exc
    if birth > study:
        raise ValueError("PatientBirthDate must not be after StudyDate")
    return study.year - birth.year - ((study.month, study.day) < (birth.month, birth.day))


def read_dicom(path: str | Path) -> DicomVideo:
    """Decode a grayscale acquisition and validate the model's 15 FPS input."""
    dataset = pydicom.dcmread(path)
    if int(getattr(dataset, "SamplesPerPixel", 1)) != 1:
        raise ValueError(f"{path}: expected a grayscale DICOM video")
    pixels = dataset.pixel_array
    if pixels.ndim == 2:
        pixels = pixels[np.newaxis, ...]
    if pixels.ndim != 3 or any(size == 0 for size in pixels.shape):
        raise ValueError(f"{path}: expected nonempty frames × height × width pixels")
    try:
        spacing = float(dataset.ImagerPixelSpacing[0])
        detector = float(dataset.DistanceSourceToDetector)
        patient = float(dataset.DistanceSourceToPatient)
    except (AttributeError, TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"{path}: valid pixel spacing and source distances are required") from exc
    if not all(np.isfinite(x) and x > 0 for x in (spacing, detector, patient)):
        raise ValueError(f"{path}: pixel spacing and source distances must be positive and finite")
    try:
        fps = float(dataset.RecommendedDisplayFrameRate)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{path}: RecommendedDisplayFrameRate is required") from exc
    if fps != 15:
        raise ValueError(f"{path}: expected 15 FPS, received {fps}")
    return DicomVideo(pixels, spacing * patient / detector, fps, get_age(dataset))
