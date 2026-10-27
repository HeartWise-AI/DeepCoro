"""Regression checks for CSV/DICOM grouping and end-to-end stage orchestration."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from deepcoro import pipeline
from deepcoro.dicom import get_age, read_dicom


def write_annotations(path, rows):
    pd.DataFrame(rows, columns=pipeline.INPUT_COLUMNS).to_csv(path, index=False)


def test_dicom_local_frames_and_csv_parent_paths(tmp_path, monkeypatch):
    csv = tmp_path / "arbitrary-name.csv"
    write_annotations(csv, [("a.dcm", "rca", 1, 2, 2, 5, 5),
                            ("b.dcm", "RCA", 4, 2, 2, 5, 5)])
    read_paths, tracked_frames, crop_frames, probabilities = [], [], [], iter([0.22, 0.23])
    weights = object()
    monkeypatch.setattr(pipeline, "resolve_models", lambda *a, **kw: tmp_path)

    def reader(path):
        read_paths.append(path)
        return SimpleNamespace(pixels=np.zeros((6, 8, 10), dtype=np.uint8),
                               pixel_spacing=1., fps=15, patient_age=60)

    def tracker(pixels, frame, *box):
        tracked_frames.append(frame)
        return {i: (0, 0) for i in range(len(pixels))}

    def crop(pixels, frame, shifts, *box):
        crop_frames.append(frame)
        return torch.zeros((1, 3, 24, 2, 2), dtype=torch.float64)

    class Bundle:
        def __init__(self, *a):
            self.calls = 0

        def segmentation(self):
            return weights

        def regression(self):
            def predict(inputs):
                assert torch.is_inference_mode_enabled()
                assert inputs[0].dtype == inputs[1].dtype == torch.float64
                return torch.tensor(next(probabilities), dtype=torch.float64)
            return predict

    monkeypatch.setattr(pipeline, "_ModelBundle", Bundle)
    monkeypatch.setattr(pipeline, "read_dicom", reader)
    monkeypatch.setattr(pipeline, "resize_coordinates", lambda *args: (2, 2, 5, 5))
    monkeypatch.setattr(pipeline, "register", tracker)
    monkeypatch.setattr(pipeline, "perform_segmentation_inference", lambda *a: np.ones((8, 10), dtype=np.uint8))
    monkeypatch.setattr(pipeline, "get_segment_center", lambda *a: "prox_rca")
    monkeypatch.setattr(pipeline, "create_cropped_registered_video", crop)
    result = pipeline.deepcoro(csv, tmp_path / "new" / "outputs", save_videos=False)
    assert tracked_frames == crop_frames == [1, 4]
    assert read_paths == [tmp_path / "a.dcm", tmp_path / "b.dcm"]
    assert result.frame.tolist() == [1, 4]
    assert result.percent_stenosis.tolist() == pytest.approx([22, 23])
    assert result.severe_stenosis.tolist() == [False, True]
    saved = pd.read_csv(tmp_path / "new" / "outputs" / "df_stenosis.csv")
    assert saved.columns.tolist() == list(pipeline.OUTPUT_COLUMNS)
    assert len(saved) == 2


def test_empty_csv_writes_schema_without_loading_weights(tmp_path, monkeypatch):
    csv = tmp_path / "empty.csv"
    write_annotations(csv, [])
    monkeypatch.setattr(pipeline, "resolve_models", lambda *a, **kw: pytest.fail("weights must not load"))
    result = pipeline.deepcoro(csv, tmp_path / "out")
    assert result.empty
    assert pd.read_csv(tmp_path / "out" / "df_stenosis.csv").columns.tolist() == list(pipeline.OUTPUT_COLUMNS)


def test_background_only_has_no_regression_or_middle_row_error():
    stenosis = {"frame": 0, "box_resized": (0, 0, 2, 2), "reg_shift": {0: (0, 0)}}
    assert pipeline.identify_segments([stenosis], np.zeros((1, 4, 5), dtype=np.uint8), "RCA") == []


@pytest.mark.parametrize("column,value", [("frame", -1), ("frame", 1.5), ("x1", float("inf")),
                                         ("artery_view", "OTHER"), ("dicom_path", " ")])
def test_invalid_csv_values_are_rejected(tmp_path, column, value):
    row = dict(zip(pipeline.INPUT_COLUMNS, ("a.dcm", "RCA", 0, 0, 0, 2, 2)))
    row[column] = value
    csv = tmp_path / "bad.csv"
    pd.DataFrame([row]).to_csv(csv, index=False)
    with pytest.raises(ValueError):
        pipeline.read_annotations(csv)


def test_mixed_views_for_one_dicom_are_rejected(tmp_path):
    csv = tmp_path / "mixed.csv"
    write_annotations(csv, [("a.dcm", "RCA", 0, 0, 0, 2, 2), ("a.dcm", "LCA", 1, 0, 0, 2, 2)])
    with pytest.raises(ValueError, match="one artery_view"):
        pipeline.read_annotations(csv)


def test_annotation_frame_and_box_bounds():
    pixels = np.zeros((2, 8, 10), dtype=np.uint8)
    for row, message in [(("a", "RCA", 2, 0, 0, 2, 2), "Frame"),
                         (("a", "RCA", 0, 0, 0, 11, 2), "Box")]:
        with pytest.raises(ValueError, match=message):
            pipeline.register_annotations(pixels, pd.DataFrame([row], columns=pipeline.INPUT_COLUMNS), 1.)


def test_tracked_box_far_outside_frame_is_background():
    region = pipeline._tracked_region(np.ones((3, 4)), (0, 0, 2, 2), (100, -100))
    assert region.shape == (2, 2)
    assert not region.any()


def test_age_completed_years_and_invalid_dates():
    assert get_age(SimpleNamespace(StudyDate="20261001", PatientBirthDate="19661002")) == 59
    with pytest.raises(ValueError, match="after StudyDate"):
        get_age(SimpleNamespace(StudyDate="20261001", PatientBirthDate="20271001"))
    with pytest.raises(ValueError, match="required"):
        get_age(SimpleNamespace(StudyDate="20261001"))


def test_dicom_rejects_fractional_framerate_and_bad_spacing(monkeypatch):
    dataset = SimpleNamespace(pixel_array=np.zeros((2, 3, 4)), SamplesPerPixel=1,
                              ImagerPixelSpacing=[.2, .2], DistanceSourceToDetector=100,
                              DistanceSourceToPatient=50, RecommendedDisplayFrameRate=15.1,
                              StudyDate="20261001", PatientBirthDate="19661002")
    monkeypatch.setattr("deepcoro.dicom.pydicom.dcmread", lambda path: dataset)
    with pytest.raises(ValueError, match="15 FPS"):
        read_dicom("test.dcm")
    dataset.RecommendedDisplayFrameRate = 15
    dataset.DistanceSourceToPatient = 0
    with pytest.raises(ValueError, match="positive and finite"):
        read_dicom("test.dcm")


def test_model_bundle_reuses_loaded_weights(tmp_path, monkeypatch):
    calls = []

    def load(path, device):
        calls.append(path)
        return torch.nn.Identity()

    monkeypatch.setattr(pipeline, "load_model", load)
    monkeypatch.setattr(pipeline, "load_regression_model", load)
    models = pipeline._ModelBundle(tmp_path, "cpu")
    first_ensemble = models.segmentation()
    first_regressor = models.regression()
    assert models.segmentation() is first_ensemble
    assert models.regression() is first_regressor
    assert len(calls) == len(pipeline.SEGMENTATION_FILES) + 1
