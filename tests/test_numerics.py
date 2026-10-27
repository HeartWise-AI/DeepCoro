import numpy as np
import pytest
import torch

from deepcoro.geometry import create_cropped_registered_video, register, resize_coordinates
from deepcoro.models import load_state_dict
from deepcoro.preprocessing import perform_segmentation_inference, process_image
from deepcoro.segments import get_segment_center


def test_physical_crop_uses_width_for_x_and_height_for_y():
    assert resize_coordinates(80, 10, 90, 20, 1, (4, 40, 100), 20) == (75, 5, 95, 25)
    assert resize_coordinates(80, 30, 90, 40, 1, (4, 40, 100), 20) == (75, 20, 95, 40)


def test_crop_larger_than_image_stays_square_and_bounded():
    x1, y1, x2, y2 = resize_coordinates(2, 3, 8, 9, .1, (4, 10, 30), 17.5)
    assert 0 <= x1 < x2 <= 30
    assert 0 <= y1 < y2 <= 10
    assert x2 - x1 == y2 - y1 == 10


@pytest.mark.parametrize("spacing", [0, -1, float('nan'), float('inf')])
def test_invalid_pixel_spacing(spacing):
    with pytest.raises(ValueError, match="Pixel spacing"):
        resize_coordinates(0, 0, 2, 2, spacing, (2, 8, 8), 17.5)


def test_blank_image_normalization_is_finite():
    output = process_image(np.zeros((32, 32), dtype=np.uint8))
    assert output.shape == (1, 32, 32)
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output) == 0


def test_segment_selection_handles_tiny_and_rectangular_regions():
    assert get_segment_center(np.array([[1]]), "RCA") == "prox_rca"
    assert get_segment_center(np.zeros((0, 3)), "RCA") == "None"
    region = np.zeros((2, 8), dtype=np.uint8)
    region[1, 4] = 5
    assert get_segment_center(region, "LCA") == "leftmain"
    region[:] = 0
    region[0, 0] = 5
    assert get_segment_center(region, "LCA") == "leftmain"


@pytest.mark.parametrize("shape,padded_shape", [((8, 10), (128, 128)), ((129, 65), (160, 128)), ((224, 224), (224, 224))])
def test_segmentation_ensemble_averages_logits_without_gradients(shape, padded_shape):
    class Fixed(torch.nn.Module):
        def forward(self, image):
            assert not torch.is_grad_enabled()
            assert image.shape == (1, 3, *padded_shape)
            out = torch.zeros((1, 26, *padded_shape), device=image.device)
            out[:, 5] = 2
            return out
    output = perform_segmentation_inference([Fixed(), Fixed()], np.zeros(shape), "cpu")
    assert output.shape == shape
    assert np.all(output == 5)


def test_tensor_checkpoint_strips_dataparallel_prefix(tmp_path):
    path = tmp_path / 'weights.pt'
    expected = torch.tensor([1., 2.], dtype=torch.float64)
    torch.save({'state_dict': {'module.layer.weight': expected}, 'best_loss': np.float64(0.1)}, path)
    actual = load_state_dict(path)
    assert torch.equal(actual['layer.weight'], expected)
    assert actual['layer.weight'].dtype == torch.float64


def test_registered_crop_temporal_padding_and_channel_order():
    video = np.full((2, 4, 6), 145, dtype=np.uint8)
    output = create_cropped_registered_video(video, 0, {0: (0, 0), 1: (0, 0)}, 0, 0, 4, 4)
    assert output.shape == (1, 3, 24, 224, 224)
    assert output.dtype == torch.float64
    assert torch.count_nonzero(output[:, :, :12]) == 0
    assert torch.count_nonzero(output[:, :, 14:]) == 0
    assert torch.allclose(output[0, :, 12:14], torch.full((3, 2, 224, 224), (145 - 144.80116) / 52.669548, dtype=torch.float64))


def test_tracking_failure_reports_frame(monkeypatch):
    class FailedTracker:
        def init(self, image, box):
            assert image.shape == (8, 8, 3)
            return None
        def update(self, image):
            return False, (0, 0, 0, 0)
    monkeypatch.setattr('deepcoro.geometry._create_tracker', FailedTracker)
    with pytest.raises(RuntimeError, match='frame 1'):
        register(np.zeros((2, 8, 8), dtype=np.uint8), 0, 1, 1, 4, 4)


def test_checkpoint_rejects_non_dictionary(tmp_path):
    path = tmp_path / "invalid.pt"
    torch.save([torch.ones(1)], path)
    with pytest.raises(ValueError, match="state dictionary"):
        load_state_dict(path)
