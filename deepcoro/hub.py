"""Resolve the published DeepCoro checkpoints without embedding weights in code/images."""
from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_REPO_ID = "heartwise/DeepCoro"
# Verified upstream snapshot: all seven segmentation models and the regressor.
DEFAULT_REVISION = "13207073d84d6c605ecb9bfa95195458796d7e39"
SEGMENTATION_FILES = (
    "algorithm4/bs_64_lr_0.00107809_model_FPN_loss_LovaszLoss.pth",
    "algorithm4/bs_64_lr_0.00242160_model_DeepLabV3Plus_loss_LovaszLoss.pth",
    "algorithm4/bs_64_lr_0.00129894_model_PAN_loss_TverskyLoss.pth",
    "algorithm4/bs_4_lr_0.00059902_model_DeepLabV3_loss_TverskyLoss.pth",
    "algorithm4/bs_16_lr_0.00993245_model_FPN_loss_LovaszLoss.pth",
    "algorithm4/bs_16_lr_0.00232125_model_DeepLabV3_loss_LovaszLoss.pth",
    "algorithm4/bs_64_lr_0.00646546_model_PAN_loss_DiceLoss.pth",
)
REGRESSION_FILE = "algorithm6.pt"
MODEL_FILES = (*SEGMENTATION_FILES, REGRESSION_FILE)


def validate_models(models_dir: str | Path) -> Path:
    """Validate a model directory (or an existing Hub clone/snapshot root)."""
    root = Path(models_dir).expanduser().resolve()
    if not (root / REGRESSION_FILE).is_file() and (root / "models").is_dir():
        root = root / "models"
    missing = [name for name in MODEL_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing DeepCoro checkpoints in {root}: {', '.join(missing)}")
    for name in MODEL_FILES:
        path = root / name
        with path.open("rb") as checkpoint:
            header = checkpoint.read(128)
        if not header:
            raise ValueError(f"Empty checkpoint: {path}")
        if header.startswith(b"version https://git-lfs.github.com/spec/"):
            raise ValueError(f"Git LFS pointer instead of model weights: {path}. Download weights with deepcoro-download.")
    return root


def resolve_models(
    models_dir: str | Path | None = None,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> Path:
    """Use explicit local weights or fetch only the eight inference checkpoints.

    Hub credentials and cache configuration use huggingface_hub's standard
    environment/login handling. ``local_files_only`` prevents network access.
    The released default is pinned; custom repositories default to their main.
    """
    if models_dir is not None:
        return validate_models(models_dir)
    if not repo_id or not repo_id.strip():
        raise ValueError("repo_id must not be empty")
    from huggingface_hub import snapshot_download

    effective_revision = revision or (DEFAULT_REVISION if repo_id == DEFAULT_REPO_ID else "main")
    snapshot = snapshot_download(
        repo_id=repo_id,
        revision=effective_revision,
        cache_dir=str(Path(cache_dir).expanduser()) if cache_dir is not None else None,
        local_files_only=local_files_only,
        allow_patterns=[f"models/{name}" for name in MODEL_FILES],
    )
    return validate_models(snapshot)


def main(args: list[str] | None = None) -> None:
    """Pre-download a pinned snapshot for offline inference."""
    parser = argparse.ArgumentParser(description="Download/cache DeepCoro's eight published inference checkpoints")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--models-dir", type=Path, help="Validate existing local checkpoints instead of downloading")
    parser.add_argument("--local-files-only", action="store_true")
    options = parser.parse_args(args)
    print(resolve_models(**vars(options)))


if __name__ == "__main__":
    main()
