from pathlib import Path

import pytest

from deepcoro.hub import DEFAULT_REPO_ID, DEFAULT_REVISION, MODEL_FILES, resolve_models


def create_models(root: Path) -> Path:
    for filename in MODEL_FILES:
        path = root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"checkpoint")
    return root


def test_local_weights_do_not_call_hub(tmp_path, monkeypatch):
    def fail(**kwargs):
        pytest.fail("Local weights must never access the Hub")
    monkeypatch.setattr("huggingface_hub.snapshot_download", fail)
    root = create_models(tmp_path / "weights")
    assert resolve_models(root) == root


def test_local_clone_root_and_missing_checkpoint(tmp_path):
    root = create_models(tmp_path / "models")
    assert resolve_models(tmp_path) == root
    (root / MODEL_FILES[0]).unlink()
    with pytest.raises(FileNotFoundError, match=MODEL_FILES[0]):
        resolve_models(root)


def test_default_snapshot_is_pinned_selective_and_offline_capable(tmp_path, monkeypatch):
    root = create_models(tmp_path / "snapshot" / "models")
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        return str(root.parent)
    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    assert resolve_models(cache_dir=tmp_path / "cache", local_files_only=True) == root
    assert calls == [{
        "repo_id": DEFAULT_REPO_ID,
        "revision": DEFAULT_REVISION,
        "cache_dir": str(tmp_path / "cache"),
        "local_files_only": True,
        "allow_patterns": [f"models/{name}" for name in MODEL_FILES],
    }]


def test_custom_revision_is_forwarded(tmp_path, monkeypatch):
    root = create_models(tmp_path / "models")
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path)
    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    resolve_models(repo_id="example/weights", revision="release-v2")
    assert calls[0]["repo_id"] == "example/weights"
    assert calls[0]["revision"] == "release-v2"
    resolve_models(repo_id="example/weights")
    assert calls[1]["revision"] == "main"


@pytest.mark.parametrize("content", [b"", b"version https://git-lfs.github.com/spec/v1\n"])
def test_incomplete_local_weights_are_rejected(tmp_path, content):
    root = create_models(tmp_path)
    (root / MODEL_FILES[0]).write_bytes(content)
    with pytest.raises(ValueError, match="checkpoint|Git LFS"):
        resolve_models(root)
