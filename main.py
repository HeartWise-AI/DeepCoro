"""Backward-compatible entry point for DeepCoro inference."""
from deepcoro.cli import main


def deepcoro(*args, **kwargs):
    from deepcoro.pipeline import deepcoro as run
    return run(*args, **kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
