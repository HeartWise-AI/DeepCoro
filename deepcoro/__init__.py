"""DeepCoro coronary angiography inference."""
__version__ = "0.2.0"


def run_inference(*args, **kwargs):
    """Run the DeepCoro pipeline (dependencies are imported on first use)."""
    from .pipeline import deepcoro
    return deepcoro(*args, **kwargs)
