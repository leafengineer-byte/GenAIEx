"""Experiment 6 - DCGAN face generator.

The generator runs inside ``exp6/.venv`` (PyTorch).
"""
from ._runner import run_task


def generate(count) -> list:
    """Generate synthetic face images from random latent noise."""
    result = run_task("gan", "gan_generate", {"count": int(count)})
    return result["images"]
