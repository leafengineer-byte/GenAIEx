"""Experiment 7 - StyleGAN-inspired generator.

No trained checkpoint ships with exp7, so this runs the architecture with random
weights as an educational demo of latent/style/noise behaviour.
"""
from ._runner import run_task


def generate(count=8, image_size=64) -> list:
    """Generate sample images from the untrained generator."""
    result = run_task(
        "stylegan",
        "stylegan_generate",
        {"count": int(count), "image_size": int(image_size)},
    )
    return result["images"]
