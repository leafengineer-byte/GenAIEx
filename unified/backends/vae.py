"""Experiment 5 - Variational Autoencoder (MNIST).

The model runs inside ``exp5/venv`` (TensorFlow 2.21 + Keras 3.12), because the
saved ``.keras`` files require that Keras version.
"""
import base64

from ._runner import run_task


def generate(count) -> list:
    """Sample random latent vectors and decode them into digit images."""
    result = run_task("vae", "vae_generate", {"count": int(count)})
    return result["images"]


def reconstruct(file_stream) -> dict:
    """Encode an uploaded digit, decode it back and plot its latent position."""
    image_base64 = base64.b64encode(file_stream.read()).decode("utf-8")
    return run_task("vae", "vae_reconstruct", {"image": image_base64})


def decode_point(z1, z2) -> dict:
    """Decode an explicit (z1, z2) latent coordinate into a digit image."""
    return run_task("vae", "vae_decode", {"z1": float(z1), "z2": float(z2)})
