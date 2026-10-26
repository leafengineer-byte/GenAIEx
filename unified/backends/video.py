"""Experiment 10 - text-to-video generation.

The video diffusion model downloads on first use and is memory heavy.
"""
from ._loader import PROJECT_ROOT
from ._models import require_heavy_hardware, require_model
from ._runner import run_task

OUTPUT_DIR = PROJECT_ROOT / "exp10" / "outputs"
VIDEO_MODEL = "damo-vilab/text-to-video-ms-1.7b"


def generate(prompt, frames=16, steps=25) -> str:
    """Generate a short video and return the saved file name."""
    text = (prompt or "").strip()
    if not text:
        raise ValueError("Please enter a text prompt.")

    require_heavy_hardware("Text-to-video")
    require_model(VIDEO_MODEL, 1.8)

    result = run_task(
        "video",
        "video_generate",
        {"prompt": text, "frames": int(frames), "steps": int(steps)},
    )
    return result["video"]
