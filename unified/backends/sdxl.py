"""Experiment 4 - Stable Diffusion XL text-to-image.

The SDXL pipeline downloads several gigabytes on first use.
"""
from ._models import require_heavy_hardware, require_model
from ._runner import run_task

SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"


def generate(prompt, negative_prompt="", steps=25, guidance_scale=7.0, seed=-1) -> str:
    """Generate a single image from a text prompt and return it as base64 PNG."""
    text = (prompt or "").strip()
    if not text:
        raise ValueError("Please enter a text prompt first.")

    require_heavy_hardware("Stable Diffusion XL")
    require_model(SDXL_MODEL, 7.0)

    result = run_task(
        "sdxl",
        "sdxl_generate",
        {
            "prompt": text,
            "negative_prompt": negative_prompt or "",
            "steps": int(steps),
            "guidance_scale": float(guidance_scale),
            "seed": int(seed),
        },
    )
    return result["image"]
