"""Experiment 8 - MusicGen text-to-music.

Requires AudioCraft and the MusicGen weights, so the first run needs a large
download; the task reports a clear error when they are unavailable.
"""
from ._loader import PROJECT_ROOT
from ._models import audiocraft_available, require_heavy_hardware, require_model
from ._runner import run_task

OUTPUT_DIR = PROJECT_ROOT / "exp8" / "outputs"
MUSIC_MODEL = "facebook/musicgen-small"


def generate(prompt, duration=8) -> str:
    """Generate a short music clip and return the saved WAV file name."""
    text = (prompt or "").strip()
    if not text:
        raise ValueError("Please enter a music prompt.")

    require_heavy_hardware("MusicGen")
    if not audiocraft_available():
        raise RuntimeError(
            "AudioCraft is not installed in this environment. Run "
            "'pip install -r exp8/requirements.txt', then retry."
        )
    require_model(MUSIC_MODEL, 2.0)

    result = run_task(
        "music", "music_generate", {"prompt": text, "duration": int(duration)}
    )
    return result["audio"]
