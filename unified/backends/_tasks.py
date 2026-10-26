"""Task implementations executed inside each experiment's own interpreter.

Each experiment lives in a folder with its own virtual environment (exp5 ships
TensorFlow 2.21 + Keras 3.12, exp6 ships PyTorch). The unified app is not able to
load every framework in a single process, so these tasks are invoked through
``unified/worker.py`` using the interpreter that belongs to the experiment.

Every task takes a single ``payload`` dict and returns JSON-serialisable data.
"""
import base64
import io
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_modules = {}


def _load(name, relative_path):
    """Import an experiment script by path, caching the module."""
    if name not in _modules:
        import importlib.util

        file_path = PROJECT_ROOT / relative_path
        if not file_path.is_file():
            raise FileNotFoundError(f"Experiment file not found: {file_path}")

        spec = importlib.util.spec_from_file_location(name, str(file_path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        _modules[name] = module
    return _modules[name]


def _to_base64(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ============================================================
# EXPERIMENT 5 - VAE
# ============================================================

def vae_generate(payload):
    from PIL import Image

    module = _load("exp5_vae_app", "exp5/app.py")
    import numpy as np

    count = max(1, min(int(payload.get("count", 9)), 25))
    latent_vectors = np.random.normal(size=(count, 2))
    generated = module.decoder.predict(latent_vectors, verbose=0)

    images = []
    for i in range(count):
        array = (generated[i].squeeze() * 255).clip(0, 255).astype(np.uint8)
        images.append(
            {
                "image": _to_base64(Image.fromarray(array)),
                "z1": round(float(latent_vectors[i, 0]), 3),
                "z2": round(float(latent_vectors[i, 1]), 3),
            }
        )
    return {"images": images}


def vae_reconstruct(payload):
    from PIL import Image

    module = _load("exp5_vae_app", "exp5/app.py")
    import numpy as np

    raw = base64.b64decode(payload["image"])
    image = Image.open(io.BytesIO(raw))

    processed = module.preprocess_image(image)
    z_mean, _z_log_var, z = module.encoder.predict(
        processed.reshape(1, 28, 28, 1), verbose=0
    )
    reconstructed = module.decoder.predict(z, verbose=0)
    array = (reconstructed[0].squeeze() * 255).clip(0, 255).astype(np.uint8)

    return {
        "input": _to_base64(Image.fromarray((processed * 255).astype(np.uint8))),
        "reconstruction": _to_base64(Image.fromarray(array)),
        "z1": round(float(z_mean[0, 0]), 4),
        "z2": round(float(z_mean[0, 1]), 4),
        "latent_graph": module.create_latent_visualization(z_mean[0]),
    }


def vae_decode(payload):
    from PIL import Image

    module = _load("exp5_vae_app", "exp5/app.py")
    import numpy as np

    z1 = float(payload.get("z1", 0.0))
    z2 = float(payload.get("z2", 0.0))
    point = np.array([[z1, z2]], dtype=np.float32)
    generated = module.decoder.predict(point, verbose=0)
    array = (generated[0].squeeze() * 255).clip(0, 255).astype(np.uint8)
    return {"image": _to_base64(Image.fromarray(array)), "z1": z1, "z2": z2}


# ============================================================
# EXPERIMENT 6 - DCGAN
# ============================================================

def gan_generate(payload):
    import torch
    from PIL import Image

    module = _load("exp6_gan_app", "exp6/app.py")

    count = max(1, min(int(payload.get("count", 4)), 8))
    with torch.no_grad():
        noise = torch.randn(count, module.NZ, 1, 1, device=module.device)
        outputs = module.generator(noise).cpu()

    images = []
    for tensor in outputs:
        array = tensor.permute(1, 2, 0).numpy()
        array = ((array + 1.0) / 2.0 * 255.0).clip(0, 255).astype("uint8")
        images.append(_to_base64(Image.fromarray(array, mode="RGB")))
    return {"images": images}


# ============================================================
# EXPERIMENT 3 - FLAN-T5
# ============================================================

def qa_answer(payload):
    module = _load("exp3_qa_app", "exp3/qa_gradio.py")
    question = (payload.get("question") or "").strip()
    if not question:
        raise ValueError("Please enter a question.")
    return {"answer": module.answer_question(question, int(payload.get("max_tokens", 128)))}


# ============================================================
# EXPERIMENT 4 - SDXL
# ============================================================

def sdxl_generate(payload):
    module = _load("exp4_sdxl_app", "exp4/text_to_image_gradio.py")
    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("Please enter a text prompt first.")

    image = module.generate_image(
        prompt,
        (payload.get("negative_prompt") or "").strip() or None,
        int(payload.get("steps", 25)),
        float(payload.get("guidance_scale", 7.0)),
        int(payload.get("seed", -1)),
    )
    return {"image": _to_base64(image)}


# ============================================================
# EXPERIMENT 7 - StyleGAN (random-init architecture demo)
# ============================================================

def stylegan_generate(payload):
    import torch
    from PIL import Image

    module = _load("exp7_stylegan", "exp7/models/stylegan.py")

    count = max(1, min(int(payload.get("count", 8)), 16))
    image_size = int(payload.get("image_size", 64))
    z_dim = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"

    generator = module.Generator(z_dim, z_dim, image_size).to(device).eval()
    with torch.no_grad():
        z = torch.randn(count, z_dim, device=device)
        outputs = ((generator(z) + 1.0) / 2.0).clamp(0, 1).cpu()

    images = []
    for tensor in outputs:
        array = (tensor.permute(1, 2, 0).numpy() * 255.0).astype("uint8")
        images.append(_to_base64(Image.fromarray(array, mode="RGB")))
    return {"images": images}


# ============================================================
# EXPERIMENT 8 - MusicGen text-to-music
# ============================================================

def music_generate(payload):
    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("Please enter a music prompt.")

    try:
        from audiocraft.models import MusicGen
        from audiocraft.data.audio import audio_write
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "AudioCraft is not installed in this environment. Install exp8/requirements.txt "
            "and free ~2 GB of disk for the MusicGen weights, then retry."
        ) from exc

    import torch

    duration = max(1, min(int(payload.get("duration", 8)), 20))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MusicGen.get_pretrained("facebook/musicgen-small", device=device)
    model.set_generation_params(duration=duration)
    wav = model.generate([prompt], progress=False)

    output_dir = PROJECT_ROOT / "exp8" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / f"music_{int(time.time())}"
    audio_write(str(base), wav[0].cpu(), model.sample_rate, strategy="loudness")
    return {"audio": base.name + ".wav"}


# ============================================================
# EXPERIMENT 9 - RAG personal assistant
# ============================================================

def assistant_answer(payload):
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    module = _load("exp9_assistant", "exp9/assistant_rag.py")

    question = (payload.get("question") or "").strip()
    if not question:
        raise ValueError("Please enter a question.")

    document = (payload.get("document") or "").strip()
    use_rag = bool(payload.get("use_rag", True)) and bool(document)

    assistant = module.RAGAssistant()
    if document:
        scratch = PROJECT_ROOT / "exp9" / "outputs" / "session_document.txt"
        scratch.parent.mkdir(parents=True, exist_ok=True)
        scratch.write_text(document, encoding="utf-8")
        assistant.add_document(scratch)

    answer, sources = assistant.answer(
        question, use_rag=use_rag, max_new_tokens=int(payload.get("max_tokens", 180))
    )
    return {"answer": answer.strip(), "sources": [s[:400] for s in sources]}


# ============================================================
# EXPERIMENT 10 - text-to-video
# ============================================================

def video_generate(payload):
    import imageio
    import torch
    from diffusers import DiffusionPipeline

    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("Please enter a text prompt.")

    model_id = "damo-vilab/text-to-video-ms-1.7b"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(device)

    result = pipe(
        prompt,
        num_frames=max(8, int(payload.get("frames", 16))),
        num_inference_steps=max(1, int(payload.get("steps", 25))),
    )

    output_dir = PROJECT_ROOT / "exp10" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"generated_{int(time.time())}.mp4"
    imageio.mimsave(
        output_dir / filename, list(result.frames[0]), fps=8, macro_block_size=1
    )
    return {"video": filename}


# ============================================================
# REGISTRY
# ============================================================

TASKS = {
    "vae_generate": vae_generate,
    "vae_reconstruct": vae_reconstruct,
    "vae_decode": vae_decode,
    "gan_generate": gan_generate,
    "qa_answer": qa_answer,
    "sdxl_generate": sdxl_generate,
    "stylegan_generate": stylegan_generate,
    "music_generate": music_generate,
    "assistant_answer": assistant_answer,
    "video_generate": video_generate,
}
