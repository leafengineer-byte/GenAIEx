"""Unified GenAI Experiments UI.

A single Flask app that exposes every experiment in this repository through one
interface. Each experiment keeps its own code; the modules in ``unified/backends``
reuse them and load their models lazily, so nothing heavy is initialised until a
feature is actually used.

Run from the project root:

    python app.py

Then open http://127.0.0.1:8000
"""
import os
import sys

from flask import Flask, jsonify, render_template, request, send_from_directory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unified.backends import assistant, gan, music, qa, sdxl, stylegan, vae, video  # noqa: E402
from unified.backends import _runner  # noqa: E402
from unified.backends._loader import PROJECT_ROOT  # noqa: E402
from unified.backends._models import (  # noqa: E402
    audiocraft_available,
    hf_cached,
    heavy_models_supported,
)

app = Flask(
    __name__,
    template_folder="unified/templates",
    static_folder="unified/static",
)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024


def _ok(**payload):
    payload["success"] = True
    return jsonify(payload)


def _fail(message, status=500):
    return jsonify({"success": False, "error": str(message)}), status


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    """Report which experiments are ready to run and on which runtime."""
    vae_ready = (
        _runner.VAE_PYTHON.is_file()
        and (PROJECT_ROOT / "exp5" / "models" / "vae_encoder.keras").is_file()
        and (PROJECT_ROOT / "exp5" / "models" / "vae_decoder.keras").is_file()
    )
    gan_ready = (
        _runner.GAN_PYTHON.is_file()
        and (PROJECT_ROOT / "exp6" / "generator.pth").is_file()
    )
    default_ready = _runner.DEFAULT_PYTHON.is_file()
    heavy_ready = heavy_models_supported()
    heavy_note = (
        "needs a GPU or 16 GB RAM · not runnable on this PC"
        if not heavy_ready
        else "needs a large model download on first run"
    )

    return _ok(
        status={
            "vae": {
                "ready": vae_ready,
                "runtime": "exp5/venv · TensorFlow + Keras",
                "note": "trained VAE · works offline",
            },
            "gan": {
                "ready": gan_ready,
                "runtime": "exp6/.venv · PyTorch",
                "note": "trained DCGAN · works offline",
            },
            "stylegan": {
                "ready": default_ready,
                "runtime": ".venv-unified · PyTorch",
                "note": "random-init demo · no trained weights",
            },
            "qa": {
                "ready": default_ready and hf_cached("google/flan-t5-base"),
                "runtime": ".venv-unified · Transformers",
                "note": "needs flan-t5-base download (~1 GB)",
            },
            "sdxl": {
                "ready": default_ready
                and heavy_ready
                and hf_cached("stabilityai/stable-diffusion-xl-base-1.0"),
                "runnable": heavy_ready,
                "runtime": ".venv-unified · Diffusers",
                "note": heavy_note,
            },
            "music": {
                "ready": default_ready
                and heavy_ready
                and audiocraft_available()
                and hf_cached("facebook/musicgen-small"),
                "runnable": heavy_ready,
                "runtime": ".venv-unified · AudioCraft",
                "note": heavy_note,
            },
            "assistant": {
                "ready": default_ready
                and hf_cached("Qwen/Qwen2.5-0.5B-Instruct")
                and hf_cached("sentence-transformers/all-MiniLM-L6-v2"),
                "runtime": ".venv-unified · Transformers + FAISS",
                "note": "works offline · cached model",
            },
            "video": {
                "ready": default_ready
                and heavy_ready
                and hf_cached("damo-vilab/text-to-video-ms-1.7b"),
                "runnable": heavy_ready,
                "runtime": ".venv-unified · Diffusers",
                "note": heavy_note,
            },
        }
    )


# ============================================================
# EXPERIMENT 1 - documentation (PDF)
# ============================================================

@app.route("/docs/exp1")
def exp1_doc():
    return send_from_directory(PROJECT_ROOT / "exp1", "vertopal.com_GenAIex1.pdf")


# ============================================================
# EXPERIMENT 3 - FLAN-T5 question answering
# ============================================================

@app.route("/api/qa/ask", methods=["POST"])
def qa_ask():
    try:
        data = request.get_json(silent=True) or {}
        answer = qa.answer(data.get("question", ""), data.get("max_tokens", 128))
        return _ok(answer=answer)
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("QA failed")
        return _fail(exc)


# ============================================================
# EXPERIMENT 4 - SDXL text-to-image
# ============================================================

@app.route("/api/sdxl/generate", methods=["POST"])
def sdxl_generate():
    try:
        data = request.get_json(silent=True) or {}
        image = sdxl.generate(
            data.get("prompt", ""),
            data.get("negative_prompt", ""),
            data.get("steps", 25),
            data.get("guidance_scale", 7.0),
            data.get("seed", -1),
        )
        return _ok(image=image)
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("SDXL generation failed")
        return _fail(exc)


# ============================================================
# EXPERIMENT 5 - VAE (MNIST)
# ============================================================

@app.route("/api/vae/generate", methods=["POST"])
def vae_generate():
    try:
        data = request.get_json(silent=True) or {}
        return _ok(images=vae.generate(data.get("count", 9)))
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("VAE generation failed")
        return _fail(exc)


@app.route("/api/vae/reconstruct", methods=["POST"])
def vae_reconstruct():
    try:
        if "image" not in request.files:
            return _fail("No image uploaded.", status=400)
        return _ok(**vae.reconstruct(request.files["image"].stream))
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("VAE reconstruction failed")
        return _fail(exc)


@app.route("/api/vae/decode", methods=["POST"])
def vae_decode():
    try:
        data = request.get_json(silent=True) or {}
        return _ok(**vae.decode_point(data.get("z1", 0.0), data.get("z2", 0.0)))
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("VAE decode failed")
        return _fail(exc)


# ============================================================
# EXPERIMENT 6 - DCGAN faces
# ============================================================

@app.route("/api/gan/generate", methods=["POST"])
def gan_generate():
    try:
        data = request.get_json(silent=True) or {}
        return _ok(images=gan.generate(data.get("count", 4)))
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("GAN generation failed")
        return _fail(exc)


# ============================================================
# EXPERIMENT 7 - StyleGAN (random-init demo)
# ============================================================

@app.route("/api/stylegan/generate", methods=["POST"])
def stylegan_generate():
    try:
        data = request.get_json(silent=True) or {}
        return _ok(images=stylegan.generate(data.get("count", 8)))
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("StyleGAN generation failed")
        return _fail(exc)


# ============================================================
# EXPERIMENT 8 - MusicGen
# ============================================================

@app.route("/api/music/generate", methods=["POST"])
def music_generate():
    try:
        data = request.get_json(silent=True) or {}
        filename = music.generate(data.get("prompt", ""), data.get("duration", 8))
        return _ok(audio=f"/media/audio/{filename}")
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Music generation failed")
        return _fail(exc)


@app.route("/media/audio/<path:filename>")
def serve_audio(filename):
    return send_from_directory(music.OUTPUT_DIR, filename)


# ============================================================
# EXPERIMENT 9 - RAG personal assistant
# ============================================================

@app.route("/api/assistant/ask", methods=["POST"])
def assistant_ask():
    try:
        data = request.get_json(silent=True) or {}
        result = assistant.ask(
            data.get("question", ""),
            data.get("document", ""),
            data.get("use_rag", True),
            data.get("max_tokens", 180),
        )
        return _ok(**result)
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Assistant failed")
        return _fail(exc)


@app.route("/api/assistant/upload", methods=["POST"])
def assistant_upload():
    try:
        if "file" not in request.files:
            return _fail("No file uploaded.", status=400)
        upload = request.files["file"]
        text = assistant.extract_text(upload.filename or "document.txt", upload.read())
        return _ok(text=text, characters=len(text))
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Document extraction failed")
        return _fail(exc)


# ============================================================
# EXPERIMENT 10 - text-to-video
# ============================================================

@app.route("/api/video/generate", methods=["POST"])
def video_generate():
    try:
        data = request.get_json(silent=True) or {}
        filename = video.generate(
            data.get("prompt", ""),
            data.get("frames", 16),
            data.get("steps", 25),
        )
        return _ok(video=f"/media/video/{filename}")
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Video generation failed")
        return _fail(exc)


@app.route("/media/video/<path:filename>")
def serve_video(filename):
    return send_from_directory(video.OUTPUT_DIR, filename)


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=int(os.environ.get("PORT", 8000)),
        debug=False,
        threaded=True,
        use_reloader=False,
    )
