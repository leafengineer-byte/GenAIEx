import os
import time
import secrets

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "generator.pth")
OUTPUT_DIR = os.path.join(BASE_DIR, "static", "generated")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# These values and layers must match the DCGAN Generator used during training.
NZ = 100
NGF = 64
NC = 3

class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.main(z)

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator(NZ, NGF, NC).to(device)
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}. Keep generator.pth beside app.py."
    )

state = torch.load(MODEL_PATH, map_location=device)
# Supports either a plain state_dict or a checkpoint containing a state_dict.
if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]
generator.load_state_dict(state)
generator.eval()

@app.route("/", methods=["GET", "POST"])
def index():
    generated_images = []
    error = None

    if request.method == "POST":
        try:
            count = int(request.form.get("count", "1"))
            count = max(1, min(count, 8))  # keep generation lightweight
            with torch.no_grad():
                noise = torch.randn(count, NZ, 1, 1, device=device)
                outputs = generator(noise).cpu()

            for tensor in outputs:
                # Generator outputs are in [-1, 1]; convert to RGB [0, 255].
                array = tensor.permute(1, 2, 0).numpy()
                array = ((array + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
                image = Image.fromarray(array, mode="RGB")
                filename = f"face_{int(time.time() * 1000)}_{secrets.token_hex(4)}.png"
                image.save(os.path.join(OUTPUT_DIR, filename))
                generated_images.append(filename)
        except Exception as exc:
            app.logger.exception("Face generation failed")
            error = f"Generation failed: {exc}"

    return render_template(
        "index.html",
        generated_images=generated_images,
        error=error,
        device=str(device),
    )

if __name__ == "__main__":
    # Open http://127.0.0.1:5000 in your browser.
    app.run(debug=True)
