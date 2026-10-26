"""Optional Flask UI for a trained educational checkpoint."""
from flask import Flask, render_template
import torch
from torchvision.utils import save_image
from pathlib import Path
from models.stylegan import Generator

app = Flask(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = Path("checkpoints/latest.pt")
OUTPUT_DIR = Path("static/generated")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

G = None
if CKPT_PATH.exists():
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    G = Generator(ckpt["z_dim"], ckpt["z_dim"], ckpt["image_size"]).to(DEVICE)
    G.load_state_dict(ckpt["generator"]); G.eval()


@app.route("/")
def index():
    image_name = None
    if G is not None:
        z = torch.randn(1, G.z_dim, device=DEVICE)
        with torch.no_grad():
            image = (G(z) + 1) / 2
        image_name = "face.png"
        save_image(image, OUTPUT_DIR / image_name)
    return render_template("index.html", image_name=image_name,
                           model_ready=G is not None)


if __name__ == "__main__":
    app.run(debug=True)
