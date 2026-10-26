import argparse
from pathlib import Path
import torch
from torchvision.utils import save_image
from models.stylegan import Generator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/latest.pt")
    p.add_argument("--count", type=int, default=8)
    p.add_argument("--out", default="outputs/generated.png")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    G = Generator(ckpt["z_dim"], ckpt["z_dim"], ckpt["image_size"]).to(device)
    G.load_state_dict(ckpt["generator"]); G.eval()
    with torch.no_grad():
        z = torch.randn(args.count, ckpt["z_dim"], device=device)
        images = (G(z) + 1) / 2
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_image(images, args.out, nrow=4)
    print(f"Saved {args.count} images to {args.out}")


if __name__ == "__main__":
    main()
