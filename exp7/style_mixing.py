"""Demonstration of layer-wise style mixing with two latent identities."""
import argparse
import torch
from torchvision.utils import save_image
from models.stylegan import Generator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/latest.pt")
    p.add_argument("--cutoff", type=int, default=2)
    p.add_argument("--out", default="outputs/style_mixing.png")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    G = Generator(ckpt["z_dim"], ckpt["z_dim"], ckpt["image_size"]).to(device)
    G.load_state_dict(ckpt["generator"]); G.eval()
    with torch.no_grad():
        z1, z2 = torch.randn(1, ckpt["z_dim"], device=device), torch.randn(1, ckpt["z_dim"], device=device)
        w1, w2 = G.map_z(z1), G.map_z(z2)
        mixed = G(z1, styles=[w1, w2], mixing_cutoff=args.cutoff)
        save_image((mixed + 1)/2, args.out)
    print(f"Saved style-mixed image to {args.out}")


if __name__ == "__main__":
    main()
