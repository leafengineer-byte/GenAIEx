"""Compare stochastic noise variation with noise disabled."""
import argparse
import torch
from torchvision.utils import save_image
from models.stylegan import Generator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/latest.pt")
    p.add_argument("--out", default="outputs/noise_comparison.png")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    G = Generator(ckpt["z_dim"], ckpt["z_dim"], ckpt["image_size"]).to(device)
    G.load_state_dict(ckpt["generator"]); G.eval()
    z = torch.randn(1, ckpt["z_dim"], device=device)
    with torch.no_grad():
        random_noise = G(z, noise_mode="random")
        zero_noise = G(z, noise_mode="zero")
        save_image(torch.cat([(random_noise+1)/2, (zero_noise+1)/2]), args.out, nrow=2)
    print(f"Saved noise comparison to {args.out}")


if __name__ == "__main__":
    main()
