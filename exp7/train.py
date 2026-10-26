import argparse, os
from pathlib import Path
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from dataset import make_loader
from models.stylegan import Generator


class Discriminator(nn.Module):
    """Compact discriminator for educational GAN training."""
    def __init__(self, image_size=64, base=64):
        super().__init__()
        layers = []
        ch = 3
        size = image_size
        out = base
        while size > 4:
            layers += [nn.Conv2d(ch, out, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
            ch, size = out, size // 2
            out = min(out * 2, 512)
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(ch * 4 * 4, 1))
    def forward(self, x):
        return self.head(self.features(x))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/celeba")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--z-dim", type=int, default=128)
    p.add_argument("--limit-images", type=int, default=0)
    p.add_argument("--out", default="checkpoints")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    loader = make_loader(args.data, args.image_size, args.batch_size, limit_images=args.limit_images)
    G = Generator(args.z_dim, args.z_dim, args.image_size).to(device)
    D = Discriminator(args.image_size).to(device)
    opt_g = Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.99))
    opt_d = Adam(D.parameters(), lr=1e-4, betas=(0.0, 0.99))
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(args.epochs):
        G.train(); D.train()
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for real, _ in loop:
            real = real.to(device)
            b = real.size(0)
            real_labels = torch.ones(b, 1, device=device)
            fake_labels = torch.zeros(b, 1, device=device)

            z = torch.randn(b, args.z_dim, device=device)
            fake = G(z).detach()
            d_loss = criterion(D(real), real_labels) + criterion(D(fake), fake_labels)
            opt_d.zero_grad(set_to_none=True); d_loss.backward(); opt_d.step()

            z = torch.randn(b, args.z_dim, device=device)
            fake = G(z)
            g_loss = criterion(D(fake), real_labels)
            opt_g.zero_grad(set_to_none=True); g_loss.backward(); opt_g.step()
            loop.set_postfix(d_loss=float(d_loss.detach()), g_loss=float(g_loss.detach()))

        torch.save({"generator": G.state_dict(), "discriminator": D.state_dict(),
                    "z_dim": args.z_dim, "image_size": args.image_size,
                    "epoch": epoch + 1}, Path(args.out) / "latest.pt")
        print(f"Saved checkpoint to {Path(args.out) / 'latest.pt'}")
    print("Training complete. This is an educational StyleGAN-inspired model, not official StyleGAN.")


if __name__ == "__main__":
    main()
