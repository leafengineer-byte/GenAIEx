"""Compact educational StyleGAN-inspired architecture (not official StyleGAN)."""
import torch
from torch import nn


class MappingNetwork(nn.Module):
    def __init__(self, z_dim=128, w_dim=128, depth=4):
        super().__init__()
        layers = []
        in_dim = z_dim
        for _ in range(depth):
            layers += [nn.Linear(in_dim, w_dim), nn.LeakyReLU(0.2, inplace=True)]
            in_dim = w_dim
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class NoiseInjection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x, noise=None):
        if noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        return x + self.weight * noise


class StyledBlock(nn.Module):
    def __init__(self, in_ch, out_ch, w_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.noise = NoiseInjection(out_ch)
        self.affine = nn.Linear(w_dim, out_ch * 2)
        self.bias = nn.Parameter(torch.zeros(1, out_ch, 1, 1))
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, w, noise=None):
        x = self.conv(x)
        x = self.noise(x, noise)
        scale, shift = self.affine(w).chunk(2, dim=1)
        x = x * (1 + scale[:, :, None, None]) + shift[:, :, None, None] + self.bias
        return self.act(x)


class Generator(nn.Module):
    def __init__(self, z_dim=128, w_dim=128, image_size=64, base_channels=128):
        super().__init__()
        if image_size not in (32, 64, 128):
            raise ValueError("image_size must be 32, 64, or 128")
        self.z_dim, self.w_dim, self.image_size = z_dim, w_dim, image_size
        self.mapping = MappingNetwork(z_dim, w_dim)
        self.num_stages = {32: 3, 64: 4, 128: 5}[image_size]
        self.constant = nn.Parameter(torch.randn(1, base_channels, 4, 4))
        channels = [base_channels, base_channels, base_channels // 2, base_channels // 4, base_channels // 8, base_channels // 16]
        blocks = []
        in_ch = base_channels
        for stage in range(self.num_stages):
            out_ch = max(16, channels[min(stage + 1, len(channels)-1)])
            blocks.append(StyledBlock(in_ch, out_ch, w_dim))
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)
        self.to_rgb = nn.Conv2d(in_ch, 3, 1)

    def forward(self, z, noise_mode="random", styles=None, mixing_cutoff=None):
        w = self.mapping(z) if styles is None else styles
        x = self.constant.expand(z.size(0), -1, -1, -1)
        for i, block in enumerate(self.blocks):
            x = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
            block_w = w
            if isinstance(w, (list, tuple)):
                block_w = w[0] if mixing_cutoff is None or i < mixing_cutoff else w[1]
            noise = None if noise_mode == "random" else torch.zeros(
                x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype
            )
            x = block(x, block_w, noise=noise)
        return torch.tanh(self.to_rgb(x))

    def map_z(self, z):
        return self.mapping(z)
