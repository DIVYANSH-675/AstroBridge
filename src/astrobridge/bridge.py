"""Conditional Brownian-bridge denoising for paired inverse imaging."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def forward_bridge(clean, observed, time, sigma, noise):
    if (
        clean.ndim != 4
        or clean.shape != observed.shape
        or noise.shape != clean.shape
        or sigma < 0
        or not math.isfinite(sigma)
    ):
        raise ValueError("Endpoint/noise shapes must match and sigma must be nonnegative")
    if (
        time.shape != (len(clean),)
        or not torch.isfinite(time).all()
        or (time < 0).any()
        or (time > 1).any()
    ):
        raise ValueError("Bridge time must lie in [0,1]")
    t = time.reshape(-1, 1, 1, 1)
    return (1 - t) * clean + t * observed + sigma * (t * (1 - t)).sqrt() * noise


def reverse_moments(current, clean_estimate, t, s, sigma):
    if current.shape != clean_estimate.shape:
        raise ValueError("Current state and clean estimate shapes must match")
    if not 0 <= s < t <= 1 or sigma < 0 or not math.isfinite(sigma):
        raise ValueError("Reverse steps require 0 <= s < t <= 1 and sigma >= 0")
    mean = (1 - s / t) * clean_estimate + (s / t) * current
    variance = sigma**2 * s * (t - s) / t
    return mean, variance


class InverseNet(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1), nn.GroupNorm(4, width), nn.SiLU()
        )
        self.deep = nn.Sequential(
            nn.Conv2d(width, 2 * width, 3, padding=1),
            nn.GroupNorm(4, 2 * width),
            nn.SiLU(),
            nn.Conv2d(2 * width, 2 * width, 3, padding=1),
            nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(3 * width, width, 3, padding=1),
            nn.GroupNorm(4, width),
            nn.SiLU(),
            nn.Conv2d(width, 1, 1),
        )
        # Start from the observation baseline, then learn a correction. This avoids
        # an initial sigmoid prediction near 0.5 on mostly dark source images.
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, current, observed, time):
        t = time.reshape(-1, 1, 1, 1).expand(-1, 1, *current.shape[-2:])
        features = self.encoder(torch.cat([current, observed, t], dim=1))
        deep = self.deep(F.avg_pool2d(features, 2))
        up = F.interpolate(deep, size=current.shape[-2:], mode="bilinear", align_corners=False)
        return observed + self.decoder(torch.cat([features, up], dim=1))


@torch.no_grad()
def sample(model, observed, steps=12, sigma=0.2, seed=0):
    if steps < 1:
        raise ValueError("steps must be positive")
    model.eval()
    generator = torch.Generator(device=observed.device).manual_seed(seed)
    current = observed.clone()
    for index in range(steps, 0, -1):
        t, s = index / steps, (index - 1) / steps
        estimate = model(current, observed, torch.full((len(observed),), t, device=observed.device))
        mean, variance = reverse_moments(current, estimate, t, s, sigma)
        current = mean + variance**0.5 * torch.randn(
            current.shape, generator=generator, device=current.device
        )
    return current


def synthetic(count=64, side=16, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[-1 : 1 : complex(side), -1 : 1 : complex(side)]
    images = []
    for _ in range(count):
        image = np.zeros((side, side))
        for _ in range(3):
            cx, cy = rng.uniform(-0.7, 0.7, 2)
            spread = rng.uniform(0.08, 0.3)
            image += rng.uniform(0.2, 1) * np.exp(
                -((x - cx) ** 2 + (y - cy) ** 2) / (2 * spread**2)
            )
        images.append(image / max(image.max(), 1e-8))
    clean = torch.tensor(np.asarray(images), dtype=torch.float32)[:, None]
    observed = F.avg_pool2d(clean, 3, stride=1, padding=1)
    observed += torch.tensor(rng.normal(0, 0.08, observed.shape), dtype=torch.float32)
    return clean.numpy(), observed.clamp(0, 1).numpy()


def load_pairs(path):
    with np.load(path, allow_pickle=False) as archive:
        clean, observed = archive["clean"], archive["observed"]
    if clean.ndim == 3:
        clean, observed = clean[:, None], observed[:, None]
    if (
        clean.shape != observed.shape
        or clean.ndim != 4
        or clean.shape[1] != 1
        or len(clean) < 5
        or min(clean.shape[-2:]) < 4
    ):
        raise ValueError("Expected matching clean/observed arrays [B,1,H,W], B>=5 and H,W>=4")
    if not np.isfinite(clean).all() or not np.isfinite(observed).all():
        raise ValueError("Images must contain finite physical values")
    return clean.astype(np.float32), observed.astype(np.float32)
