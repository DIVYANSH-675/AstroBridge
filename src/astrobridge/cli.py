import argparse
import json

import numpy as np
import torch

from . import __version__
from .bridge import InverseNet, ensemble_moments, forward_bridge, load_pairs, sample, synthetic
from .io import seed_all, start_run, write_json, write_npz


def main():
    p = argparse.ArgumentParser(description="AstroBridge paired inverse-image research prototype")
    p.add_argument("--version", action="version", version="astrobridge " + __version__)
    p.add_argument("command", choices=["generate", "train", "evaluate"])
    p.add_argument("--data")
    p.add_argument("--checkpoint")
    p.add_argument("--output", default=None)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--count", type=int, default=64)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--sigma", type=float, default=None)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--ensemble", type=int, default=4)
    a = p.parse_args()
    if a.batch_size < 1:
        p.error("batch-size must be positive")
    if not np.isfinite(a.learning_rate) or a.learning_rate <= 0:
        p.error("learning-rate must be finite and positive")
    sigma = 0.2 if a.sigma is None else a.sigma
    if not np.isfinite(sigma) or sigma < 0:
        p.error("sigma must be finite and nonnegative")
    a.output = a.output or (
        "runs/report.json"
        if a.command == "evaluate"
        else "runs/data.npz"
        if a.command in ("generate", "bank")
        else "runs/train"
    )
    if min(a.epochs, a.steps, a.ensemble) < 1 or a.count < 5:
        p.error("Positive epochs/steps/ensemble and count >= 5 are required")
    seed_all(a.seed)
    arrays = load_pairs(a.data) if a.data else synthetic(a.count, seed=a.seed)
    if a.command == "generate":
        write_npz(a.output, clean=arrays[0], observed=arrays[1])
        return
    model = InverseNet()
    if a.command == "evaluate":
        if not a.checkpoint:
            p.error("evaluate requires --checkpoint")
        checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        sigma = checkpoint.get("config", {}).get("sigma", 0.2) if a.sigma is None else a.sigma
        offset, scale = checkpoint["offset"], checkpoint["scale"]
        clean, observed = [torch.tensor((v - offset) / scale) for v in arrays]
        prediction, std = ensemble_moments(
            model, observed, ensemble=a.ensemble, steps=a.steps, sigma=sigma, seed=a.seed
        )
        report = dict(
            rmse=float((prediction - clean).square().mean().sqrt()) * scale,
            observation_rmse=float((observed - clean).square().mean().sqrt()) * scale,
            mean_ensemble_std=float(std.mean()) * scale,
            nominal_90_coverage=float(((prediction - clean).abs() <= 1.645 * std).float().mean()),
            ensemble_size=a.ensemble,
            sigma=sigma,
            note="Ensemble spread is not a calibrated posterior guarantee.",
        )
        write_json(a.output, report)
        print(json.dumps(report))
        return
    a.sigma = sigma
    output = start_run(a.output, vars(a))
    order = np.random.default_rng(a.seed).permutation(len(arrays[0]))
    cut = max(1, len(order) // 5)
    valid_ids, train_ids = order[:cut], order[cut:]
    offset = float(min(v[train_ids].min() for v in arrays))
    scale = max(float(max(v[train_ids].max() for v in arrays) - offset), 1e-6)
    clean, observed = [torch.tensor((v - offset) / scale) for v in arrays]
    optimizer = torch.optim.Adam(model.parameters(), lr=a.learning_rate)
    history = []
    best = float("inf")
    for epoch in range(a.epochs):
        model.train()
        total = 0.0
        permutation = torch.tensor(train_ids)[torch.randperm(len(train_ids))]
        for ids in permutation.split(a.batch_size):
            t = 0.02 + 0.98 * torch.rand(len(ids))
            current = forward_bridge(
                clean[ids], observed[ids], t, sigma, torch.randn_like(clean[ids])
            )
            loss = (model(current, observed[ids], t) - clean[ids]).square().mean()
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(ids)
        prediction = sample(model, observed[valid_ids], a.steps, sigma=sigma, seed=123)
        validation = float((prediction - clean[valid_ids]).square().mean())
        history.append(
            dict(epoch=epoch, train_mse=total / len(train_ids), validation_mse=validation)
        )
        if validation < best:
            best = validation
            torch.save(
                dict(model=model.state_dict(), offset=offset, scale=scale, config=vars(a)),
                output / "best.pt",
            )
    write_json(output / "metrics.json", history)
    write_json(output / "split.json", dict(train=train_ids.tolist(), validation=valid_ids.tolist()))
    print(json.dumps(history[-1]))
