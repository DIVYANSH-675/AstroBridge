# AstroBridge

Conditional Brownian-bridge inverse imaging with reproducible ensemble evaluation.

AstroBridge trains a small time-conditioned image model to reconstruct clean images along a noisy bridge between paired clean and observed images. It includes synthetic blurred source images, reverse sampling, and ensemble diagnostics.

## Status and provenance

Version 0.1.0 is an original, AI-assisted implementation developed for **Divyansh Gupta**. This repository was written independently: no upstream source files or Git history were imported. Its scientific inspiration is [AstroDSB](https://arxiv.org/abs/2506.08065); the research ideas remain credited to that paper's authors. The implementation choices and limitations below describe this repository's scope. It is an experimental prototype, not a claimed reproduction of published results.

## Setup

Python 3.10 or newer:

```bash
python -m venv .venv
# Linux/macOS; on Windows use .venv\Scripts\activate
source .venv/bin/activate
# Install CPU-only PyTorch first (omit this line to choose another build).
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
```

## Run

```bash
astrobridge train --count 64 --epochs 8 --steps 8 --seed 7 --output runs/train
astrobridge evaluate --checkpoint runs/train/best.pt --count 32 --seed 202 --steps 8 --ensemble 8 --output runs/evaluation.json
astrobridge generate --count 64 --seed 9 --output runs/pairs.npz
astrobridge train --data runs/pairs.npz --epochs 8 --output runs/custom
```

Use fresh output paths for new experiments. Training artifacts and generated data are ignored by Git. For independent evaluation, use a different generation seed or a separate dataset; evaluating on the training data is not an unbiased test.

## Implementation

The forward path is

```text
x_t = (1-t) x_clean + t x_observed + sigma sqrt(t(1-t)) noise
```

The network estimates the clean endpoint from `(x_t, x_observed, t)`. For a reverse step `s < t`, the plug-in Gaussian mean is `(1-s/t) clean_estimate + (s/t) x_t` and variance is `sigma^2 s(t-s)/t`. The default sigma is 0.2 in normalized units.

A compact encoder/decoder with GroupNorm handles odd spatial dimensions. A zero-initialized residual output head starts from the observed image and learns its correction; outputs are not artificially clipped to the training intensity range. Paired arrays are normalized using only the training partition's range. A fixed validation split and fixed sampling seed select `best.pt`. Evaluation uses multiple seeded reverse trajectories, compares mean reconstruction RMSE with the observation baseline, and reports empirical coverage of a nominal mean-plus/minus-1.645-standard-deviation interval.

## Data and artifacts

Training writes `config.json`, `split.json`, `metrics.json`, and `best.pt` including normalization parameters. Input NPZ keys `clean` and `observed` must share shape `[B,1,H,W]` or `[B,H,W]`, with B>=5 and H,W>=4. Inputs must be finite. Metrics are reported in the input physical scale. Evaluation's ensemble uses population standard deviation. All commands run on CPU.

## Validation

```bash
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m build
```

The test suite covers: Bridge endpoint identities and final reverse step; Monte Carlo forward mean/variance; deterministic seeded sampling and odd image dimensions; finite gradients; full training/checkpoint/ensemble evaluation workflow.

CI runs the suite on Python 3.10 and 3.12. Small synthetic runs validate software behavior; they do not establish a research result. See [VALIDATION.md](VALIDATION.md) for the release checks actually performed.

## Limitations

This is a Brownian-bridge denoising prototype, not AstroDSB's full training procedure, a telescope forward model, or a validated astrophysical posterior sampler. The generator uses Gaussian sources, blur, and noise. Its ensemble spread is not calibrated uncertainty; one-step sampling and single-member ensembles cannot meaningfully estimate it. The unconstrained residual output may predict negative intensities or values outside the physical range; the model does not impose positivity or flux conservation. Use real paired data and calibration experiments before interpreting scientific reconstructions.

## Calling the bridge equations

The low-level forward bridge expects clean endpoints, observed endpoints, and noise with an identical four-dimensional `[batch, channels, height, width]` shape. Its time tensor contains one value per batch item. The bundled image network uses one channel.

The reverse-moment function requires the current state and clean estimate to have identical shapes. It rejects a single clean estimate broadcast across a larger batch, which could otherwise silently pair observations with the wrong reconstruction.

The NPZ loader accepts `[batch, height, width]` arrays and adds the channel axis. When calling `forward_bridge` directly, add that axis yourself with `images[:, None, :, :]`. Reverse steps still require `0 <= s < t <= 1` and a finite nonnegative noise scale.

## License

The original code in this repository is MIT licensed, copyright 2026 Divyansh Gupta. See [LICENSE](LICENSE). The cited paper is a separate work; this code's license does not relicense the paper or external datasets.
