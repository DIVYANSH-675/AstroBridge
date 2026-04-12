import pytest
import torch

from astrobridge.bridge import InverseNet, forward_bridge, reverse_moments, sample


def test_endpoints_and_posterior_final_step():
    clean = torch.zeros(2, 1, 4, 4)
    observed = torch.ones_like(clean)
    noise = torch.ones_like(clean)
    torch.testing.assert_close(forward_bridge(clean, observed, torch.zeros(2), 0.3, noise), clean)
    torch.testing.assert_close(forward_bridge(clean, observed, torch.ones(2), 0.3, noise), observed)
    mean, var = reverse_moments(observed, clean, 0.5, 0.0, 0.3)
    torch.testing.assert_close(mean, clean)
    assert var == 0
    with pytest.raises(ValueError):
        reverse_moments(observed, clean, 0.0, 0.0, 0.3)


def test_forward_moments_monte_carlo():
    torch.manual_seed(0)
    shape = (20000, 1, 1, 1)
    clean = torch.zeros(shape)
    observation = torch.ones(shape)
    result = forward_bridge(
        clean, observation, torch.full((shape[0],), 0.5), 0.4, torch.randn(shape)
    )
    assert result.mean().item() == pytest.approx(0.5, abs=0.005)
    assert result.var().item() == pytest.approx(0.04, abs=0.002)


def test_seeded_sampling_and_backward():
    torch.manual_seed(1)
    torch.set_num_threads(1)
    model = InverseNet()
    observation = torch.rand(2, 1, 9, 11)
    torch.testing.assert_close(
        sample(model, observation, 3, seed=5), sample(model, observation, 3, seed=5)
    )
    prediction = model(observation, observation, torch.ones(2))
    prediction.mean().backward()
    assert prediction.shape == observation.shape
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_initial_endpoint_predictor_matches_observation_baseline():
    model = InverseNet()
    observation = torch.rand(2, 1, 8, 8)
    torch.testing.assert_close(
        model(torch.randn_like(observation), observation, torch.rand(2)), observation
    )


def test_bridge_rejects_shape_broadcasting():
    clean = torch.zeros(2, 4, 4)
    with pytest.raises(ValueError):
        forward_bridge(clean, clean, torch.zeros(2), 0.2, clean)
    with pytest.raises(ValueError, match="shapes must match"):
        reverse_moments(torch.zeros(2, 1, 4, 4), torch.zeros(1, 1, 4, 4), 1.0, 0.5, 0.2)


def test_pair_loader_rejects_overflow_and_mismatched_rank(tmp_path):
    import numpy as np

    from astrobridge.bridge import load_pairs

    clean = np.zeros((5, 4, 4), dtype=np.float64)
    observed = clean.copy()
    observed[0, 0, 0] = 1e100
    path = tmp_path / "pairs.npz"
    np.savez(path, clean=clean, observed=observed)
    with pytest.raises(ValueError, match="finite float32"):
        load_pairs(path)
    np.savez(path, clean=clean, observed=np.array(1.0))
    with pytest.raises(ValueError, match="matching shapes"):
        load_pairs(path)


def test_network_width_matches_group_normalization_contract():
    for width in (0, 3, 6, True):
        with pytest.raises(ValueError, match="multiple of four"):
            InverseNet(width)
    model = InverseNet(8)
    observed = torch.ones(1, 1, 4, 4)
    assert model(observed, observed, torch.ones(1)).shape == observed.shape


def test_online_ensemble_matches_stacked_trajectories():
    from astrobridge.bridge import ensemble_moments

    torch.manual_seed(5)
    torch.set_num_threads(1)
    model = InverseNet()
    with torch.no_grad():
        model.decoder[-1].weight.normal_(0, 0.02)
    observed = torch.rand(2, 1, 8, 8)
    stacked = torch.stack([sample(model, observed, 3, seed=20 + i) for i in range(3)])
    mean, std = ensemble_moments(model, observed, 3, 3, seed=20)
    torch.testing.assert_close(mean, stacked.mean(0))
    torch.testing.assert_close(std, stacked.std(0, unbiased=False))
    assert std.max() > 0


def test_interval_diagnostics_expose_zero_spread_and_undercoverage():
    from astrobridge.bridge import interval_report

    report = interval_report(torch.tensor([0.0, 0.0]), torch.zeros(2), torch.tensor([0.0, 1.0]))
    assert report["coverage"] == 0.5 and report["zero_spread_fraction"] == 1
    assert report["mean_width"] == 0
    with pytest.raises(ValueError):
        interval_report(torch.zeros(1), torch.tensor([-1.0]), torch.zeros(1))
