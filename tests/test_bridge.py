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
