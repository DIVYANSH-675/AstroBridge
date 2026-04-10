import sys

import pytest

from astrobridge import cli


def test_invalid_bridge_scale_fails_before_training(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["astrobridge", "train", "--sigma", "-1"])
    with pytest.raises(SystemExit):
        cli.main()
    assert "sigma must" in capsys.readouterr().err


def test_invalid_learning_rate_fails_before_training(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["astrobridge", "train", "--learning-rate", "0"])
    with pytest.raises(SystemExit):
        cli.main()
    assert "learning-rate must" in capsys.readouterr().err


def test_nondefault_training_options_round_trip_through_checkpoint(monkeypatch, tmp_path):
    import json

    import torch

    output = tmp_path / "train"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "astrobridge",
            "train",
            "--count",
            "6",
            "--epochs",
            "1",
            "--steps",
            "2",
            "--batch-size",
            "2",
            "--learning-rate",
            ".002",
            "--sigma",
            "0",
            "--output",
            str(output),
        ],
    )
    cli.main()
    checkpoint = torch.load(output / "best.pt", weights_only=True)
    assert checkpoint["config"]["sigma"] == 0 and checkpoint["config"]["batch_size"] == 2
    report = tmp_path / "evaluation.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "astrobridge",
            "evaluate",
            "--checkpoint",
            str(output / "best.pt"),
            "--count",
            "5",
            "--steps",
            "2",
            "--ensemble",
            "2",
            "--output",
            str(report),
        ],
    )
    cli.main()
    result = json.loads(report.read_text())
    assert result["sigma"] == 0 and result["mean_ensemble_std"] == 0
