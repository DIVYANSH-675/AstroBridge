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
