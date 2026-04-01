import json
import math
import subprocess
import sys


def test_command_line_round_trip(tmp_path):
    def run(*args):
        result = subprocess.run(
            [sys.executable, "-m", "astrobridge", *map(str, args)],
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + "\n" + result.stderr

    run("generate", "--count", "10", "--output", tmp_path / "data.npz")
    run(
        "train",
        "--data",
        tmp_path / "data.npz",
        "--epochs",
        "2",
        "--steps",
        "3",
        "--output",
        tmp_path / "train",
    )
    run(
        "evaluate",
        "--count",
        "5",
        "--seed",
        "101",
        "--checkpoint",
        tmp_path / "train/best.pt",
        "--steps",
        "3",
        "--ensemble",
        "2",
        "--output",
        tmp_path / "report.json",
    )
    report = json.loads((tmp_path / "report.json").read_text())
    assert math.isfinite(report["rmse"]) and 0 <= report["nominal_90_coverage"] <= 1
    assert report["mean_ensemble_std"] > 0
