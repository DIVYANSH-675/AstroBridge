import subprocess
import sys

from astrobridge import __version__


def test_version_does_not_require_a_command():
    result = subprocess.run(
        [sys.executable, "-m", "astrobridge", "--version"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "astrobridge " + __version__
