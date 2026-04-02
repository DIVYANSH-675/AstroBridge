# Release validation

Executed locally on 2026-11-17 on Linux CPU.

Environment: Python 3.12.3; pytest 9.2.0; Ruff 0.17.5; PyTorch 2.15.0+cpu; NumPy 2.5.5.

- Editable installation succeeded.
- Ruff lint and format checks passed.
- **5 tests passed**, including a command-line workflow test that creates its own data/artifacts.
- Wheel and source-distribution builds succeeded.
- The experiments below ran successfully; raw evaluation output is preserved in [examples/release-report.json](examples/release-report.json).

## Exploratory run

The final residual-head model trained on 64 synthetic paired images for 8 epochs, seed 7, with 8 reverse steps. Evaluation used 32 new images, seed 202, and 8 trajectories per image.

| Metric | Value |
|---|---:|
| Reconstruction RMSE | 0.049431 |
| Observed-image baseline RMSE | 0.077840 |
| Mean ensemble standard deviation | 0.014962 |
| Empirical coverage of nominal 90% intervals | 48.6572% |

The residual head starts from the observed image. It replaced a sigmoid endpoint head that underperformed on a development evaluation using seed 101 (RMSE 0.223961 versus observation RMSE 0.079769). The final exploratory evaluation uses a different seed, but these are development-scale measurements, not a preregistered benchmark.

The final reconstruction improves this synthetic baseline. **The intervals substantially under-cover and must not be treated as calibrated posterior uncertainty.**

## Reproduction and scope

Use the README commands, matching the seeds and sizes above. AstroBridge's final evaluation used seed 202. Exact floating-point values can differ across dependency versions and hardware. Generated training checkpoints remain local under ignored `runs/` directories; they are not bundled in the repository or Python distribution.

The CI workflow validates Python 3.10 and 3.12 independently; its current result is visible in [GitHub Actions](https://github.com/DIVYANSH-675/AstroBridge/actions). No GPU benchmark, large external dataset, or paper-result reproduction was run. There is no claim that these development experiments establish general research superiority.

## Input-validation follow-up (2026-11-17)

Validate bridge state shapes before tensor arithmetic. Regression tests cover the rejected inputs and valid usage.
The updated full suite passed locally: **6 tests**, including the CLI workflow.
Ruff lint and format checks also passed. The exploratory experiment results above
remain the original measurements; this follow-up did not rerun or replace them.
