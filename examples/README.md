## Examples

This folder should contain source example scripts only.

Current examples:
- `two_moons_diffusion.py`: conditional diffusion / CFG sampling on two moons
- `two_moons_flow.py`: unconditional flow matching on two moons
- `two_moons_inverse.py`: inverse-problem samplers on top of diffusion
- `test_evaluator.py`: evaluator and metric smoke test

Generated outputs should not be written back into `examples/`.
Use `runs/examples/<example-name>/` for plots, checkpoints, MLflow data, and sampled arrays.

Source-only policy:
- keep `.py` example scripts here
- keep short documentation here
- do not keep generated `.png`, `.npz`, checkpoints, `mlruns/`, `logs/`, or nested output folders here
