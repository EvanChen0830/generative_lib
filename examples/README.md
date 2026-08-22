## Examples

This folder should contain source example scripts only.

Current examples:
- `two_moons_diffusion.py`: conditional diffusion / CFG sampling on two moons
- `two_moons_consistency.py`: consistency training on two moons. Run conditional
  and unconditional models separately with ``--mode conditional`` and
  ``--mode unconditional``; the default EDM warm start and eight-step sampler
  are the quality-oriented configuration.
- `two_moons_flow.py`: unconditional flow matching on two moons
- `two_moons_inverse.py`: inverse-problem samplers on top of diffusion
- `test_evaluator.py`: evaluator and metric smoke test

Generated outputs should not be written back into `examples/`.
Use `runs/examples/<example-name>/` for plots, checkpoints, Weights & Biases data, and sampled arrays.

Source-only policy:
- keep `.py` example scripts here
- keep short documentation here
- do not keep generated `.png`, `.npz`, checkpoints, `Weights & Biases run data, `logs/`, or nested output folders here
