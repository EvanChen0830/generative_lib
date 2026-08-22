---
name: generative-lib
description: Use this repository's generative_lib package to train, sample, run inference, solve inverse problems, and evaluate generative models. Covers GaussianDiffusion, CFGDiffusion, FlowMatching, ConsistencyModel, ModelTracker checkpoints, Weights & Biases logging, dictionary DataLoader contracts, sampler shapes, and Evaluator/Frechet-distance evaluation.
license: Complete terms in LICENSE
---

# Generative Lib Training, Inference, and Evaluation

## Overview

Use this skill when a user wants to build or run experiments with this repo's `generative_lib` package. The package is a modular PyTorch library for generative modeling with diffusion, flow matching, consistency models, inverse problem samplers, Weights & Biases logging, checkpoints, and Frechet-distance evaluation.

The main contract is simple: datasets yield dictionaries, trainers map dictionary keys into `(x, condition)`, samplers produce tensors, and evaluators compare generated tensors against held-out real data.

## Key Directives

1. Read existing examples before creating a new experiment. Prefer adapting `examples/two_moons_diffusion.py`, `examples/two_moons_flow.py`, `examples/two_moons_inverse.py`, or `examples/test_evaluator.py`.
2. Always make the `DataLoader` return a flat Python dictionary. Non-dictionary batches fail in training.
3. Use `feature_keys` for conditioning inputs and `label_keys` for the target to generate or denoise.
4. Keep model forward signatures compatible with the method. Most methods call `model(x, t, condition)`.
5. Use `ModelTracker` for `best.pt` and `last.pt`; resume with `trainer.fit(..., resume=True)`.
6. For inference, load the best checkpoint before sampling unless the user explicitly wants current in-memory weights.
7. Check sample shapes carefully. Direct conditional sampling returns `[B, num_samples, *shape]`; unconditional sampling returns `[num_samples, *shape]`; loader sampling returns `[N, num_samples, *shape]`.
8. Evaluate against the same normalized data space used for training. Denormalize only for reporting or plotting unless the metric should operate on original units.

## Repository Map

Important modules:

- `generative_lib.diffusion.method.gaussian_diffusion.GaussianDiffusion`
- `generative_lib.diffusion.method.cfg_diffusion.CFGDiffusion`
- `generative_lib.diffusion.trainer.base.BaseDiffusionTrainer`
- `generative_lib.diffusion.sampler.base.BaseDiffusionSampler`
- `generative_lib.flow_matching.method.flow_matching.FlowMatching`
- `generative_lib.flow_matching.trainer.base.BaseFlowMatchingTrainer`
- `generative_lib.flow_matching.sampler.base.BaseFlowMatchingSampler`
- `generative_lib.consistency_model.method.consistency_model.ConsistencyModel`
- `generative_lib.consistency_model.trainer.base.BaseConsistencyModelTrainer`
- `generative_lib.consistency_model.sampler.base.BaseConsistencyModelSampler`
- `generative_lib.diffusion.sampler.dps.DPSSampler`
- `generative_lib.diffusion.sampler.repaint.RepaintSampler`
- `generative_lib.diffusion.sampler.sdedit.SDEditSampler`
- `generative_lib.core.base_operator.MaskingOperator`
- `generative_lib.evaluator.evaluator.Evaluator`
- `generative_lib.metrics.distance.calculate_frechet_distance`
- `generative_lib.metrics.distance.compute_statistics`
- `generative_lib.utils.logger.Logger`
- `generative_lib.utils.tracker.ModelTracker`

## Data Contract

Each dataset item must be a flat dictionary:

```python
{
    "features": torch.Tensor(...),  # conditioning input, optional
    "labels": torch.Tensor(...),    # generated target
    "info": metadata,               # optional metadata
}
```

Trainer key mapping:

- `feature_keys=["features"]` means concatenate those tensors into `condition`.
- `label_keys=["labels"]` means concatenate those tensors into `x`, the target passed to the generative method.
- `feature_keys=[]` creates unconditional training.
- Multiple keys are concatenated along the last dimension.

Sampler key mapping:

- `feature_keys` controls which dictionary fields are extracted as conditioning data during `batch_sample`.
- Direct `sample(..., condition=cond)` bypasses dictionary extraction.
- If `feature_keys` is empty and `batch_sample` receives dictionary batches, sampling is unconditional but still uses batch size from the loader.

## Train a Diffusion Model

Use this pattern for DDPM or CFG diffusion.

```python
import torch
from generative_lib.diffusion.method.gaussian_diffusion import GaussianDiffusion
from generative_lib.diffusion.method.cfg_diffusion import CFGDiffusion
from generative_lib.diffusion.trainer.base import BaseDiffusionTrainer
from generative_lib.utils.logger import Logger
from generative_lib.utils.tracker import ModelTracker

device = "cuda" if torch.cuda.is_available() else "cpu"

model = MyModel(...).to(device)  # forward(self, x, t, condition)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

method = GaussianDiffusion(timesteps=1000, schedule="linear")
# For classifier-free guidance training:
# method = CFGDiffusion(timesteps=1000, schedule="linear", unconditional_value=-1.0)

logger = Logger(
    project_name="MyProject",
    run_name="diffusion_baseline",
    use_wandb=True,
    wandb_mode="offline",
)

tracker = ModelTracker(
    exp_name="MyProject",
    model_name="diffusion_baseline",
    save_dir="./checkpoints/diffusion_baseline",
    logger=logger,
    best_metric="loss",
    mode="min",
)

trainer = BaseDiffusionTrainer(
    method=method,
    model=model,
    optimizer=optimizer,
    feature_keys=["features"],
    label_keys=["labels"],
    device=device,
    tracker=tracker,
)

trainer.fit(train_loader, val_loader=val_loader, epochs=100, resume=True)
logger.finish()
```

Training notes:

- The trainer logs metrics returned by `method.compute_loss`.
- The checkpoint score is `val_loss` when validation exists; otherwise it uses training loss.
- `ModelTracker` writes `last.pt` each epoch and updates `best.pt` when the metric improves.
- If Weights & Biases is enabled, call `Logger.finish()` after the run.

## Train a Flow Matching Model

Use flow matching when the task should learn a continuous transport velocity instead of a diffusion denoising model.

```python
import torch
from generative_lib.flow_matching.method.flow_matching import FlowMatching
from generative_lib.flow_matching.trainer.base import BaseFlowMatchingTrainer

device = "cuda" if torch.cuda.is_available() else "cpu"

model = MyFlowModel(...).to(device)  # forward(self, x, t, condition=None)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
method = FlowMatching(sigma_min=0.0)

trainer = BaseFlowMatchingTrainer(
    method=method,
    model=model,
    optimizer=optimizer,
    feature_keys=[],
    label_keys=["features"],
    device=device,
    tracker=tracker,
)

trainer.fit(train_loader, val_loader=val_loader, epochs=100, resume=True)
```

For unconditional flow matching, leave `feature_keys=[]` and put the generated target under `label_keys`.

## Run Inference and Sampling

Load the checkpoint first, then create the sampler that matches the method.

```python
from generative_lib.diffusion.sampler.base import BaseDiffusionSampler

checkpoint = tracker.load_best(model)
if checkpoint:
    model.load_state_dict(checkpoint["model_state"])

sampler = BaseDiffusionSampler(
    method=method,
    model=model,
    device=device,
    steps=50,
    feature_keys=["features"],
    sampler_type="ddim",  # "ddim" or "ddpm"
    guidance_scale=1.0,
)

samples = sampler.sample(
    num_samples=20,
    shape=[target_dim],
    condition=condition_tensor,
)
# Direct conditional output shape: [B, 20, target_dim]

samples = sampler.batch_sample(
    num_samples=20,
    shape=[target_dim],
    dataloader=test_loader,
)
# Loader output shape: [N, 20, target_dim]
```

For CFG inference, use `CFGDiffusion` weights and set guidance parameters:

```python
sampler = BaseDiffusionSampler(
    method=method,
    model=model,
    device=device,
    steps=50,
    feature_keys=["class"],
    sampler_type="ddim",
    guidance_scale=3.0,
    unconditional_value=-1.0,
)
```

Flow matching inference:

```python
from generative_lib.flow_matching.sampler.base import BaseFlowMatchingSampler

sampler = BaseFlowMatchingSampler(
    method=method,
    model=model,
    device=device,
    steps=50,
    feature_keys=["features"],
)

samples = sampler.batch_sample(
    num_samples=20,
    shape=[target_dim],
    dataloader=test_loader,
)
```

Consistency model inference:

```python
from generative_lib.consistency_model.sampler.base import BaseConsistencyModelSampler

sampler = BaseConsistencyModelSampler(
    method=method,
    model=model,
    device=device,
    steps=1,
)

samples = sampler.sample(num_samples=100, shape=[target_dim])
```

## Inverse Problem Inference

Use inverse samplers with a trained diffusion model when generated samples must satisfy a measurement constraint.

```python
import torch
from generative_lib.core.base_operator import MaskingOperator
from generative_lib.diffusion.sampler.dps import DPSSampler
from generative_lib.diffusion.sampler.repaint import RepaintSampler
from generative_lib.diffusion.sampler.sdedit import SDEditSampler

mask = torch.tensor([0.0, 1.0], device=device)  # 1 means observed, 0 means unknown
operator = MaskingOperator(mask)
```

DPS uses soft differentiable measurement guidance:

```python
sampler = DPSSampler(
    method,
    model,
    device,
    steps=100,
    operator=operator,
    y=y_observation,
    zeta=1.0,
    init_x=corrupted_data,
    start_ratio=0.5,
)

samples = sampler.sample(num_samples=200, shape=[2])
```

RePaint uses hard replacement for inpainting:

```python
sampler = RepaintSampler(
    method,
    model,
    device,
    steps=100,
    operator=operator,
    y=y_observation,
    jump_n_sample=10,
    init_x=corrupted_data,
    start_ratio=0.5,
)

samples = sampler.sample(num_samples=200, shape=[2])
```

SDEdit denoises or edits a corrupted input without a measurement operator:

```python
sampler = SDEditSampler(
    method,
    model,
    device,
    steps=100,
    init_x=corrupted_data,
    start_ratio=0.5,
)

samples = sampler.sample(num_samples=200, shape=[2])
```

## Evaluate a Model

Use `Evaluator` when generated output should be compared against a real-data key in a loader.

```python
from generative_lib.evaluator.evaluator import Evaluator

generated = sampler.batch_sample(
    num_samples=1,
    shape=[target_dim],
    dataloader=test_loader,
)

# If generated shape is [N, 1, D], squeeze before evaluating against [N, D].
generated_eval = generated.squeeze(1)

evaluator = Evaluator(feature_key="labels", logger=logger)
metrics = evaluator.evaluate(generated_eval, test_loader, step=epoch)
print(metrics)  # {"fid": ...}
```

Use raw metric helpers when manually comparing arrays:

```python
from generative_lib.metrics.distance import calculate_frechet_distance, compute_statistics

real_np = real_tensor.detach().cpu().numpy()
gen_np = generated_tensor.detach().cpu().numpy()

mu_real, sig_real = compute_statistics(real_np)
mu_gen, sig_gen = compute_statistics(gen_np)
fd = calculate_frechet_distance(mu_real, sig_real, mu_gen, sig_gen)
```

Evaluation notes:

- `Evaluator` flattens tensors with rank greater than 2.
- `Evaluator(feature_key=...)` must point at the real target key in the dataloader.
- If `num_samples > 1`, decide whether to evaluate all samples flattened, the mean sample, or a selected sample. Do this explicitly rather than relying on accidental reshaping.
- Use held-out validation or test loaders for reported metrics.

## Common Failure Modes

- `ValueError: BaseTrainer expects DataLoader to return dictionaries.` Fix the dataset `__getitem__` to return a dict.
- `Feature key ... not found` or `Label key ... not found`. Align `feature_keys` and `label_keys` with actual batch keys.
- Model forward errors. Make sure the model accepts `x`, `t`, and `condition`; handle `condition=None` for unconditional runs.
- Shape mismatch in model input. Remember multiple keys are concatenated along the last dimension.
- Empty checkpoint loads. `tracker.load_best(model)` returns `{}` when `best.pt` does not exist; guard before assuming weights loaded.
- CFG output ignores guidance. Ensure training used `CFGDiffusion`, inference used `guidance_scale > 1.0`, and `unconditional_value` matches training.
- Evaluator key mismatch. `Evaluator(feature_key=...)` should name the real target, not the conditioning key, unless that is exactly what should be evaluated.
- Inconsistent normalization. Train, sample, and evaluate in the same feature space; only denormalize for human-facing plots or downstream business metrics.

## Quick Checklist

Before training:

- Dataset returns flat dictionaries.
- `feature_keys` and `label_keys` are correct.
- Model forward signature matches `model(x, t, condition)`.
- Optimizer, logger, and tracker are initialized.
- `save_dir` is unique for the experiment.

Before inference:

- Best or last checkpoint is loaded.
- Sampler class matches method family.
- `shape` matches one generated target, excluding batch and sample dimensions.
- `feature_keys` or direct `condition` provides the intended conditioning.

Before reporting evaluation:

- Generated tensors are shaped intentionally.
- Evaluation uses held-out real data.
- Metrics are computed in the intended normalized or original data space.
- Weights & Biases run is finished if logging is enabled.

## Useful Examples

- `examples/two_moons_diffusion.py`: CFG diffusion training, DDPM/DDIM/CFG sampling, Frechet-distance comparison.
- `examples/two_moons_flow.py`: unconditional flow matching training and sampling.
- `examples/two_moons_inverse.py`: inverse problem samplers.
- `examples/test_evaluator.py`: `Evaluator` integration with generated samples.
