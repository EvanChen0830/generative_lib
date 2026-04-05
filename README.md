# Generative Lib

A modular, extensible PyTorch library for generative models, currently supporting **Gaussian Diffusion**, **Flow Matching**, and **Inverse Problem Solving** (DPS, RePaint, SDEdit). Designed with a focus on clean architecture, research flexibility, and robust experiment tracking via MLflow.

## 🚀 Features

- **Modular Design**: unified `BaseMethod`, `BaseTrainer`, and `BaseSampler` API.
- **Multiple Methods**:
  - Gaussian Diffusion (DDPM) and Classifier-Free Guidance (CFG Diffusion)
  - Flow Matching (Optimal Transport Conditional Flow Matching)
  - Consistency Models (Skeleton)
- **Inverse Problem Solving**:
  - **DPS** (Diffusion Posterior Sampling) — gradient-based soft guidance toward measurement consistency
  - **RePaint** — hard replacement inpainting with jump-back resampling for known/unknown region harmonization
  - **SDEdit** — mid-step stochastic differential editing for denoising corrupted inputs back onto the learned manifold
- **Measurement Operators**: Pluggable `BaseOperator` / `MaskingOperator` abstraction for defining forward measurement models
- **Experiment Tracking**: First-class **MLflow** integration for metrics, parameters, and artifact logging.
- **Resume Capability**: Seamlessly interrupt and resume training runs with full state restoration (model, optimizer, run ID).
- **Flexible Data Handling**: Strictly dictionary-based data flow for complex multi-modal or conditional setups.

## 📦 Installation

```bash
# Clone the repository
git clone <repository_url>
cd generative_lib

# Install in editable mode
pip install -e .
```

## 📂 Directory Structure

```text
generative_lib/
├── core/               # Abstract base classes
│   ├── base_method.py      # Abstract Physics Interface (SDE/ODE)
│   ├── base_trainer.py     # Abstract Loop Engine
│   ├── base_sampler.py     # Abstract Inference Engine
│   ├── base_operator.py    # Measurement operators (BaseOperator, MaskingOperator)
│   └── base_evaluator.py   # Evaluation interface
├── diffusion/          # Gaussian Diffusion implementation
│   ├── method/         # DDPM and CFG logic
│   ├── sampler/        # Sampling strategies
│   │   ├── base.py           # Standard forward sampler (DDPM/DDIM)
│   │   ├── base_inverse.py   # Base class for inverse problem samplers
│   │   ├── dps.py            # Diffusion Posterior Sampling
│   │   ├── repaint.py        # RePaint inpainting
│   │   └── sdedit.py         # SDEdit mid-step denoising
│   └── trainer/        # Diffusion-specific trainer
├── flow_matching/      # Flow Matching implementation
├── consistency_model/  # Consistency Model implementation
├── evaluator/          # Evaluation utilities
├── metrics/            # Distribution metrics (Wasserstein, etc.)
└── utils/              # Logger, ModelTracker, etc.

examples/               # Example scripts (Two Moons, etc.)
```

## 🛠️ Usage

### 1. Dataset Format (Crucial)

**All datasets must return a Python Dictionary.**
This design choice allows the trainer to map keys flexibly to model inputs (targets, conditions, etc.) without hardcoding argument positions.

**Example Dataset:**

```python
class MyDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        # ... logic to load data ...
        return {
            "x": cond_tensor,          # The conditioning features
            "y": target_tensor,        # The target data to generate/denoise
            "info": some_meta          # Other info
        }
```

### 2. Training Example (Diffusion)

Here is a typical workflow for training a Diffusion model:

```python
from generative_lib.diffusion.method.gaussian_diffusion import GaussianDiffusion
from generative_lib.diffusion.method.cfg_diffusion import CFGDiffusion
from generative_lib.diffusion.trainer.base import BaseDiffusionTrainer
from generative_lib.utils.logger import Logger
from generative_lib.utils.tracker import ModelTracker

# 1. Setup Method (DDPM or CFG)
method = GaussianDiffusion(timesteps=1000, schedule="linear")
# OR for CFG: method = CFGDiffusion(timesteps=1000, schedule="linear", unconditional_value=0.0)

# 2. Setup Logger & Tracker
logger = Logger(
    project_name="MyProject", 
    run_name="gaussian_simple_mlp", 
    use_mlflow=True, 
    mlflow_uri="file:./mlruns"
)
tracker = ModelTracker(
    exp_name="MyProject", 
    model_name="gaussian_simple_mlp", 
    save_dir="./checkpoints/MyProject_gaussian_simple_mlp", 
    logger=logger
)

# 3. Setup Trainer
# feature_keys: Data to denoise (usually target 'y')
# label_keys: Condition (usually features 'x')
trainer = BaseDiffusionTrainer(
    method=method,
    model=my_model,
    optimizer=my_opt,
    feature_keys=["x"], 
    label_keys=["y"], 
    device="cuda:0",
    tracker=tracker
)

# 4. Train
trainer.fit(train_dataloader, val_loader=val_dataloader, epochs=100, resume=True)
```

### 3. Inference Example (Sampling)

After training, you can generate samples using the `BaseDiffusionSampler` and best weights from the tracker:

```python
from generative_lib.diffusion.sampler.base import BaseDiffusionSampler

# 1. Load best model weights
ckpt = tracker.load_best(my_model)
my_model.load_state_dict(ckpt['model_state'])

# 2. Setup Sampler
sampler = BaseDiffusionSampler(
    method=method, 
    model=my_model, 
    device="cuda:0", 
    steps=100, 
    feature_keys=["x"]
)

# 3. Batch Sample
# Generate num_samples candidates per instance in the dataloader
generated_samples = sampler.batch_sample(
    num_samples=10,
    shape=(1,), # Modify shape according to the target dimension
    dataloader=test_dataloader
)
```

### 4. Inverse Problem Solving

The library provides three inverse problem solvers that leverage a pre-trained unconditional (or conditional) diffusion model to recover data subject to measurement constraints.

#### Operators

Define the forward measurement model using `BaseOperator`. For inpainting-style problems, use `MaskingOperator`:

```python
from generative_lib.core.base_operator import MaskingOperator

# mask: 1 = known/observed, 0 = unknown/to-generate
mask = torch.tensor([0.0, 1.0])  # Observe dim 1, generate dim 0
operator = MaskingOperator(mask)
```

#### DPS (Diffusion Posterior Sampling)

Soft gradient-based guidance. Steers the reverse process toward measurement consistency via ∇ₓ‖A(x̂₀) − y‖². Best for general inverse problems with differentiable operators.

```python
from generative_lib.diffusion.sampler.dps import DPSSampler

sampler = DPSSampler(
    method, model, device, steps=100,
    operator=operator,       # Forward measurement operator
    y=y_observation,         # Observed measurement [1, D]
    zeta=1.0,                # Guidance strength (gradient-normalized)
    init_x=corrupted_data,   # Optional: start from corrupted data (SDEdit-style)
    start_ratio=0.5          # Optional: start denoising from midpoint
)
samples = sampler.sample(num_samples=200, shape=[2])
```

#### RePaint (Hard Replacement Inpainting)

Hard replacement of known regions at each denoising step, with jump-back resampling for harmonization. Best for inpainting where known regions are exactly observed.

```python
from generative_lib.diffusion.sampler.repaint import RepaintSampler

sampler = RepaintSampler(
    method, model, device, steps=100,
    operator=operator,       # Must be a MaskingOperator
    y=y_observation,         # Full observation vector (masked dims ignored)
    jump_n_sample=10,        # Number of jump-back resampling iterations per step
    init_x=corrupted_data,   # Optional
    start_ratio=0.5          # Optional
)
samples = sampler.sample(num_samples=200, shape=[2])
```

#### SDEdit (Stochastic Differential Editing)

Pure mid-step denoising — corrupts the input to a chosen noise level, then denoises back onto the learned manifold. No measurement operator needed.

```python
from generative_lib.diffusion.sampler.sdedit import SDEditSampler

sampler = SDEditSampler(
    method, model, device, steps=100,
    init_x=corrupted_data,   # Required: data to denoise
    start_ratio=0.5          # Noise level (0=no noise, 1=full noise)
)
samples = sampler.sample(num_samples=200, shape=[2])
```

### 5. MLflow Logging

- **View UI**: Run `mlflow ui` inside the directory where your `mlruns` are stored.
- **Resume**: If `resume=True`, the trainer will look for the last checkpoint and automatically attach to the **same** MLflow Run ID, ensuring continuous learning curves.

## 📝 Examples

Check the `examples/` folder for running code:

- `two_moons_diffusion.py`: Conditional Diffusion on 2D data.
- `two_moons_flow.py`: Flow Matching on 2D data.
- `two_moons_inverse.py`: Inverse problem solving (DPS, RePaint, SDEdit) on 2D data.
- `test_evaluator.py`: Evaluation utilities demo.

## 🤝 Contributing

1.  Follow the dictionary-based data return format.
2.  Use `Logger` for all metrics.
3.  Ensure `BaseMethod.compute_loss` returns a `Dict[str, Tensor]`.
4.  Inverse samplers should inherit from `BaseInverseSampler` and implement `_sample_batch`.
