## 1. Directory Structure

The library uses a "Core vs. Implementation" split. Specific logic (like Diffusion loops) lives within its family folder (`methods/diffusion`), while abstract interfaces live in `core`.

```text
generative_lib/
├── core/
│   ├── __init__.py
│   ├── base_method.py       # Abstract Physics Interface (SDE/ODE)
│   ├── base_trainer.py      # Abstract Loop Engine
│   └── base_sampler.py      # Abstract Inference Engine
├── diffusion/
│   ├── __init__.py
│   └── method/           # Diffusion Family
│       ├── __init__.py
│       ├── gaussian_diffusion.py        # GaussianDiffusion
│       ├── cfg_diffusion.py       # CFGDiffusion
│   └── trainer/           # Diffusion Family
│       ├── __init__.py
│       ├── base_diffusion_trainer.py        # BaseDiffusionTrainer & inherit from base_trainer

│   └── sampler/           # Diffusion Family
│       ├── __init__.py
│       ├── base_diffusion_sampler.py        # BaseDiffusionSampler & inherit from base_sampler
├── flow_matching/
│   ├── __init__.py
│   └── method/           # Flow Matching Family
│       ├── __init__.py
│       ├── flow_matching.py        # FlowMatching
│   └── trainer/           # Flow Matching Family
│       ├── __init__.py
│       ├── base_flow_matching_trainer.py        # BaseFlowMatchingTrainer & inherit from base_trainer

│   └── sampler/           # Flow Matching Family
│       ├── __init__.py
│       ├── base_flow_matching_sampler.py        # BaseFlowMatchingSampler & inherit from base_sampler
├── consistency_model/
│   ├── __init__.py
│   └── method/           # Consistency Model Family
│       ├── __init__.py
│       ├── consistency_model.py        # ConsistencyModel
│   └── trainer/           # Consistency Model Family
│       ├── __init__.py
│       ├── base_consistency_model_trainer.py        # BaseConsistencyModelTrainer & inherit from base_trainer

│   └── sampler/           # Consistency Model Family
│       ├── __init__.py
│       ├── base_consistency_model_sampler.py        # BaseConsistencyModelSampler & inherit from base_sampler
├── utils/
│   ├── __init__.py
│   ├── tracker.py           # ModelTracker (Best/Last/Resume)
│   ├── logger.py            # Logger (MLflow/Tensorboard wrapper)
├── metrics/
│   ├── __init__.py
│   ├── dist_metrics.py           # DistributionMetrics


```

---

## 2. The Usage Contract (Target Workflow)

This is the canonical example the agent must support.

```python
import torch
from generative_lib.utils.logger import Logger
from generative_lib.utils.tracker import ModelTracker
from generative_lib.diffusion.method.gaussian_diffusion import GaussianDiffusion
from generative_lib.diffusion.trainer.base_diffusion_trainer import BaseDiffusionTrainer
from generative_lib.diffusion.sampler.base_diffusion_sampler import BaseDiffusionSampler
from generative_lib.metrics.dist_metrics import DistributionMetrics

# 1. Infrastructure Setup

logger = Logger(project_name="HFT_Gen", run_name="Exp_001", mlflow = True, dir = "./logs")
# here if the logger enable mlflow, the tracker will load the metrics and all the other stuff onto mlflow
tracker = ModelTracker(
    exp_name="Exp_001",
    model_name="DiT_Small",
    save_dir="./checkpoints",
    logger=logger,
    best_metric="val_loss",
    mode="min"
)

# 2. Model & Physics Setup
# Note: 'model' is the Neural Net (UNet/DiT). 'method' is the Physics (Gaussian Diffusion).
net = MyNeuralNet(input_dim=..., cond_dim=...)
method = GaussianDiffusion(
    feature_keys=["features"], # Conditioning
    label_keys=["labels"],     # Generation Target
    schedule="linear",
    timesteps=1000
)

# 3. Training
# The Trainer orchestrates the method, model, and tracker.
trainer = BaseDiffusionTrainer(
    method=method,
    model=net,
    tracker=tracker,
    device="cuda:0",
    optimizer=opt,
    use_ema=True
)

# Input from loader: {"features": [B, F], "labels": [B, L], "info": [B, I]}
trainer.fit(train_dataloader, val_dataloader, epochs=100)

# 4. Inference
# Load best weights automatically
best_model = tracker.load_best_valid(net)

sampler = BaseDiffusionSampler(
    method=method,
    model=best_model,
    device="cuda:0",
    solver="ddim", # or 'euler'
    steps=50
)

# 5. Generation & Evaluation
# Returns: Tensor of shape [Dataset_Size, Num_Samples, *Label_Dim]
output = sampler.sample(
    dataloader=test_dataloader,
    num_samples=100  # Generate 100 paths per feature
)

# 6. Metrics
metrics = DistributionMetrics(generated_data=output, reference_loader=test_dataloader)
print(metrics.compute_wasserstein())

```

---

## 3. Data Contracts

### 3.1 Input Format (DataLoader)

The user's DataLoader `__getitem__` must return a **flat dictionary**.

- **Structure:**

```python
{
    "features": Tensor,  # [Batch, Feature_Dim] (Conditioning)
    "labels":   Tensor,  # [Batch, Label_Dim]   (Target to generate)
    "info":     Any      # [Batch, ...]         (Metadata/Timestamp)
}

```

### 3.2 Output Format (Sampler)

The Sampler aggregates results across the entire dataset.

- **Structure:** `torch.Tensor` or `np.ndarray`
- **Shape:** `[N_Total_Data, N_Samples_Per_Data, *Label_Dimensions]`
- _Example:_ If testing on 100 days, generating 50 paths per day, and price vector is dim 10.
- _Shape:_ `[100, 50, 10]`

---

## 4. Class Specifications

### 4.1 Core (`core/`)

#### `BaseMethod`

- **Role:** Defines the mathematical rules (Probability Flow).
- **Key Methods:**
- `compute_loss(model, batch) -> Dict`: Calculates loss for training.
- `get_prior(shape, device) -> Tensor`: Returns (noise).
- `predict(model, x_t, t, cond) -> Tensor`: Returns driver term (noise/velocity).

#### `BaseTrainer`

- **Role:** Boilerplate training loop.
- **Key Methods:**
- `fit(train_loader, val_loader)`: Main loop.
- `_train_epoch()`: Iterates loader, calls `method.compute_loss`, steps optimizer.
- `_validate()`: Calcs metrics, calls `tracker.save_checkpoint`.

#### `BaseSampler`

- **Role:** Boilerplate inference loop structure.
- **Key Methods:**
- `sample(dataloader, num_samples)`: Iterates dataloader, runs `_sample_batch` N times, aggregates results.

---

### 4.2 Methods (`methods/diffusion/`)

#### `GaussianDiffusion(BaseMethod)`

- **Responsibility:**
- Store `betas`, `alphas`.
- Implement `compute_loss`: Sample , add noise, predict noise, MSE loss.
- Implement `predict`: Return .

#### `CFGDiffusion(GaussianDiffusion)`

- **Responsibility:**
- Inherit from GaussianDiffusion.
- Override `compute_loss`: Randomly mask `features` (Conditioning) with probability .
- (Note: The Sampler handles the inference-time CFG combination).

#### `BaseDiffusionTrainer(BaseTrainer)`

- **Responsibility:**
- Specific logging for diffusion (e.g., Min/Max of sampled, Gradient Norms).
- If no specific logic is needed, it can simply inherit `BaseTrainer` with `pass`.

#### `BaseDiffusionSampler(BaseSampler)`

- **Responsibility:**
- Implement `_sample_step(x_t, t, cond)`:
- Uses `method.predict` to get noise.
- Applies solver (DDPM step, DDIM step, or Euler).
- Handles Classifier-Free Guidance math if `cfg_scale > 1`.

---

### 4.3 Utilities (`utils/`)

#### `Logger`

- **Methods:** `log_metrics(dict, step)`, `log_params(dict)`.
- **Backend:** Wraps MLflow or Tensorboard.

#### `ModelTracker`

- **Inputs:** `logger`, `save_dir`.
- **Methods:**
- `save_checkpoint(model, optimizer, epoch, metric_val)`: Saves `last.pt`. Copies to `best.pt` if metric improves.
- `load_best_valid(model)`: Loads weights from `best.pt`.
- `load_last(model)`: For resuming training.

#### `DistributionMetrics`

- **Inputs:** `generated_tensor`, `reference_loader`.
- **Methods:**
- `compute_wasserstein()`: Compares distribution of generated labels vs real labels.
- `compute_discriminative_score()`: Train a classifier to distinguish real vs fake.
