# Generative Lib

A modular, extensible PyTorch library for generative models, currently supporting **Gaussian Diffusion** and **Flow Matching**. Designed with a focus on clean architecture, research flexibility, and robust experiment tracking via MLflow.

## 🚀 Features

- **Modular Design**: unified `BaseMethod`, `BaseTrainer`, and `BaseSampler` API.
- **Multiple Methods**:
  - Gaussian Diffusion (DDPM)
  - Flow Matching (Optimal Transport Conditional Flow Matching)
  - Consistency Models (Skeleton)
- **Experiment Tracking**: First-class **MLflow** integration for metrics, parameters, and artifact logging.
- **Resume Capability**: Seamlessly interrupt and resume training runs with full state restoration (model, optimizer, run ID).
- **Flexible Data Handling**: Strictly dictionary-based data flow for complex multi-modal or conditional setups.

## 📦 Installation

```bash
# Clone the repository
git clone <repository_url>
cd generative_lib

# Install dependencies
pip install -r requirements.txt
```

## 📂 Directory Structure

```text
generative_lib/
├── core/               # Abstract base classes (Method, Trainer, Sampler)
├── diffusion/          # Gaussian Diffusion implementation
│   ├── method/         # DDPM logic
│   ├── sampler/        # Sampling strategies
│   └── trainer/        # Diffusion-specific trainer
├── flow_matching/      # Flow Matching implementation
├── consistency_model/  # Consistency Model implementation
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
            "image": img_tensor,       # The data to generate
            "class_label": label,      # Conditional label
            "meta_info": some_meta     # Other info
        }
```

**Trainer Configuration:**

When initializing the **Trainer**, you specify which keys correspond to what:

```python
trainer = BaseDiffusionTrainer(
    method=method,
    model=model,
    optimizer=optimizer,
    # Map dataset keys to model interactions
    feature_keys=["image"],    # Keys treated as 'x' (target)
    label_keys=["class_label"] # Keys treated as 'condition'
)
```

### 2. Training Example (Diffusion)

```python
from generative_lib.diffusion.method.gaussian_diffusion import GaussianDiffusion
from generative_lib.diffusion.trainer.base import BaseDiffusionTrainer
from generative_lib.utils.logger import Logger
from generative_lib.utils.tracker import ModelTracker

# 1. Setup Method
method = GaussianDiffusion(timesteps=1000)

# 2. Setup Logger & Tracker
logger = Logger(project_name="MyProject", use_mlflow=True, mlflow_uri="file:./examples/mlruns")
tracker = ModelTracker(exp_name="MyExp", model_name="Diff", logger=logger)

# 3. Setup Trainer
trainer = BaseDiffusionTrainer(
    method=method,
    model=my_model,
    optimizer=my_opt,
    feature_keys=["image"],
    label_keys=["class_label"],
    tracker=tracker
)

# 4. Train
trainer.fit(dataloader, epochs=100, resume=True)
```

### 3. MLflow Logging

- **View UI**: Run `mlflow ui` inside the `examples/` directory (or wherever your `mlruns` are stored).
- **Resume**: If `resume=True`, the trainer will look for the last checkpoint and automatically attach to the **same** MLflow Run ID, ensuring continuous learning curves.

## 📝 Examples

Check the `examples/` folder for running code:

- `two_moons_diffusion.py`: Conditional Diffusion on 2D data.
- `two_moons_flow.py`: Flow Matching on 2D data.
- `test_resume.py`: Verifies training interruption and resumption.

## 🤝 Contributing

1.  Follow the dictionary-based data return format.
2.  Use `Logger` for all metrics.
3.  Ensure `BaseMethod.compute_loss` returns a `Dict[str, Tensor]`.
