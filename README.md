# Generative Lib

A modular, extensible PyTorch library for generative models, currently supporting **Gaussian Diffusion** and **Flow Matching**. Designed with a focus on clean architecture, research flexibility, and robust experiment tracking via MLflow.

## 🚀 Features

- **Modular Design**: unified `BaseMethod`, `BaseTrainer`, and `BaseSampler` API.
- **Multiple Methods**:
  - Gaussian Diffusion (DDPM) and Classifier-Free Guidance (CFG Diffusion)
  - Flow Matching (Optimal Transport Conditional Flow Matching)
  - Consistency Models (Continuous-Time Karras EDM w/ EMA Preconditioning)
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
│   ├── method/         # DDPM and CFG logic
│   ├── sampler/        # Sampling strategies (base.py)
│   └── trainer/        # Diffusion-specific trainer (base.py)
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
            "x": cond_tensor,          # The conditioning features
            "y": target_tensor,        # The target data to generate/denoise
            "info": some_meta          # Other info
        }
```

### 2. Training Example (Diffusion)

Here is a typical workflow for training a Diffusion model (as seen in `experiments/04_diffusion`):

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

### 4. MLflow Logging

- **View UI**: Run `mlflow ui` inside the directory where your `mlruns` are stored.
- **Resume**: If `resume=True`, the trainer will look for the last checkpoint and automatically attach to the **same** MLflow Run ID, ensuring continuous learning curves.

## 📝 Examples

Check the `examples/` folder for running code:

- `two_moons_diffusion.py`: Conditional Diffusion on 2D data.
- `two_moons_flow.py`: Flow Matching on 2D data.
- `two_moons_consistency.py`: Multi-Step Consistency Models on 2D data.
- `test_resume.py`: Verifies training interruption and resumption.

## 🤝 Contributing

1.  Follow the dictionary-based data return format.
2.  Use `Logger` for all metrics.
3.  Ensure `BaseMethod.compute_loss` returns a `Dict[str, Tensor]`.
