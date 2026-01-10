# Project Roadmap & TODOs

## 🔥 Priority 1: Evaluation & Metrics

A standardized evaluation framework is critical for comparing generative models objectively.

- [ ] **`BaseEvaluator` Class**: Abstract base for evaluation pipelines.
- [ ] **Metric Implementations**:
  - [ ] **FID (Fréchet Inception Distance)**: For image quality/diversity.
  - [ ] **Recall/Precision**: For mode coverage.
  - [ ] **Log-Likelihood (NLL)**: Essential for determining exact density estimation quality (especially for Flow Matching).
- [ ] **Integration**: Hook into `BaseTrainer.validate()` to log these metrics to MLflow automatically.

## 🚀 Priority 2: Advanced Sampling & Guidance

Enhance the generation quality and controllability.

- [ ] **DDIM (Denoising Diffusion Implicit Models)**:
  - [ ] Deterministic sampling.
  - [ ] Faster generation (fewer steps).
  - [ ] Inversion capabilities for image editing.
- [ ] **Classifier-Free Guidance (CFG)**:
  - [ ] Implement training support (randomly dropping conditions).
  - [ ] Update `BaseSampler` to handle guidance scale `w` during inference: $\epsilon_{pred} = \epsilon_{uncond} + w (\epsilon_{cond} - \epsilon_{uncond})$.

## ⚡ Priority 3: Optimal Transport & Flow Improvements

Push the state-of-the-art in Flow Matching.

- [ ] **Optimal Transport (OT) Condtional Flow Matching**:
  - [ ] Implement OT-based coupling (linear interpolation between data and noise with minimal cost).
  - [ ] Reduces transport cost -> straighter flow paths -> faster/better ODE solving.
- [ ] **Solver Choices**:
  - [ ] Support higher-order ODE solvers (Runge-Kutta 4) in the sampler for better precision with fewer regular steps.
