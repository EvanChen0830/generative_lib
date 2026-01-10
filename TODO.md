# Project Roadmap & TODOs

## 🔥 Priority 1: Evaluation & Metrics (Completed)

**Branch**: `develop` -> `main`

- [x] **`BaseEvaluator` Class**:
  - [x] Standalone `Evaluator(logger=...)`.
  - [x] Interface `evaluate(generated_data, dataloader)`.
  - [x] FID Metric Logic.
  - [x] MLflow Integration.

## 🚀 Priority 2: Advanced Sampling & Guidance

**Branch**: `feat/diffusion`

Enhance the generation quality and controllability for diffusion models.

- [x] **DDIM (Denoising Diffusion Implicit Models)**:
  - [x] Implement `DDIMSampler` (Unified in `BaseDiffusionSampler`).
  - [x] Inversion capabilities.
- [x] **Classifier-Free Guidance (CFG)**:
  - [x] Unified `BaseDiffusionSampler` to handle `guidance_scale`.
  - [x] Dual-Loss Training via `CFGDiffusion`.
  - [x] Verified on Two Moons (Conditional).

## ⚡ Priority 3: Flow Matching Enhancements

**Branch**: `feat/flow_matching`

Push the state-of-the-art in Flow Matching.

- [x] **Conditional Flow Matching (CFM)**:
  - [x] Ensure explicit support for class-conditional vector fields.
- [x] **Optimal Transport (OT)**:
  - [x] Implement OT-Plan for cleaner couplings between noise and data.
  - [x] Reduces curvature -> faster sampling.
- [x] **Rectified Flow**:
  - [x] Implement Reflow procedure (training on generated data-noise pairs) to straighten trajectories further.
  - [x] 1-Rectified Flow, 2-Rectified Flow.

## 🔄 Priority 4: Consistency Models

**Branch**: `feat/consistency`

Fast one-step or few-step generation.

- [ ] **Consistency Training (CT)**:
  - [ ] Implement Consistency Loss (enforcing $f(x_t, t) = f(x_{t'}, t')$).
  - [ ] Support searching for optimal time discretization.
- [ ] **Consistency Distillation (CD)**:
  - [ ] Distill a pre-trained Diffusion model into a Consistency Model.

## 📦 Priority 5: Solvers & Architecture

**Branch**: `develop` (or specific feature branch if large)

- [ ] **Solver Choices**: Runge-Kutta 4, Euler, Heun.
- [ ] **Architectures**: U-Net 1D/2D, DiT.
