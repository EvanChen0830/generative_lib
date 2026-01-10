---
description: overall workflow
---

1. Architecture & Design (SOLID)
   Abstraction First: You define clear Abstract Base Classes (BaseMethod,

BaseSampler
, etc.) before implementation.
Modularity: You prefer deep directory structures with concise filenames (e.g.,

diffusion/sampler/base.py
instead of .../base_diffusion_sampler.py).
Flexibility: Code must not hardcode data keys (use feature_keys, label_keys) or model architectures. 2. "No-Defensive" Coding Rule
Trust Types: If a signature says Tensor, assume it is one. No if x is None checks.
Fail Fast: No generic try-except blocks. Let errors bubble up so the stack trace is visible.
Performance: Vectorized operations (NumPy/Torch) are strictly preferred over loops. 3. Development Process
Verification-Driven: Implementation is immediately followed by a "toy" experiment (e.g., Two Moons) to prove it works before moving on.
Iterative Refinement: You refine the API (like splitting

sample
vs

batch_sample
) and hyperparameters (20k samples, BatchNorm) until the verification is undeniable. 4. Git Etiquette
Gradual Commits: Never commit everything at once. Group changes logically.
Conventional Commits: Use structural prefixes (feat, chore, test).
Imperative Messages: Describe what changed, never why or "I thought...".
