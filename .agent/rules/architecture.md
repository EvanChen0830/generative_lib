---
trigger: always_on
---

# Architecture & SOLID Principles

## 1. Single Responsibility (SRP)

- Functions should do exactly one thing.
- If a function exceeds 20-30 lines, strictly evaluate if it handles multiple logic streams (e.g., data loading AND processing). Split it if so.

## 2. Open/Closed (OCP)

- Classes should be open for extension but closed for modification.
- Use abstract base classes (ABCs) for Generative Models (e.g., `BaseScoreModel`) so new models can be added by subclassing, not by adding `if model_type == 'new_model'` statements.

## 3. Dependency Inversion (DIP)

- High-level modules (e.g., the Training Loop) should not import low-level modules (e.g., a specific CSV loader) directly. Both should depend on abstractions.

## 4. HFT/Performance Considerations

- Prioritize vectorized operations (NumPy/Torch) over Python loops.
- Avoid unnecessary data copying.
