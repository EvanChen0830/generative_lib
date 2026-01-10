---
trigger: always_on
---

# Operational Constraints & Anti-Patterns

## 1. The "No-Defensive" Coding Rule

- **Trust the Type System:** If a function signature defines `input: np.ndarray`, assume the input **IS** a numpy array.
- **NO Input Validation:** Do not write checks like `if data is None` or `if len(data) == 0`.
- **NO Try-Except:** \* Do not wrap imports in `try-except`. Assume the environment is set up correctly.
  - Do not wrap logic in generic `try-except` blocks. Let errors bubble up so we can see the stack trace.

## 2. Commit Message Standards

- **Format:** Strict imperative mood.
- **Content:** ONLY describe the functional change.
- **Prohibited:** Do not include "thinking," "reasoning," or meta-text like "I updated the file to...".
  - _Bad:_ "I added a new function because the user asked for it."
  - _Good:_ "Add generic flow matching scheduler."

## 3. Code Cleanliness

- **No Nesting:** Return early. If an `if` block returns, do not use `else`.
  - _Bad:_ `if x: return True else: return False`
  - _Good:_ `if x: return True \n return False`
