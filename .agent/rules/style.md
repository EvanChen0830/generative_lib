---
trigger: always_on
---

# Python Code Style & Documentation Guidelines

## 1. Docstrings (Google Style)

- **Mandatory:** All functions, classes, and public modules must use **Google Style** docstrings.
- **Format:**

  ```python
  def function_name(param1: Type, param2: Type) -> ReturnType:
      """Short summary.

      Args:
          param1: Description of param1.
          param2: Description of param2.

      Returns:
          Description of the return value.
      """
  ```

## 2. Type Hinting

- Use Python 3.10+ syntax (e.g., `list[str] | None` instead of `Optional[List[str]]`).
- Be explicit. Do not use `Any` unless absolutely necessary.

## 3. Imports

- Group imports: Standard library -> Third party -> Local application.
- Use absolute imports for local modules (e.g., `from my_lib.models import score_sde` not `from ..models import score_sde`).
