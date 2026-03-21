## Summary
<!-- 1-3 bullet points describing what this PR does and why -->

## Test plan
<!-- How did you verify this works? Check all that apply: -->
- [ ] Unit tests pass (`pytest tests/`)
- [ ] Lint passes (`ruff check addaxai/`)
- [ ] Type check passes (`mypy addaxai/ --ignore-missing-imports --no-strict-optional`)
- [ ] GUI smoke test passes (manual or `pytest tests/test_gui_smoke.py`)
- [ ] Tested manually in the GUI

## Checklist
- [ ] No new `global` declarations (use `AppState` instead)
- [ ] Type hints use `typing` generics (Python 3.8 compatible)
- [ ] Updated CLAUDE.md if architecture or conventions changed
