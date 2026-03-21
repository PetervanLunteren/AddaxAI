.PHONY: test lint typecheck dev test-gui test-smoke all

VENV_PYTHON = .venv/Scripts/python
GUI_PYTHON = C:/Users/Topam/AddaxAI_files/envs/env-base/python.exe

test:
	$(VENV_PYTHON) -m pytest tests/ -v

lint:
	$(VENV_PYTHON) -m ruff check addaxai/

typecheck:
	$(VENV_PYTHON) -m mypy addaxai/ --ignore-missing-imports --no-strict-optional

test-gui:
	$(GUI_PYTHON) -m pytest tests/test_gui_integration.py -v

test-smoke:
	$(GUI_PYTHON) -m pytest tests/test_gui_smoke.py -v

dev:
	$(GUI_PYTHON) dev_launch.py

all: lint typecheck test
