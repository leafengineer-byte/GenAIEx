"""Helpers for loading the existing experiment scripts as importable modules.

Each experiment lives in its own folder as a standalone script (or Gradio app).
The unified app reuses that code instead of duplicating model logic, loading the
files lazily so heavy models are only initialised when a feature is first used.
"""
import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_module(module_name: str, relative_path: str):
    """Load a standalone Python file from the project root under a unique name."""
    if module_name in sys.modules:
        return sys.modules[module_name]

    file_path = PROJECT_ROOT / relative_path
    if not file_path.is_file():
        raise FileNotFoundError(f"Experiment file not found: {file_path}")

    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
