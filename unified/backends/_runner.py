"""Spawn experiment tasks in the interpreter that belongs to each experiment."""
import json
import subprocess
from pathlib import Path

from ._loader import PROJECT_ROOT

RESULT_MARKER = "@@RESULT@@"

# Interpreters that already contain the libraries each experiment needs.
VAE_PYTHON = PROJECT_ROOT / "exp5" / "venv" / "Scripts" / "python.exe"
GAN_PYTHON = PROJECT_ROOT / "exp6" / ".venv" / "Scripts" / "python.exe"
DEFAULT_PYTHON = PROJECT_ROOT / ".venv-unified" / "Scripts" / "python.exe"

WORKER = PROJECT_ROOT / "unified" / "worker.py"


def interpreter_for(experiment: str) -> Path:
    if experiment == "vae":
        return VAE_PYTHON
    if experiment == "gan":
        return GAN_PYTHON
    return DEFAULT_PYTHON


def run_task(experiment: str, task: str, payload: dict, timeout: int = 3600) -> dict:
    """Run a task in the given experiment's interpreter and return its result."""
    python_exe = interpreter_for(experiment)
    if not python_exe.is_file():
        raise RuntimeError(
            f"Python interpreter for '{experiment}' not found: {python_exe}"
        )

    try:
        completed = subprocess.run(
            [str(python_exe), str(WORKER), task],
            input=json.dumps(payload).encode("utf-8"),
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Task '{task}' timed out after {timeout}s")

    stdout = completed.stdout.decode("utf-8", "replace")
    result_line = next(
        (line for line in stdout.splitlines() if line.startswith(RESULT_MARKER)), None
    )

    if result_line is None:
        stderr = completed.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(
            "Experiment process failed.\n" + (stderr[-1200:] or "no output produced")
        )

    data = json.loads(result_line[len(RESULT_MARKER):])
    if not data.get("success"):
        raise RuntimeError(data.get("error", "unknown error"))
    return data["result"]
