"""Worker entry point.

Runs a single task from ``unified/backends/_tasks.py`` inside whichever Python
interpreter invokes it, and prints the JSON result on stdout behind a marker so
any log noise from the frameworks can be ignored by the parent process.

Usage:
    python worker.py <task_name>      # payload JSON is read from stdin
"""
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backends._tasks import TASKS  # noqa: E402

RESULT_MARKER = "@@RESULT@@"


def main() -> int:
    if len(sys.argv) < 2:
        print(RESULT_MARKER + json.dumps({"success": False, "error": "missing task name"}))
        return 1

    task_name = sys.argv[1]
    task = TASKS.get(task_name)
    if task is None:
        print(RESULT_MARKER + json.dumps({"success": False, "error": f"unknown task: {task_name}"}))
        return 1

    try:
        payload = json.loads(sys.stdin.read() or "{}")
        result = task(payload)
        print(RESULT_MARKER + json.dumps({"success": True, "result": result}))
        return 0
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        print(RESULT_MARKER + json.dumps({"success": False, "error": f"{type(exc).__name__}: {exc}"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
