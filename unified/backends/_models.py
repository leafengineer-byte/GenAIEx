"""Model availability helpers.

Heavy experiments download multi-gigabyte weights on first use. Rather than let a
request hang or fail deep inside a framework, these helpers check the local
Hugging Face cache and the free disk space so the UI can fail fast with a clear,
actionable message. If there is enough space, the request proceeds and the model
downloads normally.
"""
import ctypes
import os
import shutil
from pathlib import Path

from ._loader import PROJECT_ROOT
from ._runner import DEFAULT_PYTHON

# The big diffusion models keep several gigabytes of fp32 weights resident while
# they run, so they need a GPU or a large amount of system memory.
HEAVY_MIN_RAM_GB = 16.0


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def total_ram_gb() -> float:
    """Total physical memory in GB, or 0.0 when it cannot be read."""
    if os.name != "nt":
        return 0.0
    status = _MemoryStatus()
    status.dwLength = ctypes.sizeof(_MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return 0.0
    return status.ullTotalPhys / 1_000_000_000


def heavy_models_supported() -> bool:
    """Whether the multi-gigabyte diffusion models can be held in memory here."""
    ram = total_ram_gb()
    return ram == 0.0 or ram >= HEAVY_MIN_RAM_GB


def require_heavy_hardware(feature: str) -> None:
    """Raise a clear error when the machine cannot fit a heavy model in memory."""
    ram = total_ram_gb()
    if ram and ram < HEAVY_MIN_RAM_GB:
        raise RuntimeError(
            f"{feature} keeps several gigabytes of model weights in memory and needs "
            f"a GPU or at least {HEAVY_MIN_RAM_GB:g} GB of RAM. This machine has "
            f"{ram:.1f} GB, so it is disabled here. Run it on a machine with a GPU "
            "using the same experiment scripts."
        )


def hf_hub_dir() -> Path:
    env = os.environ.get("HUGGINGFACE_HUB_CACHE") or os.environ.get("HF_HUB_CACHE")
    if env:
        return Path(env)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


WEIGHT_SUFFIXES = {
    ".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".h5", ".onnx",
    ".msgpack", ".gguf",
}


def hf_cached(repo_id: str, min_bytes: int = 1_000_000) -> bool:
    """True when a model's weights are fully present in the local cache.

    A repository directory can exist after a partial download that only fetched
    config files, so checking for the directory is not enough. Hugging Face keeps
    the real payload either as a blob (with a matching file in ``snapshots/``) or
    directly under ``snapshots/``; require at least one complete weight file above
    ``min_bytes`` before treating the model as available.
    """
    repo = hf_hub_dir() / ("models--" + repo_id.replace("/", "--"))
    if not repo.is_dir():
        return False
    for path in repo.rglob("*"):
        if path.suffix.lower() not in WEIGHT_SUFFIXES:
            continue
        try:
            if path.is_file() and path.stat().st_size >= min_bytes:
                return True
        except OSError:
            continue
    return False


def audiocraft_available() -> bool:
    site_packages = DEFAULT_PYTHON.parent.parent / "Lib" / "site-packages"
    return (site_packages / "audiocraft").is_dir()


def free_gb() -> float:
    return shutil.disk_usage(str(PROJECT_ROOT)).free / 1_000_000_000


def require_model(repo_id: str, size_gb: float, install_hint: str = "") -> None:
    """Raise a clear error when a model is missing and cannot be downloaded."""
    if hf_cached(repo_id):
        return

    if free_gb() < size_gb:
        hint = f" {install_hint}" if install_hint else ""
        raise RuntimeError(
            f"{repo_id} is not in the local cache and only {free_gb():.2f} GB of disk "
            f"is free, but roughly {size_gb:g} GB is needed.{hint} Free some disk space "
            "and run this experiment again to download the weights."
        )
