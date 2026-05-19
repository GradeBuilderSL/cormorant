"""
_matmul_hw_config.py — resolve MatmulKernel compile-time bounds from the
platform JSON, mirroring _pool_hw_config.py.

Currently the only consumer is ``OnnxGraph._rewrite_pointwise_conv_as_matmul``,
which needs ``MM_MAX_K`` to skip pointwise-Conv rewrites whose in-channel
count would exceed the MatmulKernel's compile-time inner-dimension cap.
The C++ build reads the same JSON via
``kernels/matmul/CMakeLists.txt::matmul_load_constants``, so this resolver
guarantees the Python and C++ sides agree.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Mapping, Tuple

# Repo layout:
#   <repo>/inference-scheduler/src/_matmul_hw_config.py   ← this file
#   <repo>/platforms/<name>.json
_SCHEDULER_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT     = _SCHEDULER_DIR.parent
_PLATFORMS_DIR = _REPO_ROOT / "platforms"

_DEFAULT_PLATFORM = "kv260"

# Required JSON fields → exported Python constant.
_REQUIRED: Tuple[Tuple[str, str], ...] = (
    ("tile_n", "MM_TILE_N"),
    ("tile_m", "MM_TILE_M"),
    ("tile_k", "MM_TILE_K"),
    ("max_k",  "MM_MAX_K"),
)


class MatmulHwConfigError(RuntimeError):
    """Platform JSON is missing / malformed for the matmul kernel."""


def _resolve_platform_path(platform_name: str) -> Path:
    path = _PLATFORMS_DIR / f"{platform_name}.json"
    if not path.is_file():
        raise MatmulHwConfigError(
            f"Platform file not found: {path}.  Set AXI_PLATFORM to a "
            f"<name> for which platforms/<name>.json exists."
        )
    return path


def _load_matmul_section(path: Path) -> Mapping[str, int]:
    try:
        with path.open("r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise MatmulHwConfigError(
            f"Platform JSON {path} is not valid JSON: {e}"
        ) from e
    try:
        section = data["kernels"]["matmul"]
    except (KeyError, TypeError) as e:
        raise MatmulHwConfigError(
            f"Platform JSON {path} is missing the 'kernels.matmul' object."
        ) from e
    if not isinstance(section, Mapping):
        raise MatmulHwConfigError(
            f"Platform JSON {path}: 'kernels.matmul' must be an object, "
            f"got {type(section).__name__}."
        )
    return section


def resolve(platform_name: str = None) -> Dict[str, int]:
    """Resolve MatmulKernel constants for the given platform.

    ``platform_name=None`` falls back to ``AXI_PLATFORM`` from the env, then
    to the built-in default ``kv260``.
    """
    if platform_name is None:
        platform_name = os.environ.get("AXI_PLATFORM", _DEFAULT_PLATFORM)
    path    = _resolve_platform_path(platform_name)
    section = _load_matmul_section(path)
    out: Dict[str, int] = {}
    for json_key, py_attr in _REQUIRED:
        if json_key not in section:
            raise MatmulHwConfigError(
                f"Platform JSON {path}: 'kernels.matmul.{json_key}' is "
                f"required but missing.  Add it (or pick a different "
                f"AXI_PLATFORM) so the Python validator and C++ kernel "
                f"build agree on bounds."
            )
        val = section[json_key]
        if not isinstance(val, int) or isinstance(val, bool):
            raise MatmulHwConfigError(
                f"Platform JSON {path}: 'kernels.matmul.{json_key}' must "
                f"be an integer, got {type(val).__name__} ({val!r})."
            )
        out[py_attr] = val
    return out


_CFG = resolve()

# Exported constants — frozen at import time for the default platform.
MM_TILE_N: int = _CFG["MM_TILE_N"]
MM_TILE_M: int = _CFG["MM_TILE_M"]
MM_TILE_K: int = _CFG["MM_TILE_K"]
MM_MAX_K : int = _CFG["MM_MAX_K"]


__all__ = (
    "MM_TILE_N",
    "MM_TILE_M",
    "MM_TILE_K",
    "MM_MAX_K",
    "MatmulHwConfigError",
    "resolve",
)
