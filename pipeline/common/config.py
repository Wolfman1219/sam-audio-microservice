"""Service-level configuration loaded from environment variables.

Each service reads only the keys it cares about. Defaults are chosen so the
service starts on a vanilla machine, and overrides come from
``docker-compose.yml`` for production runs.

We deliberately don't use Pydantic Settings here — it pulls a lot of weight
for what is essentially `os.environ.get(...)` with type coercion.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch

_PREFIX = "SAMP_"


def _env(key: str, default: Any = None) -> str | None:
    return os.environ.get(_PREFIX + key, default)


def _env_int(key: str, default: int) -> int:
    raw = _env(key)
    return int(raw) if raw is not None else default


def _env_float(key: str, default: float) -> float:
    raw = _env(key)
    return float(raw) if raw is not None else default


def _env_bool(key: str, default: bool) -> bool:
    raw = _env(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


_DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def _env_dtype(key: str, default: str) -> torch.dtype:
    raw = (_env(key) or default).strip().lower()
    if raw not in _DTYPES:
        raise ValueError(f"Unknown dtype '{raw}'; expected one of {list(_DTYPES)}")
    return _DTYPES[raw]


@dataclass(frozen=True)
class CommonConfig:
    """Settings shared by every service."""

    device: torch.device
    dtype: torch.dtype
    batch_max_size: int
    batch_max_wait_ms: int

    @classmethod
    def from_env(
        cls,
        *,
        default_batch_max_size: int = 8,
        default_batch_max_wait_ms: int = 10,
    ) -> "CommonConfig":
        device_str = _env("DEVICE", "cuda:0")
        if device_str.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"SAMP_DEVICE={device_str} but no CUDA device is visible. "
                "Check container GPU assignment."
            )
        return cls(
            device=torch.device(device_str),
            dtype=_env_dtype("DTYPE", "bfloat16"),
            batch_max_size=_env_int("BATCH_MAX_SIZE", default_batch_max_size),
            batch_max_wait_ms=_env_int("BATCH_MAX_WAIT_MS", default_batch_max_wait_ms),
        )


# Service-specific helpers — kept here so config logic lives in one file.

def model_name(default: str = "t5-base") -> str:
    return _env("MODEL_NAME", default)


def model_id(default: str = "facebook/sam-audio-large") -> str:
    return _env("MODEL_ID", default)


def ode_method(default: str = "midpoint") -> str:
    return _env("ODE_METHOD", default)


def ode_step_size(default: float = 2 / 32) -> float:
    return _env_float("ODE_STEP_SIZE", default)


def torch_compile_enabled(default: bool = False) -> bool:
    return _env_bool("TORCH_COMPILE", default)
