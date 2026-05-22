"""Tensor and payload (de)serialisation for inter-service HTTP.

Every service exchanges payloads as a single ``application/octet-stream``
body containing a torch-serialised dictionary. This keeps the wire format
trivial — no protobuf, no JSON-with-base64 — and torch's pickle protocol
preserves dtype/device-aware tensor metadata without us having to invent
anything.

For typical loads on a 30 s clip (audio features ~3 MB bf16, text features
~30 KB) the encode + transfer + decode cost is single-digit milliseconds
on loopback, which the request pipelining hides behind DiT compute.
"""

from __future__ import annotations

import io
from typing import Any

import torch


# `weights_only=True` is the safe default for cross-process loads; it
# prevents arbitrary code execution from a tampered payload. We pay a small
# compatibility cost (no custom classes in payloads), which is fine — every
# payload here is a flat dict of tensors and primitives.
_LOAD_WEIGHTS_ONLY = True


def encode_payload(payload: dict[str, Any]) -> bytes:
    """Serialise a dict of tensors + primitives to bytes.

    Tensors are moved to CPU and made contiguous before serialisation so the
    receiving end can load them onto any device without surprises. The
    caller's tensors are not mutated (we ``.cpu().contiguous()`` copies).
    """
    safe: dict[str, Any] = {}
    for k, v in payload.items():
        if isinstance(v, torch.Tensor):
            # `.contiguous()` after `.cpu()` because a non-contiguous CUDA
            # tensor would otherwise serialise its full storage including
            # the slicing offsets — wasteful.
            safe[k] = v.detach().to("cpu").contiguous()
        else:
            safe[k] = v
    buf = io.BytesIO()
    torch.save(safe, buf)
    return buf.getvalue()


def decode_payload(data: bytes, device: str | torch.device = "cpu") -> dict[str, Any]:
    """Inverse of ``encode_payload``. Materialises tensors on ``device``."""
    buf = io.BytesIO(data)
    obj = torch.load(buf, weights_only=_LOAD_WEIGHTS_ONLY, map_location=device)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict payload, got {type(obj).__name__}")
    return obj
