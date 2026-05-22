"""Compatibility shims for sam_audio against newer dep versions.

Two patches live here, both applied at import time:

1. ``BaseModel._from_pretrained`` — newer ``huggingface_hub`` releases
   stopped forwarding ``proxies`` and ``resume_download``; the sam_audio
   override still declares them as required, so we inject ``None``
   defaults.

2. ``SAMAudio.load_state_dict`` — the upstream override is wrapped in
   ``if strict:`` with **no else branch**. When ``strict=False`` is
   passed, the method becomes a silent no-op and the model keeps its
   random initialisation. We replace it with a version that always
   performs the actual load and only changes its *strictness check*
   based on the flag.

Plus one helper for services: ``log_weight_stats(module, log, label=...)``
prints the mean/std of the first parameter in a named submodule, so the
service log shows immediately whether the trained weights took effect.
A pristine random initialisation produces predictable Xavier/He stats;
a trained checkpoint produces something very different. Two restarts
of the same service should report identical stats (the checkpoint is
deterministic). A drifting value across restarts means weights aren't
being loaded.

Importing this module has the side effect of applying both patches; it
must run **before** any sam_audio model construction or load.
"""

from __future__ import annotations

import logging
import re

import torch
import torch.nn as nn

from sam_audio.model.base import BaseModel
from sam_audio.model.model import SAMAudio

LOG = logging.getLogger(__name__)


# ---- Patch 1: hf_hub-compat ------------------------------------------

_orig_from_pretrained = BaseModel._from_pretrained.__func__


def _patched_from_pretrained(cls, **kwargs):
    kwargs.setdefault("proxies", None)
    kwargs.setdefault("resume_download", None)
    return _orig_from_pretrained(cls, **kwargs)


BaseModel._from_pretrained = classmethod(_patched_from_pretrained)
LOG.debug("patched BaseModel._from_pretrained for huggingface_hub compat")


# ---- Patch 2: SAMAudio.load_state_dict no-op-when-not-strict --------
#
# Upstream method body (paraphrased):
#
#     def load_state_dict(self, state_dict, strict=True):
#         if strict:
#             missing, unexpected = super().load_state_dict(state_dict, strict=False)
#             # filter expected-missing submodule keys
#             # raise if anything remains
#
# When strict=False is passed, the entire `if` block is skipped — no
# super().load_state_dict() is invoked, and the model keeps random init.
# Replacement always calls the base loader; strict controls whether to
# raise on residual missing/unexpected keys.

_SKIP_RE = re.compile("(^text_encoder|^visual_ranker|^text_ranker|^span_predictor)")


def _patched_load_state_dict(self, state_dict, strict=True):
    missing, unexpected = nn.Module.load_state_dict(self, state_dict, strict=False)
    if strict:
        residual_missing = [k for k in missing if not _SKIP_RE.search(k)]
        if residual_missing or unexpected:
            raise RuntimeError(
                f"Missing keys: {residual_missing}, unexpected_keys: {unexpected}"
            )
    return missing, unexpected


SAMAudio.load_state_dict = _patched_load_state_dict
LOG.debug("patched SAMAudio.load_state_dict to always actually load")


# ---- Diagnostic helper ------------------------------------------------

def log_weight_stats(
    module: torch.nn.Module, log: logging.Logger, *, label: str
) -> None:
    """Log the mean / std / abs-max of the first parameter of ``module``.

    Use this at service startup right after loading a model. Two restarts
    of the same service should produce identical numbers (the checkpoint
    is deterministic). If they drift, weights aren't being loaded.
    """
    try:
        p = next(module.parameters())
    except StopIteration:
        log.warning("[weight-check %s] module has no parameters", label)
        return
    log.info(
        "[weight-check %s] shape=%s mean=%.5f std=%.5f absmax=%.5f",
        label,
        tuple(p.shape),
        p.float().mean().item(),
        p.float().std().item(),
        p.float().abs().max().item(),
    )


__all__ = ["log_weight_stats"]
