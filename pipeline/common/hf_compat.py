"""Compatibility shim for sam_audio + recent huggingface_hub releases.

In newer ``huggingface_hub`` versions, ``ModelHubMixin.from_pretrained``
no longer forwards ``proxies`` and ``resume_download`` to subclass
``_from_pretrained`` overrides. ``sam_audio.model.base.BaseModel``
declares both as required keyword-only arguments, so any call to
``SAMAudio.from_pretrained(...)`` raises::

    TypeError: BaseModel._from_pretrained() missing 2 required
    keyword-only arguments: 'proxies' and 'resume_download'

We patch BaseModel to inject ``None`` defaults for those names before
any model load. Importing this module has the side effect of applying
the patch; it must run **before** the first ``from_pretrained`` call,
so each service imports it at the top of its ``app.py`` (before any
``sam_audio`` import that would trigger a load).

The patch uses ``setdefault``, so if a future ``huggingface_hub``
version starts passing these names through again, the original values
win and our defaults are silently ignored.
"""

from __future__ import annotations

import logging

from sam_audio.model.base import BaseModel

LOG = logging.getLogger(__name__)

# Capture the bound classmethod's underlying function so we can rebuild
# the classmethod after wrapping. `__func__` is the unbound function
# the classmethod descriptor wraps.
_orig_from_pretrained = BaseModel._from_pretrained.__func__


def _patched_from_pretrained(cls, **kwargs):
    kwargs.setdefault("proxies", None)
    kwargs.setdefault("resume_download", None)
    return _orig_from_pretrained(cls, **kwargs)


BaseModel._from_pretrained = classmethod(_patched_from_pretrained)
LOG.debug(
    "patched sam_audio.BaseModel._from_pretrained for huggingface_hub compat"
)

# Nothing to export — the import is the patch.
__all__: list[str] = []
