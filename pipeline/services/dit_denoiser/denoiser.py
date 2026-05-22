"""Core denoising logic for the DiT service.

Lifts the ODE loop out of ``SAMAudio.separate()`` and re-expresses it on
pure tensors so it can be driven from an HTTP request. Inputs come
pre-encoded from the text-encoder and audio-codec services; the output is
the final latent tensor ready to send to the audio-codec for decoding.

This file deliberately mirrors the flow in the original ``separate()`` and
``_get_forward_args()`` so behaviour is identical bit-for-bit
(modulo the random noise sample, which the caller can pin via a seed).

Two key adjustments from the original::

1. ``_get_audio_features`` did ``cat([codec_out, codec_out], dim=2)``
   producing a 2C-wide tensor that's used both as conditioning and as the
   shape of the noise. The audio-codec service returns plain C-wide
   features, so we do the doubling here, inside the service that needs it.

2. Candidate expansion (``_repeat_for_reranking``) is also here: callers
   pass ``n_candidates`` and we replicate everything internally before the
   ODE loop. The orchestrator does not need to know about this.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
from torchdiffeq import odeint

LOG = logging.getLogger("dit_denoiser.core")


@dataclass
class DenoiseInputs:
    """Pre-encoded inputs for one denoise call.

    Shapes use the convention:
        B  – original batch size (number of distinct items)
        T  – audio feature length (after DACVAE encoding)
        Tt – text token length
        C  – DACVAE codebook dim (128 for sam-audio-large)
        D  – text encoder hidden size (768 for t5-base)
    """

    audio_features: torch.Tensor      # (B, T_max, C) bf16 or fp32
    feature_sizes: torch.Tensor       # (B,) int64 — valid T per item
    text_features: torch.Tensor       # (B, Tt_max, D)
    text_mask: torch.Tensor           # (B, Tt_max) bool
    n_candidates: int = 1
    noise_seed: Optional[int] = None  # If set, deterministic noise.
    noise: Optional[torch.Tensor] = None  # If set, used directly (shape: (B*n_cand, T_max, 2*C))


def _default_anchors(
    audio_pad_mask: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the no-anchor default anchor_ids / anchor_alignment.

    Mirrors ``Batch.process_anchors(anchors=None)``: token 0 = "<null>",
    token 1 = "<pad>". Each timestep points at <null>; padded timesteps
    point at <pad>.
    """
    B, T = audio_pad_mask.shape
    # Two anchor tokens per item: [<null>, <pad>].
    anchor_ids = torch.zeros(B, 2, dtype=torch.long, device=device)
    anchor_ids[:, 1] = 3  # "<pad>" index in the dict
    # 0 = <null>, 1 = <pad>. In Batch.process_anchors the alignment uses
    # *position within `current`* rather than the dict id, but here we
    # have only two positions (<null>=0, <pad>=1) so it matches.
    anchor_alignment = torch.zeros(B, T, dtype=torch.long, device=device)
    anchor_alignment[~audio_pad_mask] = 1
    return anchor_ids, anchor_alignment


def _mask_from_sizes(sizes: torch.Tensor, max_T: int) -> torch.Tensor:
    arange = torch.arange(max_T, device=sizes.device)
    return arange.unsqueeze(0) < sizes.unsqueeze(1)


def _repeat(x: torch.Tensor, n: int) -> torch.Tensor:
    """Expand the batch dim by interleaving n copies, matching SAMAudio._repeat_for_reranking."""
    if n <= 1:
        return x
    B = x.size(0)
    return x.unsqueeze(1).expand(B, n, *x.shape[1:]).reshape(B * n, *x.shape[1:])


class DenoiseRunner:
    """Runs the full ODE loop against a loaded SAMAudio (DiT only)."""

    def __init__(
        self,
        model,                       # SAMAudio with submodules nulled
        *,
        device: torch.device,
        dtype: torch.dtype,
        ode_method: str = "midpoint",
        ode_step_size: float = 2 / 32,
    ):
        self.model = model
        self.device = device
        self.dtype = dtype
        self.ode_opt = {"method": ode_method, "options": {"step_size": ode_step_size}}

    @torch.inference_mode()
    def __call__(self, inputs: DenoiseInputs) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one ODE solve. Returns (latents, feature_sizes) on CPU.

        Output ``latents`` shape: (B * n_candidates * 2, C, T_max), with the
        target and residual interleaved per the original code's
        ``wavs.view(B, 2, -1)`` reshape — i.e. for item i, candidate k::

            latents[ (i*n_candidates + k) * 2 + 0 ]   # target
            latents[ (i*n_candidates + k) * 2 + 1 ]   # residual

        ``feature_sizes`` (B * n_candidates,) gives the per-item valid T so
        the codec can trim padding before decoding.
        """
        device = self.device
        n_cand = inputs.n_candidates

        # Move inputs to GPU once.
        audio_features_C = inputs.audio_features.to(device=device, dtype=self.dtype, non_blocking=True)
        text_features = inputs.text_features.to(device=device, dtype=self.dtype, non_blocking=True)
        text_mask = inputs.text_mask.to(device=device, non_blocking=True)
        feature_sizes = inputs.feature_sizes.to(device=device, non_blocking=True)

        B, T_max, C = audio_features_C.shape

        # Reproduce SAMAudio._get_audio_features doubling: the DiT expects
        # (B, T, 2*C) conditioning where the two halves are identical copies
        # of the codec output. See the README of this service file.
        audio_features = torch.cat([audio_features_C, audio_features_C], dim=2)

        # Build padding mask and default anchors from feature sizes.
        audio_pad_mask = _mask_from_sizes(feature_sizes, T_max).to(device)
        anchor_ids, anchor_alignment = _default_anchors(audio_pad_mask, device=device)

        # Expand for re-ranking candidates. Doing this server-side keeps the
        # client interface simple (one item in, n_candidates results out).
        af = _repeat(audio_features, n_cand)
        tf = _repeat(text_features, n_cand)
        tm = _repeat(text_mask, n_cand)
        pm = _repeat(audio_pad_mask, n_cand)
        aid = _repeat(anchor_ids, n_cand)
        aal = _repeat(anchor_alignment, n_cand)
        fs_expanded = _repeat(feature_sizes, n_cand)

        forward_args = {
            "audio_features": af,
            "text_features": tf,
            "text_mask": tm,
            "masked_video_features": None,
            "anchor_ids": aid,
            "anchor_alignment": aal,
            "audio_pad_mask": pm,
        }

        # Sample noise. Same shape as `audio_features` (already 2C-wide).
        # Priority: explicit tensor > seed > default torch.randn_like.
        if inputs.noise is not None:
            noise = inputs.noise.to(device=device, dtype=self.dtype, non_blocking=True)
            if noise.shape != af.shape:
                raise ValueError(
                    f"noise tensor shape {tuple(noise.shape)} does not match "
                    f"expanded audio_features shape {tuple(af.shape)}; "
                    f"expected (B*n_cand={af.size(0)}, T={af.size(1)}, 2*C={af.size(2)})"
                )
        elif inputs.noise_seed is not None:
            g = torch.Generator(device=device).manual_seed(inputs.noise_seed)
            noise = torch.randn(
                af.shape, generator=g, device=device, dtype=self.dtype,
            )
        else:
            noise = torch.randn_like(af)

        def vector_field(t: torch.Tensor, noisy_audio: torch.Tensor) -> torch.Tensor:
            return self.model.forward(
                noisy_audio=noisy_audio,
                time=t.expand(noisy_audio.size(0)),
                **forward_args,
            )

        states = odeint(
            vector_field,
            noise,
            torch.tensor([0.0, 1.0], device=device),
            **self.ode_opt,
        )
        # generated_features: (B*n_cand, T, 2C) → (B*n_cand, 2C, T)
        generated_features = states[-1].transpose(1, 2)
        BN = generated_features.size(0)

        # Reshape (B*n_cand, 2C, T) → (B*n_cand * 2, C, T) so that for each
        # item the target latent comes first then the residual. This matches
        # the original `view(2*B, C, T)` reshape before decode.
        latents = generated_features.reshape(BN * 2, C, T_max)

        # feature_sizes is per (item, candidate) — we report it once per
        # item-candidate, the codec will use the same value for both
        # target and residual of that pair.
        return latents.to(dtype=torch.float32).cpu(), fs_expanded.cpu()
