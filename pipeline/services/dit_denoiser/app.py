"""DiT denoiser microservice — the bottleneck of the pipeline.

Loads the full ``SAMAudio`` checkpoint, nulls out every submodule that
isn't part of the DiT inference path (T5, DACVAE, vision encoder, span
predictor, both rankers), and serves a single endpoint ``POST /denoise``
that runs the entire ODE loop in one RPC.

Each ``/denoise`` call returns the target+residual latents already
shaped for the audio-codec ``/decode`` endpoint.

Per-request batching: callers send one item per request (one audio
clip + its text features); the batcher coalesces concurrent items so the
DiT processes a real batch each forward pass. Re-ranking candidate
expansion happens inside this service, so the effective batch on the GPU
is ``batch_max_size * n_candidates``.
"""

from __future__ import annotations

import asyncio
import gc
import logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, Request

from sam_audio.model.model import SAMAudio

from pipeline.common import config as cfg
from pipeline.common.batching import AsyncBatcher
from pipeline.common.http import read_payload, write_payload

from .denoiser import DenoiseInputs, DenoiseRunner

LOG = logging.getLogger("dit_denoiser")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


def _load_dit_only(model_id: str, device: torch.device, dtype: torch.dtype):
    """Load SAMAudio, strip non-DiT submodules, cast and move to device."""
    LOG.info("loading SAMAudio for DiT-only inference: %s", model_id)
    # Load whole model on CPU so the unused submodules never touch the GPU.
    # strict=False because the checkpoint may not include the T5 / ranker
    # weights that SAMAudio's load_state_dict already tolerates.
    model = SAMAudio.from_pretrained(model_id, map_location="cpu", strict=False)

    # Null out everything we don't need. Order matters slightly — drop
    # references then collect.
    for attr in (
        "text_encoder",
        "audio_codec",
        "vision_encoder",
        "visual_ranker",
        "text_ranker",
        "span_predictor",
        "span_predictor_transform",
    ):
        if hasattr(model, attr):
            setattr(model, attr, None)
    gc.collect()
    torch.cuda.empty_cache()

    # Cast the DiT + small wiring layers to working dtype, move to GPU.
    # We intentionally keep the alignment helpers (proj, align_masked_video,
    # embed_anchors, memory_proj, timestep_emb) and the transformer.
    model = model.to(device=device, dtype=dtype).eval()
    return model


def _maybe_compile(model, enabled: bool):
    """Compile the DiT forward if requested. Compile cost amortises in a few requests."""
    if not enabled:
        return
    LOG.info("torch.compile(mode='reduce-overhead') on transformer")
    try:
        # We compile the transformer module specifically; SAMAudio.forward
        # has Python-level branches that confuse the compiler.
        model.transformer = torch.compile(model.transformer, mode="reduce-overhead")
    except Exception:  # noqa: BLE001
        LOG.exception("torch.compile failed; falling back to eager")


# ---- Batching glue -----------------------------------------------------

async def _process_denoise_batch(
    runner: DenoiseRunner, items: list[dict]
) -> list[dict]:
    """Combine items into one batched ODE call.

    Each item is a dict carrying the pre-encoded tensors plus the candidate
    count. We require all items in the batch to use the same ``n_candidates``
    so the expanded effective batch shape is consistent. If you need to
    mix, route different candidate counts to different batchers (or just
    fall back to per-item processing). For dataset cleaning you typically
    pin one value globally, so this is fine.
    """
    n_cands = {item["n_candidates"] for item in items}
    if len(n_cands) > 1:
        # Run each candidate-count subset separately. Rare path.
        out: list[dict | None] = [None] * len(items)
        for n in n_cands:
            sub_idxs = [i for i, it in enumerate(items) if it["n_candidates"] == n]
            sub_items = [items[i] for i in sub_idxs]
            sub_results = await _process_denoise_batch(runner, sub_items)
            for i, res in zip(sub_idxs, sub_results):
                out[i] = res
        return out  # type: ignore[return-value]

    n_candidates = next(iter(n_cands))

    # Pad audio + text features to the batch's max lengths.
    B = len(items)
    T_max = max(item["audio_features"].size(1) for item in items)
    C = items[0]["audio_features"].size(-1)
    Tt_max = max(item["text_features"].size(1) for item in items)
    D = items[0]["text_features"].size(-1)

    audio_features = torch.zeros(B, T_max, C, dtype=items[0]["audio_features"].dtype)
    feature_sizes = torch.zeros(B, dtype=torch.long)
    text_features = torch.zeros(B, Tt_max, D, dtype=items[0]["text_features"].dtype)
    text_mask = torch.zeros(B, Tt_max, dtype=torch.bool)

    for i, item in enumerate(items):
        T = item["audio_features"].size(1)
        Tt = item["text_features"].size(1)
        audio_features[i, :T] = item["audio_features"][0]
        feature_sizes[i] = T
        text_features[i, :Tt] = item["text_features"][0]
        text_mask[i, :Tt] = item["text_mask"][0]

    inputs = DenoiseInputs(
        audio_features=audio_features,
        feature_sizes=feature_sizes,
        text_features=text_features,
        text_mask=text_mask,
        n_candidates=n_candidates,
        # No shared seed across the batch; per-item seeds would require
        # building noise outside the runner. If determinism matters,
        # restrict to batch_max_size=1.
        noise_seed=None,
    )

    latents, fs_expanded = runner(inputs)
    # latents: (B * n_candidates * 2, C, T_max)
    # fs_expanded: (B * n_candidates,)

    # Scatter back to per-item dicts. Each item gets `n_candidates` latent
    # pairs (target + residual) and matching feature sizes.
    out: list[dict] = []
    for i in range(B):
        start = i * n_candidates * 2
        end = (i + 1) * n_candidates * 2
        out.append({
            "latents": latents[start:end],                 # (n_cand*2, C, T_max)
            "feature_sizes": fs_expanded[i * n_candidates : (i + 1) * n_candidates],
        })
    return out


@asynccontextmanager
async def lifespan(app: FastAPI):
    common = cfg.CommonConfig.from_env(
        default_batch_max_size=4, default_batch_max_wait_ms=20,
    )
    model = _load_dit_only(cfg.model_id(), common.device, common.dtype)
    _maybe_compile(model, cfg.torch_compile_enabled())

    runner = DenoiseRunner(
        model,
        device=common.device,
        dtype=common.dtype,
        ode_method=cfg.ode_method(),
        ode_step_size=cfg.ode_step_size(),
    )

    batcher = AsyncBatcher(
        max_batch_size=common.batch_max_size,
        max_wait_ms=common.batch_max_wait_ms,
        process_fn=lambda items: _process_denoise_batch(runner, items),
        name="dit-denoise",
    )
    await batcher.start()
    app.state.batcher = batcher
    app.state.runner = runner
    LOG.info("dit-denoiser ready on %s (dtype=%s, compile=%s)",
             common.device, common.dtype, cfg.torch_compile_enabled())
    try:
        yield
    finally:
        await batcher.stop()


app = FastAPI(lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True}


@app.post("/denoise")
async def denoise(request: Request):
    """Run the ODE loop and return the target+residual latents.

    Request payload::

        {
            "audio_features": Tensor[1, T, C],
            "feature_sizes":  Tensor[1] (int64),
            "text_features":  Tensor[1, Tt, D],
            "text_mask":      Tensor[1, Tt] (bool),
            "n_candidates":   int  (>=1),
            "noise":          Tensor[n_candidates, T, 2*C]  (optional, pinned noise),
            "noise_seed":     int  (optional, alternative to raw noise),
        }

    Response payload::

        {
            "latents":        Tensor[n_candidates*2, C, T],  # target,resid,target,resid,...
            "feature_sizes":  Tensor[n_candidates] (int64),
        }

    When ``noise`` is supplied (parity testing / determinism), the request
    bypasses the cross-item batcher and runs alone — sharing a batch
    would require us to mix per-item noise tensors, which we don't.
    """
    payload = await read_payload(request)
    audio_features = payload["audio_features"]
    feature_sizes = payload["feature_sizes"]
    text_features = payload["text_features"]
    text_mask = payload["text_mask"]
    n_candidates = int(payload.get("n_candidates", 1))
    raw_noise = payload.get("noise", None)
    noise_seed = payload.get("noise_seed", None)
    if n_candidates < 1:
        return write_payload({"error": "n_candidates must be >= 1"})
    if audio_features.size(0) != 1:
        return write_payload({"error": "send one item per request; batching is server-side"})

    # Trim padded features to their valid extent before submitting; the
    # batcher will re-pad to its own batch's max length.
    T = int(feature_sizes[0].item())
    Tt = int(text_mask.sum(dim=1).max().item())

    if raw_noise is not None or noise_seed is not None:
        # Determinism path — bypass batcher entirely, run as batch-of-1.
        # The batcher combines per-item noise from different requests into
        # one big tensor; honouring a per-item seed there would require
        # routing each seeded item separately anyway.
        from .denoiser import DenoiseInputs
        runner: "DenoiseRunner" = request.app.state.runner
        inputs = DenoiseInputs(
            audio_features=audio_features[:, :T].contiguous(),
            feature_sizes=torch.tensor([T], dtype=torch.long),
            text_features=text_features[:, :Tt].contiguous(),
            text_mask=text_mask[:, :Tt].contiguous(),
            n_candidates=n_candidates,
            noise=raw_noise,
            noise_seed=int(noise_seed) if noise_seed is not None else None,
        )
        latents, fs_expanded = runner(inputs)
        return write_payload({
            "latents": latents.reshape(n_candidates * 2, latents.size(1), latents.size(2)),
            "feature_sizes": fs_expanded,
        })

    result = await request.app.state.batcher.submit({
        "audio_features": audio_features[:, :T].contiguous(),
        "text_features": text_features[:, :Tt].contiguous(),
        "text_mask": text_mask[:, :Tt].contiguous(),
        "n_candidates": n_candidates,
    })

    return write_payload({
        "latents": result["latents"],
        "feature_sizes": result["feature_sizes"],
    })
