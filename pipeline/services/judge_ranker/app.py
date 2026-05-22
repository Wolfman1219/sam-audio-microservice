"""Judge-ranker microservice.

Wraps ``sam_audio.ranking.judge.JudgeRanker``, which scores a set of
separated audio candidates against the original input mixture + text
description. The underlying ``SAMAudioJudgeModel`` is itself a multi-model
stack (DAC encoder + two PE-AV transformers + ModernBert + a head); it
deserves its own GPU when memory is tight, but on a 48 GB L40 it shares
comfortably with the lighter services.

Wire payload (request) — one item per HTTP call::

    {
        "extracted_audio":      Tensor[n_cand, S_max],   # candidate target waveforms
        "extracted_wav_sizes":  Tensor[n_cand] (int64),  # valid lengths
        "input_audio":          Tensor[S_in],            # original mixture, mono
        "input_wav_size":       Tensor[] (int64),
        "description":          str,
        "n_candidates":         int,
        "sample_rate":          int (default 48000),
    }

Wire payload (response)::

    {"scores": Tensor[n_cand] (float32)}

The Judge returns four heads (overall, recall, precision, faithfulness);
we report ``overall`` here because that's what the upstream
``JudgeRanker.forward`` exposes and it's the score the pipeline cares
about. If you want the breakdown later, we can add a second endpoint that
returns all four.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

# Patch sam_audio.BaseModel for newer huggingface_hub versions. Must
# precede any sam_audio import that triggers a model load.
from pipeline.common import hf_compat  # noqa: F401

import torch
from fastapi import FastAPI, Request

from sam_audio.model.config import JudgeRankerConfig
from sam_audio.ranking.judge import JudgeRanker

from pipeline.common import config as cfg
from pipeline.common.batching import AsyncBatcher
from pipeline.common.http import read_payload, write_payload

LOG = logging.getLogger("judge_ranker")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


def _env_str(key: str, default: str) -> str:
    import os
    return os.environ.get("SAMP_" + key, default)


class JudgeModelWrapper:
    def __init__(self, device: torch.device, dtype: torch.dtype, model_id: str):
        LOG.info("loading Judge model: %s", model_id)
        ranker_cfg = JudgeRankerConfig(checkpoint_or_model_id=model_id)
        ranker = JudgeRanker(ranker_cfg)
        # JudgeRanker is an nn.Module containing self.model (also a Module).
        # `.to(device, dtype)` propagates through nn.Module submodules.
        ranker = ranker.to(device=device, dtype=dtype).eval()
        # The processor is not an nn.Module; nothing to move there.
        self.ranker = ranker
        self.device = device
        self.dtype = dtype
        LOG.info("Judge ready on %s (dtype=%s)", device, dtype)

    @torch.inference_mode()
    def score(
        self,
        input_audio_list: list[torch.Tensor],
        extracted_audio_list: list[torch.Tensor],
        descriptions: list[str],
        sample_rate: int,
    ) -> torch.Tensor:
        """Return (B, n_cand) overall scores.

        ``input_audio_list[i]`` and ``extracted_audio_list[i]`` must both
        have shape ``(n_cand, T)`` — the same n_cand across items.
        """
        # JudgeRanker handles the per-item flattening internally; the
        # processor pads variable-length inputs across the flat batch.
        return self.ranker(
            input_audio=input_audio_list,
            extracted_audio=extracted_audio_list,
            descriptions=descriptions,
            sample_rate=sample_rate,
        )


# ---- Batching ---------------------------------------------------------

async def _process_score_batch(
    model: JudgeModelWrapper, items: list[dict]
) -> list[dict]:
    """Coalesce per-item score requests into one model call.

    All items in a coalesced batch must share the same n_candidates; the
    Judge's reshape ``view(bsz, ncandidates)`` requires it. We bucket by
    n_candidates and recurse for safety, but for the default
    dataset-cleaning flow n_candidates is fixed so this is a no-op.
    """
    cands = {item["n_candidates"] for item in items}
    if len(cands) > 1:
        out: list[dict | None] = [None] * len(items)
        for n in cands:
            sub_idxs = [i for i, it in enumerate(items) if it["n_candidates"] == n]
            sub = [items[i] for i in sub_idxs]
            sub_out = await _process_score_batch(model, sub)
            for i, r in zip(sub_idxs, sub_out):
                out[i] = r
        return out  # type: ignore[return-value]

    sample_rate = items[0]["sample_rate"]
    n_cand = items[0]["n_candidates"]

    input_audio_list: list[torch.Tensor] = []
    extracted_audio_list: list[torch.Tensor] = []
    descriptions: list[str] = []
    for item in items:
        # Trim padding from the original input, then expand to (n_cand, S_in).
        # Inputs must match the model's dtype — Judge contains a DACVAE
        # encoder whose Conv1d weights are cast along with the rest of the
        # model when we move it to bf16, so feeding fp32 audio in would
        # raise a dtype mismatch.
        in_size = int(item["input_wav_size"].item())
        input_wav = item["input_audio"][:in_size].to(model.device, dtype=model.dtype)
        input_audio_list.append(input_wav.unsqueeze(0).expand(n_cand, -1).contiguous())

        # Trim per-candidate extracted waveforms to their declared sizes,
        # then pad to the max within this item so we have a clean (n_cand, S) tensor.
        ex = item["extracted_audio"]               # (n_cand, S_max_padded)
        ex_sizes = item["extracted_wav_sizes"]     # (n_cand,)
        max_S = int(ex_sizes.max().item())
        ex_trimmed = torch.zeros(n_cand, max_S, dtype=model.dtype, device=model.device)
        for c in range(n_cand):
            s = int(ex_sizes[c].item())
            ex_trimmed[c, :s] = ex[c, :s].to(model.dtype)
        extracted_audio_list.append(ex_trimmed)

        descriptions.append(item["description"])

    scores = model.score(
        input_audio_list, extracted_audio_list, descriptions, sample_rate=sample_rate,
    )  # (B, n_cand)
    scores_cpu = scores.detach().float().cpu()

    return [{"scores": scores_cpu[i]} for i in range(len(items))]


@asynccontextmanager
async def lifespan(app: FastAPI):
    common = cfg.CommonConfig.from_env(
        default_batch_max_size=4, default_batch_max_wait_ms=20,
    )
    model_id = _env_str("JUDGE_MODEL_ID", "facebook/sam-audio-judge")
    model = JudgeModelWrapper(common.device, common.dtype, model_id)

    batcher = AsyncBatcher(
        max_batch_size=common.batch_max_size,
        max_wait_ms=common.batch_max_wait_ms,
        process_fn=lambda items: _process_score_batch(model, items),
        name="judge",
    )
    await batcher.start()
    app.state.batcher = batcher
    try:
        yield
    finally:
        await batcher.stop()


app = FastAPI(lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True}


@app.post("/score")
async def score(request: Request):
    payload = await read_payload(request)
    required = ("extracted_audio", "extracted_wav_sizes", "input_audio",
                "input_wav_size", "description", "n_candidates")
    missing = [k for k in required if k not in payload]
    if missing:
        return write_payload({"error": f"missing keys: {missing}"})

    item = {
        "extracted_audio": payload["extracted_audio"],
        "extracted_wav_sizes": payload["extracted_wav_sizes"],
        "input_audio": payload["input_audio"],
        "input_wav_size": payload["input_wav_size"],
        "description": payload["description"],
        "n_candidates": int(payload["n_candidates"]),
        "sample_rate": int(payload.get("sample_rate", 48_000)),
    }
    result = await request.app.state.batcher.submit(item)
    return write_payload({"scores": result["scores"]})