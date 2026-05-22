"""CLAP-ranker microservice.

Wraps ``sam_audio.ranking.clap.ClapRanker`` (LAION CLAP HTSAT-tiny).
Unlike the Judge, CLAP only needs the candidate waveforms + description —
no reference to the original mixture. The internal data prep handles its
own resampling to 48 kHz and 10 s windowing via the LAION data helpers.

Wire payload (request) — one item per HTTP call::

    {
        "extracted_audio":     Tensor[n_cand, S_max],
        "extracted_wav_sizes": Tensor[n_cand] (int64),
        "description":         str,
        "n_candidates":        int,
        "sample_rate":         int (default 48000),
    }

Wire payload (response)::

    {"scores": Tensor[n_cand] (float32)}
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

# Patch sam_audio.BaseModel for newer huggingface_hub versions. Must
# precede any sam_audio import that triggers a model load.
from pipeline.common import hf_compat  # noqa: F401

import torch
from fastapi import FastAPI, Request

from sam_audio.model.config import ClapRankerConfig
from sam_audio.ranking.clap import ClapRanker

from pipeline.common import config as cfg
from pipeline.common.batching import AsyncBatcher
from pipeline.common.http import read_payload, write_payload

LOG = logging.getLogger("clap_ranker")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


class ClapModelWrapper:
    def __init__(self, device: torch.device, dtype: torch.dtype):
        LOG.info("loading CLAP (HTSAT-tiny + text branch)")
        # ClapRankerConfig has an optional checkpoint override; default is fine.
        ranker = ClapRanker(ClapRankerConfig())
        # CLAP's internals (laion_clap) prefer fp32 for the data prep, so we
        # cast the audio/text projection heads to dtype but let the model
        # internals decide. Move to device first; CLAP modules accept it.
        ranker = ranker.to(device=device).eval()
        # Don't force dtype on the whole module — laion_clap has its own
        # tokenizer + data utilities that expect fp32 inputs and cause
        # cryptic errors otherwise. The HTSAT trunk we'll let stay fp32
        # too; on L40 it's a few hundred MB, no reason to fight it.
        self.ranker = ranker
        self.device = device

    @torch.inference_mode()
    def score(
        self,
        extracted_audio_list: list[torch.Tensor],
        descriptions: list[str],
        sample_rate: int,
    ) -> torch.Tensor:
        return self.ranker(
            extracted_audio=extracted_audio_list,
            descriptions=descriptions,
            sample_rate=sample_rate,
        )


async def _process_score_batch(
    model: ClapModelWrapper, items: list[dict]
) -> list[dict]:
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

    extracted_audio_list: list[torch.Tensor] = []
    descriptions: list[str] = []
    for item in items:
        ex = item["extracted_audio"]
        ex_sizes = item["extracted_wav_sizes"]
        max_S = int(ex_sizes.max().item())
        ex_trimmed = torch.zeros(n_cand, max_S, dtype=torch.float32)
        for c in range(n_cand):
            s = int(ex_sizes[c].item())
            ex_trimmed[c, :s] = ex[c, :s].float()
        extracted_audio_list.append(ex_trimmed.to(model.device))
        descriptions.append(item["description"])

    scores = model.score(extracted_audio_list, descriptions, sample_rate=sample_rate)
    # ClapRanker returns (B, n_cand). Some versions return (B, n_cand, 1) —
    # squeeze defensively.
    if scores.ndim == 3:
        scores = scores.squeeze(-1)
    scores_cpu = scores.detach().float().cpu()
    return [{"scores": scores_cpu[i]} for i in range(len(items))]


@asynccontextmanager
async def lifespan(app: FastAPI):
    common = cfg.CommonConfig.from_env(
        default_batch_max_size=8, default_batch_max_wait_ms=15,
    )
    model = ClapModelWrapper(common.device, common.dtype)

    batcher = AsyncBatcher(
        max_batch_size=common.batch_max_size,
        max_wait_ms=common.batch_max_wait_ms,
        process_fn=lambda items: _process_score_batch(model, items),
        name="clap",
    )
    await batcher.start()
    app.state.batcher = batcher
    LOG.info("clap-ranker ready on %s", common.device)
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
    required = ("extracted_audio", "extracted_wav_sizes", "description", "n_candidates")
    missing = [k for k in required if k not in payload]
    if missing:
        return write_payload({"error": f"missing keys: {missing}"})
    item = {
        "extracted_audio": payload["extracted_audio"],
        "extracted_wav_sizes": payload["extracted_wav_sizes"],
        "description": payload["description"],
        "n_candidates": int(payload["n_candidates"]),
        "sample_rate": int(payload.get("sample_rate", 48_000)),
    }
    result = await request.app.state.batcher.submit(item)
    return write_payload({"scores": result["scores"]})
