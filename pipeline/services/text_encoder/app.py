"""Text-encoder microservice.

Wraps ``sam_audio.model.text_encoder.T5TextEncoder``. One endpoint,
``POST /encode``, accepts a list of strings and returns the per-token
hidden states + attention mask.

Wire payload (request)::

    {"descriptions": ["man speaking", "thunder", ...]}

Wire payload (response)::

    {
        "text_features": Tensor[B, T, D],  # bf16 by default
        "text_mask":     Tensor[B, T] of bool,
    }

T tokens are padded across the batch (model uses pad_mode='longest').
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

# Patch sam_audio.BaseModel for newer huggingface_hub versions. Side-effect
# import; must precede any `sam_audio` import that triggers a model load.
from pipeline.common import hf_compat  # noqa: F401

import torch
from fastapi import FastAPI, Request

from sam_audio.model.config import T5EncoderConfig
from sam_audio.model.text_encoder import T5TextEncoder

from pipeline.common import config as cfg
from pipeline.common.batching import AsyncBatcher
from pipeline.common.http import read_payload, write_payload

LOG = logging.getLogger("text_encoder")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


# ---- Model wrapper -----------------------------------------------------

class TextEncoderModel:
    def __init__(self, device: torch.device, dtype: torch.dtype, name: str):
        LOG.info("loading T5 text encoder: %s", name)
        config = T5EncoderConfig(name=name)
        model = T5TextEncoder(config)
        # T5 is fp32 by default; cast to working dtype. Embedding weights
        # tolerate bf16 well.
        model = model.to(device=device, dtype=dtype).eval()
        self.model = model
        self.device = device
        self.dtype = dtype
        # Compile is overkill for T5; skip.

    @torch.inference_mode()
    def encode(self, descriptions: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        # T5TextEncoder.forward handles tokenisation + padding internally.
        text_features, text_mask = self.model(descriptions)
        return text_features.contiguous(), text_mask.contiguous()


# ---- Batching glue -----------------------------------------------------

# Per-request item: one description. The batcher gathers them into one
# `encode` call so T5 sees a real batch.

async def _process_batch(model: TextEncoderModel, items: list[str]) -> list[dict]:
    text_features, text_mask = model.encode(items)
    # Split the batch back into per-item dicts so the batcher hands each
    # caller their slice. Doing this on CPU avoids holding GPU memory
    # while httpx serialises responses.
    text_features = text_features.cpu()
    text_mask = text_mask.cpu()
    results = []
    for i in range(len(items)):
        # Slice keeps the time dimension as-is (batch was padded uniformly).
        results.append({
            "text_features": text_features[i : i + 1],
            "text_mask": text_mask[i : i + 1],
        })
    return results


# ---- FastAPI app -------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    common = cfg.CommonConfig.from_env(
        default_batch_max_size=32, default_batch_max_wait_ms=5,
    )
    model = TextEncoderModel(common.device, common.dtype, cfg.model_name("t5-base"))

    async def process_fn(items: list[str]) -> list[dict]:
        return await _process_batch(model, items)

    batcher: AsyncBatcher[str, dict] = AsyncBatcher(
        max_batch_size=common.batch_max_size,
        max_wait_ms=common.batch_max_wait_ms,
        process_fn=process_fn,
        name="text-encoder",
    )
    await batcher.start()
    app.state.batcher = batcher
    LOG.info("text-encoder ready on %s (dtype=%s)", common.device, common.dtype)
    try:
        yield
    finally:
        await batcher.stop()


app = FastAPI(lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True}


@app.post("/encode")
async def encode(request: Request):
    payload = await read_payload(request)
    descriptions = payload.get("descriptions")
    if not isinstance(descriptions, list) or not all(isinstance(d, str) for d in descriptions):
        return write_payload({"error": "descriptions must be list[str]"})

    batcher: AsyncBatcher[str, dict] = request.app.state.batcher

    # Submit each description individually so the batcher can mix items
    # from different incoming HTTP requests. asyncio.gather preserves order.
    import asyncio
    per_item = await asyncio.gather(*(batcher.submit(d) for d in descriptions))

    # Stitch per-item results back into a single batched payload. Need to
    # pad to the max length since T5's batched encode would have done so.
    max_T = max(item["text_features"].size(1) for item in per_item)
    B = len(per_item)
    D = per_item[0]["text_features"].size(-1)
    text_features = torch.zeros(B, max_T, D, dtype=per_item[0]["text_features"].dtype)
    text_mask = torch.zeros(B, max_T, dtype=torch.bool)
    for i, item in enumerate(per_item):
        T = item["text_features"].size(1)
        text_features[i, :T] = item["text_features"][0]
        text_mask[i, :T] = item["text_mask"][0]

    return write_payload({"text_features": text_features, "text_mask": text_mask})
