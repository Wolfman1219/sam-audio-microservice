"""Audio-codec microservice (DACVAE encode + decode).

Loads the full ``SAMAudio`` model and immediately discards every submodule
except the DACVAE, freeing the T5 / vision / DiT / ranker weights. This is
wasteful at startup but simple and reliable; the alternative (custom
partial loaders) is more code without much steady-state benefit.

Endpoints:

* ``POST /encode``  – takes ``{wavs: Tensor[B, 1, S], wav_sizes: Tensor[B]}``
                       and returns ``{audio_features: Tensor[B, T, C],
                       feature_sizes: Tensor[B]}``. C is ``codec.codebook_dim``
                       (128 for sam-audio-large). The DiT service is the one
                       that doubles to 2C; the codec stays generic.

* ``POST /decode``  – takes ``{latents: Tensor[N, C, T]}`` and returns
                       ``{wavs: Tensor[N, S]}``. The caller is responsible
                       for the target+residual reshape; the codec just
                       decodes whatever you hand it.

DACVAE runs with ``cudnn`` disabled internally and expects fp32. We
respect that here rather than fighting it.
"""

from __future__ import annotations

import gc
import logging
from contextlib import asynccontextmanager

# Patch sam_audio.BaseModel for newer huggingface_hub versions. Must
# precede any sam_audio import that triggers a model load.
from pipeline.common import hf_compat  # noqa: F401

import torch
from fastapi import FastAPI, Request

from sam_audio.model.model import SAMAudio

from pipeline.common import config as cfg
from pipeline.common.batching import AsyncBatcher
from pipeline.common.http import read_payload, write_payload

LOG = logging.getLogger("audio_codec")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


class AudioCodecModel:
    """Wraps DACVAE encode + decode lifted out of a loaded SAMAudio."""

    def __init__(self, device: torch.device, model_id: str):
        LOG.info("loading SAMAudio checkpoint to extract DACVAE: %s", model_id)
        # Load to CPU first so we don't peak-allocate the unused submodules
        # on the target GPU.
        model = SAMAudio.from_pretrained(model_id, map_location="cpu", strict=False)
        codec = model.audio_codec

        # Drop everything we don't need before moving to GPU.
        del model
        gc.collect()
        torch.cuda.empty_cache()

        codec = codec.to(device=device).eval()
        # DACVAE has its own dtype expectations; keep fp32 weights.
        self.codec = codec
        self.device = device
        self.sample_rate = codec.sample_rate
        self.hop_length = codec.hop_length
        LOG.info(
            "DACVAE on %s, sample_rate=%d, hop_length=%d",
            device, self.sample_rate, self.hop_length,
        )

    @torch.inference_mode()
    def encode(self, wavs: torch.Tensor) -> torch.Tensor:
        # wavs: (B, 1, S) fp32 in [-1, 1]. Returns (B, T, C).
        wavs = wavs.to(self.device, dtype=torch.float32, non_blocking=True)
        features = self.codec(wavs)  # (B, C, T)
        return features.transpose(1, 2).contiguous()

    @torch.inference_mode()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        # latents: (N, C, T). Returns (N, S).
        latents = latents.to(self.device, dtype=torch.float32, non_blocking=True)
        wavs = self.codec.decode(latents)  # (N, 1, S)
        return wavs.squeeze(1).contiguous()


# Encode and decode use very different batch shapes, so they have separate
# batchers. Encode batches one (wav, size) item per request; decode batches
# one (latent,) item per request.

async def _process_encode_batch(
    model: AudioCodecModel, items: list[dict]
) -> list[dict]:
    # Pad waveforms to longest in the batch.
    max_S = max(item["wav"].size(-1) for item in items)
    B = len(items)
    wavs = torch.zeros(B, 1, max_S, dtype=torch.float32)
    wav_sizes = torch.zeros(B, dtype=torch.long)
    for i, item in enumerate(items):
        S = item["wav"].size(-1)
        wavs[i, 0, :S] = item["wav"].view(-1).float()
        wav_sizes[i] = S

    features = model.encode(wavs)  # (B, T_max, C)
    feature_sizes = (wav_sizes.float() / model.hop_length).ceil().long()

    features_cpu = features.cpu()
    out = []
    for i in range(B):
        T = int(feature_sizes[i].item())
        out.append({
            "audio_features": features_cpu[i : i + 1, :T],  # trim padding
            "feature_size": int(feature_sizes[i].item()),
            "wav_size": int(wav_sizes[i].item()),
        })
    return out


async def _process_decode_batch(
    model: AudioCodecModel, items: list[torch.Tensor]
) -> list[dict]:
    # Each item is (C, T) for one latent.
    max_T = max(item.size(-1) for item in items)
    C = items[0].size(0)
    N = len(items)
    batch = torch.zeros(N, C, max_T, dtype=torch.float32)
    lengths_T = torch.zeros(N, dtype=torch.long)
    for i, lat in enumerate(items):
        T = lat.size(-1)
        batch[i, :, :T] = lat.float()
        lengths_T[i] = T

    wavs = model.decode(batch)  # (N, S_max)
    wavs_cpu = wavs.cpu()

    # Crop to the actual sample length corresponding to T frames.
    out = []
    for i in range(N):
        S = int(lengths_T[i].item()) * model.hop_length
        out.append({"wav": wavs_cpu[i : i + 1, :S]})
    return out


@asynccontextmanager
async def lifespan(app: FastAPI):
    common = cfg.CommonConfig.from_env(
        default_batch_max_size=8, default_batch_max_wait_ms=10,
    )
    model = AudioCodecModel(common.device, cfg.model_id())

    encode_batcher = AsyncBatcher(
        max_batch_size=common.batch_max_size,
        max_wait_ms=common.batch_max_wait_ms,
        process_fn=lambda items: _process_encode_batch(model, items),
        name="codec-encode",
    )
    decode_batcher = AsyncBatcher(
        max_batch_size=common.batch_max_size,
        max_wait_ms=common.batch_max_wait_ms,
        process_fn=lambda items: _process_decode_batch(model, items),
        name="codec-decode",
    )
    await encode_batcher.start()
    await decode_batcher.start()
    app.state.encode_batcher = encode_batcher
    app.state.decode_batcher = decode_batcher
    app.state.hop_length = model.hop_length
    app.state.sample_rate = model.sample_rate
    LOG.info("audio-codec ready on %s", common.device)
    try:
        yield
    finally:
        await encode_batcher.stop()
        await decode_batcher.stop()


app = FastAPI(lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True}


@app.get("/info")
async def info(request: Request) -> dict:
    """Static metadata clients need to do feature/frame arithmetic locally."""
    return {
        "sample_rate": request.app.state.sample_rate,
        "hop_length": request.app.state.hop_length,
    }


@app.post("/encode")
async def encode(request: Request):
    """Encode waveforms to DACVAE latents.

    Request: ``{"wavs": Tensor[B, 1, S], "wav_sizes": Tensor[B] (int64)}``
    Response: ``{"audio_features": Tensor[B, T_max, C], "feature_sizes": Tensor[B]}``
    """
    payload = await read_payload(request)
    wavs = payload["wavs"]
    wav_sizes = payload["wav_sizes"]
    if wavs.ndim != 3 or wavs.size(1) != 1:
        return write_payload({"error": "wavs must have shape (B, 1, S) (mono)"})
    B = wavs.size(0)

    import asyncio
    per_item = await asyncio.gather(*(
        request.app.state.encode_batcher.submit({
            "wav": wavs[i, 0, : int(wav_sizes[i].item())]
        })
        for i in range(B)
    ))

    max_T = max(item["feature_size"] for item in per_item)
    C = per_item[0]["audio_features"].size(-1)
    features = torch.zeros(B, max_T, C, dtype=per_item[0]["audio_features"].dtype)
    feature_sizes = torch.zeros(B, dtype=torch.long)
    for i, item in enumerate(per_item):
        T = item["feature_size"]
        features[i, :T] = item["audio_features"][0]
        feature_sizes[i] = T

    return write_payload({
        "audio_features": features,
        "feature_sizes": feature_sizes,
    })


@app.post("/decode")
async def decode(request: Request):
    """Decode DACVAE latents to waveforms.

    Request: ``{"latents": Tensor[N, C, T_max], "feature_sizes": Tensor[N] (int64)}``
    Response: ``{"wavs": Tensor[N, S_max], "wav_sizes": Tensor[N]}``
    """
    payload = await read_payload(request)
    latents = payload["latents"]
    feature_sizes = payload["feature_sizes"]
    if latents.ndim != 3:
        return write_payload({"error": "latents must have shape (N, C, T)"})
    N = latents.size(0)

    import asyncio
    per_item = await asyncio.gather(*(
        request.app.state.decode_batcher.submit(
            latents[i, :, : int(feature_sizes[i].item())]
        )
        for i in range(N)
    ))

    max_S = max(item["wav"].size(-1) for item in per_item)
    wavs = torch.zeros(N, max_S, dtype=per_item[0]["wav"].dtype)
    wav_sizes = torch.zeros(N, dtype=torch.long)
    for i, item in enumerate(per_item):
        S = item["wav"].size(-1)
        wavs[i, :S] = item["wav"][0]
        wav_sizes[i] = S

    return write_payload({"wavs": wavs, "wav_sizes": wav_sizes})
