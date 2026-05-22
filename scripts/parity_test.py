"""Parity smoke test: original SAMAudio.separate() vs the microservice stack.

Picks one audio clip + description, runs both paths with a pinned noise
tensor (so the only allowed sources of divergence are dtype and CUDA
kernel non-determinism), then compares the produced target waveforms.

This is a *correctness smoke test*, not a bit-identical check. Expected
deltas:

* The DiT service runs in bf16 by default; the original SAMAudio runs in
  the dtype the user picks. We force ``cosine similarity > 0.99`` and
  ``MAE < 0.02`` rather than ``allclose`` because bf16 + fp32 fan-in
  inside the ODE solver makes exact-bitwise unreasonable.

* T5 features differ between bf16 and fp32 enough to shift the ODE
  trajectory slightly. To pin this we run the original path with bf16
  text features too, by casting the text encoder output before passing
  it to ``separate``. If you'd rather compare against a fully-fp32
  baseline, run the test with SAMP_DTYPE=float32 on the services.

Usage::

    python scripts/parity_test.py \\
        --model facebook/sam-audio-large \\
        --audio /path/to/test.wav \\
        --description "thunder" \\
        --dit-url http://localhost:18003

The test assumes the microservices are running with default
``2-GPU`` layout and that the original ``SAMAudio`` model fits on the
GPU it can find (``cuda:0`` is default — set ``CUDA_VISIBLE_DEVICES``
to pick a different one).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

import torch
import torchaudio

# Ensure the pipeline package is importable when this is run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.common.http import ServiceClient  # noqa: E402


SR = 48_000


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def _mae(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().mean())


def _load(path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav.mean(0)  # (S,) mono


async def _run_microservice_path(
    audio_url: str,
    text_url: str,
    dit_url: str,
    wav: torch.Tensor,
    description: str,
    noise: torch.Tensor,
) -> torch.Tensor:
    """Run the same clip through the microservice pipeline with pinned noise.

    Returns the decoded target waveform (n_cand=1, so just the single target).
    """
    text = ServiceClient(text_url)
    audio = ServiceClient(audio_url)
    dit = ServiceClient(dit_url)
    try:
        text_resp = await text.call("/encode", {"descriptions": [description]})
        audio_resp = await audio.call("/encode", {
            "wavs": wav.view(1, 1, -1),
            "wav_sizes": torch.tensor([wav.size(-1)], dtype=torch.long),
        })

        dit_resp = await dit.call("/denoise", {
            "audio_features": audio_resp["audio_features"],
            "feature_sizes": audio_resp["feature_sizes"],
            "text_features": text_resp["text_features"],
            "text_mask": text_resp["text_mask"],
            "n_candidates": 1,
            "noise": noise,  # pinned, bypasses batcher
        })
        latents = dit_resp["latents"]  # (2, C, T) — target,resid
        feature_sizes = dit_resp["feature_sizes"]
        fs_for_decode = feature_sizes.repeat_interleave(2)

        decode_resp = await audio.call("/decode", {
            "latents": latents,
            "feature_sizes": fs_for_decode,
        })
        wavs = decode_resp["wavs"]
        wav_sizes = decode_resp["wav_sizes"]
        # Target is the first row.
        return wavs[0, : int(wav_sizes[0].item())]
    finally:
        await text.close()
        await audio.close()
        await dit.close()


def _run_original_path(
    model_id: str,
    wav: torch.Tensor,
    description: str,
    noise: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Run SAMAudio.separate() directly, with the same pinned noise."""
    from sam_audio import SAMAudio, SAMAudioProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SAMAudio.from_pretrained(model_id).to(device=device, dtype=dtype).eval()
    processor = SAMAudioProcessor.from_pretrained(model_id)

    batch = processor(audios=[wav], descriptions=[description]).to(device)

    with torch.inference_mode():
        result = model.separate(
            batch,
            noise=noise.to(device=device, dtype=dtype),
            predict_spans=False,
            reranking_candidates=1,
        )
    # `result.target` is a list of tensors, one per item in the batch.
    return result.target[0].cpu()


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="facebook/sam-audio-large")
    p.add_argument("--audio", required=True, help="Path to test audio file")
    p.add_argument("--description", required=True)
    p.add_argument("--text-url", default=os.environ.get(
        "SAMP_URL_TEXT_ENCODER", "http://localhost:18001"))
    p.add_argument("--audio-url", default=os.environ.get(
        "SAMP_URL_AUDIO_CODEC", "http://localhost:18002"))
    p.add_argument("--dit-url", default=os.environ.get(
        "SAMP_URL_DIT_DENOISER", "http://localhost:18003"))
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float32", "float16"])
    p.add_argument("--cos-threshold", type=float, default=0.99)
    p.add_argument("--mae-threshold", type=float, default=0.02)
    args = p.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}[args.dtype]
    wav = _load(args.audio)
    print(f"[parity] loaded {args.audio}: {wav.size(-1)} samples ({wav.size(-1)/SR:.2f}s)")

    # Pre-compute the noise we'll pin into both paths. Shape must be
    # (B*n_cand=1, T, 2*C). C is the DACVAE codebook_dim, 128 for sam-audio-large.
    # T is ceil(S / hop_length); hop_length = 2*8*10*12 = 1920.
    hop = 2 * 8 * 10 * 12
    T = (wav.size(-1) + hop - 1) // hop
    C = 128
    torch.manual_seed(42)
    noise = torch.randn(1, T, 2 * C, dtype=dtype)
    print(f"[parity] noise shape={tuple(noise.shape)} dtype={dtype}")

    # Run original path.
    print("[parity] running original SAMAudio.separate()...")
    t0 = time.perf_counter()
    orig = _run_original_path(args.model, wav, args.description, noise, dtype)
    print(f"[parity]   done in {time.perf_counter() - t0:.2f}s, shape={tuple(orig.shape)}")

    # Free the original-path GPU memory before talking to the services
    # (which themselves hold their own GPU allocations on a different card,
    # but it's good hygiene).
    torch.cuda.empty_cache()

    # Run microservice path.
    print("[parity] running microservice pipeline...")
    t0 = time.perf_counter()
    micro = await _run_microservice_path(
        args.audio_url, args.text_url, args.dit_url,
        wav, args.description, noise,
    )
    print(f"[parity]   done in {time.perf_counter() - t0:.2f}s, shape={tuple(micro.shape)}")

    # Lengths can differ by one sample due to hop-length rounding. Trim.
    n = min(orig.size(-1), micro.size(-1))
    orig = orig[..., :n]
    micro = micro[..., :n]

    cos = _cos(orig, micro)
    mae = _mae(orig, micro)
    print(f"[parity] cosine similarity = {cos:.6f}")
    print(f"[parity] mean abs error    = {mae:.6f}")

    ok = cos >= args.cos_threshold and mae <= args.mae_threshold
    if ok:
        print(f"[parity] PASS (cos >= {args.cos_threshold}, mae <= {args.mae_threshold})")
        return 0
    print(f"[parity] FAIL (thresholds: cos >= {args.cos_threshold}, mae <= {args.mae_threshold})")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
