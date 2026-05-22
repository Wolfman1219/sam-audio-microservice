"""Dataset-cleaning driver.

Reads a manifest (TSV of ``audio_path<TAB>description``), runs the
separation pipeline for each row, and writes target waveforms to an
output directory.

Pipeline per item::

    text-encoder  ┐
                  ├─► dit-denoiser ─► audio-codec /decode ─► ranker fan-out ─► best
    audio-codec  ┘                                            (judge + clap)

The Judge and CLAP rankers are called in parallel and their scores are
combined with configurable weights. Defaults (0.5 / 0.5) are a starting
point — tune to your dataset.

Usage::

    python -m pipeline.orchestrator.separator \\
        --manifest /data/clean_me.tsv \\
        --output-dir /data/separated \\
        --candidates 8 \\
        --concurrency 4
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torchaudio

from pipeline.common.http import ServiceClient

LOG = logging.getLogger("orchestrator")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


def _url(service: str, default: str) -> str:
    return os.environ.get(f"SAMP_URL_{service.upper().replace('-', '_')}", default)


@dataclass
class PipelineClients:
    text_encoder: ServiceClient
    audio_codec: ServiceClient
    dit_denoiser: ServiceClient
    judge_ranker: Optional[ServiceClient]
    clap_ranker: Optional[ServiceClient]

    @classmethod
    def from_env(cls, *, use_judge: bool = True, use_clap: bool = True) -> "PipelineClients":
        return cls(
            text_encoder=ServiceClient(_url("text-encoder", "http://text-encoder:8000")),
            audio_codec=ServiceClient(_url("audio-codec", "http://audio-codec:8000")),
            dit_denoiser=ServiceClient(_url("dit-denoiser", "http://dit-denoiser:8000")),
            judge_ranker=ServiceClient(_url("judge-ranker", "http://judge-ranker:8000")) if use_judge else None,
            clap_ranker=ServiceClient(_url("clap-ranker", "http://clap-ranker:8000")) if use_clap else None,
        )

    async def close(self) -> None:
        clients = [self.text_encoder, self.audio_codec, self.dit_denoiser,
                   self.judge_ranker, self.clap_ranker]
        await asyncio.gather(*(c.close() for c in clients if c is not None))


async def _wait_healthy(clients: PipelineClients, *, timeout_s: float = 600) -> None:
    deadline = time.monotonic() + timeout_s
    targets = [(name, c) for name, c in [
        ("text-encoder", clients.text_encoder),
        ("audio-codec", clients.audio_codec),
        ("dit-denoiser", clients.dit_denoiser),
        ("judge-ranker", clients.judge_ranker),
        ("clap-ranker", clients.clap_ranker),
    ] if c is not None]

    while time.monotonic() < deadline:
        ok = await asyncio.gather(*(c.healthz() for _, c in targets))
        if all(ok):
            LOG.info("all services healthy")
            return
        not_ready = [name for (name, _), o in zip(targets, ok) if not o]
        LOG.info("waiting on: %s", ", ".join(not_ready))
        await asyncio.sleep(2.0)
    raise RuntimeError("services did not become healthy in time")


def _load_mono_48k(path: str, target_sr: int = 48_000) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.mean(0)  # (S,) mono


def _normalize_min_max(scores: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Map scores to [0, 1] within the candidate set.

    Judge and CLAP score on incompatible scales (Judge ≈ MOS 1–5, CLAP
    is dot-product similarity). Without normalisation the heavier-scaled
    ranker dominates any weighted sum. Min-max within the candidates is
    the simplest fix; rank-based fusion is the obvious next iteration.
    """
    if scores.numel() <= 1:
        return torch.zeros_like(scores)
    lo, hi = scores.min(), scores.max()
    return (scores - lo) / (hi - lo + eps)


async def _separate_one(
    clients: PipelineClients,
    audio_path: str,
    description: str,
    n_candidates: int,
    *,
    sample_rate: int = 48_000,
    judge_weight: float = 0.5,
    clap_weight: float = 0.5,
) -> tuple[torch.Tensor, int, torch.Tensor, Optional[torch.Tensor]]:
    """Run one item end-to-end.

    Returns ``(best_target, sample_rate, all_target_candidates, combined_scores)``.
    ``combined_scores`` is ``None`` when no ranker was consulted (n_candidates=1).
    """
    wav = _load_mono_48k(audio_path, sample_rate)  # (S,)
    wav_size = wav.size(-1)

    # 1. Encode text and audio in parallel.
    text_task = clients.text_encoder.call(
        "/encode", {"descriptions": [description]}
    )
    audio_task = clients.audio_codec.call(
        "/encode",
        {
            "wavs": wav.view(1, 1, -1),
            "wav_sizes": torch.tensor([wav_size], dtype=torch.long),
        },
    )
    text_resp, audio_resp = await asyncio.gather(text_task, audio_task)

    # 2. Denoise (ODE loop runs inside the DiT service).
    dit_resp = await clients.dit_denoiser.call("/denoise", {
        "audio_features": audio_resp["audio_features"],
        "feature_sizes": audio_resp["feature_sizes"],
        "text_features": text_resp["text_features"],
        "text_mask": text_resp["text_mask"],
        "n_candidates": n_candidates,
    })
    latents = dit_resp["latents"]               # (n_cand*2, C, T)
    feature_sizes = dit_resp["feature_sizes"]   # (n_cand,)

    # 3. Decode. latents has target+residual interleaved per candidate.
    fs_for_decode = feature_sizes.repeat_interleave(2)  # (n_cand*2,)
    decode_resp = await clients.audio_codec.call("/decode", {
        "latents": latents,
        "feature_sizes": fs_for_decode,
    })
    wavs = decode_resp["wavs"]            # (n_cand*2, S_max)
    wav_sizes = decode_resp["wav_sizes"]  # (n_cand*2,)

    # Split target (even rows) and residual (odd rows).
    targets = wavs[0::2]
    target_sizes = wav_sizes[0::2]

    # 4. Ranker fan-out. Skip entirely when only one candidate exists.
    combined: Optional[torch.Tensor] = None
    if n_candidates > 1 and (clients.judge_ranker or clients.clap_ranker):
        ranker_tasks = []
        if clients.judge_ranker is not None:
            ranker_tasks.append(("judge", clients.judge_ranker.call("/score", {
                "extracted_audio": targets,
                "extracted_wav_sizes": target_sizes,
                "input_audio": wav,
                "input_wav_size": torch.tensor(wav_size, dtype=torch.long),
                "description": description,
                "n_candidates": n_candidates,
                "sample_rate": sample_rate,
            })))
        if clients.clap_ranker is not None:
            ranker_tasks.append(("clap", clients.clap_ranker.call("/score", {
                "extracted_audio": targets,
                "extracted_wav_sizes": target_sizes,
                "description": description,
                "n_candidates": n_candidates,
                "sample_rate": sample_rate,
            })))

        results = await asyncio.gather(*(t for _, t in ranker_tasks))
        score_map = dict(zip((name for name, _ in ranker_tasks), results))

        # Min-max normalise each ranker so weights mean what they look like.
        norm_scores = {
            name: _normalize_min_max(r["scores"]) for name, r in score_map.items()
        }
        if "judge" in norm_scores and "clap" in norm_scores:
            combined = judge_weight * norm_scores["judge"] + clap_weight * norm_scores["clap"]
        elif "judge" in norm_scores:
            combined = norm_scores["judge"]
        elif "clap" in norm_scores:
            combined = norm_scores["clap"]

        best_idx = int(combined.argmax().item())
    else:
        best_idx = 0

    best_target = targets[best_idx, : int(target_sizes[best_idx].item())]
    return best_target, sample_rate, targets, combined


async def run_manifest(
    manifest_path: Path,
    output_dir: Path,
    *,
    n_candidates: int,
    concurrency: int,
    judge_weight: float,
    clap_weight: float,
    use_judge: bool,
    use_clap: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    clients = PipelineClients.from_env(use_judge=use_judge, use_clap=use_clap)
    try:
        await _wait_healthy(clients)

        with manifest_path.open() as f:
            reader = csv.reader(f, delimiter="\t")
            rows = [(audio_path, description) for audio_path, description in reader]

        sem = asyncio.Semaphore(concurrency)

        async def process(row_idx: int, audio_path: str, description: str) -> None:
            async with sem:
                try:
                    t0 = time.perf_counter()
                    target, sr, _candidates, _scores = await _separate_one(
                        clients, audio_path, description, n_candidates,
                        judge_weight=judge_weight, clap_weight=clap_weight,
                    )
                    out_path = output_dir / f"{Path(audio_path).stem}.target.wav"
                    torchaudio.save(str(out_path), target.unsqueeze(0), sr)
                    dt = time.perf_counter() - t0
                    LOG.info("[%d] %s -> %s (%.2fs)", row_idx, audio_path, out_path, dt)
                except Exception:  # noqa: BLE001
                    LOG.exception("[%d] failed on %s", row_idx, audio_path)

        await asyncio.gather(*(
            process(i, ap, desc) for i, (ap, desc) in enumerate(rows)
        ))
    finally:
        await clients.close()


def _cli() -> None:
    p = argparse.ArgumentParser(description="SAM-Audio pipeline orchestrator")
    p.add_argument("--manifest", type=Path, required=True,
                   help="TSV file: audio_path<TAB>description per line")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--candidates", type=int, default=8,
                   help="Re-ranking candidates per item")
    p.add_argument("--concurrency", type=int, default=4,
                   help="Inflight items at once; ~2x DiT replicas is a good start")
    p.add_argument("--judge-weight", type=float, default=0.5)
    p.add_argument("--clap-weight", type=float, default=0.5)
    p.add_argument("--no-judge", action="store_true",
                   help="Skip the Judge ranker (e.g. if it isn't deployed)")
    p.add_argument("--no-clap", action="store_true",
                   help="Skip the CLAP ranker")
    args = p.parse_args()
    asyncio.run(run_manifest(
        args.manifest, args.output_dir,
        n_candidates=args.candidates,
        concurrency=args.concurrency,
        judge_weight=args.judge_weight,
        clap_weight=args.clap_weight,
        use_judge=not args.no_judge,
        use_clap=not args.no_clap,
    ))


if __name__ == "__main__":
    _cli()
