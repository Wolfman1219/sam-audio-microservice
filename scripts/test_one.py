#!/usr/bin/env python3
"""Single-file smoke test of the running pipeline.

Drives one audio clip + description through every service and prints
per-stage timings. Saves:

* ``{stem}.target.wav``     — best-scoring target (the deliverable)
* ``{stem}.residual.wav``   — everything else  (``--keep-residual``)
* ``{stem}.cand_N.wav``     — all target candidates  (``--keep-candidates``)
* ``{stem}.scores.json``    — per-candidate Judge + CLAP + combined scores

Usage::

    python scripts/test_one.py \\
        --audio /path/to/clip.wav \\
        --description "thunder" \\
        --output-dir ./out \\
        --candidates 8

First-time tip: start with ``--candidates 1`` (skips rerankers entirely,
~5× faster) to confirm the pipeline produces sensible audio. Then bump
to ``--candidates 8`` once you trust it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torchaudio

# Make the pipeline package importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.common.http import ServiceClient  # noqa: E402

LOG = logging.getLogger("test_one")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SR = 48_000


def _u(name: str, default: str) -> str:
    return os.environ.get(f"SAMP_URL_{name.upper().replace('-', '_')}", default)


def _load_mono_48k(path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav.mean(0)


def _norm(scores: torch.Tensor) -> torch.Tensor:
    if scores.numel() <= 1:
        return torch.zeros_like(scores)
    lo, hi = scores.min(), scores.max()
    return (scores - lo) / (hi - lo + 1e-8)


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--audio", required=True, help="Path to input audio (any sr; we resample to 48k mono)")
    p.add_argument("--description", required=True, help='Lowercase NP/VP, e.g. "thunder", "man speaking"')
    p.add_argument("--output-dir", type=Path, default=Path("./out"))
    p.add_argument("--candidates", type=int, default=8)
    p.add_argument("--judge-weight", type=float, default=0.5)
    p.add_argument("--clap-weight", type=float, default=0.5)
    p.add_argument("--keep-residual", action="store_true",
                   help="Also save the residual (everything except the target)")
    p.add_argument("--keep-candidates", action="store_true",
                   help="Save every candidate, not just the best")
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--no-clap", action="store_true")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.audio).stem

    text = ServiceClient(_u("text-encoder", "http://localhost:18001"))
    audio = ServiceClient(_u("audio-codec", "http://localhost:18002"))
    dit = ServiceClient(_u("dit-denoiser", "http://localhost:18003"))
    judge = ServiceClient(_u("judge-ranker", "http://localhost:18004")) if not args.no_judge else None
    clap = ServiceClient(_u("clap-ranker", "http://localhost:18005")) if not args.no_clap else None

    try:
        wav = _load_mono_48k(args.audio)
        dur = wav.size(-1) / SR
        LOG.info("loaded %s: %.2fs (%d samples @ %d Hz)", args.audio, dur, wav.size(-1), SR)
        LOG.info("description: %r", args.description)
        LOG.info("candidates:  %d", args.candidates)

        # --- 1. Text + audio encoding in parallel -------------------------
        t = time.perf_counter()
        text_task = text.call("/encode", {"descriptions": [args.description]})
        audio_task = audio.call("/encode", {
            "wavs": wav.view(1, 1, -1),
            "wav_sizes": torch.tensor([wav.size(-1)], dtype=torch.long),
        })
        text_resp, audio_resp = await asyncio.gather(text_task, audio_task)
        LOG.info("[%6.2fs]  encoders done  (text+audio in parallel)", time.perf_counter() - t)
        LOG.info("            text_features=%s  audio_features=%s",
                 tuple(text_resp["text_features"].shape),
                 tuple(audio_resp["audio_features"].shape))

        # --- 2. Denoise ---------------------------------------------------
        t = time.perf_counter()
        dit_resp = await dit.call("/denoise", {
            "audio_features": audio_resp["audio_features"],
            "feature_sizes": audio_resp["feature_sizes"],
            "text_features": text_resp["text_features"],
            "text_mask": text_resp["text_mask"],
            "n_candidates": args.candidates,
        })
        latents = dit_resp["latents"]
        feature_sizes = dit_resp["feature_sizes"]
        LOG.info("[%6.2fs]  dit denoise    (latents=%s)",
                 time.perf_counter() - t, tuple(latents.shape))

        # --- 3. Decode ----------------------------------------------------
        t = time.perf_counter()
        fs_for_decode = feature_sizes.repeat_interleave(2)
        decode_resp = await audio.call("/decode", {
            "latents": latents,
            "feature_sizes": fs_for_decode,
        })
        wavs = decode_resp["wavs"]              # (n_cand*2, S)
        wav_sizes = decode_resp["wav_sizes"]
        targets = wavs[0::2]
        residuals = wavs[1::2]
        target_sizes = wav_sizes[0::2]
        residual_sizes = wav_sizes[1::2]
        LOG.info("[%6.2fs]  decode         (wavs=%s)",
                 time.perf_counter() - t, tuple(wavs.shape))

        # --- 4. Score (skipped when candidates==1) ------------------------
        scores_blob: dict = {"n_candidates": args.candidates, "description": args.description}
        best_idx = 0

        if args.candidates > 1 and (judge or clap):
            t = time.perf_counter()
            tasks = []
            if judge is not None:
                tasks.append(("judge", judge.call("/score", {
                    "extracted_audio": targets,
                    "extracted_wav_sizes": target_sizes,
                    "input_audio": wav,
                    "input_wav_size": torch.tensor(wav.size(-1), dtype=torch.long),
                    "description": args.description,
                    "n_candidates": args.candidates,
                    "sample_rate": SR,
                })))
            if clap is not None:
                tasks.append(("clap", clap.call("/score", {
                    "extracted_audio": targets,
                    "extracted_wav_sizes": target_sizes,
                    "description": args.description,
                    "n_candidates": args.candidates,
                    "sample_rate": SR,
                })))
            results = await asyncio.gather(*(c for _, c in tasks))
            score_map = dict(zip((n for n, _ in tasks), results))

            norm_scores = {k: _norm(v["scores"]) for k, v in score_map.items()}
            if "judge" in norm_scores and "clap" in norm_scores:
                combined = args.judge_weight * norm_scores["judge"] + args.clap_weight * norm_scores["clap"]
            else:
                combined = next(iter(norm_scores.values()))
            best_idx = int(combined.argmax().item())

            scores_blob.update({
                "judge_raw":      score_map["judge"]["scores"].tolist() if "judge" in score_map else None,
                "clap_raw":       score_map["clap"]["scores"].tolist()  if "clap"  in score_map else None,
                "judge_norm":     norm_scores.get("judge").tolist() if "judge" in norm_scores else None,
                "clap_norm":      norm_scores.get("clap").tolist()  if "clap"  in norm_scores else None,
                "combined":       combined.tolist(),
                "weights":        {"judge": args.judge_weight, "clap": args.clap_weight},
                "best_candidate": best_idx,
            })
            LOG.info("[%6.2fs]  rank           (best=%d, combined=%s)",
                     time.perf_counter() - t, best_idx,
                     [round(v, 3) for v in combined.tolist()])
        else:
            scores_blob["best_candidate"] = 0
            LOG.info("           ranking skipped (candidates=%d)", args.candidates)

        # --- 5. Save outputs ---------------------------------------------
        # Best target.
        target = targets[best_idx, : int(target_sizes[best_idx].item())].unsqueeze(0)
        target_path = args.output_dir / f"{stem}.target.wav"
        torchaudio.save(str(target_path), target, SR)
        LOG.info("saved  %s", target_path)

        if args.keep_residual:
            residual = residuals[best_idx, : int(residual_sizes[best_idx].item())].unsqueeze(0)
            r_path = args.output_dir / f"{stem}.residual.wav"
            torchaudio.save(str(r_path), residual, SR)
            LOG.info("saved  %s", r_path)

        if args.keep_candidates:
            for i in range(args.candidates):
                c = targets[i, : int(target_sizes[i].item())].unsqueeze(0)
                cp = args.output_dir / f"{stem}.cand_{i}.wav"
                torchaudio.save(str(cp), c, SR)
            LOG.info("saved  %d candidate wavs", args.candidates)

        scores_path = args.output_dir / f"{stem}.scores.json"
        with scores_path.open("w") as fp:
            json.dump(scores_blob, fp, indent=2)
        LOG.info("saved  %s", scores_path)

        return 0
    finally:
        await asyncio.gather(
            text.close(), audio.close(), dit.close(),
            *(c.close() for c in (judge, clap) if c is not None),
        )


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

