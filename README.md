# SAM-Audio Pipeline (Microservice Edition)

A throughput-optimised, dockerised decomposition of SAM-Audio for offline
dataset cleaning. The original `SAMAudio.separate(...)` loads every
sub-model onto one GPU and runs everything sequentially. This stack
splits the workload across services, pins each one to its own GPU
(or shares an L40 between the light ones), and pipelines requests so
the bottleneck (DiT) is rarely idle.

## Services

| Service          | Wraps                            | Endpoints              | GPU role         |
| ---------------- | -------------------------------- | ---------------------- | ---------------- |
| `text-encoder`   | `T5TextEncoder`                  | `POST /encode`         | shared (light)   |
| `audio-codec`    | `DACVAE` (encoder + decoder)     | `POST /encode`, `/decode` | shared (light) |
| `dit-denoiser`   | `SAMAudio` DiT + ODE loop        | `POST /denoise`        | **dedicated**    |
| `judge-ranker`   | `SAMAudioJudgeModel`             | `POST /score`          | shared (light)   |
| `clap-ranker`    | `ClapRanker` (LAION CLAP)        | `POST /score`          | shared (light)   |

The orchestrator is a plain Python script — it owns the dataflow,
batches items, fans out to ranker services in parallel, picks the
winner. It is not a service, so you embed it directly in your
dataset-cleaning loop.

## GPU layout (8× L40, plan for ≤4)

**Default (2 GPUs, recommended starting point):**

```
GPU 0 → dit-denoiser
GPU 1 → text-encoder + audio-codec + judge-ranker + clap-ranker
        (all light, ~6 GB combined on a 48 GB card)
```

**Throughput mode (4 GPUs):**

```
GPU 0 → dit-denoiser (replica 1)
GPU 1 → dit-denoiser (replica 2)
GPU 2 → text-encoder + audio-codec
GPU 3 → judge-ranker + clap-ranker
```

Switching between these is a `docker-compose.yml` edit, not a code
change. The orchestrator round-robins across DiT replicas (driven by
your reverse proxy or — easier — by setting `SAMP_URL_DIT_DENOISER` to
a comma-separated list once we wire that in).

## Why HTTP and not gRPC/Ray

- Each service is a separate image: independent dependencies, restart
  cycles, log streams. True container isolation, no shared Ray runtime.
- Communication is loopback HTTP between containers on the same compose
  network — sub-millisecond per hop.
- Tensor payloads (a few MB of bf16 features per 30 s clip) are
  serialised with `torch.save` to a `BytesIO` and sent as
  `application/octet-stream`. This costs ~5 ms per transfer and is
  hidden by request pipelining.
- Each service does **server-side async batching**: requests arriving
  within a small window are coalesced into one model forward pass.

## The hot path

The DiT runs the entire 32-step ODE loop **inside one `/denoise` RPC**.
Sending 32 individual requests over the network would destroy
throughput; sending one keeps the GPU saturated.

## Scoring

After decode, Judge and CLAP are called **in parallel** with the
candidate set. Their scores are min-max normalised within the candidate
set (so the two heterogeneous scales — Judge MOS-like 1–5 vs CLAP
cosine — combine sanely) and then weighted-summed. Defaults
`--judge-weight 0.5 --clap-weight 0.5`; tune empirically.

## Bringing the stack up

```bash
# 0. one-time setup: place the sam-audio source tree at ./sam_audio_src
ln -s /path/to/sam-audio sam_audio_src

# 1. set HF_TOKEN in .env (request access to the gated checkpoints first)
cp .env.example .env
$EDITOR .env

# 2. build the shared base image once (slow; CUDA + deps)
docker compose --profile build-only build base

# 3. build the per-service images (fast; layered on the base)
docker compose build

# 4. start the stack
docker compose up -d

# 5. wait for warm-up (dit-denoiser pulls the checkpoint on first boot)
python scripts/healthcheck.py

# 6. run separation
python -m pipeline.orchestrator.separator \
    --manifest /data/clean_me.tsv \
    --output-dir /data/separated \
    --candidates 8 \
    --concurrency 4

# 7. (optional) parity smoke test against the unmodified SAMAudio
python scripts/parity_test.py \
    --audio /data/test.wav \
    --description "thunder" \
    --model facebook/sam-audio-large
```

## Auth & weights

The DiT, Judge, and text-encoder containers pull from gated HF repos.
Mount an HF cache from the host (already wired via the `hf_cache` named
volume) and set `HF_TOKEN` in `.env`.

## Layout

```
sam-audio-pipeline/
├── docker-compose.yml
├── docker/
│   ├── base.Dockerfile       # CUDA + Python + sam_audio installed
│   └── service.Dockerfile    # Per-service: copies code, sets CMD
├── pipeline/
│   ├── common/               # serialization, async batching, config, http
│   ├── services/
│   │   ├── text_encoder/
│   │   ├── audio_codec/
│   │   ├── dit_denoiser/
│   │   ├── judge_ranker/
│   │   └── clap_ranker/
│   └── orchestrator/         # The dataset-cleaning driver
└── scripts/
    ├── healthcheck.py
    └── parity_test.py
```

## Tuning knobs

Every service reads `SAMP_*` env vars at startup (see
`pipeline/common/config.py`). The ones you'll touch most:

| Var                       | What it does                                            |
| ------------------------- | ------------------------------------------------------- |
| `SAMP_BATCH_MAX_SIZE`     | Hard cap on items per model forward pass.               |
| `SAMP_BATCH_MAX_WAIT_MS`  | How long to wait for siblings before flushing a batch.  |
| `SAMP_DTYPE`              | `bfloat16` / `float16` / `float32` for model weights.   |
| `SAMP_TORCH_COMPILE`      | DiT only — `1` to compile the transformer.              |
| `SAMP_DEVICE`             | The GPU device the service sees as `cuda:0`.            |
