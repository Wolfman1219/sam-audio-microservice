#!/usr/bin/env python3
"""Quick check: every service responds 200 on /healthz.

Defaults point at the host ports exposed by docker-compose
(18001..18005). Override with SAMP_URL_<SERVICE> env vars if you've
rewired the topology.
"""

from __future__ import annotations

import asyncio
import os
import sys

import httpx

SERVICES = {
    "text-encoder": "http://localhost:18001",
    "audio-codec": "http://localhost:18002",
    "dit-denoiser": "http://localhost:18003",
    "judge-ranker": "http://localhost:18004",
    "clap-ranker": "http://localhost:18005",
}


def _url(name: str, default: str) -> str:
    return os.environ.get(f"SAMP_URL_{name.upper().replace('-', '_')}", default)


async def _check(name: str, base_url: str) -> tuple[str, bool, str]:
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            r = await client.get(base_url + "/healthz")
            return name, r.status_code == 200, f"HTTP {r.status_code}"
        except httpx.HTTPError as e:
            return name, False, str(e)


async def main() -> int:
    results = await asyncio.gather(*(
        _check(name, _url(name, default)) for name, default in SERVICES.items()
    ))
    all_ok = True
    for name, ok, msg in results:
        mark = "OK " if ok else "FAIL"
        print(f"  [{mark}] {name:<14} {msg}")
        all_ok &= ok
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
