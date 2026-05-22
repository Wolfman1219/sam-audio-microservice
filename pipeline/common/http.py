"""HTTP helpers: a generic FastAPI endpoint shape and a thin async client.

The wire format is uniform across services:

* Request body  : ``application/octet-stream``, a ``torch.save``-ed dict.
* Response body : same.

Each service exposes ``POST /<verb>`` where ``<verb>`` is e.g. ``encode``,
``denoise``, ``score``. There's no JSON layer because every payload is
tensor-heavy and JSON would force base64 round-trips for no benefit.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from fastapi import HTTPException, Request, Response

from .serialization import decode_payload, encode_payload

LOG = logging.getLogger(__name__)

OCTET = "application/octet-stream"


async def read_payload(request: Request) -> dict[str, Any]:
    """Read and decode the request body. Raises 400 on malformed input."""
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="empty request body")
    try:
        return decode_payload(body, device="cpu")
    except Exception as exc:  # noqa: BLE001
        LOG.warning("failed to decode payload: %s", exc)
        raise HTTPException(status_code=400, detail=f"bad payload: {exc}") from exc


def write_payload(payload: dict[str, Any]) -> Response:
    return Response(content=encode_payload(payload), media_type=OCTET)


class ServiceClient:
    """Async HTTP client targeting one peer service.

    One ``ServiceClient`` per peer, shared by the whole orchestrator process.
    httpx keeps a connection pool, so repeated calls reuse TCP sockets.
    """

    def __init__(self, base_url: str, *, timeout_s: float = 600.0):
        # Long timeout because DiT denoise on a 30 s clip with 8 candidates
        # plus batching can take several seconds. The cap is mostly a
        # backstop against hung peers.
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout_s, connect=5.0),
            limits=httpx.Limits(
                max_connections=64, max_keepalive_connections=32
            ),
        )
        self.base_url = base_url

    async def close(self) -> None:
        await self._client.aclose()

    async def call(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = encode_payload(payload)
        resp = await self._client.post(
            path,
            content=body,
            headers={"content-type": OCTET, "accept": OCTET},
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"{self.base_url}{path} returned {resp.status_code}: "
                f"{resp.text[:512]}"
            )
        return decode_payload(resp.content, device="cpu")

    async def healthz(self) -> bool:
        try:
            resp = await self._client.get("/healthz", timeout=5.0)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def __aenter__(self) -> "ServiceClient":
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.close()
