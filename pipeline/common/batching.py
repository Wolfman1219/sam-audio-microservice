"""Async micro-batcher for FastAPI services.

Pattern: each service registers a single ``process_fn(items: list) -> list``
that runs one model forward pass over the given batch. The batcher coalesces
items arriving within a short time window (or up to a max batch size) so the
GPU sees real batches even when callers send one request at a time.

Two parameters tune the trade-off:

* ``max_batch_size``  — hard cap on items per forward pass.
* ``max_wait_ms``     — once the queue has at least one item, how long to
                        wait for siblings before flushing.

For DiT we use a longer wait because each forward is expensive enough that
gathering a wider batch wins; for the text encoder we use a tiny wait
because individual forwards are cheap and waiting hurts latency.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable, Generic, TypeVar

LOG = logging.getLogger(__name__)

T = TypeVar("T")  # request item type
R = TypeVar("R")  # response item type


class AsyncBatcher(Generic[T, R]):
    """Coalesces concurrent requests into one ``process_fn`` invocation.

    Usage::

        batcher = AsyncBatcher(
            max_batch_size=8, max_wait_ms=10, process_fn=run_model
        )
        await batcher.start()
        # ... in a request handler:
        result = await batcher.submit(item)

    ``process_fn(batch)`` must return a sequence the same length as
    ``batch``; items and results are matched positionally. Exceptions
    raised by ``process_fn`` are propagated to every submitter in the batch.
    """

    def __init__(
        self,
        *,
        max_batch_size: int,
        max_wait_ms: int,
        process_fn: Callable[[list[T]], Awaitable[list[R]]],
        name: str = "batcher",
    ):
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if max_wait_ms < 0:
            raise ValueError("max_wait_ms must be >= 0")
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.process_fn = process_fn
        self.name = name
        self._queue: asyncio.Queue[tuple[T, asyncio.Future[R]]] = asyncio.Queue()
        self._loop_task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._loop_task is not None:
            return
        self._loop_task = asyncio.create_task(self._run_loop(), name=f"{self.name}-loop")

    async def stop(self) -> None:
        if self._loop_task is None:
            return
        self._loop_task.cancel()
        try:
            await self._loop_task
        except asyncio.CancelledError:
            pass
        self._loop_task = None

    async def submit(self, item: T) -> R:
        fut: asyncio.Future[R] = asyncio.get_event_loop().create_future()
        await self._queue.put((item, fut))
        return await fut

    async def _run_loop(self) -> None:
        while True:
            try:
                batch, futures = await self._collect_batch()
            except asyncio.CancelledError:
                raise
            except Exception:
                LOG.exception("[%s] error while collecting batch", self.name)
                continue

            t0 = time.perf_counter()
            try:
                results = await self.process_fn(batch)
            except Exception as exc:
                # Propagate to every waiter; don't crash the loop.
                LOG.exception("[%s] process_fn failed for batch of %d", self.name, len(batch))
                for fut in futures:
                    if not fut.done():
                        fut.set_exception(exc)
                continue

            if len(results) != len(futures):
                err = RuntimeError(
                    f"process_fn returned {len(results)} results for "
                    f"{len(futures)} items"
                )
                for fut in futures:
                    if not fut.done():
                        fut.set_exception(err)
                continue

            for fut, res in zip(futures, results):
                if not fut.done():
                    fut.set_result(res)

            dt_ms = (time.perf_counter() - t0) * 1000
            LOG.debug("[%s] flushed batch=%d in %.1fms", self.name, len(batch), dt_ms)

    async def _collect_batch(self) -> tuple[list[T], list[asyncio.Future[R]]]:
        # Block until at least one item is available.
        first_item, first_fut = await self._queue.get()
        batch: list[T] = [first_item]
        futures: list[asyncio.Future] = [first_fut]

        # Coalesce additional items until window closes or cap hits.
        if self.max_wait_ms > 0 and self.max_batch_size > 1:
            deadline = time.monotonic() + self.max_wait_ms / 1000.0
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item, fut = await asyncio.wait_for(
                        self._queue.get(), timeout=remaining
                    )
                except asyncio.TimeoutError:
                    break
                batch.append(item)
                futures.append(fut)
        return batch, futures
