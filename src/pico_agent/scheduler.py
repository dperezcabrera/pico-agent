"""Concurrency scheduler for async agent operations.

``PlatformScheduler`` uses an ``asyncio.Semaphore`` to limit the number of
concurrent LLM calls, preventing resource exhaustion during map-reduce
workflows and parallel agent invocations.
"""

import asyncio
import os

from pico_ioc import component


@component(scope="singleton")
class PlatformScheduler:
    """Asyncio-based concurrency limiter for parallel agent operations.

    The concurrency limit is read from the ``PICO_AGENT_MAX_CONCURRENCY``
    environment variable (default: ``10``).

    Example:
        >>> async with scheduler.semaphore:
        ...     result = await some_llm_call()
    """

    def __init__(self):
        self.limit = int(os.getenv("PICO_AGENT_MAX_CONCURRENCY", "10"))
        self._semaphore = asyncio.Semaphore(self.limit)

    async def acquire(self):
        """Acquire a concurrency slot (blocks if the limit is reached)."""
        await self._semaphore.acquire()

    def release(self):
        """Release a concurrency slot."""
        self._semaphore.release()

    @property
    def semaphore(self):
        """The underlying ``asyncio.Semaphore`` for use with ``async with``."""
        return self._semaphore
