import asyncio
import os

from pico_ioc import component


@component(scope="singleton")
class PlatformScheduler:
    def __init__(self):
        self.limit = int(os.getenv("PICO_AGENT_MAX_CONCURRENCY", "10"))
        self._semaphore = asyncio.Semaphore(self.limit)

    async def acquire(self):
        await self._semaphore.acquire()

    def release(self):
        self._semaphore.release()

    @property
    def semaphore(self):
        return self._semaphore
