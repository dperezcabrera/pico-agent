import asyncio
import os
from unittest.mock import patch

import pytest

from pico_agent.scheduler import PlatformScheduler


class TestPlatformScheduler:
    def test_default_concurrency_limit(self):
        scheduler = PlatformScheduler()
        assert scheduler.limit == 10

    def test_custom_concurrency_limit_from_env(self):
        with patch.dict(os.environ, {"PICO_AGENT_MAX_CONCURRENCY": "5"}):
            scheduler = PlatformScheduler()
            assert scheduler.limit == 5

    def test_semaphore_property(self):
        scheduler = PlatformScheduler()
        assert isinstance(scheduler.semaphore, asyncio.Semaphore)

    @pytest.mark.asyncio
    async def test_acquire_and_release(self):
        scheduler = PlatformScheduler()
        await scheduler.acquire()
        # Should complete without blocking
        scheduler.release()

    @pytest.mark.asyncio
    async def test_concurrent_access_limited(self):
        with patch.dict(os.environ, {"PICO_AGENT_MAX_CONCURRENCY": "2"}):
            scheduler = PlatformScheduler()

        acquired_count = 0
        max_concurrent = 0

        async def task():
            nonlocal acquired_count, max_concurrent
            await scheduler.acquire()
            acquired_count += 1
            max_concurrent = max(max_concurrent, acquired_count)
            await asyncio.sleep(0.01)
            acquired_count -= 1
            scheduler.release()

        # Run 5 tasks with concurrency limit of 2
        await asyncio.gather(*[task() for _ in range(5)])

        # Max concurrent should not exceed limit
        assert max_concurrent <= scheduler.limit

    @pytest.mark.asyncio
    async def test_semaphore_context_manager_pattern(self):
        scheduler = PlatformScheduler()

        # Using the semaphore directly as context manager
        async with scheduler.semaphore:
            # Task is running
            pass
        # Semaphore automatically released

    def test_scheduler_limit_type(self):
        with patch.dict(os.environ, {"PICO_AGENT_MAX_CONCURRENCY": "15"}):
            scheduler = PlatformScheduler()
            assert isinstance(scheduler.limit, int)
            assert scheduler.limit == 15


class TestSchedulerEdgeCases:
    def test_empty_env_uses_default(self):
        # Ensure PICO_AGENT_MAX_CONCURRENCY is not set
        env = os.environ.copy()
        if "PICO_AGENT_MAX_CONCURRENCY" in env:
            del env["PICO_AGENT_MAX_CONCURRENCY"]

        with patch.dict(os.environ, env, clear=True):
            with patch.dict(os.environ, {}, clear=False):
                # Re-add other env vars but not PICO_AGENT_MAX_CONCURRENCY
                scheduler = PlatformScheduler()
                # Default is 10
                assert scheduler.limit == 10

    @pytest.mark.asyncio
    async def test_multiple_acquires_up_to_limit(self):
        with patch.dict(os.environ, {"PICO_AGENT_MAX_CONCURRENCY": "3"}):
            scheduler = PlatformScheduler()

        # Should be able to acquire up to the limit
        await scheduler.acquire()
        await scheduler.acquire()
        await scheduler.acquire()

        # Release all
        scheduler.release()
        scheduler.release()
        scheduler.release()
