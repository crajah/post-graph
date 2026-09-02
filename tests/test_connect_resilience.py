"""Pool defaults and connect() retry under a deploy herd.

Unit tests: create_pool is monkeypatched, no database needed. The scenario is
the production one -- a fleet's workers all connecting at deploy time, the
server briefly refusing with "too many clients", and the survivors staggering
in via backoff instead of crashing.
"""
import asyncpg
import pytest

from post_graph import AsyncPostGraph

pytestmark = pytest.mark.asyncio


def test_lazy_pool_default():
    c = AsyncPostGraph(dsn="postgresql://x/x")
    cfg = c.get_pool_config()
    assert cfg["min_size"] == 1        # a fleet holds N idle conns, not 10N
    assert cfg["max_size"] == 10


async def test_connect_retries_through_herd(monkeypatch):
    attempts = []

    async def flaky_create_pool(dsn=None, **kw):
        attempts.append(1)
        if len(attempts) < 3:
            raise asyncpg.TooManyConnectionsError("sorry, too many clients already")
        return object()                # stands in for the pool

    monkeypatch.setattr(asyncpg, "create_pool", flaky_create_pool)
    c = AsyncPostGraph(dsn="postgresql://x/x")
    await c.connect(retries=5, retry_base_delay=0.01)
    assert len(attempts) == 3
    assert c.connection is not None


async def test_connect_retries_exhausted(monkeypatch):
    async def always_full(dsn=None, **kw):
        raise asyncpg.TooManyConnectionsError("sorry, too many clients already")

    monkeypatch.setattr(asyncpg, "create_pool", always_full)
    c = AsyncPostGraph(dsn="postgresql://x/x")
    with pytest.raises(asyncpg.TooManyConnectionsError):
        await c.connect(retries=2, retry_base_delay=0.01)


async def test_retries_zero_fails_immediately(monkeypatch):
    attempts = []

    async def always_full(dsn=None, **kw):
        attempts.append(1)
        raise asyncpg.TooManyConnectionsError("sorry, too many clients already")

    monkeypatch.setattr(asyncpg, "create_pool", always_full)
    c = AsyncPostGraph(dsn="postgresql://x/x")
    with pytest.raises(asyncpg.TooManyConnectionsError):
        await c.connect(retries=0)
    assert len(attempts) == 1


async def test_other_errors_do_not_retry(monkeypatch):
    attempts = []

    async def bad_auth(dsn=None, **kw):
        attempts.append(1)
        raise asyncpg.InvalidPasswordError("nope")

    monkeypatch.setattr(asyncpg, "create_pool", bad_auth)
    c = AsyncPostGraph(dsn="postgresql://x/x")
    with pytest.raises(asyncpg.InvalidPasswordError):
        await c.connect(retries=5, retry_base_delay=0.01)
    assert len(attempts) == 1          # auth failure is not a herd; fail loud
