"""Dispatch on what a pool can do, not on whether it is an ``asyncpg.Pool``.

``AsyncPostGraph`` takes either a connection or a pool and has to tell them
apart at every call site. It used to ask ``isinstance(self.connection,
asyncpg.Pool)``, which meant that anything that was not literally that class
was driven as a single connection -- so ``acquire()`` was never called and the
first query died on ``AttributeError: 'X' object has no attribute 'fetch'``.

The object that falls through that test is the useful one: a wrapper whose
``acquire()`` does something to the connection before handing it over -- sets a
role, a ``search_path``, a statement timeout. That is the only safe shape
behind PgBouncer in transaction mode, where a session-level ``SET ROLE`` would
leave a server connection wearing one tenant's role for the next client that
borrows it. The alternative -- subclassing ``asyncpg.Pool`` -- obliges a caller
to import asyncpg, which is the thing a caller using this library to own its
SQL is trying not to do.

So: anything with an ``acquire()`` that is not an asyncpg ``Connection`` is a
pool.
"""

import uuid
from contextlib import asynccontextmanager

import asyncpg
import pytest
from conftest import DSN, drop_role, grant_role

from post_graph import AsyncPostGraph
from post_graph.errors import PostGraphError


class MinimalPool:
    """The whole protocol: an ``acquire()`` that is an async context manager.

    Deliberately has nothing else -- no ``fetch``, no ``execute``, no
    ``transaction`` -- so any site that drives it as a connection raises
    ``AttributeError`` and names itself.
    """

    def __init__(self, pool):
        self._pool = pool
        self.acquisitions = 0

    @asynccontextmanager
    async def acquire(self):
        self.acquisitions += 1
        async with self._pool.acquire() as conn:
            yield conn


class RoleAssumingPool(MinimalPool):
    """A pool wrapper that does the thing wrappers exist for.

    Each acquisition opens a transaction and assumes a role for its duration,
    which is what makes it safe through a transaction-mode pooler: nothing
    about the role survives the acquisition.
    """

    def __init__(self, pool, role):
        super().__init__(pool)
        self.role = role

    @asynccontextmanager
    async def acquire(self):
        self.acquisitions += 1
        async with self._pool.acquire() as conn:
            tx = conn.transaction()
            await tx.start()
            try:
                await conn.execute(f'SET LOCAL ROLE "{self.role}"')
                yield conn
            except BaseException:
                await tx.rollback()
                raise
            else:
                await tx.commit()


@pytest.fixture()
async def raw_pool():
    pool = await asyncpg.create_pool(DSN, min_size=1, max_size=4)
    yield pool
    await pool.close()


class TestTheProtocol:
    """What ``_is_pool`` answers, before any SQL runs."""

    @pytest.mark.asyncio
    async def test_a_wrapper_with_acquire_is_a_pool(self, raw_pool):
        client = AsyncPostGraph(connection_or_pool=MinimalPool(raw_pool))
        assert client._is_pool is True

    @pytest.mark.asyncio
    async def test_an_asyncpg_pool_is_still_a_pool(self, raw_pool):
        assert AsyncPostGraph(connection_or_pool=raw_pool)._is_pool is True

    @pytest.mark.asyncio
    async def test_a_connection_is_not(self):
        conn = await asyncpg.connect(DSN)
        try:
            assert AsyncPostGraph(connection_or_pool=conn)._is_pool is False
        finally:
            await conn.close()

    @pytest.mark.asyncio
    async def test_a_connection_borrowed_from_a_pool_is_not(self, raw_pool):
        """The edge the capability test has to get right.

        A caller holding a borrowed connection from ``async with
        pool.acquire()`` and constructing a client over it must get the
        connection branch, or the explicit transaction it opened is silently
        abandoned. Both halves of the test exclude it: a
        ``PoolConnectionProxy`` is an ``asyncpg.Connection`` for ``isinstance``
        *and* has no ``acquire``, since attribute lookup on it delegates to the
        connection. Asserted here because it is the belt and the braces, and a
        test that knows only one of them would not notice losing the other.
        """
        async with raw_pool.acquire() as conn:
            assert isinstance(conn, asyncpg.Connection)
            assert not hasattr(conn, "acquire")
            assert AsyncPostGraph(connection_or_pool=conn)._is_pool is False

    def test_no_connection_yet_is_not_a_pool(self):
        """A client built from a DSN has no connection until ``connect()``."""
        assert AsyncPostGraph(dsn=DSN)._is_pool is False

    def test_the_isinstance_test_is_gone(self):
        """A grep, as a test. The defect was 22 identical lines, and a 23rd
        added later would reintroduce it in one place only -- which is the
        hardest version to find."""
        import inspect

        from post_graph import client_asyncpg
        source = inspect.getsource(client_asyncpg)
        offending = [line.strip() for line in source.splitlines()
                     if "isinstance(self.connection, asyncpg.Pool)" in line]
        assert not offending, offending


class TestItActuallyRuns:
    """The reproduction from the report, and then the point of it."""

    @pytest.mark.asyncio
    async def test_fetch_through_a_minimal_wrapper(self, raw_pool):
        wrapper = MinimalPool(raw_pool)
        client = AsyncPostGraph(connection_or_pool=wrapper)
        rows = await client._fetch("SELECT 1 AS one")
        assert rows[0]["one"] == 1
        assert wrapper.acquisitions == 1

    @pytest.mark.asyncio
    async def test_the_write_and_read_paths_both_acquire(self, raw_pool, realm):
        """A write goes through ``_run_in_tx``, a read does not; the old test
        broke both, so both are checked."""
        wrapper = MinimalPool(raw_pool)
        client = AsyncPostGraph(connection_or_pool=wrapper)
        table = f"pp_{uuid.uuid4().hex[:8]}"
        try:
            await client.create_vertex_table(table, realm=realm)
            v = await client.add_vertex(table, realm=realm, payload={"n": 1})
            got = await client.get_vertex(table, realm, v.id)
            assert got.payload == {"n": 1}
            assert len(await client.get_vertices(table, realm=realm)) == 1
            assert await client.count_vertices(table, realm=realm) == 1
        finally:
            await client._execute(f'DROP TABLE IF EXISTS "{table}" CASCADE')
            await client._execute(f'DROP TABLE IF EXISTS "{table}_audit" CASCADE')

    @pytest.mark.asyncio
    async def test_every_acquisition_carries_the_role(self, raw_pool, realm, grantee):
        """The use the whole change is for.

        The table is owned by the test's own role and the wrapper's role holds
        only ``SELECT`` and ``INSERT`` on it, so a statement that reached the
        database outside an acquisition -- or an acquisition whose ``SET LOCAL
        ROLE`` was skipped -- would run as the owner and the grant would prove
        nothing. The negative control below is the other half of that.
        """
        owner = await asyncpg.connect(DSN)
        table = f"pp_{uuid.uuid4().hex[:8]}"
        try:
            setup = AsyncPostGraph(connection_or_pool=owner)
            await setup.create_vertex_table(table, realm=realm, audited=False)
            await grant_role(owner, grantee, [
                f'GRANT SELECT, INSERT ON "{table}" TO "{grantee}"',
                f'GRANT USAGE, SELECT ON SEQUENCE "{table}_id_seq" TO "{grantee}"',
            ])

            wrapper = RoleAssumingPool(raw_pool, grantee)
            client = AsyncPostGraph(connection_or_pool=wrapper)

            v = await client.add_vertex(table, realm=realm, payload={"n": 1})
            assert (await client.get_vertex(table, realm, v.id)).payload == {"n": 1}
            # One acquisition per call, each with its own role assumption.
            assert wrapper.acquisitions >= 2

            # The role may not delete: the grant is the containment, and it is
            # in force on the connections post-graph is using.
            with pytest.raises(asyncpg.InsufficientPrivilegeError):
                await client.delete_vertex(table, realm=realm, vertex_id=v.id)
        finally:
            await drop_role(owner, grantee)
            await owner.execute(f'DROP TABLE IF EXISTS "{table}" CASCADE')
            await owner.close()

    @pytest.mark.asyncio
    async def test_the_negative_control(self, raw_pool, realm, grantee):
        """Without the grant the same calls are refused, so the test above is
        measuring the role and not merely finding a permissive database."""
        owner = await asyncpg.connect(DSN)
        table = f"pp_{uuid.uuid4().hex[:8]}"
        try:
            setup = AsyncPostGraph(connection_or_pool=owner)
            await setup.create_vertex_table(table, realm=realm, audited=False)
            await grant_role(owner, grantee, [])        # no table grants at all

            client = AsyncPostGraph(
                connection_or_pool=RoleAssumingPool(raw_pool, grantee))
            with pytest.raises(asyncpg.InsufficientPrivilegeError):
                await client.add_vertex(table, realm=realm, payload={"n": 1})
        finally:
            await drop_role(owner, grantee)
            await owner.execute(f'DROP TABLE IF EXISTS "{table}" CASCADE')
            await owner.close()


class TestWhatStillRefusesAPool:
    """Capability dispatch must not widen what a pool is allowed to do."""

    @pytest.mark.asyncio
    async def test_advisory_xact_lock_refuses_a_wrapper_too(self, raw_pool, realm):
        """A transaction-scoped lock on a borrowed connection is released by
        the acquisition that took it, so it is already gone when the caller
        writes. That is true of a wrapper exactly as it is of a pool, and the
        error has to say so rather than locking nothing."""
        client = AsyncPostGraph(connection_or_pool=MinimalPool(raw_pool))
        with pytest.raises(PostGraphError, match="connection-bound"):
            await client.advisory_xact_lock("whatever", realm=realm)
