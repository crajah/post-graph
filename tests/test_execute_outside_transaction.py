"""A statement PostgreSQL refuses inside a transaction needs a path of its own.

Every write in this library runs inside a transaction, deliberately. A small
class of statements cannot: VACUUM, CREATE INDEX CONCURRENTLY, DROP INDEX
CONCURRENTLY, REINDEX CONCURRENTLY, CREATE DATABASE, DROP DATABASE, ALTER
SYSTEM, CREATE/DROP TABLESPACE. PostgreSQL rejects them with
ActiveSQLTransactionError, and a caller whose architecture forbids opening its
own connection -- because a connection opened behind the client's back would
make "every DDL call is inside a transaction" unenforceable for every other
caller -- then has nowhere to issue them from.

`execute` happens not to open a transaction, so it happens to work. That is an
implementation detail of `execute`, not a contract, and the point of this
method is the guarantee rather than the behaviour.

The case worth reading first is in TestWhatItRefuses: a pool wrapper whose
acquire() opens a transaction to hold a tenant's SET LOCAL ROLE -- the only
safe shape behind a transaction-mode pooler -- cannot serve this method. It is
the shape a caller most likely already has, and left undetected it reproduces
the exact error the method exists to escape.
"""

import uuid
from contextlib import asynccontextmanager

import asyncpg
import pytest
from conftest import DSN

from post_graph import AsyncPostGraph
from post_graph.errors import PostGraphError


class TxOpeningPool:
    """Wraps a pool and yields each connection already inside a transaction."""

    def __init__(self, pool):
        self._pool = pool

    @asynccontextmanager
    async def acquire(self):
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                yield conn


class IdlePool:
    """Wraps a pool and yields an idle session, as a plain pool does."""

    def __init__(self, pool):
        self._pool = pool
        self.acquisitions = 0

    @asynccontextmanager
    async def acquire(self):
        self.acquisitions += 1
        async with self._pool.acquire() as conn:
            yield conn


@pytest.fixture()
async def raw_pool():
    pool = await asyncpg.create_pool(DSN, min_size=1, max_size=3)
    yield pool
    await pool.close()


class TestItRuns:

    @pytest.mark.asyncio
    async def test_vacuum_through_a_pool(self, pg_client):
        assert await pg_client.execute_outside_transaction("VACUUM") == "VACUUM"

    @pytest.mark.asyncio
    async def test_vacuum_through_a_pool_wrapper_that_yields_an_idle_session(
        self, raw_pool
    ):
        wrapper = IdlePool(raw_pool)
        client = AsyncPostGraph(connection_or_pool=wrapper)
        assert await client.execute_outside_transaction("VACUUM") == "VACUUM"
        assert wrapper.acquisitions == 1

    @pytest.mark.asyncio
    async def test_on_a_connection_with_no_transaction_open(self):
        conn = await asyncpg.connect(DSN)
        try:
            client = AsyncPostGraph(connection_or_pool=conn)
            assert await client.execute_outside_transaction("VACUUM") == "VACUUM"
        finally:
            await conn.close()

    @pytest.mark.asyncio
    async def test_create_index_concurrently(self, pg_client, clean_realm):
        """The next statement in the class somebody will want."""
        await pg_client.create_vertex_table("things", realm=clean_realm)
        index = f"idx_cc_{uuid.uuid4().hex[:8]}"
        try:
            await pg_client.execute_outside_transaction(
                f'CREATE INDEX CONCURRENTLY "{index}" ON "things" (realm, id)')
            row = await pg_client._fetchrow(
                f'SELECT to_regclass(\'"{index}"\') AS oid')
            assert row["oid"] is not None, "the index was not created"
        finally:
            await pg_client.execute_outside_transaction(
                f'DROP INDEX CONCURRENTLY IF EXISTS "{index}"')

    @pytest.mark.asyncio
    async def test_the_same_statement_fails_on_the_transactional_path(
        self, pg_client
    ):
        """The control: this is a real restriction, not a style preference."""
        with pytest.raises(asyncpg.ActiveSQLTransactionError):
            await pg_client._run_in_tx(lambda conn: conn.execute("VACUUM"))


class TestWhatItRefuses:

    @pytest.mark.asyncio
    async def test_a_pool_wrapper_that_opens_a_transaction(self, raw_pool):
        """The trap a caller most likely already has.

        A wrapper that assumes a tenant's role per acquisition has to open a
        transaction to hold `SET LOCAL ROLE`, because a session-level `SET ROLE`
        would outlive the acquisition through a transaction-mode pooler. The
        statement then fails with the driver's own message about a transaction
        the caller never opened, which is why the refusal names the wrapper.
        """
        client = AsyncPostGraph(connection_or_pool=TxOpeningPool(raw_pool))
        with pytest.raises(PostGraphError, match="already inside a transaction"):
            await client.execute_outside_transaction("VACUUM")

    @pytest.mark.asyncio
    async def test_a_connection_whose_caller_holds_a_transaction(self):
        conn = await asyncpg.connect(DSN)
        try:
            client = AsyncPostGraph(connection_or_pool=conn)
            tx = conn.transaction()
            await tx.start()
            with pytest.raises(PostGraphError, match="holds an open transaction"):
                await client.execute_outside_transaction("VACUUM")
            await tx.rollback()
            # The caller's transaction is intact, not consumed by the refusal.
            assert await conn.fetchval("SELECT 1") == 1
        finally:
            await conn.close()

    @pytest.mark.asyncio
    async def test_it_does_not_open_a_connection_of_its_own(self):
        """The refusal above must not be a fallback that opens a side channel.

        A client with a DSN and no connection could reach the database by
        opening one. That is exactly the thing a caller forbidding its own
        connections is protecting, so it refuses instead.
        """
        client = AsyncPostGraph(dsn=DSN)
        with pytest.raises(PostGraphError, match="no connection"):
            await client.execute_outside_transaction("VACUUM")

    @pytest.mark.parametrize("statement", [
        "INSERT INTO t (a) VALUES (1)",
        "update t set a = 1",
        "DELETE FROM t",
        "TRUNCATE t",
        "COPY t FROM STDIN",
        "WITH x AS (SELECT 1) INSERT INTO t SELECT * FROM x",
    ])
    @pytest.mark.asyncio
    async def test_a_statement_with_a_transactional_path(self, pg_client, statement):
        """Routing a write through here would skip app.current_user_id, so an
        audited table would record it against nobody. A CTE is in the list
        because it is the obvious way round a first-word check."""
        with pytest.raises(PostGraphError, match="transactional path"):
            await pg_client.execute_outside_transaction(statement)

    @pytest.mark.asyncio
    async def test_an_empty_statement(self, pg_client):
        for empty in ("", "   ", None):
            with pytest.raises(PostGraphError, match="needs a statement"):
                await pg_client.execute_outside_transaction(empty)


class TestOneStatementPerCall:
    """Enforced by PostgreSQL, not by policy -- which is worth a test, because
    it is the reason this takes a statement rather than holding a connection
    for a block of them."""

    @pytest.mark.asyncio
    async def test_postgres_refuses_the_class_in_a_multi_statement_string(self):
        """A multi-statement simple query is one implicit transaction, so the
        statement fails with the error this method exists to escape -- naming a
        transaction the caller never opened."""
        conn = await asyncpg.connect(DSN)
        try:
            with pytest.raises(asyncpg.ActiveSQLTransactionError):
                await conn.execute("VACUUM; VACUUM;")
        finally:
            await conn.close()

    @pytest.mark.asyncio
    async def test_and_a_failure_rolls_back_what_came_before_it(self):
        """The other half: the implicit transaction is a real one."""
        conn = await asyncpg.connect(DSN)
        table = f"eot_{uuid.uuid4().hex[:8]}"
        try:
            await conn.execute(f"CREATE TABLE {table} (a int)")
            with pytest.raises(asyncpg.DivisionByZeroError):
                await conn.execute(
                    f"INSERT INTO {table} VALUES (1); SELECT 1/0;")
            assert await conn.fetchval(f"SELECT count(*) FROM {table}") == 0
        finally:
            await conn.execute(f"DROP TABLE IF EXISTS {table}")
            await conn.close()

    @pytest.mark.asyncio
    async def test_the_client_says_so_rather_than_letting_the_driver(self, pg_client):
        with pytest.raises(PostGraphError, match="takes one statement"):
            await pg_client.execute_outside_transaction("VACUUM; VACUUM;")

    @pytest.mark.asyncio
    async def test_a_trailing_semicolon_is_fine(self, pg_client):
        assert await pg_client.execute_outside_transaction("VACUUM;") == "VACUUM"


class TestSQLAlchemyParity:

    @pytest.mark.asyncio
    async def test_vacuum_through_an_engine(self, sa_client):
        assert await sa_client.execute_outside_transaction("VACUUM") is None

    @pytest.mark.asyncio
    async def test_the_same_statement_fails_on_the_transactional_path(self, sa_engine):
        from sqlalchemy import text
        with pytest.raises(Exception, match="cannot run inside a transaction"):
            async with sa_engine.begin() as conn:
                await conn.execute(text("VACUUM"))

    @pytest.mark.asyncio
    async def test_a_connection_whose_caller_holds_a_transaction(self, sa_engine):
        from post_graph.client_sqlalchemy import SQLAlchemyPostGraph
        async with sa_engine.connect() as conn:
            async with conn.begin():
                client = SQLAlchemyPostGraph(conn)
                with pytest.raises(PostGraphError, match="holds an open transaction"):
                    await client.execute_outside_transaction("VACUUM")

    @pytest.mark.asyncio
    async def test_a_statement_with_a_transactional_path(self, sa_client):
        with pytest.raises(PostGraphError, match="transactional path"):
            await sa_client.execute_outside_transaction("DELETE FROM t")
