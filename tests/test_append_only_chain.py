"""Reading the newest row, and serialising the writers who append after it.

Together these make an append-only hash chain writable: read the current head,
compute the next row from it, insert, commit -- with no two appenders reading
the same head.

The second is what makes the first mean anything. A sequence does not order
commits: two appenders can take nextval 5 and 6 and commit in the other order,
so both can read the same newest row and record it as their predecessor.
`ORDER BY id DESC LIMIT 1` is trustworthy only while one appender is in flight.
"""

import asyncio

import asyncpg
import pytest
from conftest import DSN

from post_graph import AsyncPostGraph
from post_graph._query import ADVISORY_CLASSID, advisory_key
from post_graph.errors import PostGraphError


class TestKeyDerivation:
    """No database needed, and the property that matters most is here."""

    def test_keys_are_stable_across_processes(self):
        """Python's hash() of a string is salted per interpreter, so deriving
        the key with it would give two replicas different keys for the same
        table -- different locks, nothing serialised. That is the exact silent
        failure the lock exists to prevent, reintroduced by the derivation."""
        import subprocess
        import sys
        code = ("from post_graph._query import ADVISORY_CLASSID, advisory_key;"
                "print(ADVISORY_CLASSID, advisory_key('r', 'access_log'))")
        seen = set()
        for seed in ("0", "1", "987654"):
            out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                                 text=True, env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"})
            seen.add(out.stdout.strip())
        assert len(seen) == 1, f"key is not process-stable: {seen}"

    def test_keys_fit_the_two_argument_lock_space(self):
        for realm, table in (("r1", "a"), ("r2", "b"), ("", "")):
            for key in (ADVISORY_CLASSID, advisory_key(realm, table)):
                assert -2 ** 31 <= key < 2 ** 31

    def test_different_tables_get_different_keys(self):
        assert advisory_key("r", "a") != advisory_key("r", "b")
        assert advisory_key("r1", "a") != advisory_key("r2", "a")


class TestNewestRow:

    @pytest.mark.asyncio
    async def test_the_newest_row_is_the_last_inserted(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("log", realm=realm)
        ids = []
        for i in range(25):
            v = await pg_client.add_vertex("log", realm=realm, payload={"seq": i})
            ids.append(v.id)

        newest = await pg_client.newest_vertex("log", realm=realm)
        assert newest is not None
        assert newest.id == ids[-1]
        assert newest.payload["seq"] == 24

    @pytest.mark.asyncio
    async def test_it_is_not_what_a_payload_timestamp_would_return(self, pg_client, clean_realm):
        """This half is what distinguishes the new capability from the one that
        already existed. The chain's order must not be something a clock skew
        between two writers can change, so the timestamps are written
        deliberately out of order and must be ignored."""
        realm = clean_realm
        await pg_client.create_vertex_table("log", realm=realm)
        # Inserted last, but claims the earliest instant.
        await pg_client.add_vertex("log", realm=realm, payload={"at": "2026-01-02", "n": 1})
        await pg_client.add_vertex("log", realm=realm, payload={"at": "2026-01-03", "n": 2})
        last = await pg_client.add_vertex("log", realm=realm, payload={"at": "2026-01-01", "n": 3})

        newest = await pg_client.newest_vertex("log", realm=realm)
        assert newest.id == last.id
        assert newest.payload["n"] == 3

        by_payload = await pg_client.find_vertices(
            "log", realm=realm, order_by="at", descending=True, limit=1)
        assert by_payload[0].payload["n"] == 2, "payload ordering disagrees, as it should"

    @pytest.mark.asyncio
    async def test_an_empty_table_has_no_newest_row(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("log", realm=realm)
        assert await pg_client.newest_vertex("log", realm=realm) is None

    @pytest.mark.asyncio
    async def test_order_by_column_works_on_edges_too(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("n", realm=realm)
        await pg_client.create_edge_table("e", from_vertex_table="n",
                                          to_vertex_table="n", realm=realm)
        a = await pg_client.add_vertex("n", realm=realm, payload={})
        b = await pg_client.add_vertex("n", realm=realm, payload={})
        last = None
        for i in range(5):
            last = await pg_client.add_edge("e", realm=realm, from_id=a.id, to_id=b.id,
                                            relation_type=f"r{i}", payload={"i": i})
        newest = await pg_client.find_edges(
            "e", realm=realm, order_by_column="id", descending=True, limit=1)
        assert newest[0].id == last.id

    @pytest.mark.asyncio
    async def test_a_payload_key_named_id_still_means_the_payload_key(self, pg_client, clean_realm):
        """order_by="id" must not have silently changed meaning: it is a
        payload key, and order_by_column="id" is the column."""
        realm = clean_realm
        await pg_client.create_vertex_table("log", realm=realm)
        await pg_client.add_vertex("log", realm=realm, payload={"id": "zzz"})
        await pg_client.add_vertex("log", realm=realm, payload={"id": "aaa"})

        by_payload = await pg_client.find_vertices(
            "log", realm=realm, order_by="id", descending=True, limit=1)
        assert by_payload[0].payload["id"] == "zzz"

        by_column = await pg_client.newest_vertex("log", realm=realm)
        assert by_column.payload["id"] == "aaa", "the column, not the payload key"

    @pytest.mark.asyncio
    async def test_an_arbitrary_column_name_is_refused(self, pg_client, clean_realm):
        with pytest.raises(ValueError):
            await pg_client.find_vertices("log", realm=clean_realm,
                                          order_by_column="payload; DROP TABLE x")

    @pytest.mark.asyncio
    async def test_both_ordering_arguments_at_once_is_refused(self, pg_client, clean_realm):
        with pytest.raises(PostGraphError):
            await pg_client.find_vertices("log", realm=clean_realm,
                                          order_by="at", order_by_column="id")


class TestTheLockRefusesWhenItCannotWork:

    @pytest.mark.asyncio
    async def test_a_pool_bound_client_raises(self, pg_client, clean_realm):
        """On a pool the lock would be taken on an arbitrary connection and
        released with its own implicit transaction, before the caller's write.
        It would appear to work and serialise nothing."""
        with pytest.raises(PostGraphError) as exc:
            await pg_client.advisory_xact_lock("log", realm=clean_realm)
        assert "connection-bound" in str(exc.value)

    @pytest.mark.asyncio
    async def test_a_connection_not_in_a_transaction_raises(self, clean_realm):
        """The same silent failure one step further in: a transaction-scoped
        lock taken outside a transaction is released by the implicit
        transaction that took it, so it is gone before the call returns."""
        conn = await asyncpg.connect(DSN)
        try:
            client = AsyncPostGraph(connection_or_pool=conn)
            with pytest.raises(PostGraphError) as exc:
                await client.advisory_xact_lock("log", realm=clean_realm)
            assert "inside a transaction" in str(exc.value)
        finally:
            await conn.close()

    @pytest.mark.asyncio
    async def test_inside_a_transaction_it_is_actually_held(self, clean_realm):
        conn = await asyncpg.connect(DSN)
        try:
            client = AsyncPostGraph(connection_or_pool=conn)
            async with conn.transaction():
                await client.advisory_xact_lock("log", realm=clean_realm)
                held = await conn.fetchval(
                    "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' "
                    "AND classid = $1 AND pid = pg_backend_pid()",
                    ADVISORY_CLASSID,
                )
                assert held == 1, "the lock must be held for the rest of the transaction"
            after = await conn.fetchval(
                "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' "
                "AND classid = $1 AND pid = pg_backend_pid()",
                ADVISORY_CLASSID,
            )
            assert after == 0, "and released at commit, with no unlock call"
        finally:
            await conn.close()


# How many appenders race. Chosen by raising it until the no-lock control
# below actually forked -- a concurrency test whose failing case has never run
# is not evidence.
APPENDERS = 12


async def _append(realm, table, use_lock):
    """One appender, on its own connection: lock, read head, insert, commit.

    Its own connection on purpose. Sharing one connection would serialise the
    appenders for the wrong reason, and the test would pass whether the lock
    worked or not.
    """
    conn = await asyncpg.connect(DSN)
    try:
        client = AsyncPostGraph(connection_or_pool=conn)
        async with conn.transaction():
            if use_lock:
                await client.advisory_xact_lock(table, realm=realm)
            head = await client.newest_vertex(table, realm=realm)
            prev = head.payload["n"] if head else 0
            await client.add_vertex(table, realm=realm, payload={"n": prev + 1, "prev": prev})
    finally:
        await conn.close()


async def _chain(pg_client, realm, table):
    rows = await pg_client.find_vertices(table, realm=realm, order_by_column="id")
    return [(r.payload["prev"], r.payload["n"]) for r in rows]


class TestSerialisation:

    @pytest.mark.asyncio
    async def test_concurrent_appenders_produce_one_unbroken_chain(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("chain", realm=realm)

        await asyncio.gather(*[_append(realm, "chain", True) for _ in range(APPENDERS)])

        chain = await _chain(pg_client, realm, "chain")
        assert len(chain) == APPENDERS

        values = [n for _, n in chain]
        assert len(set(values)) == APPENDERS, f"a value appeared twice: {values}"
        assert values == sorted(values) == list(range(1, APPENDERS + 1))
        for prev, n in chain:
            assert prev == n - 1, f"row {n} records predecessor {prev}"

    @pytest.mark.asyncio
    async def test_without_the_lock_the_chain_forks(self, pg_client, clean_realm):
        """The control. If this ever stops forking, the test above has stopped
        being evidence and APPENDERS needs raising until it forks again."""
        realm = clean_realm
        await pg_client.create_vertex_table("chain", realm=realm)

        await asyncio.gather(*[_append(realm, "chain", False) for _ in range(APPENDERS)])

        values = [n for _, n in await _chain(pg_client, realm, "chain")]
        assert len(set(values)) < len(values), (
            f"no fork at APPENDERS={APPENDERS}: {values}. The locked test above is "
            f"not proving anything until this case can fail."
        )

    @pytest.mark.asyncio
    async def test_the_lock_does_not_serialise_a_different_table(self, pg_client, clean_realm):
        """The key is derived per (realm, table), so appending to one table
        must not block another."""
        realm = clean_realm
        for table in ("chain_a", "chain_b"):
            await pg_client.create_vertex_table(table, realm=realm)
        await asyncio.gather(
            *[_append(realm, "chain_a", True) for _ in range(4)],
            *[_append(realm, "chain_b", True) for _ in range(4)],
        )
        for table in ("chain_a", "chain_b"):
            values = [n for _, n in await _chain(pg_client, realm, table)]
            assert values == [1, 2, 3, 4], f"{table}: {values}"


class TestSQLAlchemyParity:
    """client_sqlalchemy's docstring says where, order_by and descending
    "behave exactly as on" the asyncpg client. Whatever lands in one belongs in
    the other, or that sentence stops being true."""

    @pytest.mark.asyncio
    async def test_newest_vertex_and_order_by_column(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_chain", realm=realm)
        last = None
        for i in range(12):
            last = await sa_client.add_vertex("sa_chain", realm=realm, payload={"n": i})

        newest = await sa_client.newest_vertex("sa_chain", realm=realm)
        assert newest is not None and newest.id == last.id and newest.payload["n"] == 11

        by_col = await sa_client.find_vertices(
            "sa_chain", realm=realm, order_by_column="id", descending=True, limit=1)
        assert by_col[0].id == last.id

    @pytest.mark.asyncio
    async def test_an_engine_bound_client_refuses_the_lock(self, sa_client, sa_clean_realm):
        """Same silent failure as a pool on the other client."""
        with pytest.raises(PostGraphError) as exc:
            await sa_client.advisory_xact_lock("sa_chain", realm=sa_clean_realm)
        assert "connection-bound" in str(exc.value)

    @pytest.mark.asyncio
    async def test_an_arbitrary_column_is_refused(self, sa_client, sa_clean_realm):
        with pytest.raises(ValueError):
            await sa_client.find_vertices("sa_chain", realm=sa_clean_realm,
                                          order_by_column="1; DROP TABLE x")

    @pytest.mark.asyncio
    async def test_the_lock_is_held_inside_a_transaction(self, sa_engine, sa_clean_realm):
        from sqlalchemy import text as _text

        from post_graph.client_sqlalchemy import SQLAlchemyPostGraph

        async with sa_engine.connect() as conn:
            async with conn.begin():
                client = SQLAlchemyPostGraph(conn)
                await client.advisory_xact_lock("sa_chain", realm=sa_clean_realm)
                held = (await conn.execute(_text(
                    "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' "
                    "AND classid = :c AND pid = pg_backend_pid()"),
                    {"c": ADVISORY_CLASSID})).scalar()
                assert held == 1
