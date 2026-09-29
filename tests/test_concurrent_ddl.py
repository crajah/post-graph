"""Provisioning must tolerate a concurrent creator.

`IF NOT EXISTS` is not atomic. It checks the catalog and then creates, and two
sessions can both pass the check before either inserts, so one gets a
uniqueness violation from a system index rather than the silent no-op the
clause implies. Every multi-replica deployment hits this on the first write to
a new space: measured before the fix, six replicas creating one table produced
five failures and one success.

Two different things go wrong and need opposite treatment. A statement that
lost a race has already achieved what the caller wanted, so it is ignored. A
statement killed to break a deadlock has achieved nothing, so it is retried.
Everything else must still surface -- an index silently swallowed becomes a
wrong answer much later.
"""
import asyncio
import uuid

import pytest
from conftest import DSN

from post_graph import AsyncPostGraph, is_concurrent_creation, is_retryable_ddl


def _err(msg, code=None):
    e = Exception(msg)
    e.sqlstate = code
    return e


class TestClassifier:
    """No database needed: the distinctions the whole fix rests on."""

    @pytest.mark.parametrize("msg,code", [
        ('duplicate key value violates unique constraint "pg_namespace_nspname_index"', "23505"),
        ('duplicate key value violates unique constraint "pg_class_relname_nsp_index"', "23505"),
        ('relation "people" already exists', "42P07"),
        ('schema "r1" already exists', "42P06"),
        ("trigger already exists", "42710"),
        ('column "space" of relation "people" already exists', "42701"),
    ])
    def test_a_lost_race_is_tolerated(self, msg, code):
        assert is_concurrent_creation(_err(msg, code))

    def test_an_application_duplicate_key_is_not_a_race(self):
        """The ambiguous case, and the one that matters. A unique_violation is
        what CREATE SCHEMA raises when it loses -- and also what an application
        insert raises on a real duplicate. Only the catalog's own indexes count."""
        assert not is_concurrent_creation(
            _err('duplicate key value violates unique constraint "idx_people_uuid"', "23505"))

    @pytest.mark.parametrize("msg,code", [
        ('relation "people" does not exist', "42P01"),
        ("permission denied for schema public", "42501"),
        ("out of shared memory", "53200"),
    ])
    def test_a_real_failure_still_surfaces(self, msg, code):
        assert not is_concurrent_creation(_err(msg, code))

    def test_a_deadlock_is_retried_not_swallowed(self):
        e = _err("deadlock detected", "40P01")
        assert is_retryable_ddl(e)
        assert not is_concurrent_creation(e), "retrying is not the same as ignoring"

    def test_a_raced_function_replace_is_retryable(self):
        assert is_retryable_ddl(_err("tuple concurrently updated", "XX000"))


REPLICAS = 6


class TestConcurrentProvisioning:

    @pytest.mark.asyncio
    @pytest.mark.parametrize("schema_per_realm", [True, False])
    async def test_replicas_racing_to_create_a_vertex_table(self, pg_client, schema_per_realm):
        realm = "cdt_" + uuid.uuid4().hex[:10]
        table = f"t_{realm[-8:]}"
        clients = [AsyncPostGraph(dsn=DSN, schema_per_realm=schema_per_realm)
                   for _ in range(REPLICAS)]
        for c in clients:
            await c.connect()
        try:
            out = await asyncio.gather(
                *[c.create_vertex_table(table, realm=realm, vector_dim=8) for c in clients],
                return_exceptions=True,
            )
            failed = [r for r in out if isinstance(r, BaseException)]
            assert not failed, f"{len(failed)}/{REPLICAS} replicas failed: {failed[:2]}"

            # Provisioned by a stampede, and still actually usable.
            v = await clients[0].add_vertex(table, realm=realm, payload={"n": "x"},
                                            embedding=[0.1] * 8)
            got = await clients[0].get_vertex(table, realm=realm, vertex_id=v.id)
            assert got is not None and got.payload["n"] == "x"
        finally:
            for c in clients:
                await c.close()
            await _cleanup(pg_client, realm, table, schema_per_realm)

    @pytest.mark.asyncio
    async def test_replicas_racing_to_create_an_edge_table(self, pg_client):
        realm = "cdt_" + uuid.uuid4().hex[:10]
        clients = [AsyncPostGraph(dsn=DSN, schema_per_realm=True) for _ in range(REPLICAS)]
        for c in clients:
            await c.connect()
        try:
            await clients[0].create_vertex_table("people", realm=realm, vector_dim=8)
            out = await asyncio.gather(
                *[c.create_edge_table("knows", from_vertex_table="people",
                                      to_vertex_table="people", realm=realm)
                  for c in clients],
                return_exceptions=True,
            )
            failed = [r for r in out if isinstance(r, BaseException)]
            assert not failed, f"{len(failed)}/{REPLICAS} replicas failed: {failed[:2]}"
        finally:
            for c in clients:
                await c.close()
            await pg_client._execute(f'DROP SCHEMA IF EXISTS "{realm}" CASCADE')

    @pytest.mark.asyncio
    async def test_replicas_racing_to_create_a_payload_index(self, pg_client):
        realm = "cdt_" + uuid.uuid4().hex[:10]
        clients = [AsyncPostGraph(dsn=DSN, schema_per_realm=True) for _ in range(REPLICAS)]
        for c in clients:
            await c.connect()
        try:
            await clients[0].create_vertex_table("events", realm=realm, vector_dim=8)
            out = await asyncio.gather(
                *[c.create_payload_index("events", realm=realm, key="due_at") for c in clients],
                return_exceptions=True,
            )
            failed = [r for r in out if isinstance(r, BaseException)]
            assert not failed, f"{len(failed)}/{REPLICAS} replicas failed: {failed[:2]}"
            assert all(r == "idx_events_payload_due_at" for r in out), \
                "every replica should get the index name back, raced or not"
        finally:
            for c in clients:
                await c.close()
            await pg_client._execute(f'DROP SCHEMA IF EXISTS "{realm}" CASCADE')

    @pytest.mark.asyncio
    async def test_a_real_ddl_failure_still_raises(self, pg_client, clean_realm):
        """Tolerance must not have become blanket suppression."""
        from post_graph.errors import TableNotFoundError
        with pytest.raises((TableNotFoundError, Exception)) as exc:
            await pg_client.create_payload_index("no_such_table", realm=clean_realm, key="k")
        assert "no_such_table" in str(exc.value) or "does not exist" in str(exc.value).lower()


async def _cleanup(pg_client, realm, table, schema_per_realm):
    if schema_per_realm:
        await pg_client._execute(f'DROP SCHEMA IF EXISTS "{realm}" CASCADE')
    else:
        for t in (table, f"{table}_audit", f"{table}_data"):
            await pg_client._execute(f'DROP TABLE IF EXISTS "{t}" CASCADE')
