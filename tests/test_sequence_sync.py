"""Tests for O(1) monotonic sequence advancement on explicit-ID writes (Issue #4).

Verifies that ``add_vertex(..., vertex_id=...)`` and ``add_edge(..., edge_id=...)``
across both ``AsyncPostGraph`` and ``SQLAlchemyPostGraph``:

1. Never emit ``SELECT ... MAX(id) FROM <table>`` (zero full-table scan).
2. Advance the table sequence monotonically when ``1 <= id <= MAX_SEQUENCE_SYNC_ID``.
3. Never rewind the sequence when a smaller explicit ID is inserted afterwards,
   including under forced space-scoped row-level security (RLS) where higher IDs
   in other spaces are invisible to the caller's ``SELECT``.
4. Succeed cleanly when the connected role holds ``USAGE`` (without ``UPDATE``)
   on the sequence, and never rewind when a role holds ``UPDATE`` without ``SELECT``.
5. Leave the sequence untouched for deterministic 63-bit hashed IDs above
   ``MAX_SEQUENCE_SYNC_ID`` (including ``INT64_MAX``) and when ``sync_sequence=False``.
"""

import uuid

import asyncpg
import pytest
from conftest import DSN, RecordingConnection, drop_role, grant_role

from post_graph import MAX_SEQUENCE_SYNC_ID, AsyncPostGraph


class TestNoTableScanOnExplicitIds:
    """Explicit-ID inserts must never scan MAX(id) on the vertex or edge table."""

    @pytest.mark.asyncio
    async def test_asyncpg_vertex_and_edge_emit_no_max_id_scan(self, pg_client_spr, clean_realm_spr):
        realm = clean_realm_spr
        v_tbl = f"v_{uuid.uuid4().hex[:8]}"
        e_tbl = f"e_{uuid.uuid4().hex[:8]}"
        await pg_client_spr.create_vertex_table(v_tbl, realm=realm, audited=False)
        await pg_client_spr.create_edge_table(
            e_tbl, from_vertex_table=v_tbl, to_vertex_table=v_tbl, realm=realm, audited=False
        )

        raw = await asyncpg.connect(DSN)
        try:
            rec = RecordingConnection(raw)
            client = AsyncPostGraph(connection_or_pool=rec, schema_per_realm=True)

            rec.queries.clear()
            a = await client.add_vertex(v_tbl, realm=realm, vertex_id=25, payload={"k": "a"})
            b = await client.add_vertex(v_tbl, realm=realm, vertex_id=30, payload={"k": "b"})
            for sql in rec.queries:
                assert "MAX(id)" not in sql.upper(), f"Unexpected MAX(id) scan: {sql}"

            rec.queries.clear()
            await client.add_edge(
                e_tbl, realm=realm, from_id=a.id, to_id=b.id, relation_type="rel", edge_id=40
            )
            for sql in rec.queries:
                assert "MAX(id)" not in sql.upper(), f"Unexpected MAX(id) scan on edge: {sql}"

            # Sequence advanced monotonically past 30 and 40
            c = await client.add_vertex(v_tbl, realm=realm, payload={"k": "c"})
            assert int(c.id) == 31
            e2 = await client.add_edge(
                e_tbl, realm=realm, from_id=a.id, to_id=c.id, relation_type="rel"
            )
            assert int(e2.id) == 41
        finally:
            await raw.close()

    @pytest.mark.asyncio
    async def test_sync_sequence_false_skips_setval_round_trip(self, pg_client_spr, clean_realm_spr):
        realm = clean_realm_spr
        v_tbl = f"v_{uuid.uuid4().hex[:8]}"
        e_tbl = f"e_{uuid.uuid4().hex[:8]}"
        await pg_client_spr.create_vertex_table(v_tbl, realm=realm, audited=False)
        await pg_client_spr.create_edge_table(
            e_tbl, from_vertex_table=v_tbl, to_vertex_table=v_tbl, realm=realm, audited=False
        )

        raw = await asyncpg.connect(DSN)
        try:
            rec = RecordingConnection(raw)
            client = AsyncPostGraph(connection_or_pool=rec, schema_per_realm=True)

            rec.queries.clear()
            a = await client.add_vertex(
                v_tbl, realm=realm, vertex_id=500, payload={}, sync_sequence=False
            )
            b = await client.add_vertex(
                v_tbl, realm=realm, vertex_id=501, payload={}, sync_sequence=False
            )
            assert not any("setval" in q.lower() for q in rec.queries)

            rec.queries.clear()
            await client.add_edge(
                e_tbl,
                realm=realm,
                from_id=a.id,
                to_id=b.id,
                relation_type="rel",
                edge_id=600,
                sync_sequence=False,
            )
            assert not any("setval" in q.lower() for q in rec.queries)

            # Auto-assigned IDs start at 1 because sync_sequence=False left sequences untouched
            auto_v = await client.add_vertex(v_tbl, realm=realm, payload={})
            assert int(auto_v.id) == 1
            auto_e = await client.add_edge(
                e_tbl, realm=realm, from_id=a.id, to_id=auto_v.id, relation_type="rel"
            )
            assert int(auto_e.id) == 1
        finally:
            await raw.close()


class TestMonotonicAndHashedIdsBothBackends:
    """Both AsyncPostGraph and SQLAlchemyPostGraph keep parity on monotonic & hashed IDs."""

    @pytest.mark.asyncio
    async def test_asyncpg_monotonic_never_rewinds_and_ignores_63bit_hashes(
        self, pg_client_spr, clean_realm_spr
    ):
        realm = clean_realm_spr
        v_tbl = f"v_{uuid.uuid4().hex[:8]}"
        e_tbl = f"e_{uuid.uuid4().hex[:8]}"
        await pg_client_spr.create_vertex_table(v_tbl, realm=realm, audited=False)
        await pg_client_spr.create_edge_table(
            e_tbl, from_vertex_table=v_tbl, to_vertex_table=v_tbl, realm=realm, audited=False
        )

        # Fresh sequence + id=1 advances sequence so next auto id is 2
        v1 = await pg_client_spr.add_vertex(v_tbl, realm=realm, vertex_id=1, payload={})
        v2 = await pg_client_spr.add_vertex(v_tbl, realm=realm, payload={})
        assert int(v1.id) == 1 and int(v2.id) == 2

        # Advance to 100, then insert smaller id=5 -> must NOT rewind to 5
        await pg_client_spr.add_vertex(v_tbl, realm=realm, vertex_id=100, payload={})
        await pg_client_spr.add_vertex(v_tbl, realm=realm, vertex_id=5, payload={})
        v_next = await pg_client_spr.add_vertex(v_tbl, realm=realm, payload={})
        assert int(v_next.id) == 101

        # 63-bit hashed IDs (including INT64_MAX) must not push sequence to ceiling
        int64_max = (1 << 63) - 1
        hashed_id = MAX_SEQUENCE_SYNC_ID + 12345
        await pg_client_spr.add_vertex(v_tbl, realm=realm, vertex_id=hashed_id, payload={})
        await pg_client_spr.add_vertex(v_tbl, realm=realm, vertex_id=int64_max, payload={})
        v_after_hash = await pg_client_spr.add_vertex(v_tbl, realm=realm, payload={})
        assert int(v_after_hash.id) == 102

        # Same guarantees on add_edge
        await pg_client_spr.add_edge(
            e_tbl, realm=realm, from_id=v1.id, to_id=v2.id, relation_type="r", edge_id=200
        )
        await pg_client_spr.add_edge(
            e_tbl, realm=realm, from_id=v1.id, to_id=v2.id, relation_type="r", edge_id=10
        )
        await pg_client_spr.add_edge(
            e_tbl, realm=realm, from_id=v1.id, to_id=v2.id, relation_type="r", edge_id=int64_max
        )
        e_next = await pg_client_spr.add_edge(
            e_tbl, realm=realm, from_id=v1.id, to_id=v2.id, relation_type="r"
        )
        assert int(e_next.id) == 201

    @pytest.mark.asyncio
    async def test_sqlalchemy_monotonic_never_rewinds_and_ignores_63bit_hashes(
        self, sa_client_spr, sa_clean_realm_spr
    ):
        realm = sa_clean_realm_spr
        v_tbl = f"v_{uuid.uuid4().hex[:8]}"
        e_tbl = f"e_{uuid.uuid4().hex[:8]}"
        await sa_client_spr.create_vertex_table(v_tbl, realm=realm, audited=False)
        await sa_client_spr.create_edge_table(
            e_tbl, from_vertex_table=v_tbl, to_vertex_table=v_tbl, realm=realm, audited=False
        )

        v1 = await sa_client_spr.add_vertex(v_tbl, realm=realm, vertex_id=1, payload={})
        v2 = await sa_client_spr.add_vertex(v_tbl, realm=realm, payload={})
        assert int(v1.id) == 1 and int(v2.id) == 2

        await sa_client_spr.add_vertex(v_tbl, realm=realm, vertex_id=100, payload={})
        await sa_client_spr.add_vertex(v_tbl, realm=realm, vertex_id=5, payload={})
        await sa_client_spr.add_vertex(
            v_tbl, realm=realm, vertex_id=500, payload={}, sync_sequence=False
        )
        int64_max = (1 << 63) - 1
        await sa_client_spr.add_vertex(v_tbl, realm=realm, vertex_id=int64_max, payload={})
        v_next = await sa_client_spr.add_vertex(v_tbl, realm=realm, payload={})
        assert int(v_next.id) == 101

        await sa_client_spr.add_edge(
            e_tbl, realm=realm, from_id=v1.id, to_id=v2.id, relation_type="r", edge_id=200
        )
        await sa_client_spr.add_edge(
            e_tbl, realm=realm, from_id=v1.id, to_id=v2.id, relation_type="r", edge_id=10
        )
        await sa_client_spr.add_edge(
            e_tbl,
            realm=realm,
            from_id=v1.id,
            to_id=v2.id,
            relation_type="r",
            edge_id=600,
            sync_sequence=False,
        )
        await sa_client_spr.add_edge(
            e_tbl, realm=realm, from_id=v1.id, to_id=v2.id, relation_type="r", edge_id=int64_max
        )
        e_next = await sa_client_spr.add_edge(
            e_tbl, realm=realm, from_id=v1.id, to_id=v2.id, relation_type="r"
        )
        assert int(e_next.id) == 201


class TestRLSAndRestrictedSequencePrivileges:
    """Space-scoped RLS and narrow sequence grants never rewind or fail explicit-ID writes."""

    @pytest.mark.asyncio
    async def test_space_scoped_rls_never_rewinds_sequence(
        self, pg_client_spr, clean_realm_spr, grantee
    ):
        realm = clean_realm_spr
        v_tbl = f"notes_{uuid.uuid4().hex[:8]}"
        await pg_client_spr.create_vertex_table(v_tbl, realm=realm, audited=False)

        # Space A advances sequence to 1000
        await pg_client_spr.add_vertex(
            v_tbl, realm=realm, vertex_id=1000, space="space_a", payload={"msg": "high"}
        )

        owner = await asyncpg.connect(DSN)
        try:
            await owner.execute(f'ALTER TABLE "{realm}"."{v_tbl}" ENABLE ROW LEVEL SECURITY')
            await owner.execute(f'ALTER TABLE "{realm}"."{v_tbl}" FORCE ROW LEVEL SECURITY')
            await grant_role(
                owner,
                grantee,
                [
                    f'GRANT USAGE ON SCHEMA "{realm}" TO "{grantee}"',
                    f'GRANT SELECT, INSERT ON "{realm}"."{v_tbl}" TO "{grantee}"',
                    f'GRANT SELECT, INSERT ON "{realm}"."{v_tbl}_data" TO "{grantee}"',
                    f'GRANT USAGE, SELECT, UPDATE ON ALL SEQUENCES IN SCHEMA "{realm}" TO "{grantee}"',
                    f"""CREATE POLICY space_b_only ON "{realm}"."{v_tbl}"
                        FOR ALL TO "{grantee}"
                        USING (space = 'space_b')
                        WITH CHECK (space = 'space_b')""",
                ],
            )

            # Grantee can only see space_b (where max id will be 5, not 1000)
            conn = await asyncpg.connect(DSN)
            try:
                client = AsyncPostGraph(connection_or_pool=conn, schema_per_realm=True)
                async with conn.transaction():
                    await conn.execute(f'SET LOCAL ROLE "{grantee}"')
                    await client.add_vertex(
                        v_tbl, realm=realm, vertex_id=5, space="space_b", payload={"msg": "low"}
                    )
            finally:
                await conn.close()

            # Sequence must still sit at 1000, so the next sequence-assigned write gets 1001
            v_auto = await pg_client_spr.add_vertex(
                v_tbl, realm=realm, space="space_a", payload={"msg": "next"}
            )
            assert int(v_auto.id) == 1001
        finally:
            await drop_role(owner, grantee)
            await owner.close()

    @pytest.mark.asyncio
    async def test_usage_only_or_update_without_select_sequence_grants(
        self, pg_client_spr, clean_realm_spr, grantee
    ):
        realm = clean_realm_spr
        v_tbl = f"items_{uuid.uuid4().hex[:8]}"
        await pg_client_spr.create_vertex_table(v_tbl, realm=realm, audited=False)
        await pg_client_spr.add_vertex(v_tbl, realm=realm, vertex_id=500, payload={})

        owner = await asyncpg.connect(DSN)
        try:
            # 1. Role with USAGE only on sequence (no UPDATE, no SELECT)
            await grant_role(
                owner,
                grantee,
                [
                    f'GRANT USAGE ON SCHEMA "{realm}" TO "{grantee}"',
                    f'GRANT SELECT, INSERT ON "{realm}"."{v_tbl}" TO "{grantee}"',
                    f'GRANT USAGE ON SEQUENCE "{realm}"."{v_tbl}_id_seq" TO "{grantee}"',
                ],
            )
            conn = await asyncpg.connect(DSN)
            try:
                client = AsyncPostGraph(connection_or_pool=conn, schema_per_realm=True)
                async with conn.transaction():
                    await conn.execute(f'SET LOCAL ROLE "{grantee}"')
                    v = await client.add_vertex(v_tbl, realm=realm, vertex_id=600, payload={})
                    assert int(v.id) == 600

                # 2. Role with USAGE + UPDATE (still no SELECT): must NOT rewind 500 -> 7
                await owner.execute(
                    f'GRANT UPDATE ON SEQUENCE "{realm}"."{v_tbl}_id_seq" TO "{grantee}"'
                )
                async with conn.transaction():
                    await conn.execute(f'SET LOCAL ROLE "{grantee}"')
                    v_low = await client.add_vertex(v_tbl, realm=realm, vertex_id=7, payload={})
                    assert int(v_low.id) == 7
            finally:
                await conn.close()

            # Sequence remained at 500 rather than rewinding to 7
            v_next = await pg_client_spr.add_vertex(v_tbl, realm=realm, payload={})
            assert int(v_next.id) == 501
        finally:
            await drop_role(owner, grantee)
            await owner.close()
