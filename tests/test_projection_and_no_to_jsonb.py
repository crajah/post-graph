"""Tests for zero-``to_jsonb`` read paths (Issue #5) and payload projection (Issue #6).

Exercises both ``AsyncPostGraph`` and ``SQLAlchemyPostGraph`` across shared-schema
and ``schema_per_realm=True`` topologies:

1. No read, search, or ``_data`` query ever emits ``to_jsonb(...)`` or queries
   ``information_schema.columns`` (``to_jsonb`` remains exclusively inside
   ``audit_trigger_func()`` for ``OLD``/``NEW`` trigger row snapshots).
2. ``_table_has_column`` / ``_table_has_embedding`` probes ``pg_attribute`` via
   ``to_regclass`` at most once per ``(realm, table, column)``, shares the cache
   across ``transaction()``, invalidates on ``drop_realm``, updates immediately
   when a non-vector table is re-declared with ``vector_dim``, and never caches
   ``False`` when ``to_regclass`` is ``NULL`` (non-existent table).
3. ``include_embedding`` (default ``True``) populates embeddings across every
   vertex, edge, search, and ``_data`` surface — including ``get_edge`` /
   ``get_edge_by_uuid`` and SQLAlchemy ``fulltext_search_*`` — and omits the
   vector column projection when ``include_embedding=False``.
4. ``payload_fields`` and ``exclude_payload_fields`` project or strip JSONB keys
   server-side across all 21 read surfaces (plus ``Vertex`` / ``Edge`` ``get_data*``
   methods), preserving explicit JSON ``null`` while never inventing absent keys.
5. Invalid keys, bare strings, and passing both ``payload_fields`` and
   ``exclude_payload_fields`` fail fast before issuing SQL; space-scoped RLS
   preserves full ``top_k`` visibility on ``vector_search(search_scope="data")``.
"""

import re
import uuid
from pathlib import Path

import asyncpg
import pytest
from conftest import DSN, RecordingConnection, drop_role, grant_role
from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine

from post_graph import (
    AsyncPostGraph,
    EdgeNotFoundError,
    PostGraphError,
    SQLAlchemyPostGraph,
    TableNotFoundError,
    VertexNotFoundError,
)


class TestNoReadPathToJsonbInSource:
    """Static guard: ``to_jsonb`` may only appear in ``audit_trigger_func``."""

    def test_only_audit_trigger_references_to_jsonb(self):
        pkg_dir = Path(__file__).resolve().parent.parent / "post_graph"
        hits = []
        for py_file in sorted(pkg_dir.rglob("*.py")):
            for lineno, line in enumerate(py_file.read_text(encoding="utf-8").splitlines(), 1):
                if "to_jsonb" in line:
                    stripped = line.strip()
                    if stripped.startswith("#"):
                        continue
                    hits.append((py_file.name, lineno, stripped))

        assert hits, "Expected audit_trigger_func() to retain to_jsonb(OLD)/to_jsonb(NEW)."
        allowed = {"v_old_row := to_jsonb(OLD);", "v_new_row := to_jsonb(NEW);"}
        unexpected = [(f, ln, s) for f, ln, s in hits if s not in allowed]
        assert not unexpected, f"Unexpected to_jsonb outside audit_trigger_func: {unexpected}"


class TestPgAttributeCacheAndRecordedSql:
    """Recorded SQL assertions on ``to_jsonb``, ``pg_attribute`` caching, and cache hygiene."""

    @pytest.mark.asyncio
    async def test_asyncpg_zero_to_jsonb_and_cached_pg_attribute_across_all_reads(
        self, pg_client_spr, clean_realm_spr, has_pgvector
    ):
        if not has_pgvector:
            pytest.skip("pgvector extension unavailable")

        realm = clean_realm_spr
        v_vec = f"vv_{uuid.uuid4().hex[:8]}"
        v_plain = f"vp_{uuid.uuid4().hex[:8]}"
        e_vec = f"ev_{uuid.uuid4().hex[:8]}"

        await pg_client_spr.create_vertex_table(v_vec, realm=realm, vector_dim=4, audited=False)
        await pg_client_spr.create_vertex_table(v_plain, realm=realm, audited=False)
        await pg_client_spr.create_edge_table(
            e_vec,
            from_vertex_table=v_vec,
            to_vertex_table=v_vec,
            realm=realm,
            vector_dim=4,
            audited=False,
        )

        raw = await asyncpg.connect(DSN)
        try:
            rec = RecordingConnection(raw)
            client = AsyncPostGraph(connection_or_pool=rec, schema_per_realm=True)

            rec.queries.clear()
            a = await client.add_vertex(
                v_vec,
                realm=realm,
                payload={"name": "alpha text", "blob": "x" * 2048},
                embedding=[1.0, 0.0, 0.0, 0.0],
            )
            b = await client.add_vertex(
                v_vec,
                realm=realm,
                payload={"name": "beta text", "blob": "y" * 2048},
                embedding=[0.0, 1.0, 0.0, 0.0],
            )
            await client.add_vertex(v_plain, realm=realm, payload={"name": "plain node"})
            edge = await client.add_edge(
                e_vec,
                realm=realm,
                from_id=a.id,
                to_id=b.id,
                relation_type="links",
                payload={"note": "connected alpha beta", "raw": "z" * 2048},
                embedding=[0.5, 0.5, 0.0, 0.0],
            )
            drec = await client.add_vertex_data(
                v_vec,
                realm=realm,
                vertex_id=a.id,
                payload={"rev": 1, "raw": "d" * 2048},
                embedding=[0.9, 0.1, 0.0, 0.0],
            )
            edrec = await client.add_edge_data(
                e_vec,
                realm=realm,
                edge_id=edge.id,
                payload={"erev": 1},
                embedding=[0.4, 0.6, 0.0, 0.0],
            )

            # Warm caches once across tables, then verify zero subsequent catalog hits.
            await client.get_vertex(v_vec, realm, a.id)
            await client.get_vertex(v_plain, realm, "1")
            await client.get_edge(e_vec, realm, edge.id)
            await client.get_vertex_data(v_vec, realm, a.id)
            await client.get_edge_data(e_vec, realm, edge.id)

            for q in rec.queries:
                assert "to_jsonb" not in q.lower(), f"Forbidden to_jsonb in query: {q}"
                assert "information_schema.columns" not in q.lower(), (
                    f"Forbidden information_schema.columns lookup: {q}"
                )

            rec.queries.clear()
            await client.get_vertex(v_vec, realm, a.id, payload_fields=["name"])
            await client.get_vertex_by_uuid(v_vec, realm, a.uuid, exclude_payload_fields=["blob"])
            await client.get_vertices(v_vec, realm, include_embedding=False)
            await client.get_vertices_multi_realm(v_vec, [realm], payload_fields=["name"])
            await client.newest_vertex(v_vec, realm, payload_fields=["name"])
            await client.find_vertices(v_vec, realm, filters={"name": "alpha text"})
            await client.find_vertices(v_plain, realm)
            await client.get_edge(e_vec, realm, edge.id, payload_fields=["note"])
            await client.get_edge_by_uuid(e_vec, realm, edge.uuid, exclude_payload_fields=["raw"])
            await client.get_edges(e_vec, realm)
            await client.get_edges_multi_realm(e_vec, [realm])
            await client.find_edges(e_vec, realm, relation_type="links")
            await client.vector_search(v_vec, realm, [1.0, 0.0, 0.0, 0.0], search_scope="main")
            await client.vector_search(v_vec, realm, [1.0, 0.0, 0.0, 0.0], search_scope="data")
            await client.vector_search(v_vec, realm, [1.0, 0.0, 0.0, 0.0], search_scope="both")
            await client.vector_search_edges(e_vec, realm, [0.5, 0.5, 0.0, 0.0])
            await client.fulltext_search_vertices(v_vec, realm, "alpha")
            await client.fulltext_search_edges(e_vec, realm, "connected")
            await client.get_vertex_data(v_vec, realm, a.id)
            await client.get_latest_vertex_data(v_vec, realm, a.id)
            await client.get_vertex_data_by_id(v_vec, realm, drec.data_id)
            await client.get_edge_data(e_vec, realm, edge.id)
            await client.get_latest_edge_data(e_vec, realm, edge.id)
            await client.get_edge_data_by_id(e_vec, realm, edrec.data_id)

            # Inside transaction(), scoped client shares _column_cache with parent.
            async with client.transaction(user_id="tester") as tx:
                await tx.get_vertex(v_vec, realm, a.id)
                await tx.get_vertex(v_plain, realm, "1")

            for q in rec.queries:
                low = q.lower()
                assert "to_jsonb" not in low, f"Forbidden to_jsonb in query: {q}"
                assert "pg_attribute" not in low, f"Uncached pg_attribute lookup on warm path: {q}"
                assert "information_schema" not in low, f"Unexpected information_schema query: {q}"
                if "from best" in low or "with best" in low:
                    assert not re.search(r"group\s+by[^)]*payload", low), (
                        f"vector_search(data) must not GROUP BY payload: {q}"
                    )
        finally:
            await raw.close()

    @pytest.mark.asyncio
    async def test_sqlalchemy_zero_to_jsonb_and_cached_pg_attribute(
        self, sa_clean_realm_spr, has_pgvector
    ):
        if not has_pgvector:
            pytest.skip("pgvector extension unavailable")

        realm = sa_clean_realm_spr
        sa_dsn = DSN.replace("postgresql://", "postgresql+asyncpg://", 1)
        engine = create_async_engine(sa_dsn)
        recorded: list[str] = []

        @event.listens_for(engine.sync_engine, "before_cursor_execute")
        def _capture(conn, cursor, statement, parameters, context, executemany):
            recorded.append(" ".join(statement.split()))

        try:
            client = SQLAlchemyPostGraph(engine, schema_per_realm=True)
            v_tbl = f"sv_{uuid.uuid4().hex[:8]}"
            e_tbl = f"se_{uuid.uuid4().hex[:8]}"
            await client.create_vertex_table(v_tbl, realm=realm, vector_dim=4, audited=False)
            await client.create_edge_table(
                e_tbl,
                from_vertex_table=v_tbl,
                to_vertex_table=v_tbl,
                realm=realm,
                vector_dim=4,
                audited=False,
            )

            recorded.clear()
            v1 = await client.add_vertex(
                v_tbl, realm=realm, payload={"title": "doc one"}, embedding=[1.0, 0.0, 0.0, 0.0]
            )
            v2 = await client.add_vertex(
                v_tbl, realm=realm, payload={"title": "doc two"}, embedding=[0.0, 1.0, 0.0, 0.0]
            )
            e1 = await client.add_edge(
                e_tbl,
                realm=realm,
                from_id=v1.id,
                to_id=v2.id,
                relation_type="rel",
                payload={"summary": "edge summary"},
                embedding=[0.5, 0.5, 0.0, 0.0],
            )
            d1 = await client.add_vertex_data(
                v_tbl,
                realm=realm,
                vertex_id=v1.id,
                payload={"step": 1},
                embedding=[0.8, 0.2, 0.0, 0.0],
            )

            await client.get_vertex(v_tbl, realm, v1.id)
            await client.get_vertices(v_tbl, realm)
            await client.get_vertices_multi_realm(v_tbl, [realm])
            await client.find_vertices(v_tbl, realm, payload_fields=["title"])
            await client.get_edge(e_tbl, realm, e1.id)
            await client.get_edges(e_tbl, realm)
            await client.get_edges_multi_realm(e_tbl, [realm])
            await client.find_edges(e_tbl, realm, exclude_payload_fields=["summary"])
            await client.vector_search(v_tbl, realm, [1.0, 0.0, 0.0, 0.0], search_scope="data")
            await client.vector_search(v_tbl, realm, [1.0, 0.0, 0.0, 0.0], search_scope="both")
            await client.vector_search_edges(e_tbl, realm, [0.5, 0.5, 0.0, 0.0])
            await client.fulltext_search_vertices(v_tbl, realm, "doc")
            await client.fulltext_search_edges(e_tbl, realm, "summary")
            await client.get_vertex_data(v_tbl, realm, v1.id)
            await client.get_vertex_data_by_id(v_tbl, realm, d1.data_id)

            for stmt in recorded:
                low = stmt.lower()
                assert "to_jsonb" not in low, f"Forbidden to_jsonb in SQLAlchemy SQL: {stmt}"
                assert "information_schema.columns" not in low, (
                    f"Forbidden information_schema.columns in SQLAlchemy SQL: {stmt}"
                )
                assert "to_regclass" not in low, (
                    f"create_vertex_table/create_edge_table should pre-seed _column_cache: {stmt}"
                )
        finally:
            await engine.dispose()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("backend", ["asyncpg", "sqlalchemy"])
    async def test_missing_table_never_poisons_cache_and_redeclare_with_vector_updates_cache(
        self, backend, pg_client_spr, sa_client_spr, clean_realm_spr, has_pgvector
    ):
        if not has_pgvector:
            pytest.skip("pgvector extension unavailable")

        realm = clean_realm_spr
        client = pg_client_spr if backend == "asyncpg" else sa_client_spr
        missing_tbl = f"late_{uuid.uuid4().hex[:8]}"

        # 1. Reading a non-existent table raises TableNotFoundError and leaves _column_cache clean.
        with pytest.raises(TableNotFoundError):
            await client.get_vertices(missing_tbl, realm)
        assert (realm, missing_tbl, "embedding") not in client._col_cache_dict()

        # Creating that table immediately afterwards with vector_dim=4 must return embeddings.
        await client.create_vertex_table(missing_tbl, realm=realm, vector_dim=4, audited=False)
        v = await client.add_vertex(
            missing_tbl, realm=realm, payload={"k": 1}, embedding=[0.25, 0.5, 0.75, 1.0]
        )
        got = await client.get_vertex(missing_tbl, realm, v.id)
        assert got is not None and got.embedding == pytest.approx([0.25, 0.5, 0.75, 1.0])

        # 2. Creating a non-vector table caches False on read; re-declaring with vector_dim=4 flips to True.
        upgrade_tbl = f"upg_{uuid.uuid4().hex[:8]}"
        await client.create_vertex_table(upgrade_tbl, realm=realm, audited=False)
        u1 = await client.add_vertex(upgrade_tbl, realm=realm, payload={"k": "before"})
        got_u1 = await client.get_vertex(upgrade_tbl, realm, u1.id)
        assert got_u1 is not None and got_u1.embedding is None
        assert client._lookup_cached_column(realm, upgrade_tbl, "embedding") is False

        await client.create_vertex_table(upgrade_tbl, realm=realm, vector_dim=4, audited=False)
        assert client._lookup_cached_column(realm, upgrade_tbl, "embedding") is True
        u2 = await client.add_vertex(
            upgrade_tbl, realm=realm, payload={"k": "after"}, embedding=[1.0, 2.0, 3.0, 4.0]
        )
        got_u2 = await client.get_vertex(upgrade_tbl, realm, u2.id)
        assert got_u2 is not None and got_u2.embedding == pytest.approx([1.0, 2.0, 3.0, 4.0])

        # 3. drop_realm evicts all cached column entries for that realm.
        sub_realm = f"drop_{uuid.uuid4().hex[:8]}"
        await client.create_vertex_table("items", realm=sub_realm, vector_dim=4, audited=False)
        assert client._lookup_cached_column(sub_realm, "items", "embedding") is True
        assert await client.drop_realm(sub_realm) is True
        assert client._lookup_cached_column(sub_realm, "items", "embedding") is None


@pytest.mark.parametrize(
    "backend,spr",
    [
        ("asyncpg", False),
        ("asyncpg", True),
        ("sqlalchemy", False),
        ("sqlalchemy", True),
    ],
)
class TestPayloadProjectionAndEmbeddingControlAllSurfaces:
    """Full-surface parity for ``payload_fields``, ``exclude_payload_fields``, and ``include_embedding``."""

    def _pick_client(self, backend, spr, pg_client, pg_client_spr, sa_client, sa_client_spr):
        if backend == "asyncpg":
            return pg_client_spr if spr else pg_client
        return sa_client_spr if spr else sa_client

    @pytest.mark.asyncio
    async def test_vertex_and_edge_projections_preserve_json_null_and_omit_absent(
        self,
        backend,
        spr,
        pg_client,
        pg_client_spr,
        sa_client,
        sa_client_spr,
        clean_realm,
        clean_realm_spr,
        has_pgvector,
    ):
        if not has_pgvector:
            pytest.skip("pgvector extension unavailable")

        client = self._pick_client(backend, spr, pg_client, pg_client_spr, sa_client, sa_client_spr)
        realm = clean_realm_spr if spr else clean_realm
        realm2 = f"{realm}_b"
        v_tbl = f"pv_{uuid.uuid4().hex[:8]}"
        e_tbl = f"pe_{uuid.uuid4().hex[:8]}"

        try:
            await client.create_vertex_table(v_tbl, realm=realm, vector_dim=4, audited=False)
            await client.create_edge_table(
                e_tbl,
                from_vertex_table=v_tbl,
                to_vertex_table=v_tbl,
                realm=realm,
                vector_dim=4,
                audited=False,
            )
            if spr:
                # Second realm deliberately WITHOUT vector_dim to exercise mixed-schema UNION ALL!
                await client.create_vertex_table(v_tbl, realm=realm2, audited=False)
                await client.create_edge_table(
                    e_tbl,
                    from_vertex_table=v_tbl,
                    to_vertex_table=v_tbl,
                    realm=realm2,
                    audited=False,
                )

            huge_blob = "X" * 40_000  # Exceeds PostgreSQL 2 KB TOAST threshold
            v1 = await client.add_vertex(
                v_tbl,
                realm=realm,
                payload={
                    "title": "First Document",
                    "ended_at": None,  # Explicit JSON null — must survive projection
                    "seq": 10,
                    "kind": "doc",
                    "transcript": huge_blob,
                },
                embedding=[1.0, 0.0, 0.0, 0.0],
            )
            v2 = await client.add_vertex(
                v_tbl,
                realm=realm,
                payload={
                    "title": "Second Document",
                    "seq": 20,
                    "kind": "doc",
                    "transcript": huge_blob,
                },
                embedding=[0.0, 1.0, 0.0, 0.0],
            )
            v_r2 = await client.add_vertex(
                v_tbl,
                realm=realm2,
                payload={"title": "Other Realm Doc", "ended_at": None, "transcript": huge_blob},
            )

            e1 = await client.add_edge(
                e_tbl,
                realm=realm,
                from_id=v1.id,
                to_id=v2.id,
                relation_type="cites",
                payload={
                    "label": "Primary Citation",
                    "revoked_at": None,
                    "weight": 5,
                    "raw_context": huge_blob,
                },
                embedding=[0.6, 0.8, 0.0, 0.0],
            )
            await client.add_edge(
                e_tbl,
                realm=realm2,
                from_id=v_r2.id,
                to_id=v_r2.id,
                relation_type="cites",
                payload={"label": "Realm2 Citation", "raw_context": huge_blob},
            )

            d1 = await client.add_vertex_data(
                v_tbl,
                realm=realm,
                vertex_id=v1.id,
                payload={"version": 1, "closed_at": None, "snapshot": huge_blob},
                embedding=[0.9, 0.1, 0.0, 0.0],
            )
            ed1 = await client.add_edge_data(
                e_tbl,
                realm=realm,
                edge_id=e1.id,
                payload={"eversion": 1, "closed_at": None, "snapshot": huge_blob},
                embedding=[0.7, 0.3, 0.0, 0.0],
            )

            inc_v = ["title", "ended_at", "absent_key", "title"]  # includes duplicate & absent key
            expected_v1_inc = {"title": "First Document", "ended_at": None}
            expected_v2_inc = {"title": "Second Document"}

            # 1. get_vertex (by id, fqid, and uuid) & get_vertex_by_uuid
            gv = await client.get_vertex(
                v_tbl, realm, v1.id, payload_fields=inc_v, include_embedding=False
            )
            assert gv is not None
            assert gv.payload == expected_v1_inc
            assert "ended_at" in gv.payload and gv.payload["ended_at"] is None
            assert "absent_key" not in gv.payload
            assert gv.embedding is None

            gv_uuid = await client.get_vertex(
                v_tbl, realm, v1.uuid, exclude_payload_fields=["transcript", "seq", "kind"]
            )
            assert gv_uuid is not None
            assert gv_uuid.payload == expected_v1_inc
            assert gv_uuid.embedding == pytest.approx([1.0, 0.0, 0.0, 0.0])

            gv_empty = await client.get_vertex_by_uuid(
                v_tbl, realm, v1.uuid, payload_fields=[], include_embedding=False
            )
            assert gv_empty is not None and gv_empty.payload == {} and gv_empty.embedding is None

            # 2. get_vertices & newest_vertex
            gvs = await client.get_vertices(
                v_tbl, realm, payload_fields=inc_v, include_embedding=False
            )
            assert [x.payload for x in gvs] == [expected_v1_inc, expected_v2_inc]
            assert all(x.embedding is None for x in gvs)

            nv = await client.newest_vertex(
                v_tbl, realm, exclude_payload_fields=["transcript"], include_embedding=False
            )
            assert nv is not None and nv.id == v2.id
            assert nv.payload == {"title": "Second Document", "seq": 20, "kind": "doc"}
            assert nv.embedding is None

            # 3. get_vertices_multi_realm (including mixed vector + non-vector schemas in SPR)
            mr_vs = await client.get_vertices_multi_realm(
                v_tbl, [realm, realm2], payload_fields=["title", "ended_at"]
            )
            assert len(mr_vs) == 3
            by_title = {x.payload["title"]: x for x in mr_vs}
            assert by_title["First Document"].payload == expected_v1_inc
            assert by_title["First Document"].embedding == pytest.approx([1.0, 0.0, 0.0, 0.0])
            assert by_title["Second Document"].payload == expected_v2_inc
            assert by_title["Other Realm Doc"].payload == {
                "title": "Other Realm Doc",
                "ended_at": None,
            }
            if spr:
                assert by_title["Other Realm Doc"].embedding is None

            # 4. find_vertices: filter & sort on unprojected keys while projecting subset
            fvs = await client.find_vertices(
                v_tbl,
                realm,
                filters={"kind": "doc"},
                where=[("seq", ">=", 10)],
                order_by="seq",
                descending=True,
                payload_fields=inc_v,
                include_embedding=False,
            )
            assert [x.id for x in fvs] == [v2.id, v1.id]
            assert [x.payload for x in fvs] == [expected_v2_inc, expected_v1_inc]
            assert all(x.embedding is None for x in fvs)

            # 5. get_edge (by id, fqid, uuid), get_edge_by_uuid, get_edges, get_edges_multi_realm, find_edges
            inc_e = ["label", "revoked_at", "missing_edge_key"]
            expected_e1_inc = {"label": "Primary Citation", "revoked_at": None}

            ge = await client.get_edge(e_tbl, realm, e1.id, payload_fields=inc_e)
            assert ge is not None
            assert ge.payload == expected_e1_inc
            assert ge.embedding == pytest.approx([0.6, 0.8, 0.0, 0.0])

            ge_no_emb = await client.get_edge(
                e_tbl,
                realm,
                e1.uuid,
                exclude_payload_fields=["raw_context", "weight"],
                include_embedding=False,
            )
            assert ge_no_emb is not None
            assert ge_no_emb.payload == expected_e1_inc
            assert ge_no_emb.embedding is None

            ges = await client.get_edges(
                e_tbl, realm, payload_fields=inc_e, include_embedding=False
            )
            assert len(ges) == 1 and ges[0].payload == expected_e1_inc and ges[0].embedding is None

            mr_es = await client.get_edges_multi_realm(
                e_tbl, [realm, realm2], exclude_payload_fields=["raw_context", "weight"]
            )
            assert len(mr_es) == 2
            e_by_label = {x.payload["label"]: x for x in mr_es}
            assert e_by_label["Primary Citation"].payload == expected_e1_inc
            assert e_by_label["Primary Citation"].embedding == pytest.approx([0.6, 0.8, 0.0, 0.0])
            assert e_by_label["Realm2 Citation"].payload == {"label": "Realm2 Citation"}

            fes = await client.find_edges(
                e_tbl,
                realm,
                where=[("weight", ">", 1)],
                payload_fields=inc_e,
                include_embedding=False,
            )
            assert len(fes) == 1 and fes[0].payload == expected_e1_inc and fes[0].embedding is None

            # 6. vector_search (main, data, both) & vector_search_edges
            for scope in ("main", "data", "both"):
                vs_hits = await client.vector_search(
                    v_tbl,
                    realm,
                    [1.0, 0.0, 0.0, 0.0],
                    top_k=2,
                    search_scope=scope,
                    payload_fields=inc_v,
                    include_embedding=False,
                )
                assert vs_hits and vs_hits[0][0].id == v1.id
                assert vs_hits[0][0].payload == expected_v1_inc
                assert vs_hits[0][0].embedding is None
                assert vs_hits[0][0].uuid == v1.uuid

            vs_with_emb = await client.vector_search(
                v_tbl,
                realm,
                [1.0, 0.0, 0.0, 0.0],
                top_k=1,
                search_scope="data",
                exclude_payload_fields=["transcript", "seq", "kind"],
            )
            assert vs_with_emb[0][0].payload == expected_v1_inc
            assert vs_with_emb[0][0].embedding == pytest.approx([1.0, 0.0, 0.0, 0.0])

            ve_hits = await client.vector_search_edges(
                e_tbl,
                realm,
                [0.6, 0.8, 0.0, 0.0],
                top_k=1,
                payload_fields=inc_e,
                include_embedding=False,
            )
            assert len(ve_hits) == 1
            assert ve_hits[0][0].payload == expected_e1_inc
            assert ve_hits[0][0].embedding is None

            # 7. fulltext_search_vertices & fulltext_search_edges (searches unprojected fields too!)
            ft_v = await client.fulltext_search_vertices(
                v_tbl,
                realm,
                "First",
                fields=["title"],
                payload_fields=inc_v,
            )
            assert len(ft_v) == 1
            assert ft_v[0].payload == expected_v1_inc
            assert ft_v[0].embedding == pytest.approx([1.0, 0.0, 0.0, 0.0])

            ft_v_no_emb = await client.fulltext_search_vertices(
                v_tbl,
                realm,
                "First",
                exclude_payload_fields=["transcript", "seq", "kind"],
                include_embedding=False,
            )
            assert len(ft_v_no_emb) == 1
            assert ft_v_no_emb[0].payload == expected_v1_inc
            assert ft_v_no_emb[0].embedding is None

            ft_e = await client.fulltext_search_edges(
                e_tbl,
                realm,
                "Citation",
                fields=["label"],
                payload_fields=inc_e,
            )
            assert len(ft_e) == 1
            assert ft_e[0].payload == expected_e1_inc
            assert ft_e[0].embedding == pytest.approx([0.6, 0.8, 0.0, 0.0])

            # 8. Data history surfaces (client methods + Vertex/Edge model helpers)
            vd_list = await v1.get_data(
                payload_fields=["version", "closed_at", "absent"], include_embedding=False
            )
            assert len(vd_list) == 1
            assert vd_list[0].payload == {"version": 1, "closed_at": None}
            assert vd_list[0].embedding is None

            vd_latest = await v1.get_latest_data(exclude_payload_fields=["snapshot"])
            assert vd_latest is not None
            assert vd_latest.payload == {"version": 1, "closed_at": None}
            assert vd_latest.embedding == pytest.approx([0.9, 0.1, 0.0, 0.0])

            vd_by_id = await v1.get_data_by_id(
                d1.data_id, payload_fields=["version"], include_embedding=False
            )
            assert vd_by_id is not None
            assert vd_by_id.payload == {"version": 1}
            assert vd_by_id.embedding is None

            ed_list = await e1.get_data(
                payload_fields=["eversion", "closed_at"], include_embedding=False
            )
            assert len(ed_list) == 1
            assert ed_list[0].payload == {"eversion": 1, "closed_at": None}
            assert ed_list[0].embedding is None

            ed_latest = await e1.get_latest_data(exclude_payload_fields=["snapshot"])
            assert ed_latest is not None
            assert ed_latest.payload == {"eversion": 1, "closed_at": None}
            assert ed_latest.embedding == pytest.approx([0.7, 0.3, 0.0, 0.0])

            ed_by_id = await e1.get_data_by_id(
                ed1.data_id, payload_fields=[], include_embedding=False
            )
            assert ed_by_id is not None and ed_by_id.payload == {} and ed_by_id.embedding is None
        finally:
            if spr:
                try:
                    await client.drop_realm(realm2)
                except Exception:
                    pass
            else:
                try:
                    await client.delete_realm(realm2)
                except Exception:
                    pass


class TestValidationErrorsAndRlsTopK:
    """Input validation, non-vector table search parity, strict errors, and RLS top-k preservation."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("backend", ["asyncpg", "sqlalchemy"])
    async def test_projection_validation_and_non_vector_search_parity(
        self, backend, pg_client_spr, sa_client_spr, clean_realm_spr
    ):
        realm = clean_realm_spr
        client = pg_client_spr if backend == "asyncpg" else sa_client_spr
        v_tbl = f"nv_{uuid.uuid4().hex[:8]}"
        e_tbl = f"ne_{uuid.uuid4().hex[:8]}"

        await client.create_vertex_table(v_tbl, realm=realm, audited=False)
        await client.create_edge_table(
            e_tbl, from_vertex_table=v_tbl, to_vertex_table=v_tbl, realm=realm, audited=False
        )

        # Passing embedding on a non-vector table succeeds cleanly on BOTH backends.
        v = await client.add_vertex(
            v_tbl, realm=realm, payload={"name": "plain"}, embedding=[1.0, 2.0]
        )
        assert v.embedding is None
        e = await client.add_edge(
            e_tbl,
            realm=realm,
            from_id=v.id,
            to_id=v.id,
            relation_type="self",
            payload={"k": "v"},
            embedding=[1.0, 2.0],
        )
        assert e.embedding is None

        # vector_search / vector_search_edges on a non-vector table returns [] (not TableNotFoundError)
        assert await client.vector_search(v_tbl, realm, [1.0, 2.0]) == []
        assert await client.vector_search_edges(e_tbl, realm, [1.0, 2.0]) == []

        # Missing table still raises TableNotFoundError
        with pytest.raises(TableNotFoundError):
            await client.vector_search("no_such_vertex_table", realm, [1.0, 2.0])
        with pytest.raises(TableNotFoundError):
            await client.vector_search_edges("no_such_edge_table", realm, [1.0, 2.0])

        # strict=True still raises VertexNotFoundError / EdgeNotFoundError with projections
        with pytest.raises(VertexNotFoundError):
            await client.get_vertex(v_tbl, realm, "999999", strict=True, payload_fields=["name"])
        with pytest.raises(EdgeNotFoundError):
            await client.get_edge(e_tbl, realm, "999999", strict=True, payload_fields=["k"])

        # Mutually exclusive payload_fields + exclude_payload_fields -> PostGraphError
        for fn in (
            lambda: client.find_vertices(
                v_tbl, realm, payload_fields=["a"], exclude_payload_fields=["b"]
            ),
            lambda: client.get_vertices(
                v_tbl, realm, payload_fields=["a"], exclude_payload_fields=["b"]
            ),
            lambda: client.get_vertex(
                v_tbl, realm, v.id, payload_fields=["a"], exclude_payload_fields=["b"]
            ),
            lambda: client.find_edges(
                e_tbl, realm, payload_fields=["a"], exclude_payload_fields=["b"]
            ),
        ):
            with pytest.raises(PostGraphError, match="not both"):
                await fn()

        # Bare string rejected with PostGraphError (never silently split into characters!)
        with pytest.raises(PostGraphError, match="not a bare string"):
            await client.find_vertices(v_tbl, realm, payload_fields="name")
        with pytest.raises(PostGraphError, match="not a bare string"):
            await client.get_vertices(v_tbl, realm, exclude_payload_fields="name")

        # Invalid payload key rejected with ValueError
        for bad_key in ("", "bad'key", "a; DROP TABLE", "space key"):
            with pytest.raises(ValueError, match="Invalid payload key"):
                await client.find_vertices(v_tbl, realm, payload_fields=[bad_key])
            with pytest.raises(ValueError, match="Invalid payload key"):
                await client.get_vertices(v_tbl, realm, exclude_payload_fields=[bad_key])

    @pytest.mark.asyncio
    async def test_rls_role_reads_and_vector_search_data_preserves_top_k(
        self, pg_client_spr, clean_realm_spr, grantee, has_pgvector
    ):
        if not has_pgvector:
            pytest.skip("pgvector extension unavailable")

        realm = clean_realm_spr
        v_tbl = f"rls_{uuid.uuid4().hex[:8]}"
        await pg_client_spr.create_vertex_table(v_tbl, realm=realm, vector_dim=4, audited=False)

        # Insert closer match in space_a (hidden by RLS on main table) and visible match in space_b
        hidden = await pg_client_spr.add_vertex(
            v_tbl,
            realm=realm,
            space="space_a",
            payload={"title": "Hidden", "secret": "s1"},
            embedding=[1.0, 0.0, 0.0, 0.0],
        )
        await pg_client_spr.add_vertex_data(
            v_tbl, realm=realm, vertex_id=hidden.id, payload={"rev": 1}, embedding=[1.0, 0.0, 0.0, 0.0]
        )

        visible = await pg_client_spr.add_vertex(
            v_tbl,
            realm=realm,
            space="space_b",
            payload={"title": "Visible", "ended_at": None, "secret": "s2"},
            embedding=[0.8, 0.2, 0.0, 0.0],
        )
        await pg_client_spr.add_vertex_data(
            v_tbl, realm=realm, vertex_id=visible.id, payload={"rev": 1}, embedding=[0.8, 0.2, 0.0, 0.0]
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
                    f'GRANT SELECT ON "{realm}"."{v_tbl}" TO "{grantee}"',
                    f'GRANT SELECT ON "{realm}"."{v_tbl}_data" TO "{grantee}"',
                    f"""CREATE POLICY space_b_read ON "{realm}"."{v_tbl}"
                        FOR SELECT TO "{grantee}"
                        USING (space = 'space_b')""",
                ],
            )

            conn = await asyncpg.connect(DSN)
            try:
                role_client = AsyncPostGraph(connection_or_pool=conn, schema_per_realm=True)
                async with conn.transaction():
                    await conn.execute(f'SET LOCAL ROLE "{grantee}"')
                    rows = await role_client.find_vertices(
                        v_tbl,
                        realm=realm,
                        payload_fields=["title", "ended_at"],
                        include_embedding=False,
                    )
                    assert len(rows) == 1
                    assert rows[0].id == visible.id
                    assert rows[0].payload == {"title": "Visible", "ended_at": None}
                    assert rows[0].embedding is None

                    # top_k=1 on search_scope="data" must still return the visible row
                    # even though hidden had smaller distance in _data!
                    hits = await role_client.vector_search(
                        v_tbl,
                        realm=realm,
                        query_vector=[1.0, 0.0, 0.0, 0.0],
                        top_k=1,
                        search_scope="data",
                        exclude_payload_fields=["secret"],
                    )
                    assert len(hits) == 1
                    assert hits[0][0].id == visible.id
                    assert hits[0][0].payload == {"title": "Visible", "ended_at": None}
            finally:
                await conn.close()
        finally:
            await drop_role(owner, grantee)
            await owner.close()
