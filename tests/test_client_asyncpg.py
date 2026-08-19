"""Integration tests for AsyncPostGraph against a live PostgreSQL instance.

Set POST_GRAPH_TEST_DSN to a PostgreSQL connection string.
Tests are skipped when the database is unreachable.
"""

import json

import pytest
from post_graph import (
    AsyncPostGraph,
    RESERVED_SPACE_ALL,
    ReservedSpaceError,
    TableExistsError,
    TableNotFoundError,
    VertexNotFoundError,
    CyclicReferenceError,
)
from post_graph.models import Vertex, Edge, DataRecord

# ---------------------------------------------------------------------------
# Identifier validation (no DB needed)
# ---------------------------------------------------------------------------

class TestValidateIdentifier:
    def test_valid(self):
        client = AsyncPostGraph(dsn="unused")
        for name in ("people", "my_table", "_private", "T1"):
            client._validate_identifier(name)

    def test_empty_raises(self):
        client = AsyncPostGraph(dsn="unused")
        with pytest.raises(ValueError):
            client._validate_identifier("")

    def test_none_raises(self):
        client = AsyncPostGraph(dsn="unused")
        with pytest.raises(ValueError):
            client._validate_identifier(None)

    def test_special_chars_raise(self):
        client = AsyncPostGraph(dsn="unused")
        for bad in ("my-table", "drop;--", "my table", "1table", "tbl$"):
            with pytest.raises(ValueError):
                client._validate_identifier(bad)


# ---------------------------------------------------------------------------
# _padded_date_sql (static, no DB needed)
# ---------------------------------------------------------------------------

class TestPaddedDateSql:
    def test_output_format(self):
        sql = AsyncPostGraph._padded_date_sql("col")
        assert "split_part" in sql
        assert "'01'" in sql


# ---------------------------------------------------------------------------
# Edge filter SQL builder (no DB needed)
# ---------------------------------------------------------------------------

class TestEdgeFilterSql:
    def _client(self):
        return AsyncPostGraph(dsn="unused")

    def test_empty_filters(self):
        sql, params = self._client()._edge_filter_sql(
            relation_types=None, as_of=None, payload_null_keys=None,
            space=None, valid_from_key="valid_from", valid_to_key="valid_to",
            first_param=3,
        )
        assert sql == ""
        assert params == []

    def test_relation_types(self):
        sql, params = self._client()._edge_filter_sql(
            relation_types=["knows", "works_at"], as_of=None,
            payload_null_keys=None, space=None,
            valid_from_key="vf", valid_to_key="vt", first_param=5,
        )
        assert "relation_type = ANY($5::text[])" in sql
        assert params == [["knows", "works_at"]]

    def test_space_filter(self):
        sql, params = self._client()._edge_filter_sql(
            relation_types=None, as_of=None, payload_null_keys=None,
            space="prod", valid_from_key="vf", valid_to_key="vt",
            first_param=5,
        )
        assert "space = $5" in sql
        assert params == ["prod"]

    def test_space_all_is_ignored(self):
        sql, params = self._client()._edge_filter_sql(
            relation_types=None, as_of=None, payload_null_keys=None,
            space=RESERVED_SPACE_ALL, valid_from_key="vf", valid_to_key="vt",
            first_param=5,
        )
        assert sql == ""
        assert params == []

    def test_as_of(self):
        sql, params = self._client()._edge_filter_sql(
            relation_types=None, as_of="2020", payload_null_keys=None,
            space=None, valid_from_key="valid_from", valid_to_key="valid_to",
            first_param=5,
        )
        assert "$5" in sql
        assert "valid_from" in sql
        assert "valid_to" in sql
        assert params == ["2020"]

    def test_payload_null_keys(self):
        sql, params = self._client()._edge_filter_sql(
            relation_types=None, as_of=None,
            payload_null_keys=["superseded_by", "deleted"],
            space=None, valid_from_key="vf", valid_to_key="vt",
            first_param=5,
        )
        assert "payload->>'superseded_by' IS NULL" in sql
        assert "payload->>'deleted' IS NULL" in sql
        assert params == []

    def test_combined_filters(self):
        sql, params = self._client()._edge_filter_sql(
            relation_types=["rel_a"], as_of="2023-06",
            payload_null_keys=["closed"], space="staging",
            valid_from_key="vf", valid_to_key="vt", first_param=5,
        )
        assert "$5" in sql  # relation_types
        assert "$6" in sql  # space
        assert "$7" in sql  # as_of
        assert len(params) == 3

    def test_payload_null_keys_rejects_bad_identifier(self):
        with pytest.raises(ValueError):
            self._client()._edge_filter_sql(
                relation_types=None, as_of=None,
                payload_null_keys=["drop;--"],
                space=None, valid_from_key="vf", valid_to_key="vt",
                first_param=5,
            )


# ---------------------------------------------------------------------------
# Table reference helpers (no DB needed)
# ---------------------------------------------------------------------------

class TestGetTableRef:
    def test_simple(self):
        c = AsyncPostGraph(dsn="unused")
        assert c._get_table_ref("people") == '"people"'

    def test_schema_per_realm(self):
        c = AsyncPostGraph(dsn="unused", schema_per_realm=True)
        assert c._get_table_ref("people", realm="acme") == '"acme"."people"'

    def test_schema_per_realm_missing_realm(self):
        c = AsyncPostGraph(dsn="unused", schema_per_realm=True)
        from post_graph.errors import PostGraphError
        with pytest.raises(PostGraphError):
            c._get_table_ref("people")

    def test_invalid_table_name_raises(self):
        c = AsyncPostGraph(dsn="unused")
        with pytest.raises(ValueError):
            c._get_table_ref("bad-name")


# ---------------------------------------------------------------------------
# Client construction (no DB needed)
# ---------------------------------------------------------------------------

class TestClientConstruction:
    def test_default_attributes(self):
        c = AsyncPostGraph(dsn="postgresql://localhost/test")
        assert c.dsn == "postgresql://localhost/test"
        assert c.schema_per_realm is False
        assert c.connection is None
        assert c._pool is None
        assert c._schema_cache == {}

    def test_schema_per_realm_flag(self):
        c = AsyncPostGraph(dsn="unused", schema_per_realm=True)
        assert c.schema_per_realm is True

    def test_connection_passed_directly(self):
        sentinel = object()
        c = AsyncPostGraph(connection_or_pool=sentinel)
        assert c.connection is sentinel


# ---------------------------------------------------------------------------
# Traverse direction validation (no DB needed)
# ---------------------------------------------------------------------------

class TestTraverseDirectionValidation:
    def test_invalid_neighbor_direction(self):
        c = AsyncPostGraph(dsn="unused")
        with pytest.raises(ValueError, match="Direction"):
            import asyncio
            asyncio.get_event_loop().run_until_complete(
                c.get_neighbors("realm", "vtable", "1", ["e"], direction="sideways")
            )

    def test_invalid_traverse_direction(self):
        c = AsyncPostGraph(dsn="unused")
        with pytest.raises(ValueError, match="Direction"):
            import asyncio
            asyncio.get_event_loop().run_until_complete(
                c.traverse("realm", "vtable", "1", ["e"], direction="sideways")
            )

    def test_invalid_shortest_path_direction(self):
        c = AsyncPostGraph(dsn="unused")
        with pytest.raises(ValueError, match="Direction"):
            import asyncio
            asyncio.get_event_loop().run_until_complete(
                c.shortest_path("realm", "a", "1", "b", "2", ["e"], direction="sideways")
            )


# ===================================================================
# Integration tests — require a running PostgreSQL
# ===================================================================

class TestVertexCRUD:
    async def test_create_table_and_add_vertex(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("people", realm=realm)
        v = await pg_client.add_vertex("people", realm=realm, payload={"name": "Alice"})
        assert isinstance(v, Vertex)
        assert v.realm == realm
        assert v.payload["name"] == "Alice"
        assert v.space == "default"
        assert v.fqid is not None
        assert v.uuid is not None

    async def test_add_vertex_with_explicit_id(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("people", realm=realm)
        v = await pg_client.add_vertex("people", realm=realm, vertex_id=100, payload={"name": "Bob"})
        assert v.id == "100"

    async def test_get_vertex_by_id(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("people", realm=realm)
        v = await pg_client.add_vertex("people", realm=realm, payload={"name": "Carol"})
        fetched = await pg_client.get_vertex("people", realm=realm, vertex_id=v.id)
        assert fetched is not None
        assert fetched.id == v.id
        assert fetched.payload["name"] == "Carol"

    async def test_get_vertex_by_uuid(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("people", realm=realm)
        v = await pg_client.add_vertex("people", realm=realm, payload={"name": "Dave"})
        fetched = await pg_client.get_vertex_by_uuid("people", realm=realm, uuid=v.uuid)
        assert fetched is not None
        assert fetched.uuid == v.uuid

    async def test_get_vertex_by_fqid(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("people", realm=realm)
        v = await pg_client.add_vertex("people", realm=realm, payload={"name": "Eve"})
        fetched = await pg_client.get_vertex("people", realm=realm, vertex_id=v.fqid)
        assert fetched is not None
        assert fetched.fqid == v.fqid

    async def test_get_vertex_not_found(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("people", realm=realm)
        fetched = await pg_client.get_vertex("people", realm=realm, vertex_id="999999")
        assert fetched is None

    async def test_get_vertices(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("people", realm=realm)
        await pg_client.add_vertex("people", realm=realm, payload={"n": 1})
        await pg_client.add_vertex("people", realm=realm, payload={"n": 2})
        await pg_client.add_vertex("people", realm=realm, payload={"n": 3})
        verts = await pg_client.get_vertices("people", realm=realm)
        assert len(verts) == 3

    async def test_get_vertices_with_limit(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("people", realm=realm)
        for i in range(5):
            await pg_client.add_vertex("people", realm=realm, payload={"n": i})
        verts = await pg_client.get_vertices("people", realm=realm, limit=2)
        assert len(verts) == 2

    async def test_upsert_vertex_insert(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("people", realm=realm)
        v = await pg_client.upsert_vertex("people", realm=realm, vertex_id=50, payload={"v": 1})
        assert v.id == "50"
        assert v.payload["v"] == 1

    async def test_upsert_vertex_update_merges_payload(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("people", realm=realm)
        await pg_client.upsert_vertex("people", realm=realm, vertex_id=60, payload={"a": 1})
        v2 = await pg_client.upsert_vertex("people", realm=realm, vertex_id=60, payload={"b": 2})
        assert v2.payload["a"] == 1
        assert v2.payload["b"] == 2

    async def test_delete_vertex(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("people", realm=realm)
        v = await pg_client.add_vertex("people", realm=realm, payload={"x": 1})
        deleted = await pg_client.delete_vertex("people", realm=realm, vertex_id=v.id)
        assert deleted is True
        fetched = await pg_client.get_vertex("people", realm=realm, vertex_id=v.id)
        assert fetched is None

    async def test_delete_vertex_not_found(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("people", realm=realm)
        deleted = await pg_client.delete_vertex("people", realm=realm, vertex_id="999999")
        assert deleted is False

    async def test_vertex_has_client_ref(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("people", realm=realm)
        v = await pg_client.add_vertex("people", realm=realm, payload={"name": "test"})
        assert v._client is pg_client


class TestSpaceIsolation:
    async def test_add_vertex_with_space(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("items", realm=realm)
        v = await pg_client.add_vertex("items", realm=realm, space="staging", payload={"x": 1})
        assert v.space == "staging"

    async def test_get_vertices_filtered_by_space(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("items", realm=realm)
        await pg_client.add_vertex("items", realm=realm, space="prod", payload={"x": 1})
        await pg_client.add_vertex("items", realm=realm, space="prod", payload={"x": 2})
        await pg_client.add_vertex("items", realm=realm, space="staging", payload={"x": 3})

        prod = await pg_client.get_vertices("items", realm=realm, space="prod")
        assert len(prod) == 2

        staging = await pg_client.get_vertices("items", realm=realm, space="staging")
        assert len(staging) == 1

    async def test_get_vertices_all_spaces(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("items", realm=realm)
        await pg_client.add_vertex("items", realm=realm, space="a", payload={"x": 1})
        await pg_client.add_vertex("items", realm=realm, space="b", payload={"x": 2})

        all_v = await pg_client.get_vertices("items", realm=realm, space=RESERVED_SPACE_ALL)
        assert len(all_v) == 2

    async def test_add_vertex_with_reserved_space_raises(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("items", realm=realm)
        with pytest.raises(ReservedSpaceError):
            await pg_client.add_vertex("items", realm=realm, space="__all__", payload={})

    async def test_upsert_vertex_with_reserved_space_raises(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("items", realm=realm)
        with pytest.raises(ReservedSpaceError):
            await pg_client.upsert_vertex("items", realm=realm, vertex_id=1, space="__all__", payload={})


class TestEdgeCRUD:
    async def _setup(self, pg_client, realm):
        await pg_client.create_vertex_table("people", realm=realm)
        await pg_client.create_edge_table(
            "knows", from_vertex_table="people", to_vertex_table="people", realm=realm
        )
        v1 = await pg_client.add_vertex("people", realm=realm, payload={"name": "A"})
        v2 = await pg_client.add_vertex("people", realm=realm, payload={"name": "B"})
        return v1, v2

    async def test_add_edge(self, pg_client, clean_realm):
        realm = clean_realm
        v1, v2 = await self._setup(pg_client, realm)
        e = await pg_client.add_edge(
            "knows", realm=realm, from_id=v1.id, to_id=v2.id,
            relation_type="friends", payload={"since": "2020"}
        )
        assert isinstance(e, Edge)
        assert e.from_id == v1.id
        assert e.to_id == v2.id
        assert e.relation_type == "friends"
        assert e.space == "default"

    async def test_add_edge_with_space(self, pg_client, clean_realm):
        realm = clean_realm
        v1, v2 = await self._setup(pg_client, realm)
        e = await pg_client.add_edge(
            "knows", realm=realm, from_id=v1.id, to_id=v2.id,
            relation_type="friends", space="prod"
        )
        assert e.space == "prod"

    async def test_add_edge_reserved_space_raises(self, pg_client, clean_realm):
        realm = clean_realm
        v1, v2 = await self._setup(pg_client, realm)
        with pytest.raises(ReservedSpaceError):
            await pg_client.add_edge(
                "knows", realm=realm, from_id=v1.id, to_id=v2.id,
                relation_type="friends", space="__all__"
            )

    async def test_get_edge(self, pg_client, clean_realm):
        realm = clean_realm
        v1, v2 = await self._setup(pg_client, realm)
        e = await pg_client.add_edge(
            "knows", realm=realm, from_id=v1.id, to_id=v2.id,
            relation_type="friends"
        )
        fetched = await pg_client.get_edge("knows", realm=realm, edge_id=e.id)
        assert fetched is not None
        assert fetched.id == e.id

    async def test_get_edge_by_uuid(self, pg_client, clean_realm):
        realm = clean_realm
        v1, v2 = await self._setup(pg_client, realm)
        e = await pg_client.add_edge(
            "knows", realm=realm, from_id=v1.id, to_id=v2.id,
            relation_type="friends"
        )
        fetched = await pg_client.get_edge_by_uuid("knows", realm=realm, uuid=e.uuid)
        assert fetched is not None
        assert fetched.uuid == e.uuid

    async def test_upsert_edge(self, pg_client, clean_realm):
        realm = clean_realm
        v1, v2 = await self._setup(pg_client, realm)
        e1 = await pg_client.upsert_edge(
            "knows", realm=realm, from_id=v1.id, to_id=v2.id,
            relation_type="friends", edge_id=1, payload={"a": 1}
        )
        e2 = await pg_client.upsert_edge(
            "knows", realm=realm, from_id=v1.id, to_id=v2.id,
            relation_type="friends", edge_id=1, payload={"b": 2}
        )
        assert e2.payload["a"] == 1
        assert e2.payload["b"] == 2

    async def test_delete_edge(self, pg_client, clean_realm):
        realm = clean_realm
        v1, v2 = await self._setup(pg_client, realm)
        e = await pg_client.add_edge(
            "knows", realm=realm, from_id=v1.id, to_id=v2.id,
            relation_type="friends"
        )
        deleted = await pg_client.delete_edge("knows", realm=realm, edge_id=e.id)
        assert deleted is True
        fetched = await pg_client.get_edge("knows", realm=realm, edge_id=e.id)
        assert fetched is None

    async def test_edge_has_client_ref(self, pg_client, clean_realm):
        realm = clean_realm
        v1, v2 = await self._setup(pg_client, realm)
        e = await pg_client.add_edge(
            "knows", realm=realm, from_id=v1.id, to_id=v2.id,
            relation_type="friends"
        )
        assert e._client is pg_client

    async def test_edge_fk_violation_raises(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("people", realm=realm)
        await pg_client.create_edge_table(
            "knows", from_vertex_table="people", to_vertex_table="people", realm=realm
        )
        with pytest.raises(VertexNotFoundError):
            await pg_client.add_edge(
                "knows", realm=realm, from_id="999", to_id="998",
                relation_type="friends"
            )

    async def test_edge_table_requires_vertex_tables(self, pg_client, clean_realm):
        realm = clean_realm
        with pytest.raises(TableNotFoundError):
            await pg_client.create_edge_table(
                "bad_edge", from_vertex_table="nonexistent", to_vertex_table="also_nonexistent",
                realm=realm
            )

    async def test_get_edges(self, pg_client, clean_realm):
        realm = clean_realm
        v1, v2 = await self._setup(pg_client, realm)
        await pg_client.add_edge("knows", realm=realm, from_id=v1.id, to_id=v2.id, relation_type="friends")
        await pg_client.add_edge("knows", realm=realm, from_id=v2.id, to_id=v1.id, relation_type="colleagues")
        edges = await pg_client.get_edges("knows", realm=realm)
        assert len(edges) == 2
        assert all(isinstance(e, Edge) for e in edges)

    async def test_get_edges_with_limit(self, pg_client, clean_realm):
        realm = clean_realm
        v1, v2 = await self._setup(pg_client, realm)
        for i in range(5):
            await pg_client.add_edge("knows", realm=realm, from_id=v1.id, to_id=v2.id, relation_type=f"r{i}")
        edges = await pg_client.get_edges("knows", realm=realm, limit=2)
        assert len(edges) == 2

    async def test_get_edges_filter_by_relation_type(self, pg_client, clean_realm):
        realm = clean_realm
        v1, v2 = await self._setup(pg_client, realm)
        await pg_client.add_edge("knows", realm=realm, from_id=v1.id, to_id=v2.id, relation_type="friends")
        await pg_client.add_edge("knows", realm=realm, from_id=v2.id, to_id=v1.id, relation_type="colleagues")
        edges = await pg_client.get_edges("knows", realm=realm, relation_type="friends")
        assert len(edges) == 1
        assert edges[0].relation_type == "friends"

    async def test_get_edges_filter_by_space(self, pg_client, clean_realm):
        realm = clean_realm
        v1, v2 = await self._setup(pg_client, realm)
        await pg_client.add_edge("knows", realm=realm, from_id=v1.id, to_id=v2.id, relation_type="friends", space="prod")
        await pg_client.add_edge("knows", realm=realm, from_id=v2.id, to_id=v1.id, relation_type="colleagues", space="staging")
        edges = await pg_client.get_edges("knows", realm=realm, space="prod")
        assert len(edges) == 1
        assert edges[0].space == "prod"

    async def test_get_edges_empty(self, pg_client, clean_realm):
        realm = clean_realm
        await self._setup(pg_client, realm)
        edges = await pg_client.get_edges("knows", realm=realm)
        assert edges == []


class TestCycleDetection:
    async def test_cycle_raises(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("nodes", realm=realm)
        await pg_client.create_edge_table(
            "links", from_vertex_table="nodes", to_vertex_table="nodes", realm=realm
        )
        a = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "A"})
        b = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "B"})
        c = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "C"})

        await pg_client.add_edge("links", realm=realm, from_id=a.id, to_id=b.id, relation_type="to")
        await pg_client.add_edge("links", realm=realm, from_id=b.id, to_id=c.id, relation_type="to")

        with pytest.raises(CyclicReferenceError):
            await pg_client.add_edge(
                "links", realm=realm, from_id=c.id, to_id=a.id,
                relation_type="to", check_cycle=True
            )

    async def test_no_false_positive(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("nodes", realm=realm)
        await pg_client.create_edge_table(
            "links", from_vertex_table="nodes", to_vertex_table="nodes", realm=realm
        )
        a = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "A"})
        b = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "B"})
        c = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "C"})

        await pg_client.add_edge("links", realm=realm, from_id=a.id, to_id=b.id, relation_type="to")
        # b -> c does not create a cycle
        e = await pg_client.add_edge(
            "links", realm=realm, from_id=b.id, to_id=c.id,
            relation_type="to", check_cycle=True
        )
        assert e is not None


class TestNeighbors:
    async def _setup_triangle(self, pg_client, realm):
        await pg_client.create_vertex_table("nodes", realm=realm)
        await pg_client.create_edge_table(
            "links", from_vertex_table="nodes", to_vertex_table="nodes", realm=realm
        )
        a = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "A"})
        b = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "B"})
        c = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "C"})
        await pg_client.add_edge("links", realm=realm, from_id=a.id, to_id=b.id, relation_type="to")
        await pg_client.add_edge("links", realm=realm, from_id=a.id, to_id=c.id, relation_type="to")
        return a, b, c

    async def test_outgoing(self, pg_client, clean_realm):
        realm = clean_realm
        a, b, c = await self._setup_triangle(pg_client, realm)
        neighbors = await pg_client.get_neighbors(realm, "nodes", a.id, ["links"], direction="out")
        assert len(neighbors) == 2
        neighbor_ids = {v.id for v, e in neighbors}
        assert b.id in neighbor_ids
        assert c.id in neighbor_ids

    async def test_incoming(self, pg_client, clean_realm):
        realm = clean_realm
        a, b, c = await self._setup_triangle(pg_client, realm)
        neighbors = await pg_client.get_neighbors(realm, "nodes", b.id, ["links"], direction="in")
        assert len(neighbors) == 1
        assert neighbors[0][0].id == a.id

    async def test_both(self, pg_client, clean_realm):
        realm = clean_realm
        a, b, c = await self._setup_triangle(pg_client, realm)
        await pg_client.add_edge("links", realm=realm, from_id=b.id, to_id=a.id, relation_type="back")
        neighbors = await pg_client.get_neighbors(realm, "nodes", a.id, ["links"], direction="both")
        assert len(neighbors) == 3  # 2 outgoing + 1 incoming


class TestTraversal:
    async def _setup_chain(self, pg_client, realm, n=4):
        """Create a linear chain: v0 -> v1 -> v2 -> ... -> v(n-1)"""
        await pg_client.create_vertex_table("nodes", realm=realm)
        await pg_client.create_edge_table(
            "links", from_vertex_table="nodes", to_vertex_table="nodes", realm=realm
        )
        verts = []
        for i in range(n):
            v = await pg_client.add_vertex("nodes", realm=realm, payload={"i": i})
            verts.append(v)
        for i in range(n - 1):
            await pg_client.add_edge(
                "links", realm=realm, from_id=verts[i].id, to_id=verts[i + 1].id,
                relation_type="next"
            )
        return verts

    async def test_traverse_depth(self, pg_client, clean_realm):
        realm = clean_realm
        verts = await self._setup_chain(pg_client, realm, n=5)
        results = await pg_client.traverse(
            realm, "nodes", verts[0].id, ["links"], max_depth=3
        )
        depths = {r["depth"] for r in results}
        assert 0 in depths  # anchor
        assert 3 in depths
        assert 4 not in depths

    async def test_traverse_full(self, pg_client, clean_realm):
        realm = clean_realm
        verts = await self._setup_chain(pg_client, realm, n=4)
        results = await pg_client.traverse(
            realm, "nodes", verts[0].id, ["links"], max_depth=10
        )
        ids = {r["id"] for r in results}
        for v in verts:
            assert v.id in ids

    async def test_traverse_with_relation_types(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("nodes", realm=realm)
        await pg_client.create_edge_table(
            "links", from_vertex_table="nodes", to_vertex_table="nodes", realm=realm
        )
        a = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "A"})
        b = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "B"})
        c = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "C"})
        await pg_client.add_edge("links", realm=realm, from_id=a.id, to_id=b.id, relation_type="alpha")
        await pg_client.add_edge("links", realm=realm, from_id=a.id, to_id=c.id, relation_type="beta")

        results = await pg_client.traverse(
            realm, "nodes", a.id, ["links"], max_depth=1,
            relation_types=["alpha"]
        )
        reached_ids = {r["id"] for r in results if r["depth"] > 0}
        assert b.id in reached_ids
        assert c.id not in reached_ids

    async def test_traverse_with_space(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("nodes", realm=realm)
        await pg_client.create_edge_table(
            "links", from_vertex_table="nodes", to_vertex_table="nodes", realm=realm
        )
        a = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "A"})
        b = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "B"})
        c = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "C"})
        await pg_client.add_edge("links", realm=realm, from_id=a.id, to_id=b.id, relation_type="r", space="prod")
        await pg_client.add_edge("links", realm=realm, from_id=a.id, to_id=c.id, relation_type="r", space="staging")

        results = await pg_client.traverse(
            realm, "nodes", a.id, ["links"], max_depth=1, space="prod"
        )
        reached_ids = {r["id"] for r in results if r["depth"] > 0}
        assert b.id in reached_ids
        assert c.id not in reached_ids

    async def test_traverse_with_payload_null_keys(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("nodes", realm=realm)
        await pg_client.create_edge_table(
            "links", from_vertex_table="nodes", to_vertex_table="nodes", realm=realm
        )
        a = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "A"})
        b = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "B"})
        c = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "C"})
        await pg_client.add_edge(
            "links", realm=realm, from_id=a.id, to_id=b.id,
            relation_type="r", payload={"superseded_by": "new_edge"}
        )
        await pg_client.add_edge(
            "links", realm=realm, from_id=a.id, to_id=c.id,
            relation_type="r", payload={"active": True}
        )

        results = await pg_client.traverse(
            realm, "nodes", a.id, ["links"], max_depth=1,
            payload_null_keys=["superseded_by"]
        )
        reached_ids = {r["id"] for r in results if r["depth"] > 0}
        assert c.id in reached_ids
        assert b.id not in reached_ids

    async def test_traverse_as_of(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("nodes", realm=realm)
        await pg_client.create_edge_table(
            "links", from_vertex_table="nodes", to_vertex_table="nodes", realm=realm
        )
        a = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "A"})
        b = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "B"})
        c = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "C"})
        await pg_client.add_edge(
            "links", realm=realm, from_id=a.id, to_id=b.id,
            relation_type="r", payload={"valid_from": "2010", "valid_to": "2015"}
        )
        await pg_client.add_edge(
            "links", realm=realm, from_id=a.id, to_id=c.id,
            relation_type="r", payload={"valid_from": "2018", "valid_to": "2025"}
        )

        results_2012 = await pg_client.traverse(
            realm, "nodes", a.id, ["links"], max_depth=1, as_of="2012"
        )
        reached_2012 = {r["id"] for r in results_2012 if r["depth"] > 0}
        assert b.id in reached_2012
        assert c.id not in reached_2012

        results_2020 = await pg_client.traverse(
            realm, "nodes", a.id, ["links"], max_depth=1, as_of="2020"
        )
        reached_2020 = {r["id"] for r in results_2020 if r["depth"] > 0}
        assert c.id in reached_2020
        assert b.id not in reached_2020

    async def test_traverse_as_of_no_dates_always_qualifies(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("nodes", realm=realm)
        await pg_client.create_edge_table(
            "links", from_vertex_table="nodes", to_vertex_table="nodes", realm=realm
        )
        a = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "A"})
        b = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "B"})
        await pg_client.add_edge(
            "links", realm=realm, from_id=a.id, to_id=b.id,
            relation_type="r", payload={}
        )

        results = await pg_client.traverse(
            realm, "nodes", a.id, ["links"], max_depth=1, as_of="2020"
        )
        reached = {r["id"] for r in results if r["depth"] > 0}
        assert b.id in reached


class TestShortestPath:
    async def test_direct_path(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("nodes", realm=realm)
        await pg_client.create_edge_table(
            "links", from_vertex_table="nodes", to_vertex_table="nodes", realm=realm
        )
        a = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "A"})
        b = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "B"})
        await pg_client.add_edge("links", realm=realm, from_id=a.id, to_id=b.id, relation_type="to")

        path = await pg_client.shortest_path(
            realm, "nodes", a.id, "nodes", b.id, ["links"]
        )
        assert path is not None
        assert path["depth"] == 1

    async def test_multi_hop_path(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("nodes", realm=realm)
        await pg_client.create_edge_table(
            "links", from_vertex_table="nodes", to_vertex_table="nodes", realm=realm
        )
        a = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "A"})
        b = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "B"})
        c = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "C"})
        await pg_client.add_edge("links", realm=realm, from_id=a.id, to_id=b.id, relation_type="to")
        await pg_client.add_edge("links", realm=realm, from_id=b.id, to_id=c.id, relation_type="to")

        path = await pg_client.shortest_path(
            realm, "nodes", a.id, "nodes", c.id, ["links"]
        )
        assert path is not None
        assert path["depth"] == 2

    async def test_no_path(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("nodes", realm=realm)
        await pg_client.create_edge_table(
            "links", from_vertex_table="nodes", to_vertex_table="nodes", realm=realm
        )
        a = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "A"})
        b = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "B"})

        path = await pg_client.shortest_path(
            realm, "nodes", a.id, "nodes", b.id, ["links"]
        )
        assert path is None


class TestDataRecords:
    async def test_add_and_get_vertex_data(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("nodes", realm=realm)
        v = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "A"})

        dr = await pg_client.add_vertex_data("nodes", realm=realm, vertex_id=v.id, payload={"v": 1})
        assert isinstance(dr, DataRecord)
        assert dr.payload["v"] == 1

        records = await pg_client.get_vertex_data("nodes", realm=realm, vertex_id=v.id)
        assert len(records) == 1
        assert records[0].data_id == dr.data_id

    async def test_multiple_data_records_ordered(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("nodes", realm=realm)
        v = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "A"})

        await pg_client.add_vertex_data("nodes", realm=realm, vertex_id=v.id, payload={"v": 1})
        await pg_client.add_vertex_data("nodes", realm=realm, vertex_id=v.id, payload={"v": 2})
        await pg_client.add_vertex_data("nodes", realm=realm, vertex_id=v.id, payload={"v": 3})

        records = await pg_client.get_vertex_data("nodes", realm=realm, vertex_id=v.id)
        assert len(records) == 3
        assert records[0].payload["v"] == 3  # most recent first

    async def test_get_latest_vertex_data(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("nodes", realm=realm)
        v = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "A"})

        await pg_client.add_vertex_data("nodes", realm=realm, vertex_id=v.id, payload={"v": 1})
        await pg_client.add_vertex_data("nodes", realm=realm, vertex_id=v.id, payload={"v": 2})

        latest = await pg_client.get_latest_vertex_data("nodes", realm=realm, vertex_id=v.id)
        assert latest is not None
        assert latest.payload["v"] == 2

    async def test_get_vertex_data_by_id(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("nodes", realm=realm)
        v = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "A"})

        dr = await pg_client.add_vertex_data("nodes", realm=realm, vertex_id=v.id, payload={"v": 42})
        fetched = await pg_client.get_vertex_data_by_id("nodes", realm=realm, data_id=dr.data_id)
        assert fetched is not None
        assert fetched.payload["v"] == 42

    async def test_edge_data(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("nodes", realm=realm)
        await pg_client.create_edge_table(
            "links", from_vertex_table="nodes", to_vertex_table="nodes", realm=realm
        )
        a = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "A"})
        b = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "B"})
        e = await pg_client.add_edge("links", realm=realm, from_id=a.id, to_id=b.id, relation_type="to")

        dr = await pg_client.add_edge_data("links", realm=realm, edge_id=e.id, payload={"w": 0.5})
        assert dr.payload["w"] == 0.5

        records = await pg_client.get_edge_data("links", realm=realm, edge_id=e.id)
        assert len(records) == 1

        latest = await pg_client.get_latest_edge_data("links", realm=realm, edge_id=e.id)
        assert latest is not None

    async def test_vertex_data_limited(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("nodes", realm=realm)
        v = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "A"})
        for i in range(5):
            await pg_client.add_vertex_data("nodes", realm=realm, vertex_id=v.id, payload={"v": i})
        records = await pg_client.get_vertex_data("nodes", realm=realm, vertex_id=v.id, limit=2)
        assert len(records) == 2


class TestDeleteRealm:
    async def test_deletes_all_data(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("people", realm=realm)
        await pg_client.add_vertex("people", realm=realm, payload={"n": "A"})
        await pg_client.add_vertex("people", realm=realm, payload={"n": "B"})

        deleted = await pg_client.delete_realm(realm)
        assert deleted >= 2

        verts = await pg_client.get_vertices("people", realm=realm)
        assert len(verts) == 0


class TestSchemaPerRealm:
    async def test_create_and_query(self, pg_client_spr, clean_realm_spr):
        realm = clean_realm_spr
        await pg_client_spr.create_vertex_table("items", realm=realm)
        v = await pg_client_spr.add_vertex("items", realm=realm, payload={"name": "thing"})
        assert v.realm == realm

        fetched = await pg_client_spr.get_vertex("items", realm=realm, vertex_id=v.id)
        assert fetched is not None
        assert fetched.payload["name"] == "thing"

    async def test_edge_operations(self, pg_client_spr, clean_realm_spr):
        realm = clean_realm_spr
        await pg_client_spr.create_vertex_table("nodes", realm=realm)
        await pg_client_spr.create_edge_table(
            "links", from_vertex_table="nodes", to_vertex_table="nodes", realm=realm
        )
        a = await pg_client_spr.add_vertex("nodes", realm=realm, payload={"n": "A"})
        b = await pg_client_spr.add_vertex("nodes", realm=realm, payload={"n": "B"})
        e = await pg_client_spr.add_edge(
            "links", realm=realm, from_id=a.id, to_id=b.id, relation_type="to"
        )
        assert e is not None

        neighbors = await pg_client_spr.get_neighbors(realm, "nodes", a.id, ["links"])
        assert len(neighbors) == 1


class TestVectorSearch:
    async def test_vertex_vector_search(self, pg_client, clean_realm, has_pgvector):
        if not has_pgvector:
            pytest.skip("pgvector not available")
        realm = clean_realm
        dim = 4
        await pg_client.create_vertex_table("items", realm=realm, vector_dim=dim)

        await pg_client.add_vertex("items", realm=realm, payload={"n": "A"}, embedding=[1.0, 0.0, 0.0, 0.0])
        await pg_client.add_vertex("items", realm=realm, payload={"n": "B"}, embedding=[0.0, 1.0, 0.0, 0.0])
        await pg_client.add_vertex("items", realm=realm, payload={"n": "C"}, embedding=[0.9, 0.1, 0.0, 0.0])

        results = await pg_client.vector_search(
            "items", realm=realm, query_vector=[1.0, 0.0, 0.0, 0.0], top_k=2
        )
        assert len(results) == 2
        assert results[0][0].payload["n"] in ("A", "C")

    async def test_vector_search_with_space(self, pg_client, clean_realm, has_pgvector):
        if not has_pgvector:
            pytest.skip("pgvector not available")
        realm = clean_realm
        dim = 4
        await pg_client.create_vertex_table("items", realm=realm, vector_dim=dim)

        await pg_client.add_vertex("items", realm=realm, space="a", payload={"n": "A"}, embedding=[1.0, 0.0, 0.0, 0.0])
        await pg_client.add_vertex("items", realm=realm, space="b", payload={"n": "B"}, embedding=[0.9, 0.1, 0.0, 0.0])

        results = await pg_client.vector_search(
            "items", realm=realm, query_vector=[1.0, 0.0, 0.0, 0.0],
            top_k=10, space="a"
        )
        assert len(results) == 1
        assert results[0][0].payload["n"] == "A"

    async def test_edge_vector_search(self, pg_client, clean_realm, has_pgvector):
        if not has_pgvector:
            pytest.skip("pgvector not available")
        realm = clean_realm
        dim = 4
        await pg_client.create_vertex_table("nodes", realm=realm)
        await pg_client.create_edge_table(
            "rels", from_vertex_table="nodes", to_vertex_table="nodes",
            realm=realm, vector_dim=dim
        )
        a = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "A"})
        b = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "B"})

        await pg_client.add_edge(
            "rels", realm=realm, from_id=a.id, to_id=b.id,
            relation_type="linked", embedding=[1.0, 0.0, 0.0, 0.0]
        )

        results = await pg_client.vector_search_edges(
            "rels", realm=realm, query_vector=[1.0, 0.0, 0.0, 0.0], top_k=5
        )
        assert len(results) == 1
        assert results[0][0].relation_type == "linked"


class TestObjectTraversal:
    """Test the OO traversal API on Vertex objects returned from the client."""

    async def test_vertex_outgoing(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("nodes", realm=realm)
        await pg_client.create_edge_table(
            "links", from_vertex_table="nodes", to_vertex_table="nodes", realm=realm
        )
        a = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "A"})
        b = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "B"})
        await pg_client.add_edge("links", realm=realm, from_id=a.id, to_id=b.id, relation_type="to")

        steps = await a.outgoing("links")
        assert len(steps) == 1
        assert steps[0].vertex().payload["n"] == "B"
        assert steps[0].edge.relation_type == "to"

    async def test_vertex_incoming(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("nodes", realm=realm)
        await pg_client.create_edge_table(
            "links", from_vertex_table="nodes", to_vertex_table="nodes", realm=realm
        )
        a = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "A"})
        b = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "B"})
        await pg_client.add_edge("links", realm=realm, from_id=a.id, to_id=b.id, relation_type="to")

        steps = await b.incoming("links")
        assert len(steps) == 1
        assert steps[0].vertex().payload["n"] == "A"

    async def test_vertex_add_edge_to(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("nodes", realm=realm)
        await pg_client.create_edge_table(
            "links", from_vertex_table="nodes", to_vertex_table="nodes", realm=realm
        )
        a = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "A"})
        b = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "B"})

        edge = await a.add_edge_to(to_id=b.id, edge_table="links")
        assert edge.from_id == a.id
        assert edge.to_id == b.id

    async def test_vertex_delete(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("nodes", realm=realm)
        v = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "A"})
        deleted = await v.delete()
        assert deleted is True

    async def test_vertex_add_and_get_data(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("nodes", realm=realm)
        v = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "A"})

        dr = await v.add_data(payload={"version": 1})
        assert dr.payload["version"] == 1

        records = await v.get_data()
        assert len(records) == 1

        latest = await v.get_latest_data()
        assert latest is not None

        by_id = await v.get_data_by_id(data_id=dr.data_id)
        assert by_id is not None


# ===================================================================
# Additional integration tests
# ===================================================================

class TestVertexAutoId:
    async def test_auto_id_increments(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("items", realm=realm)
        v1 = await pg_client.add_vertex("items", realm=realm, payload={"n": 1})
        v2 = await pg_client.add_vertex("items", realm=realm, payload={"n": 2})
        assert int(v2.id) > int(v1.id)

    async def test_auto_id_after_explicit_id(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("items", realm=realm)
        await pg_client.add_vertex("items", realm=realm, vertex_id=100, payload={"n": 1})
        v2 = await pg_client.add_vertex("items", realm=realm, payload={"n": 2})
        assert int(v2.id) > 100


class TestVertexEmbedding:
    async def test_vertex_embedding_roundtrip(self, pg_client, clean_realm, has_pgvector):
        if not has_pgvector:
            pytest.skip("pgvector not available")
        realm = clean_realm
        dim = 4
        await pg_client.create_vertex_table("items", realm=realm, vector_dim=dim)
        emb = [0.1, 0.2, 0.3, 0.4]
        v = await pg_client.add_vertex("items", realm=realm, payload={"n": "A"}, embedding=emb)
        assert v.embedding is not None
        assert len(v.embedding) == 4
        for a, b in zip(v.embedding, emb):
            assert abs(a - b) < 1e-5

    async def test_upsert_vertex_updates_embedding(self, pg_client, clean_realm, has_pgvector):
        if not has_pgvector:
            pytest.skip("pgvector not available")
        realm = clean_realm
        dim = 4
        await pg_client.create_vertex_table("items", realm=realm, vector_dim=dim)
        v1 = await pg_client.upsert_vertex(
            "items", realm=realm, vertex_id=1, payload={"n": "A"},
            embedding=[1.0, 0.0, 0.0, 0.0]
        )
        v2 = await pg_client.upsert_vertex(
            "items", realm=realm, vertex_id=1, payload={"n": "A"},
            embedding=[0.0, 1.0, 0.0, 0.0]
        )
        assert v2.embedding is not None
        assert abs(v2.embedding[0]) < 1e-5
        assert abs(v2.embedding[1] - 1.0) < 1e-5


class TestEdgeDefaultTableName:
    async def test_default_edge_table_name(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("people", realm=realm)
        await pg_client.create_vertex_table("companies", realm=realm)
        await pg_client.create_edge_table(
            from_vertex_table="people", to_vertex_table="companies", realm=realm
        )
        v1 = await pg_client.add_vertex("people", realm=realm, payload={"n": "A"})
        v2 = await pg_client.add_vertex("companies", realm=realm, payload={"n": "X"})
        e = await pg_client.add_edge(
            "peopleTOcompanies", realm=realm, from_id=v1.id, to_id=v2.id,
            relation_type="works_at"
        )
        assert e is not None
        assert e.relation_type == "works_at"


class TestEdgeExplicitId:
    async def test_edge_with_explicit_id(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("nodes", realm=realm)
        await pg_client.create_edge_table(
            "links", from_vertex_table="nodes", to_vertex_table="nodes", realm=realm
        )
        a = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "A"})
        b = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "B"})
        e = await pg_client.add_edge(
            "links", realm=realm, from_id=a.id, to_id=b.id,
            relation_type="to", edge_id=500
        )
        assert e.id == "500"


class TestCascadeDeleteFromVertex:
    async def test_deleting_vertex_cascades_edges(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("nodes", realm=realm)
        await pg_client.create_edge_table(
            "links", from_vertex_table="nodes", to_vertex_table="nodes", realm=realm
        )
        a = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "A"})
        b = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "B"})
        e = await pg_client.add_edge(
            "links", realm=realm, from_id=a.id, to_id=b.id, relation_type="to"
        )
        await pg_client.delete_vertex("nodes", realm=realm, vertex_id=a.id)
        fetched = await pg_client.get_edge("links", realm=realm, edge_id=e.id)
        assert fetched is None


class TestEdgeUpsertReservedSpace:
    async def test_upsert_edge_reserved_space_raises(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("nodes", realm=realm)
        await pg_client.create_edge_table(
            "links", from_vertex_table="nodes", to_vertex_table="nodes", realm=realm
        )
        a = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "A"})
        b = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "B"})
        with pytest.raises(ReservedSpaceError):
            await pg_client.upsert_edge(
                "links", realm=realm, from_id=a.id, to_id=b.id,
                relation_type="to", edge_id=1, space="__all__"
            )


class TestVertexGetByInvalidUuid:
    async def test_invalid_uuid_returns_none(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("nodes", realm=realm)
        fetched = await pg_client.get_vertex_by_uuid("nodes", realm=realm, uuid="not-a-uuid")
        assert fetched is None


class TestEdgeGetByInvalidUuid:
    async def test_invalid_uuid_returns_none(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("nodes", realm=realm)
        await pg_client.create_edge_table(
            "links", from_vertex_table="nodes", to_vertex_table="nodes", realm=realm
        )
        fetched = await pg_client.get_edge_by_uuid("links", realm=realm, uuid="not-a-uuid")
        assert fetched is None


class TestGetVerticesEmptyRealm:
    async def test_empty_result(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("nodes", realm=realm)
        verts = await pg_client.get_vertices("nodes", realm=realm)
        assert verts == []


class TestGetVertexDataEmpty:
    async def test_no_data_returns_empty(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("nodes", realm=realm)
        v = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "A"})
        records = await pg_client.get_vertex_data("nodes", realm=realm, vertex_id=v.id)
        assert records == []

    async def test_get_latest_data_returns_none_when_empty(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("nodes", realm=realm)
        v = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "A"})
        latest = await pg_client.get_latest_vertex_data("nodes", realm=realm, vertex_id=v.id)
        assert latest is None

    async def test_get_data_by_nonexistent_id(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("nodes", realm=realm)
        fetched = await pg_client.get_vertex_data_by_id("nodes", realm=realm, data_id=999999)
        assert fetched is None


class TestDeleteRealmWithEdges:
    async def test_deletes_vertices_and_edges(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("nodes", realm=realm)
        await pg_client.create_edge_table(
            "links", from_vertex_table="nodes", to_vertex_table="nodes", realm=realm
        )
        a = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "A"})
        b = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "B"})
        await pg_client.add_edge("links", realm=realm, from_id=a.id, to_id=b.id, relation_type="to")

        deleted = await pg_client.delete_realm(realm)
        assert deleted >= 3  # 2 vertices + 1 edge (plus possible audit records)


class TestTraverseMultipleEdgeTables:
    async def test_traverse_across_two_edge_tables(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("multi_nodes", realm=realm)
        await pg_client.create_edge_table(
            "multi_knows", from_vertex_table="multi_nodes", to_vertex_table="multi_nodes", realm=realm
        )
        await pg_client.create_edge_table(
            "multi_works_with", from_vertex_table="multi_nodes", to_vertex_table="multi_nodes", realm=realm
        )
        a = await pg_client.add_vertex("multi_nodes", realm=realm, payload={"n": "A"})
        b = await pg_client.add_vertex("multi_nodes", realm=realm, payload={"n": "B"})
        c = await pg_client.add_vertex("multi_nodes", realm=realm, payload={"n": "C"})
        await pg_client.add_edge("multi_knows", realm=realm, from_id=a.id, to_id=b.id, relation_type="knows")
        await pg_client.add_edge("multi_works_with", realm=realm, from_id=b.id, to_id=c.id, relation_type="works_with")

        results = await pg_client.traverse(
            realm, "multi_nodes", a.id, ["multi_knows", "multi_works_with"], max_depth=2
        )
        reached = {r["id"] for r in results}
        assert a.id in reached
        assert b.id in reached
        assert c.id in reached


class TestTraverseIncomingDirection:
    async def test_traverse_incoming(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("nodes", realm=realm)
        await pg_client.create_edge_table(
            "links", from_vertex_table="nodes", to_vertex_table="nodes", realm=realm
        )
        a = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "A"})
        b = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "B"})
        c = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "C"})
        await pg_client.add_edge("links", realm=realm, from_id=a.id, to_id=c.id, relation_type="to")
        await pg_client.add_edge("links", realm=realm, from_id=b.id, to_id=c.id, relation_type="to")

        results = await pg_client.traverse(
            realm, "nodes", c.id, ["links"], max_depth=1, direction="in"
        )
        reached = {r["id"] for r in results if r["depth"] > 0}
        assert a.id in reached
        assert b.id in reached


class TestTraverseCycleProtection:
    async def test_traverse_does_not_loop(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("nodes", realm=realm)
        await pg_client.create_edge_table(
            "links", from_vertex_table="nodes", to_vertex_table="nodes", realm=realm
        )
        a = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "A"})
        b = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "B"})
        await pg_client.add_edge("links", realm=realm, from_id=a.id, to_id=b.id, relation_type="to")
        await pg_client.add_edge("links", realm=realm, from_id=b.id, to_id=a.id, relation_type="to")

        results = await pg_client.traverse(
            realm, "nodes", a.id, ["links"], max_depth=10
        )
        ids = [r["id"] for r in results]
        assert len(ids) == len(set(ids))


class TestVectorSearchScopes:
    async def test_vector_search_data_scope(self, pg_client, clean_realm, has_pgvector):
        if not has_pgvector:
            pytest.skip("pgvector not available")
        realm = clean_realm
        dim = 4
        await pg_client.create_vertex_table("items", realm=realm, vector_dim=dim)
        v = await pg_client.add_vertex("items", realm=realm, payload={"n": "A"}, embedding=[1.0, 0.0, 0.0, 0.0])
        await pg_client.add_vertex_data("items", realm=realm, vertex_id=v.id,
                                         payload={"v": 1}, embedding=[0.0, 1.0, 0.0, 0.0])

        results = await pg_client.vector_search(
            "items", realm=realm, query_vector=[0.0, 1.0, 0.0, 0.0],
            top_k=5, search_scope="data"
        )
        assert len(results) >= 1

    async def test_vector_search_both_scope(self, pg_client, clean_realm, has_pgvector):
        if not has_pgvector:
            pytest.skip("pgvector not available")
        realm = clean_realm
        dim = 4
        await pg_client.create_vertex_table("items", realm=realm, vector_dim=dim)
        await pg_client.add_vertex("items", realm=realm, payload={"n": "A"}, embedding=[1.0, 0.0, 0.0, 0.0])

        results = await pg_client.vector_search(
            "items", realm=realm, query_vector=[1.0, 0.0, 0.0, 0.0],
            top_k=5, search_scope="both"
        )
        assert len(results) >= 1

    async def test_vector_search_l2_metric(self, pg_client, clean_realm, has_pgvector):
        if not has_pgvector:
            pytest.skip("pgvector not available")
        realm = clean_realm
        dim = 4
        await pg_client.create_vertex_table("items", realm=realm, vector_dim=dim)
        await pg_client.add_vertex("items", realm=realm, payload={"n": "A"}, embedding=[1.0, 0.0, 0.0, 0.0])

        results = await pg_client.vector_search(
            "items", realm=realm, query_vector=[1.0, 0.0, 0.0, 0.0],
            top_k=5, distance_metric="l2"
        )
        assert len(results) >= 1

    async def test_vector_search_inner_product_metric(self, pg_client, clean_realm, has_pgvector):
        if not has_pgvector:
            pytest.skip("pgvector not available")
        realm = clean_realm
        dim = 4
        await pg_client.create_vertex_table("items", realm=realm, vector_dim=dim)
        await pg_client.add_vertex("items", realm=realm, payload={"n": "A"}, embedding=[1.0, 0.0, 0.0, 0.0])

        results = await pg_client.vector_search(
            "items", realm=realm, query_vector=[1.0, 0.0, 0.0, 0.0],
            top_k=5, distance_metric="inner_product"
        )
        assert len(results) >= 1

    async def test_vector_search_all_spaces(self, pg_client, clean_realm, has_pgvector):
        if not has_pgvector:
            pytest.skip("pgvector not available")
        realm = clean_realm
        dim = 4
        await pg_client.create_vertex_table("items", realm=realm, vector_dim=dim)
        await pg_client.add_vertex("items", realm=realm, space="a", payload={"n": "A"}, embedding=[1.0, 0.0, 0.0, 0.0])
        await pg_client.add_vertex("items", realm=realm, space="b", payload={"n": "B"}, embedding=[0.9, 0.1, 0.0, 0.0])

        results = await pg_client.vector_search(
            "items", realm=realm, query_vector=[1.0, 0.0, 0.0, 0.0],
            top_k=10, space="__all__"
        )
        assert len(results) == 2


class TestEdgeVectorSearchFilters:
    async def test_edge_vector_search_by_relation_type(self, pg_client, clean_realm, has_pgvector):
        if not has_pgvector:
            pytest.skip("pgvector not available")
        realm = clean_realm
        dim = 4
        await pg_client.create_vertex_table("nodes", realm=realm)
        await pg_client.create_edge_table(
            "rels", from_vertex_table="nodes", to_vertex_table="nodes",
            realm=realm, vector_dim=dim
        )
        a = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "A"})
        b = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "B"})
        c = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "C"})
        await pg_client.add_edge(
            "rels", realm=realm, from_id=a.id, to_id=b.id,
            relation_type="alpha", embedding=[1.0, 0.0, 0.0, 0.0]
        )
        await pg_client.add_edge(
            "rels", realm=realm, from_id=a.id, to_id=c.id,
            relation_type="beta", embedding=[0.9, 0.1, 0.0, 0.0]
        )

        results = await pg_client.vector_search_edges(
            "rels", realm=realm, query_vector=[1.0, 0.0, 0.0, 0.0],
            top_k=10, relation_type="alpha"
        )
        assert len(results) == 1
        assert results[0][0].relation_type == "alpha"

    async def test_edge_vector_search_by_space(self, pg_client, clean_realm, has_pgvector):
        if not has_pgvector:
            pytest.skip("pgvector not available")
        realm = clean_realm
        dim = 4
        await pg_client.create_vertex_table("nodes", realm=realm)
        await pg_client.create_edge_table(
            "rels", from_vertex_table="nodes", to_vertex_table="nodes",
            realm=realm, vector_dim=dim
        )
        a = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "A"})
        b = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "B"})
        c = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "C"})
        await pg_client.add_edge(
            "rels", realm=realm, from_id=a.id, to_id=b.id,
            relation_type="r", space="prod", embedding=[1.0, 0.0, 0.0, 0.0]
        )
        await pg_client.add_edge(
            "rels", realm=realm, from_id=a.id, to_id=c.id,
            relation_type="r", space="staging", embedding=[0.9, 0.1, 0.0, 0.0]
        )

        results = await pg_client.vector_search_edges(
            "rels", realm=realm, query_vector=[1.0, 0.0, 0.0, 0.0],
            top_k=10, space="prod"
        )
        assert len(results) == 1
        assert results[0][0].space == "prod"


class TestEdgeSchema:
    async def test_get_edge_schema(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("people", realm=realm)
        await pg_client.create_vertex_table("companies", realm=realm)
        await pg_client.create_edge_table(
            "works_at", from_vertex_table="people", to_vertex_table="companies", realm=realm
        )
        schema = await pg_client.get_edge_schema("works_at", realm=realm)
        assert schema["from_id"] == "people"
        assert schema["to_id"] == "companies"

    async def test_get_edge_schema_caching(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("nodes", realm=realm)
        await pg_client.create_edge_table(
            "links", from_vertex_table="nodes", to_vertex_table="nodes", realm=realm
        )
        s1 = await pg_client.get_edge_schema("links", realm=realm)
        s2 = await pg_client.get_edge_schema("links", realm=realm)
        assert s1 is s2


class TestVertexSpaceInheritance:
    async def test_add_edge_to_inherits_space(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("nodes", realm=realm)
        await pg_client.create_edge_table(
            "links", from_vertex_table="nodes", to_vertex_table="nodes", realm=realm
        )
        a = await pg_client.add_vertex("nodes", realm=realm, space="prod", payload={"n": "A"})
        b = await pg_client.add_vertex("nodes", realm=realm, space="prod", payload={"n": "B"})
        edge = await a.add_edge_to(to_id=b.id, edge_table="links")
        assert edge.space == "prod"

    async def test_add_edge_to_override_space(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("nodes", realm=realm)
        await pg_client.create_edge_table(
            "links", from_vertex_table="nodes", to_vertex_table="nodes", realm=realm
        )
        a = await pg_client.add_vertex("nodes", realm=realm, space="prod", payload={"n": "A"})
        b = await pg_client.add_vertex("nodes", realm=realm, space="prod", payload={"n": "B"})
        edge = await a.add_edge_to(to_id=b.id, edge_table="links", space="staging")
        assert edge.space == "staging"


class TestSchemaPerRealmTraversal:
    async def test_traverse_in_schema_per_realm(self, pg_client_spr, clean_realm_spr):
        realm = clean_realm_spr
        await pg_client_spr.create_vertex_table("nodes", realm=realm)
        await pg_client_spr.create_edge_table(
            "links", from_vertex_table="nodes", to_vertex_table="nodes", realm=realm
        )
        a = await pg_client_spr.add_vertex("nodes", realm=realm, payload={"n": "A"})
        b = await pg_client_spr.add_vertex("nodes", realm=realm, payload={"n": "B"})
        await pg_client_spr.add_edge("links", realm=realm, from_id=a.id, to_id=b.id, relation_type="to")

        results = await pg_client_spr.traverse(
            realm, "nodes", a.id, ["links"], max_depth=1
        )
        reached = {r["id"] for r in results}
        assert a.id in reached
        assert b.id in reached


class TestVertexPayloadTypes:
    async def test_nested_json_payload(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("nodes", realm=realm)
        payload = {"nested": {"list": [1, 2, 3], "obj": {"a": True}}, "top": "val"}
        v = await pg_client.add_vertex("nodes", realm=realm, payload=payload)
        assert v.payload["nested"]["list"] == [1, 2, 3]
        assert v.payload["nested"]["obj"]["a"] is True

        fetched = await pg_client.get_vertex("nodes", realm=realm, vertex_id=v.id)
        assert fetched.payload == payload

    async def test_empty_payload(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("nodes", realm=realm)
        v = await pg_client.add_vertex("nodes", realm=realm, payload={})
        assert v.payload == {}

    async def test_null_payload_defaults_to_empty(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("nodes", realm=realm)
        v = await pg_client.add_vertex("nodes", realm=realm, payload=None)
        assert v.payload == {}


class TestUpsertVertexSpaceUpdate:
    async def test_upsert_updates_space(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("nodes", realm=realm)
        v1 = await pg_client.upsert_vertex("nodes", realm=realm, vertex_id=1,
                                             space="prod", payload={"n": 1})
        assert v1.space == "prod"
        v2 = await pg_client.upsert_vertex("nodes", realm=realm, vertex_id=1,
                                             space="staging", payload={"n": 2})
        assert v2.space == "staging"
        assert v2.payload["n"] == 2
        assert v2.payload.get("n") == 2


class TestEdgeGetNotFound:
    async def test_get_edge_nonexistent(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("nodes", realm=realm)
        await pg_client.create_edge_table(
            "links", from_vertex_table="nodes", to_vertex_table="nodes", realm=realm
        )
        fetched = await pg_client.get_edge("links", realm=realm, edge_id="999999")
        assert fetched is None

    async def test_delete_edge_not_found(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("nodes", realm=realm)
        await pg_client.create_edge_table(
            "links", from_vertex_table="nodes", to_vertex_table="nodes", realm=realm
        )
        deleted = await pg_client.delete_edge("links", realm=realm, edge_id="999999")
        assert deleted is False


class TestVertexFqidFormat:
    async def test_fqid_format(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("people", realm=realm)
        v = await pg_client.add_vertex("people", realm=realm, payload={"n": "A"})
        assert v.fqid == f"{realm}/people/{v.id}"


class TestEdgeFqidFormat:
    async def test_edge_fqid_format(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("people", realm=realm)
        await pg_client.create_vertex_table("companies", realm=realm)
        await pg_client.create_edge_table(
            "works_at", from_vertex_table="people", to_vertex_table="companies", realm=realm
        )
        v1 = await pg_client.add_vertex("people", realm=realm, payload={"n": "A"})
        v2 = await pg_client.add_vertex("companies", realm=realm, payload={"n": "X"})
        e = await pg_client.add_edge(
            "works_at", realm=realm, from_id=v1.id, to_id=v2.id, relation_type="r"
        )
        assert e.fqid == f"{realm}/people-companies/{e.id}"


class TestVertexAddEdgeFrom:
    async def test_add_edge_from(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("nodes", realm=realm)
        await pg_client.create_edge_table(
            "links", from_vertex_table="nodes", to_vertex_table="nodes", realm=realm
        )
        a = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "A"})
        b = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "B"})
        edge = await b.add_edge_from(from_id=a.id, edge_table="links")
        assert edge.from_id == a.id
        assert edge.to_id == b.id


class TestEdgeDeleteViaModel:
    async def test_edge_delete_via_model(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("nodes", realm=realm)
        await pg_client.create_edge_table(
            "links", from_vertex_table="nodes", to_vertex_table="nodes", realm=realm
        )
        a = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "A"})
        b = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "B"})
        e = await pg_client.add_edge("links", realm=realm, from_id=a.id, to_id=b.id, relation_type="to")
        deleted = await e.delete()
        assert deleted is True
        fetched = await pg_client.get_edge("links", realm=realm, edge_id=e.id)
        assert fetched is None


class TestEdgeDataViaModel:
    async def test_edge_data_via_model(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("nodes", realm=realm)
        await pg_client.create_edge_table(
            "links", from_vertex_table="nodes", to_vertex_table="nodes", realm=realm
        )
        a = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "A"})
        b = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "B"})
        e = await pg_client.add_edge("links", realm=realm, from_id=a.id, to_id=b.id, relation_type="to")

        dr = await e.add_data(payload={"w": 1.0})
        assert dr.payload["w"] == 1.0

        records = await e.get_data()
        assert len(records) == 1

        latest = await e.get_latest_data()
        assert latest is not None
        assert latest.payload["w"] == 1.0

        by_id = await e.get_data_by_id(data_id=dr.data_id)
        assert by_id is not None


class TestTraversalStepChaining:
    async def test_chained_traversal(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("nodes", realm=realm)
        await pg_client.create_edge_table(
            "links", from_vertex_table="nodes", to_vertex_table="nodes", realm=realm
        )
        a = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "A"})
        b = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "B"})
        c = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "C"})
        await pg_client.add_edge("links", realm=realm, from_id=a.id, to_id=b.id, relation_type="to")
        await pg_client.add_edge("links", realm=realm, from_id=b.id, to_id=c.id, relation_type="to")

        steps_1 = await a.outgoing("links")
        assert len(steps_1) == 1
        assert steps_1[0].vertex().payload["n"] == "B"

        steps_2 = await steps_1[0].vertex().outgoing("links")
        assert len(steps_2) == 1
        assert steps_2[0].vertex().payload["n"] == "C"

    async def test_step_add_edge_to(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("nodes", realm=realm)
        await pg_client.create_edge_table(
            "links", from_vertex_table="nodes", to_vertex_table="nodes", realm=realm
        )
        a = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "A"})
        b = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "B"})
        c = await pg_client.add_vertex("nodes", realm=realm, payload={"n": "C"})
        await pg_client.add_edge("links", realm=realm, from_id=a.id, to_id=b.id, relation_type="to")

        steps = await a.outgoing("links")
        edge = await steps[0].add_edge_to(to_id=c.id, edge_table="links")
        assert edge.from_id == b.id
        assert edge.to_id == c.id


# ------------------------------------------------- vertex data, schema-per-realm

@pytest.mark.asyncio
async def test_add_vertex_data_works_in_schema_per_realm():
    """Regression: RETURNING cannot name the target table schema-qualified.

    Every add_vertex_data variant ended with `to_jsonb("realm"."t_data")`, which
    PostgreSQL rejects with "missing FROM-clause entry". That made the whole
    append-only history unusable in schema_per_realm mode — the recommended
    mode — while the shared-table mode it was tested in worked fine.

    Builds its own client rather than using the session-scoped fixtures: those
    share one connection across tests and fail with "another operation is in
    progress", which is what hides every live test in this file.
    """
    import os
    import uuid

    dsn = os.getenv("POST_GRAPH_TEST_DSN", "postgresql://localhost/post_graph_test")

    realm = f"test_{uuid.uuid4().hex[:12]}"
    c = AsyncPostGraph(dsn=dsn, schema_per_realm=True)
    try:
        await c.connect()
    except Exception as exc:
        pytest.skip(f"PostgreSQL not reachable: {exc}")

    try:
        await c.create_vertex_table("hist", realm=realm)
        v = await c.add_vertex("hist", realm=realm, payload={"name": "thing"})

        await c.add_vertex_data(table_name="hist", realm=realm, vertex_id=int(v.id),
                                payload={"version": "1.0.0"})
        await c.add_vertex_data(table_name="hist", realm=realm, vertex_id=int(v.id),
                                payload={"version": "1.0.1"})

        records = await c.get_vertex_data(table_name="hist", realm=realm, vertex_id=int(v.id))
        assert len(records) == 2, "both history records must be readable back"

        versions = []
        for r in records:
            payload = r.to_dict()["payload"]
            if isinstance(payload, str):
                payload = json.loads(payload)
            versions.append(payload["version"])
        # Newest first: this ordering is what version resolution relies on.
        assert versions == ["1.0.1", "1.0.0"]
    finally:
        try:
            await c._execute(f'DROP SCHEMA IF EXISTS "{realm}" CASCADE')
        finally:
            await c.close()


# ---------------------------------------------------------------------------
# Public fetch/execute/fetchrow API
# ---------------------------------------------------------------------------

class TestPublicQueryAPI:
    @pytest.mark.asyncio
    async def test_fetch_returns_rows(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("people")
        await pg_client.add_vertex("people", realm, payload={"k": "v1"})
        await pg_client.add_vertex("people", realm, payload={"k": "v2"})
        rows = await pg_client.fetch(
            'SELECT payload FROM "people" WHERE realm = $1', realm
        )
        assert len(rows) == 2

    @pytest.mark.asyncio
    async def test_execute_runs_statement(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("people")
        await pg_client.add_vertex("people", realm, payload={"x": 1})
        status = await pg_client.execute(
            'DELETE FROM "people" WHERE realm = $1', realm
        )
        assert "DELETE" in status

    @pytest.mark.asyncio
    async def test_fetchrow_returns_one(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("people")
        await pg_client.add_vertex("people", realm, payload={"only": True})
        row = await pg_client.fetchrow(
            'SELECT payload FROM "people" WHERE realm = $1', realm
        )
        assert row is not None

    @pytest.mark.asyncio
    async def test_aliases_match_underlying_function(self, pg_client):
        assert type(pg_client).fetch is type(pg_client)._fetch
        assert type(pg_client).execute is type(pg_client)._execute
        assert type(pg_client).fetchrow is type(pg_client)._fetchrow
