"""Integration tests for SQLAlchemyPostGraph against a live PostgreSQL instance.

Mirrors the asyncpg test suite to ensure both clients behave identically.
Tests are skipped when the database is unreachable.
"""

import json
import uuid

import pytest
from post_graph.client_sqlalchemy import SQLAlchemyPostGraph
from post_graph import (
    RESERVED_SPACE_ALL,
    ReservedSpaceError,
    TableExistsError,
    TableNotFoundError,
    VertexNotFoundError,
    EdgeNotFoundError,
    CyclicReferenceError,
    PostGraphError,
)
from post_graph.models import Vertex, Edge, DataRecord


# ---------------------------------------------------------------------------
# Identifier validation (no DB needed)
# ---------------------------------------------------------------------------

class TestValidateIdentifier:
    def test_valid(self):
        client = SQLAlchemyPostGraph(engine_or_connection=object())
        for name in ("people", "my_table", "_private", "T1"):
            client._validate_identifier(name)

    def test_empty_raises(self):
        client = SQLAlchemyPostGraph(engine_or_connection=object())
        with pytest.raises(ValueError):
            client._validate_identifier("")

    def test_none_raises(self):
        client = SQLAlchemyPostGraph(engine_or_connection=object())
        with pytest.raises(ValueError):
            client._validate_identifier(None)

    def test_special_chars_raise(self):
        client = SQLAlchemyPostGraph(engine_or_connection=object())
        for bad in ("my-table", "drop;--", "my table", "1table", "tbl$"):
            with pytest.raises(ValueError):
                client._validate_identifier(bad)


# ---------------------------------------------------------------------------
# _get_table_ref (no DB needed)
# ---------------------------------------------------------------------------

class TestGetTableRef:
    def test_simple(self):
        client = SQLAlchemyPostGraph(engine_or_connection=object())
        assert client._get_table_ref("people") == '"people"'

    def test_schema_per_realm(self):
        client = SQLAlchemyPostGraph(engine_or_connection=object(), schema_per_realm=True)
        assert client._get_table_ref("people", "myrealm") == '"myrealm"."people"'

    def test_schema_per_realm_missing_realm(self):
        client = SQLAlchemyPostGraph(engine_or_connection=object(), schema_per_realm=True)
        with pytest.raises(PostGraphError):
            client._get_table_ref("people")

    def test_no_validation_on_table_name(self):
        client = SQLAlchemyPostGraph(engine_or_connection=object())
        ref = client._get_table_ref("any_name")
        assert ref == '"any_name"'


# ---------------------------------------------------------------------------
# _padded_date_sql (static, no DB needed)
# ---------------------------------------------------------------------------

class TestPaddedDateSql:
    def test_output_format(self):
        sql = SQLAlchemyPostGraph._padded_date_sql("col")
        assert "split_part" in sql
        assert "'01'" in sql


# ---------------------------------------------------------------------------
# Edge filter SQL builder (no DB needed)
# ---------------------------------------------------------------------------

class TestEdgeFilterSql:
    def _client(self):
        return SQLAlchemyPostGraph(engine_or_connection=object())

    def test_empty_filters(self):
        sql, params = self._client()._edge_filter_sql(
            relation_types=None, as_of=None, payload_null_keys=None,
            space=None, valid_from_key="valid_from", valid_to_key="valid_to",
        )
        assert sql == ""
        assert params == {}

    def test_relation_types(self):
        sql, params = self._client()._edge_filter_sql(
            relation_types=["knows", "works_at"], as_of=None,
            payload_null_keys=None, space=None,
            valid_from_key="vf", valid_to_key="vt",
        )
        assert "relation_type" in sql
        assert "rel_types" in params

    def test_space_filter(self):
        sql, params = self._client()._edge_filter_sql(
            relation_types=None, as_of=None, payload_null_keys=None,
            space="prod", valid_from_key="vf", valid_to_key="vt",
        )
        assert "space" in sql
        assert params["trav_space"] == "prod"

    def test_space_all_is_ignored(self):
        sql, params = self._client()._edge_filter_sql(
            relation_types=None, as_of=None, payload_null_keys=None,
            space=RESERVED_SPACE_ALL, valid_from_key="vf", valid_to_key="vt",
        )
        assert sql == ""
        assert params == {}

    def test_as_of(self):
        sql, params = self._client()._edge_filter_sql(
            relation_types=None, as_of="2020-06-15",
            payload_null_keys=None, space=None,
            valid_from_key="valid_from", valid_to_key="valid_to",
        )
        assert "trav_as_of" in params
        assert "split_part" in sql

    def test_payload_null_keys(self):
        sql, params = self._client()._edge_filter_sql(
            relation_types=None, as_of=None,
            payload_null_keys=["superseded_by"],
            space=None, valid_from_key="vf", valid_to_key="vt",
        )
        assert "superseded_by" in sql
        assert "IS NULL" in sql

    def test_payload_null_keys_rejects_bad_identifier(self):
        with pytest.raises(ValueError):
            self._client()._edge_filter_sql(
                relation_types=None, as_of=None,
                payload_null_keys=["drop;--"],
                space=None, valid_from_key="vf", valid_to_key="vt",
            )

    def test_combined_filters(self):
        sql, params = self._client()._edge_filter_sql(
            relation_types=["knows"], as_of="2020",
            payload_null_keys=["deleted"],
            space="prod", valid_from_key="vf", valid_to_key="vt",
        )
        assert "rel_types" in params
        assert "trav_space" in params
        assert "trav_as_of" in params
        assert "deleted" in sql


# ---------------------------------------------------------------------------
# Client construction (no DB needed)
# ---------------------------------------------------------------------------

class TestClientConstruction:
    def test_default_attributes(self):
        client = SQLAlchemyPostGraph(engine_or_connection=object())
        assert client.schema_per_realm is False
        assert client._schema_cache == {}

    def test_schema_per_realm_flag(self):
        client = SQLAlchemyPostGraph(engine_or_connection=object(), schema_per_realm=True)
        assert client.schema_per_realm is True


# ---------------------------------------------------------------------------
# Direction validation (no DB needed)
# ---------------------------------------------------------------------------

class TestDirectionValidation:
    def test_invalid_neighbor_direction(self):
        client = SQLAlchemyPostGraph(engine_or_connection=object())
        with pytest.raises(ValueError):
            import asyncio
            asyncio.get_event_loop().run_until_complete(
                client.get_neighbors("r", "t", "1", ["e"], direction="sideways")
            )

    def test_invalid_traverse_direction(self):
        client = SQLAlchemyPostGraph(engine_or_connection=object())
        with pytest.raises(ValueError):
            import asyncio
            asyncio.get_event_loop().run_until_complete(
                client.traverse("r", "t", "1", ["e"], direction="up")
            )

    def test_invalid_shortest_path_direction(self):
        client = SQLAlchemyPostGraph(engine_or_connection=object())
        with pytest.raises(ValueError):
            import asyncio
            asyncio.get_event_loop().run_until_complete(
                client.shortest_path("r", "t", "1", "t", "2", ["e"], direction="diagonal")
            )


# ===========================================================================
# Integration tests — require a live PostgreSQL database
# ===========================================================================

class TestVertexCRUD:
    async def test_create_table_and_add_vertex(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        v = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "Alice"})
        assert isinstance(v, Vertex)
        assert v.realm == realm
        assert v.payload["name"] == "Alice"
        assert v.table_name == "sa_people"

    async def test_add_vertex_with_explicit_id(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        v = await sa_client.add_vertex("sa_people", realm=realm, vertex_id=500, payload={"name": "Bob"})
        assert v.id == "500"

    async def test_get_vertex_by_id(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        v = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "Carol"})
        fetched = await sa_client.get_vertex("sa_people", realm, v.id)
        assert fetched is not None
        assert fetched.id == v.id
        assert fetched.payload["name"] == "Carol"

    async def test_get_vertex_by_uuid(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        v = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "Dan"})
        fetched = await sa_client.get_vertex_by_uuid("sa_people", realm, v.uuid)
        assert fetched is not None
        assert fetched.id == v.id

    async def test_get_vertex_by_fqid(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        v = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "Eve"})
        fetched = await sa_client.get_vertex("sa_people", realm, v.fqid)
        assert fetched is not None
        assert fetched.fqid == v.fqid

    async def test_get_vertex_not_found(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        result = await sa_client.get_vertex("sa_people", realm, "99999")
        assert result is None

    async def test_get_vertices(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "A"})
        await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "B"})
        verts = await sa_client.get_vertices("sa_people", realm)
        assert len(verts) >= 2

    async def test_get_vertices_with_limit(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        for i in range(5):
            await sa_client.add_vertex("sa_people", realm=realm, payload={"n": i})
        verts = await sa_client.get_vertices("sa_people", realm, limit=3)
        assert len(verts) == 3

    async def test_upsert_vertex_insert(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        v = await sa_client.upsert_vertex("sa_people", realm=realm, vertex_id=700, payload={"name": "Upserted"})
        assert v.id == "700"
        assert v.payload["name"] == "Upserted"

    async def test_upsert_vertex_update_merges_payload(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        v1 = await sa_client.add_vertex("sa_people", realm=realm, vertex_id=800, payload={"a": 1})
        v2 = await sa_client.upsert_vertex("sa_people", realm=realm, vertex_id=800, payload={"b": 2})
        assert v2.payload["a"] == 1
        assert v2.payload["b"] == 2

    async def test_delete_vertex(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        v = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "Doomed"})
        deleted = await sa_client.delete_vertex("sa_people", realm, v.id)
        assert deleted is True
        assert await sa_client.get_vertex("sa_people", realm, v.id) is None

    async def test_delete_vertex_not_found(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        deleted = await sa_client.delete_vertex("sa_people", realm, "99999")
        assert deleted is False

    async def test_vertex_has_client_ref(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        v = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "HasClient"})
        assert v._client is sa_client


class TestSpaceIsolation:
    async def test_add_vertex_with_space(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        v = await sa_client.add_vertex("sa_people", realm=realm, payload={"n": 1}, space="prod")
        assert v.space == "prod"

    async def test_get_vertices_filtered_by_space(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        await sa_client.add_vertex("sa_people", realm=realm, payload={"n": 1}, space="prod")
        await sa_client.add_vertex("sa_people", realm=realm, payload={"n": 2}, space="dev")
        prod = await sa_client.get_vertices("sa_people", realm, space="prod")
        assert all(v.space == "prod" for v in prod)

    async def test_get_vertices_all_spaces(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        await sa_client.add_vertex("sa_people", realm=realm, payload={"n": 1}, space="prod")
        await sa_client.add_vertex("sa_people", realm=realm, payload={"n": 2}, space="dev")
        all_v = await sa_client.get_vertices("sa_people", realm, space=RESERVED_SPACE_ALL)
        assert len(all_v) >= 2

    async def test_add_vertex_with_reserved_space_raises(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        with pytest.raises(ReservedSpaceError):
            await sa_client.add_vertex("sa_people", realm=realm, payload={}, space=RESERVED_SPACE_ALL)

    async def test_upsert_vertex_with_reserved_space_raises(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        with pytest.raises(ReservedSpaceError):
            await sa_client.upsert_vertex("sa_people", realm=realm, vertex_id=1, payload={}, space=RESERVED_SPACE_ALL)


class TestEdgeCRUD:
    async def test_add_edge(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        await sa_client.create_edge_table("sa_knows", from_vertex_table="sa_people", to_vertex_table="sa_people", realm=realm)
        a = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "A"})
        b = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "B"})
        e = await sa_client.add_edge("sa_knows", realm=realm, from_id=a.id, to_id=b.id, relation_type="knows")
        assert isinstance(e, Edge)
        assert e.from_id == a.id
        assert e.to_id == b.id

    async def test_add_edge_with_space(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        await sa_client.create_edge_table("sa_knows", from_vertex_table="sa_people", to_vertex_table="sa_people", realm=realm)
        a = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "A"})
        b = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "B"})
        e = await sa_client.add_edge("sa_knows", realm=realm, from_id=a.id, to_id=b.id, relation_type="knows", space="prod")
        assert e.space == "prod"

    async def test_add_edge_reserved_space_raises(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        await sa_client.create_edge_table("sa_knows", from_vertex_table="sa_people", to_vertex_table="sa_people", realm=realm)
        a = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "A"})
        b = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "B"})
        with pytest.raises(ReservedSpaceError):
            await sa_client.add_edge("sa_knows", realm=realm, from_id=a.id, to_id=b.id, relation_type="knows", space=RESERVED_SPACE_ALL)

    async def test_get_edge(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        await sa_client.create_edge_table("sa_knows", from_vertex_table="sa_people", to_vertex_table="sa_people", realm=realm)
        a = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "A"})
        b = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "B"})
        e = await sa_client.add_edge("sa_knows", realm=realm, from_id=a.id, to_id=b.id, relation_type="knows")
        fetched = await sa_client.get_edge("sa_knows", realm, e.id)
        assert fetched is not None
        assert fetched.id == e.id

    async def test_get_edge_by_uuid(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        await sa_client.create_edge_table("sa_knows", from_vertex_table="sa_people", to_vertex_table="sa_people", realm=realm)
        a = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "A"})
        b = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "B"})
        e = await sa_client.add_edge("sa_knows", realm=realm, from_id=a.id, to_id=b.id, relation_type="knows")
        fetched = await sa_client.get_edge_by_uuid("sa_knows", realm, e.uuid)
        assert fetched is not None
        assert fetched.id == e.id

    async def test_upsert_edge(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        await sa_client.create_edge_table("sa_knows", from_vertex_table="sa_people", to_vertex_table="sa_people", realm=realm)
        a = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "A"})
        b = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "B"})
        e1 = await sa_client.add_edge("sa_knows", realm=realm, from_id=a.id, to_id=b.id, relation_type="knows", edge_id=900, payload={"w": 1})
        e2 = await sa_client.upsert_edge("sa_knows", realm=realm, from_id=a.id, to_id=b.id, relation_type="knows", edge_id=900, payload={"x": 2})
        assert e2.payload["w"] == 1
        assert e2.payload["x"] == 2

    async def test_delete_edge(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        await sa_client.create_edge_table("sa_knows", from_vertex_table="sa_people", to_vertex_table="sa_people", realm=realm)
        a = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "A"})
        b = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "B"})
        e = await sa_client.add_edge("sa_knows", realm=realm, from_id=a.id, to_id=b.id, relation_type="knows")
        deleted = await sa_client.delete_edge("sa_knows", realm, e.id)
        assert deleted is True

    async def test_edge_has_client_ref(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        await sa_client.create_edge_table("sa_knows", from_vertex_table="sa_people", to_vertex_table="sa_people", realm=realm)
        a = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "A"})
        b = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "B"})
        e = await sa_client.add_edge("sa_knows", realm=realm, from_id=a.id, to_id=b.id, relation_type="knows")
        assert e._client is sa_client

    async def test_edge_fk_violation_raises(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        await sa_client.create_edge_table("sa_knows", from_vertex_table="sa_people", to_vertex_table="sa_people", realm=realm)
        with pytest.raises(VertexNotFoundError):
            await sa_client.add_edge("sa_knows", realm=realm, from_id="99998", to_id="99999", relation_type="knows")

    async def test_edge_table_requires_vertex_tables(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        with pytest.raises(TableNotFoundError):
            await sa_client.create_edge_table("sa_bad_edge", from_vertex_table="nonexistent", to_vertex_table="nonexistent", realm=realm)

    async def test_default_edge_table_name(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_src", realm=realm)
        await sa_client.create_vertex_table("sa_tgt", realm=realm)
        await sa_client.create_edge_table(from_vertex_table="sa_src", to_vertex_table="sa_tgt", realm=realm)
        a = await sa_client.add_vertex("sa_src", realm=realm, payload={})
        b = await sa_client.add_vertex("sa_tgt", realm=realm, payload={})
        e = await sa_client.add_edge("sa_srcTOsa_tgt", realm=realm, from_id=a.id, to_id=b.id, relation_type="linked")
        assert isinstance(e, Edge)

    async def test_get_edges(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        await sa_client.create_edge_table("sa_knows", from_vertex_table="sa_people", to_vertex_table="sa_people", realm=realm)
        a = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "A"})
        b = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "B"})
        await sa_client.add_edge("sa_knows", realm=realm, from_id=a.id, to_id=b.id, relation_type="friends")
        await sa_client.add_edge("sa_knows", realm=realm, from_id=b.id, to_id=a.id, relation_type="colleagues")
        edges = await sa_client.get_edges("sa_knows", realm=realm)
        assert len(edges) == 2
        assert all(isinstance(e, Edge) for e in edges)

    async def test_get_edges_with_limit(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        await sa_client.create_edge_table("sa_knows", from_vertex_table="sa_people", to_vertex_table="sa_people", realm=realm)
        a = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "A"})
        b = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "B"})
        for i in range(5):
            await sa_client.add_edge("sa_knows", realm=realm, from_id=a.id, to_id=b.id, relation_type=f"r{i}")
        edges = await sa_client.get_edges("sa_knows", realm=realm, limit=2)
        assert len(edges) == 2

    async def test_get_edges_filter_by_relation_type(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        await sa_client.create_edge_table("sa_knows", from_vertex_table="sa_people", to_vertex_table="sa_people", realm=realm)
        a = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "A"})
        b = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "B"})
        await sa_client.add_edge("sa_knows", realm=realm, from_id=a.id, to_id=b.id, relation_type="friends")
        await sa_client.add_edge("sa_knows", realm=realm, from_id=b.id, to_id=a.id, relation_type="colleagues")
        edges = await sa_client.get_edges("sa_knows", realm=realm, relation_type="friends")
        assert len(edges) == 1
        assert edges[0].relation_type == "friends"

    async def test_get_edges_filter_by_space(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        await sa_client.create_edge_table("sa_knows", from_vertex_table="sa_people", to_vertex_table="sa_people", realm=realm)
        a = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "A"})
        b = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "B"})
        await sa_client.add_edge("sa_knows", realm=realm, from_id=a.id, to_id=b.id, relation_type="friends", space="prod")
        await sa_client.add_edge("sa_knows", realm=realm, from_id=b.id, to_id=a.id, relation_type="colleagues", space="staging")
        edges = await sa_client.get_edges("sa_knows", realm=realm, space="prod")
        assert len(edges) == 1
        assert edges[0].space == "prod"

    async def test_get_edges_empty(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        await sa_client.create_edge_table("sa_knows", from_vertex_table="sa_people", to_vertex_table="sa_people", realm=realm)
        edges = await sa_client.get_edges("sa_knows", realm=realm)
        assert edges == []


class TestStrictMode:
    async def test_get_vertex_strict_raises(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        with pytest.raises(VertexNotFoundError):
            await sa_client.get_vertex("sa_people", realm=realm, vertex_id="999999", strict=True)

    async def test_get_vertex_strict_returns_when_found(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        v = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "found"})
        fetched = await sa_client.get_vertex("sa_people", realm=realm, vertex_id=v.id, strict=True)
        assert fetched is not None
        assert fetched.id == v.id

    async def test_get_vertex_by_uuid_strict_raises(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        with pytest.raises(VertexNotFoundError):
            await sa_client.get_vertex_by_uuid("sa_people", realm=realm, uuid="00000000-0000-0000-0000-000000000000", strict=True)

    async def test_get_edge_strict_raises(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        await sa_client.create_edge_table("sa_knows", from_vertex_table="sa_people", to_vertex_table="sa_people", realm=realm)
        with pytest.raises(EdgeNotFoundError):
            await sa_client.get_edge("sa_knows", realm=realm, edge_id="999999", strict=True)

    async def test_get_edge_strict_returns_when_found(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        await sa_client.create_edge_table("sa_knows", from_vertex_table="sa_people", to_vertex_table="sa_people", realm=realm)
        a = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "A"})
        b = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "B"})
        e = await sa_client.add_edge("sa_knows", realm=realm, from_id=a.id, to_id=b.id, relation_type="friends")
        fetched = await sa_client.get_edge("sa_knows", realm=realm, edge_id=e.id, strict=True)
        assert fetched is not None
        assert fetched.id == e.id

    async def test_get_edge_by_uuid_strict_raises(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        await sa_client.create_edge_table("sa_knows", from_vertex_table="sa_people", to_vertex_table="sa_people", realm=realm)
        with pytest.raises(EdgeNotFoundError):
            await sa_client.get_edge_by_uuid("sa_knows", realm=realm, uuid="00000000-0000-0000-0000-000000000000", strict=True)


class TestFindVertices:
    async def test_find_by_payload(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "Alice", "role": "eng"})
        await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "Bob", "role": "pm"})
        await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "Carol", "role": "eng"})
        results = await sa_client.find_vertices("sa_people", realm=realm, filters={"role": "eng"})
        assert len(results) == 2
        names = {v.payload["name"] for v in results}
        assert names == {"Alice", "Carol"}

    async def test_find_multiple_filters(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "Alice", "role": "eng"})
        await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "Bob", "role": "eng"})
        results = await sa_client.find_vertices("sa_people", realm=realm, filters={"role": "eng", "name": "Alice"})
        assert len(results) == 1
        assert results[0].payload["name"] == "Alice"

    async def test_find_no_match(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "Alice"})
        results = await sa_client.find_vertices("sa_people", realm=realm, filters={"name": "Nobody"})
        assert results == []

    async def test_find_with_limit(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        for i in range(5):
            await sa_client.add_vertex("sa_people", realm=realm, payload={"group": "a"})
        results = await sa_client.find_vertices("sa_people", realm=realm, filters={"group": "a"}, limit=2)
        assert len(results) == 2


class TestFindEdges:
    async def test_find_by_payload(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        await sa_client.create_edge_table("sa_knows", from_vertex_table="sa_people", to_vertex_table="sa_people", realm=realm)
        v1 = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "A"})
        v2 = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "B"})
        await sa_client.add_edge("sa_knows", realm=realm, from_id=v1.id, to_id=v2.id, relation_type="friends", payload={"weight": "5"})
        await sa_client.add_edge("sa_knows", realm=realm, from_id=v2.id, to_id=v1.id, relation_type="friends", payload={"weight": "3"})
        results = await sa_client.find_edges("sa_knows", realm=realm, filters={"weight": "5"})
        assert len(results) == 1
        assert results[0].payload["weight"] == "5"

    async def test_find_with_relation_type(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        await sa_client.create_edge_table("sa_knows", from_vertex_table="sa_people", to_vertex_table="sa_people", realm=realm)
        v1 = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "A"})
        v2 = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "B"})
        await sa_client.add_edge("sa_knows", realm=realm, from_id=v1.id, to_id=v2.id, relation_type="friends", payload={"tag": "x"})
        await sa_client.add_edge("sa_knows", realm=realm, from_id=v2.id, to_id=v1.id, relation_type="colleagues", payload={"tag": "x"})
        results = await sa_client.find_edges("sa_knows", realm=realm, filters={"tag": "x"}, relation_type="friends")
        assert len(results) == 1
        assert results[0].relation_type == "friends"

    async def test_find_no_match(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        await sa_client.create_edge_table("sa_knows", from_vertex_table="sa_people", to_vertex_table="sa_people", realm=realm)
        v1 = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "A"})
        v2 = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "B"})
        await sa_client.add_edge("sa_knows", realm=realm, from_id=v1.id, to_id=v2.id, relation_type="friends", payload={"w": "1"})
        results = await sa_client.find_edges("sa_knows", realm=realm, filters={"w": "999"})
        assert results == []


class TestMultiRealm:
    async def test_get_vertices_multi_realm(self, sa_client):
        r1 = f"test_mr_{uuid.uuid4().hex[:8]}"
        r2 = f"test_mr_{uuid.uuid4().hex[:8]}"
        try:
            await sa_client.create_vertex_table("sa_people", realm=r1)
            await sa_client.create_vertex_table("sa_people", realm=r2)
            await sa_client.add_vertex("sa_people", realm=r1, payload={"name": "Alice"})
            await sa_client.add_vertex("sa_people", realm=r2, payload={"name": "Bob"})
            results = await sa_client.get_vertices_multi_realm("sa_people", realms=[r1, r2])
            assert len(results) == 2
            names = {v.payload["name"] for v in results}
            assert names == {"Alice", "Bob"}
        finally:
            await sa_client.delete_realm(r1)
            await sa_client.delete_realm(r2)

    async def test_get_edges_multi_realm(self, sa_client):
        r1 = f"test_mr_{uuid.uuid4().hex[:8]}"
        r2 = f"test_mr_{uuid.uuid4().hex[:8]}"
        try:
            await sa_client.create_vertex_table("sa_people", realm=r1)
            await sa_client.create_vertex_table("sa_people", realm=r2)
            await sa_client.create_edge_table("sa_knows", from_vertex_table="sa_people", to_vertex_table="sa_people", realm=r1)
            await sa_client.create_edge_table("sa_knows", from_vertex_table="sa_people", to_vertex_table="sa_people", realm=r2)
            v1 = await sa_client.add_vertex("sa_people", realm=r1, payload={"name": "A"})
            v2 = await sa_client.add_vertex("sa_people", realm=r1, payload={"name": "B"})
            v3 = await sa_client.add_vertex("sa_people", realm=r2, payload={"name": "C"})
            v4 = await sa_client.add_vertex("sa_people", realm=r2, payload={"name": "D"})
            await sa_client.add_edge("sa_knows", realm=r1, from_id=v1.id, to_id=v2.id, relation_type="friends")
            await sa_client.add_edge("sa_knows", realm=r2, from_id=v3.id, to_id=v4.id, relation_type="colleagues")
            results = await sa_client.get_edges_multi_realm("sa_knows", realms=[r1, r2])
            assert len(results) == 2
            realms_found = {e.realm for e in results}
            assert realms_found == {r1, r2}
        finally:
            await sa_client.delete_realm(r1)
            await sa_client.delete_realm(r2)


class TestBatchUpsert:
    async def test_batch_upsert_vertices(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        items = [
            {"payload": {"name": "Alice"}},
            {"payload": {"name": "Bob"}},
            {"payload": {"name": "Carol"}},
        ]
        results = await sa_client.batch_upsert_vertices("sa_people", realm=realm, items=items)
        assert len(results) == 3
        assert all(isinstance(v, Vertex) for v in results)
        names = {v.payload["name"] for v in results}
        assert names == {"Alice", "Bob", "Carol"}

    async def test_batch_upsert_edges(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        await sa_client.create_edge_table("sa_knows", from_vertex_table="sa_people", to_vertex_table="sa_people", realm=realm)
        v1 = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "A"})
        v2 = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "B"})
        v3 = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "C"})
        items = [
            {"from_id": v1.id, "to_id": v2.id, "relation_type": "friends", "payload": {"w": 1}},
            {"from_id": v2.id, "to_id": v3.id, "relation_type": "colleagues", "payload": {"w": 2}},
        ]
        results = await sa_client.batch_upsert_edges("sa_knows", realm=realm, items=items)
        assert len(results) == 2
        assert all(isinstance(e, Edge) for e in results)
        assert results[0].relation_type == "friends"
        assert results[1].relation_type == "colleagues"


class TestCycleDetection:
    async def test_cycle_raises(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_nodes", realm=realm)
        await sa_client.create_edge_table("sa_links", from_vertex_table="sa_nodes", to_vertex_table="sa_nodes", realm=realm)
        a = await sa_client.add_vertex("sa_nodes", realm=realm, payload={"n": "A"})
        b = await sa_client.add_vertex("sa_nodes", realm=realm, payload={"n": "B"})
        await sa_client.add_edge("sa_links", realm=realm, from_id=a.id, to_id=b.id, relation_type="to")
        with pytest.raises(CyclicReferenceError):
            await sa_client.add_edge("sa_links", realm=realm, from_id=b.id, to_id=a.id, relation_type="to", check_cycle=True)

    async def test_no_false_positive(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_nodes", realm=realm)
        await sa_client.create_edge_table("sa_links", from_vertex_table="sa_nodes", to_vertex_table="sa_nodes", realm=realm)
        a = await sa_client.add_vertex("sa_nodes", realm=realm, payload={"n": "A"})
        b = await sa_client.add_vertex("sa_nodes", realm=realm, payload={"n": "B"})
        c = await sa_client.add_vertex("sa_nodes", realm=realm, payload={"n": "C"})
        await sa_client.add_edge("sa_links", realm=realm, from_id=a.id, to_id=b.id, relation_type="to")
        e = await sa_client.add_edge("sa_links", realm=realm, from_id=b.id, to_id=c.id, relation_type="to", check_cycle=True)
        assert isinstance(e, Edge)


class TestNeighbors:
    async def test_outgoing(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        await sa_client.create_edge_table("sa_knows", from_vertex_table="sa_people", to_vertex_table="sa_people", realm=realm)
        a = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "A"})
        b = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "B"})
        await sa_client.add_edge("sa_knows", realm=realm, from_id=a.id, to_id=b.id, relation_type="knows")
        neighbors = await sa_client.get_neighbors(realm, "sa_people", a.id, ["sa_knows"], direction="out")
        assert len(neighbors) == 1
        assert neighbors[0][0].id == b.id

    async def test_incoming(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        await sa_client.create_edge_table("sa_knows", from_vertex_table="sa_people", to_vertex_table="sa_people", realm=realm)
        a = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "A"})
        b = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "B"})
        await sa_client.add_edge("sa_knows", realm=realm, from_id=a.id, to_id=b.id, relation_type="knows")
        neighbors = await sa_client.get_neighbors(realm, "sa_people", b.id, ["sa_knows"], direction="in")
        assert len(neighbors) == 1
        assert neighbors[0][0].id == a.id

    async def test_both(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        await sa_client.create_edge_table("sa_knows", from_vertex_table="sa_people", to_vertex_table="sa_people", realm=realm)
        a = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "A"})
        b = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "B"})
        c = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "C"})
        await sa_client.add_edge("sa_knows", realm=realm, from_id=a.id, to_id=b.id, relation_type="knows")
        await sa_client.add_edge("sa_knows", realm=realm, from_id=c.id, to_id=b.id, relation_type="knows")
        neighbors = await sa_client.get_neighbors(realm, "sa_people", b.id, ["sa_knows"], direction="both")
        ids = {n[0].id for n in neighbors}
        assert a.id in ids
        assert c.id in ids


class TestTraversal:
    async def test_traverse_depth(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        await sa_client.create_edge_table("sa_knows", from_vertex_table="sa_people", to_vertex_table="sa_people", realm=realm)
        a = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "A"})
        b = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "B"})
        c = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "C"})
        await sa_client.add_edge("sa_knows", realm=realm, from_id=a.id, to_id=b.id, relation_type="knows")
        await sa_client.add_edge("sa_knows", realm=realm, from_id=b.id, to_id=c.id, relation_type="knows")
        results = await sa_client.traverse(realm, "sa_people", a.id, ["sa_knows"], max_depth=1)
        reached = {r["id"] for r in results}
        assert a.id in reached
        assert b.id in reached
        assert c.id not in reached

    async def test_traverse_full(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        await sa_client.create_edge_table("sa_knows", from_vertex_table="sa_people", to_vertex_table="sa_people", realm=realm)
        a = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "A"})
        b = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "B"})
        c = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "C"})
        await sa_client.add_edge("sa_knows", realm=realm, from_id=a.id, to_id=b.id, relation_type="knows")
        await sa_client.add_edge("sa_knows", realm=realm, from_id=b.id, to_id=c.id, relation_type="knows")
        results = await sa_client.traverse(realm, "sa_people", a.id, ["sa_knows"], max_depth=3)
        reached = {r["id"] for r in results}
        assert a.id in reached
        assert b.id in reached
        assert c.id in reached

    async def test_traverse_with_relation_types(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        await sa_client.create_edge_table("sa_knows", from_vertex_table="sa_people", to_vertex_table="sa_people", realm=realm)
        a = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "A"})
        b = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "B"})
        c = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "C"})
        await sa_client.add_edge("sa_knows", realm=realm, from_id=a.id, to_id=b.id, relation_type="knows")
        await sa_client.add_edge("sa_knows", realm=realm, from_id=b.id, to_id=c.id, relation_type="works_with")
        results = await sa_client.traverse(realm, "sa_people", a.id, ["sa_knows"], max_depth=3, relation_types=["knows"])
        reached = {r["id"] for r in results}
        assert b.id in reached
        assert c.id not in reached

    async def test_traverse_with_space(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        await sa_client.create_edge_table("sa_knows", from_vertex_table="sa_people", to_vertex_table="sa_people", realm=realm)
        a = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "A"})
        b = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "B"})
        c = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "C"})
        await sa_client.add_edge("sa_knows", realm=realm, from_id=a.id, to_id=b.id, relation_type="knows", space="prod")
        await sa_client.add_edge("sa_knows", realm=realm, from_id=b.id, to_id=c.id, relation_type="knows", space="dev")
        results = await sa_client.traverse(realm, "sa_people", a.id, ["sa_knows"], max_depth=3, space="prod")
        reached = {r["id"] for r in results}
        assert b.id in reached
        assert c.id not in reached

    async def test_traverse_as_of(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        await sa_client.create_edge_table("sa_knows", from_vertex_table="sa_people", to_vertex_table="sa_people", realm=realm)
        a = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "A"})
        b = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "B"})
        c = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "C"})
        await sa_client.add_edge("sa_knows", realm=realm, from_id=a.id, to_id=b.id, relation_type="k", payload={"valid_from": "2020", "valid_to": "2025"})
        await sa_client.add_edge("sa_knows", realm=realm, from_id=b.id, to_id=c.id, relation_type="k", payload={"valid_from": "2030", "valid_to": "2035"})
        results = await sa_client.traverse(realm, "sa_people", a.id, ["sa_knows"], max_depth=3, as_of="2022")
        reached = {r["id"] for r in results}
        assert b.id in reached
        assert c.id not in reached

    async def test_traverse_as_of_no_dates_always_qualifies(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        await sa_client.create_edge_table("sa_knows", from_vertex_table="sa_people", to_vertex_table="sa_people", realm=realm)
        a = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "A"})
        b = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "B"})
        await sa_client.add_edge("sa_knows", realm=realm, from_id=a.id, to_id=b.id, relation_type="k", payload={})
        results = await sa_client.traverse(realm, "sa_people", a.id, ["sa_knows"], max_depth=3, as_of="2022")
        reached = {r["id"] for r in results}
        assert b.id in reached

    async def test_traverse_with_payload_null_keys(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        await sa_client.create_edge_table("sa_knows", from_vertex_table="sa_people", to_vertex_table="sa_people", realm=realm)
        a = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "A"})
        b = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "B"})
        c = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "C"})
        await sa_client.add_edge("sa_knows", realm=realm, from_id=a.id, to_id=b.id, relation_type="k", payload={})
        await sa_client.add_edge("sa_knows", realm=realm, from_id=b.id, to_id=c.id, relation_type="k", payload={"superseded": "yes"})
        results = await sa_client.traverse(realm, "sa_people", a.id, ["sa_knows"], max_depth=3, payload_null_keys=["superseded"])
        reached = {r["id"] for r in results}
        assert b.id in reached
        assert c.id not in reached


class TestShortestPath:
    async def test_direct_path(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        await sa_client.create_edge_table("sa_knows", from_vertex_table="sa_people", to_vertex_table="sa_people", realm=realm)
        a = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "A"})
        b = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "B"})
        await sa_client.add_edge("sa_knows", realm=realm, from_id=a.id, to_id=b.id, relation_type="knows")
        path = await sa_client.shortest_path(realm, "sa_people", a.id, "sa_people", b.id, ["sa_knows"])
        assert path is not None
        assert path["depth"] == 1

    async def test_multi_hop_path(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        await sa_client.create_edge_table("sa_knows", from_vertex_table="sa_people", to_vertex_table="sa_people", realm=realm)
        a = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "A"})
        b = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "B"})
        c = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "C"})
        await sa_client.add_edge("sa_knows", realm=realm, from_id=a.id, to_id=b.id, relation_type="knows")
        await sa_client.add_edge("sa_knows", realm=realm, from_id=b.id, to_id=c.id, relation_type="knows")
        path = await sa_client.shortest_path(realm, "sa_people", a.id, "sa_people", c.id, ["sa_knows"])
        assert path is not None
        assert path["depth"] == 2

    async def test_no_path(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        await sa_client.create_edge_table("sa_knows", from_vertex_table="sa_people", to_vertex_table="sa_people", realm=realm)
        a = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "A"})
        b = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "B"})
        path = await sa_client.shortest_path(realm, "sa_people", a.id, "sa_people", b.id, ["sa_knows"])
        assert path is None


class TestDataRecords:
    async def test_add_and_get_vertex_data(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        v = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "A"})
        dr = await sa_client.add_vertex_data("sa_people", realm, v.id, payload={"version": 1})
        assert isinstance(dr, DataRecord)
        assert dr.payload["version"] == 1
        records = await sa_client.get_vertex_data("sa_people", realm, v.id)
        assert len(records) >= 1

    async def test_multiple_data_records_ordered(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        v = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "A"})
        await sa_client.add_vertex_data("sa_people", realm, v.id, payload={"v": 1})
        await sa_client.add_vertex_data("sa_people", realm, v.id, payload={"v": 2})
        records = await sa_client.get_vertex_data("sa_people", realm, v.id)
        assert records[0].payload["v"] == 2

    async def test_get_latest_vertex_data(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        v = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "A"})
        await sa_client.add_vertex_data("sa_people", realm, v.id, payload={"v": 1})
        await sa_client.add_vertex_data("sa_people", realm, v.id, payload={"v": 2})
        latest = await sa_client.get_latest_vertex_data("sa_people", realm, v.id)
        assert latest.payload["v"] == 2

    async def test_get_vertex_data_by_id(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        v = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "A"})
        dr = await sa_client.add_vertex_data("sa_people", realm, v.id, payload={"v": 1})
        fetched = await sa_client.get_vertex_data_by_id("sa_people", realm, dr.data_id)
        assert fetched is not None
        assert fetched.data_id == dr.data_id

    async def test_edge_data(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        await sa_client.create_edge_table("sa_knows", from_vertex_table="sa_people", to_vertex_table="sa_people", realm=realm)
        a = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "A"})
        b = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "B"})
        e = await sa_client.add_edge("sa_knows", realm=realm, from_id=a.id, to_id=b.id, relation_type="knows")
        dr = await sa_client.add_edge_data("sa_knows", realm, e.id, payload={"note": "first"})
        assert isinstance(dr, DataRecord)
        records = await sa_client.get_edge_data("sa_knows", realm, e.id)
        assert len(records) >= 1

    async def test_vertex_data_limited(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        v = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "A"})
        for i in range(5):
            await sa_client.add_vertex_data("sa_people", realm, v.id, payload={"v": i})
        records = await sa_client.get_vertex_data("sa_people", realm, v.id, limit=2)
        assert len(records) == 2


class TestDeleteRealm:
    async def test_deletes_all_data(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_nodes", realm=realm)
        await sa_client.create_edge_table("sa_links", from_vertex_table="sa_nodes", to_vertex_table="sa_nodes", realm=realm)
        a = await sa_client.add_vertex("sa_nodes", realm=realm, payload={"n": "A"})
        b = await sa_client.add_vertex("sa_nodes", realm=realm, payload={"n": "B"})
        await sa_client.add_edge("sa_links", realm=realm, from_id=a.id, to_id=b.id, relation_type="to")
        deleted = await sa_client.delete_realm(realm)
        assert deleted >= 3


class TestTraverseMultipleEdgeTables:
    async def test_traverse_across_two_edge_tables(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_multi", realm=realm)
        await sa_client.create_edge_table("sa_mk", from_vertex_table="sa_multi", to_vertex_table="sa_multi", realm=realm)
        await sa_client.create_edge_table("sa_mw", from_vertex_table="sa_multi", to_vertex_table="sa_multi", realm=realm)
        a = await sa_client.add_vertex("sa_multi", realm=realm, payload={"n": "A"})
        b = await sa_client.add_vertex("sa_multi", realm=realm, payload={"n": "B"})
        c = await sa_client.add_vertex("sa_multi", realm=realm, payload={"n": "C"})
        await sa_client.add_edge("sa_mk", realm=realm, from_id=a.id, to_id=b.id, relation_type="knows")
        await sa_client.add_edge("sa_mw", realm=realm, from_id=b.id, to_id=c.id, relation_type="works_with")
        results = await sa_client.traverse(realm, "sa_multi", a.id, ["sa_mk", "sa_mw"], max_depth=2)
        reached = {r["id"] for r in results}
        assert a.id in reached
        assert b.id in reached
        assert c.id in reached


class TestTraverseIncomingDirection:
    async def test_traverse_incoming(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_nodes", realm=realm)
        await sa_client.create_edge_table("sa_links", from_vertex_table="sa_nodes", to_vertex_table="sa_nodes", realm=realm)
        a = await sa_client.add_vertex("sa_nodes", realm=realm, payload={"n": "A"})
        b = await sa_client.add_vertex("sa_nodes", realm=realm, payload={"n": "B"})
        c = await sa_client.add_vertex("sa_nodes", realm=realm, payload={"n": "C"})
        await sa_client.add_edge("sa_links", realm=realm, from_id=a.id, to_id=c.id, relation_type="to")
        await sa_client.add_edge("sa_links", realm=realm, from_id=b.id, to_id=c.id, relation_type="to")
        results = await sa_client.traverse(realm, "sa_nodes", c.id, ["sa_links"], max_depth=1, direction="in")
        reached = {r["id"] for r in results}
        assert a.id in reached
        assert b.id in reached


class TestTraverseCycleProtection:
    async def test_traverse_does_not_loop(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_nodes", realm=realm)
        await sa_client.create_edge_table("sa_links", from_vertex_table="sa_nodes", to_vertex_table="sa_nodes", realm=realm)
        a = await sa_client.add_vertex("sa_nodes", realm=realm, payload={"n": "A"})
        b = await sa_client.add_vertex("sa_nodes", realm=realm, payload={"n": "B"})
        await sa_client.add_edge("sa_links", realm=realm, from_id=a.id, to_id=b.id, relation_type="to")
        await sa_client.add_edge("sa_links", realm=realm, from_id=b.id, to_id=a.id, relation_type="to")
        results = await sa_client.traverse(realm, "sa_nodes", a.id, ["sa_links"], max_depth=10, direction="out")
        unique_ids = {r["id"] for r in results}
        assert a.id in unique_ids
        assert b.id in unique_ids
        assert len(results) <= 4


class TestVectorSearch:
    async def test_vertex_vector_search(self, sa_client, sa_clean_realm, has_pgvector):
        if not has_pgvector:
            pytest.skip("pgvector not available")
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_vecs", realm=realm, vector_dim=3)
        await sa_client.add_vertex("sa_vecs", realm=realm, payload={"n": "A"}, embedding=[1.0, 0.0, 0.0])
        await sa_client.add_vertex("sa_vecs", realm=realm, payload={"n": "B"}, embedding=[0.0, 1.0, 0.0])
        results = await sa_client.vector_search("sa_vecs", realm, [1.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2
        assert results[0][0].payload["n"] == "A"
        assert results[0][1] < results[1][1]

    async def test_vector_search_with_space(self, sa_client, sa_clean_realm, has_pgvector):
        if not has_pgvector:
            pytest.skip("pgvector not available")
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_vecs", realm=realm, vector_dim=3)
        await sa_client.add_vertex("sa_vecs", realm=realm, payload={"n": "A"}, embedding=[1.0, 0.0, 0.0], space="alpha")
        await sa_client.add_vertex("sa_vecs", realm=realm, payload={"n": "B"}, embedding=[0.9, 0.1, 0.0], space="beta")
        results = await sa_client.vector_search("sa_vecs", realm, [1.0, 0.0, 0.0], top_k=5, space="alpha")
        assert len(results) == 1
        assert results[0][0].space == "alpha"

    async def test_vector_search_l2_metric(self, sa_client, sa_clean_realm, has_pgvector):
        if not has_pgvector:
            pytest.skip("pgvector not available")
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_vecs", realm=realm, vector_dim=3)
        await sa_client.add_vertex("sa_vecs", realm=realm, payload={"n": "A"}, embedding=[1.0, 0.0, 0.0])
        results = await sa_client.vector_search("sa_vecs", realm, [1.0, 0.0, 0.0], top_k=1, distance_metric="l2")
        assert len(results) == 1

    async def test_vector_search_data_scope(self, sa_client, sa_clean_realm, has_pgvector):
        if not has_pgvector:
            pytest.skip("pgvector not available")
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_vecs", realm=realm, vector_dim=3)
        v = await sa_client.add_vertex("sa_vecs", realm=realm, payload={"n": "A"}, embedding=[1.0, 0.0, 0.0])
        await sa_client.add_vertex_data("sa_vecs", realm, v.id, payload={"v": 1}, embedding=[0.0, 1.0, 0.0])
        results = await sa_client.vector_search("sa_vecs", realm, [0.0, 1.0, 0.0], top_k=5, search_scope="data")
        assert len(results) >= 1

    async def test_vector_search_both_scope(self, sa_client, sa_clean_realm, has_pgvector):
        if not has_pgvector:
            pytest.skip("pgvector not available")
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_vecs", realm=realm, vector_dim=3)
        v = await sa_client.add_vertex("sa_vecs", realm=realm, payload={"n": "A"}, embedding=[1.0, 0.0, 0.0])
        await sa_client.add_vertex_data("sa_vecs", realm, v.id, payload={"v": 1}, embedding=[0.0, 1.0, 0.0])
        results = await sa_client.vector_search("sa_vecs", realm, [1.0, 0.0, 0.0], top_k=5, search_scope="both")
        assert len(results) >= 1


class TestSchemaPerRealm:
    async def test_create_and_query(self, sa_client_spr, sa_clean_realm_spr):
        realm = sa_clean_realm_spr
        await sa_client_spr.create_vertex_table("people", realm=realm)
        v = await sa_client_spr.add_vertex("people", realm=realm, payload={"name": "SPR"})
        assert isinstance(v, Vertex)
        fetched = await sa_client_spr.get_vertex("people", realm, v.id)
        assert fetched is not None
        assert fetched.payload["name"] == "SPR"

    async def test_edge_operations(self, sa_client_spr, sa_clean_realm_spr):
        realm = sa_clean_realm_spr
        await sa_client_spr.create_vertex_table("people", realm=realm)
        await sa_client_spr.create_edge_table("knows", from_vertex_table="people", to_vertex_table="people", realm=realm)
        a = await sa_client_spr.add_vertex("people", realm=realm, payload={"name": "A"})
        b = await sa_client_spr.add_vertex("people", realm=realm, payload={"name": "B"})
        e = await sa_client_spr.add_edge("knows", realm=realm, from_id=a.id, to_id=b.id, relation_type="knows")
        assert isinstance(e, Edge)

    async def test_traverse_in_schema_per_realm(self, sa_client_spr, sa_clean_realm_spr):
        realm = sa_clean_realm_spr
        await sa_client_spr.create_vertex_table("people", realm=realm)
        await sa_client_spr.create_edge_table("knows", from_vertex_table="people", to_vertex_table="people", realm=realm)
        a = await sa_client_spr.add_vertex("people", realm=realm, payload={"name": "A"})
        b = await sa_client_spr.add_vertex("people", realm=realm, payload={"name": "B"})
        await sa_client_spr.add_edge("knows", realm=realm, from_id=a.id, to_id=b.id, relation_type="knows")
        results = await sa_client_spr.traverse(realm, "people", a.id, ["knows"], max_depth=1)
        reached = {r["id"] for r in results}
        assert b.id in reached

    async def test_data_records_in_schema_per_realm(self, sa_client_spr, sa_clean_realm_spr):
        realm = sa_clean_realm_spr
        await sa_client_spr.create_vertex_table("people", realm=realm)
        v = await sa_client_spr.add_vertex("people", realm=realm, payload={"name": "A"})
        dr = await sa_client_spr.add_vertex_data("people", realm, v.id, payload={"v": 1})
        assert isinstance(dr, DataRecord)
        latest = await sa_client_spr.get_latest_vertex_data("people", realm, v.id)
        assert latest.payload["v"] == 1


class TestVertexAutoId:
    async def test_auto_id_increments(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        v1 = await sa_client.add_vertex("sa_people", realm=realm, payload={"n": 1})
        v2 = await sa_client.add_vertex("sa_people", realm=realm, payload={"n": 2})
        assert int(v2.id) > int(v1.id)

    async def test_auto_id_after_explicit_id(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        v1 = await sa_client.add_vertex("sa_people", realm=realm, vertex_id=100, payload={"n": 1})
        v2 = await sa_client.add_vertex("sa_people", realm=realm, payload={"n": 2})
        assert int(v2.id) > int(v1.id)


class TestVertexEmbedding:
    async def test_vertex_embedding_roundtrip(self, sa_client, sa_clean_realm, has_pgvector):
        if not has_pgvector:
            pytest.skip("pgvector not available")
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_vecs", realm=realm, vector_dim=3)
        emb = [0.1, 0.2, 0.3]
        v = await sa_client.add_vertex("sa_vecs", realm=realm, payload={"n": "A"}, embedding=emb)
        assert v.embedding is not None
        assert len(v.embedding) == 3
        for a, b in zip(v.embedding, emb):
            assert abs(a - b) < 1e-6

    async def test_upsert_vertex_updates_embedding(self, sa_client, sa_clean_realm, has_pgvector):
        if not has_pgvector:
            pytest.skip("pgvector not available")
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_vecs", realm=realm, vector_dim=3)
        v = await sa_client.add_vertex("sa_vecs", realm=realm, vertex_id=50, payload={"n": "A"}, embedding=[1.0, 0.0, 0.0])
        v2 = await sa_client.upsert_vertex("sa_vecs", realm=realm, vertex_id=50, payload={"n": "A"}, embedding=[0.0, 1.0, 0.0])
        assert v2.embedding is not None
        assert abs(v2.embedding[1] - 1.0) < 1e-6


class TestEdgeExplicitId:
    async def test_edge_with_explicit_id(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        await sa_client.create_edge_table("sa_knows", from_vertex_table="sa_people", to_vertex_table="sa_people", realm=realm)
        a = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "A"})
        b = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "B"})
        e = await sa_client.add_edge("sa_knows", realm=realm, from_id=a.id, to_id=b.id, relation_type="knows", edge_id=42)
        assert e.id == "42"


class TestCascadeDeleteFromVertex:
    async def test_deleting_vertex_cascades_edges(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        await sa_client.create_edge_table("sa_knows", from_vertex_table="sa_people", to_vertex_table="sa_people", realm=realm)
        a = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "A"})
        b = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "B"})
        e = await sa_client.add_edge("sa_knows", realm=realm, from_id=a.id, to_id=b.id, relation_type="knows")
        await sa_client.delete_vertex("sa_people", realm, a.id)
        edge = await sa_client.get_edge("sa_knows", realm, e.id)
        assert edge is None


class TestEdgeUpsertReservedSpace:
    async def test_upsert_edge_reserved_space_raises(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        await sa_client.create_edge_table("sa_knows", from_vertex_table="sa_people", to_vertex_table="sa_people", realm=realm)
        a = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "A"})
        b = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "B"})
        with pytest.raises(ReservedSpaceError):
            await sa_client.upsert_edge("sa_knows", realm=realm, from_id=a.id, to_id=b.id, relation_type="knows", space=RESERVED_SPACE_ALL)


class TestVertexGetByInvalidUuid:
    async def test_invalid_uuid_returns_none(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        result = await sa_client.get_vertex_by_uuid("sa_people", realm, "not-a-valid-uuid-at-all-nope-1")
        assert result is None


class TestEdgeGetByInvalidUuid:
    async def test_invalid_uuid_returns_none(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        await sa_client.create_edge_table("sa_knows", from_vertex_table="sa_people", to_vertex_table="sa_people", realm=realm)
        result = await sa_client.get_edge_by_uuid("sa_knows", realm, "not-a-valid-uuid-at-all-nope-2")
        assert result is None


class TestGetVerticesEmptyRealm:
    async def test_empty_result(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        verts = await sa_client.get_vertices("sa_people", realm)
        realm_verts = [v for v in verts if v.realm == realm]
        assert realm_verts == []


class TestGetVertexDataEmpty:
    async def test_no_data_returns_empty(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        v = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "A"})
        records = await sa_client.get_vertex_data("sa_people", realm, v.id)
        assert records == []

    async def test_get_latest_data_returns_none_when_empty(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        v = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "A"})
        result = await sa_client.get_latest_vertex_data("sa_people", realm, v.id)
        assert result is None

    async def test_get_data_by_nonexistent_id(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        result = await sa_client.get_vertex_data_by_id("sa_people", realm, 99999)
        assert result is None


class TestVertexPayloadTypes:
    async def test_nested_json_payload(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        payload = {"name": "A", "meta": {"tags": ["x", "y"], "count": 42}}
        v = await sa_client.add_vertex("sa_people", realm=realm, payload=payload)
        assert v.payload["meta"]["tags"] == ["x", "y"]

    async def test_empty_payload(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        v = await sa_client.add_vertex("sa_people", realm=realm, payload={})
        assert v.payload == {}

    async def test_null_payload_defaults_to_empty(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        v = await sa_client.add_vertex("sa_people", realm=realm, payload=None)
        assert v.payload == {}


class TestUpsertVertexSpaceUpdate:
    async def test_upsert_updates_space(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        v1 = await sa_client.add_vertex("sa_people", realm=realm, vertex_id=300, payload={}, space="old")
        v2 = await sa_client.upsert_vertex("sa_people", realm=realm, vertex_id=300, payload={}, space="new")
        assert v2.space == "new"


class TestEdgeGetNotFound:
    async def test_get_edge_nonexistent(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        await sa_client.create_edge_table("sa_knows", from_vertex_table="sa_people", to_vertex_table="sa_people", realm=realm)
        result = await sa_client.get_edge("sa_knows", realm, "99999")
        assert result is None

    async def test_delete_edge_not_found(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        await sa_client.create_edge_table("sa_knows", from_vertex_table="sa_people", to_vertex_table="sa_people", realm=realm)
        deleted = await sa_client.delete_edge("sa_knows", realm, "99999")
        assert deleted is False


class TestVertexFqidFormat:
    async def test_fqid_format(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        v = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "A"})
        assert v.fqid == f"{realm}/sa_people/{v.id}"


class TestEdgeFqidFormat:
    async def test_edge_fqid_format(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        await sa_client.create_edge_table("sa_knows", from_vertex_table="sa_people", to_vertex_table="sa_people", realm=realm)
        a = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "A"})
        b = await sa_client.add_vertex("sa_people", realm=realm, payload={"name": "B"})
        e = await sa_client.add_edge("sa_knows", realm=realm, from_id=a.id, to_id=b.id, relation_type="knows")
        assert e.fqid == f"{realm}/sa_people-sa_people/{e.id}"


class TestEdgeSchema:
    async def test_get_edge_schema(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        await sa_client.create_edge_table("sa_knows", from_vertex_table="sa_people", to_vertex_table="sa_people", realm=realm)
        from sqlalchemy import text
        async with sa_client.engine_or_connection.connect() as conn:
            schema = await sa_client.get_edge_schema(conn, "sa_knows")
            assert schema["from_id"] == "sa_people"
            assert schema["to_id"] == "sa_people"

    async def test_get_edge_schema_caching(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_people", realm=realm)
        await sa_client.create_edge_table("sa_knows", from_vertex_table="sa_people", to_vertex_table="sa_people", realm=realm)
        from sqlalchemy import text
        async with sa_client.engine_or_connection.connect() as conn:
            s1 = await sa_client.get_edge_schema(conn, "sa_knows")
            s2 = await sa_client.get_edge_schema(conn, "sa_knows")
            assert s1 is s2


class TestFulltextSearch:
    @pytest.fixture(autouse=True)
    async def _setup(self, sa_client, sa_clean_realm):
        self.client = sa_client
        self.realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_ft_verts")
        self.v1 = await sa_client.add_vertex("sa_ft_verts", sa_clean_realm,
                                              payload={"title": "PostgreSQL Performance Tuning", "body": "Indexes improve query speed"})
        self.v2 = await sa_client.add_vertex("sa_ft_verts", sa_clean_realm,
                                              payload={"title": "Python Web Development", "body": "Flask and Django are popular frameworks"})
        self.v3 = await sa_client.add_vertex("sa_ft_verts", sa_clean_realm,
                                              payload={"title": "Graph Databases Overview", "body": "Vertices and edges form a graph structure"})
        await sa_client.create_edge_table("sa_ft_edges", from_vertex_table="sa_ft_verts", to_vertex_table="sa_ft_verts")
        await sa_client.add_edge("sa_ft_edges", sa_clean_realm, from_id=self.v1.id, to_id=self.v2.id,
                                 relation_type="related_to",
                                 payload={"note": "PostgreSQL can power Python web apps"})
        await sa_client.add_edge("sa_ft_edges", sa_clean_realm, from_id=self.v2.id, to_id=self.v3.id,
                                 relation_type="related_to",
                                 payload={"note": "Graph databases use different paradigms"})

    @pytest.mark.asyncio
    async def test_search_all_fields(self):
        results = await self.client.fulltext_search_vertices("sa_ft_verts", self.realm, "PostgreSQL")
        assert len(results) >= 1
        assert any(r.id == self.v1.id for r in results)

    @pytest.mark.asyncio
    async def test_search_specific_field(self):
        results = await self.client.fulltext_search_vertices("sa_ft_verts", self.realm, "popular frameworks", fields=["body"])
        assert len(results) >= 1
        assert any(r.id == self.v2.id for r in results)

    @pytest.mark.asyncio
    async def test_search_no_match(self):
        results = await self.client.fulltext_search_vertices("sa_ft_verts", self.realm, "kubernetes containerization")
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_edges(self):
        results = await self.client.fulltext_search_edges("sa_ft_edges", self.realm, "PostgreSQL Python")
        assert len(results) >= 1
        found_ids = {r.from_id for r in results}
        assert self.v1.id in found_ids

    @pytest.mark.asyncio
    async def test_search_edges_specific_field(self):
        results = await self.client.fulltext_search_edges("sa_ft_edges", self.realm, "graph paradigms", fields=["note"])
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_search_with_limit(self):
        results = await self.client.fulltext_search_vertices("sa_ft_verts", self.realm, "PostgreSQL graph", limit=1)
        assert len(results) <= 1


class TestMultiVector:
    @pytest.fixture(autouse=True)
    async def _setup(self, sa_client, sa_clean_realm):
        self.client = sa_client
        self.realm = sa_clean_realm
        await sa_client.create_vertex_table(
            "sa_mv_verts",
            vector_columns={"entity_emb": 3, "fact_emb": 3},
        )

    @pytest.mark.asyncio
    async def test_add_vertex_with_named_embeddings(self):
        v = await self.client.add_vertex(
            "sa_mv_verts", self.realm,
            payload={"name": "test"},
            embeddings={"entity_emb": [0.1, 0.2, 0.3], "fact_emb": [0.4, 0.5, 0.6]},
        )
        assert v.embeddings is not None
        assert "entity_emb" in v.embeddings

    @pytest.mark.asyncio
    async def test_upsert_vertex_with_named_embeddings(self):
        v = await self.client.upsert_vertex(
            "sa_mv_verts", self.realm,
            payload={"name": "upserted"},
            embeddings={"entity_emb": [0.7, 0.8, 0.9]},
        )
        assert v.embeddings is not None
        assert "entity_emb" in v.embeddings

    @pytest.mark.asyncio
    async def test_vector_search_on_named_column(self):
        await self.client.add_vertex(
            "sa_mv_verts", self.realm,
            payload={"name": "close"},
            embeddings={"entity_emb": [0.1, 0.2, 0.3]},
        )
        await self.client.add_vertex(
            "sa_mv_verts", self.realm,
            payload={"name": "far"},
            embeddings={"entity_emb": [0.9, 0.8, 0.7]},
        )
        results = await self.client.vector_search(
            "sa_mv_verts", self.realm,
            query_vector=[0.1, 0.2, 0.3],
            top_k=2,
            column_name="entity_emb",
        )
        assert len(results) == 2
        assert results[0][0].payload["name"] == "close"

    @pytest.mark.asyncio
    async def test_search_different_columns_return_different_order(self):
        await self.client.add_vertex(
            "sa_mv_verts", self.realm,
            payload={"name": "a"},
            embeddings={"entity_emb": [1.0, 0.0, 0.0], "fact_emb": [0.0, 0.0, 1.0]},
        )
        await self.client.add_vertex(
            "sa_mv_verts", self.realm,
            payload={"name": "b"},
            embeddings={"entity_emb": [0.0, 0.0, 1.0], "fact_emb": [1.0, 0.0, 0.0]},
        )
        entity_results = await self.client.vector_search(
            "sa_mv_verts", self.realm, [1.0, 0.0, 0.0], top_k=2, column_name="entity_emb",
        )
        fact_results = await self.client.vector_search(
            "sa_mv_verts", self.realm, [1.0, 0.0, 0.0], top_k=2, column_name="fact_emb",
        )
        assert entity_results[0][0].payload["name"] == "a"
        assert fact_results[0][0].payload["name"] == "b"


class TestPoolConfig:
    @pytest.mark.asyncio
    async def test_pool_status(self, sa_client):
        status = sa_client.get_pool_status()
        assert status is not None
        assert "size" in status
        assert "checked_in" in status

    @pytest.mark.asyncio
    async def test_from_dsn_factory(self):
        import os
        dsn = os.environ.get("POST_GRAPH_TEST_DSN", "postgresql+asyncpg://crajah@localhost/post_graph_test")
        if not dsn.startswith("postgresql+asyncpg"):
            dsn = dsn.replace("postgresql://", "postgresql+asyncpg://")
        client = SQLAlchemyPostGraph.from_dsn(dsn, pool_size=2, max_overflow=3)
        status = client.get_pool_status()
        assert status is not None
        await client.engine_or_connection.dispose()


class TestWeightedShortestPath:
    @pytest.fixture(autouse=True)
    async def _setup(self, sa_client, sa_clean_realm):
        self.client = sa_client
        self.realm = sa_clean_realm
        await sa_client.create_vertex_table("sa_wsp_nodes")
        self.a = await sa_client.add_vertex("sa_wsp_nodes", sa_clean_realm, payload={"name": "A"})
        self.b = await sa_client.add_vertex("sa_wsp_nodes", sa_clean_realm, payload={"name": "B"})
        self.c = await sa_client.add_vertex("sa_wsp_nodes", sa_clean_realm, payload={"name": "C"})
        self.d = await sa_client.add_vertex("sa_wsp_nodes", sa_clean_realm, payload={"name": "D"})
        await sa_client.create_edge_table("sa_wsp_edges", from_vertex_table="sa_wsp_nodes", to_vertex_table="sa_wsp_nodes")
        await sa_client.add_edge("sa_wsp_edges", sa_clean_realm, self.a.id, self.b.id, relation_type="road",
                                 payload={"weight": 1.0})
        await sa_client.add_edge("sa_wsp_edges", sa_clean_realm, self.b.id, self.c.id, relation_type="road",
                                 payload={"weight": 1.0})
        await sa_client.add_edge("sa_wsp_edges", sa_clean_realm, self.a.id, self.c.id, relation_type="road",
                                 payload={"weight": 10.0})
        await sa_client.add_edge("sa_wsp_edges", sa_clean_realm, self.c.id, self.d.id, relation_type="road",
                                 payload={"weight": 1.0})

    @pytest.mark.asyncio
    async def test_weighted_shortest_path_prefers_light_edges(self):
        result = await self.client.weighted_shortest_path(
            self.realm, "sa_wsp_nodes", self.a.id, "sa_wsp_nodes", self.c.id,
            edge_tables=["sa_wsp_edges"],
        )
        assert result is not None
        assert result["total_weight"] == 2.0
        assert result["depth"] == 2

    @pytest.mark.asyncio
    async def test_weighted_shortest_path_no_path(self):
        e = await self.client.add_vertex("sa_wsp_nodes", self.realm, payload={"name": "E"})
        result = await self.client.weighted_shortest_path(
            self.realm, "sa_wsp_nodes", self.a.id, "sa_wsp_nodes", e.id,
            edge_tables=["sa_wsp_edges"],
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_weighted_path_returns_total_weight(self):
        result = await self.client.weighted_shortest_path(
            self.realm, "sa_wsp_nodes", self.a.id, "sa_wsp_nodes", self.d.id,
            edge_tables=["sa_wsp_edges"],
        )
        assert result is not None
        assert result["total_weight"] == 3.0
        assert result["depth"] == 3


class TestConnectionMode:
    async def test_works_with_async_connection(self, sa_engine, sa_clean_realm):
        """Test that SQLAlchemyPostGraph works when given an AsyncConnection."""
        realm = sa_clean_realm
        async with sa_engine.connect() as conn:
            async with conn.begin():
                client = SQLAlchemyPostGraph(conn)
                await client.create_vertex_table("sa_conn_test", realm=realm)
                v = await client.add_vertex("sa_conn_test", realm=realm, payload={"n": 1})
                assert isinstance(v, Vertex)
                fetched = await client.get_vertex("sa_conn_test", realm, v.id)
                assert fetched is not None
