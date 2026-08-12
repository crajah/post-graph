"""Tests for the post_graph.models module — pure unit tests, no database needed."""

import pytest
from datetime import datetime, timezone

from post_graph.models import Vertex, Edge, DataRecord, TraversalStep
from post_graph.errors import PostGraphError


# ---------------------------------------------------------------------------
# Vertex
# ---------------------------------------------------------------------------

class TestVertex:
    def test_defaults(self):
        v = Vertex(realm="acme", id="1")
        assert v.realm == "acme"
        assert v.id == "1"
        assert v.space == "default"
        assert v.payload == {}
        assert v.created_at is None
        assert v.updated_at is None
        assert v.table_name is None
        assert v.embedding is None
        assert v.uuid is None
        assert v._client is None

    def test_fqid_auto_generated(self):
        v = Vertex(realm="acme", id="42", table_name="people")
        assert v.fqid == "acme/people/42"

    def test_fqid_not_generated_without_table(self):
        v = Vertex(realm="acme", id="1")
        assert v.fqid is None

    def test_fqid_not_overwritten_when_provided(self):
        v = Vertex(realm="acme", id="1", table_name="people", fqid="custom/fqid/1")
        assert v.fqid == "custom/fqid/1"

    def test_to_dict(self):
        now = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        v = Vertex(
            realm="acme", id="1", space="prod", table_name="people",
            payload={"name": "Alice"}, created_at=now, updated_at=now,
            embedding=[0.1, 0.2], uuid="abc-123"
        )
        d = v.to_dict()
        assert d["realm"] == "acme"
        assert d["id"] == "1"
        assert d["space"] == "prod"
        assert d["payload"] == {"name": "Alice"}
        assert d["created_at"] == now.isoformat()
        assert d["updated_at"] == now.isoformat()
        assert d["embedding"] == [0.1, 0.2]
        assert d["uuid"] == "abc-123"
        assert d["table_name"] == "people"
        assert d["fqid"] == "acme/people/1"

    def test_to_dict_none_dates(self):
        v = Vertex(realm="r", id="1")
        d = v.to_dict()
        assert d["created_at"] is None
        assert d["updated_at"] is None

    def test_equality_ignores_client(self):
        v1 = Vertex(realm="r", id="1", _client="mock_a")
        v2 = Vertex(realm="r", id="1", _client="mock_b")
        assert v1 == v2

    def test_repr_excludes_client(self):
        v = Vertex(realm="r", id="1", _client="secret")
        assert "secret" not in repr(v)


class TestVertexWithoutClient:
    """All traversal / mutation methods require a _client reference."""

    def _vertex(self):
        return Vertex(realm="r", id="1", table_name="t")

    @pytest.mark.asyncio
    async def test_to_raises(self):
        with pytest.raises(PostGraphError, match="client"):
            await self._vertex().to("edges")

    @pytest.mark.asyncio
    async def test_from_raises(self):
        with pytest.raises(PostGraphError, match="client"):
            await self._vertex().from_("edges")

    @pytest.mark.asyncio
    async def test_incoming_raises(self):
        with pytest.raises(PostGraphError, match="client"):
            await self._vertex().incoming("edges")

    @pytest.mark.asyncio
    async def test_outgoing_raises(self):
        with pytest.raises(PostGraphError, match="client"):
            await self._vertex().outgoing("edges")

    @pytest.mark.asyncio
    async def test_add_edge_to_raises(self):
        with pytest.raises(PostGraphError, match="client"):
            await self._vertex().add_edge_to(to_id="2", edge_table="e")

    @pytest.mark.asyncio
    async def test_add_edge_from_raises(self):
        with pytest.raises(PostGraphError, match="client"):
            await self._vertex().add_edge_from(from_id="2", edge_table="e")

    @pytest.mark.asyncio
    async def test_delete_raises(self):
        with pytest.raises(PostGraphError, match="client"):
            await self._vertex().delete()

    @pytest.mark.asyncio
    async def test_add_data_raises(self):
        with pytest.raises(PostGraphError, match="client"):
            await self._vertex().add_data(payload={"a": 1})

    @pytest.mark.asyncio
    async def test_get_data_raises(self):
        with pytest.raises(PostGraphError, match="client"):
            await self._vertex().get_data()

    @pytest.mark.asyncio
    async def test_get_latest_data_raises(self):
        with pytest.raises(PostGraphError, match="client"):
            await self._vertex().get_latest_data()

    @pytest.mark.asyncio
    async def test_get_data_by_id_raises(self):
        with pytest.raises(PostGraphError, match="client"):
            await self._vertex().get_data_by_id(data_id=1)


# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------

class TestEdge:
    def test_defaults(self):
        e = Edge(realm="r", id="1", from_id="10", to_id="20", relation_type="knows")
        assert e.space == "default"
        assert e.payload == {}
        assert e.embedding is None

    def test_fqid_auto_generated(self):
        e = Edge(realm="r", id="5", from_id="1", to_id="2",
                 relation_type="rel", table_name="edges")
        assert e.fqid == "r/edges/5"

    def test_fqid_not_generated_without_table(self):
        e = Edge(realm="r", id="1", from_id="1", to_id="2", relation_type="rel")
        assert e.fqid is None

    def test_to_dict(self):
        now = datetime(2025, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
        e = Edge(
            realm="r", id="3", from_id="1", to_id="2", relation_type="knows",
            space="prod", table_name="edges", payload={"w": 0.9},
            created_at=now, updated_at=now, embedding=[0.5], uuid="u-1"
        )
        d = e.to_dict()
        assert d["from_id"] == "1"
        assert d["to_id"] == "2"
        assert d["relation_type"] == "knows"
        assert d["payload"] == {"w": 0.9}
        assert d["embedding"] == [0.5]
        assert d["space"] == "prod"

    def test_equality_ignores_client(self):
        base = dict(realm="r", id="1", from_id="a", to_id="b", relation_type="x")
        e1 = Edge(**base, _client="c1")
        e2 = Edge(**base, _client="c2")
        assert e1 == e2


class TestEdgeWithoutClient:
    def _edge(self):
        return Edge(realm="r", id="1", from_id="1", to_id="2",
                    relation_type="rel", table_name="edges")

    @pytest.mark.asyncio
    async def test_delete_raises(self):
        with pytest.raises(PostGraphError, match="client"):
            await self._edge().delete()

    @pytest.mark.asyncio
    async def test_add_data_raises(self):
        with pytest.raises(PostGraphError, match="client"):
            await self._edge().add_data(payload={"a": 1})

    @pytest.mark.asyncio
    async def test_get_data_raises(self):
        with pytest.raises(PostGraphError, match="client"):
            await self._edge().get_data()

    @pytest.mark.asyncio
    async def test_get_latest_data_raises(self):
        with pytest.raises(PostGraphError, match="client"):
            await self._edge().get_latest_data()

    @pytest.mark.asyncio
    async def test_get_data_by_id_raises(self):
        with pytest.raises(PostGraphError, match="client"):
            await self._edge().get_data_by_id(data_id=1)


# ---------------------------------------------------------------------------
# DataRecord
# ---------------------------------------------------------------------------

class TestDataRecord:
    def test_defaults(self):
        dr = DataRecord(data_id="1", realm="r", id="10")
        assert dr.space == "default"
        assert dr.payload == {}
        assert dr.timestamp is None
        assert dr.embedding is None

    def test_to_dict(self):
        now = datetime(2025, 3, 1, tzinfo=timezone.utc)
        dr = DataRecord(
            data_id="7", realm="r", id="10", space="staging",
            payload={"v": 2}, timestamp=now, embedding=[1.0, 2.0]
        )
        d = dr.to_dict()
        assert d["data_id"] == "7"
        assert d["space"] == "staging"
        assert d["timestamp"] == now.isoformat()
        assert d["embedding"] == [1.0, 2.0]


# ---------------------------------------------------------------------------
# TraversalStep
# ---------------------------------------------------------------------------

class TestTraversalStep:
    def _step(self):
        v = Vertex(realm="r", id="2", table_name="people")
        e = Edge(realm="r", id="1", from_id="1", to_id="2",
                 relation_type="knows", table_name="edges")
        return TraversalStep(edge=e, neighbor_vertex=v)

    def test_vertex_accessor(self):
        step = self._step()
        assert step.vertex() is step.neighbor_vertex
        assert step.vertex().id == "2"

    @pytest.mark.asyncio
    async def test_add_edge_to_delegates(self):
        step = self._step()
        with pytest.raises(PostGraphError, match="client"):
            await step.add_edge_to(to_id="3", edge_table="e")

    @pytest.mark.asyncio
    async def test_add_edge_from_delegates(self):
        step = self._step()
        with pytest.raises(PostGraphError, match="client"):
            await step.add_edge_from(from_id="3", edge_table="e")
