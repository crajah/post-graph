"""Integration tests proving realm isolation in post-graph.

Verifies that data in one realm is invisible from another realm's queries,
covering vertices, edges, traversal, vector search, and delete_realm.
"""

import uuid

import pytest
import pytest_asyncio

from conftest import requires_pg

pytestmark = [requires_pg, pytest.mark.asyncio(loop_scope="session")]


@pytest_asyncio.fixture()
async def two_realms(pg_client):
    """Create two isolated realms with identical table schemas but different data."""
    realm_a = f"realm_a_{uuid.uuid4().hex[:8]}"
    realm_b = f"realm_b_{uuid.uuid4().hex[:8]}"

    await pg_client.create_vertex_table("people", realm=realm_a)
    await pg_client.create_vertex_table("people", realm=realm_b)
    await pg_client.create_edge_table(
        "knows", from_vertex_table="people", to_vertex_table="people", realm=realm_a
    )
    await pg_client.create_edge_table(
        "knows", from_vertex_table="people", to_vertex_table="people", realm=realm_b
    )

    yield realm_a, realm_b

    for r in (realm_a, realm_b):
        try:
            await pg_client.delete_realm(r)
        except Exception:
            pass


@pytest_asyncio.fixture()
async def two_realms_vector(pg_client, has_pgvector):
    """Two realms with vector-enabled tables for similarity search tests."""
    if not has_pgvector:
        pytest.skip("pgvector not available")

    tag = uuid.uuid4().hex[:6]
    vtable = f"vitems_{tag}"
    etable = f"vlinks_{tag}"
    realm_a = f"vec_a_{uuid.uuid4().hex[:8]}"
    realm_b = f"vec_b_{uuid.uuid4().hex[:8]}"

    for r in (realm_a, realm_b):
        await pg_client.create_vertex_table(vtable, realm=r, vector_dim=3)
        await pg_client.create_edge_table(
            etable,
            from_vertex_table=vtable,
            to_vertex_table=vtable,
            realm=r,
            vector_dim=3,
        )

    yield realm_a, realm_b, vtable, etable

    for r in (realm_a, realm_b):
        try:
            await pg_client.delete_realm(r)
        except Exception:
            pass
    for tbl in (etable, f"{etable}_audit", f"{etable}_data",
                vtable, f"{vtable}_audit", f"{vtable}_data"):
        try:
            await pg_client._execute(f'DROP TABLE IF EXISTS "{tbl}" CASCADE')
        except Exception:
            pass


class TestVertexRealmIsolation:
    async def test_vertex_invisible_across_realms(self, pg_client, two_realms):
        realm_a, realm_b = two_realms

        va = await pg_client.add_vertex("people", realm_a, payload={"name": "Alice"})
        assert va.realm == realm_a

        vb = await pg_client.add_vertex("people", realm_b, payload={"name": "Bob"})
        assert vb.realm == realm_b

        vertices_a = await pg_client.get_vertices("people", realm_a)
        vertices_b = await pg_client.get_vertices("people", realm_b)

        a_ids = {v.id for v in vertices_a}
        b_ids = {v.id for v in vertices_b}

        assert va.id in a_ids
        assert vb.id not in a_ids
        assert vb.id in b_ids
        assert va.id not in b_ids

    async def test_get_vertex_wrong_realm_returns_none(self, pg_client, two_realms):
        realm_a, realm_b = two_realms

        va = await pg_client.add_vertex("people", realm_a, payload={"name": "Alice"})

        result = await pg_client.get_vertex("people", realm_b, va.id)
        assert result is None

    async def test_same_id_different_realms(self, pg_client, two_realms):
        realm_a, realm_b = two_realms

        va = await pg_client.add_vertex(
            "people", realm_a, vertex_id=999, payload={"name": "Alice"}
        )
        vb = await pg_client.add_vertex(
            "people", realm_b, vertex_id=999, payload={"name": "Bob"}
        )

        assert va.id == vb.id == "999"

        fetched_a = await pg_client.get_vertex("people", realm_a, "999")
        fetched_b = await pg_client.get_vertex("people", realm_b, "999")

        assert fetched_a.payload["name"] == "Alice"
        assert fetched_b.payload["name"] == "Bob"


class TestEdgeRealmIsolation:
    async def test_edge_invisible_across_realms(self, pg_client, two_realms):
        realm_a, realm_b = two_realms

        a1 = await pg_client.add_vertex("people", realm_a, payload={"name": "A1"})
        a2 = await pg_client.add_vertex("people", realm_a, payload={"name": "A2"})
        edge_a = await pg_client.add_edge(
            "knows", realm_a, from_id=a1.id, to_id=a2.id, relation_type="knows"
        )

        b1 = await pg_client.add_vertex("people", realm_b, payload={"name": "B1"})
        b2 = await pg_client.add_vertex("people", realm_b, payload={"name": "B2"})

        fetched = await pg_client.get_edge("knows", realm_b, edge_a.id)
        assert fetched is None

    async def test_delete_edge_wrong_realm(self, pg_client, two_realms):
        realm_a, realm_b = two_realms

        a1 = await pg_client.add_vertex("people", realm_a, payload={"name": "A1"})
        a2 = await pg_client.add_vertex("people", realm_a, payload={"name": "A2"})
        edge_a = await pg_client.add_edge(
            "knows", realm_a, from_id=a1.id, to_id=a2.id, relation_type="knows"
        )

        deleted = await pg_client.delete_edge("knows", realm_b, edge_a.id)
        assert not deleted

        still_there = await pg_client.get_edge("knows", realm_a, edge_a.id)
        assert still_there is not None


class TestTraversalRealmIsolation:
    async def test_traverse_stays_within_realm(self, pg_client, two_realms):
        realm_a, realm_b = two_realms

        a1 = await pg_client.add_vertex("people", realm_a, payload={"name": "A1"})
        a2 = await pg_client.add_vertex("people", realm_a, payload={"name": "A2"})
        await pg_client.add_edge(
            "knows", realm_a, from_id=a1.id, to_id=a2.id, relation_type="knows"
        )

        b1 = await pg_client.add_vertex("people", realm_b, payload={"name": "B1"})
        b2 = await pg_client.add_vertex("people", realm_b, payload={"name": "B2"})
        await pg_client.add_edge(
            "knows", realm_b, from_id=b1.id, to_id=b2.id, relation_type="knows"
        )

        result_a = await pg_client.traverse(
            realm=realm_a,
            start_table="people",
            start_id=a1.id,
            edge_tables=["knows"],
            max_depth=3,
        )
        found_ids = {step["id"] for step in result_a}
        assert a2.id in found_ids or str(a2.id) in {str(x) for x in found_ids}

        for step in result_a:
            assert step.get("realm", realm_a) == realm_a

    async def test_neighbors_isolated_by_realm(self, pg_client, two_realms):
        realm_a, realm_b = two_realms

        a1 = await pg_client.add_vertex("people", realm_a, payload={"name": "A1"})
        a2 = await pg_client.add_vertex("people", realm_a, payload={"name": "A2"})
        await pg_client.add_edge(
            "knows", realm_a, from_id=a1.id, to_id=a2.id, relation_type="knows"
        )

        b1 = await pg_client.add_vertex("people", realm_b, payload={"name": "B1"})
        b2 = await pg_client.add_vertex("people", realm_b, payload={"name": "B2"})
        await pg_client.add_edge(
            "knows", realm_b, from_id=b1.id, to_id=b2.id, relation_type="knows"
        )

        neighbors_a = await pg_client.get_neighbors(
            realm=realm_a,
            vertex_table="people",
            vertex_id=a1.id,
            edge_tables=["knows"],
        )
        neighbor_realms = {v.realm for v, e in neighbors_a}
        assert neighbor_realms <= {realm_a}

        neighbors_b = await pg_client.get_neighbors(
            realm=realm_b,
            vertex_table="people",
            vertex_id=b1.id,
            edge_tables=["knows"],
        )
        neighbor_realms_b = {v.realm for v, e in neighbors_b}
        assert neighbor_realms_b <= {realm_b}


class TestVectorSearchRealmIsolation:
    async def test_vector_search_isolated(self, pg_client, two_realms_vector):
        realm_a, realm_b, vtable, etable = two_realms_vector

        await pg_client.add_vertex(
            vtable, realm_a, payload={"name": "alpha"}, embedding=[1.0, 0.0, 0.0]
        )
        await pg_client.add_vertex(
            vtable, realm_b, payload={"name": "beta"}, embedding=[1.0, 0.0, 0.0]
        )

        results_a = await pg_client.vector_search(
            vtable, realm_a, query_vector=[1.0, 0.0, 0.0], top_k=10
        )
        results_b = await pg_client.vector_search(
            vtable, realm_b, query_vector=[1.0, 0.0, 0.0], top_k=10
        )

        a_names = {v.payload["name"] for v, _ in results_a}
        b_names = {v.payload["name"] for v, _ in results_b}

        assert "alpha" in a_names
        assert "beta" not in a_names
        assert "beta" in b_names
        assert "alpha" not in b_names

    async def test_edge_vector_search_isolated(self, pg_client, two_realms_vector):
        realm_a, realm_b, vtable, etable = two_realms_vector

        a1 = await pg_client.add_vertex(vtable, realm_a, payload={"name": "a1"})
        a2 = await pg_client.add_vertex(vtable, realm_a, payload={"name": "a2"})
        await pg_client.add_edge(
            etable,
            realm_a,
            from_id=a1.id,
            to_id=a2.id,
            relation_type="similar",
            embedding=[1.0, 0.0, 0.0],
        )

        b1 = await pg_client.add_vertex(vtable, realm_b, payload={"name": "b1"})
        b2 = await pg_client.add_vertex(vtable, realm_b, payload={"name": "b2"})
        await pg_client.add_edge(
            etable,
            realm_b,
            from_id=b1.id,
            to_id=b2.id,
            relation_type="related",
            embedding=[1.0, 0.0, 0.0],
        )

        results_a = await pg_client.vector_search_edges(
            etable, realm_a, query_vector=[1.0, 0.0, 0.0], top_k=10
        )
        results_b = await pg_client.vector_search_edges(
            etable, realm_b, query_vector=[1.0, 0.0, 0.0], top_k=10
        )

        a_types = {e.relation_type for e, _ in results_a}
        b_types = {e.relation_type for e, _ in results_b}

        assert "similar" in a_types
        assert "related" not in a_types
        assert "related" in b_types
        assert "similar" not in b_types


class TestDeleteRealmIsolation:
    async def test_delete_realm_only_affects_target(self, pg_client, two_realms):
        realm_a, realm_b = two_realms

        await pg_client.add_vertex("people", realm_a, payload={"name": "Alice"})
        await pg_client.add_vertex("people", realm_b, payload={"name": "Bob"})

        count = await pg_client.delete_realm(realm_a)
        assert count >= 1

        remaining_a = await pg_client.get_vertices("people", realm_a)
        remaining_b = await pg_client.get_vertices("people", realm_b)

        assert len(remaining_a) == 0
        assert len(remaining_b) >= 1
        assert remaining_b[0].payload["name"] == "Bob"


class TestUpsertRealmIsolation:
    async def test_upsert_does_not_cross_realms(self, pg_client, two_realms):
        realm_a, realm_b = two_realms

        va = await pg_client.add_vertex(
            "people", realm_a, vertex_id=42, payload={"name": "Alice"}
        )

        vb = await pg_client.upsert_vertex(
            "people", realm_b, vertex_id=42, payload={"name": "Bob"}
        )

        fetched_a = await pg_client.get_vertex("people", realm_a, "42")
        fetched_b = await pg_client.get_vertex("people", realm_b, "42")

        assert fetched_a.payload["name"] == "Alice"
        assert fetched_b.payload["name"] == "Bob"


class TestSpaceWithinRealm:
    async def test_space_filters_within_realm(self, pg_client, two_realms):
        realm_a, _ = two_realms

        await pg_client.add_vertex(
            "people", realm_a, payload={"name": "Alice"}, space="hr"
        )
        await pg_client.add_vertex(
            "people", realm_a, payload={"name": "Bob"}, space="eng"
        )

        hr_only = await pg_client.get_vertices("people", realm_a, space="hr")
        eng_only = await pg_client.get_vertices("people", realm_a, space="eng")
        all_spaces = await pg_client.get_vertices("people", realm_a)

        assert len(hr_only) == 1
        assert hr_only[0].payload["name"] == "Alice"
        assert len(eng_only) == 1
        assert eng_only[0].payload["name"] == "Bob"
        assert len(all_spaces) >= 2


class TestDataRecordRealmIsolation:
    async def test_vertex_data_isolated_by_realm(self, pg_client, two_realms):
        realm_a, realm_b = two_realms

        va = await pg_client.add_vertex("people", realm_a, payload={"name": "Alice"})
        vb = await pg_client.add_vertex("people", realm_b, payload={"name": "Bob"})

        await va.add_data(payload={"note": "realm_a data"})
        await vb.add_data(payload={"note": "realm_b data"})

        data_a = await va.get_data()
        data_b = await vb.get_data()

        assert len(data_a) == 1
        assert data_a[0].payload["note"] == "realm_a data"
        assert len(data_b) == 1
        assert data_b[0].payload["note"] == "realm_b data"
