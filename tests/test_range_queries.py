"""Server-side range filtering, ordering, counting and bulk deletion.

The consumer this serves polls monotonically growing vertex tables: an event
scheduler asking for the few rows whose due_at has passed, a work queue asking
for undone rows. Before these methods, every poll fetched the whole table and
filtered client-side; each test here is shaped like one of those queries.

Every test runs against both clients, and the schema_per_realm variants run
the same assertions through per-schema realms.
"""
import pytest
import pytest_asyncio
from conftest import requires_pg

pytestmark = [pytest.mark.asyncio(loop_scope="session"), requires_pg]


@pytest_asyncio.fixture(params=["asyncpg", "sqlalchemy"])
async def any_client(request, pg_client, sa_client):
    return pg_client if request.param == "asyncpg" else sa_client


@pytest_asyncio.fixture(params=["asyncpg_spr", "sqlalchemy_spr"])
async def any_client_spr(request, pg_client_spr, sa_client_spr):
    return pg_client_spr if request.param == "asyncpg_spr" else sa_client_spr


ROWS = [
    # name      due_at (fixed-width, text-sortable)  priority  done_at
    ("a", "2026-01-01T00:00:00", 1,  None),      # explicit JSON null
    ("b", "2026-01-02T00:00:00", 5,  "ABSENT"),  # key absent entirely
    ("c", "2026-01-03T00:00:00", 10, "2026-01-04T00:00:00"),
    ("d", "2026-02-01T00:00:00", 2,  "ABSENT"),
    ("e", "2026-03-01T00:00:00", 7,  "2026-03-02T00:00:00"),
]


async def seed(c, realm, table="rq_events"):
    await c.create_vertex_table(table, realm=realm)
    for name, due, prio, done in ROWS:
        payload = {"name": name, "due_at": due, "priority": prio}
        if done != "ABSENT":
            payload["done_at"] = done
        await c.add_vertex(table, realm=realm, payload=payload)
    return table


def names(vs):
    return [v.payload["name"] for v in vs]


class TestOperators:
    async def test_eq_and_ne(self, any_client, clean_realm):
        c, realm = any_client, clean_realm
        t = await seed(c, realm)
        got = await c.find_vertices(t, realm=realm, where=[("name", "=", "c")])
        assert names(got) == ["c"]
        got = await c.find_vertices(t, realm=realm, where=[("name", "!=", "c")])
        assert sorted(names(got)) == ["a", "b", "d", "e"]

    async def test_text_range_on_sortable_timestamps(self, any_client, clean_realm):
        c, realm = any_client, clean_realm
        t = await seed(c, realm)
        got = await c.find_vertices(
            t, realm=realm, where=[("due_at", "<=", "2026-01-31T23:59:59")])
        assert sorted(names(got)) == ["a", "b", "c"]
        got = await c.find_vertices(
            t, realm=realm, where=[("due_at", ">", "2026-01-31T23:59:59")])
        assert sorted(names(got)) == ["d", "e"]

    async def test_numeric_range(self, any_client, clean_realm):
        c, realm = any_client, clean_realm
        t = await seed(c, realm)
        got = await c.find_vertices(t, realm=realm, where=[("priority", ">=", 5)])
        assert sorted(names(got)) == ["b", "c", "e"]
        got = await c.find_vertices(t, realm=realm, where=[("priority", "<", 5)])
        assert sorted(names(got)) == ["a", "d"]

    async def test_numeric_vs_text_cast_rule(self, any_client, clean_realm):
        """int 10 compares numerically (9 < 10); the string '10' compares as
        text, where '10' < '9' -- both behaviours are deliberate."""
        c, realm = any_client, clean_realm
        t = await c.create_vertex_table("rq_cast", realm=realm) or "rq_cast"
        for v in ("9", "10", "11"):
            await c.add_vertex("rq_cast", realm=realm, payload={"v": v, "n": int(v)})
        num = await c.find_vertices("rq_cast", realm=realm, where=[("n", "<", 10)])
        assert [x.payload["n"] for x in num] == [9]
        txt = await c.find_vertices("rq_cast", realm=realm, where=[("v", "<", "9")])
        # text ordering: '10' and '11' sort below '9'
        assert sorted(x.payload["v"] for x in txt) == ["10", "11"]

    async def test_is_null_covers_null_and_absent(self, any_client, clean_realm):
        c, realm = any_client, clean_realm
        t = await seed(c, realm)
        got = await c.find_vertices(t, realm=realm, where=[("done_at", "is_null", None)])
        # a: explicit null; b, d: key absent
        assert sorted(names(got)) == ["a", "b", "d"]
        got = await c.find_vertices(t, realm=realm, where=[("done_at", "not_null", None)])
        assert sorted(names(got)) == ["c", "e"]

    async def test_in_text_and_numeric(self, any_client, clean_realm):
        c, realm = any_client, clean_realm
        t = await seed(c, realm)
        got = await c.find_vertices(t, realm=realm, where=[("name", "in", ["a", "e"])])
        assert sorted(names(got)) == ["a", "e"]
        got = await c.find_vertices(t, realm=realm, where=[("priority", "in", [1, 2, 3])])
        assert sorted(names(got)) == ["a", "d"]

    async def test_conjunction_with_filters(self, any_client, clean_realm):
        c, realm = any_client, clean_realm
        t = await seed(c, realm)
        got = await c.find_vertices(
            t, realm=realm, filters={"name": "d"},
            where=[("done_at", "is_null", None)])
        assert names(got) == ["d"]
        got = await c.find_vertices(
            t, realm=realm, filters={"name": "c"},
            where=[("done_at", "is_null", None)])
        assert got == []


class TestOrderingAndLimit:
    async def test_order_by_text_with_limit(self, any_client, clean_realm):
        """The scheduler's exact shape: undone, due, oldest first, capped."""
        c, realm = any_client, clean_realm
        t = await seed(c, realm)
        got = await c.find_vertices(
            t, realm=realm,
            where=[("done_at", "is_null", None),
                   ("due_at", "<=", "2026-12-31T00:00:00")],
            order_by="due_at", limit=2)
        assert names(got) == ["a", "b"]

    async def test_order_descending(self, any_client, clean_realm):
        c, realm = any_client, clean_realm
        t = await seed(c, realm)
        got = await c.find_vertices(t, realm=realm, where=[],
                                    order_by="due_at", descending=True, limit=2)
        assert names(got) == ["e", "d"]

    async def test_order_numeric_follows_where_cast(self, any_client, clean_realm):
        """With a numeric predicate on the key, ordering is numeric too:
        priorities 1,2,5,7,10 must not sort as text (1,10,2,5,7)."""
        c, realm = any_client, clean_realm
        t = await seed(c, realm)
        got = await c.find_vertices(t, realm=realm,
                                    where=[("priority", ">", 0)],
                                    order_by="priority")
        assert [v.payload["priority"] for v in got] == [1, 2, 5, 7, 10]


class TestCount:
    async def test_count_matches_find(self, any_client, clean_realm):
        c, realm = any_client, clean_realm
        t = await seed(c, realm)
        n = await c.count_vertices(t, realm=realm,
                                   where=[("done_at", "is_null", None)])
        assert n == 3
        assert await c.count_vertices(t, realm=realm) == 5
        assert await c.count_vertices(t, realm=realm, filters={"name": "a"}) == 1


class TestDelete:
    async def test_bulk_delete_returns_count(self, any_client, clean_realm):
        c, realm = any_client, clean_realm
        t = await seed(c, realm)
        purged = await c.delete_vertices(
            t, realm=realm,
            where=[("done_at", "not_null", None),
                   ("done_at", "<", "2026-02-01T00:00:00")])
        assert purged == 1                                    # only c
        assert await c.count_vertices(t, realm=realm) == 4

    async def test_delete_refuses_empty_where(self, any_client, clean_realm):
        c, realm = any_client, clean_realm
        t = await seed(c, realm)
        with pytest.raises(ValueError, match="delete_realm"):
            await c.delete_vertices(t, realm=realm, where=[])
        with pytest.raises((ValueError, TypeError)):
            await c.delete_vertices(t, realm=realm, where=None)
        assert await c.count_vertices(t, realm=realm) == 5


class TestPayloadIndex:
    async def test_create_and_idempotent(self, any_client, clean_realm):
        c, realm = any_client, clean_realm
        t = await seed(c, realm)
        name1 = await c.create_payload_index(t, realm=realm, key="due_at")
        name2 = await c.create_payload_index(t, realm=realm, key="due_at")
        assert name1 == name2 == f"idx_{t}_payload_due_at"
        num = await c.create_payload_index(t, realm=realm, key="priority", numeric=True)
        assert num.endswith("_num")

    async def test_hostile_key_rejected(self, any_client, clean_realm):
        c, realm = any_client, clean_realm
        t = await seed(c, realm)
        with pytest.raises(ValueError, match="[Ii]nvalid payload key"):
            await c.create_payload_index(t, realm=realm, key="x'); DROP TABLE x; --")
        with pytest.raises(ValueError, match="[Ii]nvalid payload key"):
            await c.find_vertices(t, realm=realm, where=[("x' OR '1'='1", "=", "x")])
        with pytest.raises(ValueError, match="[Ii]nvalid payload key"):
            await c.find_vertices(t, realm=realm, where=[], order_by="x; --")

    async def test_bad_op_and_bad_triple(self, any_client, clean_realm):
        c, realm = any_client, clean_realm
        t = await seed(c, realm)
        with pytest.raises(ValueError, match="Unknown where op"):
            await c.find_vertices(t, realm=realm, where=[("name", "like", "a%")])
        with pytest.raises(ValueError, match="triples"):
            await c.find_vertices(t, realm=realm, where=[("name", "=")])
        with pytest.raises(ValueError, match="is_null"):
            await c.find_vertices(t, realm=realm, where=[("name", "=", None)])


class TestSchemaPerRealm:
    """The same shapes through per-schema realms, for every new method."""

    async def test_all_methods_spr(self, any_client_spr, realm):
        c = any_client_spr
        try:
            t = await seed(c, realm)
            due = await c.find_vertices(
                t, realm=realm,
                where=[("done_at", "is_null", None),
                       ("due_at", "<=", "2026-12-31T00:00:00")],
                order_by="due_at", limit=200)
            assert names(due) == ["a", "b", "d"]
            assert await c.count_vertices(
                t, realm=realm, where=[("done_at", "is_null", None)]) == 3
            idx = await c.create_payload_index(t, realm=realm, key="due_at")
            assert idx == f"idx_{t}_payload_due_at"
            purged = await c.delete_vertices(
                t, realm=realm, where=[("done_at", "not_null", None)])
            assert purged == 2
            assert await c.count_vertices(t, realm=realm) == 3
        finally:
            try:
                await c.delete_realm(realm)
            except Exception:
                pass


class TestEdgeRangeQueries:
    """The same where/order_by semantics on edges, which is what a bi-temporal
    relation store needs: t_created/t_expired live on edge payloads."""

    async def _seed_edges(self, c, realm):
        await c.create_vertex_table("rq_nodes2", realm=realm)
        await c.create_edge_table("rq_rel", from_vertex_table="rq_nodes2",
                                  to_vertex_table="rq_nodes2", realm=realm)
        a = await c.add_vertex("rq_nodes2", realm=realm, payload={"name": "a"})
        b = await c.add_vertex("rq_nodes2", realm=realm, payload={"name": "b"})
        stamps = ["2026-01-01T00:00:00", "2026-02-01T00:00:00", "2026-03-01T00:00:00"]
        for i, ts in enumerate(stamps):
            payload = {"t_created": ts, "k": i}
            if i == 0:
                payload["t_expired"] = "2026-02-15T00:00:00"
            await c.add_edge("rq_rel", realm=realm, from_id=a.id, to_id=b.id,
                             relation_type=f"r{i}", payload=payload)
        return "rq_rel"

    async def test_edge_where_text_range_and_order(self, any_client, clean_realm):
        c, realm = any_client, clean_realm
        t = await self._seed_edges(c, realm)
        got = await c.find_edges(t, realm=realm,
                                 where=[("t_created", ">", "2026-01-15T00:00:00")],
                                 order_by="t_created", descending=True)
        assert [e.payload["k"] for e in got] == [2, 1]

    async def test_edge_is_null_and_not_null(self, any_client, clean_realm):
        c, realm = any_client, clean_realm
        t = await self._seed_edges(c, realm)
        expired = await c.find_edges(t, realm=realm,
                                     where=[("t_expired", "not_null", None)])
        assert [e.payload["k"] for e in expired] == [0]
        live = await c.find_edges(t, realm=realm,
                                  where=[("t_expired", "is_null", None)])
        assert sorted(e.payload["k"] for e in live) == [1, 2]

    async def test_count_edges(self, any_client, clean_realm):
        c, realm = any_client, clean_realm
        t = await self._seed_edges(c, realm)
        assert await c.count_edges(t, realm=realm) == 3
        assert await c.count_edges(
            t, realm=realm,
            where=[("t_created", ">=", "2026-02-01T00:00:00")]) == 2
        assert await c.count_edges(t, realm=realm, relation_type="r0") == 1

    async def test_edge_numeric_where(self, any_client, clean_realm):
        c, realm = any_client, clean_realm
        t = await self._seed_edges(c, realm)
        got = await c.find_edges(t, realm=realm, where=[("k", ">=", 1)],
                                 order_by="k")
        assert [e.payload["k"] for e in got] == [1, 2]

    async def test_edge_payload_index(self, any_client, clean_realm):
        c, realm = any_client, clean_realm
        t = await self._seed_edges(c, realm)
        n1 = await c.create_payload_index(t, realm=realm, key="t_created")
        n2 = await c.create_payload_index(t, realm=realm, key="t_created")
        assert n1 == n2 == f"idx_{t}_payload_t_created"


class TestFilteredVectorSearch:
    """where= inside vector_search: a filtered top-k must be a genuine top-k.

    The scenario is the one post-filtering fails: many near neighbours of the
    query are excluded by the predicate, and the wanted rows sit further out.
    """

    async def _seed(self, c, realm):
        await c.create_vertex_table("rq_vec", realm=realm, vector_dim=3)
        # Ten level-0 rows hugging the query vector, two level-1 rows far away.
        for i in range(10):
            await c.add_vertex("rq_vec", realm=realm,
                               payload={"name": f"n{i}", "level": 0},
                               embedding=[1.0, 0.0, 0.001 * i])
        await c.add_vertex("rq_vec", realm=realm,
                           payload={"name": "far1", "level": 1},
                           embedding=[0.0, 1.0, 0.0])
        await c.add_vertex("rq_vec", realm=realm,
                           payload={"name": "far2", "level": 1},
                           embedding=[0.0, 0.9, 0.1])

    async def test_filtered_topk_returns_k_not_remainder(self, any_client, clean_realm):
        c, realm = any_client, clean_realm
        await self._seed(c, realm)
        hits = await c.vector_search("rq_vec", realm=realm,
                                     query_vector=[1.0, 0.0, 0.0], top_k=2,
                                     where=[("level", "=", 1)])
        # Post-filtering top-2 would return zero rows here; in-search
        # filtering returns exactly the two level-1 rows.
        assert len(hits) == 2
        assert {v.payload["name"] for v, _d in hits} == {"far1", "far2"}

    async def test_unfiltered_unchanged(self, any_client, clean_realm):
        c, realm = any_client, clean_realm
        await self._seed(c, realm)
        hits = await c.vector_search("rq_vec", realm=realm,
                                     query_vector=[1.0, 0.0, 0.0], top_k=3)
        assert all(v.payload["level"] == 0 for v, _d in hits)

    async def test_where_rejected_on_other_scopes(self, any_client, clean_realm):
        c, realm = any_client, clean_realm
        await self._seed(c, realm)
        with pytest.raises(ValueError, match="main"):
            await c.vector_search("rq_vec", realm=realm,
                                  query_vector=[1.0, 0.0, 0.0],
                                  search_scope="both",
                                  where=[("level", "=", 1)])
