"""Typed filter matching in find_vertices / find_edges.

The previous implementation compared ``str(value)`` against ``payload->>'key'``,
which silently matched nothing for booleans (Python renders ``True``, JSONB
renders ``true``) and for None (``= NULL`` is never true), and interpolated the
filter *key* into the SQL text unvalidated. Matching is now JSONB containment:
type-sensitive, key-safe, GIN-indexable, with explicit sentinels for the two
states None used to conflate.

Every class runs against both clients via the parametrised ``any_client``
fixture, because both carried the same implementation.
"""
import pytest
import pytest_asyncio

from post_graph import JSON_NULL, ABSENT

from conftest import requires_pg

pytestmark = [pytest.mark.asyncio(loop_scope="session"), requires_pg]


@pytest_asyncio.fixture(params=["asyncpg", "sqlalchemy"])
async def any_client(request, pg_client, sa_client):
    """Each test runs once per client implementation."""
    return pg_client if request.param == "asyncpg" else sa_client


@pytest_asyncio.fixture()
async def rich_realm(any_client, realm):
    """A realm holding one vertex per interesting payload shape, plus edges."""
    c = any_client
    await c.create_vertex_table("ff_things", realm=realm)
    await c.create_edge_table("ff_links", from_vertex_table="ff_things",
                              to_vertex_table="ff_things", realm=realm)
    payloads = {
        "flag_on":     {"name": "flag_on", "active": True},
        "flag_off":    {"name": "flag_off", "active": False},
        "flag_null":   {"name": "flag_null", "active": None},
        "flag_absent": {"name": "flag_absent"},
        "num_int":     {"name": "num_int", "count": 42},
        "num_str":     {"name": "num_str", "count": "42"},
        "num_float":   {"name": "num_float", "ratio": 0.5},
        "nested":      {"name": "nested", "tags": ["a", "b"], "meta": {"k": 1}},
    }
    ids = {}
    for key, payload in payloads.items():
        v = await c.add_vertex("ff_things", realm=realm, payload=payload)
        ids[key] = v.id
    e = await c.add_edge("ff_links", realm=realm, from_id=ids["flag_on"],
                         to_id=ids["flag_off"], relation_type="tested",
                         payload={"verified": True, "note": None})
    yield c, realm, ids, e
    try:
        await c.delete_realm(realm)
    except Exception:
        pass


def names(vertices):
    return {v.payload["name"] for v in vertices}


class TestTypedMatching:
    """Booleans and numbers must match by JSON type, not by str() coincidence."""

    async def test_true_matches_only_json_true(self, rich_realm):
        c, realm, _, _ = rich_realm
        got = await c.find_vertices("ff_things", realm=realm, filters={"active": True})
        assert names(got) == {"flag_on"}

    async def test_false_matches_only_json_false(self, rich_realm):
        c, realm, _, _ = rich_realm
        got = await c.find_vertices("ff_things", realm=realm, filters={"active": False})
        assert names(got) == {"flag_off"}

    async def test_int_does_not_match_numeric_string(self, rich_realm):
        c, realm, _, _ = rich_realm
        got = await c.find_vertices("ff_things", realm=realm, filters={"count": 42})
        assert names(got) == {"num_int"}

    async def test_string_does_not_match_number(self, rich_realm):
        c, realm, _, _ = rich_realm
        got = await c.find_vertices("ff_things", realm=realm, filters={"count": "42"})
        assert names(got) == {"num_str"}

    async def test_float(self, rich_realm):
        c, realm, _, _ = rich_realm
        got = await c.find_vertices("ff_things", realm=realm, filters={"ratio": 0.5})
        assert names(got) == {"num_float"}

    async def test_string_filters_still_work(self, rich_realm):
        c, realm, _, _ = rich_realm
        got = await c.find_vertices("ff_things", realm=realm, filters={"name": "nested"})
        assert names(got) == {"nested"}

    async def test_conjunction_of_mixed_types(self, rich_realm):
        c, realm, _, _ = rich_realm
        got = await c.find_vertices(
            "ff_things", realm=realm, filters={"name": "flag_on", "active": True})
        assert names(got) == {"flag_on"}
        got = await c.find_vertices(
            "ff_things", realm=realm, filters={"name": "flag_off", "active": True})
        assert got == []


class TestNullAndAbsent:
    """None is rejected; each state it conflated has its own sentinel."""

    async def test_none_raises(self, rich_realm):
        c, realm, _, _ = rich_realm
        with pytest.raises(ValueError, match="ambiguous"):
            await c.find_vertices("ff_things", realm=realm, filters={"active": None})

    async def test_none_error_names_both_sentinels(self, rich_realm):
        c, realm, _, _ = rich_realm
        with pytest.raises(ValueError, match="JSON_NULL.*ABSENT"):
            await c.find_vertices("ff_things", realm=realm, filters={"active": None})

    async def test_json_null_matches_explicit_null_only(self, rich_realm):
        c, realm, _, _ = rich_realm
        got = await c.find_vertices("ff_things", realm=realm, filters={"active": JSON_NULL})
        # flag_null has active: null. flag_absent has no key at all: under
        # containment {"active": null} those differ, and only the former hits.
        assert names(got) == {"flag_null"}

    async def test_absent_matches_missing_key_only(self, rich_realm):
        c, realm, _, _ = rich_realm
        got = await c.find_vertices("ff_things", realm=realm, filters={"active": ABSENT})
        # Every vertex without an 'active' key qualifies, including flag_absent
        # and the numeric/nested ones. flag_null does NOT: its key is present.
        got_names = names(got)
        assert "flag_absent" in got_names
        assert "flag_null" not in got_names
        assert "flag_on" not in got_names and "flag_off" not in got_names

    async def test_absent_composes_with_containment(self, rich_realm):
        c, realm, _, _ = rich_realm
        got = await c.find_vertices(
            "ff_things", realm=realm,
            filters={"name": "flag_absent", "active": ABSENT})
        assert names(got) == {"flag_absent"}
        got = await c.find_vertices(
            "ff_things", realm=realm,
            filters={"name": "flag_null", "active": ABSENT})
        assert got == []


class TestContainmentSemantics:
    """Nested values match by containment, documented rather than discovered."""

    async def test_array_subset_matches(self, rich_realm):
        c, realm, _, _ = rich_realm
        got = await c.find_vertices("ff_things", realm=realm, filters={"tags": ["a"]})
        assert names(got) == {"nested"}

    async def test_array_superset_does_not(self, rich_realm):
        c, realm, _, _ = rich_realm
        got = await c.find_vertices(
            "ff_things", realm=realm, filters={"tags": ["a", "b", "c"]})
        assert got == []

    async def test_nested_object(self, rich_realm):
        c, realm, _, _ = rich_realm
        got = await c.find_vertices("ff_things", realm=realm, filters={"meta": {"k": 1}})
        assert names(got) == {"nested"}
        got = await c.find_vertices("ff_things", realm=realm, filters={"meta": {"k": 2}})
        assert got == []


class TestKeySafety:
    """Filter keys never enter the SQL text, so no key can break the query."""

    INJECTION_KEYS = [
        "x' OR t.realm=t.realm OR '",
        "x'; DROP TABLE ff_things; --",
        'x" OR 1=1 --',
    ]

    async def test_hostile_keys_match_nothing_and_do_not_error(self, rich_realm):
        c, realm, _, _ = rich_realm
        for key in self.INJECTION_KEYS:
            got = await c.find_vertices("ff_things", realm=realm, filters={key: "x"})
            assert got == [], f"hostile key matched rows: {key!r}"
        # The table must have survived every attempt.
        got = await c.find_vertices("ff_things", realm=realm, filters={"name": "flag_on"})
        assert len(got) == 1

    async def test_hostile_absent_keys_are_safe_too(self, rich_realm):
        c, realm, _, _ = rich_realm
        for key in self.INJECTION_KEYS:
            got = await c.find_vertices("ff_things", realm=realm, filters={key: ABSENT})
            # The key is absent from every payload, so ABSENT matches all rows
            # -- the point is that it parameterises rather than injects.
            assert len(got) == 8

    async def test_unusual_but_legal_json_keys(self, rich_realm):
        c, realm, ids, _ = rich_realm
        await c.add_vertex("ff_things", realm=realm,
                           payload={"name": "oddkeys", "key with space": 1,
                                    "key-with-dash": True})
        got = await c.find_vertices("ff_things", realm=realm,
                                    filters={"key with space": 1})
        assert names(got) == {"oddkeys"}
        got = await c.find_vertices("ff_things", realm=realm,
                                    filters={"key-with-dash": True})
        assert names(got) == {"oddkeys"}


class TestFindEdges:
    """Edges share the implementation and the guarantees."""

    async def test_bool_filter(self, rich_realm):
        c, realm, _, e = rich_realm
        got = await c.find_edges("ff_links", realm=realm, filters={"verified": True})
        assert len(got) == 1 and got[0].id == e.id
        got = await c.find_edges("ff_links", realm=realm, filters={"verified": False})
        assert got == []

    async def test_json_null_on_edge(self, rich_realm):
        c, realm, _, e = rich_realm
        got = await c.find_edges("ff_links", realm=realm, filters={"note": JSON_NULL})
        assert len(got) == 1 and got[0].id == e.id

    async def test_none_raises_on_edge(self, rich_realm):
        c, realm, _, _ = rich_realm
        with pytest.raises(ValueError, match="ambiguous"):
            await c.find_edges("ff_links", realm=realm, filters={"note": None})

    async def test_relation_type_composes(self, rich_realm):
        c, realm, _, _ = rich_realm
        got = await c.find_edges("ff_links", realm=realm,
                                 filters={"verified": True}, relation_type="tested")
        assert len(got) == 1
        got = await c.find_edges("ff_links", realm=realm,
                                 filters={"verified": True}, relation_type="other")
        assert got == []


class TestUnchangedBehaviour:
    """The shapes that worked before must keep working identically."""

    async def test_empty_filters_matches_all(self, rich_realm):
        c, realm, _, _ = rich_realm
        got = await c.find_vertices("ff_things", realm=realm, filters={})
        assert len(got) == 8

    async def test_limit(self, rich_realm):
        c, realm, _, _ = rich_realm
        got = await c.find_vertices("ff_things", realm=realm, filters={}, limit=3)
        assert len(got) == 3

    async def test_no_match(self, rich_realm):
        c, realm, _, _ = rich_realm
        got = await c.find_vertices("ff_things", realm=realm, filters={"name": "nobody"})
        assert got == []


class TestInterpolationSitesValidate:
    """Config-supplied keys that do enter SQL text are identifier-checked."""

    async def test_lexical_fields_reject_hostile_names(self, any_client, clean_realm):
        c, realm = any_client, clean_realm
        await c.create_vertex_table("ff_docs", realm=realm)
        with pytest.raises(ValueError, match="[Ii]nvalid identifier"):
            await c.fulltext_search_vertices(
                "ff_docs", realm=realm, query="x",
                fields=["title", "x', ''); DROP TABLE ff_docs; --"])

    async def test_weight_field_rejects_hostile_names(self, any_client, clean_realm):
        c, realm = any_client, clean_realm
        await c.create_vertex_table("ff_nodes", realm=realm)
        await c.create_edge_table("ff_ties", from_vertex_table="ff_nodes",
                                  to_vertex_table="ff_nodes", realm=realm)
        v1 = await c.add_vertex("ff_nodes", realm=realm, payload={"name": "a"})
        v2 = await c.add_vertex("ff_nodes", realm=realm, payload={"name": "b"})
        with pytest.raises(ValueError, match="[Ii]nvalid identifier"):
            await c.weighted_shortest_path(
                realm, start_table="ff_nodes", start_id=v1.id,
                target_table="ff_nodes", target_id=v2.id,
                edge_tables=["ff_ties"],
                weight_field="w')::float, 1.0)); DROP TABLE ff_nodes; --")
