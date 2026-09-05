"""openCypher subset: lexing, parsing, translation and execution.

The translator's job is to answer the question that was asked or refuse. So
alongside the happy paths these tests pin down two things that matter more:
every literal reaches SQL as a bind parameter rather than as text, and every
construct outside the subset raises rather than being approximated.
"""
import pytest
from conftest import requires_pg

from post_graph.cypher import ast as A
from post_graph.cypher import parse, tokenize
from post_graph.cypher.lexer import CypherSyntaxError
from post_graph.cypher.translator import CypherTranslationError, Translator


def _ref(table, realm):
    return f'"{realm}".{table}'


def _translator(**kw):
    kw.setdefault('vertex_tables', ['person', 'company'])
    kw.setdefault('edge_tables', ['knows', 'works_at'])
    kw.setdefault('edge_schemas', {'knows': ('person', 'person'),
                                   'works_at': ('person', 'company')})
    return Translator(realm='R', table_ref=_ref, **kw)


def _sql(query, **kw):
    tr = _translator(**kw)
    sql, params, cols = tr.build_select(parse(query))
    return sql, params, cols


# ------------------------------------------------------------------- lexing

class TestLexer:
    def test_keywords_are_case_insensitive(self):
        assert [t.value for t in tokenize("match Match MATCH")[:3]] == ['MATCH'] * 3

    def test_identifiers_keep_their_case(self):
        assert tokenize("MATCH (myNode)")[2].value == 'myNode'

    def test_string_escapes(self):
        assert tokenize(r"'it\'s'")[0].value == "it's"

    def test_backticks_allow_reserved_words_as_names(self):
        t = tokenize("`match`")[0]
        assert t.kind == 'IDENT' and t.value == 'match'

    def test_comments_are_skipped(self):
        assert [t.kind for t in tokenize("MATCH // trailing\n(n)")][:2] == ['KEYWORD', 'PUNCT']

    def test_numbers(self):
        vals = [t.value for t in tokenize("1 2.5")[:2]]
        assert vals == [1, 2.5]

    def test_unexpected_character_reports_position(self):
        with pytest.raises(CypherSyntaxError) as e:
            tokenize("MATCH (n) RETURN n # bad")
        assert '^' in str(e.value)


# ------------------------------------------------------------------ parsing

class TestParser:
    def test_node_labels_and_properties(self):
        q = parse("MATCH (n:Person {name:'x'}) RETURN n.name")
        node = q.clauses[0].patterns[0].nodes[0]
        assert node.variable == 'n' and node.labels == ['Person']
        assert isinstance(node.properties['name'], A.Literal)

    @pytest.mark.parametrize("pattern,direction", [
        ("(a)-[r]->(b)", 'out'),
        ("(a)<-[r]-(b)", 'in'),
        ("(a)-[r]-(b)", 'both'),
    ])
    def test_directions(self, pattern, direction):
        q = parse(f"MATCH {pattern} RETURN a")
        assert q.clauses[0].patterns[0].rels[0].direction == direction

    @pytest.mark.parametrize("text,lo,hi", [
        ("*", 1, None), ("*2", 2, 2), ("*1..3", 1, 3), ("*..5", 1, 5), ("*2..", 2, None),
    ])
    def test_variable_length_bounds(self, text, lo, hi):
        rel = parse(f"MATCH (a)-[r{text}]->(b) RETURN a").clauses[0].patterns[0].rels[0]
        assert (rel.min_hops, rel.max_hops) == (lo, hi)

    def test_relationship_type_alternatives(self):
        rel = parse("MATCH (a)-[r:A|B|C]->(b) RETURN a").clauses[0].patterns[0].rels[0]
        assert rel.types == ['A', 'B', 'C']

    def test_boolean_precedence_binds_and_tighter_than_or(self):
        w = parse("MATCH (n) WHERE n.a = 1 OR n.b = 2 AND n.c = 3 RETURN n").clauses[0].where
        assert w.op == 'OR' and w.right.op == 'AND'

    @pytest.mark.parametrize("predicate", [
        "n.x IS NULL", "n.x IS NOT NULL", "n.x STARTS WITH 'a'", "n.x ENDS WITH 'a'",
        "n.x CONTAINS 'a'", "n.x IN ['a','b']", "NOT n.x = 1", "n.x =~ '^a'",
    ])
    def test_predicates_parse(self, predicate):
        assert parse(f"MATCH (n) WHERE {predicate} RETURN n") is not None

    def test_return_modifiers(self):
        r = parse("MATCH (n) RETURN DISTINCT n.a AS a ORDER BY a DESC SKIP 5 LIMIT 10").clauses[1]
        assert r.distinct and r.items[0].alias == 'a'
        assert r.order_by[0].descending
        assert r.skip.value == 5 and r.limit.value == 10

    def test_merge_with_on_create_and_on_match(self):
        m = parse("MERGE (n:P {k:'v'}) ON CREATE SET n.a = 1 ON MATCH SET n.b = 2").clauses[0]
        assert len(m.on_create) == 1 and len(m.on_match) == 1

    def test_detach_delete(self):
        d = parse("MATCH (n) DETACH DELETE n").clauses[1]
        assert d.detach and d.variables == ['n']

    def test_unknown_function_is_rejected_with_suggestions(self):
        with pytest.raises(CypherSyntaxError) as e:
            parse("MATCH (n) RETURN frobnicate(n)")
        assert 'Supported' in str(e.value)

    def test_union_is_refused_rather_than_ignored(self):
        with pytest.raises(CypherSyntaxError):
            parse("MATCH (n) RETURN n UNION MATCH (m) RETURN m")

    def test_empty_query_rejected(self):
        with pytest.raises(CypherSyntaxError):
            parse("   ")

    def test_error_carries_position_marker(self):
        with pytest.raises(CypherSyntaxError) as e:
            parse("MATCH (n RETURN n")
        assert '^' in str(e.value)


# -------------------------------------------------------------- translation

class TestTranslation:
    def test_literals_become_bind_parameters(self):
        sql, params, _ = _sql("MATCH (n:Person) WHERE n.name = 'Bob' RETURN n.name")
        assert 'Bob' not in sql and 'Bob' in params

    def test_injection_attempt_is_a_parameter_not_sql(self):
        nasty = "'; DROP TABLE person; --"
        sql, params, _ = _sql(f"MATCH (n:Person) WHERE n.name = \"{nasty}\" RETURN n.name")
        assert 'DROP TABLE' not in sql
        assert nasty in params

    def test_property_key_must_be_an_identifier(self):
        # Keys are interpolated into payload->>'...', so they are validated.
        with pytest.raises(ValueError):
            _sql("MATCH (n:Person) WHERE n.`bad key` = 'x' RETURN n.name")

    def test_numeric_comparison_is_numeric_not_lexical(self):
        sql, _, _ = _sql("MATCH (n:Person) WHERE n.age > 9 RETURN n.name")
        assert '::numeric' in sql

    def test_unknown_label_is_rejected(self):
        with pytest.raises(CypherTranslationError) as e:
            _sql("MATCH (n:Unicorn) RETURN n.name")
        assert 'Unknown label' in str(e.value)

    def test_missing_label_is_rejected(self):
        with pytest.raises(CypherTranslationError):
            _sql("MATCH (n) RETURN n.name")

    def test_no_joining_edge_table_is_rejected(self):
        with pytest.raises(CypherTranslationError):
            _sql("MATCH (c:Company)-[:KNOWS]->(p:Person) RETURN p.name")

    def test_aggregate_adds_group_by(self):
        sql, _, _ = _sql("MATCH (n:Person) RETURN n.name AS nm, count(*) AS c")
        assert 'GROUP BY' in sql

    def test_no_aggregate_means_no_group_by(self):
        sql, _, _ = _sql("MATCH (n:Person) RETURN n.name AS nm")
        assert 'GROUP BY' not in sql

    def test_missing_parameter_is_reported(self):
        with pytest.raises(CypherTranslationError) as e:
            _sql("MATCH (n:Person) WHERE n.name = $who RETURN n.name")
        assert '$who' in str(e.value)

    def test_promoted_column_is_preferred_over_payload(self):
        sql, _, _ = _sql("MATCH (n:Person) WHERE n.valid_from = '2020' RETURN n.name",
                         promoted_columns={'person': {'pt_valid_from'}})
        assert 'pt_valid_from' in sql and "payload->>'valid_from'" not in sql

    def test_real_columns_are_read_directly(self):
        sql, _, _ = _sql("MATCH (n:Person) WHERE n.uuid = 'x' RETURN n.fqid")
        assert '"uuid"' in sql and "payload->>'uuid'" not in sql

    def test_variable_length_builds_recursive_cte(self):
        sql, _, _ = _sql("MATCH (a:Person)-[:KNOWS*1..3]->(b:Person) RETURN b.name")
        assert 'WITH RECURSIVE' in sql

    def test_variable_length_across_labels_is_refused(self):
        with pytest.raises(CypherTranslationError):
            _sql("MATCH (a:Person)-[:WORKS_AT*1..3]->(b:Company) RETURN b.name")

    def test_return_star_is_refused(self):
        with pytest.raises(CypherTranslationError):
            _sql("MATCH (n:Person) RETURN *")

    def test_optional_match_is_refused_not_silently_inner(self):
        # Answering an OPTIONAL MATCH as an inner join would drop rows the
        # caller explicitly asked to keep.
        with pytest.raises(CypherTranslationError):
            _sql("OPTIONAL MATCH (n:Person) RETURN n.name")

    def test_with_is_refused(self):
        with pytest.raises(CypherTranslationError):
            _sql("MATCH (n:Person) WITH n RETURN n.name")


# -------------------------------------------------------------- integration

@requires_pg
class TestCypherExecution:
    """End to end against a live graph. Reads translate to SQL; writes go
    through the client, so these also confirm the two halves agree."""

    @pytest.fixture()
    async def graph(self, pg_client_spr, clean_realm_spr):
        from post_graph import CypherSession
        pg_client = pg_client_spr
        realm = clean_realm_spr
        await pg_client.create_vertex_table("person", realm=realm)
        await pg_client.create_vertex_table("company", realm=realm)
        await pg_client.create_edge_table("knows", from_vertex_table="person",
                                          to_vertex_table="person", realm=realm)
        await pg_client.create_edge_table("works_at", from_vertex_table="person",
                                          to_vertex_table="company", realm=realm)
        people = {}
        for name, age, city in [("Alice", 34, "London"), ("Bob", 28, "Leeds"),
                                ("Carol", 41, "London"), ("Dan", 23, "Lisbon")]:
            people[name] = await pg_client.add_vertex(
                "person", realm, payload={"name": name, "age": age, "city": city})
        acme = await pg_client.add_vertex("company", realm, payload={"name": "Acme"})
        for a, b in [("Alice", "Bob"), ("Bob", "Carol"), ("Carol", "Dan"), ("Alice", "Carol")]:
            await pg_client.add_edge("knows", realm, people[a].id, people[b].id,
                                     "KNOWS", payload={"since": "2020"})
        await pg_client.add_edge("works_at", realm, people["Alice"].id, acme.id,
                                 "WORKS_AT", payload={})
        return CypherSession(pg_client, realm), people, realm

    @pytest.mark.asyncio(loop_scope="session")
    async def test_filter_and_order(self, graph):
        s, _, _ = graph
        rows = await s.run("MATCH (p:Person) WHERE p.age > 30 RETURN p.name AS name ORDER BY name")
        assert [r['name'] for r in rows] == ['Alice', 'Carol']

    @pytest.mark.asyncio(loop_scope="session")
    async def test_numeric_comparison_is_not_lexical(self, graph):
        s, _, _ = graph
        # Lexically '9' > '28'; numerically it is not. Dan is 23, Bob 28.
        rows = await s.run("MATCH (p:Person) WHERE p.age > 9 RETURN p.name AS name")
        assert len(rows) == 4

    @pytest.mark.asyncio(loop_scope="session")
    async def test_parameters(self, graph):
        s, _, _ = graph
        rows = await s.run("MATCH (p:Person) WHERE p.city = $city RETURN p.name AS name ORDER BY name",
                           {"city": "London"})
        assert [r['name'] for r in rows] == ['Alice', 'Carol']

    @pytest.mark.asyncio(loop_scope="session")
    async def test_relationship_traversal(self, graph):
        s, _, _ = graph
        rows = await s.run("MATCH (a:Person)-[:KNOWS]->(b:Person) WHERE a.name = 'Alice' "
                           "RETURN b.name AS b ORDER BY b")
        assert [r['b'] for r in rows] == ['Bob', 'Carol']

    @pytest.mark.asyncio(loop_scope="session")
    async def test_incoming_direction(self, graph):
        s, _, _ = graph
        rows = await s.run("MATCH (a:Person)<-[:KNOWS]-(b:Person) WHERE a.name = 'Carol' "
                           "RETURN b.name AS b ORDER BY b")
        assert [r['b'] for r in rows] == ['Alice', 'Bob']

    @pytest.mark.asyncio(loop_scope="session")
    async def test_variable_length_reaches_transitively(self, graph):
        s, _, _ = graph
        rows = await s.run("MATCH (a:Person)-[:KNOWS*1..3]->(b:Person) WHERE a.name = 'Alice' "
                           "RETURN DISTINCT b.name AS r ORDER BY r")
        assert [r['r'] for r in rows] == ['Bob', 'Carol', 'Dan']

    @pytest.mark.asyncio(loop_scope="session")
    async def test_variable_length_respects_upper_bound(self, graph):
        s, _, _ = graph
        rows = await s.run("MATCH (a:Person)-[:KNOWS*1..1]->(b:Person) WHERE a.name = 'Alice' "
                           "RETURN DISTINCT b.name AS r ORDER BY r")
        assert [r['r'] for r in rows] == ['Bob', 'Carol']

    @pytest.mark.asyncio(loop_scope="session")
    async def test_aggregate_with_grouping(self, graph):
        s, _, _ = graph
        rows = await s.run("MATCH (a:Person)-[:KNOWS]->(b:Person) "
                           "RETURN a.name AS a, count(*) AS n ORDER BY n DESC, a")
        assert rows[0] == {'a': 'Alice', 'n': 2}

    @pytest.mark.asyncio(loop_scope="session")
    async def test_cross_label_relationship(self, graph):
        s, _, _ = graph
        rows = await s.run("MATCH (p:Person)-[:WORKS_AT]->(c:Company) "
                           "RETURN p.name AS who, c.name AS employer")
        assert rows == [{'who': 'Alice', 'employer': 'Acme'}]

    @pytest.mark.asyncio(loop_scope="session")
    async def test_node_value_is_a_dict(self, graph):
        s, _, _ = graph
        rows = await s.run("MATCH (p:Person) WHERE p.name = 'Bob' RETURN p")
        assert rows[0]['p']['properties']['name'] == 'Bob'
        assert rows[0]['p']['label'] == 'person'

    @pytest.mark.asyncio(loop_scope="session")
    async def test_relationship_properties_and_type(self, graph):
        s, _, _ = graph
        rows = await s.run("MATCH (a:Person)-[r:KNOWS]->(b:Person) WHERE a.name = 'Bob' "
                           "RETURN type(r) AS t, r.since AS since")
        assert rows == [{'t': 'KNOWS', 'since': '2020'}]

    @pytest.mark.asyncio(loop_scope="session")
    async def test_string_predicates(self, graph):
        s, _, _ = graph
        rows = await s.run("MATCH (p:Person) WHERE p.city STARTS WITH 'Lo' "
                           "RETURN p.name AS name ORDER BY name")
        assert [r['name'] for r in rows] == ['Alice', 'Carol']

    @pytest.mark.asyncio(loop_scope="session")
    async def test_in_list(self, graph):
        s, _, _ = graph
        rows = await s.run("MATCH (p:Person) WHERE p.name IN ['Bob','Dan'] "
                           "RETURN p.name AS name ORDER BY name")
        assert [r['name'] for r in rows] == ['Bob', 'Dan']

    @pytest.mark.asyncio(loop_scope="session")
    async def test_skip_and_limit(self, graph):
        s, _, _ = graph
        rows = await s.run("MATCH (p:Person) RETURN p.name AS name ORDER BY name SKIP 1 LIMIT 2")
        assert [r['name'] for r in rows] == ['Bob', 'Carol']

    @pytest.mark.asyncio(loop_scope="session")
    async def test_realm_isolation(self, pg_client_spr, graph):
        """A Cypher query must not see another realm's rows."""
        s, _, realm = graph
        other = f"{realm}_other"
        await pg_client_spr.create_vertex_table("person", realm=other)
        await pg_client_spr.add_vertex("person", other, payload={"name": "Zed", "age": 99})
        rows = await s.run("MATCH (p:Person) RETURN p.name AS name")
        assert 'Zed' not in [r['name'] for r in rows]
        await pg_client_spr._execute(f'DROP SCHEMA IF EXISTS "{other}" CASCADE')

    @pytest.mark.asyncio(loop_scope="session")
    async def test_injection_does_not_execute(self, graph):
        s, _, _ = graph
        rows = await s.run("MATCH (p:Person) WHERE p.name = $n RETURN p.name AS name",
                           {"n": "'; DROP TABLE person; --"})
        assert rows == []
        # The table is still there.
        assert await s.run("MATCH (p:Person) RETURN count(*) AS c")

    @pytest.mark.asyncio(loop_scope="session")
    async def test_create_and_read_back(self, graph):
        s, _, _ = graph
        await s.run("CREATE (p:Person {name:'Eve', age:30})")
        rows = await s.run("MATCH (p:Person) WHERE p.name = 'Eve' RETURN p.age AS age")
        assert rows == [{'age': '30'}]

    @pytest.mark.asyncio(loop_scope="session")
    async def test_create_relationship(self, graph):
        s, _, _ = graph
        await s.run("CREATE (a:Person {name:'Gina'})-[:KNOWS {since:'2021'}]->(b:Person {name:'Hank'})")
        rows = await s.run("MATCH (a:Person)-[r:KNOWS]->(b:Person) WHERE a.name='Gina' "
                           "RETURN b.name AS b, r.since AS since")
        assert rows == [{'b': 'Hank', 'since': '2021'}]

    @pytest.mark.asyncio(loop_scope="session")
    async def test_merge_is_idempotent(self, graph):
        s, _, _ = graph
        await s.run("MERGE (p:Person {name:'Ivy'})")
        await s.run("MERGE (p:Person {name:'Ivy'})")
        rows = await s.run("MATCH (p:Person) WHERE p.name = 'Ivy' RETURN count(*) AS c")
        assert rows[0]['c'] == 1

    @pytest.mark.asyncio(loop_scope="session")
    async def test_merge_on_match_updates(self, graph):
        s, _, _ = graph
        await s.run("MERGE (p:Person {name:'Kim'})")
        await s.run("MERGE (p:Person {name:'Kim'}) ON MATCH SET p.seen = 'again'")
        rows = await s.run("MATCH (p:Person) WHERE p.name='Kim' RETURN p.seen AS seen")
        assert rows == [{'seen': 'again'}]

    @pytest.mark.asyncio(loop_scope="session")
    async def test_explain_returns_sql(self, graph):
        s, _, _ = graph
        sql = await s.explain("MATCH (p:Person) RETURN p.name AS name")
        assert sql.upper().startswith('SELECT')


class TestUnaryAndRowCounts:
    """Gaps the openCypher TCK harness surfaced."""

    def test_unary_minus_on_property(self):
        from post_graph.cypher.ast import UnaryOp
        expr = parse("MATCH (n) RETURN -n.a AS neg").clauses[1].items[0].expression
        assert isinstance(expr, UnaryOp) and expr.op == '-'

    def test_unary_minus_binds_tighter_than_multiplication(self):
        from post_graph.cypher.ast import UnaryOp
        expr = parse("MATCH (n) RETURN -n.a * n.b AS x").clauses[1].items[0].expression
        assert isinstance(expr.left, UnaryOp)

    def test_negative_literal_folds_to_a_literal(self):
        # Folding keeps numeric-comparison detection working for n.a > -1.
        w = parse("MATCH (n) WHERE n.a > -1 RETURN n").clauses[0].where
        assert isinstance(w.right, A.Literal) and w.right.value == -1

    def test_unary_plus_is_identity(self):
        expr = parse("MATCH (n) RETURN +n.a AS x").clauses[1].items[0].expression
        assert isinstance(expr, A.Property)

    def test_double_negation(self):
        from post_graph.cypher.ast import UnaryOp
        expr = parse("MATCH (n) RETURN --n.a AS x").clauses[1].items[0].expression
        assert isinstance(expr, UnaryOp) and isinstance(expr.operand, UnaryOp)

    def test_unary_minus_translates(self):
        sql, _, _ = _sql("MATCH (n:Person) RETURN -n.age AS neg")
        assert '::numeric' in sql

    @pytest.mark.parametrize("clause,value", [
        ("LIMIT", "-1"), ("SKIP", "-1"), ("LIMIT", "1.5"), ("SKIP", "2.5"),
    ])
    def test_row_counts_must_be_non_negative_integers(self, clause, value):
        # Passed through, these surface as a driver error naming a SQL clause
        # the caller never wrote.
        with pytest.raises(CypherTranslationError):
            _sql(f"MATCH (n:Person) RETURN n.name {clause} {value}")

    @pytest.mark.parametrize("clause", ["LIMIT", "SKIP"])
    def test_valid_row_counts_are_accepted(self, clause):
        sql, _, _ = _sql(f"MATCH (n:Person) RETURN n.name {clause} 5")
        assert clause.replace('SKIP', 'OFFSET') in sql

    def test_negative_row_count_via_parameter_is_caught(self):
        with pytest.raises(CypherTranslationError):
            _sql("MATCH (n:Person) RETURN n.name LIMIT $n", parameters={'n': -3})


@requires_pg
class TestExplain:
    """explain() must describe reads and writes differently, because they are
    executed differently — one SQL statement versus a sequence of client calls."""

    @pytest.fixture()
    async def session(self, pg_client_spr, clean_realm_spr):
        from post_graph import CypherSession
        realm = clean_realm_spr
        await pg_client_spr.create_vertex_table("person", realm=realm)
        await pg_client_spr.create_edge_table("knows", from_vertex_table="person",
                                              to_vertex_table="person", realm=realm)
        return CypherSession(pg_client_spr, realm)

    @pytest.mark.asyncio(loop_scope="session")
    async def test_read_returns_sql(self, session):
        sql = await session.explain("MATCH (p:Person) RETURN p.name AS n")
        assert sql.upper().startswith('SELECT')

    @pytest.mark.asyncio(loop_scope="session")
    async def test_create_describes_client_operations(self, session):
        plan = await session.explain("CREATE (p:Person {name: 'X'})")
        assert 'add_vertex' in plan
        # It must not look like SQL: a write is not one statement, and saying so
        # is the point of the header line.
        assert 'not as one SQL statement' in plan
        assert not plan.upper().startswith('SELECT')

    @pytest.mark.asyncio(loop_scope="session")
    async def test_create_relationship_lists_both_endpoints_and_the_edge(self, session):
        plan = await session.explain(
            "CREATE (a:Person {name:'A'})-[:KNOWS]->(b:Person {name:'B'})")
        assert plan.count('add_vertex') == 2 and 'add_edge' in plan

    @pytest.mark.asyncio(loop_scope="session")
    async def test_merge_shows_both_branches(self, session):
        plan = await session.explain("MERGE (p:Person {name:'X'}) ON MATCH SET p.seen = 'y'")
        assert 'if found' in plan and 'if absent' in plan

    @pytest.mark.asyncio(loop_scope="session")
    async def test_set_is_described(self, session):
        plan = await session.explain("CREATE (p:Person {name:'Y'}) SET p.age = 44")
        assert 'upsert_vertex' in plan and 'age' in plan

    @pytest.mark.asyncio(loop_scope="session")
    async def test_delete_is_described(self, session):
        plan = await session.explain("CREATE (p:Person {name:'Z'}) DETACH DELETE p")
        assert 'delete_vertex' in plan and 'DETACH' in plan

    @pytest.mark.asyncio(loop_scope="session")
    async def test_explain_does_not_execute(self, session):
        await session.explain("CREATE (p:Person {name: 'ghost'})")
        rows = await session.run("MATCH (p:Person) WHERE p.name = 'ghost' RETURN count(*) AS c")
        assert rows[0]['c'] == 0


@requires_pg
class TestWriteAtomicity:
    """Cypher treats a query as a unit. A CREATE that fails partway must leave
    nothing behind — otherwise a rejected query silently half-applies, which is
    worse than either succeeding or failing."""

    @pytest.fixture()
    async def session(self, pg_client_spr, clean_realm_spr):
        from post_graph import CypherSession
        realm = clean_realm_spr
        await pg_client_spr.create_vertex_table("person", realm=realm)
        await pg_client_spr.create_vertex_table("company", realm=realm)
        await pg_client_spr.create_edge_table("knows", from_vertex_table="person",
                                              to_vertex_table="person", realm=realm)
        return CypherSession(pg_client_spr, realm)

    @staticmethod
    async def _people(session):
        return (await session.run("MATCH (p:Person) RETURN count(*) AS n"))[0]['n']

    @pytest.mark.asyncio(loop_scope="session")
    async def test_failed_relationship_rolls_back_its_vertices(self, session):
        # Both vertices are created before the relationship is found to be
        # unroutable, so a non-transactional write would leave them behind.
        with pytest.raises(CypherTranslationError):
            await session.run(
                "CREATE (a:Person {name:'A'})-[:WORKS_AT]->(b:Company {name:'Acme'})")
        assert await self._people(session) == 0

    @pytest.mark.asyncio(loop_scope="session")
    async def test_failed_second_pattern_rolls_back_the_first(self, session):
        with pytest.raises(CypherTranslationError):
            await session.run(
                "CREATE (x:Person {name:'P1'}), (y:Person {name:'P2'}), (z:Unicorn {name:'P3'})")
        assert await self._people(session) == 0

    @pytest.mark.asyncio(loop_scope="session")
    async def test_successful_write_still_commits(self, session):
        await session.run("CREATE (a:Person {name:'A'})-[:KNOWS]->(b:Person {name:'B'})")
        assert await self._people(session) == 2
        rows = await session.run("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN b.name AS b")
        assert rows == [{'b': 'B'}]

    @pytest.mark.asyncio(loop_scope="session")
    async def test_set_on_an_unknown_variable_rolls_back_the_create(self, session):
        with pytest.raises(CypherTranslationError):
            await session.run("CREATE (p:Person {name:'X'}) SET q.age = 1")
        assert await self._people(session) == 0
