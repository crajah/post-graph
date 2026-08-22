---
title: openCypher over post-graph
---

# openCypher over post-graph

A documented subset of openCypher, translated to a single SQL statement against
post-graph's own tables. It is a query surface over the graph you already have,
not a second storage engine.

```python
from post_graph import AsyncPostGraph, CypherSession

client = AsyncPostGraph(dsn="postgresql://localhost/mydb")
await client.connect()

session = CypherSession(client, realm="my_realm")
rows = await session.run(
    "MATCH (p:Person)-[:KNOWS]->(f:Person) "
    "WHERE p.name = $name AND f.age > 30 "
    "RETURN f.name AS friend ORDER BY friend",
    {"name": "Alice"},
)
```

`session.explain(query)` shows what a query will do without doing it: the SQL for
a read, and for a write the sequence of client operations it will perform, since
a write is not one statement. Running it changes nothing.

## How Cypher maps onto post-graph

| Cypher | post-graph |
| :--- | :--- |
| `(n:Person)` | the `person` vertex table |
| `[r:KNOWS]` | an edge row whose `relation_type` is `KNOWS` |
| `n.name` | `payload->>'name'`, or a promoted column when one exists |
| `n.uuid`, `n.fqid`, `id(n)` | the real columns of those names |
| `RETURN n` | a JSON object: `id`, `uuid`, `label`, `properties` |

A label is a table, so a node has exactly one. Relationship *types* are values in
a column, not tables — `[:KNOWS]` filters rows rather than choosing where to
look. The edge table is inferred from the labels at each end; if two edge tables
join the same pair, the query is rejected as ambiguous rather than guessed.

Labels and relationship types are matched case-insensitively against your table
names, so `(p:Person)` finds a table called `person`.

## Supported

**Reading** — `MATCH`, `WHERE`, `RETURN`, `DISTINCT`, `ORDER BY` (`ASC`/`DESC`),
`SKIP`, `LIMIT`.

**Patterns** — nodes with labels and inline property maps; relationships with
types, alternatives (`[:A|B]`), direction (`->`, `<-`, undirected), and inline
properties; multi-hop chains; variable-length paths (`*`, `*2`, `*1..3`, `*..5`),
which must begin and end on the same label.

**Predicates** — `=`, `<>`, `<`, `<=`, `>`, `>=`, `AND`, `OR`, `XOR`, `NOT`,
`IS NULL`, `IS NOT NULL`, `IN`, `STARTS WITH`, `ENDS WITH`, `CONTAINS`, `=~`,
arithmetic, and unary `-`/`+`.

**Functions** — `count`, `sum`, `avg`, `min`, `max`, `collect` (with `DISTINCT`),
`id`, `labels`, `type`, `properties`, `keys`, `toUpper`, `toLower`, `trim`,
`length`, `size`, `coalesce`, `toString`, `toInteger`, `toFloat`, `abs`, `ceil`,
`floor`, `round`, `exists`. Using an aggregate adds the `GROUP BY` for you.

**Writing** — `CREATE` for nodes and relationships, `MERGE` on node properties
with `ON CREATE SET` and `ON MATCH SET`, `SET`, `DELETE`. Writes go through the
client's own methods rather than generated SQL, so audit tables, triggers and
realm rules behave exactly as they do for any other caller.

A write query is **atomic**. The whole query runs in one transaction, so a
`CREATE` whose relationship cannot be routed does not leave its nodes behind.
If you constructed the client with a bare connection rather than a pool, you are
managing the transaction yourself and the query joins yours rather than opening
a second one.

**Parameters** — `$name`, supplied as a dict. Every literal and parameter is
bound, never interpolated, so a value that looks like SQL is data.

## Not supported

These raise `CypherSyntaxError` or `CypherTranslationError` rather than being
approximated, because a query answered slightly differently from how it was
written is worse than one that is refused:

- `WITH`, `UNION`, `UNWIND`, `OPTIONAL MATCH`, `CASE`
- path variables (`p = (a)-[]->(b)`) and path functions
- binding a variable to a variable-length relationship
- multiple labels on one node
- `RETURN *`
- `MATCH` combined with a write clause in one query — read first, then write
- computed values in `CREATE`/`SET`; only literals and parameters are written

## Numbers, text and types

Properties are stored in JSONB and read back as text. Comparing a property to a
number compares numerically, so `n.age > 9` does not exclude someone aged 28 the
way a lexical comparison would. Comparing to a string compares as text. Values
returned from `payload` arrive as strings; `toInteger()` and `toFloat()` convert
where you need a number in the result.

## Performance

Property predicates use a [promoted column](../post_graph/promoted.py) when the
table has one, and fall back to `payload->>` otherwise — same rows either way,
but the promoted path is indexed. Promote the keys you filter on:

```python
await client.create_edge_table(
    "relations",
    from_vertex_table="entities", to_vertex_table="entities",
    realm=realm,
    promoted_keys=["t_expired", "status"],
)
```

`valid_from` and `valid_to` are promoted by default.

Variable-length patterns become a recursive CTE bounded at 8 hops when you give
no upper bound, so an unbounded `*` cannot walk a cyclic graph forever.

## Conformance

Conformance is measured against the [openCypher TCK](https://github.com/opencypher/openCypher/tree/master/tck),
not asserted. Fetch the corpus and run it:

```bash
tests/tck/fetch_tck.sh
pytest tests/test_tck.py
```

Of 1,615 TCK scenarios:

| | |
| :--- | ---: |
| passed | 179 |
| failed | 0 |
| errored | 0 |
| unsupported — refused by this dialect | 1,132 |
| skipped — needs TCK machinery this harness does not model | 304 |

Read that honestly. 179 is not a coverage score: most of the corpus assumes a
schema-free graph where `CREATE (:Foo)` invents a label and `MATCH (n)` scans
every node, which is not post-graph's model — a label is a table that has to
exist. Those scenarios are counted as *unsupported*, which is the documented
behaviour, rather than hidden.

What the harness is for is the two numbers that must stay at zero. **errored**
means a scenario raised something other than a clean refusal — always a bug.
**failed** means a query ran and returned the wrong answer — always a bug. The
suite asserts both are zero and that the passing count does not regress.

It has already earned that: it found unary minus missing entirely, and negative
and fractional `SKIP`/`LIMIT` being passed through to PostgreSQL, where they
surfaced as a driver error naming a SQL clause the caller never wrote.

The 304 skipped scenarios need `CALL` procedures, side-effect assertions, or the
TCK's named fixture graphs — machinery this harness does not model, and separate
from what the dialect supports.
