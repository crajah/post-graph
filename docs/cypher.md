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

`session.explain(query)` returns the SQL a read becomes, which is the honest way
to check what you are actually asking the database.

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
and arithmetic.

**Functions** — `count`, `sum`, `avg`, `min`, `max`, `collect` (with `DISTINCT`),
`id`, `labels`, `type`, `properties`, `keys`, `toUpper`, `toLower`, `trim`,
`length`, `size`, `coalesce`, `toString`, `toInteger`, `toFloat`, `abs`, `ceil`,
`floor`, `round`, `exists`. Using an aggregate adds the `GROUP BY` for you.

**Writing** — `CREATE` for nodes and relationships, `MERGE` on node properties
with `ON CREATE SET` and `ON MATCH SET`, `SET`, `DELETE`. Writes go through the
client's own methods rather than generated SQL, so audit tables, triggers and
realm rules behave exactly as they do for any other caller.

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
