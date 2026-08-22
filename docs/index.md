---
layout: default
title: "Introducing post-graph: A Graph Database That Is Just PostgreSQL"
description: "Multi-tenant realms, audit history, pgvector search and filtered traversal — as ordinary tables you own, not an engine you operate."
---

# Introducing post-graph: A Graph Database That Is Just PostgreSQL

### Multi-tenant realms, audit history, pgvector search and filtered traversal — as ordinary tables you own, not an engine you operate

**[GitHub](https://github.com/crajah/post-graph)** · **[PyPI](https://pypi.org/project/post-graph/)** · `pip install post-graph` · Apache 2.0

---

Most teams reach for a graph database at the moment they realise a join is really a walk. The usual next step is to stand up Neo4j, or install Apache AGE, or accept that the graph lives somewhere other than the data it describes.

There is a less obvious option: PostgreSQL is already a competent graph database, provided something takes care of the schema, the traversal SQL, the tenancy and the audit trail for you.

That is what `post-graph` is. Not a storage engine, not an extension — a Python library that builds the tables, writes the recursive CTEs, and leaves you with a graph made of ordinary PostgreSQL objects that you can index, constrain, back up and inspect with `psql`.

```bash
pip install post-graph
```

## The claim, stated plainly

A property graph is two tables. Vertices with a JSONB payload, edges with a from-id, a to-id and a type. Everything a graph database sells you on top of that — traversal, pattern matching, tenancy, indexing — is either SQL you can generate, or infrastructure PostgreSQL already has.

What you gain by staying in PostgreSQL is not philosophical. It is that **the graph and the rest of your data share one transaction, one backup and one consistency model.** You can write a vertex, its embedding and an application row in a single transaction and have all three commit or none.

What you give up is openCypher, and a genuinely mature ecosystem of graph tooling. That trade is discussed honestly at the end.

## Where the alternatives sit

**Neo4j** is the reference implementation of the property-graph model, with Cypher, a mature query planner and a large ecosystem. It is also a separate system: separate operational surface, separate backup story, and no transaction that spans it and your relational data.

**Apache AGE** is the closest comparison, because it shares the premise — graphs belong in PostgreSQL — and reaches a different conclusion about how. AGE is a PostgreSQL extension providing openCypher. Each graph becomes a schema; labels become tables inheriting from `_ag_label_vertex` and `_ag_label_edge`; properties are stored as `agtype`, described in its own documentation as *"a superset of Json and a custom implementation of JsonB."*

The consequential difference is one sentence in the AGE manual:

> "It is recommended that no DML or DDL commands are executed in the namespace that is reserved for the graph."

That is a reasonable thing for an extension to say — it owns those tables and rewrites them across versions. But it forecloses most of what follows in this article. You cannot safely add a `vector` column with an HNSW index to an AGE vertex, nor your own foreign keys, nor audit triggers, nor a unique index on a property to enforce entity identity.

There is also a practical constraint worth checking before adopting: AGE's setup documentation lists support for **PostgreSQL 11 through 15**, with the project homepage announcing compatibility with **16** in a 1.5.0 release candidate. On PostgreSQL 17 or 18, that means pinning backwards.

**pgvector alone** solves similarity search beautifully and models no relationships at all. The two are complementary rather than competing — post-graph uses pgvector.

|  | Neo4j | Apache AGE | post-graph |
| :--- | :---: | :---: | :---: |
| Runs inside your PostgreSQL | ❌ | ✅ | ✅ |
| openCypher | ✅ | ✅ | ⚠️ documented subset |
| Tables you may index and constrain | n/a | ❌ discouraged | ✅ |
| Vector search on vertices *and* edges | via plugin | ❌ | ✅ |
| Shadow audit log of every mutation | enterprise | ❌ | ✅ |
| Append-only history per element | ❌ | ❌ | ✅ |
| Schema-per-tenant isolation | enterprise | per graph | ✅ |
| Needs a recent PostgreSQL | n/a | ⚠️ 11–16 | ✅ any supported |

## What the tables actually look like

Vertex and edge tables are created for you, with the constraints a graph needs and a relational database can enforce:

```python
from post_graph import AsyncPostGraph

pg = AsyncPostGraph(dsn="postgresql://localhost/mydb", schema_per_realm=True)
await pg.connect()

await pg.create_vertex_table("people",  realm="acme", vector_dim=1536)
await pg.create_vertex_table("companies", realm="acme", vector_dim=1536)
await pg.create_edge_table("works_at",
                           from_vertex_table="people",
                           to_vertex_table="companies",
                           realm="acme")
```

Inspect the result and there is no mystery layer:

```
Foreign-key constraints:
    "works_at_realm_from_id_fkey" FOREIGN KEY (realm, from_id)
        REFERENCES acme.people(realm, id) ON DELETE CASCADE
    "works_at_realm_to_id_fkey"   FOREIGN KEY (realm, to_id)
        REFERENCES acme.companies(realm, id) ON DELETE CASCADE
```

Two details in there matter more than they look.

**The endpoints are typed independently.** A `works_at` edge structurally cannot point its source at a company, because the foreign key references `people`. Cypher's label model expresses that as a convention; here PostgreSQL refuses the write.

**The keys are composite on `(realm, …)`.** Tenancy is not a filter applied by a query builder that might be forgotten — an edge in one realm cannot reference a vertex in another, because the database will not permit the row.

## Two levels of tenancy

`realm` is macro-isolation. With `schema_per_realm=True` each tenant gets its own PostgreSQL schema, so isolation is physical and dropping a tenant is dropping a schema. Without it, realms partition by column in shared tables.

`space` is micro-isolation *within* a realm — `production`, `staging`, `sandbox`, per-workspace — carried on both vertices and edges. The reserved space `__all__` queries across every space, and is rejected at write time so nothing can be created into it.

The distinction earns its keep the moment a traversal is involved, which is why every read path — including `traverse()` — takes a space. A walk that begins from a correctly scoped vertex and then leaves the first hop unfiltered is a cross-tenant leak that no single-hop test will catch.

## Vectors on vertices and on edges

Each vertex table can carry an `embedding` column with an HNSW index, so similarity search is a method rather than a second datastore:

```python
hits = await pg.vector_search("people", realm="acme", space="production",
                              query_embedding=vec, limit=10)
```

Edges may be embedded too — `vector_search_edges` — which is unusual and mostly optional. Relationships are normally reached by traversing from a matched vertex rather than by similarity. It exists for the cases where the *description* of a relationship is the thing worth searching.

The payoff for keeping this in PostgreSQL is the single transaction: a vertex and its embedding commit together, so there is no window in which the graph and the vector index disagree about what exists.

## Audit and history, because graphs are edited

Every table gets a **shadow audit table** and a trigger recording each insert, update and delete with old and new row images, alongside an **append-only `_data` history table** for element-level versions. Deleting a vertex cascades its edges away; the audit trail of what was there does not go with it.

This is the part that is genuinely awkward to bolt onto an engine that owns its own storage, and nearly free when the tables are yours: it is triggers and one extra table per element type.

## Traversal is generated SQL, not a query language

Recursive CTEs do the walking, exposed as three methods:

```python
await pg.get_neighbors(realm, "people", pid, edge_tables=["works_at"])
await pg.traverse(realm, "people", pid, edge_tables=["works_at"], max_depth=3)
await pg.shortest_path(realm, "people", a, "people", b, edge_tables=["knows"])
```

Since 0.6.0 the walk is filterable, and the filters apply to **every step** rather than to the final result:

```python
await pg.traverse(
    realm, "entities", start_id, edge_tables=["relations"],
    max_depth=3,
    relation_types=["generates_cash_flow", "consumes_cash"],
    as_of="2020",                       # only edges whose stated validity covers it
    payload_null_keys=["superseded_by"],  # skip edges the caller has closed
    space="production",
)
```

That distinction is the whole design. Filtering a result set still allows a path to travel *through* an excluded edge to reach something that then presents as reachable — the offending edge vanishes from the output while the conclusion it enabled survives. Constraining each step means the path is never walked.

`as_of` follows one rule worth stating: **an edge that states no period qualifies at every date.** Silence about when something held means it held throughout, not that it held nowhere. A graph that records no dates is therefore completely unaffected by temporal filtering.

`payload_null_keys` is deliberately generic. post-graph does not know what "superseded" means; it knows how to skip edges whose payload key is absent. The meaning stays in the layer that has it.

## Properties you filter on, as indexed columns

Properties live in `payload` JSONB, which is flexible and opaque to the planner. `payload->>'valid_from' <= '2024-01-01'` cannot use an index, so the `as_of` filter — running on every step of every walk — degraded to a sequential scan.

Since 1.0.0 the hot keys are promoted to generated columns that PostgreSQL maintains from `payload`. Nothing about writing changes:

```python
await pg.create_edge_table(
    "relations",
    from_vertex_table="entities", to_vertex_table="entities",
    realm=realm,
    promoted_keys=["superseded_by"],   # p_superseded_by, indexed
)

await pg.add_edge("relations", realm, a, b, "generates_cash_flow",
                  payload={"valid_from": "2024-06"})   # unchanged
```

`valid_from` and `valid_to` are promoted by default as `pt_valid_from` and `pt_valid_to`, holding the date normalised to `YYYY-MM-DD` so a partial date like `'2024'` orders correctly against a full one. On 65k rows the temporal filter moved from a 588-buffer sequential scan to a 27-buffer bitmap index scan — roughly 31ms to 2.5ms warm — and the margin grows with the table.

Two things this deliberately does not do. It does not change the model: `Vertex` and `Edge` carry no new fields, because a promoted column is derived, read-only, and present only on tables created since the feature existed. And it does not touch existing tables — a realm created earlier keeps working and returns the same rows through `payload->>`, without the index.

## A Cypher subset, measured rather than claimed

```python
from post_graph import CypherSession

session = CypherSession(pg, realm="my_realm")
await session.run(
    "MATCH (p:Person)-[:KNOWS*1..3]->(f:Person) "
    "WHERE p.name = $name AND f.age > 30 "
    "RETURN DISTINCT f.name AS friend ORDER BY friend",
    {"name": "Alice"},
)
```

A label is a vertex table, a relationship type is a value in `relation_type`, a property is a payload key — read through a promoted column when one exists, so Cypher inherits the index work for free.

Reads compile to a single SQL statement. Writes do not: `CREATE`, `MERGE`, `SET` and `DELETE` run through the client's own methods so audit tables, triggers and realm rules behave as for any other caller, and the whole query runs in one transaction, so a `CREATE` whose relationship cannot be routed leaves nothing behind. `session.explain(query)` shows either — the SQL for a read, the operation sequence for a write — without running it.

The interesting part is what it refuses. `WITH`, `UNION`, `OPTIONAL MATCH`, path variables, multiple labels and `RETURN *` all raise, with a position or a reason. A query answered slightly differently from how it was written is worse than one that is rejected, because the caller cannot tell which happened.

Conformance is measured, not asserted. The openCypher TCK is 1,615 Cucumber scenarios; run against post-graph, 179 pass, 0 fail and 0 error, with 1,132 refused as outside the subset and 304 needing TCK machinery the harness does not model. That 179 is not a coverage score — most of the corpus assumes a schema-free graph where `CREATE (:Foo)` invents a label and `MATCH (n)` scans every node, which is not this model. The numbers that matter are the two held at zero, and the harness has already earned its place: it found unary minus missing entirely, and negative `SKIP`/`LIMIT` reaching PostgreSQL as a driver error naming a clause the caller never wrote.

## What this is not

**Cypher is a subset, not the whole language.** Since 1.0.0 there is a Cypher query surface (see below), but it is a documented subset and it refuses what it cannot express. If full Cypher is your primary interface, Neo4j or AGE remains the better tool. What post-graph offers is filtered traversal that composes with the tenancy, temporal and vector predicates the same query already needs — which a separate graph engine cannot do — with Cypher available on top of it.

**There is no query planner for graph patterns.** Recursive CTEs are executed by PostgreSQL's planner, which is excellent at set operations and knows nothing about graph cardinality. Deep unbounded traversal on a dense graph will find that out: three hops from one well-connected vertex in a real 3,000-vertex graph reached over 25,000 rows. Bound your depth and cap your fan-out.

**It is a library, not a server.** No graph browser, no visualisation, no Bloom. `psql` and your existing PostgreSQL tooling are the interface.

## Who it is for

Reach for post-graph when the graph is *part of* an application whose data already lives in PostgreSQL, when tenancy and audit are requirements rather than nice-to-haves, and when the queries are bounded traversals rather than open-ended pattern matching. Multi-tenant SaaS with per-customer graphs, knowledge graphs with embeddings, entity resolution over documents, anything where "which rows changed and who changed them" is a question someone will eventually ask.

Reach for Neo4j when graph querying *is* the product and Cypher is the interface your team wants. Reach for AGE when you want Cypher inside PostgreSQL and can live on a supported major version without adding your own columns, constraints or triggers to the graph tables.

## Try it

```bash
pip install post-graph
createdb mydb && psql -d mydb -c "CREATE EXTENSION vector;"
```

`post-graph` is Apache 2.0 licensed and on PyPI. It is also the storage layer beneath **[post-graph-rag](https://github.com/crajah/post-graph-rag)** ([write-up](https://crajah.github.io/post-graph-rag/)), a Graph RAG library that leans on every feature described here — the vector columns for retrieval, the composite keys for tenancy, the audit history for re-indexing, and the filtered traversal for multi-hop question answering over corpora whose facts change.

A paper covering the architecture and evaluation methodology is going to arXiv — link to follow.

**GitHub:** https://github.com/crajah/post-graph · **PyPI:** `pip install post-graph`

The broader point survives the library: before adopting a second datastore for graph-shaped data, it is worth checking what your existing one already does. A property graph is two tables, a traversal is a recursive CTE, and everything else is the part you were going to have to operate anyway.
