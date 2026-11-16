---
layout: default
title: "Indexing JSONB for temporal queries: 31ms to 2.5ms with generated columns"
description: "A date stored in JSONB cannot use a plain index, casting it to DATE is rejected outright, and PostgreSQL 18 changed a default that silently breaks the fix."
---

# Indexing JSONB for temporal queries

### 31ms to 2.5ms with generated columns — and the PostgreSQL 18 default that silently breaks it

Properties in JSONB are flexible and opaque to the planner. That trade is usually
fine, right up to the moment a hot predicate lives inside the document:

```sql
SELECT * FROM relations
WHERE realm = 'acme'
  AND payload->>'valid_from' <= '2005-06-15'
  AND (payload->>'valid_to' IS NULL OR payload->>'valid_to' > '2005-06-15');
```

This is an as-of filter: *what did we believe on this date*. In a temporal graph
it runs on every step of every traversal, which makes it the hottest path in
the system — and it scans.

Getting it to use an index turned out to involve three obstacles, each of which
rules out the obvious answer to the one before. The last of them is new in
PostgreSQL 18 and will silently undo the whole technique if you don't know
about it.

## Obstacle 1: the cast you want is not allowed

The instinct is to index the value as a real date:

```sql
ALTER TABLE relations
  ADD COLUMN pt_valid_from DATE
  GENERATED ALWAYS AS ((payload->>'valid_from')::date) STORED;
```

PostgreSQL refuses:

```
ERROR:  generation expression is not immutable
```

Casting text to `date` is only **STABLE**, not **IMMUTABLE** — the result depends
on the `DateStyle` setting, so the same input can mean different things in two
sessions. A stored generated column has to be immutable, because its value is
written to disk once and must stay correct for ever.

That rules out storing a date. Which leaves storing text — and text comparison
has its own problem.

## Obstacle 2: partial dates sort as shorter strings

ISO-8601 has a property worth leaning on: lexical order and chronological order
agree. `'2004-12-31' < '2005-06-15'` as text and as dates alike. So text
comparison is fine — until the data contains partial dates, which extracted data
constantly does. An LLM reading "signed in 2004" produces `'2004'`, not
`'2004-01-01'`.

And `'2004' > '2005-06-15'`? No — but `'2024' > '2024-06-15'`? Also no. The
failure is subtler: `'2024'` sorts *before* every full date in 2024, which is
right for a start date and wrong for an end date, and the shortness is doing the
work rather than the meaning.

So the value has to be normalised before it is compared. With a date cast off
the table, that means string surgery using only immutable functions:

```python
def padded_date_sql(expr: str) -> str:
    """'2024' becomes '2024-01-01'; '2024-06' becomes '2024-06-01'."""
    return (
        f"(split_part({expr}, '-', 1) || '-' || "
        f"COALESCE(NULLIF(split_part({expr}, '-', 2), ''), '01') || '-' || "
        f"COALESCE(NULLIF(split_part({expr}, '-', 3), ''), '01'))"
    )
```

`split_part`, `COALESCE` and `NULLIF` are all IMMUTABLE, so the composition is
too, and PostgreSQL accepts it in a generated column.

One detail worth copying: a NULL key must yield NULL rather than `'--'`.

```python
src = f"payload->>'{key}'"
expr = f"CASE WHEN {src} IS NULL THEN NULL ELSE {padded_date_sql(src)} END"
```

Without the `CASE`, "no stated period" and "a period that failed to parse"
become the same value, and you lose the ability to tell a gap in the data from a
bug in the extractor.

## Obstacle 3: why not just an expression index?

A fair objection: PostgreSQL will happily index an expression, so why add a
column at all?

```sql
CREATE INDEX ON relations ((payload->>'valid_from'));
```

That works, and for a bare accessor it is the right tool. It stops working as
soon as the indexed value is the *normalised* one, because the planner matches
an expression index only when the query repeats the expression verbatim. Every
call site would have to carry this:

```sql
WHERE (split_part(payload->>'valid_from','-',1) || '-' ||
       COALESCE(NULLIF(split_part(payload->>'valid_from','-',2),''),'01') || '-' ||
       COALESCE(NULLIF(split_part(payload->>'valid_from','-',3),''),'01')) <= '2005-06-15'
```

Miss one character and you get a sequential scan with no warning. A stored
generated column names the expression once, so the query is a plain column
reference that the planner matches trivially:

```sql
WHERE pt_valid_from <= '2005-06-15'
```

The normalisation is then computed once per write rather than once per row per
query, which is the other half of the win.

## The measurement

Measured on 65,000 relation rows, the as-of predicate above:

| | buffers | warm latency |
| :--- | ---: | ---: |
| `payload->>'valid_from'`, sequential scan | 588 | ~31 ms |
| `pt_valid_from`, bitmap index scan | 27 | ~2.5 ms |

A 12× time improvement and a 22× reduction in buffers touched. The margin grows
with the table, because one side of it is a scan.

## The PostgreSQL 18 trap

Here is the part that is newly dangerous. The full DDL is:

```sql
ALTER TABLE relations ADD COLUMN IF NOT EXISTS "pt_valid_from" TEXT
  GENERATED ALWAYS AS (
    CASE WHEN payload->>'valid_from' IS NULL THEN NULL
         ELSE (split_part(payload->>'valid_from','-',1) || '-' ||
               COALESCE(NULLIF(split_part(payload->>'valid_from','-',2),''),'01') || '-' ||
               COALESCE(NULLIF(split_part(payload->>'valid_from','-',3),''),'01'))
    END
  ) STORED;

CREATE INDEX IF NOT EXISTS "idx_relations_pt_valid_from"
  ON relations ("pt_valid_from");
```

Note the `STORED`. Until PostgreSQL 17, omitting it was a syntax error — the
keyword was mandatory, so you could not get this wrong.

**PostgreSQL 18 added virtual generated columns and made VIRTUAL the default.**
The same DDL without the keyword now succeeds and means something else:

```sql
-- On PostgreSQL 18:
CREATE TEMP TABLE probe (a int, b int GENERATED ALWAYS AS (a * 2));
SELECT attname, attgenerated FROM pg_attribute
WHERE attrelid = 'probe'::regclass AND attname = 'b';
--  attname | attgenerated
--  b       | v              ← virtual
```

A virtual generated column is computed on read and **cannot be indexed**. So on
PostgreSQL 18, a promoted column declared without `STORED` creates the column
successfully, then fails at `CREATE INDEX` — or worse, if the index creation is
best-effort, succeeds at everything and quietly returns you to the sequential
scan this whole technique exists to remove. On 17 the identical code raised at
the first statement.

If you use generated columns for indexability, that is worth a test rather than
a comment:

```python
rows = await conn.fetch(
    "SELECT attname, attgenerated FROM pg_attribute "
    "WHERE attrelid = $1::regclass AND attgenerated <> ''", table)
for r in rows:
    assert r["attgenerated"].decode() == "s", (r["attname"], "not stored")
```

(`attgenerated` is PostgreSQL's one-byte `"char"` type, which asyncpg returns as
bytes — `b's'`, not `'s'`. That one cost me a test run.)

## In short

- A date in JSONB cannot be indexed usefully as a date, because `text::date` is
  STABLE and generated columns demand IMMUTABLE.
- Normalise partial dates with `split_part` / `COALESCE` / `NULLIF`, which are
  immutable, and lean on ISO-8601 sorting lexically.
- Prefer a stored generated column over an expression index once the expression
  is non-trivial, so the planner matches a column reference instead of a string
  your call sites have to reproduce exactly.
- Write `STORED` explicitly, and assert it, because PostgreSQL 18 changed what
  omitting it means.

---

This is from [post-graph](https://github.com/crajah/post-graph), an Apache 2.0
library that makes PostgreSQL behave like a graph database — realms, audit
history, pgvector search and filtered traversal, as ordinary tables you own
rather than an engine you operate. The temporal promotion described here is what
makes `traverse(as_of=...)` an index scan.

`pip install post-graph`
