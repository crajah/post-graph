# Examples

Ten runnable scripts. They need **PostgreSQL and nothing else** — no LLM, no
API keys, no network service. Point them at a database once:

```bash
export POSTGRES_URI=postgresql://localhost:5432/postgres
psql -d postgres -c "CREATE EXTENSION IF NOT EXISTS vector;"   # for 02 only

pip install post-graph
cd examples && python 01_quickstart.py
```

Each uses its own timestamped realm, so they never collide, can be run in any
order, and leave your other data alone.

| | what it shows |
| :--- | :--- |
| [`01_quickstart.py`](01_quickstart.py) | Vertices, edges, a one-hop walk, a recursive-CTE traversal and a shortest path |
| [`02_vector_search.py`](02_vector_search.py) | pgvector similarity across both main and history tables |
| [`03_realms_and_spaces.py`](03_realms_and_spaces.py) | `realm` for tenant isolation, `space` for sub-grouping within it |
| [`04_schema_per_realm.py`](04_schema_per_realm.py) | Physical isolation — a PostgreSQL schema per tenant |
| [`05_range_queries.py`](05_range_queries.py) | `>`/`<`/`>=` on JSONB fields, ordering, counting, payload indexes, bulk delete |
| [`06_history_and_audit.py`](06_history_and_audit.py) | Append-only history you write, and a trigger-written audit log you don't |
| [`07_opencypher.py`](07_opencypher.py) | openCypher queries against the same graph |
| [`08_transactions.py`](08_transactions.py) | Graph writes and your own tables in one transaction |
| [`09_uuid_keys.py`](09_uuid_keys.py) | Automatic UUIDs alongside the generated integer ids |
| [`10_full_tour.py`](10_full_tour.py) | Everything above end to end — the long version |

## Two that are worth reading even if you never run them

**`05_range_queries.py`** shows the comparison rule that trips people up:
numbers compare numerically, strings compare as text. That is deliberate —
zero-padded timestamps and version strings sort correctly as text and would
not if everything were coerced to a number. It also shows `delete_vertices`
refusing an empty predicate, because a bulk delete that silently means
"everything" is not a convenience.

**`06_history_and_audit.py`** shows the difference between the two tables
every graph table gets. History is what you chose to record. The audit log is
what happened — written by triggers, attributed to a session user, including
the operations you would rather had not occurred.

## Note

These were all executed against a live PostgreSQL before being committed. If
one fails for you, that is a bug worth reporting rather than a setup mistake
on your side — see [CONTRIBUTING.md](../CONTRIBUTING.md).
