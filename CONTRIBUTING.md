# Contributing to post-graph

Bug reports, failing test cases and pull requests are all welcome. This file
covers the parts of the setup that are specific to this project — the ones you
would otherwise discover by losing an afternoon to them.

## What you need

**PostgreSQL with pgvector.** The test suite talks to a real database rather
than mocking one, because most of what it verifies — generated columns, JSONB
predicates, vector search, cascade behaviour — lives in the database and cannot
be mocked without testing the mock instead.

```bash
# macOS
brew install postgresql@16 pgvector

# Debian/Ubuntu (match your server version)
sudo apt install postgresql-16 postgresql-16-pgvector

createdb post_graph_test
psql -d post_graph_test -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

**Note that the suite skips rather than fails** when PostgreSQL is
unreachable, so a run reporting no failures has not necessarily tested
anything. Check the count, and check for skips.

## Setup and tests

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"

export POST_GRAPH_TEST_DSN=postgresql://localhost/post_graph_test
pytest -q                  # ~635 tests, around five minutes
pytest tests/test_range_queries.py -q   # one file
pytest -q -k traversal                  # one behaviour
```

The suite must be green before you open a pull request. CI runs the same
command on Python 3.9 and 3.13 against a pgvector container.

## Linting

```bash
pip install ruff
ruff check .               # must pass; CI enforces it
ruff check . --fix         # fixes most of what it finds
```

The configuration in `pyproject.toml` selects `F`, `B`, `E9` and `I` — rules
that catch defects, not house style. **Do not run `ruff format`.** It would
rewrite several thousand lines and take the history behind them with it, for
no defect caught. Formatting is deliberately not enforced; match the style of
the file you are editing.

## Pull requests

- **One change per pull request.** A bug fix and a refactor in the same diff
  take several times longer to review than the two separately.
- **A test that fails before your change and passes after** is the most useful
  thing in a pull request. For a bug, that test is more valuable than the fix;
  send it on its own if you would rather.
- **Explain the why in the commit message.** What changed is visible in the
  diff. Why it changed, and what you ruled out, is not.
- **Say what you measured.** For anything touching query construction or
  indexing, include the SQL you expect to be generated, or a benchmark. Both
  client backends (asyncpg and SQLAlchemy) must behave identically, so a
  change to one usually needs the same change and the same test in the other.

## Things worth knowing before you change them

- **Two backends, one behaviour.** `client_asyncpg.py` and
  `client_sqlalchemy.py` are held to the same semantics. A feature added to one
  is unfinished until it exists in both, with tests in both.
- **Payload keys are validated before they reach SQL.** Range queries and
  payload indexes build SQL from key names, so those names go through
  `_validate_payload_key` first. Do not route around it.
- **`B904` is currently ignored in `post_graph/`.** The clients translate
  driver exceptions into domain errors deliberately. Adding `from e` at those
  sites would improve tracebacks and is a genuinely good first contribution —
  as its own change, not mixed into another one.

## Reporting a bug

Include the table definition, the call you made, and what you expected. The
generated SQL is the most useful thing you can attach; most bugs here are
visible in it.

## Licence

Apache 2.0. Contributions are accepted under the same licence.
