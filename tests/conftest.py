import os
import uuid

import pytest
import pytest_asyncio

DSN = os.environ.get("POST_GRAPH_TEST_DSN", "postgresql://localhost/post_graph_test")

requires_pg = pytest.mark.skipif(
    os.environ.get("POST_GRAPH_SKIP_INTEGRATION", "0") == "1",
    reason="POST_GRAPH_SKIP_INTEGRATION is set",
)


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def pg_client():
    """Session-scoped AsyncPostGraph client connected to a test database.

    Skips the entire session if the database is unreachable.
    """
    from post_graph import AsyncPostGraph

    client = AsyncPostGraph(dsn=DSN)
    try:
        await client.connect()
    except Exception as exc:
        pytest.skip(f"PostgreSQL not reachable at {DSN}: {exc}")
    yield client
    await client.close()


@pytest.fixture()
def realm():
    """A unique realm name for test isolation."""
    return f"test_{uuid.uuid4().hex[:12]}"


@pytest.fixture()
async def clean_realm(pg_client, realm):
    """Yield a unique realm and delete it after the test."""
    yield realm
    try:
        await pg_client.delete_realm(realm)
    except Exception:
        pass


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def pg_client_spr():
    """Session-scoped AsyncPostGraph client with schema_per_realm=True."""
    from post_graph import AsyncPostGraph

    client = AsyncPostGraph(dsn=DSN, schema_per_realm=True)
    try:
        await client.connect()
    except Exception as exc:
        pytest.skip(f"PostgreSQL not reachable at {DSN}: {exc}")
    yield client
    await client.close()


@pytest.fixture()
async def clean_realm_spr(pg_client_spr, realm):
    """Yield a unique realm and drop its schema after the test."""
    yield realm
    try:
        await pg_client_spr.delete_realm(realm)
    except Exception:
        pass
    try:
        await pg_client_spr._execute(f'DROP SCHEMA IF EXISTS "{realm}" CASCADE')
    except Exception:
        pass


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def has_pgvector(pg_client):
    """Return True if the test database has the pgvector extension available."""
    try:
        await pg_client._execute("CREATE EXTENSION IF NOT EXISTS vector")
        return True
    except Exception:
        return False


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def sa_engine():
    """Session-scoped SQLAlchemy AsyncEngine for integration tests."""
    from sqlalchemy.ext.asyncio import create_async_engine

    sa_dsn = DSN.replace("postgresql://", "postgresql+asyncpg://", 1)
    engine = create_async_engine(sa_dsn)
    try:
        async with engine.connect() as conn:
            await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
    except ImportError:
        # A broken install, not an absent database. Skipping here is how a
        # missing greenlet turned the whole SQLAlchemy half of the suite --
        # some 240 tests -- into skips for four consecutive CI runs, with one
        # stray test the only thing that went red. The skip below exists so
        # the suite runs where there is no PostgreSQL; it must not also
        # swallow the case where the dependency itself is gone.
        raise
    except Exception as exc:
        pytest.skip(f"SQLAlchemy engine cannot connect at {sa_dsn}: {exc}")
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def sa_client(sa_engine):
    """Session-scoped SQLAlchemyPostGraph client."""
    from post_graph.client_sqlalchemy import SQLAlchemyPostGraph

    return SQLAlchemyPostGraph(sa_engine)


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def sa_client_spr(sa_engine):
    """Session-scoped SQLAlchemyPostGraph client with schema_per_realm=True."""
    from post_graph.client_sqlalchemy import SQLAlchemyPostGraph

    return SQLAlchemyPostGraph(sa_engine, schema_per_realm=True)


@pytest.fixture()
async def sa_clean_realm(sa_client, realm):
    """Yield a unique realm and delete it after the test (SQLAlchemy client)."""
    yield realm
    try:
        await sa_client.delete_realm(realm)
    except Exception:
        pass


@pytest.fixture()
async def sa_clean_realm_spr(sa_client_spr, realm):
    """Yield a unique realm and drop its schema after the test (SQLAlchemy SPR)."""
    yield realm
    try:
        await sa_client_spr.delete_realm(realm)
    except Exception:
        pass
    try:
        from sqlalchemy import text
        async with sa_client_spr.engine_or_connection.begin() as conn:
            await conn.execute(text(f'DROP SCHEMA IF EXISTS "{realm}" CASCADE'))
    except Exception:
        pass


# --------------------------------------------------------------- role helpers
#
# Privilege tests need a role that holds exactly the grants under test and
# nothing else, and a way to see the SQL a method emits rather than only its
# effect. Both started life inside test_update_vertex.py; they are here because
# three test modules now measure the same kind of thing.


async def grant_role(conn, role, grants):
    """Create a NOLOGIN role and apply *grants*, a list of SQL fragments."""
    await conn.execute(f'CREATE ROLE "{role}" NOLOGIN')
    await conn.execute(f'GRANT USAGE ON SCHEMA public TO "{role}"')
    for g in grants:
        await conn.execute(g)


async def drop_role(conn, role):
    try:
        await conn.execute(f'REASSIGN OWNED BY "{role}" TO CURRENT_USER')
        await conn.execute(f'DROP OWNED BY "{role}"')
        await conn.execute(f'DROP ROLE "{role}"')
    except Exception:
        pass


class RecordingConnection:
    """Delegates to a real asyncpg connection and keeps the SQL it is asked
    to run. Used to assert the shape of a statement, not its effect."""

    def __init__(self, conn):
        self._conn = conn
        self.queries = []

    def transaction(self, *a, **kw):
        return self._conn.transaction(*a, **kw)

    def is_in_transaction(self):
        return self._conn.is_in_transaction()

    async def execute(self, query, *args, **kw):
        self.queries.append(" ".join(query.split()))
        return await self._conn.execute(query, *args, **kw)

    async def fetchrow(self, query, *args, **kw):
        self.queries.append(" ".join(query.split()))
        return await self._conn.fetchrow(query, *args, **kw)

    async def fetch(self, query, *args, **kw):
        self.queries.append(" ".join(query.split()))
        return await self._conn.fetch(query, *args, **kw)

    async def fetchval(self, query, *args, **kw):
        self.queries.append(" ".join(query.split()))
        return await self._conn.fetchval(query, *args, **kw)


@pytest.fixture()
def grantee():
    """A unique, unused role name for one privilege test."""
    return f"pgt_role_{uuid.uuid4().hex[:10]}"
