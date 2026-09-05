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
