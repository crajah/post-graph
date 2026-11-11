"""Handing back objects a subordinate role happened to create.

The shape that produces the problem: two login roles, where the writer is
`IN ROLE` the owner and carries a longer `statement_timeout` for bulk work. If
the writer is the first session to reach a realm, it is the writer that creates
the tables -- and everything it makes is owned by it: the tables, their `_data`
and `_audit` companions, every BIGSERIAL sequence and every index.

The owner can then no longer ALTER those tables, and `pg_dump` cannot take
ACCESS SHARE on them, so nightly backups fail with `permission denied for
table documents` on a table the owning role is supposed to own. Re-indexing to
fix it costs hours of extraction; the ownership transfer is a catalogue update
measured in milliseconds.

`REASSIGN OWNED BY CURRENT_USER TO "<role>"` is that update. The first class is
the repair end to end, and the rest guard the ways it could stop being safe.
"""

import uuid

import asyncpg
import pytest
from conftest import DSN, drop_role

from post_graph import AsyncPostGraph


def _dsn_as(role):
    """The test DSN, connecting as *role*. Roles below are created NOLOGIN-free
    with a password so a real connection can be made as them."""
    head, _, tail = DSN.partition("://")
    host_and_db = tail.split("@")[-1]
    return f"{head}://{role}:{role}@{host_and_db}"


@pytest.fixture()
def roles():
    tag = uuid.uuid4().hex[:8]
    return f"owner_{tag}", f"writer_{tag}"


async def _make_roles(admin, owner, writer):
    """An owner, and a writer that inherits it -- neither a superuser."""
    await admin.execute(
        f'CREATE ROLE "{owner}" LOGIN INHERIT PASSWORD \'{owner}\' '
        f"NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE")
    await admin.execute(
        f'CREATE ROLE "{writer}" LOGIN INHERIT PASSWORD \'{writer}\' '
        f'IN ROLE "{owner}" NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE')
    await admin.execute(f'GRANT USAGE ON SCHEMA public TO "{owner}", "{writer}"')
    db = await admin.fetchval("SELECT current_database()")
    # create_vertex_table issues CREATE SCHEMA IF NOT EXISTS whether or not the
    # schema is there, so both roles need CREATE on the database.
    await admin.execute(f'GRANT CREATE, CONNECT ON DATABASE "{db}" TO "{owner}", "{writer}"')
    ext = await admin.fetchval(
        "SELECT n.nspname FROM pg_extension e JOIN pg_namespace n "
        "ON n.oid = e.extnamespace WHERE e.extname = 'vector'")
    if ext:
        await admin.execute(f'GRANT USAGE ON SCHEMA "{ext}" TO "{owner}", "{writer}"')


class TestTheRepair:
    """The incident, reproduced and then fixed."""

    @pytest.mark.asyncio
    async def test_a_writer_hands_its_tables_back_to_the_owner(self, realm, roles):
        owner, writer = roles
        admin = await asyncpg.connect(DSN)
        try:
            await _make_roles(admin, owner, writer)
            await admin.execute(f'CREATE SCHEMA "{realm}" AUTHORIZATION "{owner}"')
            await admin.execute(f'GRANT CREATE, USAGE ON SCHEMA "{realm}" TO "{writer}"')

            w = AsyncPostGraph(dsn=_dsn_as(writer), schema_per_realm=True)
            o = AsyncPostGraph(dsn=_dsn_as(owner), schema_per_realm=True)
            await w.connect()
            await o.connect()
            try:
                who = await w._fetchrow(
                    "SELECT current_user AS u, (SELECT rolsuper FROM pg_roles "
                    "WHERE rolname = current_user) AS s")
                assert who["u"] == writer and who["s"] is False

                # The writer gets there first and creates the realm's tables.
                await w.create_vertex_table("documents", realm=realm, audited=True)
                await w.add_vertex("documents", realm=realm,
                                   payload={"key": "doc-1"}, space="shelf")

                owned_by = await admin.fetchval(
                    "SELECT tableowner FROM pg_tables "
                    "WHERE schemaname = $1 AND tablename = 'documents'", realm)
                assert owned_by == writer, "precondition: the writer owns them"

                # The owner is locked out of its own realm.
                with pytest.raises(asyncpg.InsufficientPrivilegeError):
                    await o.find_vertices("documents", realm=realm, space="shelf")
                with pytest.raises(asyncpg.InsufficientPrivilegeError):
                    await o.create_vertex_table("documents", realm=realm, audited=True)

                # The repair.
                await w.reassign_owned_to(owner)

                # The owner can read what the writer indexed, unchanged...
                rows = await o.find_vertices("documents", realm=realm, space="shelf")
                assert [r.payload["key"] for r in rows] == ["doc-1"]
                # ...and ALTER them again, idempotently.
                await o.create_vertex_table("documents", realm=realm, audited=True)

                # ...and the writer can still write, because it inherits the
                # owner's rights on the tables and the sequences it gave away.
                await w.add_vertex("documents", realm=realm,
                                   payload={"key": "doc-2"}, space="shelf")
                rows = await o.find_vertices("documents", realm=realm, space="shelf")
                assert sorted(r.payload["key"] for r in rows) == ["doc-1", "doc-2"]
            finally:
                await w.close()
                await o.close()
        finally:
            await admin.execute(f'DROP SCHEMA IF EXISTS "{realm}" CASCADE')
            for r in (writer, owner):
                await drop_role(admin, r)
            await admin.close()

    @pytest.mark.asyncio
    async def test_the_companions_and_sequences_move_too(self, realm, roles):
        """The reason this is one statement and not an enumeration.

        A vertex table brings `_data` and `_audit` companions and a BIGSERIAL
        sequence per table. Missing any one of them leaves the owner unable to
        ALTER the set, or the writer unable to insert.
        """
        owner, writer = roles
        admin = await asyncpg.connect(DSN)
        try:
            await _make_roles(admin, owner, writer)
            await admin.execute(f'CREATE SCHEMA "{realm}" AUTHORIZATION "{owner}"')
            await admin.execute(f'GRANT CREATE, USAGE ON SCHEMA "{realm}" TO "{writer}"')

            w = AsyncPostGraph(dsn=_dsn_as(writer), schema_per_realm=True)
            await w.connect()
            try:
                await w.create_vertex_table("documents", realm=realm, audited=True)
                await w.reassign_owned_to(owner)
            finally:
                await w.close()

            tables = {r["tablename"]: r["tableowner"] for r in await admin.fetch(
                "SELECT tablename, tableowner FROM pg_tables WHERE schemaname = $1",
                realm)}
            assert tables, "the writer created nothing"
            assert set(tables) >= {"documents", "documents_data", "documents_audit"}
            assert set(tables.values()) == {owner}, tables

            seqs = await admin.fetch(
                "SELECT c.relname, pg_get_userbyid(c.relowner) AS owner "
                "FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = $1 AND c.relkind = 'S'", realm)
            assert seqs, "no sequences were created"
            assert {r["owner"] for r in seqs} == {owner}

            idx = await admin.fetch(
                "SELECT c.relname, pg_get_userbyid(c.relowner) AS owner "
                "FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = $1 AND c.relkind = 'i'", realm)
            assert {r["owner"] for r in idx} == {owner}
        finally:
            await admin.execute(f'DROP SCHEMA IF EXISTS "{realm}" CASCADE')
            for r in (writer, owner):
                await drop_role(admin, r)
            await admin.close()


class TestItIsSafeToCallAlways:

    @pytest.mark.asyncio
    async def test_calling_it_twice_is_a_no_op(self, realm, roles):
        owner, writer = roles
        admin = await asyncpg.connect(DSN)
        try:
            await _make_roles(admin, owner, writer)
            await admin.execute(f'CREATE SCHEMA "{realm}" AUTHORIZATION "{owner}"')
            await admin.execute(f'GRANT CREATE, USAGE ON SCHEMA "{realm}" TO "{writer}"')
            w = AsyncPostGraph(dsn=_dsn_as(writer), schema_per_realm=True)
            o = AsyncPostGraph(dsn=_dsn_as(owner), schema_per_realm=True)
            await w.connect()
            await o.connect()
            try:
                await w.create_vertex_table("documents", realm=realm)
                assert await w.reassign_owned_to(owner) is None
                assert await w.reassign_owned_to(owner) is None   # nothing left
                assert await o.reassign_owned_to(owner) is None   # to itself
            finally:
                await w.close()
                await o.close()
        finally:
            await admin.execute(f'DROP SCHEMA IF EXISTS "{realm}" CASCADE')
            for r in (writer, owner):
                await drop_role(admin, r)
            await admin.close()

    @pytest.mark.asyncio
    async def test_a_role_owning_nothing(self, roles):
        """Safe to call unconditionally at start-up, which is the point.

        Not run as the database owner: that role owns objects the system
        requires, and PostgreSQL refuses to reassign those -- which is its own
        answer to "could this method be pointed somewhere dangerous".
        """
        owner, writer = roles
        admin = await asyncpg.connect(DSN)
        try:
            await _make_roles(admin, owner, writer)
            w = AsyncPostGraph(dsn=_dsn_as(writer))
            await w.connect()
            try:
                assert await w.reassign_owned_to(owner) is None
            finally:
                await w.close()
        finally:
            for r in (writer, owner):
                await drop_role(admin, r)
            await admin.close()


class TestWhatItRefuses:

    @pytest.mark.parametrize("hostile", [
        'owner"; DROP SCHEMA reference CASCADE; --',
        "owner role",
        "9lives",
        "",
        None,
    ])
    @pytest.mark.asyncio
    async def test_a_hostile_identifier(self, pg_client, hostile):
        """REASSIGN OWNED takes no bound parameter, so validating the
        identifier before quoting it is the only defence there is."""
        with pytest.raises(ValueError):
            await pg_client.reassign_owned_to(hostile)

    @pytest.mark.asyncio
    async def test_it_refuses_before_touching_the_connection(self, realm):
        """The refusal must not depend on reaching the database."""
        client = AsyncPostGraph(dsn=DSN)          # never connected
        with pytest.raises(ValueError):
            await client.reassign_owned_to('x"; DROP SCHEMA y; --')

    @pytest.mark.asyncio
    async def test_the_source_role_cannot_be_chosen(self):
        """Fixed to CURRENT_USER, so this can only ever give away.

        A source parameter would make it a method for pulling objects away
        from another role, which PostgreSQL would permit to any session
        holding privileges of both.
        """
        import inspect
        src = inspect.getsource(AsyncPostGraph.reassign_owned_to)
        body = src.split('"""')[2]
        assert "REASSIGN OWNED BY CURRENT_USER TO" in body
        assert list(inspect.signature(
            AsyncPostGraph.reassign_owned_to).parameters) == ["self", "role"]


class TestItDoesNotSwallowAFailure:
    """The hazard in routing this through the creation-tolerant DDL path."""

    def test_a_concurrent_catalogue_update_is_not_a_benign_race(self):
        """`tuple concurrently updated` means "done already" for CREATE OR
        REPLACE FUNCTION, and the opposite for a reassignment: the catalogue
        row did not move. Swallowing it would report success over objects
        whose ownership never changed -- backups still failing, nothing to
        say why. It is retryable either way, which is what we want instead.
        """
        from post_graph._ddl import is_concurrent_creation, is_retryable_ddl
        e = Exception("tuple concurrently updated")
        assert is_concurrent_creation(e), "the classifier this must not use"
        assert is_retryable_ddl(e), "the classifier this relies on"

    @pytest.mark.asyncio
    async def test_the_method_asks_for_no_tolerance(self):
        import inspect
        assert "tolerate_existing=False" in inspect.getsource(
            AsyncPostGraph.reassign_owned_to)

    @pytest.mark.asyncio
    async def test_a_real_failure_still_raises(self, pg_client):
        """A role that does not exist must fail loudly, not be swallowed."""
        with pytest.raises(asyncpg.PostgresError):
            await pg_client.reassign_owned_to("no_such_role_" + uuid.uuid4().hex[:8])

    @pytest.mark.asyncio
    async def test_creation_still_tolerates_a_race(self, pg_client, clean_realm):
        """The default is unchanged: _execute_ddl still swallows for creations."""
        await pg_client.create_vertex_table("things", realm=clean_realm)
        await pg_client.create_vertex_table("things", realm=clean_realm)


class TestSQLAlchemyParity:

    @pytest.mark.asyncio
    async def test_it_hands_tables_back(self, realm, roles):
        from sqlalchemy.ext.asyncio import create_async_engine

        from post_graph.client_sqlalchemy import SQLAlchemyPostGraph

        owner, writer = roles
        admin = await asyncpg.connect(DSN)
        engine = None
        try:
            await _make_roles(admin, owner, writer)
            await admin.execute(f'CREATE SCHEMA "{realm}" AUTHORIZATION "{owner}"')
            await admin.execute(f'GRANT CREATE, USAGE ON SCHEMA "{realm}" TO "{writer}"')

            engine = create_async_engine(
                _dsn_as(writer).replace("postgresql://", "postgresql+asyncpg://", 1))
            w = SQLAlchemyPostGraph(engine, schema_per_realm=True)
            await w.create_vertex_table("documents", realm=realm)
            assert await admin.fetchval(
                "SELECT tableowner FROM pg_tables WHERE schemaname = $1 "
                "AND tablename = 'documents'", realm) == writer

            await w.reassign_owned_to(owner)
            assert await admin.fetchval(
                "SELECT tableowner FROM pg_tables WHERE schemaname = $1 "
                "AND tablename = 'documents'", realm) == owner
        finally:
            if engine is not None:
                await engine.dispose()
            await admin.execute(f'DROP SCHEMA IF EXISTS "{realm}" CASCADE')
            for r in (writer, owner):
                await drop_role(admin, r)
            await admin.close()

    @pytest.mark.asyncio
    async def test_a_hostile_identifier(self, sa_client):
        with pytest.raises(ValueError):
            await sa_client.reassign_owned_to('x"; DROP SCHEMA y; --')
