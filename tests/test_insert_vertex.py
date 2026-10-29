"""An append an ``INSERT``-only role can run.

An append-only ledger's containment is its grant: the writer holds ``INSERT``
and nothing else, so no code path it reaches can amend or remove what it has
recorded. Its idempotency is then the primary key -- the row id is derived from
the event, and a redelivered message must collide with the row already there
and change nothing.

Neither existing write can do that. ``upsert_vertex`` needs ``UPDATE`` for its
``ON CONFLICT DO UPDATE``, which is the privilege the ledger exists not to
grant. ``add_vertex`` needs ``SELECT`` -- not for a read the caller asked for,
but for its own ``RETURNING`` clause, which PostgreSQL privilege-checks column
by column, and for the ``setval`` that resynchronises the sequence from
``MAX(id)``.

So ``insert_vertex`` emits one statement, ``INSERT ... ON CONFLICT DO
NOTHING``, with no ``RETURNING``, no conflict target and no ``setval``, and
reads the answer out of the command tag. The first class below is the point of
the method; the rest guard the ways it could quietly stop being that statement.
"""

import uuid

import asyncpg
import pytest
from conftest import DSN, RecordingConnection, drop_role, grant_role

from post_graph import RESERVED_SPACE_ALL, AsyncPostGraph
from post_graph.errors import PostGraphError, ReservedSpaceError, TableNotFoundError


class TestThePrivilegeProperty:
    """The reason the method exists."""

    @pytest.mark.asyncio
    async def test_an_insert_only_role_can_append_where_the_others_cannot(
        self, pg_client, clean_realm, grantee
    ):
        """One role holding ``INSERT`` and nothing else; three statements.

        The table is unaudited so the only privilege in play is the one under
        test. No grant on the sequence either: a supplied id never evaluates
        the column default, so the writer of an id-keyed ledger does not need
        the sequence at all.
        """
        realm = clean_realm
        await pg_client.create_vertex_table("usage_events", realm=realm, audited=False)

        owner = await asyncpg.connect(DSN)
        try:
            await grant_role(owner, grantee, [
                f'GRANT INSERT ON "usage_events" TO "{grantee}"',
            ])

            conn = await asyncpg.connect(DSN)
            try:
                client = AsyncPostGraph(connection_or_pool=conn)
                event_id = 20261029001

                # add_vertex: refused, and not for the insert. Its RETURNING
                # clause needs SELECT on every column it names, and its setval
                # reads MAX(id) from the table.
                tx = conn.transaction()
                await tx.start()
                await conn.execute(f'SET LOCAL ROLE "{grantee}"')
                with pytest.raises(asyncpg.InsufficientPrivilegeError):
                    await client.add_vertex("usage_events", realm=realm,
                                            vertex_id=event_id, payload={"n": 1})
                await tx.rollback()

                # upsert_vertex: refused for UPDATE, which is the privilege an
                # append-only ledger is defined by not holding.
                tx = conn.transaction()
                await tx.start()
                await conn.execute(f'SET LOCAL ROLE "{grantee}"')
                with pytest.raises(asyncpg.InsufficientPrivilegeError):
                    await client.upsert_vertex("usage_events", realm=realm,
                                               vertex_id=event_id, payload={"n": 1})
                await tx.rollback()

                # insert_vertex: written, on INSERT alone.
                tx = conn.transaction()
                await tx.start()
                await conn.execute(f'SET LOCAL ROLE "{grantee}"')
                written = await client.insert_vertex(
                    "usage_events", realm=realm, vertex_id=event_id,
                    payload={"case_id": "c-1", "n": 1}, space="metering")
                await tx.commit()
                assert written is True

                # The redelivery: same id, absorbed, and it reports so.
                tx = conn.transaction()
                await tx.start()
                await conn.execute(f'SET LOCAL ROLE "{grantee}"')
                again = await client.insert_vertex(
                    "usage_events", realm=realm, vertex_id=event_id,
                    payload={"case_id": "c-1", "n": 999}, space="metering")
                await tx.commit()
                assert again is False
            finally:
                await conn.close()

            # And the ledger holds one row, with what was first recorded.
            rows = await owner.fetch(
                'SELECT id, space, payload FROM "usage_events" WHERE realm = $1', realm)
            assert len(rows) == 1
            assert rows[0]["id"] == 20261029001
            assert rows[0]["space"] == "metering"
            assert '"n": 1' in rows[0]["payload"]
        finally:
            await drop_role(owner, grantee)
            await owner.close()

    @pytest.mark.asyncio
    async def test_naming_the_conflict_target_is_what_the_untargeted_form_avoids(
        self, pg_client, clean_realm, grantee
    ):
        """Why the statement has no ``(realm, id)`` after ``ON CONFLICT``.

        The targeted form reads the index to identify the arbiter and is
        refused to a role without ``SELECT``; the untargeted form is not. This
        is measured rather than asserted because it is the one piece of the
        statement's shape that looks like it could be tightened, and
        tightening it would break the method for its only caller.
        """
        realm = clean_realm
        await pg_client.create_vertex_table("usage_events", realm=realm, audited=False)

        owner = await asyncpg.connect(DSN)
        try:
            await grant_role(owner, grantee, [
                f'GRANT INSERT ON "usage_events" TO "{grantee}"',
            ])
            conn = await asyncpg.connect(DSN)
            try:
                base = ('INSERT INTO "usage_events" (realm, id, space, payload) '
                        "VALUES ($1, $2, 'default', '{}'::jsonb) ")

                tx = conn.transaction()
                await tx.start()
                await conn.execute(f'SET LOCAL ROLE "{grantee}"')
                with pytest.raises(asyncpg.InsufficientPrivilegeError):
                    await conn.execute(base + "ON CONFLICT (realm, id) DO NOTHING",
                                       realm, 1)
                await tx.rollback()

                tx = conn.transaction()
                await tx.start()
                await conn.execute(f'SET LOCAL ROLE "{grantee}"')
                status = await conn.execute(base + "ON CONFLICT DO NOTHING", realm, 1)
                await tx.commit()
                assert status == "INSERT 0 1"
            finally:
                await conn.close()
        finally:
            await drop_role(owner, grantee)
            await owner.close()

    @pytest.mark.asyncio
    async def test_an_audited_table_also_needs_the_audit_insert_grant(
        self, pg_client, clean_realm, grantee
    ):
        """The adjacent trap, for a caller who audits its ledger.

        The audit trigger is ``SECURITY INVOKER``, so its ``INSERT INTO
        <table>_audit`` runs as the appending role, not as the table's owner.
        The grant on the table alone is therefore not enough, and the failure
        arrives as a permission error on a table the caller never named.
        """
        realm = clean_realm
        await pg_client.create_vertex_table("usage_events", realm=realm, audited=True)

        owner = await asyncpg.connect(DSN)
        try:
            await grant_role(owner, grantee, [
                f'GRANT INSERT ON "usage_events" TO "{grantee}"',
            ])
            conn = await asyncpg.connect(DSN)
            try:
                client = AsyncPostGraph(connection_or_pool=conn)

                tx = conn.transaction()
                await tx.start()
                await conn.execute(f'SET LOCAL ROLE "{grantee}"')
                with pytest.raises(asyncpg.InsufficientPrivilegeError) as exc:
                    await client.insert_vertex("usage_events", realm=realm,
                                               vertex_id=1, payload={}, space="s")
                await tx.rollback()
                assert "usage_events_audit" in str(exc.value)

                # The second grant, which the docstring names.
                await owner.execute(
                    f'GRANT INSERT ON "usage_events_audit" TO "{grantee}"')
                await owner.execute(
                    f'GRANT USAGE ON SEQUENCE "usage_events_audit_audit_id_seq" '
                    f'TO "{grantee}"')

                tx = conn.transaction()
                await tx.start()
                await conn.execute(f'SET LOCAL ROLE "{grantee}"')
                assert await client.insert_vertex(
                    "usage_events", realm=realm, vertex_id=1, payload={}, space="s")
                await tx.commit()
            finally:
                await conn.close()
        finally:
            await drop_role(owner, grantee)
            await owner.close()


class TestTheStatementShape:
    """What the method emits, not what it achieves."""

    @pytest.mark.asyncio
    async def test_one_statement_no_returning_no_target_no_setval(
        self, pg_client, clean_realm
    ):
        realm = clean_realm
        await pg_client.create_vertex_table("usage_events", realm=realm, audited=False)

        conn = await asyncpg.connect(DSN)
        try:
            spy = RecordingConnection(conn)
            client = AsyncPostGraph(connection_or_pool=spy)
            assert await client.insert_vertex("usage_events", realm=realm,
                                              vertex_id=7, payload={"a": 1},
                                              space="metering")
        finally:
            await conn.close()

        touching = [q for q in spy.queries if "usage_events" in q]
        assert len(touching) == 1, f"expected one statement, got {touching}"
        sql = touching[0].upper()
        assert "RETURNING" not in sql, "a RETURNING clause needs SELECT on its columns"
        assert "ON CONFLICT DO NOTHING" in sql
        assert "ON CONFLICT (" not in sql, "a named arbiter needs SELECT on the index"
        assert "SETVAL" not in sql, "setval reads MAX(id) from the table"
        assert "SELECT" not in sql, "no read of any kind"


class TestTheReturnValue:
    """True wrote, False was already there -- and False is not a merge."""

    @pytest.mark.asyncio
    async def test_the_absorbed_duplicate_leaves_the_stored_row_alone(
        self, pg_client, clean_realm
    ):
        realm = clean_realm
        await pg_client.create_vertex_table("usage_events", realm=realm)
        assert await pg_client.insert_vertex("usage_events", realm=realm, vertex_id=3,
                                             payload={"n": 1}, space="metering") is True
        assert await pg_client.insert_vertex("usage_events", realm=realm, vertex_id=3,
                                             payload={"n": 2, "extra": True},
                                             space="other") is False

        v = await pg_client.get_vertex("usage_events", realm, "3")
        assert v.payload == {"n": 1}, "DO NOTHING, not a merge"
        assert v.space == "metering", "and not a move between spaces"

    @pytest.mark.asyncio
    async def test_a_second_unique_constraint_also_reports_false(
        self, pg_client, clean_realm
    ):
        """The documented caveat of an untargeted conflict clause.

        ``ON CONFLICT DO NOTHING`` without an arbiter absorbs *any* unique
        violation on the row, not only one on the primary key. A table with a
        second unique constraint therefore reports False for a collision on
        that one too -- which is right for a ledger whose idempotency key is
        that constraint, and surprising for anyone expecting False to mean
        "this id exists". The docstring says so because of this test.
        """
        realm = clean_realm
        await pg_client.create_vertex_table("usage_events", realm=realm)
        idx = f"uq_{uuid.uuid4().hex[:10]}"
        await pg_client._execute(
            f'CREATE UNIQUE INDEX "{idx}" ON "usage_events" '
            f"(realm, (payload->>'call_id'))")
        try:
            assert await pg_client.insert_vertex(
                "usage_events", realm=realm, vertex_id=1,
                payload={"call_id": "k"}, space="metering") is True
            # A different id, the same call_id.
            assert await pg_client.insert_vertex(
                "usage_events", realm=realm, vertex_id=2,
                payload={"call_id": "k"}, space="metering") is False
        finally:
            await pg_client._execute(f'DROP INDEX IF EXISTS "{idx}"')


class TestArgumentHandling:

    @pytest.mark.asyncio
    async def test_space_is_keyword_only_and_required(self, pg_client, clean_realm):
        """No default, deliberately: a silently defaulted space on a ledger
        partitioned by tenant is rows written where nobody will look."""
        await pg_client.create_vertex_table("usage_events", realm=clean_realm)
        with pytest.raises(TypeError):
            await pg_client.insert_vertex("usage_events", realm=clean_realm,
                                          vertex_id=1, payload={})
        with pytest.raises(TypeError):
            await pg_client.insert_vertex("usage_events", clean_realm, 1, {}, "metering")

    @pytest.mark.asyncio
    async def test_the_reserved_space_is_refused(self, pg_client, clean_realm):
        await pg_client.create_vertex_table("usage_events", realm=clean_realm)
        with pytest.raises(ReservedSpaceError):
            await pg_client.insert_vertex("usage_events", realm=clean_realm,
                                          vertex_id=1, payload={},
                                          space=RESERVED_SPACE_ALL)

    @pytest.mark.asyncio
    async def test_a_missing_id_is_refused_rather_than_assigned(
        self, pg_client, clean_realm
    ):
        await pg_client.create_vertex_table("usage_events", realm=clean_realm)
        with pytest.raises(PostGraphError):
            await pg_client.insert_vertex("usage_events", realm=clean_realm,
                                          vertex_id=None, payload={}, space="s")

    @pytest.mark.asyncio
    async def test_an_fqid_shaped_id_is_normalised(self, pg_client, clean_realm):
        await pg_client.create_vertex_table("usage_events", realm=clean_realm)
        assert await pg_client.insert_vertex("usage_events", realm=clean_realm,
                                             vertex_id="usage_events/42",
                                             payload={}, space="s") is True
        assert await pg_client.get_vertex("usage_events", clean_realm, "42") is not None

    @pytest.mark.asyncio
    async def test_a_missing_table_raises_the_librarys_error(self, pg_client, realm):
        with pytest.raises(TableNotFoundError):
            await pg_client.insert_vertex("no_such_" + uuid.uuid4().hex[:8],
                                          realm=realm, vertex_id=1, payload={},
                                          space="s")

    @pytest.mark.asyncio
    async def test_an_empty_payload_is_an_empty_object(self, pg_client, clean_realm):
        await pg_client.create_vertex_table("usage_events", realm=clean_realm)
        assert await pg_client.insert_vertex("usage_events", realm=clean_realm,
                                             vertex_id=1, payload=None, space="s")
        v = await pg_client.get_vertex("usage_events", clean_realm, "1")
        assert v.payload == {}


class TestInsideACallersTransaction:
    """A connection-bound client keeps the caller's transaction, and its role."""

    @pytest.mark.asyncio
    async def test_the_append_rolls_back_with_the_callers_transaction(
        self, pg_client, clean_realm
    ):
        realm = clean_realm
        await pg_client.create_vertex_table("usage_events", realm=realm)

        conn = await asyncpg.connect(DSN)
        try:
            client = AsyncPostGraph(connection_or_pool=conn)
            tx = conn.transaction()
            await tx.start()
            assert await client.insert_vertex("usage_events", realm=realm,
                                              vertex_id=5, payload={"n": 1},
                                              space="s") is True
            # Visible inside the caller's transaction...
            assert await conn.fetchval(
                'SELECT count(*) FROM "usage_events" WHERE realm = $1 AND id = 5',
                realm) == 1
            await tx.rollback()
        finally:
            await conn.close()

        # ...and gone with it.
        assert await pg_client.get_vertex("usage_events", realm, "5") is None


class TestSchemaPerRealm:

    @pytest.mark.asyncio
    async def test_it_writes_into_the_realms_schema(self, pg_client_spr,
                                                    clean_realm_spr):
        realm = clean_realm_spr
        await pg_client_spr.create_vertex_table("usage_events", realm=realm)
        assert await pg_client_spr.insert_vertex("usage_events", realm=realm,
                                                 vertex_id=1, payload={"n": 1},
                                                 space="s") is True
        assert await pg_client_spr.insert_vertex("usage_events", realm=realm,
                                                 vertex_id=1, payload={"n": 2},
                                                 space="s") is False
        v = await pg_client_spr.get_vertex("usage_events", realm, "1")
        assert v.payload == {"n": 1}


class TestSQLAlchemyParity:

    @pytest.mark.asyncio
    async def test_true_then_false(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("usage_events", realm=realm)
        assert await sa_client.insert_vertex("usage_events", realm=realm,
                                             vertex_id=11, payload={"n": 1},
                                             space="metering") is True
        assert await sa_client.insert_vertex("usage_events", realm=realm,
                                             vertex_id=11, payload={"n": 2},
                                             space="metering") is False
        v = await sa_client.get_vertex("usage_events", realm, "11")
        assert v.payload == {"n": 1}
        assert v.space == "metering"

    @pytest.mark.asyncio
    async def test_space_is_keyword_only_there_too(self, sa_client, sa_clean_realm):
        await sa_client.create_vertex_table("usage_events", realm=sa_clean_realm)
        with pytest.raises(TypeError):
            await sa_client.insert_vertex("usage_events", realm=sa_clean_realm,
                                          vertex_id=1, payload={})
