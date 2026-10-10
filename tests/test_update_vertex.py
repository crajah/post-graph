"""An update-only vertex write, so a role granted UPDATE can perform one.

`upsert_vertex` emits `INSERT ... ON CONFLICT DO UPDATE`, and PostgreSQL
checks the INSERT privilege on that statement unconditionally -- not only when
the insert path is taken, because the check is on the statement and not on the
branch the data happens to take. So a role granted `SELECT, UPDATE` cannot
change a row through post-graph although it can change the same row through
SQL.

That is not a style complaint. Where a caller may spend a single-use token but
must not be able to mint one, the grant *is* the containment: `UPDATE` without
`INSERT` is the whole of the boundary, and an upsert asks for it to be given
up. `update_vertex` emits a bare UPDATE, so `UPDATE` is all it needs.

The first test in TestThePrivilegeProperty is the point of the feature. The
rest guard the ways a bare UPDATE could quietly turn back into an upsert.
"""

import json
import uuid

import asyncpg
import pytest
from conftest import DSN

from post_graph import AsyncPostGraph
from post_graph.errors import ReservedSpaceError


async def _grant_role(conn, role, grants):
    """Create a NOLOGIN role and apply *grants*, a list of SQL fragments."""
    await conn.execute(f'CREATE ROLE "{role}" NOLOGIN')
    await conn.execute(f'GRANT USAGE ON SCHEMA public TO "{role}"')
    for g in grants:
        await conn.execute(g)


async def _drop_role(conn, role):
    try:
        await conn.execute(f'REASSIGN OWNED BY "{role}" TO CURRENT_USER')
        await conn.execute(f'DROP OWNED BY "{role}"')
        await conn.execute(f'DROP ROLE "{role}"')
    except Exception:
        pass


class _RecordingConnection:
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
def role():
    return f"upd_role_{uuid.uuid4().hex[:10]}"


class TestThePrivilegeProperty:
    """The reason the method exists."""

    @pytest.mark.asyncio
    async def test_a_select_update_role_updates_where_upsert_is_denied(
        self, pg_client, clean_realm, role
    ):
        """The whole of the request: one role, one row, two statements.

        The table is unaudited so that the only privilege in play is the one
        under test; the audit trigger's own INSERT is the next test.
        """
        realm = clean_realm
        await pg_client.create_vertex_table("invitations", realm=realm, audited=False)
        v = await pg_client.add_vertex(
            "invitations", realm=realm, payload={"code": "abc", "redeemed_at": None}
        )

        owner = await asyncpg.connect(DSN)
        try:
            await _grant_role(owner, role, [
                f'GRANT SELECT, UPDATE ON "invitations" TO "{role}"',
            ])

            conn = await asyncpg.connect(DSN)
            try:
                client = AsyncPostGraph(connection_or_pool=conn)

                # The grant is correct, the row exists, and the upsert's
                # statement is the wrong shape for the privilege.
                tx = conn.transaction()
                await tx.start()
                await conn.execute(f'SET LOCAL ROLE "{role}"')
                with pytest.raises(asyncpg.InsufficientPrivilegeError):
                    await client.upsert_vertex(
                        "invitations", realm=realm, vertex_id=v.id,
                        payload={"redeemed_at": "2026-10-10T09:00:00Z"},
                    )
                await tx.rollback()

                # The same change, through a statement the grant permits.
                tx = conn.transaction()
                await tx.start()
                await conn.execute(f'SET LOCAL ROLE "{role}"')
                updated = await client.update_vertex(
                    "invitations", realm=realm, vertex_id=v.id,
                    payload={"redeemed_at": "2026-10-10T09:00:00Z"},
                )
                await tx.commit()

                assert updated is not None
                assert updated.payload["redeemed_at"] == "2026-10-10T09:00:00Z"
                assert updated.payload["code"] == "abc", "merge, not replace"
            finally:
                await conn.close()
        finally:
            await _drop_role(owner, role)
            await owner.close()

    @pytest.mark.asyncio
    async def test_an_audited_table_also_needs_the_audit_grants(
        self, pg_client, clean_realm, role
    ):
        """A trap that is not update_vertex's doing, and would block a caller
        who read only the headline.

        post-graph puts an AFTER UPDATE trigger on an audited table, and the
        trigger runs `INSERT INTO <table>_audit` as the *invoking* role. So on
        the default audited table a SELECT,UPDATE role still fails -- with
        `permission denied for table invitations_audit`, naming a table that
        does not appear in the grant at all.

        The fix is two write-only grants, and this test asserts the thing that
        makes them acceptable: they do not widen the role's reach into the
        audited table, so a caller that may spend an invitation still cannot
        mint one.
        """
        realm = clean_realm
        await pg_client.create_vertex_table("invitations", realm=realm)  # audited
        v = await pg_client.add_vertex("invitations", realm=realm, payload={"code": "abc"})

        owner = await asyncpg.connect(DSN)
        try:
            await _grant_role(owner, role, [
                f'GRANT SELECT, UPDATE ON "invitations" TO "{role}"',
            ])

            conn = await asyncpg.connect(DSN)
            try:
                client = AsyncPostGraph(connection_or_pool=conn)

                tx = conn.transaction()
                await tx.start()
                await conn.execute(f'SET LOCAL ROLE "{role}"')
                with pytest.raises(asyncpg.InsufficientPrivilegeError) as exc:
                    await client.update_vertex(
                        "invitations", realm=realm, vertex_id=v.id, payload={"spent": True}
                    )
                await tx.rollback()
                assert "invitations_audit" in str(exc.value), (
                    "the denial names the audit shadow, not the granted table"
                )

                await owner.execute(f'GRANT INSERT ON "invitations_audit" TO "{role}"')
                await owner.execute(
                    f'GRANT USAGE ON SEQUENCE "invitations_audit_audit_id_seq" TO "{role}"'
                )

                tx = conn.transaction()
                await tx.start()
                await conn.execute(f'SET LOCAL ROLE "{role}"')
                updated = await client.update_vertex(
                    "invitations", realm=realm, vertex_id=v.id, payload={"spent": True}
                )
                assert updated is not None and updated.payload["spent"] is True

                # Containment survives the two grants: still no INSERT on the
                # audited table, and still no read of the audit trail.
                with pytest.raises(asyncpg.InsufficientPrivilegeError):
                    await conn.execute(
                        'INSERT INTO "invitations" (realm, id, space, payload) '
                        "VALUES ($1, 9999, 'default', '{}'::jsonb)", realm
                    )
                await tx.rollback()

                tx = conn.transaction()
                await tx.start()
                await conn.execute(f'SET LOCAL ROLE "{role}"')
                with pytest.raises(asyncpg.InsufficientPrivilegeError):
                    await conn.fetch('SELECT * FROM "invitations_audit"')
                await tx.rollback()
            finally:
                await conn.close()
        finally:
            await _drop_role(owner, role)
            await owner.close()

    @pytest.mark.asyncio
    async def test_the_statement_it_emits_contains_no_insert(
        self, pg_client, clean_realm
    ):
        """Read the SQL off the wire rather than trusting the source.

        The privilege test above is the real proof -- a statement containing an
        INSERT could not have run under that grant -- but it needs a role, and
        a role needs privileges to create. This one needs nothing, so it still
        runs where the suite is pointed at a database the test user does not
        own, and it names the failure precisely if someone reintroduces an
        upsert here.
        """
        realm = clean_realm
        await pg_client.create_vertex_table("t", realm=realm, audited=False)
        v = await pg_client.add_vertex("t", realm=realm, payload={"a": 1})

        conn = await asyncpg.connect(DSN)
        try:
            spy = _RecordingConnection(conn)
            client = AsyncPostGraph(connection_or_pool=spy)
            await client.update_vertex("t", realm=realm, vertex_id=v.id, payload={"b": 2})
        finally:
            await conn.close()

        writes = [q for q in spy.queries if "t" in q and "set_config" not in q]
        assert writes, "no statement was recorded"
        sql = " ".join(writes).upper()
        assert "INSERT" not in sql, f"update_vertex emitted an INSERT: {writes}"
        assert "ON CONFLICT" not in sql
        assert "UPDATE" in sql and "WHERE REALM = $1 AND ID = $2" in sql


class TestItNeverInserts:

    @pytest.mark.asyncio
    async def test_an_unknown_id_returns_none_and_changes_nothing(
        self, pg_client, clean_realm
    ):
        """A method that quietly created the row would be upsert_vertex with a
        different name, and the caller would be writing an invitation rather
        than spending one."""
        realm = clean_realm
        await pg_client.create_vertex_table("invitations", realm=realm, audited=False)
        await pg_client.add_vertex("invitations", realm=realm, payload={"a": 1})

        before = await pg_client._fetchrow(
            'SELECT count(*) AS n FROM "invitations" WHERE realm = $1', realm
        )
        result = await pg_client.update_vertex(
            "invitations", realm=realm, vertex_id=10 ** 9, payload={"b": 2}
        )
        after = await pg_client._fetchrow(
            'SELECT count(*) AS n FROM "invitations" WHERE realm = $1', realm
        )

        assert result is None
        assert before["n"] == after["n"] == 1

    @pytest.mark.asyncio
    async def test_an_unknown_realm_returns_none(self, pg_client, clean_realm):
        """The realm is a column, not a namespace, in the default mode: an id
        that exists under another realm must not be reachable."""
        realm = clean_realm
        await pg_client.create_vertex_table("invitations", realm=realm, audited=False)
        v = await pg_client.add_vertex("invitations", realm=realm, payload={"a": 1})

        assert await pg_client.update_vertex(
            "invitations", realm=f"{realm}_other", vertex_id=v.id, payload={"b": 2}
        ) is None
        kept = await pg_client.get_vertex("invitations", realm=realm, vertex_id=v.id)
        assert "b" not in kept.payload


class TestMergeParityWithUpsert:
    """A caller moving from upsert_vertex to update_vertex must get the same
    row, so the comparisons here run both and diff the payloads."""

    @pytest.mark.asyncio
    async def test_merge_keeps_unmentioned_keys(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("t", realm=realm, audited=False)
        v = await pg_client.add_vertex("t", realm=realm, payload={"keep": 1, "change": "old"})
        out = await pg_client.update_vertex(
            "t", realm=realm, vertex_id=v.id, payload={"change": "new", "add": True}
        )
        assert out.payload == {"keep": 1, "change": "new", "add": True}

    @pytest.mark.asyncio
    async def test_replace_overwrites_and_so_removes_keys(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("t", realm=realm, audited=False)
        v = await pg_client.add_vertex("t", realm=realm, payload={"gone": 1, "stays": 2})
        out = await pg_client.update_vertex(
            "t", realm=realm, vertex_id=v.id, payload={"stays": 2}, replace=True
        )
        assert out.payload == {"stays": 2}

    @pytest.mark.asyncio
    async def test_json_null_behaves_as_it_does_under_upsert(self, pg_client, clean_realm):
        """JSON null is a value, not an absence: `||` overwrites with it. The
        same must be true of both writes, or a caller nulling a field gets a
        different row depending on which it called."""
        realm = clean_realm
        await pg_client.create_vertex_table("t", realm=realm, audited=False)
        a = await pg_client.add_vertex("t", realm=realm, payload={"x": 1, "y": 2})
        b = await pg_client.add_vertex("t", realm=realm, payload={"x": 1, "y": 2})

        via_upsert = await pg_client.upsert_vertex(
            "t", realm=realm, vertex_id=a.id, payload={"y": None}
        )
        via_update = await pg_client.update_vertex(
            "t", realm=realm, vertex_id=b.id, payload={"y": None}
        )
        assert via_update.payload == via_upsert.payload == {"x": 1, "y": None}

    @pytest.mark.asyncio
    async def test_an_empty_payload_is_a_no_op_on_the_payload(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("t", realm=realm, audited=False)
        v = await pg_client.add_vertex("t", realm=realm, payload={"x": 1})
        out = await pg_client.update_vertex("t", realm=realm, vertex_id=v.id)
        assert out.payload == {"x": 1}

    @pytest.mark.asyncio
    async def test_updated_at_moves_and_created_at_does_not(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("t", realm=realm, audited=False)
        v = await pg_client.add_vertex("t", realm=realm, payload={"x": 1})
        out = await pg_client.update_vertex("t", realm=realm, vertex_id=v.id, payload={"x": 2})
        assert out.created_at == v.created_at
        assert out.updated_at >= v.updated_at

    @pytest.mark.asyncio
    async def test_the_identity_columns_are_untouched(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("t", realm=realm, audited=False)
        v = await pg_client.add_vertex("t", realm=realm, payload={"x": 1})
        out = await pg_client.update_vertex("t", realm=realm, vertex_id=v.id, payload={"x": 2})
        assert (out.id, out.fqid, out.uuid) == (v.id, v.fqid, v.uuid)


class TestSpaceHandling:

    @pytest.mark.asyncio
    async def test_omitting_space_leaves_the_row_where_it_is(self, pg_client, clean_realm):
        """upsert_vertex defaults space to "default" and always writes it,
        because it may be creating the row. Inheriting that here would
        relocate a tenant's row whenever the caller did not mention a space,
        which is a surprising way to lose data on an update."""
        realm = clean_realm
        await pg_client.create_vertex_table("t", realm=realm, audited=False)
        v = await pg_client.add_vertex("t", realm=realm, payload={"x": 1}, space="firm_a")
        out = await pg_client.update_vertex("t", realm=realm, vertex_id=v.id, payload={"x": 2})
        assert out.space == "firm_a"

    @pytest.mark.asyncio
    async def test_passing_space_moves_the_row(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("t", realm=realm, audited=False)
        v = await pg_client.add_vertex("t", realm=realm, payload={"x": 1}, space="firm_a")
        out = await pg_client.update_vertex(
            "t", realm=realm, vertex_id=v.id, payload={"x": 2}, space="firm_b"
        )
        assert out.space == "firm_b"

    @pytest.mark.asyncio
    async def test_the_reserved_space_is_refused(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("t", realm=realm, audited=False)
        v = await pg_client.add_vertex("t", realm=realm, payload={"x": 1})
        with pytest.raises(ReservedSpaceError):
            await pg_client.update_vertex(
                "t", realm=realm, vertex_id=v.id, payload={"x": 2}, space="__all__"
            )


class TestInsideACallersTransaction:

    @pytest.mark.asyncio
    async def test_read_then_write_under_the_lock_is_one_transaction(
        self, pg_client, clean_realm
    ):
        """How the caller uses it: take the table's advisory lock, read the
        row, refuse if already spent, update it. The lock added in 1.9.0 makes
        that atomic only if the write joins the same transaction, so this
        asserts the write is visible inside it and that rolling the caller's
        transaction back takes the update with it."""
        realm = clean_realm
        await pg_client.create_vertex_table("invitations", realm=realm, audited=False)
        v = await pg_client.add_vertex(
            "invitations", realm=realm, payload={"redeemed_at": None}
        )

        conn = await asyncpg.connect(DSN)
        try:
            client = AsyncPostGraph(connection_or_pool=conn)
            tx = conn.transaction()
            await tx.start()
            await client.advisory_xact_lock("invitations", realm=realm)
            row = await client.get_vertex("invitations", realm=realm, vertex_id=v.id)
            assert row.payload["redeemed_at"] is None
            await client.update_vertex(
                "invitations", realm=realm, vertex_id=v.id,
                payload={"redeemed_at": "2026-10-10T09:00:00Z"},
            )
            inside = await conn.fetchval(
                'SELECT payload->>\'redeemed_at\' FROM "invitations" '
                "WHERE realm = $1 AND id = $2", realm, int(v.id)
            )
            assert inside == "2026-10-10T09:00:00Z", "the write must join the caller's transaction"
            await tx.rollback()
        finally:
            await conn.close()

        after = await pg_client.get_vertex("invitations", realm=realm, vertex_id=v.id)
        assert after.payload["redeemed_at"] is None, "rollback must take the update with it"

    @pytest.mark.asyncio
    async def test_a_second_update_in_the_same_transaction_sees_the_first(
        self, pg_client, clean_realm
    ):
        realm = clean_realm
        await pg_client.create_vertex_table("t", realm=realm, audited=False)
        v = await pg_client.add_vertex("t", realm=realm, payload={"n": 0})
        conn = await asyncpg.connect(DSN)
        try:
            client = AsyncPostGraph(connection_or_pool=conn)
            async with conn.transaction():
                await client.update_vertex("t", realm=realm, vertex_id=v.id, payload={"a": 1})
                out = await client.update_vertex(
                    "t", realm=realm, vertex_id=v.id, payload={"b": 2}
                )
                assert out.payload == {"n": 0, "a": 1, "b": 2}
        finally:
            await conn.close()


class TestAuditTrail:

    @pytest.mark.asyncio
    async def test_an_audited_update_is_recorded_as_an_update(
        self, pg_client, clean_realm
    ):
        """The audit row must say UPDATE, with both sides. For a single-use
        token this trail is the evidence of who spent it."""
        realm = clean_realm
        await pg_client.create_vertex_table("t", realm=realm)
        v = await pg_client.add_vertex("t", realm=realm, payload={"spent": False})
        await pg_client.update_vertex(
            "t", realm=realm, vertex_id=v.id, payload={"spent": True}, user_id="auditor-7"
        )
        rows = await pg_client._fetch(
            'SELECT action, changed_by, old_row, new_row FROM "t_audit" '
            "WHERE realm = $1 AND action = 'UPDATE' ORDER BY audit_id DESC",
            realm,
        )
        assert rows, "the update must leave an audit row"
        rec = rows[0]
        assert rec["changed_by"] == "auditor-7"
        old = json.loads(rec["old_row"]) if isinstance(rec["old_row"], str) else rec["old_row"]
        new = json.loads(rec["new_row"]) if isinstance(rec["new_row"], str) else rec["new_row"]
        assert old["payload"]["spent"] is False
        assert new["payload"]["spent"] is True


class TestTableNameHandling:

    @pytest.mark.asyncio
    async def test_a_missing_table_raises_table_not_found(self, pg_client, clean_realm):
        from post_graph.errors import TableNotFoundError
        with pytest.raises(TableNotFoundError):
            await pg_client.update_vertex(
                "no_such_table", realm=clean_realm, vertex_id=1, payload={"a": 1}
            )

    @pytest.mark.asyncio
    async def test_an_invalid_identifier_is_refused(self, pg_client, clean_realm):
        with pytest.raises(Exception):
            await pg_client.update_vertex(
                'bad"; DROP TABLE x; --', realm=clean_realm, vertex_id=1, payload={"a": 1}
            )

    @pytest.mark.asyncio
    async def test_an_fqid_style_vertex_id_is_accepted(self, pg_client, clean_realm):
        """Parity with upsert_vertex, which splits on '/' so a caller can pass
        a vertex's fqid straight back in."""
        realm = clean_realm
        await pg_client.create_vertex_table("t", realm=realm, audited=False)
        v = await pg_client.add_vertex("t", realm=realm, payload={"x": 1})
        out = await pg_client.update_vertex(
            "t", realm=realm, vertex_id=v.fqid, payload={"x": 2}
        )
        assert out is not None and out.payload["x"] == 2


class TestSchemaPerRealm:

    @pytest.mark.asyncio
    async def test_it_works_in_schema_per_realm_mode(self, pg_client_spr, clean_realm_spr):
        realm = clean_realm_spr
        await pg_client_spr.create_vertex_table("t", realm=realm, audited=False)
        v = await pg_client_spr.add_vertex("t", realm=realm, payload={"x": 1})
        out = await pg_client_spr.update_vertex(
            "t", realm=realm, vertex_id=v.id, payload={"x": 2}
        )
        assert out is not None and out.payload["x"] == 2
        assert await pg_client_spr.update_vertex(
            "t", realm=realm, vertex_id=10 ** 9, payload={"x": 3}
        ) is None


class TestSQLAlchemyParity:
    """client_sqlalchemy.py's docstring claims parity, so it has to hold."""

    @pytest.mark.asyncio
    async def test_merge_and_absent_row(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("t", realm=realm, audited=False)
        v = await sa_client.add_vertex("t", realm=realm, payload={"keep": 1, "change": "old"})
        out = await sa_client.update_vertex(
            "t", realm=realm, vertex_id=v.id, payload={"change": "new"}
        )
        assert out.payload == {"keep": 1, "change": "new"}
        assert await sa_client.update_vertex(
            "t", realm=realm, vertex_id=10 ** 9, payload={"x": 1}
        ) is None

    @pytest.mark.asyncio
    async def test_replace_and_space(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("t", realm=realm, audited=False)
        v = await sa_client.add_vertex("t", realm=realm, payload={"gone": 1, "stays": 2}, space="firm_a")
        out = await sa_client.update_vertex(
            "t", realm=realm, vertex_id=v.id, payload={"stays": 2}, replace=True
        )
        assert out.payload == {"stays": 2}
        assert out.space == "firm_a", "an omitted space must not relocate the row"
        moved = await sa_client.update_vertex(
            "t", realm=realm, vertex_id=v.id, payload={}, space="firm_b"
        )
        assert moved.space == "firm_b"

    @pytest.mark.asyncio
    async def test_it_never_inserts(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("t", realm=realm, audited=False)
        await sa_client.add_vertex("t", realm=realm, payload={"x": 1})
        rows = await sa_client.find_vertices("t", realm=realm)
        await sa_client.update_vertex("t", realm=realm, vertex_id=10 ** 9, payload={"x": 1})
        assert len(await sa_client.find_vertices("t", realm=realm)) == len(rows)

    @pytest.mark.asyncio
    async def test_the_reserved_space_is_refused(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table("t", realm=realm, audited=False)
        v = await sa_client.add_vertex("t", realm=realm, payload={"x": 1})
        with pytest.raises(ReservedSpaceError):
            await sa_client.update_vertex(
                "t", realm=realm, vertex_id=v.id, payload={"x": 2}, space="__all__"
            )
