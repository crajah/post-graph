"""The primitives that remove a caller's last reasons to hold raw SQL.

A caller whose rule is that every statement goes through this library still
needed its own connection for six things: creating a database before anything
can connect to it, grouping several writes into one transaction, a unique index
over a payload key, compare-and-swap, asking the catalogue what exists, and
removing one record's shadow audit rows.

Each test here is the thing the caller could not previously express, not a
demonstration that SQL works.
"""

import asyncio
import uuid

import asyncpg
import pytest
from conftest import DSN

from post_graph import AsyncPostGraph
from post_graph.errors import (
    CommitFailedError,
    ConcurrentModificationError,
    NestedTransactionError,
    PostGraphError,
    TableExistsError,
    UniqueConstraintError,
)


class TestEnsureDatabase:

    @pytest.mark.asyncio
    async def test_it_creates_and_returns_a_usable_dsn(self):
        name = f"pg_ensure_{uuid.uuid4().hex[:10]}"
        admin = await asyncpg.connect(DSN)
        try:
            resolved = await AsyncPostGraph.ensure_database(DSN, database=name)
            assert resolved.endswith(f"/{name}")
            client = AsyncPostGraph(dsn=resolved, schema_per_realm=True)
            await client.connect()
            try:
                realm = f"r_{uuid.uuid4().hex[:8]}"
                await client.create_vertex_table("Thing", realm=realm)
                assert await client.table_exists("Thing", realm) is True
            finally:
                await client.close()
        finally:
            await admin.execute(f'DROP DATABASE IF EXISTS "{name}"')
            await admin.close()

    @pytest.mark.asyncio
    async def test_it_is_idempotent_and_survives_a_deploy_herd(self):
        """Replicas starting together all find it absent and race to create it.
        The loser's DuplicateDatabaseError is success: it exists."""
        name = f"pg_herd_{uuid.uuid4().hex[:10]}"
        admin = await asyncpg.connect(DSN)
        try:
            results = await asyncio.gather(*(
                AsyncPostGraph.ensure_database(DSN, database=name)
                for _ in range(4)))
            assert all(r.endswith(f"/{name}") for r in results)
            # And again, now that it plainly exists.
            assert (await AsyncPostGraph.ensure_database(
                DSN, database=name)).endswith(f"/{name}")
            assert await admin.fetchval(
                "SELECT count(*) FROM pg_database WHERE datname = $1", name) == 1
        finally:
            await admin.execute(f'DROP DATABASE IF EXISTS "{name}"')
            await admin.close()

    @pytest.mark.parametrize("hostile", [
        'bad"; DROP DATABASE postgres; --',
        "has space",
        "9lives",
        "",
        "x" * 64,
    ])
    @pytest.mark.asyncio
    async def test_a_hostile_name_is_refused_before_connecting(self, hostile):
        """CREATE DATABASE takes no bound parameter, so the name is
        interpolated; it is validated before anything is opened."""
        with pytest.raises(ValueError):
            await AsyncPostGraph.ensure_database(
                "postgresql://127.0.0.1:1/x", database=hostile)

    @pytest.mark.asyncio
    async def test_a_maintenance_database_that_is_not_there(self):
        with pytest.raises(PostGraphError):
            await AsyncPostGraph.ensure_database(
                DSN, database=f"d_{uuid.uuid4().hex[:8]}",
                maintenance_database=f"absent_{uuid.uuid4().hex[:8]}")


class TestTransaction:

    @pytest.mark.asyncio
    async def test_the_block_is_atomic_and_invisible_until_commit(
        self, pg_client_spr, clean_realm_spr
    ):
        realm = clean_realm_spr
        await pg_client_spr.create_vertex_table("Item", realm=realm, audited=True)
        other = AsyncPostGraph(dsn=DSN, schema_per_realm=True)
        await other.connect()
        try:
            async with pg_client_spr.transaction(user_id="usr_alice") as tx:
                await tx.add_vertex("Item", realm=realm, payload={"k": "1"})
                await tx.add_vertex("Item", realm=realm, payload={"k": "2"})
                # Nothing is visible outside the transaction yet.
                assert await other.count_vertices("Item", realm=realm) == 0
            assert await other.count_vertices("Item", realm=realm) == 2
        finally:
            await other.close()

    @pytest.mark.asyncio
    async def test_the_actor_is_remembered_and_can_be_overridden(
        self, pg_client_spr, clean_realm_spr
    ):
        """The detail that makes grouped writes auditable.

        A call inside the block that names no user_id used to blank
        app.current_user_id, so its audit row recorded no actor -- losing
        attribution on exactly the writes important enough to group. An
        explicit user_id on one call must not change the ones after it.
        """
        realm = clean_realm_spr
        await pg_client_spr.create_vertex_table("Item", realm=realm, audited=True)
        async with pg_client_spr.transaction(user_id="usr_alice") as tx:
            v1 = await tx.add_vertex("Item", realm=realm, payload={"k": "1"})
            v2 = await tx.add_vertex("Item", realm=realm, payload={"k": "2"},
                                     user_id="usr_bob")
            v3 = await tx.add_vertex("Item", realm=realm, payload={"k": "3"})

        rows = await pg_client_spr._fetch(
            f'SELECT changed_by, new_row->>\'id\' AS id FROM "{realm}"."Item_audit" '
            f"WHERE action = 'INSERT' ORDER BY audit_id")
        actors = {r["id"]: r["changed_by"] for r in rows}
        assert actors[v1.id] == "usr_alice"
        assert actors[v2.id] == "usr_bob", "an explicit actor must be honoured"
        assert actors[v3.id] == "usr_alice", "and must not persist to later calls"

    @pytest.mark.asyncio
    async def test_an_exception_rolls_the_whole_block_back(
        self, pg_client_spr, clean_realm_spr
    ):
        realm = clean_realm_spr
        await pg_client_spr.create_vertex_table("Item", realm=realm)
        with pytest.raises(RuntimeError, match="boom"):
            async with pg_client_spr.transaction(user_id="usr_alice") as tx:
                await tx.add_vertex("Item", realm=realm, payload={"k": "1"})
                raise RuntimeError("boom")
        assert await pg_client_spr.count_vertices("Item", realm=realm) == 0

    @pytest.mark.asyncio
    async def test_a_nested_transaction_is_refused(self, pg_client_spr, realm):
        async with pg_client_spr.transaction(user_id="usr_alice") as tx:
            with pytest.raises(NestedTransactionError):
                async with tx.transaction(user_id="usr_alice"):
                    pass

    @pytest.mark.asyncio
    async def test_connect_and_close_are_refused_on_the_scoped_client(
        self, pg_client_spr, realm
    ):
        """It borrowed the connection; closing it would strand the block."""
        async with pg_client_spr.transaction(user_id="usr_alice") as tx:
            with pytest.raises(PostGraphError):
                await tx.connect()
            with pytest.raises(PostGraphError):
                await tx.close()

    @pytest.mark.asyncio
    async def test_the_scoped_client_is_poisoned_after_the_block(
        self, pg_client_spr, clean_realm_spr
    ):
        """A reference kept past the block would otherwise write outside any
        transaction, on a connection now serving another request."""
        realm = clean_realm_spr
        await pg_client_spr.create_vertex_table("Item", realm=realm)
        async with pg_client_spr.transaction(user_id="usr_alice") as tx:
            await tx.add_vertex("Item", realm=realm, payload={"k": "1"})
        stashed = tx
        with pytest.raises(PostGraphError, match="has exited"):
            await stashed.add_vertex("Item", realm=realm, payload={"k": "2"})
        with pytest.raises(PostGraphError, match="has exited"):
            await stashed.find_vertices("Item", realm=realm)

    @pytest.mark.asyncio
    async def test_the_idle_timeout_is_transaction_local(
        self, pg_client_spr, realm
    ):
        """SET LOCAL, so it reverts on commit and cannot leak onto a pooled
        backend the next request borrows."""
        async with pg_client_spr.transaction(
            user_id="usr_alice", idle_in_transaction_timeout_ms=5000
        ) as tx:
            setting = await tx._fetchrow(
                "SELECT current_setting('idle_in_transaction_session_timeout') AS v")
            assert setting["v"] == "5s"
        after = await pg_client_spr._fetchrow(
            "SELECT current_setting('idle_in_transaction_session_timeout') AS v")
        assert after["v"] in ("0", "0ms"), after["v"]

    @pytest.mark.parametrize("bad", [True, False, 0, -1, 1.5, "5000"])
    @pytest.mark.asyncio
    async def test_a_bad_idle_timeout_is_refused(self, pg_client_spr, bad):
        """True is an int in Python and would become '1ms', firing at once."""
        with pytest.raises(ValueError):
            async with pg_client_spr.transaction(
                user_id="usr_alice", idle_in_transaction_timeout_ms=bad):
                pass

    @pytest.mark.parametrize("bad", [None, "", "   ", 7])
    @pytest.mark.asyncio
    async def test_an_empty_actor_is_refused(self, pg_client_spr, bad):
        with pytest.raises(ValueError):
            async with pg_client_spr.transaction(user_id=bad):
                pass

    @pytest.mark.asyncio
    async def test_a_failed_commit_is_its_own_error(self, pg_client_spr,
                                                    clean_realm_spr):
        """The caller's situation differs: the block ran, so any external
        effect has happened, and only the writes are in doubt.

        Produced with a DEFERRABLE INITIALLY DEFERRED constraint trigger, which
        runs at COMMIT rather than at the statement. Two earlier attempts
        proved nothing: a CHECK constraint cannot be deferred at all, and a
        UNIQUE constraint cannot be built on the expression index a payload key
        needs.
        """
        realm = clean_realm_spr
        await pg_client_spr.create_vertex_table("Item", realm=realm)
        await pg_client_spr._execute(
            f'CREATE OR REPLACE FUNCTION "{realm}".fail_at_commit() '
            f"RETURNS TRIGGER AS $$ BEGIN "
            f"RAISE EXCEPTION 'deferred failure at commit'; "
            f"END $$ LANGUAGE plpgsql")
        await pg_client_spr._execute(
            f'CREATE CONSTRAINT TRIGGER fail_late AFTER INSERT ON "{realm}"."Item" '
            f"DEFERRABLE INITIALLY DEFERRED FOR EACH ROW "
            f'EXECUTE FUNCTION "{realm}".fail_at_commit()')

        with pytest.raises(CommitFailedError) as exc:
            async with pg_client_spr.transaction(user_id="usr_alice") as tx:
                await tx.add_vertex("Item", realm=realm, payload={"k": "one"})
        assert "COMMIT failed" in str(exc.value)
        assert "deferred failure at commit" in str(exc.value)
        # Nothing was written: a failed commit is still a rollback as far as
        # the database is concerned. What it cannot undo is anything the block
        # did outside the database, which is why it has its own type.
        assert await pg_client_spr.count_vertices("Item", realm=realm) == 0


class TestUniquePayloadIndexes:

    @pytest.mark.asyncio
    async def test_uniqueness_per_space(self, pg_client_spr, clean_realm_spr):
        """Two tenants may each hold an 'oliver'; one tenant may not hold two."""
        realm = clean_realm_spr
        await pg_client_spr.create_vertex_table("Credential", realm=realm)
        name = await pg_client_spr.create_payload_index(
            "Credential", realm, "username_key", unique=True,
            scope_by_space=True, index_name="uq_credential_username_per_space")
        assert name == "uq_credential_username_per_space"

        await pg_client_spr.add_vertex("Credential", realm=realm,
                                       payload={"username_key": "oliver"},
                                       space="institution:a")
        await pg_client_spr.add_vertex("Credential", realm=realm,
                                       payload={"username_key": "oliver"},
                                       space="institution:b")

        for writer in ("add_vertex", "upsert_vertex"):
            with pytest.raises(UniqueConstraintError) as exc:
                await getattr(pg_client_spr, writer)(
                    "Credential", realm=realm,
                    payload={"username_key": "oliver"},
                    space="institution:a")
            assert exc.value.constraint_name == "uq_credential_username_per_space"
            # Still a TableExistsError, so a caller written against 1.13.0 and
            # earlier keeps working.
            assert isinstance(exc.value, TableExistsError)
            # And the message does not claim the *id* was taken, which it was not.
            assert "uq_credential_username_per_space" in str(exc.value)

    @pytest.mark.asyncio
    async def test_a_composite_index_across_spaces(self, pg_client_spr, clean_realm_spr):
        realm = clean_realm_spr
        await pg_client_spr.create_vertex_table("Course", realm=realm)
        await pg_client_spr.create_payload_index(
            "Course", realm, ("institution_id", "slug"),
            unique=True, scope_by_space=False, index_name="uq_course_inst_slug")
        await pg_client_spr.add_vertex(
            "Course", realm=realm, payload={"institution_id": "i1", "slug": "s"},
            space="one")
        with pytest.raises(UniqueConstraintError) as exc:
            await pg_client_spr.add_vertex(
                "Course", realm=realm,
                payload={"institution_id": "i1", "slug": "s"}, space="two")
        assert exc.value.constraint_name == "uq_course_inst_slug"
        # A different institution is fine.
        await pg_client_spr.add_vertex(
            "Course", realm=realm, payload={"institution_id": "i2", "slug": "s"},
            space="two")

    @pytest.mark.asyncio
    async def test_an_update_that_collides(self, pg_client_spr, clean_realm_spr):
        """An UPDATE can violate a unique payload index too, and used to let
        the driver's exception escape."""
        realm = clean_realm_spr
        await pg_client_spr.create_vertex_table("Credential", realm=realm)
        await pg_client_spr.create_payload_index(
            "Credential", realm, "username_key", unique=True,
            index_name="uq_cred_u")
        await pg_client_spr.add_vertex("Credential", realm=realm,
                                       payload={"username_key": "a"}, space="s")
        v = await pg_client_spr.add_vertex("Credential", realm=realm,
                                           payload={"username_key": "b"}, space="s")
        with pytest.raises(UniqueConstraintError) as exc:
            await pg_client_spr.update_vertex("Credential", realm, v.id,
                                              payload={"username_key": "a"})
        assert exc.value.constraint_name == "uq_cred_u"

    @pytest.mark.asyncio
    async def test_the_legacy_shape_is_unchanged(self, pg_client_spr, clean_realm_spr):
        """A non-unique single-key caller must not acquire a second index."""
        realm = clean_realm_spr
        await pg_client_spr.create_vertex_table("Thing", realm=realm)
        assert await pg_client_spr.create_payload_index(
            "Thing", realm, "due_at") == "idx_Thing_payload_due_at"
        assert await pg_client_spr.create_payload_index(
            "Thing", realm, "n", numeric=True) == "idx_Thing_payload_n_num"

    @pytest.mark.asyncio
    async def test_a_name_too_long_is_refused(self, pg_client_spr, realm):
        """PostgreSQL truncates silently, so two names sharing a 63-byte
        prefix would become one index and drop a uniqueness constraint."""
        with pytest.raises(ValueError, match="63"):
            await pg_client_spr.create_payload_index(
                "Thing", realm, "k", unique=True, index_name="x" * 64)

    @pytest.mark.asyncio
    async def test_a_derived_name_too_long_is_hashed_not_truncated(
        self, pg_client_spr, clean_realm_spr
    ):
        realm = clean_realm_spr
        await pg_client_spr.create_vertex_table("T", realm=realm)
        keys_a = tuple(f"key_number_{i}" for i in range(6))
        keys_b = keys_a[:-1] + ("key_number_99",)
        a = await pg_client_spr.create_payload_index("T", realm, keys_a, unique=True)
        b = await pg_client_spr.create_payload_index("T", realm, keys_b, unique=True)
        assert a != b, "two key tuples must not collide under 63 bytes"
        assert len(a.encode()) <= 63 and len(b.encode()) <= 63

    @pytest.mark.asyncio
    async def test_an_empty_key_sequence_and_numeric_composite(
        self, pg_client_spr, realm
    ):
        with pytest.raises(ValueError):
            await pg_client_spr.create_payload_index("T", realm, [])
        with pytest.raises(ValueError, match="numeric"):
            await pg_client_spr.create_payload_index(
                "T", realm, ("a", "b"), numeric=True)

    @pytest.mark.asyncio
    async def test_building_over_existing_duplicates_says_so(
        self, pg_client_spr, clean_realm_spr
    ):
        realm = clean_realm_spr
        await pg_client_spr.create_vertex_table("Dup", realm=realm)
        for _ in range(2):
            await pg_client_spr.add_vertex("Dup", realm=realm,
                                           payload={"k": "same"}, space="s")
        with pytest.raises(UniqueConstraintError, match="duplicate"):
            await pg_client_spr.create_payload_index(
                "Dup", realm, "k", unique=True, index_name="uq_dup_k")


class TestCompareAndSwap:

    @pytest.mark.asyncio
    async def test_a_fresh_token_updates_and_a_stale_one_refuses(
        self, pg_client_spr, clean_realm_spr
    ):
        realm = clean_realm_spr
        await pg_client_spr.create_vertex_table("RoleGrant", realm=realm)
        v = await pg_client_spr.add_vertex("RoleGrant", realm=realm,
                                           payload={"roles": ["instructor"]})
        updated = await pg_client_spr.update_vertex(
            "RoleGrant", realm, v.id, payload={"roles": ["instructor", "admin"]},
            expected_updated_at=v.updated_at)
        assert updated is not None
        assert updated.updated_at > v.updated_at

        for call in (
            lambda: pg_client_spr.update_vertex(
                "RoleGrant", realm, v.id, payload={"roles": ["x"]},
                expected_updated_at=v.updated_at),
            lambda: pg_client_spr.upsert_vertex(
                "RoleGrant", realm, vertex_id=v.id, payload={"roles": ["y"]},
                expected_updated_at=v.updated_at),
        ):
            with pytest.raises(ConcurrentModificationError):
                await call()

        # The winner's payload survived both refusals.
        final = await pg_client_spr.get_vertex("RoleGrant", realm, v.id)
        assert final.payload["roles"] == ["instructor", "admin"]

    @pytest.mark.asyncio
    async def test_a_missing_row_is_not_a_lost_race(
        self, pg_client_spr, clean_realm_spr
    ):
        """Nothing to update and somebody else updated it are different
        answers, and the caller has to be able to tell them apart."""
        realm = clean_realm_spr
        await pg_client_spr.create_vertex_table("RoleGrant", realm=realm)
        v = await pg_client_spr.add_vertex("RoleGrant", realm=realm, payload={})
        assert await pg_client_spr.update_vertex(
            "RoleGrant", realm, 999999, payload={"x": 1},
            expected_updated_at=v.updated_at) is None

    @pytest.mark.asyncio
    async def test_an_upsert_that_creates_needs_no_token(
        self, pg_client_spr, clean_realm_spr
    ):
        realm = clean_realm_spr
        await pg_client_spr.create_vertex_table("RoleGrant", realm=realm)
        # A supplied id that does not exist yet: the INSERT path runs.
        created = await pg_client_spr.upsert_vertex(
            "RoleGrant", realm, vertex_id=4242, payload={"k": "v"},
            expected_updated_at=None)
        assert created.id == "4242"

    @pytest.mark.asyncio
    async def test_a_token_without_an_id_is_refused(self, pg_client_spr, realm):
        with pytest.raises(ValueError, match="requires vertex_id"):
            await pg_client_spr.upsert_vertex(
                "RoleGrant", realm, payload={},
                expected_updated_at=__import__("datetime").datetime.now())

    @pytest.mark.asyncio
    async def test_updated_at_advances_within_one_transaction(
        self, pg_client_spr, clean_realm_spr
    ):
        """Why the trigger uses clock_timestamp() and not now().

        now() is the transaction's start time, so two writes to one row inside
        a transaction would leave updated_at unchanged between them -- and a
        compare-and-swap token taken from the first would still match after the
        second, which is the opposite of what it is for.
        """
        realm = clean_realm_spr
        await pg_client_spr.create_vertex_table("Item", realm=realm)
        v = await pg_client_spr.add_vertex("Item", realm=realm, payload={"n": 0})
        async with pg_client_spr.transaction(user_id="usr_alice") as tx:
            first = await tx.update_vertex("Item", realm, v.id, payload={"n": 1})
            second = await tx.update_vertex("Item", realm, v.id, payload={"n": 2})
        assert second.updated_at > first.updated_at


class TestRealmAndCatalogue:

    @pytest.mark.asyncio
    async def test_list_realms_matches_the_prefix_literally(self, pg_client_spr):
        """`_` is a wildcard in LIKE, so an unescaped prefix of 'learner_'
        also matches 'learnerXYZ'. That decides which schemas get dropped."""
        tag = uuid.uuid4().hex[:8]
        names = [f"lrn{tag}_001", f"lrn{tag}_002", f"lrn{tag}s_archive",
                 f"lrn{tag}X_other"]
        try:
            for n in names:
                await pg_client_spr.create_vertex_table("Note", realm=n)
            found = await pg_client_spr.list_realms(prefix=f"lrn{tag}_")
            assert found == [f"lrn{tag}_001", f"lrn{tag}_002"], found
        finally:
            for n in names:
                await pg_client_spr._execute(f'DROP SCHEMA IF EXISTS "{n}" CASCADE')

    @pytest.mark.asyncio
    async def test_realm_exists_table_exists_and_drop(self, pg_client_spr):
        realm = f"r_{uuid.uuid4().hex[:10]}"
        assert await pg_client_spr.realm_exists(realm) is False
        await pg_client_spr.create_vertex_table("Note", realm=realm, audited=True)
        assert await pg_client_spr.realm_exists(realm) is True
        assert await pg_client_spr.table_exists("Note", realm) is True
        assert await pg_client_spr.table_exists("Note_audit", realm) is True
        assert await pg_client_spr.table_exists("Absent", realm) is False

        assert await pg_client_spr.drop_realm(realm) is True
        assert await pg_client_spr.realm_exists(realm) is False
        assert await pg_client_spr.drop_realm(realm) is False, "already absent"

    @pytest.mark.asyncio
    async def test_drop_realm_refuses_without_schema_per_realm(self, pg_client):
        with pytest.raises(PostGraphError, match="schema_per_realm"):
            await pg_client.drop_realm("whatever")

    @pytest.mark.asyncio
    async def test_tables_containing_text_finds_the_audit_copy(self, pg_client_spr):
        """The residue check: a deletion that misses the shadow tables is the
        failure this is for, so it has to see them."""
        realm = f"r_{uuid.uuid4().hex[:10]}"
        other = f"o_{uuid.uuid4().hex[:10]}"
        marker = f"SECRET_MARKER_{uuid.uuid4().hex[:8]}"
        try:
            await pg_client_spr.create_vertex_table("Note", realm=realm, audited=True)
            await pg_client_spr.create_vertex_table("Note", realm=other, audited=True)
            await pg_client_spr.add_vertex("Note", realm=realm,
                                           payload={"text": marker})
            hits = await pg_client_spr.tables_containing_text(
                marker, include_realms=[realm, other])
            assert hits == {f"{realm}.Note", f"{realm}.Note_audit"}, hits
            assert await pg_client_spr.tables_containing_text(
                f"absent_{uuid.uuid4().hex[:8]}", include_realms=[realm]) == set()
        finally:
            for n in (realm, other):
                await pg_client_spr._execute(f'DROP SCHEMA IF EXISTS "{n}" CASCADE')

    @pytest.mark.asyncio
    async def test_a_needle_with_wildcards_is_matched_literally(self, pg_client_spr):
        realm = f"r_{uuid.uuid4().hex[:10]}"
        try:
            await pg_client_spr.create_vertex_table("Note", realm=realm)
            await pg_client_spr.add_vertex("Note", realm=realm,
                                           payload={"text": "a_b"})
            assert await pg_client_spr.tables_containing_text(
                "aXb", include_realms=[realm]) == set()
            assert await pg_client_spr.tables_containing_text(
                "a_b", include_realms=[realm]) == {
                    f"{realm}.Note", f"{realm}.Note_audit"}
        finally:
            await pg_client_spr._execute(f'DROP SCHEMA IF EXISTS "{realm}" CASCADE')

    @pytest.mark.asyncio
    async def test_exclude_prefixes_unless_included(self, pg_client_spr):
        tag = uuid.uuid4().hex[:8]
        a, b = f"ex{tag}_keep", f"ex{tag}_skip"
        marker = f"MARK_{tag}"
        try:
            for n in (a, b):
                await pg_client_spr.create_vertex_table("Note", realm=n)
                await pg_client_spr.add_vertex("Note", realm=n,
                                               payload={"text": marker})
            hits = await pg_client_spr.tables_containing_text(
                marker, include_realms=[a], exclude_realm_prefixes=[f"ex{tag}_"])
            assert hits == {f"{a}.Note", f"{a}.Note_audit"}, hits
        finally:
            for n in (a, b):
                await pg_client_spr._execute(f'DROP SCHEMA IF EXISTS "{n}" CASCADE')

    @pytest.mark.asyncio
    async def test_an_empty_needle_is_refused(self, pg_client_spr):
        with pytest.raises(ValueError):
            await pg_client_spr.tables_containing_text("")


class TestPurgeRecordAudit:

    @pytest.mark.asyncio
    async def test_it_removes_every_version_of_one_record(self, pg_client_spr):
        """Insert, update and delete leave three audit rows, and the contents
        live in new_row for the insert and old_row for the delete. Matching
        only one of them would leave the oldest copy in place."""
        realm = f"r_{uuid.uuid4().hex[:10]}"
        secret = f"MY_SECRET_{uuid.uuid4().hex[:8]}"
        keep = f"KEEP_ME_{uuid.uuid4().hex[:8]}"
        try:
            await pg_client_spr.create_vertex_table("Document", realm=realm,
                                                    audited=True)
            v1 = await pg_client_spr.add_vertex("Document", realm=realm,
                                                payload={"text": secret})
            await pg_client_spr.update_vertex("Document", realm, v1.id,
                                              payload={"text": secret, "n": 1})
            await pg_client_spr.delete_vertex("Document", realm=realm,
                                              vertex_id=v1.id)
            v2 = await pg_client_spr.add_vertex("Document", realm=realm,
                                                payload={"text": keep})

            assert await pg_client_spr.tables_containing_text(
                secret, include_realms=[realm]) == {f"{realm}.Document_audit"}

            purged = await pg_client_spr.purge_record_audit(
                "Document", realm, [v1.id])
            assert purged == 3, "INSERT, UPDATE and DELETE audit rows"

            assert await pg_client_spr.tables_containing_text(
                secret, include_realms=[realm]) == set()
            # The other record's history is untouched.
            assert await pg_client_spr.tables_containing_text(
                keep, include_realms=[realm]) == {
                    f"{realm}.Document", f"{realm}.Document_audit"}
            assert v2.id
        finally:
            await pg_client_spr._execute(f'DROP SCHEMA IF EXISTS "{realm}" CASCADE')

    @pytest.mark.asyncio
    async def test_empty_ids_and_an_unaudited_table(self, pg_client_spr,
                                                    clean_realm_spr):
        realm = clean_realm_spr
        await pg_client_spr.create_vertex_table("Plain", realm=realm, audited=False)
        assert await pg_client_spr.purge_record_audit("Plain", realm, []) == 0
        assert await pg_client_spr.purge_record_audit("Plain", realm, None) == 0
        # No audit table: the caller asked for the audit rows gone, and
        # there are none.
        assert await pg_client_spr.purge_record_audit("Plain", realm, [1]) == 0

    @pytest.mark.asyncio
    async def test_it_accepts_one_id_or_an_fqid(self, pg_client_spr, clean_realm_spr):
        realm = clean_realm_spr
        await pg_client_spr.create_vertex_table("Doc", realm=realm, audited=True)
        v = await pg_client_spr.add_vertex("Doc", realm=realm, payload={"a": 1})
        assert await pg_client_spr.purge_record_audit("Doc", realm, v.id) == 1
        v2 = await pg_client_spr.add_vertex("Doc", realm=realm, payload={"a": 2})
        assert await pg_client_spr.purge_record_audit(
            "Doc", realm, f"{realm}/Doc/{v2.id}") == 1

    @pytest.mark.asyncio
    async def test_a_hostile_table_name(self, pg_client_spr, realm):
        with pytest.raises(ValueError):
            await pg_client_spr.purge_record_audit(
                'x"; DROP SCHEMA y; --', realm, [1])


class TestSQLAlchemyParity:
    """The two clients have diverged before, always the same way: a rule lived
    in one and was reimplemented or forgotten in the other, invisibly, because
    each client's tests only exercised that client. These assert the surfaces
    match and then exercise the SQLAlchemy side of each primitive."""

    def test_the_public_surfaces_match(self):
        import inspect

        from post_graph.client_asyncpg import AsyncPostGraph as A
        from post_graph.client_sqlalchemy import SQLAlchemyPostGraph as S

        def public(cls):
            return {n for n, _ in inspect.getmembers(cls, callable)
                    if not n.startswith("_")}

        # connect/close/get_pool_config belong to the client that owns a pool;
        # the SQLAlchemy one is handed an engine.
        pool_lifecycle = {"connect", "close", "get_pool_config"}
        assert public(A) - public(S) - pool_lifecycle == set()

        for name in ("create_payload_index", "upsert_vertex", "update_vertex",
                     "purge_record_audit", "drop_realm", "list_realms",
                     "tables_containing_text", "transaction"):
            pa = list(inspect.signature(getattr(A, name)).parameters)
            ps = list(inspect.signature(getattr(S, name)).parameters)
            assert [p for p in pa if p not in ps] == [], (name, pa, ps)

    def test_the_shared_rules_are_shared_and_not_copied(self):
        """Each rule has one implementation, so the clients cannot disagree."""
        from post_graph._shared import SharedLogic
        from post_graph.client_asyncpg import AsyncPostGraph as A
        from post_graph.client_sqlalchemy import SQLAlchemyPostGraph as S

        assert issubclass(A, SharedLogic) and issubclass(S, SharedLogic)
        for name in ("_like_literal", "_normalise_ids", "_unique_error",
                     "_plan_payload_index", "_bounded_index_name",
                     "_validate_database_name"):
            assert name not in A.__dict__, f"{name} is copied in AsyncPostGraph"
            assert name not in S.__dict__, f"{name} is copied in SQLAlchemyPostGraph"

    @pytest.mark.asyncio
    async def test_purge_record_audit(self, sa_client_spr, sa_clean_realm_spr):
        realm = sa_clean_realm_spr
        secret = f"SA_SECRET_{uuid.uuid4().hex[:8]}"
        await sa_client_spr.create_vertex_table("Doc", realm=realm, audited=True)
        v = await sa_client_spr.add_vertex("Doc", realm=realm,
                                          payload={"text": secret})
        await sa_client_spr.update_vertex("Doc", realm, v.id,
                                          payload={"text": secret, "n": 1})
        assert await sa_client_spr.tables_containing_text(
            secret, include_realms=[realm]) == {
                f"{realm}.Doc", f"{realm}.Doc_audit"}
        assert await sa_client_spr.purge_record_audit("Doc", realm, [v.id]) == 2
        assert await sa_client_spr.tables_containing_text(
            secret, include_realms=[realm]) == {f"{realm}.Doc"}

    @pytest.mark.asyncio
    async def test_realm_lifecycle(self, sa_client_spr):
        realm = f"sa_{uuid.uuid4().hex[:10]}"
        assert await sa_client_spr.realm_exists(realm) is False
        await sa_client_spr.create_vertex_table("Note", realm=realm, audited=True)
        assert await sa_client_spr.realm_exists(realm) is True
        assert await sa_client_spr.table_exists("Note_audit", realm) is True
        assert await sa_client_spr.drop_realm(realm) is True
        assert await sa_client_spr.drop_realm(realm) is False

    @pytest.mark.asyncio
    async def test_list_realms_escapes_the_prefix(self, sa_client_spr):
        tag = uuid.uuid4().hex[:8]
        names = [f"sa{tag}_1", f"sa{tag}_2", f"sa{tag}X_3"]
        try:
            for n in names:
                await sa_client_spr.create_vertex_table("Note", realm=n)
            assert await sa_client_spr.list_realms(prefix=f"sa{tag}_") == [
                f"sa{tag}_1", f"sa{tag}_2"]
        finally:
            for n in names:
                await sa_client_spr.drop_realm(n)

    @pytest.mark.asyncio
    async def test_a_unique_payload_index(self, sa_client_spr, sa_clean_realm_spr):
        realm = sa_clean_realm_spr
        await sa_client_spr.create_vertex_table("Cred", realm=realm)
        name = await sa_client_spr.create_payload_index(
            "Cred", realm, "username_key", unique=True, scope_by_space=True,
            index_name="uq_sa_cred_user")
        assert name == "uq_sa_cred_user"
        await sa_client_spr.add_vertex("Cred", realm=realm,
                                       payload={"username_key": "oliver"},
                                       space="inst:a")
        await sa_client_spr.add_vertex("Cred", realm=realm,
                                       payload={"username_key": "oliver"},
                                       space="inst:b")
        with pytest.raises(UniqueConstraintError) as exc:
            await sa_client_spr.add_vertex("Cred", realm=realm,
                                           payload={"username_key": "oliver"},
                                           space="inst:a")
        assert exc.value.constraint_name == "uq_sa_cred_user"
        assert isinstance(exc.value, TableExistsError)

    @pytest.mark.asyncio
    async def test_compare_and_swap(self, sa_client_spr, sa_clean_realm_spr):
        realm = sa_clean_realm_spr
        await sa_client_spr.create_vertex_table("Grant", realm=realm)
        v = await sa_client_spr.add_vertex("Grant", realm=realm, payload={"r": 1})
        updated = await sa_client_spr.update_vertex(
            "Grant", realm, v.id, payload={"r": 2},
            expected_updated_at=v.updated_at)
        assert updated.updated_at > v.updated_at
        with pytest.raises(ConcurrentModificationError):
            await sa_client_spr.update_vertex(
                "Grant", realm, v.id, payload={"r": 3},
                expected_updated_at=v.updated_at)
        with pytest.raises(ConcurrentModificationError):
            await sa_client_spr.upsert_vertex(
                "Grant", realm, vertex_id=v.id, payload={"r": 4},
                expected_updated_at=v.updated_at)
        # A missing row is still None, not a lost race.
        assert await sa_client_spr.update_vertex(
            "Grant", realm, 999999, payload={"r": 9},
            expected_updated_at=v.updated_at) is None

    @pytest.mark.asyncio
    async def test_transaction_atomicity_and_actor(self, sa_client_spr,
                                                   sa_clean_realm_spr):
        realm = sa_clean_realm_spr
        await sa_client_spr.create_vertex_table("Item", realm=realm, audited=True)
        async with sa_client_spr.transaction(user_id="usr_alice") as tx:
            v1 = await tx.add_vertex("Item", realm=realm, payload={"k": 1})
            v2 = await tx.add_vertex("Item", realm=realm, payload={"k": 2},
                                     user_id="usr_bob")
            v3 = await tx.add_vertex("Item", realm=realm, payload={"k": 3})

        rows = await sa_client_spr._provision(lambda conn: conn.execute(__import__(
            "sqlalchemy").text(
            f'SELECT changed_by, new_row->>\'id\' AS id '
            f'FROM "{realm}"."Item_audit" WHERE action = \'INSERT\'')))
        actors = {r[1]: r[0] for r in rows.fetchall()}
        assert actors[v1.id] == "usr_alice"
        assert actors[v2.id] == "usr_bob"
        assert actors[v3.id] == "usr_alice"

        with pytest.raises(RuntimeError):
            async with sa_client_spr.transaction(user_id="usr_alice") as tx2:
                await tx2.add_vertex("Item", realm=realm, payload={"k": 4})
                raise RuntimeError("boom")
        assert await sa_client_spr.count_vertices("Item", realm=realm) == 3

    @pytest.mark.asyncio
    async def test_transaction_refuses_nesting_and_poisons_on_exit(
        self, sa_client_spr, sa_clean_realm_spr
    ):
        realm = sa_clean_realm_spr
        await sa_client_spr.create_vertex_table("Item", realm=realm)
        async with sa_client_spr.transaction(user_id="usr_alice") as tx:
            with pytest.raises(NestedTransactionError):
                async with tx.transaction(user_id="usr_alice"):
                    pass
            await tx.add_vertex("Item", realm=realm, payload={"k": 1})
        with pytest.raises(PostGraphError, match="has exited"):
            await tx.add_vertex("Item", realm=realm, payload={"k": 2})
        with pytest.raises(PostGraphError, match="has exited"):
            await tx.find_vertices("Item", realm=realm)

    @pytest.mark.asyncio
    async def test_ensure_database(self):
        from post_graph.client_sqlalchemy import SQLAlchemyPostGraph

        name = f"sa_ensure_{uuid.uuid4().hex[:8]}"
        sa_dsn = DSN.replace("postgresql://", "postgresql+asyncpg://", 1)
        admin = await asyncpg.connect(DSN)
        try:
            resolved = await SQLAlchemyPostGraph.ensure_database(
                sa_dsn, database=name)
            assert resolved.endswith(f"/{name}")
            assert await admin.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = $1", name) == 1
            # Idempotent.
            await SQLAlchemyPostGraph.ensure_database(sa_dsn, database=name)
            with pytest.raises(ValueError):
                await SQLAlchemyPostGraph.ensure_database(
                    sa_dsn, database='x"; DROP DATABASE postgres; --')
        finally:
            await admin.execute(f'DROP DATABASE IF EXISTS "{name}"')
            await admin.close()
