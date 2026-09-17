"""Auditing is a choice: a table can decline it, and its record can be removed.

Every table previously got a shadow audit table and a trigger whether or not
anything would ever read them, and once created there was no way to remove one.
That is wrong in both directions: a high-churn queue accumulates an audit table
larger than the data it shadows, and a table that no longer needs auditing kept
recording for ever.

Removing the record stays an explicit call. Nothing here drops an audit table as
a side effect of something else -- not creating a table unaudited, not deleting
a realm's rows.
"""


def _t(realm, base):
    """A table name unique to this test.

    The default client runs realm-as-column, so every realm shares one set of
    physical tables. Without a unique name a test that creates `knows`
    unaudited would see the audit table left by whichever test created `knows`
    first, and the collision looks exactly like the feature not working.
    """
    return f"{base}_{realm[-8:]}"


async def _audit_exists(client, realm, table):
    ref = client._get_table_ref(f"{table}_audit", realm)

    async def _op(conn):
        try:                                    # asyncpg
            return await conn.fetchval("SELECT to_regclass($1) IS NOT NULL", ref)
        except AttributeError:                  # SQLAlchemy connection
            from sqlalchemy import text
            res = await conn.execute(text("SELECT to_regclass(:r) IS NOT NULL"), {"r": ref})
            return res.scalar()

    return bool(await client._run_in_tx(_op))


async def _audit_rows(client, realm, table):
    ref = client._get_table_ref(f"{table}_audit", realm)

    async def _op(conn):
        try:                                    # asyncpg
            return await conn.fetchval(f"SELECT count(*) FROM {ref} WHERE realm = $1", realm)
        except AttributeError:                  # SQLAlchemy connection
            from sqlalchemy import text
            res = await conn.execute(
                text(f"SELECT count(*) FROM {ref} WHERE realm = :r"), {"r": realm})
            return res.scalar()

    return int(await client._run_in_tx(_op))


class TestUnauditedTables:

    async def test_audited_is_the_default(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table(_t(realm, "people"), realm=realm)
        assert await _audit_exists(pg_client, realm, _t(realm, "people"))
        await pg_client.add_vertex(_t(realm, "people"), realm=realm, payload={"n": "a"})
        assert await _audit_rows(pg_client, realm, _t(realm, "people")) == 1

    async def test_unaudited_table_creates_no_audit_table(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table(_t(realm, "cache"), realm=realm, audited=False)
        assert not await _audit_exists(pg_client, realm, _t(realm, "cache"))

        # The table itself must be entirely usable.
        v = await pg_client.add_vertex(_t(realm, "cache"), realm=realm, payload={"n": "a"})
        v = await pg_client.upsert_vertex(_t(realm, "cache"), realm=realm, vertex_id=v.id,
                                          payload={"n": "b"})
        assert v.payload["n"] == "b"
        got = await pg_client.get_vertex(_t(realm, "cache"), realm=realm, vertex_id=v.id)
        assert got.payload["n"] == "b"

    async def test_unaudited_edge_table(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table(_t(realm, "people"), realm=realm)
        await pg_client.create_edge_table(_t(realm, "knows"), from_vertex_table=_t(realm, "people"),
                                          to_vertex_table=_t(realm, "people"), realm=realm,
                                          audited=False)
        assert not await _audit_exists(pg_client, realm, _t(realm, "knows"))
        a = await pg_client.add_vertex(_t(realm, "people"), realm=realm, payload={"n": "a"})
        b = await pg_client.add_vertex(_t(realm, "people"), realm=realm, payload={"n": "b"})
        e = await pg_client.add_edge(_t(realm, "knows"), realm=realm, from_id=a.id, to_id=b.id,
                                     relation_type=_t(realm, "knows"), payload={})
        assert e.id

    async def test_redeclaring_unaudited_stops_recording_but_keeps_the_record(
            self, pg_client, clean_realm):
        """Declaring a table unaudited must actually take effect on an existing
        table -- but it must not discard what was already recorded."""
        realm = clean_realm
        await pg_client.create_vertex_table(_t(realm, "people"), realm=realm)
        await pg_client.add_vertex(_t(realm, "people"), realm=realm, payload={"n": "a"})
        assert await _audit_rows(pg_client, realm, _t(realm, "people")) == 1

        await pg_client.create_vertex_table(_t(realm, "people"), realm=realm, audited=False)
        await pg_client.add_vertex(_t(realm, "people"), realm=realm, payload={"n": "b"})

        assert await _audit_exists(pg_client, realm, _t(realm, "people")), "record must survive"
        assert await _audit_rows(pg_client, realm, _t(realm, "people")) == 1, "recording must stop"

    async def test_auditing_can_be_turned_back_on(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table(_t(realm, "people"), realm=realm, audited=False)
        await pg_client.add_vertex(_t(realm, "people"), realm=realm, payload={"n": "a"})

        await pg_client.create_vertex_table(_t(realm, "people"), realm=realm, audited=True)
        await pg_client.add_vertex(_t(realm, "people"), realm=realm, payload={"n": "b"})
        assert await _audit_rows(pg_client, realm, _t(realm, "people")) == 1


class TestDropAuditTable:

    async def test_drop_removes_the_table_and_stops_recording(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table(_t(realm, "people"), realm=realm)
        await pg_client.add_vertex(_t(realm, "people"), realm=realm, payload={"n": "a"})
        assert await _audit_rows(pg_client, realm, _t(realm, "people")) == 1

        assert await pg_client.drop_audit_table(_t(realm, "people"), realm=realm) is True
        assert not await _audit_exists(pg_client, realm, _t(realm, "people"))

        # The base table must remain fully writable with no audit target.
        v = await pg_client.add_vertex(_t(realm, "people"), realm=realm, payload={"n": "b"})
        await pg_client.upsert_vertex(_t(realm, "people"), realm=realm, vertex_id=v.id,
                                      payload={"n": "c"})
        await pg_client.delete_vertex(_t(realm, "people"), realm=realm, vertex_id=v.id)

    async def test_drop_is_idempotent(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table(_t(realm, "people"), realm=realm)
        assert await pg_client.drop_audit_table(_t(realm, "people"), realm=realm) is True
        assert await pg_client.drop_audit_table(_t(realm, "people"), realm=realm) is False

    async def test_drop_on_an_unaudited_table_is_a_no_op(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table(_t(realm, "cache"), realm=realm, audited=False)
        assert await pg_client.drop_audit_table(_t(realm, "cache"), realm=realm) is False

    async def test_drop_works_for_edge_tables(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table(_t(realm, "people"), realm=realm)
        await pg_client.create_edge_table(_t(realm, "knows"), from_vertex_table=_t(realm, "people"),
                                          to_vertex_table=_t(realm, "people"), realm=realm)
        assert await pg_client.drop_audit_table(_t(realm, "knows"), realm=realm) is True
        a = await pg_client.add_vertex(_t(realm, "people"), realm=realm, payload={"n": "a"})
        b = await pg_client.add_vertex(_t(realm, "people"), realm=realm, payload={"n": "b"})
        assert await pg_client.add_edge(_t(realm, "knows"), realm=realm, from_id=a.id, to_id=b.id,
                                        relation_type=_t(realm, "knows"), payload={})

    async def test_dropping_one_table_audit_leaves_others_alone(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table(_t(realm, "people"), realm=realm)
        await pg_client.create_vertex_table(_t(realm, "places"), realm=realm)
        await pg_client.drop_audit_table(_t(realm, "people"), realm=realm)

        assert await _audit_exists(pg_client, realm, _t(realm, "places"))
        await pg_client.add_vertex(_t(realm, "places"), realm=realm, payload={"n": "x"})
        assert await _audit_rows(pg_client, realm, _t(realm, "places")) == 1

    async def test_recreating_restores_auditing(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table(_t(realm, "people"), realm=realm)
        await pg_client.add_vertex(_t(realm, "people"), realm=realm, payload={"n": "a"})
        await pg_client.drop_audit_table(_t(realm, "people"), realm=realm)

        await pg_client.create_vertex_table(_t(realm, "people"), realm=realm, audited=True)
        await pg_client.add_vertex(_t(realm, "people"), realm=realm, payload={"n": "b"})
        assert await _audit_rows(pg_client, realm, _t(realm, "people")) == 1, "discarded rows stay gone"


class TestSQLAlchemyParity:
    """The second backend must behave identically; it is a supported client,
    not a mirror that is allowed to drift."""

    async def test_unaudited_and_drop(self, sa_client, sa_clean_realm):
        realm = sa_clean_realm
        await sa_client.create_vertex_table(_t(realm, "sa_cache"), realm=realm, audited=False)
        assert not await _audit_exists(sa_client, realm, _t(realm, "sa_cache"))
        assert await sa_client.drop_audit_table(_t(realm, "sa_cache"), realm=realm) is False

        await sa_client.create_vertex_table(_t(realm, "sa_people"), realm=realm)
        assert await _audit_exists(sa_client, realm, _t(realm, "sa_people"))
        await sa_client.add_vertex(_t(realm, "sa_people"), realm=realm, payload={"n": "a"})
        assert await _audit_rows(sa_client, realm, _t(realm, "sa_people")) == 1

        assert await sa_client.drop_audit_table(_t(realm, "sa_people"), realm=realm) is True
        assert not await _audit_exists(sa_client, realm, _t(realm, "sa_people"))
        await sa_client.add_vertex(_t(realm, "sa_people"), realm=realm, payload={"n": "b"})
