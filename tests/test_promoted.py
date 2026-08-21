"""Promoted payload columns: correctness first, speed second.

The whole point of promotion is that it changes the plan without changing the
answer, so most of these tests assert *parity* — a promoted table and an
unpromoted one must return the same rows for the same query. A promotion that
is merely fast is a bug.
"""
import pytest

from post_graph import promoted as pr
from conftest import requires_pg


class TestNames:
    def test_column_names(self):
        assert pr.temporal_column('valid_from') == 'pt_valid_from'
        assert pr.generic_column('t_expired') == 'p_t_expired'

    @pytest.mark.parametrize("bad", ["", "no-hyphens", "1leading", "semi;colon", None, 7])
    def test_rejects_unsafe_keys(self, bad):
        # Payload keys become column names, so anything that is not a plain
        # identifier has to be refused rather than quoted and hoped for.
        with pytest.raises(ValueError):
            pr.validate_key(bad)

    def test_ddl_is_idempotent_in_form(self):
        stmts = pr.column_ddl('public.t', 't', 'valid_from', temporal=True)
        assert any('ADD COLUMN IF NOT EXISTS' in s for s in stmts)
        assert any('CREATE INDEX IF NOT EXISTS' in s for s in stmts)

    def test_defaults_cover_the_temporal_pair(self):
        ddl = "\n".join(pr.all_column_ddl('public.t', 't'))
        assert 'pt_valid_from' in ddl and 'pt_valid_to' in ddl


@requires_pg
class TestGeneratedValues:
    @pytest.mark.asyncio(loop_scope="session")
    async def test_partial_dates_normalise(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("pv", realm=realm)
        await pg_client.create_edge_table("pe", from_vertex_table="pv",
                                          to_vertex_table="pv", realm=realm,
                                          promoted_keys=["t_expired"])
        a = await pg_client.add_vertex("pv", realm, payload={})
        b = await pg_client.add_vertex("pv", realm, payload={})
        for vf in ("2024", "2024-06", "2024-06-15"):
            await pg_client.add_edge("pe", realm, a.id, b.id, "R", payload={"valid_from": vf})
        rows = await pg_client._fetch(
            f"SELECT payload->>'valid_from' AS raw, pt_valid_from FROM "
            f"{pg_client._get_table_ref('pe', realm)} WHERE realm = $1 ORDER BY id", realm)
        got = {r['raw']: r['pt_valid_from'] for r in rows}
        # A year and a year-month must land on a real date, or they sort as
        # shorter strings and compare wrongly against a full date.
        assert got == {"2024": "2024-01-01", "2024-06": "2024-06-01",
                       "2024-06-15": "2024-06-15"}

    @pytest.mark.asyncio(loop_scope="session")
    async def test_absent_key_is_null_not_empty(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("pv2", realm=realm)
        await pg_client.create_edge_table("pe2", from_vertex_table="pv2",
                                          to_vertex_table="pv2", realm=realm)
        a = await pg_client.add_vertex("pv2", realm, payload={})
        b = await pg_client.add_vertex("pv2", realm, payload={})
        await pg_client.add_edge("pe2", realm, a.id, b.id, "R", payload={})
        row = await pg_client._fetchrow(
            f"SELECT pt_valid_from FROM {pg_client._get_table_ref('pe2', realm)} WHERE realm = $1", realm)
        # NULL means 'no stated period', which the as-of filter treats as
        # always-valid. '--' would be a date that parses and is always false.
        assert row['pt_valid_from'] is None

    @pytest.mark.asyncio(loop_scope="session")
    async def test_column_tracks_payload_updates(self, pg_client, clean_realm):
        realm = clean_realm
        await pg_client.create_vertex_table("pv3", realm=realm)
        await pg_client.create_edge_table("pe3", from_vertex_table="pv3",
                                          to_vertex_table="pv3", realm=realm)
        a = await pg_client.add_vertex("pv3", realm, payload={})
        b = await pg_client.add_vertex("pv3", realm, payload={})
        e = await pg_client.add_edge("pe3", realm, a.id, b.id, "R", payload={"valid_from": "2020"})
        await pg_client.upsert_edge("pe3", realm, a.id, b.id, "R",
                                    edge_id=e.id, payload={"valid_from": "2030"})
        row = await pg_client._fetchrow(
            f"SELECT pt_valid_from FROM {pg_client._get_table_ref('pe3', realm)} WHERE realm = $1 AND id = $2",
            realm, int(e.id))
        # Generated columns are maintained by PostgreSQL, so an update through
        # any path keeps them correct without the client doing anything.
        assert row['pt_valid_from'] == "2030-01-01"


@requires_pg
class TestParityWithUnpromoted:
    """Promotion must change the plan, never the answer.

    A table whose promoted columns are dropped stands in for a database created
    before this feature existed. Both must agree on every query, or upgrading
    silently changes results.
    """

    @staticmethod
    async def _build(client, realm, suffix, drop_promoted):
        v, e = f"parv{suffix}", f"pare{suffix}"
        await client.create_vertex_table(v, realm=realm)
        await client.create_edge_table(e, from_vertex_table=v, to_vertex_table=v,
                                       realm=realm, promoted_keys=["t_expired"])
        if drop_promoted:
            ref = client._get_table_ref(e, realm)
            for col in ("pt_valid_from", "pt_valid_to", "p_t_expired"):
                await client._execute(f'ALTER TABLE {ref} DROP COLUMN IF EXISTS "{col}"')
            client._promoted_cache.pop((realm, e), None)
        a = await client.add_vertex(v, realm, payload={"n": "a"})
        rows = []
        for payload in (
            {"valid_from": "2020", "valid_to": "2021"},
            {"valid_from": "2024-06"},
            {"valid_to": "2019-12-31"},
            {},                                   # no period: valid at every date
            {"valid_from": "2022", "t_expired": "2023-01-01"},
        ):
            t = await client.add_vertex(v, realm, payload={"n": "t"})
            await client.add_edge(e, realm, a.id, t.id, "R", payload=payload)
            rows.append(t)
        return v, e, a

    @pytest.mark.asyncio(loop_scope="session")
    @pytest.mark.parametrize("as_of", [None, "2019-01-01", "2020-06-15", "2022-05-05",
                                       "2024-07-01", "2030-01-01"])
    async def test_as_of_parity(self, pg_client, clean_realm, as_of):
        realm = clean_realm
        v1, e1, a1 = await self._build(pg_client, realm, "p", drop_promoted=False)
        v2, e2, a2 = await self._build(pg_client, realm, "u", drop_promoted=True)

        promoted = await pg_client.traverse(realm=realm, start_table=v1, start_id=a1.id,
                                            edge_tables=[e1], max_depth=1, as_of=as_of)
        legacy = await pg_client.traverse(realm=realm, start_table=v2, start_id=a2.id,
                                          edge_tables=[e2], max_depth=1, as_of=as_of)
        assert len(promoted) == len(legacy), f"as_of={as_of}: {len(promoted)} vs {len(legacy)}"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_payload_null_keys_parity(self, pg_client, clean_realm):
        realm = clean_realm
        v1, e1, a1 = await self._build(pg_client, realm, "np", drop_promoted=False)
        v2, e2, a2 = await self._build(pg_client, realm, "nu", drop_promoted=True)
        promoted = await pg_client.traverse(realm=realm, start_table=v1, start_id=a1.id,
                                            edge_tables=[e1], max_depth=1,
                                            payload_null_keys=["t_expired"])
        legacy = await pg_client.traverse(realm=realm, start_table=v2, start_id=a2.id,
                                          edge_tables=[e2], max_depth=1,
                                          payload_null_keys=["t_expired"])
        assert len(promoted) == len(legacy)
        # The edge carrying t_expired must be excluded by both.
        assert len(promoted) < 6

    @pytest.mark.asyncio(loop_scope="session")
    async def test_legacy_table_still_traversable(self, pg_client, clean_realm):
        """Dropping the columns must degrade to the payload expression, not error."""
        realm = clean_realm
        v, e, a = await self._build(pg_client, realm, "lg", drop_promoted=True)
        cols = await pg_client._promoted_columns(e, realm)
        assert cols == set()
        res = await pg_client.traverse(realm=realm, start_table=v, start_id=a.id,
                                       edge_tables=[e], max_depth=1, as_of="2020-06-15")
        assert res
