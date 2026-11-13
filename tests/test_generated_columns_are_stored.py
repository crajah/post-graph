"""Generated columns must be STORED, and PostgreSQL 18 made that worth testing.

Until PostgreSQL 17, `GENERATED ALWAYS AS (expr)` without a storage keyword was
a syntax error: STORED was the only option and had to be written. PostgreSQL 18
added virtual generated columns and made VIRTUAL the default when the keyword
is omitted -- so the same DDL that used to fail loudly now succeeds and means
something different.

A virtual column cannot be indexed. That is survivable for `fqid`, which is
read rather than searched, and fatal for a promoted payload column, whose only
reason to exist is to carry an index: every expression index built over it
would fail, and the temporal traversal it was added for would go back to the
sequential scan it was measured to replace -- silently, on PostgreSQL 18 only.

Every generated column in this library writes STORED explicitly, so none of
that happens. These tests are here so it keeps being true, on both versions.
"""

import re
import uuid
from pathlib import Path

import pytest

SOURCE_FILES = ("post_graph/client_asyncpg.py",
                "post_graph/client_sqlalchemy.py",
                "post_graph/promoted.py")


def _storage(value) -> str:
    """`pg_attribute.attgenerated` is PostgreSQL's one-byte "char" type, which
    asyncpg hands back as bytes. 's' is stored, 'v' is virtual, '' is neither."""
    return value.decode() if isinstance(value, (bytes, bytearray)) else (value or "")



class TestTheSourceSaysStored:
    """A grep, because the failure is a missing word rather than wrong logic."""

    def test_every_generated_column_declares_its_storage(self):
        root = Path(__file__).resolve().parent.parent
        offenders = []
        for name in SOURCE_FILES:
            for lineno, line in enumerate(
                    (root / name).read_text().splitlines(), start=1):
                if "GENERATED ALWAYS AS" not in line:
                    continue
                # The keyword may be on this line or the clause may continue;
                # these are all single-line declarations today.
                if not re.search(r"STORED|VIRTUAL", line):
                    offenders.append(f"{name}:{lineno}")
        assert not offenders, (
            "these generated columns name no storage, so they are STORED on "
            "PostgreSQL 17 and earlier (an error, in fact) and VIRTUAL on 18 "
            f"and later, where they cannot be indexed: {offenders}")

    def test_none_of_them_are_virtual(self):
        root = Path(__file__).resolve().parent.parent
        virtual = [name for name in SOURCE_FILES
                   if "VIRTUAL" in (root / name).read_text()]
        assert not virtual, (
            f"a virtual generated column cannot be indexed, which is the "
            f"whole purpose of the promoted columns: {virtual}")


class TestTheDatabaseAgrees:
    """The behavioural half, which is what the PostgreSQL 18 CI job is for.

    ``pg_attribute.attgenerated`` is 's' for stored and 'v' for virtual.
    """

    @pytest.mark.asyncio
    async def test_a_vertex_tables_fqid_is_stored(self, pg_client_spr,
                                                  clean_realm_spr):
        realm = clean_realm_spr
        await pg_client_spr.create_vertex_table("Thing", realm=realm)
        row = await pg_client_spr._fetchrow(
            "SELECT attgenerated FROM pg_attribute "
            "WHERE attrelid = $1::regclass AND attname = 'fqid'",
            f'"{realm}"."Thing"')
        assert _storage(row["attgenerated"]) == "s", "fqid must be stored"

    @pytest.mark.asyncio
    async def test_an_edge_tables_fqid_is_stored(self, pg_client_spr,
                                                 clean_realm_spr):
        realm = clean_realm_spr
        await pg_client_spr.create_vertex_table("Node", realm=realm)
        await pg_client_spr.create_edge_table(
            "Link", from_vertex_table="Node", to_vertex_table="Node",
            realm=realm)
        row = await pg_client_spr._fetchrow(
            "SELECT attgenerated FROM pg_attribute "
            "WHERE attrelid = $1::regclass AND attname = 'fqid'",
            f'"{realm}"."Link"')
        assert _storage(row["attgenerated"]) == "s", "fqid must be stored"

    @pytest.mark.asyncio
    async def test_a_promoted_column_is_stored_and_indexable(
        self, pg_client_spr, clean_realm_spr
    ):
        """The one that would actually break. A promoted column exists to be
        indexed, so if it came out virtual the index would fail rather than
        the column."""
        realm = clean_realm_spr
        await pg_client_spr.create_vertex_table(
            "Event", realm=realm, promoted_keys=("valid_from",))
        rows = await pg_client_spr._fetch(
            "SELECT attname, attgenerated FROM pg_attribute "
            "WHERE attrelid = $1::regclass AND attgenerated <> ''",
            f'"{realm}"."Event"')
        assert rows, "no generated columns were created"
        promoted = [r for r in rows if r["attname"] != "fqid"]
        assert promoted, "no promoted column was created"
        for r in rows:
            assert _storage(r["attgenerated"]) == "s", (
                r["attname"], _storage(r["attgenerated"]))

        # And the thing the column is for: an index over it builds.
        idx = f"idx_prom_{uuid.uuid4().hex[:8]}"
        await pg_client_spr._execute(
            f'CREATE INDEX "{idx}" ON "{realm}"."Event" (pt_valid_from)')
        present = await pg_client_spr._fetchrow(
            "SELECT to_regclass($1) AS oid", f'"{realm}"."{idx}"')
        assert present["oid"] is not None
