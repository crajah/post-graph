"""A duplicate vertex id is answered by this library's error, not the driver's.

``add_vertex`` documents that it "Raises TableExistsError if it already
exists", and it has a handler that says so. For four releases the handler was
unreachable for the case it names: the ``INSERT`` -- the only statement that
can raise a unique violation on ``(realm, id)`` -- sat *above* the ``try``, so
a duplicate id escaped as a raw ``asyncpg.UniqueViolationError``.

The SQLAlchemy client never had the bug, which is how the two clients came to
disagree about one method's documented contract while every test passed.

Why it matters more than tidiness: ``add_vertex`` is the write a caller uses
when ``upsert_vertex`` is wrong -- when "the row already exists" must be
refused rather than merged, and when the role deliberately lacks the ``UPDATE``
that ``ON CONFLICT DO UPDATE`` demands. A sign-in path that creates an account
row on first use is exactly that, and its duplicate case is reachable by an
ordinary user action: a magic link opened in two tabs. Both requests read, both
find nothing, both write, and the loser should get a refusal rather than a
driver exception nothing upstream recognises.
"""

import ast
import inspect
import textwrap
import uuid

import asyncpg
import pytest
from conftest import DSN

from post_graph import AsyncPostGraph
from post_graph.errors import TableExistsError, TableNotFoundError


class TestTheLibrarysOwnError:
    """The reason the fix exists."""

    @pytest.mark.asyncio
    async def test_duplicate_id_raises_table_exists_error(self, pg_client, clean_realm):
        """The headline: the library's type, naming the id, the table and the realm."""
        await pg_client.create_vertex_table("accounts", realm=clean_realm)
        await pg_client.add_vertex("accounts", realm=clean_realm, vertex_id=4242,
                                   payload={"email": "first@example.com"})

        with pytest.raises(TableExistsError) as exc:
            await pg_client.add_vertex("accounts", realm=clean_realm, vertex_id=4242,
                                       payload={"email": "second@example.com"})

        message = str(exc.value)
        assert "4242" in message
        assert "accounts" in message
        assert clean_realm in message

    @pytest.mark.asyncio
    async def test_no_asyncpg_error_escapes(self, pg_client, clean_realm):
        """Not an instance check on the happy type: nothing from the driver gets out.

        ``TableExistsError`` is not a subclass of any asyncpg exception, so
        this fails if the raw error escapes even should some other handler
        start producing a differently-named library error.
        """
        await pg_client.create_vertex_table("accounts", realm=clean_realm)
        await pg_client.add_vertex("accounts", realm=clean_realm, vertex_id=1,
                                   payload={})

        try:
            await pg_client.add_vertex("accounts", realm=clean_realm, vertex_id=1,
                                       payload={})
        except Exception as e:                                   # noqa: BLE001
            assert not isinstance(e, asyncpg.PostgresError), (
                f"a driver exception escaped add_vertex: {type(e).__name__}: {e}")
            assert isinstance(e, TableExistsError)
        else:
            pytest.fail("the duplicate insert did not raise")

    @pytest.mark.asyncio
    async def test_the_two_tabs_race(self, clean_realm):
        """The real shape: the row arrives on a second connection mid-flight.

        A single-connection test cannot produce this. Here the caller reads,
        finds nothing, and by the time it writes another connection has
        written the same id -- which is what two tabs on one magic link do.
        """
        owner = AsyncPostGraph(dsn=DSN)
        other = AsyncPostGraph(dsn=DSN)
        await owner.connect()
        await other.connect()
        try:
            await owner.create_vertex_table("accounts", realm=clean_realm)

            # The caller's read: nobody holds this id.
            assert await owner.get_vertex("accounts", clean_realm, "77") is None

            # The other tab wins the race, on its own connection and its own
            # transaction, and commits.
            await other.add_vertex("accounts", realm=clean_realm, vertex_id=77,
                                   payload={"email": "tab-two@example.com"})

            # The caller writes anyway, as it must -- its read said the id was
            # free. It should be refused in the library's own terms.
            with pytest.raises(TableExistsError):
                await owner.add_vertex("accounts", realm=clean_realm, vertex_id=77,
                                       payload={"email": "tab-one@example.com"})

            # And the winner's row is untouched: a refusal, not a merge.
            row = await owner.get_vertex("accounts", clean_realm, "77")
            assert row.payload["email"] == "tab-two@example.com"
        finally:
            await owner.close()
            await other.close()

    @pytest.mark.asyncio
    async def test_a_missing_table_also_raises_the_librarys_error(self, pg_client, realm):
        """The sibling handler, also unreachable before the fix.

        ``pg_get_serial_sequence`` on an absent table raises
        ``UndefinedTableError`` from the *first* statement in the method, which
        was likewise above the try. So this failed as a driver error too, for
        both the supplied-id and the sequence-id branch.
        """
        with pytest.raises(TableNotFoundError):
            await pg_client.add_vertex("no_such_table_" + uuid.uuid4().hex[:8],
                                       realm=realm, payload={})


class TestUpsertIsUnchanged:
    """The sibling with ``ON CONFLICT DO UPDATE`` has no violation to raise."""

    @pytest.mark.asyncio
    async def test_upsert_merges_and_returns_the_row(self, pg_client, clean_realm):
        await pg_client.create_vertex_table("accounts", realm=clean_realm)
        await pg_client.upsert_vertex("accounts", realm=clean_realm, vertex_id=9,
                                      payload={"email": "a@example.com", "tier": "free"})
        v = await pg_client.upsert_vertex("accounts", realm=clean_realm, vertex_id=9,
                                          payload={"tier": "paid"})
        assert v is not None
        assert v.id == "9"
        # Merged, not replaced -- the long-standing contract.
        assert v.payload == {"email": "a@example.com", "tier": "paid"}


class TestItStaysFixed:
    """A structural guard, so the statement cannot drift back out.

    The behavioural tests above pass whenever the handler is reachable, but
    they cannot say *why* it was unreachable, and the regression is a two-line
    move that no reviewer would notice. This reads the method itself.
    """

    def test_every_statement_in_add_vertex_is_inside_the_try(self):
        source = textwrap.dedent(inspect.getsource(AsyncPostGraph.add_vertex))
        op = next(n for n in ast.walk(ast.parse(source))
                  if isinstance(n, ast.AsyncFunctionDef) and n.name == "_op")

        outside = []
        for stmt in op.body:
            if isinstance(stmt, ast.Try):
                continue
            if any(isinstance(n, ast.Await) for n in ast.walk(stmt)):
                outside.append(ast.unparse(stmt).splitlines()[0][:90])

        assert not outside, (
            "these awaited statements in add_vertex sit outside the try that "
            "translates asyncpg's errors, so the error they raise escapes as "
            f"the driver's own type: {outside}")

    def test_the_try_still_translates_both_errors(self):
        """Guards the other half: a try covering everything but handling nothing."""
        source = textwrap.dedent(inspect.getsource(AsyncPostGraph.add_vertex))
        op = next(n for n in ast.walk(ast.parse(source))
                  if isinstance(n, ast.AsyncFunctionDef) and n.name == "_op")
        tries = [n for n in op.body if isinstance(n, ast.Try)]
        assert len(tries) == 1
        caught = {ast.unparse(h.type) for h in tries[0].handlers}
        assert caught == {"asyncpg.UniqueViolationError", "asyncpg.UndefinedTableError"}


class TestSQLAlchemyParity:
    """The client that never had the bug must keep not having it."""

    @pytest.mark.asyncio
    async def test_duplicate_id_raises_table_exists_error(self, sa_client, sa_clean_realm):
        await sa_client.create_vertex_table("accounts", realm=sa_clean_realm)
        await sa_client.add_vertex("accounts", realm=sa_clean_realm, vertex_id=5,
                                   payload={"email": "a@example.com"})
        with pytest.raises(TableExistsError) as exc:
            await sa_client.add_vertex("accounts", realm=sa_clean_realm, vertex_id=5,
                                       payload={"email": "b@example.com"})
        assert "5" in str(exc.value)
