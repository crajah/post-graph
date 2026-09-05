"""Two kinds of memory: append-only history, and a shadow audit log.

Every vertex and edge table gets a `_data` table for timestamped payload
versions you write deliberately, and an `_audit` table populated by triggers
that records every insert, update and delete whether you meant it or not.

The distinction matters. History is what you chose to keep. The audit log is
what happened, including the things you would rather had not.
"""
import asyncio

from _shared import DSN, banner, fresh_realm
from post_graph import AsyncPostGraph


async def main():
    client = AsyncPostGraph(dsn=DSN)
    await client.connect()
    realm = fresh_realm("ex_history")
    try:
        await client.create_vertex_table("accounts", realm=realm)
        acct = await client.add_vertex("accounts", realm=realm,
                                       payload={"name": "Acme", "tier": "free"},
                                       user_id="alice")

        banner("Append-only history: versions you record on purpose")
        await acct.add_data({"tier": "free", "seats": 1})
        await acct.add_data({"tier": "pro", "seats": 25})
        await acct.add_data({"tier": "enterprise", "seats": 400})
        for rec in await acct.get_data():
            print(f"  {rec.timestamp:%Y-%m-%d %H:%M:%S}  {rec.payload}")

        banner("Updating the vertex itself, as a different user")
        await client.upsert_vertex("accounts", realm=realm, vertex_id=acct.id,
                                   payload={"name": "Acme", "tier": "enterprise"},
                                   user_id="bob")
        current = await client.get_vertex("accounts", realm=realm, vertex_id=acct.id)
        print(f"  now: {current.payload}")

        banner("Audit log: what happened, and who did it")
        # The audit table is an ordinary table written by triggers, so it is
        # read with SQL rather than through an API of its own.
        ref = client._get_table_ref("accounts_audit", realm)
        for row in await client._fetch(
                f"SELECT action, changed_by, changed_at FROM {ref} ORDER BY audit_id"):
            print(f"  {row['action']:<7} by {row['changed_by'] or '-':<6} "
                  f"at {row['changed_at']:%H:%M:%S}")
        print("\n  The insert is attributed to alice and the update to bob, taken")
        print("  from a session variable by the trigger rather than from anything")
        print("  the caller wrote into the row.")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
