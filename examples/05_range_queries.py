"""Filter, order, count and bulk-delete on JSONB fields — in the database.

Payload filtering used to mean equality only, so anything else meant pulling
rows into Python and discarding most of them. These predicates compile to SQL
and run where the data is.

The comparison rule is worth knowing: numbers compare numerically, strings
compare as text. That is deliberate — zero-padded timestamps and version
strings sort correctly as text, and would not if everything were coerced to a
number.
"""
import asyncio

from _shared import DSN, banner, fresh_realm
from post_graph import AsyncPostGraph


async def main():
    client = AsyncPostGraph(dsn=DSN)
    await client.connect()
    realm = fresh_realm("ex_range")
    try:
        await client.create_vertex_table("readings", realm=realm)
        for i, (sensor, temp, ts) in enumerate([
                ("s1", 18.5, "2026-01-02T09:00:00Z"),
                ("s1", 22.1, "2026-01-02T12:00:00Z"),
                ("s2", 31.7, "2026-01-02T12:05:00Z"),
                ("s2", 12.0, "2026-01-03T08:30:00Z"),
                ("s3", 27.4, "2026-01-03T15:45:00Z")]):
            await client.add_vertex("readings", realm=realm,
                                    payload={"sensor": sensor, "temp": temp, "ts": ts})

        banner("Numeric range: temp > 20")
        for v in await client.find_vertices("readings", realm=realm,
                                            where=[("temp", ">", 20)]):
            print(f"  {v.payload['sensor']}  {v.payload['temp']}")

        banner("Ordered, descending, limited")
        for v in await client.find_vertices("readings", realm=realm,
                                            order_by="temp", descending=True, limit=3):
            print(f"  {v.payload['temp']}  {v.payload['sensor']}")

        banner("Strings compare as text, so ISO timestamps sort correctly")
        for v in await client.find_vertices(
                "readings", realm=realm,
                where=[("ts", ">=", "2026-01-03T00:00:00Z")], order_by="ts"):
            print(f"  {v.payload['ts']}  {v.payload['sensor']}")

        banner("Counting without transferring rows")
        n = await client.count_vertices("readings", realm=realm,
                                        where=[("temp", "<", 20)])
        print(f"  readings below 20 degrees: {n}")

        banner("An index for the predicate you actually run")
        await client.create_payload_index("readings", realm=realm, key="temp",
                                          numeric=True)
        print("  numeric expression index created on temp")

        banner("Bulk delete, which refuses an empty predicate")
        removed = await client.delete_vertices("readings", realm=realm,
                                               where=[("temp", "<", 15)])
        print(f"  deleted {removed}; remaining "
              f"{await client.count_vertices('readings', realm=realm)}")
        try:
            await client.delete_vertices("readings", realm=realm, where=[])
        except ValueError as e:
            print(f"  empty predicate refused: {e}")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
