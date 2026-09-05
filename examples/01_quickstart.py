"""Vertices, edges and a traversal — the whole model in one file.

post-graph maps a graph onto ordinary tables: one table per vertex type, one
per edge type. That means foreign keys, indexes and constraints are real, and
anything that speaks SQL can read your graph.
"""
import asyncio

from _shared import DSN, banner, fresh_realm
from post_graph import AsyncPostGraph


async def main():
    client = AsyncPostGraph(dsn=DSN)
    await client.connect()
    realm = fresh_realm("ex_quickstart")
    try:
        await client.create_vertex_table("people", realm=realm)
        await client.create_edge_table("knows", from_vertex_table="people",
                                       to_vertex_table="people", realm=realm)

        ada = await client.add_vertex("people", realm=realm,
                                      payload={"name": "Ada", "role": "engineer"})
        bob = await client.add_vertex("people", realm=realm,
                                      payload={"name": "Bob", "role": "analyst"})
        cai = await client.add_vertex("people", realm=realm,
                                      payload={"name": "Cai", "role": "designer"})
        await client.add_edge("knows", realm=realm, from_id=ada.id, to_id=bob.id,
                              relation_type="colleague")
        await client.add_edge("knows", realm=realm, from_id=bob.id, to_id=cai.id,
                              relation_type="colleague")

        banner("Every vertex carries a generated fqid")
        for v in await client.find_vertices("people", realm=realm, filters={}):
            print(f"  {v.fqid}  {v.payload}")

        banner("One hop from Ada")
        for step in await ada.to("knows"):
            print(f"  Ada --[{step.edge.relation_type}]--> "
                  f"{step.vertex().payload['name']}")

        banner("Two hops, in the database, via a recursive CTE")
        reached = await client.traverse(realm=realm, start_table="people",
                                        start_id=ada.id, edge_tables=["knows"],
                                        max_depth=2)
        # traverse returns ids and the path taken, not payloads: the walk
        # happens in the database and stays cheap regardless of row width.
        for r in reached:
            print(f"  depth {r['depth']}: {r['table_name']}:{r['id']} "
                  f"via {' -> '.join(r['path'])}")

        banner("Shortest path Ada -> Cai")
        path = await client.shortest_path(realm=realm, start_table="people",
                                          start_id=ada.id, target_table="people",
                                          target_id=cai.id, edge_tables=["knows"])
        print(f"  {path}")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
