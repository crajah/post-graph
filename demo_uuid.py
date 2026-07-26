"""Demo script testing automatic UUID creation and lookup in post-graph."""
import asyncio
import os
from post_graph import AsyncPostGraph

DB_URI = os.getenv("POSTGRES_URI", "postgresql://crajah@localhost:5432/postgres")

async def main():
    print("=" * 60)
    print("POST-GRAPH AUTOMATIC UUID DEMO")
    print("=" * 60)

    client = AsyncPostGraph(dsn=DB_URI)
    await client.connect()
    realm = "uuid_demo_realm"

    print("\n[+] Creating vertex and edge tables...")
    await client.create_vertex_table("users", realm=realm)
    await client.create_vertex_table("posts", realm=realm)
    await client.create_edge_table("authored", from_vertex_table="users", to_vertex_table="posts", realm=realm)

    print("\n[+] Adding vertex 'Alice'...")
    alice = await client.add_vertex("users", realm=realm, payload={"name": "Alice", "role": "author"})
    print(f"    Alice ID: {alice.id}")
    print(f"    Alice FQID: {alice.fqid}")
    print(f"    Alice Automatically Assigned UUID: {alice.uuid}")

    print("\n[+] Adding vertex 'Post 1'...")
    post1 = await client.add_vertex("posts", realm=realm, payload={"title": "Hello World"})
    print(f"    Post 1 ID: {post1.id}")
    print(f"    Post 1 Automatically Assigned UUID: {post1.uuid}")

    print("\n[+] Adding edge 'authored'...")
    edge = await client.add_edge("authored", realm=realm, from_id=alice.id, to_id=post1.id, relation_type="authored", payload={"created_year": 2026})
    print(f"    Edge ID: {edge.id}")
    print(f"    Edge Automatically Assigned UUID: {edge.uuid}")

    print(f"\n[+] Searching Vertex 'users' by UUID ({alice.uuid})...")
    fetched_user = await client.get_vertex_by_uuid("users", realm=realm, uuid=alice.uuid)
    print(f"    Found Vertex: {fetched_user.payload['name']} (ID: {fetched_user.id}, UUID: {fetched_user.uuid})")

    print(f"\n[+] Searching Vertex 'users' via get_vertex() passing UUID string ({alice.uuid})...")
    fetched_user_gen = await client.get_vertex("users", realm=realm, vertex_id=alice.uuid)
    print(f"    Found Vertex via get_vertex: {fetched_user_gen.payload['name']}")

    print(f"\n[+] Searching Edge 'authored' by UUID ({edge.uuid})...")
    fetched_edge = await client.get_edge_by_uuid("authored", realm=realm, uuid=edge.uuid)
    print(f"    Found Edge: Relation={fetched_edge.relation_type} (ID: {fetched_edge.id}, UUID: {fetched_edge.uuid})")

    print("\n[+] Cleaning up demo tables...")
    for tbl in ["authored", "posts", "users"]:
        t_ref = client._get_table_ref(tbl, realm=realm)
        a_ref = client._get_table_ref(f"{tbl}_audit", realm=realm)
        d_ref = client._get_table_ref(f"{tbl}_data", realm=realm)
        await client._execute(f"DROP TABLE IF EXISTS {d_ref} CASCADE;")
        await client._execute(f"DROP TABLE IF EXISTS {a_ref} CASCADE;")
        await client._execute(f"DROP TABLE IF EXISTS {t_ref} CASCADE;")

    await client.close()
    print("[+] Demo complete!")

if __name__ == "__main__":
    asyncio.run(main())
