"""Demo script for post-graph Space Sub-grouping.

Demonstrates segregating vertices, edges, and data records by {space} within a {realm}.
"""
import asyncio
import os
import sys

# Add parent dir to path
sys.path.insert(0, os.path.dirname(__file__))

from post_graph import AsyncPostGraph

async def main():
    dsn = os.getenv("POSTGRES_URI", "postgresql://crajah:postgrespassword@localhost:5432/postgres")
    client = AsyncPostGraph(dsn=dsn)
    try:
        await client.connect()
        print("Successfully connected to PostgreSQL for space verification.")
    except Exception as e:
        print(f"PostgreSQL connection skipped: {e}")
        return

    realm = "proj_alpha_civilization"

    # Create table with space support
    await client.create_vertex_table("agent_registry", realm=realm)
    print("Created vertex table 'agent_registry' with space column.")

    # 1. Upsert vertices into 'production' space
    v1 = await client.upsert_vertex(
        table_name="agent_registry",
        realm=realm,
        vertex_id=101,
        space="production",
        payload={"name": "Production Agent 1", "caste": "genesis"}
    )
    print(f"Upserted vertex into space 'production': {v1.id} (space={v1.space})")

    # 2. Upsert vertices into 'sandbox' space
    v2 = await client.upsert_vertex(
        table_name="agent_registry",
        realm=realm,
        vertex_id=102,
        space="sandbox",
        payload={"name": "Sandbox Test Agent", "caste": "progeny"}
    )
    print(f"Upserted vertex into space 'sandbox': {v2.id} (space={v2.space})")

    # 3. Query all vertices in realm
    all_v = await client.get_vertices(table_name="agent_registry", realm=realm)
    print(f"Total vertices in realm '{realm}': {len(all_v)}")

    # 4. Query vertices segregated by space 'production'
    prod_v = await client.get_vertices(table_name="agent_registry", realm=realm, space="production")
    print(f"Vertices in space 'production': {len(prod_v)} -> {[v.payload['name'] for v in prod_v]}")

    # 5. Query vertices segregated by space 'sandbox'
    sandbox_v = await client.get_vertices(table_name="agent_registry", realm=realm, space="sandbox")
    print(f"Vertices in space 'sandbox': {len(sandbox_v)} -> {[v.payload['name'] for v in sandbox_v]}")

    await client.close()
    print("Space demo completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
