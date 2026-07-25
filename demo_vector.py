"""Demo script verifying pgvector support and vector similarity search in post-graph."""
import asyncio
import os
from post_graph import AsyncPostGraph, TableNotFoundError

DB_URI = os.getenv("POSTGRES_URI", "postgresql://crajah@localhost:5432/postgres")

async def main():
    print("=" * 60)
    print("POST-GRAPH PGVECTOR DEMO")
    print("=" * 60)

    client = AsyncPostGraph(dsn=DB_URI)
    await client.connect()

    table_name = "doc_chunks"
    realm = "rag_demo"

    table_ref = client._get_table_ref(table_name, realm)
    audit_table_ref = client._get_table_ref(f"{table_name}_audit", realm)
    data_table_ref = client._get_table_ref(f"{table_name}_data", realm)

    await client._execute(f"DROP TABLE IF EXISTS {data_table_ref} CASCADE;")
    await client._execute(f"DROP TABLE IF EXISTS {audit_table_ref} CASCADE;")
    await client._execute(f"DROP TABLE IF EXISTS {table_ref} CASCADE;")

    print("\n[+] Creating vertex table 'doc_chunks' with vector_dim=4...")
    try:
        await client.create_vertex_table(table_name, realm=realm, vector_dim=4)
        print("    [OK] Table created successfully with vector_dim=4")
    except Exception as e:
        print(f"    [SKIP] pgvector extension not installed in Postgres: {e}")
        await client.close()
        return

    print("\n[+] Inserting document chunk vertices with 4D embeddings...")
    v1 = await client.add_vertex(
        table_name, realm=realm,
        payload={"text": "PostgreSQL is a relational database system."},
        embedding=[1.0, 0.0, 0.0, 0.0]
    )
    print(f"    Inserted V1 ID={v1.id}, embedding={v1.embedding}")

    v2 = await client.add_vertex(
        table_name, realm=realm,
        payload={"text": "Post-graph enables graph storage on PostgreSQL."},
        embedding=[0.9, 0.1, 0.0, 0.0]
    )
    print(f"    Inserted V2 ID={v2.id}, embedding={v2.embedding}")

    v3 = await client.add_vertex(
        table_name, realm=realm,
        payload={"text": "Deep Learning models extract knowledge graphs."},
        embedding=[0.0, 0.0, 1.0, 0.0]
    )
    print(f"    Inserted V3 ID={v3.id}, embedding={v3.embedding}")

    print("\n[+] Performing Cosine Vector Similarity Search for query [1.0, 0.05, 0.0, 0.0]...")
    query_vec = [1.0, 0.05, 0.0, 0.0]
    results = await client.vector_search(table_name, realm=realm, query_vector=query_vec, top_k=3, distance_metric="cosine")

    for rank, (vertex, distance) in enumerate(results, 1):
        print(f"    Rank {rank}: ID={vertex.id}, Cosine Distance={distance:.4f}, Text='{vertex.payload['text']}'")

    print("\n[+] Cleaning up demo table...")
    await client._execute(f"DROP TABLE IF EXISTS {data_table_ref} CASCADE;")
    await client._execute(f"DROP TABLE IF EXISTS {audit_table_ref} CASCADE;")
    await client._execute(f"DROP TABLE IF EXISTS {table_ref} CASCADE;")
    await client.close()
    print("[+] Done!")

if __name__ == "__main__":
    asyncio.run(main())
