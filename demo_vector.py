"""Demo script verifying pgvector support and vector similarity search in post-graph across main and data tables."""
import asyncio
import os
from post_graph import AsyncPostGraph, TableNotFoundError

DB_URI = os.getenv("POSTGRES_URI", "postgresql://crajah@localhost:5432/postgres")

async def main():
    print("=" * 60)
    print("POST-GRAPH PGVECTOR DEMO (MAIN & DATA TABLE SEARCH)")
    print("=" * 60)

    client = AsyncPostGraph(dsn=DB_URI)
    await client.connect()

    table_name = "doc_chunks"
    realm = "rag_demo"

    table_ref = client._get_table_ref(table_name, realm)
    audit_table_ref = client._get_table_ref(f"{table_name}_data_audit", realm)
    audit_table_main = client._get_table_ref(f"{table_name}_audit", realm)
    data_table_ref = client._get_table_ref(f"{table_name}_data", realm)

    await client._execute(f"DROP TABLE IF EXISTS {data_table_ref} CASCADE;")
    await client._execute(f"DROP TABLE IF EXISTS {audit_table_main} CASCADE;")
    await client._execute(f"DROP TABLE IF EXISTS {table_ref} CASCADE;")

    print("\n[+] Creating vertex table 'doc_chunks' with vector_dim=4...")
    try:
        await client.create_vertex_table(table_name, realm=realm, vector_dim=4)
        print("    [OK] Table created successfully with vector_dim=4 for main & data tables")
    except Exception as e:
        print(f"    [SKIP] pgvector extension not installed in Postgres: {e}")
        await client.close()
        return

    print("\n[+] Inserting document chunk vertices into MAIN table with 4D embeddings...")
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

    print("\n[+] Adding historical data records to DATA table with 4D embeddings for V1 & V2...")
    d1 = await v1.add_data(
        payload={"version": "v1.1", "note": "PostgreSQL 16 release notes"},
        embedding=[0.0, 1.0, 0.0, 0.0]
    )
    print(f"    Added V1 DataRecord ID={d1.data_id}, embedding={d1.embedding}")

    d2 = await v2.add_data(
        payload={"version": "v1.2", "note": "Post-graph pgvector extension release"},
        embedding=[0.0, 0.95, 0.05, 0.0]
    )
    print(f"    Added V2 DataRecord ID={d2.data_id}, embedding={d2.embedding}")

    query_vec_main = [1.0, 0.05, 0.0, 0.0]
    print(f"\n[+] 1. Vector Search [Scope: MAIN] for query {query_vec_main}...")
    results_main = await client.vector_search(table_name, realm=realm, query_vector=query_vec_main, top_k=2, search_scope="main")
    for rank, (vertex, dist) in enumerate(results_main, 1):
        print(f"    Rank {rank}: Vertex ID={vertex.id}, Distance={dist:.4f}, Text='{vertex.payload['text']}'")

    query_vec_data = [0.0, 1.0, 0.0, 0.0]
    print(f"\n[+] 2. Vector Search [Scope: DATA] for query {query_vec_data}...")
    results_data = await client.vector_search(table_name, realm=realm, query_vector=query_vec_data, top_k=2, search_scope="data")
    for rank, (vertex, dist) in enumerate(results_data, 1):
        print(f"    Rank {rank}: Vertex ID={vertex.id}, Distance={dist:.4f}, Text='{vertex.payload['text']}'")

    print(f"\n[+] 3. Vector Search [Scope: BOTH] for query {query_vec_data}...")
    results_both = await client.vector_search(table_name, realm=realm, query_vector=query_vec_data, top_k=2, search_scope="both")
    for rank, (vertex, dist) in enumerate(results_both, 1):
        print(f"    Rank {rank}: Vertex ID={vertex.id}, Distance={dist:.4f}, Text='{vertex.payload['text']}'")

    print("\n[+] Cleaning up demo table...")
    await client._execute(f"DROP TABLE IF EXISTS {data_table_ref} CASCADE;")
    await client._execute(f"DROP TABLE IF EXISTS {audit_table_main} CASCADE;")
    await client._execute(f"DROP TABLE IF EXISTS {table_ref} CASCADE;")
    await client.close()
    print("[+] Done!")

if __name__ == "__main__":
    asyncio.run(main())
