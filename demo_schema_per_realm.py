import os
import asyncio
import getpass
import json
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from post_graph import AsyncPostGraph, SQLAlchemyPostGraph
from post_graph.errors import TableExistsError, TableNotFoundError, VertexNotFoundError, CyclicReferenceError

# Autodetect connection DSN
username = getpass.getuser()
raw_url = os.environ.get("DATABASE_URL", f"postgresql://{username}@localhost:5432/postgres")
if raw_url.startswith("postgresql://"):
    SQLALCHEMY_URL = raw_url.replace("postgresql://", "postgresql+asyncpg://")
else:
    SQLALCHEMY_URL = raw_url
ASYNC_DSN = raw_url


async def run_asyncpg_schema_demo():
    print("=" * 60)
    print("RUNNING ASYNCPG SCHEMA-PER-REALM INTEGRATION DEMO")
    print("=" * 60)

    client = AsyncPostGraph(dsn=ASYNC_DSN, schema_per_realm=True)
    await client.connect()

    realm_a = "tenant_a"
    realm_b = "tenant_b"

    try:
        # Clean start
        print("\n[+] Dropping old schema namespaces...")
        await client._execute(f'DROP SCHEMA IF EXISTS "{realm_a}" CASCADE;')
        await client._execute(f'DROP SCHEMA IF EXISTS "{realm_b}" CASCADE;')

        # 1. Create tables inside schemas
        print(f"[+] Creating tables in '{realm_a}' and '{realm_b}' schemas...")
        await client.create_vertex_table("users", realm=realm_a)
        await client.create_vertex_table("users", realm=realm_b)

        await client.create_vertex_table("posts", realm=realm_a)
        await client.create_vertex_table("posts", realm=realm_b)

        await client.create_edge_table("follows", from_vertex_table="users", to_vertex_table="users", realm=realm_a)
        await client.create_edge_table("follows", from_vertex_table="users", to_vertex_table="users", realm=realm_b)

        # 2. Add Vertices & Edges (Verification of isolation)
        print("\n[+] Populating vertices and edges...")
        # 2. Add Vertices & Edges (Verification of isolation)
        print("\n[+] Populating vertices and edges...")
        # Add alice in tenant_a
        alice_a = await client.add_vertex("users", realm=realm_a, vertex_id=1, payload={"name": "Alice in Wonderland"})
        # Add alice in tenant_b (different payload)
        alice_b = await client.add_vertex("users", realm=realm_b, vertex_id=1, payload={"name": "Alice in Chains"})
        
        # Add bob in both
        bob_a = await client.add_vertex("users", realm=realm_a, vertex_id=2, payload={"name": "Bob A"})
        bob_b = await client.add_vertex("users", realm=realm_b, vertex_id=2, payload={"name": "Bob B"})

        # Link followers
        await client.add_edge("follows", realm=realm_a, edge_id=10, from_id=1, to_id=2, relation_type="friend")
        await client.add_edge("follows", realm=realm_b, edge_id=10, from_id=1, to_id=2, relation_type="colleague")

        # 3. Retrieve and assert isolation
        print("\n[+] Verifying logical/physical data isolation...")
        v_a = await client.get_vertex("users", realm=realm_a, vertex_id=1)
        v_b = await client.get_vertex("users", realm=realm_b, vertex_id=1)
        print(f"    Loaded from '{realm_a}': {v_a.payload['name']}")
        print(f"    Loaded from '{realm_b}': {v_b.payload['name']}")
        assert v_a.payload['name'] == "Alice in Wonderland"
        assert v_b.payload['name'] == "Alice in Chains"

        e_a = await client.get_edge("follows", realm=realm_a, edge_id=10)
        e_b = await client.get_edge("follows", realm=realm_b, edge_id=10)
        print(f"    Edge relation in '{realm_a}': {e_a.relation_type}")
        print(f"    Edge relation in '{realm_b}': {e_b.relation_type}")
        assert e_a.relation_type == "friend"
        assert e_b.relation_type == "colleague"

        # 4. Traversals
        print("\n[+] Performing traversal queries within schema namespaces...")
        steps_a = await v_a.to("follows")
        print(f"    Alice's neighbors in '{realm_a}': {[s.vertex().payload['name'] for s in steps_a]}")
        assert len(steps_a) == 1
        assert steps_a[0].vertex().payload['name'] == "Bob A"

        steps_b = await v_b.to("follows")
        print(f"    Alice's neighbors in '{realm_b}': {[s.vertex().payload['name'] for s in steps_b]}")
        assert len(steps_b) == 1
        assert steps_b[0].vertex().payload['name'] == "Bob B"

        # 5. Schema-local cascade deletes
        print("\n[+] Verifying schema-local cascade deletes...")
        await client.delete_vertex("users", realm=realm_a, vertex_id=2)
        
        # In tenant_a, the follow edge should be gone
        check_e_a = await client.get_edge("follows", realm=realm_a, edge_id=10)
        print(f"    Does follows edge exist in '{realm_a}' after Bob is deleted? {check_e_a is not None} (Expected: False)")
        assert check_e_a is None

        # In tenant_b, the follows edge must be intact!
        check_e_b = await client.get_edge("follows", realm=realm_b, edge_id=10)
        print(f"    Does follows edge exist in '{realm_b}'? {check_e_b is not None} (Expected: True)")
        assert check_e_b is not None

        # 6. Cycle prevention within schema namespace
        print("\n[+] Testing cycle prevention in schema-per-realm mode...")
        # Re-add bob in A
        await client.add_vertex("users", realm=realm_a, vertex_id=2, payload={"name": "Bob A"})
        # Add carl in A
        await client.add_vertex("users", realm=realm_a, vertex_id=3, payload={"name": "Carl A"})

        # Link: alice (1) -> bob (2) -> carl (3)
        await client.add_edge("follows", realm=realm_a, edge_id=11, from_id=1, to_id=2, relation_type="friend")
        await client.add_edge("follows", realm=realm_a, edge_id=12, from_id=2, to_id=3, relation_type="friend")

        # Try to link carl (3) -> alice (1) with check_cycle=True
        try:
            await client.add_edge("follows", realm=realm_a, from_id=3, to_id=1, relation_type="friend", check_cycle=True)
            print("    [-] Error: Cyclic edge was incorrectly allowed!")
            assert False
        except CyclicReferenceError as e:
            print(f"    [OK] Cycle check prevented loop: {e}")

        # 7. Delete Realm
        print("\n[+] Verifying delete_realm under schema_per_realm mode...")
        deleted_rows = await client.delete_realm(realm=realm_a)
        print(f"    Rows deleted in '{realm_a}': {deleted_rows}")

        # Verify tenant_a vertices and edges are completely wiped
        check_alice_a = await client.get_vertex("users", realm=realm_a, vertex_id=1)
        print(f"    Does Alice in '{realm_a}' still exist? {check_alice_a is not None} (Expected: False)")
        assert check_alice_a is None

        # Verify tenant_b is completely untouched
        check_alice_b = await client.get_vertex("users", realm=realm_b, vertex_id=1)
        print(f"    Does Alice in '{realm_b}' still exist? {check_alice_b is not None} (Expected: True)")
        assert check_alice_b is not None

    finally:
        print("\n[+] Cleaning up schema namespaces...")
        await client._execute(f'DROP SCHEMA IF EXISTS "{realm_a}" CASCADE;')
        await client._execute(f'DROP SCHEMA IF EXISTS "{realm_b}" CASCADE;')
        await client.close()
        print("[+] Finished asyncpg schema demo.")


async def run_sqlalchemy_schema_demo():
    print("\n" + "=" * 60)
    print("RUNNING SQLALCHEMY SCHEMA-PER-REALM INTEGRATION DEMO")
    print("=" * 60)

    engine = create_async_engine(SQLALCHEMY_URL)
    client = SQLAlchemyPostGraph(engine, schema_per_realm=True)

    realm_a = "sa_tenant_a"
    realm_b = "sa_tenant_b"

    try:
        # Clean start
        print("\n[+] Dropping old schema namespaces...")
        async with engine.begin() as conn:
            await conn.execute(text(f'DROP SCHEMA IF EXISTS "{realm_a}" CASCADE;'))
            await conn.execute(text(f'DROP SCHEMA IF EXISTS "{realm_b}" CASCADE;'))

        # 1. Create tables inside schemas
        print(f"[+] Creating tables in '{realm_a}' and '{realm_b}' schemas...")
        await client.create_vertex_table("sa_users", realm=realm_a)
        await client.create_vertex_table("sa_users", realm=realm_b)

        await client.create_vertex_table("sa_posts", realm=realm_a)
        await client.create_vertex_table("sa_posts", realm=realm_b)

        await client.create_edge_table("sa_follows", from_vertex_table="sa_users", to_vertex_table="sa_users", realm=realm_a)
        await client.create_edge_table("sa_follows", from_vertex_table="sa_users", to_vertex_table="sa_users", realm=realm_b)

        # 2. Add Vertices & Edges (Verification of isolation)
        print("\n[+] Populating vertices and edges...")
        alice_a = await client.add_vertex("sa_users", realm=realm_a, vertex_id=1, payload={"name": "Alice in Wonderland"})
        alice_b = await client.add_vertex("sa_users", realm=realm_b, vertex_id=1, payload={"name": "Alice in Chains"})
        
        bob_a = await client.add_vertex("sa_users", realm=realm_a, vertex_id=2, payload={"name": "Bob A"})
        bob_b = await client.add_vertex("sa_users", realm=realm_b, vertex_id=2, payload={"name": "Bob B"})

        # Link followers
        await client.add_edge("sa_follows", realm=realm_a, edge_id=10, from_id=1, to_id=2, relation_type="friend")
        await client.add_edge("sa_follows", realm=realm_b, edge_id=10, from_id=1, to_id=2, relation_type="colleague")

        # 3. Retrieve and assert isolation
        print("\n[+] Verifying logical/physical data isolation...")
        v_a = await client.get_vertex("sa_users", realm=realm_a, vertex_id=1)
        v_b = await client.get_vertex("sa_users", realm=realm_b, vertex_id=1)
        print(f"    Loaded from '{realm_a}': {v_a.payload['name']}")
        print(f"    Loaded from '{realm_b}': {v_b.payload['name']}")
        assert v_a.payload['name'] == "Alice in Wonderland"
        assert v_b.payload['name'] == "Alice in Chains"

        e_a = await client.get_edge("sa_follows", realm=realm_a, edge_id=10)
        e_b = await client.get_edge("sa_follows", realm=realm_b, edge_id=10)
        print(f"    Edge relation in '{realm_a}': {e_a.relation_type}")
        print(f"    Edge relation in '{realm_b}': {e_b.relation_type}")
        assert e_a.relation_type == "friend"
        assert e_b.relation_type == "colleague"

        # 4. Traversals
        print("\n[+] Performing traversal queries within schema namespaces...")
        steps_a = await v_a.to("sa_follows")
        print(f"    Alice's neighbors in '{realm_a}': {[s.vertex().payload['name'] for s in steps_a]}")
        assert len(steps_a) == 1
        assert steps_a[0].vertex().payload['name'] == "Bob A"

        steps_b = await v_b.to("sa_follows")
        print(f"    Alice's neighbors in '{realm_b}': {[s.vertex().payload['name'] for s in steps_b]}")
        assert len(steps_b) == 1
        assert steps_b[0].vertex().payload['name'] == "Bob B"

        # 5. Schema-local cascade deletes
        print("\n[+] Verifying schema-local cascade deletes...")
        await client.delete_vertex("sa_users", realm=realm_a, vertex_id=2)
        
        # In tenant_a, the follow edge should be gone
        check_e_a = await client.get_edge("sa_follows", realm=realm_a, edge_id=10)
        print(f"    Does follows edge exist in '{realm_a}' after Bob is deleted? {check_e_a is not None} (Expected: False)")
        assert check_e_a is None

        # In tenant_b, the follows edge must be intact!
        check_e_b = await client.get_edge("sa_follows", realm=realm_b, edge_id=10)
        print(f"    Does follows edge exist in '{realm_b}'? {check_e_b is not None} (Expected: True)")
        assert check_e_b is not None

        # 6. Cycle prevention within schema namespace
        print("\n[+] Testing cycle prevention in schema-per-realm mode...")
        # Re-add bob in A
        await client.add_vertex("sa_users", realm=realm_a, vertex_id=2, payload={"name": "Bob A"})
        # Add carl in A
        await client.add_vertex("sa_users", realm=realm_a, vertex_id=3, payload={"name": "Carl A"})

        # Link: alice (1) -> bob (2) -> carl (3)
        await client.add_edge("sa_follows", realm=realm_a, edge_id=11, from_id=1, to_id=2, relation_type="friend")
        await client.add_edge("sa_follows", realm=realm_a, edge_id=12, from_id=2, to_id=3, relation_type="friend")

        # Try to link carl (3) -> alice (1) with check_cycle=True
        try:
            await client.add_edge("sa_follows", realm=realm_a, from_id=3, to_id=1, relation_type="friend", check_cycle=True)
            print("    [-] Error: Cyclic edge was incorrectly allowed!")
            assert False
        except CyclicReferenceError as e:
            print(f"    [OK] Cycle check prevented loop: {e}")

        # 7. Delete Realm
        print("\n[+] Verifying delete_realm under schema_per_realm mode...")
        deleted_count = await client.delete_realm(realm=realm_a)
        print(f"    Rows deleted by delete_realm: {deleted_count}")

        # Verify tenant_a vertices and edges are completely wiped
        check_alice_a = await client.get_vertex("sa_users", realm=realm_a, vertex_id=1)
        print(f"    Does Alice in '{realm_a}' still exist? {check_alice_a is not None} (Expected: False)")
        assert check_alice_a is None

        # Verify tenant_b is completely untouched
        check_alice_b = await client.get_vertex("sa_users", realm=realm_b, vertex_id=1)
        print(f"    Does Alice in '{realm_b}' still exist? {check_alice_b is not None} (Expected: True)")
        assert check_alice_b is not None

    finally:
        print("\n[+] Cleaning up schema namespaces...")
        async with engine.begin() as conn:
            await conn.execute(text(f'DROP SCHEMA IF EXISTS "{realm_a}" CASCADE;'))
            await conn.execute(text(f'DROP SCHEMA IF EXISTS "{realm_b}" CASCADE;'))
        await engine.dispose()
        print("[+] Finished SQLAlchemy schema demo.")


async def main():
    await run_asyncpg_schema_demo()
    await run_sqlalchemy_schema_demo()


if __name__ == "__main__":
    asyncio.run(main())
