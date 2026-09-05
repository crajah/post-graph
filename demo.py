import asyncio
import os
import sys

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

# Ensure the local package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

import getpass

from post_graph import AsyncPostGraph, SQLAlchemyPostGraph, TableNotFoundError

default_user = getpass.getuser()
DATABASE_URL = os.environ.get("DATABASE_URL", f"postgresql://{default_user}@localhost:5432/postgres")
# For SQLAlchemy we need postgresql+asyncpg
SQLALCHEMY_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")


async def run_asyncpg_demo():
    print("\n" + "=" * 50)
    print("RUNNING RAW ASYNCPG DEMO")
    print("=" * 50)

    # Initialize client
    client = AsyncPostGraph(dsn=DATABASE_URL)
    try:
        await client.connect()
    except Exception as e:
        print(f"[-] Failed to connect using raw asyncpg: {e}")
        print("    Please ensure PostgreSQL is running and DATABASE_URL is correct.")
        print("    E.g. run: docker run -d --name post-graph-pg -p 5432:5432 -e POSTGRES_PASSWORD=postgres postgres")
        return False

    try:
        # Clean start: drop existing tables
        for table in ["usersTOusers", "likes", "follows", "follows_cascade", "users", "posts"]:
            await client._execute(f'DROP TABLE IF EXISTS "{table}_data" CASCADE;')
            await client._execute(f'DROP TABLE IF EXISTS "{table}_audit" CASCADE;')
            await client._execute(f'DROP TABLE IF EXISTS "{table}" CASCADE;')

        # 1. Create tables
        print("[+] Creating vertex tables 'users' and 'posts'...")
        await client.create_vertex_table("users")
        await client.create_vertex_table("posts")

        print("[+] Creating edge tables 'follows' (users -> users) and 'likes' (users -> posts)...")
        await client.create_edge_table("follows", from_vertex_table="users", to_vertex_table="users")
        await client.create_edge_table("likes", from_vertex_table="users", to_vertex_table="posts")

        # Verify that the explicit edge tables were created
        try:
            follows_exists = await client._table_exists("follows")
            likes_exists = await client._table_exists("likes")
            print(f"    [OK] Edge table 'follows' exists: {follows_exists}")
            print(f"    [OK] Edge table 'likes' exists: {likes_exists}")
        except Exception as e:
            print(f"    [WARN] Could not verify explicit edge tables: {e}")

        # Verify default naming when table_name not provided
        print("[+] Testing default edge table naming (should create 'usersTOusers')...")
        await client.create_edge_table(from_vertex_table="users", to_vertex_table="users", realm="realm_a")
        default_name = "usersTOusers"
        try:
            default_exists = await client._table_exists(default_name, realm="realm_a")
            print(f"    [OK] Default edge table '{default_name}' exists: {default_exists}")
        except Exception as e:
            print(f"    [WARN] Could not verify default edge table naming: {e}")

        # Verify that referencing non-existent vertex tables raises TableNotFoundError
        try:
            print("[+] Verifying TableNotFoundError for invalid create_edge_table...")
            await client.create_edge_table("invalid_edge", from_vertex_table="non_existent_1", to_vertex_table="posts")
        except TableNotFoundError as e:
            print(f"    [OK] TableNotFoundError caught correctly: {e}")

        # 2. Insert vertices in different realms
        print("\n[+] Inserting vertices with user auditing...")
        # Realm A
        u1 = await client.add_vertex("users", realm="realm_a", vertex_id=1, payload={"name": "Alice Smith"}, user_id="admin_1")
        await client.add_vertex("users", realm="realm_a", vertex_id=2, payload={"name": "Bob Jones"}, user_id="admin_1")
        await client.add_vertex("posts", realm="realm_a", vertex_id=10, payload={"title": "Post Graph is Awesome"}, user_id="alice_user")

        # Realm B (Tenant Isolation verification)
        u3 = await client.add_vertex("users", realm="realm_b", vertex_id=1, payload={"name": "Alice Cooper"}, user_id="admin_2")
        print(f"    Inserted Alice in realm_a: {u1.payload}")
        print(f"    Inserted Alice in realm_b (isolated): {u3.payload}")

        # 3. Add edges (relationships)
        print("\n[+] Adding edges...")
        # Alice (1) follows Bob (2) in Realm A
        e1 = await client.add_edge(
            table_name="follows", realm="realm_a", edge_id=100,
            from_id=1, to_id=2, relation_type="friend",
            payload={"since": "2026-01-01"}, user_id="alice_user"
        )
        # Bob (2) likes Post 10 in Realm A
        e2 = await client.add_edge(
            table_name="likes", realm="realm_a", edge_id=101,
            from_id=2, to_id=10, relation_type="like",
            payload={"reaction": "heart"}, user_id="bob_user"
        )
        print(f"    Edge follows: {e1.from_id} -> {e1.to_id} (type: {e1.relation_type})")
        print(f"    Edge likes: {e2.from_id} -> {e2.to_id} (type: {e2.relation_type})")

        # Verify that we cannot link across realms (should fail constraint check)
        try:
            print("[+] Attempting invalid cross-realm edge (realm_a -> realm_b)...")
            await client.add_edge(
                table_name="follows", realm="realm_a", edge_id=999,
                from_id=1, to_id=9999, relation_type="friend"
            )
        except Exception as e:
            print(f"    [OK] Cross-realm constraint verification succeeded: {e}")

        # 4. View Shadow Audit Logs
        print("\n[+] Querying shadow audit logs for 'users'...")
        audit_rows = await client._fetch("SELECT * FROM users_audit ORDER BY audit_id")
        for row in audit_rows:
            print(f"    Audit {row['audit_id']}: Action={row['action']}, Realm={row['realm']}, User={row['changed_by']}, Timestamp={row['changed_at']}")
            print(f"      Old: {row['old_row']}")
            print(f"      New: {row['new_row']}")

        # 5. Perform Update and Check audit log
        print("\n[+] Updating Bob's payload...")
        await client.upsert_vertex("users", realm="realm_a", vertex_id=2, payload={"age": 30}, user_id="bob_user")
        
        # Check audit log for update
        update_audit = await client._fetch("SELECT * FROM users_audit WHERE action = 'UPDATE' ORDER BY audit_id DESC LIMIT 1")
        if update_audit:
            print(f"    Audit Update: Action={update_audit[0]['action']}, User={update_audit[0]['changed_by']}")
            print(f"      Old: {update_audit[0]['old_row']}")
            print(f"      New: {update_audit[0]['new_row']}")

        # 6. Graph Query: Get Neighbors
        print("\n[+] Getting neighbors of '1' (alice) in 'realm_a'...")
        neighbors = await client.get_neighbors(realm="realm_a", vertex_table="users", vertex_id=1, edge_tables=["follows"])
        for neighbor_v, edge in neighbors:
            print(f"    Neighbor ID: {neighbor_v.id} (Table: {neighbor_v.table_name or 'users'}) via relation: {edge.relation_type}")

        # Demonstration of direct vertex traversal API
        print("\n[+] Testing direct object traversal: get_vertex(1).to('follows')[0].vertex()...")
        alice_v = await client.get_vertex("users", "realm_a", 1)
        follows_steps = await alice_v.to("follows")
        if follows_steps:
            neighbor = follows_steps[0].vertex()
            print(f"    Direct traversal result: {alice_v.id} --[follows]--> {neighbor.id} ({neighbor.payload})")

        # Demonstration of direct reverse traversal chaining
        print("\n[+] Testing direct reverse traversal chaining: get_vertex(2).from_('follows')[0].vertex()...")
        bob_v = await client.get_vertex("users", "realm_a", 2)
        reverse_steps = await bob_v.from_("follows")
        if reverse_steps:
            neighbor = reverse_steps[0].vertex()
            print(f"    Reverse traversal result: {bob_v.id} <--[follows]-- {neighbor.id} ({neighbor.payload})")

        # Demonstration of get_vertex(id).from_(relation)[0].add_edge_to(to_id, relation)
        print("\n[+] Testing mutation via traversal step: get_vertex(2).from_('follows')[0].add_edge_to(10, 'likes')...")
        if reverse_steps:
            # reverse_steps[0].neighbor_vertex is alice. This will add an edge from alice to post 10 in likes.
            new_edge = await reverse_steps[0].add_edge_to(to_id=10, edge_table="likes")
            print(f"    Edge added via step: {new_edge.from_id} --[{new_edge.relation_type}]--> {new_edge.to_id} (ID: {new_edge.id})")

        # 7. Recursive CTE Traversal
        print("\n[+] Traversing graph from '1' (alice) (max_depth=3) in 'realm_a'...")
        paths = await client.traverse(realm="realm_a", start_table="users", start_id=1, edge_tables=["follows", "likes"])
        for path in paths:
            print(f"    Reachable: ID={path['id']}, Table={path['table_name']}, Depth={path['depth']}")
            print(f"      Path: {' -> '.join(path['path'])}")
            print(f"      Edges: {' -> '.join(path['edge_path'])}")

        # 8. Shortest Path Execution
        print("\n[+] Finding shortest path from '1' (alice, users) to '10' (post_1, posts) in 'realm_a'...")
        sp = await client.shortest_path(
            realm="realm_a", start_table="users", start_id=1,
            target_table="posts", target_id=10, edge_tables=["follows", "likes"]
        )
        if sp:
            print(f"    Shortest Path Found (Length={sp['depth']}):")
            print(f"      Nodes: {' -> '.join(sp['path'])}")
            print(f"      Edges: {' -> '.join(sp['edge_path'])}")
        else:
            print("    [-] No path found.")

        # 9. Cascade Delete Check
        print("\n[+] Deleting vertex '2' (bob) and verifying cascade on edge...")
        bob_v = await client.get_vertex("users", "realm_a", 2)
        await bob_v.delete(user_id="admin_1")
        
        # Verify edge follows was deleted
        edge_check = await client.get_edge("follows", realm="realm_a", edge_id=100)
        print(f"    Edge follows exists after deleting target vertex? {edge_check is not None}")

        # Check edge audit log for delete
        edge_audit = await client._fetch("SELECT * FROM follows_audit WHERE action = 'DELETE' LIMIT 1")
        if edge_audit:
            print(f"    Edge Audit Delete: Action={edge_audit[0]['action']}, User={edge_audit[0]['changed_by']}")
            print(f"      Old: {edge_audit[0]['old_row']}")

        # 10. Edge-to-Vertex Cascade Delete Check
        print("\n[+] Creating edge table 'follows_cascade' with cascade_delete_from=True and cascade_delete_to=True...")
        await client.create_edge_table("follows_cascade", from_vertex_table="users", to_vertex_table="users", cascade_delete_from=True, cascade_delete_to=True)
        
        print("    -> Adding temporary vertices 201 and 202...")
        await client.add_vertex("users", realm="realm_a", vertex_id=201)
        await client.add_vertex("users", realm="realm_a", vertex_id=202)
        
        print("    -> Linking with edge 200 in 'follows_cascade'")
        await client.add_edge("follows_cascade", realm="realm_a", edge_id=200, from_id=201, to_id=202, relation_type="temp_rel")
        
        print("    -> Deleting the edge 200...")
        await client.delete_edge("follows_cascade", realm="realm_a", edge_id=200)
        
        # Verify both vertices were automatically deleted
        v1 = await client.get_vertex("users", "realm_a", 201)
        v2 = await client.get_vertex("users", "realm_a", 202)
        print(f"    -> Verification: Does 201 still exist? {v1 is not None} (Expected: False)")
        print(f"    -> Verification: Does 202 still exist? {v2 is not None} (Expected: False)")

        # 11. Cycle Detection Check
        print("\n[+] Testing Cyclic Reference Detection...")
        # Create a small chain: node 301 -> 302 -> 303
        await client.add_vertex("users", realm="realm_a", vertex_id=301)
        await client.add_vertex("users", realm="realm_a", vertex_id=302)
        await client.add_vertex("users", realm="realm_a", vertex_id=303)
        
        await client.add_edge("follows", realm="realm_a", edge_id=300, from_id=301, to_id=302, relation_type="knows")
        await client.add_edge("follows", realm="realm_a", edge_id=301, from_id=302, to_id=303, relation_type="knows")
        
        # Attempt to add edge 303 -> 301 with check_cycle=True
        try:
            print("    -> Attempting to link 303 back to 301 (Check cycle = True)...")
            from post_graph.errors import CyclicReferenceError
            await client.add_edge(
                table_name="follows", realm="realm_a", edge_id=302,
                from_id=303, to_id=301, relation_type="knows",
                check_cycle=True
            )
            print("    [-] ERROR: Allowed cyclic edge creation!")
        except CyclicReferenceError as e:
            print(f"    [OK] CyclicReferenceError caught successfully: {e}")

        # 12. Complex Multi-Node & Cascade Delete Scenarios
        print("\n[+] Testing Complex Multi-Node Cascading Deletions...")
        await client.add_vertex("users", realm="realm_a", vertex_id=401)
        await client.add_vertex("users", realm="realm_a", vertex_id=402)
        await client.add_vertex("users", realm="realm_a", vertex_id=403)
        await client.add_vertex("posts", realm="realm_a", vertex_id=404)
        await client.add_vertex("posts", realm="realm_a", vertex_id=405)

        await client.add_edge("follows", realm="realm_a", edge_id=400, from_id=401, to_id=402, relation_type="branch")
        await client.add_edge("follows", realm="realm_a", edge_id=401, from_id=401, to_id=403, relation_type="branch")
        await client.add_edge("likes", realm="realm_a", edge_id=402, from_id=402, to_id=404, relation_type="leaf")
        await client.add_edge("likes", realm="realm_a", edge_id=403, from_id=403, to_id=405, relation_type="leaf")

        print("    -> Traversing tree from root_node (401)...")
        paths = await client.traverse(realm="realm_a", start_table="users", start_id=401, edge_tables=["follows", "likes"])
        for path in paths:
            print(f"       Path: {' -> '.join(path['path'])}")

        print("    -> Deleting branch_1 (402) (should cascade delete edges 400 and 402)...")
        b1 = await client.get_vertex("users", "realm_a", 402)
        await b1.delete()

        check_e_r_b1 = await client.get_edge("follows", realm="realm_a", edge_id=400)
        check_e_b1_l1 = await client.get_edge("likes", realm="realm_a", edge_id=402)
        print(f"       Does edge 400 still exist? {check_e_r_b1 is not None} (Expected: False)")
        print(f"       Does edge 402 still exist? {check_e_b1_l1 is not None} (Expected: False)")
        
        check_b2 = await client.get_vertex("users", "realm_a", 403)
        check_l1 = await client.get_vertex("posts", "realm_a", 404)
        print(f"       Does vertex 403 still exist? {check_b2 is not None} (Expected: True)")
        print(f"       Does vertex 404 still exist? {check_l1 is not None} (Expected: True)")

        print("    -> Deleting root_node (401) (should cascade delete edge 401)...")
        rn = await client.get_vertex("users", "realm_a", 401)
        await rn.delete()

        check_e_r_b2 = await client.get_edge("follows", realm="realm_a", edge_id=401)
        print(f"       Does edge 401 still exist? {check_e_r_b2 is not None} (Expected: False)")

        # 12.5. Append-Only Data Table Verification ({table_name}_data)
        print("\n[+] Testing Append-Only Data Tables ({table_name}_data)...")
        v_alice = await client.get_vertex("users", "realm_a", 1)
        if v_alice:
            await v_alice.add_data({"status": "active", "login_count": 1})
            await v_alice.add_data({"status": "idle", "login_count": 2})
            v_data = await v_alice.get_data()
            print(f"    Vertex Data Records for Alice: count={len(v_data)}")
            for r in v_data:
                print(f"      -> Data ID: {r.data_id}, Timestamp: {r.timestamp}, Payload: {r.payload}")
            assert len(v_data) == 2

        e_like = await client.get_edge("likes", "realm_a", 2)
        if e_like:
            await e_like.add_data({"event": "heart_clicked", "device": "mobile"})
            e_data = await e_like.get_data()
            print(f"    Edge Data Records for Like Edge: count={len(e_data)}, Payload: {e_data[0].payload}")
            assert len(e_data) == 1

        # 13. ID Autogeneration Verification
        print("\n[+] Testing Autogenerated ID formats...")
        v_auto = await client.add_vertex("users", realm="realm_a", payload={"name": "Auto Node"})
        print(f"    Generated Vertex ID: {v_auto.id}, FQID: {v_auto.fqid}")
        assert v_auto.fqid.startswith("realm_a/users/")
        assert v_auto.id.isdigit()

        e_auto = await client.add_edge("follows", realm="realm_a", from_id=1, to_id=301, relation_type="knows")
        print(f"    Generated Edge ID:   {e_auto.id}, FQID: {e_auto.fqid}")
        assert e_auto.fqid.startswith("realm_a/users-users/")
        assert e_auto.id.isdigit()

        # 14. delete_realm Verification
        print("\n[+] Testing delete_realm...")
        # Add test nodes in realm 'realm_to_delete'
        await client.add_vertex("users", realm="realm_to_delete", vertex_id=501)
        await client.add_vertex("users", realm="realm_to_delete", vertex_id=502)
        await client.add_edge("follows", realm="realm_to_delete", edge_id=500, from_id=501, to_id=502, relation_type="knows")

        # Delete realm
        deleted_count = await client.delete_realm("realm_to_delete")
        print(f"    Rows deleted by delete_realm: {deleted_count}")
        
        # Verify they don't exist anymore
        v1_check = await client.get_vertex("users", "realm_to_delete", 501)
        e1_check = await client.get_edge("follows", realm="realm_to_delete", edge_id=500)
        print(f"    Does 501 still exist? {v1_check is not None} (Expected: False)")
        print(f"    Does 500 still exist? {e1_check is not None} (Expected: False)")

        # Verify other realms are not affected
        alice_realm_b = await client.get_vertex("users", "realm_b", 1)
        print(f"    Does '1' in realm_b still exist? {alice_realm_b is not None} (Expected: True)")

    finally:
        # Cleanup DDL
        print("\n[+] Cleaning up database tables...")
        for table in ["usersTOusers", "likes", "follows", "follows_cascade", "users", "posts"]:
            await client._execute(f'DROP TABLE IF EXISTS "{table}_data" CASCADE;')
            await client._execute(f'DROP TABLE IF EXISTS "{table}_audit" CASCADE;')
            await client._execute(f'DROP TABLE IF EXISTS "{table}" CASCADE;')
        await client.close()
        print("[+] Finished cleanup.")
    return True


async def run_sqlalchemy_demo():
    print("\n" + "=" * 50)
    print("RUNNING SQLALCHEMY ASYNC DEMO")
    print("=" * 50)

    engine = create_async_engine(SQLALCHEMY_URL)
    client = SQLAlchemyPostGraph(engine)

    try:
        # Clean start
        async with engine.begin() as conn:
            for table in ["sa_usersTOusers", "sa_likes", "sa_follows", "sa_follows_cascade", "sa_users", "sa_posts"]:
                await conn.execute(text(f'DROP TABLE IF EXISTS "{table}_data" CASCADE;'))
                await conn.execute(text(f'DROP TABLE IF EXISTS "{table}_audit" CASCADE;'))
                await conn.execute(text(f'DROP TABLE IF EXISTS "{table}" CASCADE;'))

        # 1. Create tables
        print("[+] Creating vertex tables...")
        await client.create_vertex_table("sa_users")
        await client.create_vertex_table("sa_posts")
        
        print("[+] Creating edge tables...")
        await client.create_edge_table("sa_follows", from_vertex_table="sa_users", to_vertex_table="sa_users")
        await client.create_edge_table("sa_likes", from_vertex_table="sa_users", to_vertex_table="sa_posts")

        # Verify that referencing non-existent vertex tables raises TableNotFoundError in SQLAlchemy
        try:
            print("[+] Verifying TableNotFoundError in SQLAlchemy client...")
            await client.create_edge_table("sa_invalid_edge", from_vertex_table="sa_users", to_vertex_table="sa_non_existent")
        except TableNotFoundError as e:
            print(f"    [OK] TableNotFoundError caught correctly: {e}")

        # 2. Insert records passing user_id context
        print("[+] Inserting records with audit user context...")
        await client.add_vertex("sa_users", realm="default", vertex_id=1, payload={"username": "sa_alice"}, user_id="creator_1")
        await client.add_vertex("sa_users", realm="default", vertex_id=2, payload={"username": "sa_bob"}, user_id="creator_1")
        await client.add_edge("sa_follows", realm="default", edge_id=100, from_id=1, to_id=2, relation_type="follow", user_id="creator_2")

        # 3. Retrieve and print audit log
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT * FROM sa_users_audit ORDER BY audit_id"))
            rows = result.all()
            print("[+] Querying sa_users_audit...")
            for row in rows:
                print(f"    Audit {row.audit_id}: Action={row.action}, User={row.changed_by}")
                print(f"      New State: {row.new_row}")

        # 4. Traversal
        print("[+] Traversing using SQLAlchemy CTE...")
        paths = await client.traverse(realm="default", start_table="sa_users", start_id=1, edge_tables=["sa_follows"])
        for path in paths:
            print(f"    Path: {' -> '.join(path['path'])}")

        # Demonstration of direct vertex traversal API in SQLAlchemy client
        print("[+] Testing direct object traversal via SQLAlchemy client...")
        u1_v = await client.get_vertex("sa_users", "default", 1)
        sa_follows_steps = await u1_v.to("sa_follows")
        if sa_follows_steps:
            neighbor = sa_follows_steps[0].vertex()
            print(f"    Direct traversal result: {u1_v.id} --[sa_follows]--> {neighbor.id} ({neighbor.payload})")

        # Demonstration of direct reverse traversal chaining in SQLAlchemy client
        print("[+] Testing direct reverse traversal chaining via SQLAlchemy client...")
        u2_v = await client.get_vertex("sa_users", "default", 2)
        reverse_sa_steps = await u2_v.from_("sa_follows")
        if reverse_sa_steps:
            neighbor = reverse_sa_steps[0].vertex()
            print(f"    Reverse traversal result: {u2_v.id} <--[sa_follows]-- {neighbor.id} ({neighbor.payload})")

        # Demonstration of mutation via traversal step in SQLAlchemy client
        print("[+] Testing mutation via traversal step in SQLAlchemy client...")
        if reverse_sa_steps:
            # reverse_sa_steps[0].neighbor_vertex is u1. This will add an edge from u1 to u2 in sa_follows.
            new_sa_edge = await reverse_sa_steps[0].add_edge_to(to_id=2, edge_table="sa_follows", edge_id=101)
            print(f"    SQLAlchemy Edge added via step: {new_sa_edge.from_id} --[{new_sa_edge.relation_type}]--> {new_sa_edge.to_id}")

        # Demonstration of direct vertex deletion in SQLAlchemy client
        print("[+] Testing direct vertex deletion via SQLAlchemy client...")
        u2_v = await client.get_vertex("sa_users", "default", 2)
        await u2_v.delete(user_id="creator_1")
        
        # Verify edge sa_follows was cascade deleted at DB level
        sa_edge_check = await client.get_edge("sa_follows", realm="default", edge_id=101)
        print(f"    SQLAlchemy Edge follows exists after deleting target vertex? {sa_edge_check is not None}")

        # Demonstration of Edge-to-Vertex Cascade Delete Check in SQLAlchemy
        print("\n[+] Testing edge-to-vertex cascade delete via SQLAlchemy client...")
        await client.create_edge_table("sa_follows_cascade", from_vertex_table="sa_users", to_vertex_table="sa_users", cascade_delete_from=True, cascade_delete_to=True)
        
        print("    -> Adding temporary vertices 201 and 202...")
        await client.add_vertex("sa_users", realm="default", vertex_id=201)
        await client.add_vertex("sa_users", realm="default", vertex_id=202)
        
        print("    -> Linking with edge 200...")
        await client.add_edge("sa_follows_cascade", realm="default", edge_id=200, from_id=201, to_id=202, relation_type="temp_rel")
        
        print("    -> Deleting the edge 200...")
        await client.delete_edge("sa_follows_cascade", realm="default", edge_id=200)
        
        # Verify both vertices were automatically deleted
        sa_v1 = await client.get_vertex("sa_users", "default", 201)
        sa_v2 = await client.get_vertex("sa_users", "default", 202)
        print(f"    -> Verification: Does 201 still exist? {sa_v1 is not None} (Expected: False)")
        print(f"    -> Verification: Does 202 still exist? {sa_v2 is not None} (Expected: False)")

        # Testing Cyclic Reference Detection in SQLAlchemy
        print("\n[+] Testing Cyclic Reference Detection via SQLAlchemy client...")
        await client.add_vertex("sa_users", realm="default", vertex_id=301)
        await client.add_vertex("sa_users", realm="default", vertex_id=302)
        await client.add_vertex("sa_users", realm="default", vertex_id=303)
        
        await client.add_edge("sa_follows", realm="default", edge_id=300, from_id=301, to_id=302, relation_type="knows")
        await client.add_edge("sa_follows", realm="default", edge_id=301, from_id=302, to_id=303, relation_type="knows")
        
        # Attempt to add edge 303 -> 301 with check_cycle=True
        try:
            print("    -> Attempting to link 303 back to 301 (Check cycle = True)...")
            from post_graph.errors import CyclicReferenceError
            await client.add_edge(
                table_name="sa_follows", realm="default", edge_id=302,
                from_id=303, to_id=301, relation_type="knows",
                check_cycle=True
            )
            print("    [-] ERROR: Allowed cyclic edge creation in SQLAlchemy!")
        except CyclicReferenceError as e:
            print(f"    [OK] CyclicReferenceError caught successfully in SQLAlchemy: {e}")

        # Complex Multi-Node & Cascade Delete Scenarios in SQLAlchemy
        print("\n[+] Testing Complex Multi-Node Cascading Deletions via SQLAlchemy client...")
        await client.add_vertex("sa_users", realm="default", vertex_id=401)
        await client.add_vertex("sa_users", realm="default", vertex_id=402)
        await client.add_vertex("sa_users", realm="default", vertex_id=403)
        await client.add_vertex("sa_posts", realm="default", vertex_id=404)
        await client.add_vertex("sa_posts", realm="default", vertex_id=405)

        await client.add_edge("sa_follows", realm="default", edge_id=400, from_id=401, to_id=402, relation_type="branch")
        await client.add_edge("sa_follows", realm="default", edge_id=401, from_id=401, to_id=403, relation_type="branch")
        await client.add_edge("sa_likes", realm="default", edge_id=402, from_id=402, to_id=404, relation_type="leaf")
        await client.add_edge("sa_likes", realm="default", edge_id=403, from_id=403, to_id=405, relation_type="leaf")

        print("    -> Traversing tree from sa_root_node (401)...")
        paths = await client.traverse(realm="default", start_table="sa_users", start_id=401, edge_tables=["sa_follows", "sa_likes"])
        for path in paths:
            print(f"       Path: {' -> '.join(path['path'])}")

        print("    -> Deleting sa_branch_1 (402) (should cascade delete edges 400 and 402)...")
        b1 = await client.get_vertex("sa_users", "default", 402)
        await b1.delete()

        check_e_r_b1 = await client.get_edge("sa_follows", realm="default", edge_id=400)
        check_e_b1_l1 = await client.get_edge("sa_likes", realm="default", edge_id=402)
        print(f"       Does edge 400 still exist? {check_e_r_b1 is not None} (Expected: False)")
        print(f"       Does edge 402 still exist? {check_e_b1_l1 is not None} (Expected: False)")
        
        check_b2 = await client.get_vertex("sa_users", "default", 403)
        check_l1 = await client.get_vertex("sa_posts", "default", 404)
        print(f"       Does vertex 403 still exist? {check_b2 is not None} (Expected: True)")
        print(f"       Does vertex 404 still exist? {check_l1 is not None} (Expected: True)")

        print("    -> Deleting sa_root_node (401) (should cascade delete edge 401)...")
        rn = await client.get_vertex("sa_users", "default", 401)
        await rn.delete()

        check_e_r_b2 = await client.get_edge("sa_follows", realm="default", edge_id=401)
        print(f"       Does edge 401 still exist? {check_e_r_b2 is not None} (Expected: False)")

        # Testing Autogenerated ID formats in SQLAlchemy
        print("\n[+] Testing Autogenerated ID formats via SQLAlchemy client...")
        v_auto = await client.add_vertex("sa_users", realm="default", payload={"name": "SA Auto Node"})
        print(f"    SQLAlchemy Generated Vertex ID: {v_auto.id}, FQID: {v_auto.fqid}")
        assert v_auto.fqid.startswith("default/sa_users/")
        assert v_auto.id.isdigit()

        e_auto = await client.add_edge("sa_follows", realm="default", from_id=1, to_id=301, relation_type="knows")
        print(f"    SQLAlchemy Generated Edge ID:   {e_auto.id}, FQID: {e_auto.fqid}")
        assert e_auto.fqid.startswith("default/sa_users-sa_users/")
        assert e_auto.id.isdigit()

        # Testing delete_realm in SQLAlchemy
        print("\n[+] Testing delete_realm via SQLAlchemy client...")
        await client.add_vertex("sa_users", realm="realm_to_delete", vertex_id=501)
        await client.add_vertex("sa_users", realm="realm_to_delete", vertex_id=502)
        await client.add_edge("sa_follows", realm="realm_to_delete", edge_id=500, from_id=501, to_id=502, relation_type="knows")

        # Delete realm
        deleted_count = await client.delete_realm("realm_to_delete")
        print(f"    SQLAlchemy Rows deleted by delete_realm: {deleted_count}")

        # Verify they don't exist anymore
        v1_check = await client.get_vertex("sa_users", "realm_to_delete", 501)
        e1_check = await client.get_edge("sa_follows", realm="realm_to_delete", edge_id=500)
        print(f"    SQLAlchemy Does 501 still exist? {v1_check is not None} (Expected: False)")
        print(f"    SQLAlchemy Does 500 still exist? {e1_check is not None} (Expected: False)")

        # Verify other realms are not affected
        sa_alice_default = await client.get_vertex("sa_users", "default", 1)
        print(f"    SQLAlchemy Does '1' in default realm still exist? {sa_alice_default is not None} (Expected: True)")

    except Exception as e:
        print(f"[-] Error during SQLAlchemy demo: {e}")
    finally:
        # Cleanup
        print("[+] Cleaning up sa_ tables...")
        async with engine.begin() as conn:
            for table in ["sa_usersTOusers", "sa_likes", "sa_follows", "sa_follows_cascade", "sa_users", "sa_posts"]:
                await conn.execute(text(f'DROP TABLE IF EXISTS "{table}_data" CASCADE;'))
                await conn.execute(text(f'DROP TABLE IF EXISTS "{table}_audit" CASCADE;'))
                await conn.execute(text(f'DROP TABLE IF EXISTS "{table}" CASCADE;'))
        await engine.dispose()
        print("[+] Finished cleanup.")


async def main():
    success = await run_asyncpg_demo()
    if success:
        await run_sqlalchemy_demo()


if __name__ == "__main__":
    # If running directly, execute the demo
    asyncio.run(main())
