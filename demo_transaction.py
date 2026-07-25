import asyncio
import getpass
import os
import sys
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

# Ensure the local package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from post_graph import AsyncPostGraph, SQLAlchemyPostGraph

# Setup connection string defaulting to current OS user
default_user = getpass.getuser()
DATABASE_URL = os.environ.get("DATABASE_URL", f"postgresql://{default_user}@localhost:5432/postgres")
SQLALCHEMY_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")


async def run_asyncpg_transaction_demo():
    print("\n" + "=" * 60)
    print("ASYNCPG TRANSACTION DEMO (RAW)")
    print("=" * 60)

    # 1. Setup client & tables
    client = AsyncPostGraph(dsn=DATABASE_URL)
    await client.connect()

    try:
        print("[+] Preparing clean tables...")
        await client._execute('DROP TABLE IF EXISTS "tx_follows_data" CASCADE;')
        await client._execute('DROP TABLE IF EXISTS "tx_follows_audit" CASCADE;')
        await client._execute('DROP TABLE IF EXISTS "tx_follows" CASCADE;')
        await client._execute('DROP TABLE IF EXISTS "tx_users_data" CASCADE;')
        await client._execute('DROP TABLE IF EXISTS "tx_users_audit" CASCADE;')
        await client._execute('DROP TABLE IF EXISTS "tx_users" CASCADE;')
        
        await client.create_vertex_table("tx_users")
        await client.create_edge_table("tx_follows", from_vertex_table="tx_users", to_vertex_table="tx_users")

        # Get raw pool reference
        pool = client.connection

        # ----------------------------------------------------
        # CASE 1: Successful Transaction (COMMIT)
        # ----------------------------------------------------
        print("\n[+] Starting Transaction 1 (Expect Commit)...")
        async with pool.acquire() as conn:
            async with conn.transaction():
                # Bind the transactional connection to a temporary client wrapper
                tx_client = AsyncPostGraph(connection_or_pool=conn)

                print("    -> Adding vertex '1' (tx_alice)")
                await tx_client.add_vertex("tx_users", realm="default", vertex_id=1, payload={"role": "admin"})
                print("    -> Adding vertex '2' (tx_bob)")
                await tx_client.add_vertex("tx_users", realm="default", vertex_id=2, payload={"role": "user"})
                print("    -> Adding edge '1' -> '2'")
                await tx_client.add_edge("tx_follows", realm="default", edge_id=1, from_id=1, to_id=2, relation_type="knows")
                
                print("    [COMMIT] Exiting transaction block normally. Committing modifications...")

        # Verify commit succeeded by reading from the main pool
        v_alice = await client.get_vertex("tx_users", "default", 1)
        v_bob = await client.get_vertex("tx_users", "default", 2)
        print(f"    -> Verification: Does 'tx_alice' (ID 1) exist? {v_alice is not None}")
        print(f"    -> Verification: Does 'tx_bob' (ID 2) exist? {v_bob is not None}")


        # ----------------------------------------------------
        # CASE 2: Failing Transaction (ROLLBACK)
        # ----------------------------------------------------
        print("\n[+] Starting Transaction 2 (Expect Rollback)...")
        try:
            async with pool.acquire() as conn:
                async with conn.transaction():
                    # Bind connection to a temporary client wrapper
                    tx_client = AsyncPostGraph(connection_or_pool=conn)

                    print("    -> Adding vertex '3' (tx_charlie)")
                    await tx_client.add_vertex("tx_users", realm="default", vertex_id=3, payload={"role": "manager"})
                    
                    print("    -> Adding vertex '4' (tx_david)")
                    await tx_client.add_vertex("tx_users", realm="default", vertex_id=4, payload={"role": "contractor"})

                    # Simulate a failure (e.g. key constraint error referencing non-existent vertex 999)
                    print("    -> Simulating error: Attempting to add edge referencing non-existent vertex ID 999")
                    await tx_client.add_edge("tx_follows", realm="default", edge_id=2, from_id=3, to_id=999, relation_type="knows")
                    
                    print("    This line will NOT be reached.")
        except Exception as e:
            print(f"    [ROLLBACK] Intercepted expected transaction failure: {e}")
            print("    Rolling back all changes made in Transaction 2...")

        # Verify rollback succeeded
        v_charlie = await client.get_vertex("tx_users", "default", 3)
        v_david = await client.get_vertex("tx_users", "default", 4)
        print(f"    -> Verification: Does 'tx_charlie' (ID 3) exist? {v_charlie is not None} (Expected: False)")
        print(f"    -> Verification: Does 'tx_david' (ID 4) exist? {v_david is not None} (Expected: False)")

    finally:
        print("\n[+] Cleaning up tables...")
        await client._execute('DROP TABLE IF EXISTS "tx_follows_data" CASCADE;')
        await client._execute('DROP TABLE IF EXISTS "tx_follows_audit" CASCADE;')
        await client._execute('DROP TABLE IF EXISTS "tx_follows" CASCADE;')
        await client._execute('DROP TABLE IF EXISTS "tx_users_data" CASCADE;')
        await client._execute('DROP TABLE IF EXISTS "tx_users_audit" CASCADE;')
        await client._execute('DROP TABLE IF EXISTS "tx_users" CASCADE;')
        await client.close()


async def run_sqlalchemy_transaction_demo():
    print("\n" + "=" * 60)
    print("SQLALCHEMY TRANSACTION DEMO")
    print("=" * 60)

    # 1. Setup client and connection engine
    engine = create_async_engine(SQLALCHEMY_URL)
    client = SQLAlchemyPostGraph(engine)

    try:
        print("[+] Preparing clean tables...")
        async with engine.begin() as conn:
            await conn.execute(text('DROP TABLE IF EXISTS "sa_tx_follows_data" CASCADE;'))
            await conn.execute(text('DROP TABLE IF EXISTS "sa_tx_follows_audit" CASCADE;'))
            await conn.execute(text('DROP TABLE IF EXISTS "sa_tx_follows" CASCADE;'))
            await conn.execute(text('DROP TABLE IF EXISTS "sa_tx_users_data" CASCADE;'))
            await conn.execute(text('DROP TABLE IF EXISTS "sa_tx_users_audit" CASCADE;'))
            await conn.execute(text('DROP TABLE IF EXISTS "sa_tx_users" CASCADE;'))
        
        await client.create_vertex_table("sa_tx_users")
        await client.create_edge_table("sa_tx_follows", from_vertex_table="sa_tx_users", to_vertex_table="sa_tx_users")

        # ----------------------------------------------------
        # CASE 1: Successful Transaction (COMMIT)
        # ----------------------------------------------------
        print("\n[+] Starting Transaction 1 (Expect Commit)...")
        async with engine.connect() as conn:
            async with conn.begin(): # Auto-commits on block exit
                tx_client = SQLAlchemyPostGraph(conn)

                print("    -> Adding vertex '1' (sa_tx_alice)")
                await tx_client.add_vertex("sa_tx_users", realm="default", vertex_id=1)
                print("    -> Adding vertex '2' (sa_tx_bob)")
                await tx_client.add_vertex("sa_tx_users", realm="default", vertex_id=2)
                print("    -> Adding edge '1' -> '2'")
                await tx_client.add_edge("sa_tx_follows", realm="default", edge_id=1, from_id=1, to_id=2, relation_type="knows")

                print("    [COMMIT] Exiting transaction block normally. Committing modifications...")

        # Verify commit succeeded
        v_alice = await client.get_vertex("sa_tx_users", "default", 1)
        v_bob = await client.get_vertex("sa_tx_users", "default", 2)
        print(f"    -> Verification: Does 'sa_tx_alice' (ID 1) exist? {v_alice is not None}")
        print(f"    -> Verification: Does 'sa_tx_bob' (ID 2) exist? {v_bob is not None}")

        # ----------------------------------------------------
        # CASE 2: Failing Transaction (ROLLBACK)
        # ----------------------------------------------------
        print("\n[+] Starting Transaction 2 (Expect Rollback)...")
        try:
            async with engine.connect() as conn:
                async with conn.begin(): # Auto-rolls back on exception
                    tx_client = SQLAlchemyPostGraph(conn)

                    print("    -> Adding vertex '3' (sa_tx_charlie)")
                    await tx_client.add_vertex("sa_tx_users", realm="default", vertex_id=3)
                    print("    -> Adding vertex '4' (sa_tx_david)")
                    await tx_client.add_vertex("sa_tx_users", realm="default", vertex_id=4)

                    # Raise a manual exception inside the block
                    print("    -> Triggering manual exception inside transaction block...")
                    raise ValueError("Manual rollback trigger exception!")

        except Exception as e:
            print(f"    [ROLLBACK] Intercepted expected transaction failure: {e}")
            print("    Rolling back all changes made in Transaction 2...")

        # Verify rollback succeeded
        v_charlie = await client.get_vertex("sa_tx_users", "default", 3)
        v_david = await client.get_vertex("sa_tx_users", "default", 4)
        print(f"    -> Verification: Does 'sa_tx_charlie' (ID 3) exist? {v_charlie is not None} (Expected: False)")
        print(f"    -> Verification: Does 'sa_tx_david' (ID 4) exist? {v_david is not None} (Expected: False)")

    finally:
        print("\n[+] Cleaning up tables...")
        async with engine.begin() as conn:
            await conn.execute(text('DROP TABLE IF EXISTS "sa_tx_follows_data" CASCADE;'))
            await conn.execute(text('DROP TABLE IF EXISTS "sa_tx_follows_audit" CASCADE;'))
            await conn.execute(text('DROP TABLE IF EXISTS "sa_tx_follows" CASCADE;'))
            await conn.execute(text('DROP TABLE IF EXISTS "sa_tx_users_data" CASCADE;'))
            await conn.execute(text('DROP TABLE IF EXISTS "sa_tx_users_audit" CASCADE;'))
            await conn.execute(text('DROP TABLE IF EXISTS "sa_tx_users" CASCADE;'))
        await engine.dispose()
        await engine.dispose()


async def main():
    await run_asyncpg_transaction_demo()
    await run_sqlalchemy_transaction_demo()


if __name__ == "__main__":
    asyncio.run(main())
