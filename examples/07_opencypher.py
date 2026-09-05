"""Demo script for post-graph openCypher queries.

Shows Cypher reading and writing the same graph the rest of the API uses: a
label is a vertex table, a relationship type is a value in the relation_type
column, and a property is a payload key. Also shows the two things that make
this dialect worth trusting — the SQL it generates is inspectable, and queries
it cannot express are refused rather than approximated.
"""
import asyncio
import os
import sys

# Add parent dir to path
sys.path.insert(0, os.path.dirname(__file__))

from post_graph import AsyncPostGraph, CypherSession, CypherSyntaxError, CypherTranslationError


async def main():
    dsn = os.getenv("POSTGRES_URI", "postgresql://localhost:5432/postgres")
    client = AsyncPostGraph(dsn=dsn, schema_per_realm=True)
    try:
        await client.connect()
        print("Successfully connected to PostgreSQL for Cypher demo.")
    except Exception as e:
        print(f"PostgreSQL connection skipped: {e}")
        return

    realm = "cypher_demo"
    await client.delete_realm(realm)
    await client._execute(f'DROP SCHEMA IF EXISTS "{realm}" CASCADE')

    # Cypher labels are vertex tables, so the tables come first. A label with no
    # table behind it is an error, not an invitation to create one.
    await client.create_vertex_table("person", realm=realm)
    await client.create_vertex_table("company", realm=realm)
    await client.create_edge_table("knows", from_vertex_table="person",
                                   to_vertex_table="person", realm=realm)
    await client.create_edge_table("works_at", from_vertex_table="person",
                                   to_vertex_table="company", realm=realm)
    print("Created vertex tables 'person', 'company' and edge tables 'knows', 'works_at'.")

    session = CypherSession(client, realm)

    # ---------------------------------------------------------------- writing
    # CREATE goes through the client's own add_vertex/add_edge, so audit tables
    # and triggers fire exactly as they do for any other caller.
    await session.run("CREATE (p:Person {name: 'Alice', age: 34, city: 'London'})")
    await session.run("CREATE (p:Person {name: 'Bob', age: 28, city: 'Leeds'})")
    await session.run("CREATE (p:Person {name: $name, age: $age, city: 'London'})",
                      {"name": "Carol", "age": 41})
    await session.run(
        "CREATE (a:Person {name: 'Dan', age: 23, city: 'Lisbon'})"
        "-[:KNOWS {since: '2021'}]->(b:Person {name: 'Erin', age: 37, city: 'Leeds'})")
    print("Created 5 people and 1 relationship via CREATE.")

    # MERGE is match-or-create, so running it twice leaves one node.
    for _ in range(2):
        await session.run("MERGE (c:Company {name: 'Acme'})")
    rows = await session.run("MATCH (c:Company) RETURN count(*) AS n")
    print(f"MERGE ran twice, companies in the graph: {rows[0]['n']}")

    # Relationships between people already created above.
    people = {r['name']: r['id'] for r in
              await session.run("MATCH (p:Person) RETURN p.name AS name, id(p) AS id")}
    for a, b in [("Alice", "Bob"), ("Bob", "Carol"), ("Carol", "Dan")]:
        await client.add_edge("knows", realm, people[a], people[b], "KNOWS",
                              payload={"since": "2020"})
    await client.add_edge("works_at", realm, people["Alice"],
                          (await session.run("MATCH (c:Company) RETURN id(c) AS id"))[0]['id'],
                          "WORKS_AT", payload={})
    print("Linked people with KNOWS and Alice to Acme with WORKS_AT.")

    # ---------------------------------------------------------------- reading
    print("\n-- Filtering and ordering --")
    for row in await session.run(
            "MATCH (p:Person) WHERE p.age > 30 RETURN p.name AS name, p.age AS age "
            "ORDER BY age DESC"):
        print(f"   {row['name']:6s} {row['age']}")

    print("\n-- Parameters --")
    rows = await session.run(
        "MATCH (p:Person) WHERE p.city = $city RETURN p.name AS name ORDER BY name",
        {"city": "London"})
    print("   in London:", [r['name'] for r in rows])

    print("\n-- Relationships, and following one backwards --")
    rows = await session.run("MATCH (a:Person)-[:KNOWS]->(b:Person) WHERE a.name = 'Alice' "
                             "RETURN b.name AS friend")
    print("   Alice knows:", [r['friend'] for r in rows])
    rows = await session.run("MATCH (a:Person)<-[:KNOWS]-(b:Person) WHERE a.name = 'Carol' "
                             "RETURN b.name AS knower")
    print("   knows Carol:", [r['knower'] for r in rows])

    print("\n-- Variable-length: who is reachable within three hops --")
    rows = await session.run(
        "MATCH (a:Person)-[:KNOWS*1..3]->(b:Person) WHERE a.name = 'Alice' "
        "RETURN DISTINCT b.name AS reached ORDER BY reached")
    print("   from Alice:", [r['reached'] for r in rows])

    print("\n-- Aggregation, with GROUP BY inferred --")
    for row in await session.run(
            "MATCH (a:Person)-[:KNOWS]->(b:Person) "
            "RETURN a.name AS who, count(*) AS friends ORDER BY friends DESC, who"):
        print(f"   {row['who']:6s} knows {row['friends']}")

    print("\n-- Crossing labels --")
    for row in await session.run("MATCH (p:Person)-[:WORKS_AT]->(c:Company) "
                                 "RETURN p.name AS who, c.name AS employer"):
        print(f"   {row['who']} works at {row['employer']}")

    print("\n-- Returning a whole node --")
    row = (await session.run("MATCH (p:Person) WHERE p.name = 'Bob' RETURN p"))[0]
    print("   ", row['p'])

    print("\n-- String predicates and IN --")
    rows = await session.run("MATCH (p:Person) WHERE p.city STARTS WITH 'L' "
                             "AND p.name IN ['Alice','Dan'] RETURN p.name AS name ORDER BY name")
    print("   ", [r['name'] for r in rows])

    # ------------------------------------------------------------ inspectable
    print("\n-- What a read becomes: one SQL statement --")
    sql = await session.explain("MATCH (p:Person) WHERE p.age > $min RETURN p.name AS name",
                                {"min": 30})
    print("   ", sql[:160], "...")

    # A write has no SQL to show. CREATE, MERGE, SET and DELETE run through the
    # client's own methods so audit tables and triggers behave, which makes a
    # write a sequence of calls rather than one statement. explain() returns
    # that sequence, and running it changes nothing.
    print("\n-- What a write becomes: client operations, not one statement --")
    for query in [
        "CREATE (p:Person {name: 'Frank', age: 51, city: 'Leeds'})",
        "CREATE (a:Person {name: 'Gina'})-[:KNOWS {since: '2023'}]->(b:Person {name: 'Hank'})",
        "MERGE (c:Company {name: 'Acme'}) ON MATCH SET c.seen = 'again'",
        "CREATE (p:Person {name: 'Iris'}) SET p.age = 29",
    ]:
        print(f"\n   {query}")
        for line in (await session.explain(query)).splitlines():
            print(f"     {line}")

    before = (await session.run("MATCH (p:Person) RETURN count(*) AS n"))[0]['n']
    await session.explain("CREATE (p:Person {name: 'NeverCreated'})")
    after = (await session.run("MATCH (p:Person) RETURN count(*) AS n"))[0]['n']
    print(f"\n   explaining a CREATE does not run it: {before} people before, {after} after")

    # ---------------------------------------------------------------- refusals
    # The subset is bounded and says so. A query it cannot express is rejected
    # rather than answered approximately, because the caller cannot tell the
    # difference once an answer comes back.
    print("\n-- Queries this dialect refuses --")
    for query, why in [
        ("MATCH (n) RETURN n.name", "no label: the vertex table is unknown"),
        ("MATCH (p:Unicorn) RETURN p.name", "unknown label"),
        ("MATCH (p:Person) RETURN p.name LIMIT -1", "negative LIMIT"),
        ("MATCH (p:Person) RETURN *", "RETURN * is not supported"),
        ("MATCH (p:Person) WITH p RETURN p.name", "WITH is not translated"),
    ]:
        try:
            await session.run(query)
            print(f"   UNEXPECTED: accepted {query!r}")
        except (CypherSyntaxError, CypherTranslationError) as exc:
            print(f"   {why:42s} -> {str(exc).splitlines()[0][:60]}")

    # Injected SQL arrives as a value, never as SQL.
    rows = await session.run("MATCH (p:Person) WHERE p.name = $n RETURN p.name AS name",
                             {"n": "'; DROP TABLE person; --"})
    still_there = await session.run("MATCH (p:Person) RETURN count(*) AS n")
    print(f"\n   injection attempt matched {len(rows)} row(s); "
          f"people still in the graph: {still_there[0]['n']}")

    await client._execute(f'DROP SCHEMA IF EXISTS "{realm}" CASCADE')
    await client.close()
    print("\nCypher demo completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
