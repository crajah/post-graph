"""openCypher over post-graph.

    session = CypherSession(client, realm="my_realm")
    rows = await session.run(
        "MATCH (p:Person)-[:KNOWS]->(f:Person) "
        "WHERE p.name = $name RETURN f.name AS friend",
        {"name": "Alice"})

Scope is a documented subset, not all of openCypher. Anything outside it raises
CypherSyntaxError or CypherTranslationError with the position or the reason —
this dialect refuses queries it cannot express rather than answering a nearby
question. See docs/cypher.md for exactly what is supported.
"""
from .engine import CypherSession
from .lexer import CypherSyntaxError, tokenize
from .parser import parse
from .translator import CypherTranslationError, Translator

__all__ = [
    'CypherSession', 'CypherSyntaxError', 'CypherTranslationError',
    'parse', 'tokenize', 'Translator',
]
