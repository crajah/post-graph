"""Tokeniser for the supported openCypher subset.

Cypher keywords are case-insensitive but identifiers are not, so keywords are
recognised by upper-cased text while the original spelling is kept for names.
"""
import re
from dataclasses import dataclass
from typing import Any, List, Optional


class CypherSyntaxError(ValueError):
    """Raised for input this dialect cannot parse.

    Carries the offset so the caller can point at the offending text rather than
    reporting that 'the query' is wrong.
    """

    def __init__(self, message: str, position: Optional[int] = None, query: Optional[str] = None):
        self.position = position
        self.query = query
        if position is not None and query is not None:
            line_start = query.rfind('\n', 0, position) + 1
            line_end = query.find('\n', position)
            line_end = len(query) if line_end == -1 else line_end
            caret = ' ' * (position - line_start) + '^'
            message = f"{message}\n  {query[line_start:line_end]}\n  {caret}"
        super().__init__(message)


KEYWORDS = {
    'MATCH', 'OPTIONAL', 'WHERE', 'RETURN', 'CREATE', 'MERGE', 'SET', 'DELETE',
    'DETACH', 'REMOVE', 'WITH', 'UNWIND', 'ORDER', 'BY', 'SKIP', 'LIMIT',
    'DISTINCT', 'AS', 'AND', 'OR', 'XOR', 'NOT', 'IN', 'IS', 'NULL', 'TRUE',
    'FALSE', 'STARTS', 'ENDS', 'CONTAINS', 'ASC', 'ASCENDING', 'DESC',
    'DESCENDING', 'ON', 'UNION', 'ALL', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END',
}

@dataclass
class Token:
    kind: str          # KEYWORD IDENT NUMBER STRING PARAM OP PUNCT EOF
    value: Any
    pos: int


_TOKEN_RE = re.compile(r"""
    (?P<WS>\s+)
  | (?P<COMMENT>//[^\n]*)
  | (?P<NUMBER>\d+\.\d+|\d+)
  | (?P<STRING>'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")
  | (?P<PARAM>\$[A-Za-z_][A-Za-z0-9_]*)
  | (?P<BACKTICK>`(?:[^`]|``)*`)
  | (?P<IDENT>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<OP><>|<=|>=|=~|\.\.|=|<|>|\+|-|\*|/|%|\^)
  | (?P<PUNCT>[()\[\]{},:.|])
""", re.VERBOSE)


def _unescape(raw: str) -> str:
    body = raw[1:-1]
    return (body.replace("\\'", "'").replace('\\"', '"')
                .replace('\\n', '\n').replace('\\t', '\t')
                .replace('\\r', '\r').replace('\\\\', '\\'))


def tokenize(query: str) -> List[Token]:
    tokens: List[Token] = []
    i = 0
    n = len(query)
    while i < n:
        m = _TOKEN_RE.match(query, i)
        if not m:
            raise CypherSyntaxError(f"Unexpected character {query[i]!r}", i, query)
        kind = m.lastgroup
        text = m.group()
        i = m.end()
        if kind in ('WS', 'COMMENT'):
            continue
        if kind == 'NUMBER':
            tokens.append(Token('NUMBER', float(text) if '.' in text else int(text), m.start()))
        elif kind == 'STRING':
            tokens.append(Token('STRING', _unescape(text), m.start()))
        elif kind == 'PARAM':
            tokens.append(Token('PARAM', text[1:], m.start()))
        elif kind == 'BACKTICK':
            # Backticks quote an identifier that would otherwise be a keyword or
            # contain punctuation; the name keeps its exact spelling.
            tokens.append(Token('IDENT', text[1:-1].replace('``', '`'), m.start()))
        elif kind == 'IDENT':
            upper = text.upper()
            tokens.append(Token('KEYWORD', upper, m.start()) if upper in KEYWORDS
                          else Token('IDENT', text, m.start()))
        else:
            tokens.append(Token(kind, text, m.start()))
    tokens.append(Token('EOF', None, n))
    return tokens
