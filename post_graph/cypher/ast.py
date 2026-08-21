"""AST for the supported openCypher subset.

Deliberately small and explicit. Anything the parser cannot build a node for is
rejected at parse time rather than translated approximately — a Cypher layer
that silently answers a slightly different question than the one asked is worse
than one that refuses.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------- expressions

@dataclass
class Literal:
    value: Any


@dataclass
class Param:
    """$name — bound at execution, never interpolated into SQL."""
    name: str


@dataclass
class Variable:
    name: str


@dataclass
class Property:
    """n.prop — a property access on a bound pattern variable."""
    variable: str
    key: str


@dataclass
class Comparison:
    op: str          # = <> < <= > >= STARTS_WITH ENDS_WITH CONTAINS IN =~
    left: Any
    right: Any


@dataclass
class IsNull:
    operand: Any
    negated: bool = False


@dataclass
class Not:
    operand: Any


@dataclass
class BoolOp:
    op: str          # AND / OR / XOR
    left: Any
    right: Any


@dataclass
class FunctionCall:
    name: str        # count, sum, avg, min, max, collect, id, labels, type, ...
    args: List[Any] = field(default_factory=list)
    distinct: bool = False


# ------------------------------------------------------------------- patterns

@dataclass
class NodePattern:
    variable: Optional[str] = None
    labels: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RelPattern:
    variable: Optional[str] = None
    types: List[str] = field(default_factory=list)
    direction: str = 'out'            # 'out' | 'in' | 'both'
    properties: Dict[str, Any] = field(default_factory=dict)
    min_hops: Optional[int] = None    # set for variable-length patterns
    max_hops: Optional[int] = None


@dataclass
class PathPattern:
    """A chain: node (rel node)* — the only pattern shape openCypher allows."""
    nodes: List[NodePattern]
    rels: List[RelPattern] = field(default_factory=list)
    path_variable: Optional[str] = None


# --------------------------------------------------------------------- clauses

@dataclass
class Match:
    patterns: List[PathPattern]
    where: Optional[Any] = None
    optional: bool = False


@dataclass
class Create:
    patterns: List[PathPattern]


@dataclass
class Merge:
    pattern: PathPattern
    on_create: List[Tuple[Property, Any]] = field(default_factory=list)
    on_match: List[Tuple[Property, Any]] = field(default_factory=list)


@dataclass
class SetClause:
    assignments: List[Tuple[Property, Any]]


@dataclass
class Remove:
    properties: List[Property]


@dataclass
class Delete:
    variables: List[str]
    detach: bool = False


@dataclass
class ReturnItem:
    expression: Any
    alias: Optional[str] = None


@dataclass
class OrderItem:
    expression: Any
    descending: bool = False


@dataclass
class Return:
    items: List[ReturnItem]
    distinct: bool = False
    order_by: List[OrderItem] = field(default_factory=list)
    skip: Optional[Any] = None
    limit: Optional[Any] = None


@dataclass
class With:
    """WITH behaves as RETURN feeding the next clause, plus an optional WHERE."""
    items: List[ReturnItem]
    distinct: bool = False
    where: Optional[Any] = None
    order_by: List[OrderItem] = field(default_factory=list)
    skip: Optional[Any] = None
    limit: Optional[Any] = None


@dataclass
class Unwind:
    expression: Any
    alias: str


@dataclass
class Query:
    clauses: List[Any]
