"""Recursive-descent parser for the supported openCypher subset.

Scope is deliberately bounded and every boundary raises CypherSyntaxError with a
position. The failure mode this avoids is the dangerous one: accepting a query
whose meaning we only approximate.
"""
from typing import Any, Dict, List, Optional, Tuple

from .ast import (
    BoolOp, Comparison, Create, Delete, FunctionCall, IsNull, Literal, Match,
    Merge, Not, NodePattern, OrderItem, Param, PathPattern, Property, Query,
    RelPattern, Remove, Return, ReturnItem, SetClause, UnaryOp, Unwind, Variable,
    With,
)
from .lexer import CypherSyntaxError, Token, tokenize

# Aggregations decide whether a projection implies GROUP BY.
AGGREGATES = {'count', 'sum', 'avg', 'min', 'max', 'collect', 'stdev', 'percentilecont'}
SCALAR_FUNCTIONS = {
    'id', 'labels', 'type', 'properties', 'keys', 'toupper', 'tolower', 'trim',
    'length', 'size', 'coalesce', 'tostring', 'tointeger', 'tofloat', 'abs',
    'ceil', 'floor', 'round', 'startnode', 'endnode', 'exists',
}


class Parser:
    def __init__(self, query: str):
        self.query = query
        self.tokens = tokenize(query)
        self.i = 0

    # ------------------------------------------------------------- utilities

    @property
    def cur(self) -> Token:
        return self.tokens[self.i]

    def at(self, kind: str, value: Any = None) -> bool:
        t = self.cur
        return t.kind == kind and (value is None or t.value == value)

    def at_keyword(self, *words: str) -> bool:
        return self.cur.kind == 'KEYWORD' and self.cur.value in words

    def advance(self) -> Token:
        t = self.tokens[self.i]
        self.i += 1
        return t

    def expect(self, kind: str, value: Any = None) -> Token:
        if not self.at(kind, value):
            want = value if value is not None else kind
            raise CypherSyntaxError(
                f"Expected {want!r} but found {self.cur.value!r}", self.cur.pos, self.query)
        return self.advance()

    def accept(self, kind: str, value: Any = None) -> Optional[Token]:
        return self.advance() if self.at(kind, value) else None

    # ---------------------------------------------------------------- query

    def parse(self) -> Query:
        clauses: List[Any] = []
        while not self.at('EOF'):
            if self.at_keyword('MATCH'):
                clauses.append(self.parse_match(optional=False))
            elif self.at_keyword('OPTIONAL'):
                self.advance()
                clauses.append(self.parse_match(optional=True))
            elif self.at_keyword('CREATE'):
                self.advance()
                clauses.append(Create(self.parse_pattern_list()))
            elif self.at_keyword('MERGE'):
                clauses.append(self.parse_merge())
            elif self.at_keyword('SET'):
                self.advance()
                clauses.append(SetClause(self.parse_assignments()))
            elif self.at_keyword('REMOVE'):
                self.advance()
                clauses.append(Remove(self.parse_property_list()))
            elif self.at_keyword('DELETE', 'DETACH'):
                clauses.append(self.parse_delete())
            elif self.at_keyword('RETURN'):
                clauses.append(self.parse_return())
            elif self.at_keyword('WITH'):
                clauses.append(self.parse_with())
            elif self.at_keyword('UNWIND'):
                self.advance()
                expr = self.parse_expression()
                self.expect('KEYWORD', 'AS')
                clauses.append(Unwind(expr, self.parse_name()))
            elif self.at_keyword('UNION'):
                raise CypherSyntaxError(
                    "UNION is not supported by this dialect", self.cur.pos, self.query)
            else:
                raise CypherSyntaxError(
                    f"Unexpected {self.cur.value!r}", self.cur.pos, self.query)
        if not clauses:
            raise CypherSyntaxError("Empty query", 0, self.query)
        return Query(clauses)

    def parse_name(self) -> str:
        # A bare keyword is a common slip ('AS count'); say so precisely.
        if self.cur.kind == 'KEYWORD':
            raise CypherSyntaxError(
                f"{self.cur.value!r} is a reserved word; quote it with backticks to use it as a name",
                self.cur.pos, self.query)
        return self.expect('IDENT').value

    # --------------------------------------------------------------- clauses

    def parse_match(self, optional: bool) -> Match:
        self.expect('KEYWORD', 'MATCH')
        patterns = self.parse_pattern_list()
        where = None
        if self.at_keyword('WHERE'):
            self.advance()
            where = self.parse_expression()
        return Match(patterns, where, optional)

    def parse_merge(self) -> Merge:
        self.expect('KEYWORD', 'MERGE')
        pattern = self.parse_pattern()
        on_create: List[Tuple[Property, Any]] = []
        on_match: List[Tuple[Property, Any]] = []
        while self.at_keyword('ON'):
            self.advance()
            if self.at_keyword('CREATE'):
                self.advance()
                self.expect('KEYWORD', 'SET')
                on_create += self.parse_assignments()
            elif self.at_keyword('MATCH'):
                self.advance()
                self.expect('KEYWORD', 'SET')
                on_match += self.parse_assignments()
            else:
                raise CypherSyntaxError("Expected CREATE or MATCH after ON",
                                        self.cur.pos, self.query)
        return Merge(pattern, on_create, on_match)

    def parse_delete(self) -> Delete:
        detach = False
        if self.at_keyword('DETACH'):
            self.advance()
            detach = True
        self.expect('KEYWORD', 'DELETE')
        names = [self.parse_name()]
        while self.accept('PUNCT', ','):
            names.append(self.parse_name())
        return Delete(names, detach)

    def parse_assignments(self) -> List[Tuple[Property, Any]]:
        out = []
        while True:
            prop = self.parse_property_ref()
            self.expect('OP', '=')
            out.append((prop, self.parse_expression()))
            if not self.accept('PUNCT', ','):
                return out

    def parse_property_list(self) -> List[Property]:
        out = [self.parse_property_ref()]
        while self.accept('PUNCT', ','):
            out.append(self.parse_property_ref())
        return out

    def parse_property_ref(self) -> Property:
        var = self.parse_name()
        self.expect('PUNCT', '.')
        return Property(var, self.parse_name())

    def parse_return(self) -> Return:
        self.expect('KEYWORD', 'RETURN')
        distinct = bool(self.accept('KEYWORD', 'DISTINCT'))
        items = self.parse_return_items()
        order, skip, limit = self.parse_tail()
        return Return(items, distinct, order, skip, limit)

    def parse_with(self) -> With:
        self.expect('KEYWORD', 'WITH')
        distinct = bool(self.accept('KEYWORD', 'DISTINCT'))
        items = self.parse_return_items()
        where = None
        if self.at_keyword('WHERE'):
            self.advance()
            where = self.parse_expression()
        order, skip, limit = self.parse_tail()
        return With(items, distinct, where, order, skip, limit)

    def parse_tail(self):
        order: List[OrderItem] = []
        skip = limit = None
        if self.at_keyword('ORDER'):
            self.advance()
            self.expect('KEYWORD', 'BY')
            while True:
                expr = self.parse_expression()
                desc = False
                if self.at_keyword('DESC', 'DESCENDING'):
                    self.advance()
                    desc = True
                elif self.at_keyword('ASC', 'ASCENDING'):
                    self.advance()
                order.append(OrderItem(expr, desc))
                if not self.accept('PUNCT', ','):
                    break
        if self.at_keyword('SKIP'):
            self.advance()
            skip = self.parse_expression()
        if self.at_keyword('LIMIT'):
            self.advance()
            limit = self.parse_expression()
        return order, skip, limit

    def parse_return_items(self) -> List[ReturnItem]:
        items = []
        while True:
            if self.at('OP', '*'):
                self.advance()
                items.append(ReturnItem(Variable('*')))
            else:
                expr = self.parse_expression()
                alias = None
                if self.at_keyword('AS'):
                    self.advance()
                    alias = self.parse_name()
                items.append(ReturnItem(expr, alias))
            if not self.accept('PUNCT', ','):
                return items

    # -------------------------------------------------------------- patterns

    def parse_pattern_list(self) -> List[PathPattern]:
        patterns = [self.parse_pattern()]
        while self.accept('PUNCT', ','):
            patterns.append(self.parse_pattern())
        return patterns

    def parse_pattern(self) -> PathPattern:
        path_var = None
        # p = (a)-[r]->(b)
        if self.cur.kind == 'IDENT' and self.tokens[self.i + 1].kind == 'OP' \
                and self.tokens[self.i + 1].value == '=':
            path_var = self.parse_name()
            self.advance()
        nodes = [self.parse_node_pattern()]
        rels: List[RelPattern] = []
        while self.at('OP', '-') or self.at('OP', '<'):
            rels.append(self.parse_rel_pattern())
            nodes.append(self.parse_node_pattern())
        return PathPattern(nodes, rels, path_var)

    def parse_node_pattern(self) -> NodePattern:
        self.expect('PUNCT', '(')
        var = None
        labels: List[str] = []
        props: Dict[str, Any] = {}
        if self.cur.kind == 'IDENT':
            var = self.advance().value
        while self.accept('PUNCT', ':'):
            labels.append(self.parse_name())
        if self.at('PUNCT', '{'):
            props = self.parse_property_map()
        self.expect('PUNCT', ')')
        return NodePattern(var, labels, props)

    def parse_rel_pattern(self) -> RelPattern:
        direction = 'both'
        if self.accept('OP', '<'):
            self.expect('OP', '-')
            direction = 'in'
        else:
            self.expect('OP', '-')

        var = None
        types: List[str] = []
        props: Dict[str, Any] = {}
        min_hops = max_hops = None

        if self.accept('PUNCT', '['):
            if self.cur.kind == 'IDENT':
                var = self.advance().value
            if self.accept('PUNCT', ':'):
                types.append(self.parse_name())
                while self.accept('PUNCT', '|'):
                    self.accept('PUNCT', ':')
                    types.append(self.parse_name())
            if self.at('OP', '*'):
                self.advance()
                min_hops, max_hops = 1, None
                if self.cur.kind == 'NUMBER':
                    min_hops = self.advance().value
                    max_hops = min_hops
                if self.accept('OP', '..'):
                    max_hops = self.advance().value if self.cur.kind == 'NUMBER' else None
            if self.at('PUNCT', '{'):
                props = self.parse_property_map()
            self.expect('PUNCT', ']')

        if self.accept('OP', '-'):
            if self.accept('OP', '>'):
                if direction == 'in':
                    raise CypherSyntaxError(
                        "A relationship cannot point both ways", self.cur.pos, self.query)
                direction = 'out'
        else:
            self.expect('OP', '-')
        return RelPattern(var, types, direction, props, min_hops, max_hops)

    def parse_property_map(self) -> Dict[str, Any]:
        self.expect('PUNCT', '{')
        props: Dict[str, Any] = {}
        if self.at('PUNCT', '}'):
            self.advance()
            return props
        while True:
            key = self.parse_name()
            self.expect('PUNCT', ':')
            props[key] = self.parse_expression()
            if not self.accept('PUNCT', ','):
                break
        self.expect('PUNCT', '}')
        return props

    # ----------------------------------------------------------- expressions

    def parse_expression(self) -> Any:
        return self.parse_or()

    def parse_or(self) -> Any:
        left = self.parse_xor()
        while self.at_keyword('OR'):
            self.advance()
            left = BoolOp('OR', left, self.parse_xor())
        return left

    def parse_xor(self) -> Any:
        left = self.parse_and()
        while self.at_keyword('XOR'):
            self.advance()
            left = BoolOp('XOR', left, self.parse_and())
        return left

    def parse_and(self) -> Any:
        left = self.parse_not()
        while self.at_keyword('AND'):
            self.advance()
            left = BoolOp('AND', left, self.parse_not())
        return left

    def parse_not(self) -> Any:
        if self.at_keyword('NOT'):
            self.advance()
            return Not(self.parse_not())
        return self.parse_comparison()

    def parse_comparison(self) -> Any:
        left = self.parse_additive()
        while True:
            if self.cur.kind == 'OP' and self.cur.value in ('=', '<>', '<', '<=', '>', '>=', '=~'):
                op = self.advance().value
                left = Comparison(op, left, self.parse_additive())
            elif self.at_keyword('IN'):
                self.advance()
                left = Comparison('IN', left, self.parse_additive())
            elif self.at_keyword('IS'):
                self.advance()
                negated = bool(self.accept('KEYWORD', 'NOT'))
                self.expect('KEYWORD', 'NULL')
                left = IsNull(left, negated)
            elif self.at_keyword('STARTS'):
                self.advance()
                self.expect('KEYWORD', 'WITH')
                left = Comparison('STARTS_WITH', left, self.parse_additive())
            elif self.at_keyword('ENDS'):
                self.advance()
                self.expect('KEYWORD', 'WITH')
                left = Comparison('ENDS_WITH', left, self.parse_additive())
            elif self.at_keyword('CONTAINS'):
                self.advance()
                left = Comparison('CONTAINS', left, self.parse_additive())
            else:
                return left

    def parse_additive(self) -> Any:
        left = self.parse_multiplicative()
        while self.cur.kind == 'OP' and self.cur.value in ('+', '-'):
            op = self.advance().value
            left = Comparison(op, left, self.parse_multiplicative())
        return left

    def parse_multiplicative(self) -> Any:
        left = self.parse_unary()
        while self.cur.kind == 'OP' and self.cur.value in ('*', '/', '%'):
            op = self.advance().value
            left = Comparison(op, left, self.parse_unary())
        return left

    def parse_unary(self) -> Any:
        if self.cur.kind == 'OP' and self.cur.value in ('-', '+'):
            op = self.advance().value
            operand = self.parse_unary()          # right-associative: --a
            # Fold a negated numeric literal so it stays a literal, which keeps
            # numeric comparison detection working for WHERE n.x > -1.
            if op == '-' and isinstance(operand, Literal) \
                    and isinstance(operand.value, (int, float)) \
                    and not isinstance(operand.value, bool):
                return Literal(-operand.value)
            if op == '+':
                return operand
            return UnaryOp(op, operand)
        return self.parse_atom()

    def parse_atom(self) -> Any:
        t = self.cur
        if t.kind == 'NUMBER' or t.kind == 'STRING':
            self.advance()
            return Literal(t.value)
        if t.kind == 'PARAM':
            self.advance()
            return Param(t.value)
        if self.at_keyword('TRUE'):
            self.advance(); return Literal(True)
        if self.at_keyword('FALSE'):
            self.advance(); return Literal(False)
        if self.at_keyword('NULL'):
            self.advance(); return Literal(None)
        if self.at('PUNCT', '('):
            self.advance()
            expr = self.parse_expression()
            self.expect('PUNCT', ')')
            return expr
        if self.at('PUNCT', '['):
            self.advance()
            items = []
            if not self.at('PUNCT', ']'):
                items.append(self.parse_expression())
                while self.accept('PUNCT', ','):
                    items.append(self.parse_expression())
            self.expect('PUNCT', ']')
            return Literal([i.value if isinstance(i, Literal) else i for i in items])
        if t.kind == 'IDENT':
            name = self.advance().value
            if self.at('PUNCT', '('):
                return self.parse_function_call(name)
            if self.accept('PUNCT', '.'):
                return Property(name, self.parse_name())
            return Variable(name)
        raise CypherSyntaxError(f"Unexpected {t.value!r} in expression", t.pos, self.query)

    def parse_function_call(self, name: str) -> FunctionCall:
        self.expect('PUNCT', '(')
        lowered = name.lower()
        if lowered not in AGGREGATES and lowered not in SCALAR_FUNCTIONS:
            raise CypherSyntaxError(
                f"Unsupported function {name!r}. Supported: "
                f"{', '.join(sorted(AGGREGATES | SCALAR_FUNCTIONS))}",
                self.cur.pos, self.query)
        distinct = bool(self.accept('KEYWORD', 'DISTINCT'))
        args: List[Any] = []
        if self.at('OP', '*'):
            self.advance()
            args.append(Variable('*'))
        elif not self.at('PUNCT', ')'):
            args.append(self.parse_expression())
            while self.accept('PUNCT', ','):
                args.append(self.parse_expression())
        self.expect('PUNCT', ')')
        return FunctionCall(lowered, args, distinct)


def parse(query: str) -> Query:
    return Parser(query).parse()
