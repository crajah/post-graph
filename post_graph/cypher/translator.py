"""Translate the parsed Cypher subset into a single SQL statement.

Mapping onto post-graph's model:

  label            a vertex table. ``(n:Person)`` reads the ``person`` table.
  relationship     a row in an edge table; the Cypher type is the
                   ``relation_type`` column, not the table name, so
                   ``[r:KNOWS]`` filters rather than selects a table.
  property         a ``payload`` key, read through a promoted column when the
                   table has one (see promoted.py) and ``payload->>`` otherwise.
  id(n)            the ``id`` column. ``n.uuid`` and ``n.fqid`` reach those
                   columns directly since they are real columns, not payload.

Every literal and parameter becomes a bind parameter. Identifiers are validated
against the same rule the rest of the library uses, so nothing reaches SQL by
interpolation except names that have been checked.
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .. import promoted as _promoted
from .ast import (
    BoolOp,
    Comparison,
    FunctionCall,
    IsNull,
    Literal,
    Match,
    NodePattern,
    Not,
    Param,
    PathPattern,
    Property,
    Query,
    RelPattern,
    Return,
    UnaryOp,
    Variable,
    With,
)

# Columns that exist on the table itself rather than inside payload.
REAL_COLUMNS = {'id', 'uuid', 'fqid', 'realm', 'space', 'created_at', 'updated_at'}
EDGE_REAL_COLUMNS = REAL_COLUMNS | {'from_id', 'to_id', 'relation_type'}

_SQL_COMPARISON = {'=': '=', '<>': '<>', '<': '<', '<=': '<=', '>': '>', '>=': '>=',
                   '+': '+', '-': '-', '*': '*', '/': '/', '%': '%'}


class CypherTranslationError(ValueError):
    """The query parsed but cannot be expressed against this schema."""


class Binding:
    """A pattern variable bound to a SQL alias and the table behind it."""

    def __init__(self, alias: str, table: str, kind: str, promoted: Optional[set] = None):
        self.alias = alias
        self.table = table
        self.kind = kind                      # 'node' | 'rel'
        self.promoted = promoted or set()


class Translator:
    def __init__(
        self,
        realm: str,
        table_ref,
        vertex_tables: Sequence[str],
        edge_tables: Sequence[str],
        promoted_columns: Optional[Dict[str, set]] = None,
        schema_per_realm: bool = False,
        parameters: Optional[Dict[str, Any]] = None,
        edge_schemas: Optional[Dict[str, Tuple[str, str]]] = None,
    ):
        self.realm = realm
        self._table_ref = table_ref
        self.vertex_tables = {t.lower(): t for t in vertex_tables}
        self.edge_tables = list(edge_tables)
        self.promoted_columns = promoted_columns or {}
        self.schema_per_realm = schema_per_realm
        self.parameters = parameters or {}
        self.edge_schemas = edge_schemas or {}
        self._ctes: List[str] = []
        self.json_columns: List[str] = []
        self.params: List[Any] = []
        self.bindings: Dict[str, Binding] = {}
        self._alias_n = 0

    # ------------------------------------------------------------- plumbing

    def _bind(self, value: Any) -> str:
        self.params.append(value)
        return f"${len(self.params)}"

    def _alias(self, prefix: str) -> str:
        self._alias_n += 1
        return f"{prefix}{self._alias_n}"

    def _resolve_label(self, labels: List[str]) -> str:
        if not labels:
            raise CypherTranslationError(
                "A node pattern needs a label so the vertex table is known, e.g. (n:Person). "
                f"Known labels: {', '.join(sorted(self.vertex_tables.values())) or 'none'}")
        if len(labels) > 1:
            raise CypherTranslationError(
                "Multiple labels on one node are not supported: a vertex lives in exactly one table")
        label = labels[0]
        table = self.vertex_tables.get(label.lower())
        if table is None:
            raise CypherTranslationError(
                f"Unknown label {label!r}. Known labels: "
                f"{', '.join(sorted(self.vertex_tables.values())) or 'none'}")
        return table

    # ---------------------------------------------------------- expressions

    def expr(self, node: Any) -> str:
        if isinstance(node, Literal):
            return 'NULL' if node.value is None else self._bind(node.value)
        if isinstance(node, Param):
            # Bound as a value, never interpolated: a parameter is data.
            if node.name not in self.parameters:
                raise CypherTranslationError(f"Parameter ${node.name} was not supplied")
            return self._bind(self.parameters[node.name])
        if isinstance(node, Property):
            return self.property_sql(node)
        if isinstance(node, Variable):
            b = self.bindings.get(node.name)
            if b is None:
                raise CypherTranslationError(f"Unbound variable {node.name!r}")
            return f'{b.alias}.id'
        if isinstance(node, Comparison):
            return self.comparison_sql(node)
        if isinstance(node, IsNull):
            inner = self.expr(node.operand)
            return f"({inner} IS {'NOT ' if node.negated else ''}NULL)"
        if isinstance(node, Not):
            return f"(NOT {self.expr(node.operand)})"
        if isinstance(node, UnaryOp):
            # Payload values are text, so negation has to be numeric explicitly.
            return f"({node.op}({self.expr(node.operand)})::numeric)"
        if isinstance(node, BoolOp):
            left, right = self.expr(node.left), self.expr(node.right)
            if node.op == 'XOR':
                return f"(({left}) <> ({right}))"
            return f"({left} {node.op} {right})"
        if isinstance(node, FunctionCall):
            return self.function_sql(node)
        raise CypherTranslationError(f"Cannot translate {type(node).__name__}")

    def property_sql(self, node: Property) -> str:
        b = self.bindings.get(node.variable)
        if b is None:
            raise CypherTranslationError(f"Unbound variable {node.variable!r}")
        real = EDGE_REAL_COLUMNS if b.kind == 'rel' else REAL_COLUMNS
        if node.key in real:
            return f'{b.alias}."{node.key}"'
        col = _promoted.generic_column(node.key)
        tcol = _promoted.temporal_column(node.key)
        # A promoted column holds exactly what payload->> would return, so using
        # it is an index opportunity rather than a change of meaning.
        if col in b.promoted:
            return f'{b.alias}."{col}"'
        if tcol in b.promoted:
            return f'{b.alias}."{tcol}"'
        return f"{b.alias}.payload->>'{_promoted.validate_key(node.key)}'"

    def comparison_sql(self, node: Comparison) -> str:
        left = self.expr(node.left)
        if node.op == 'IN':
            if isinstance(node.right, Literal) and isinstance(node.right.value, list):
                return f"({left} = ANY({self._bind([str(v) for v in node.right.value])}::text[]))"
            return f"({left} = ANY({self.expr(node.right)}))"
        right = self.expr(node.right)
        if node.op == 'STARTS_WITH':
            return f"({left} LIKE {right} || '%')"
        if node.op == 'ENDS_WITH':
            return f"({left} LIKE '%' || {right})"
        if node.op == 'CONTAINS':
            return f"(POSITION({right} IN {left}) > 0)"
        if node.op == '=~':
            return f"({left} ~ {right})"
        sql_op = _SQL_COMPARISON.get(node.op)
        if sql_op is None:
            raise CypherTranslationError(f"Unsupported operator {node.op!r}")
        # Payload values are text; comparing them to a number must compare
        # numerically or '9' > '10' lexically.
        if sql_op in ('<', '<=', '>', '>=') and self._is_numeric(node.right):
            return f"(({left})::numeric {sql_op} ({right})::numeric)"
        return f"({left} {sql_op} {right})"

    def _is_numeric(self, node: Any) -> bool:
        if isinstance(node, Literal):
            value = node.value
        elif isinstance(node, Param):
            value = self.parameters.get(node.name)
        else:
            return False
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def function_sql(self, node: FunctionCall) -> str:
        name = node.name
        args = node.args
        if name == 'count':
            if args and isinstance(args[0], Variable) and args[0].name == '*':
                return 'COUNT(*)'
            inner = self.expr(args[0])
            return f"COUNT({'DISTINCT ' if node.distinct else ''}{inner})"
        if name in ('sum', 'avg', 'min', 'max'):
            inner = self.expr(args[0])
            numeric = f"({inner})::numeric"
            return f"{name.upper()}({'DISTINCT ' if node.distinct else ''}{numeric})"
        if name == 'collect':
            inner = self.expr(args[0])
            return f"ARRAY_AGG({'DISTINCT ' if node.distinct else ''}{inner})"
        if name == 'id':
            b = self._binding_of(args[0])
            return f'{b.alias}."id"'
        if name == 'type':
            b = self._binding_of(args[0])
            if b.kind != 'rel':
                raise CypherTranslationError("type() takes a relationship")
            return f'{b.alias}."relation_type"'
        if name == 'labels':
            b = self._binding_of(args[0])
            return f'{self._bind([b.table])}::text[]'
        if name == 'properties':
            b = self._binding_of(args[0])
            return f'{b.alias}.payload'
        if name == 'keys':
            b = self._binding_of(args[0])
            return f'ARRAY(SELECT jsonb_object_keys({b.alias}.payload))'
        if name in ('toupper', 'tolower'):
            return f"{'UPPER' if name == 'toupper' else 'LOWER'}({self.expr(args[0])})"
        if name == 'trim':
            return f"BTRIM({self.expr(args[0])})"
        if name in ('length', 'size'):
            return f"LENGTH({self.expr(args[0])})"
        if name == 'coalesce':
            return f"COALESCE({', '.join(self.expr(a) for a in args)})"
        if name == 'tostring':
            return f"({self.expr(args[0])})::text"
        if name == 'tointeger':
            return f"({self.expr(args[0])})::bigint"
        if name == 'tofloat':
            return f"({self.expr(args[0])})::double precision"
        if name in ('abs', 'ceil', 'floor', 'round'):
            return f"{name.upper()}(({self.expr(args[0])})::numeric)"
        if name == 'exists':
            return f"({self.expr(args[0])} IS NOT NULL)"
        raise CypherTranslationError(f"Unsupported function {name!r}")

    def _binding_of(self, node: Any) -> Binding:
        if not isinstance(node, Variable):
            raise CypherTranslationError("Expected a pattern variable")
        b = self.bindings.get(node.name)
        if b is None:
            raise CypherTranslationError(f"Unbound variable {node.name!r}")
        return b

    @staticmethod
    def has_aggregate(node: Any) -> bool:
        if isinstance(node, FunctionCall):
            from .parser import AGGREGATES
            if node.name in AGGREGATES:
                return True
            return any(Translator.has_aggregate(a) for a in node.args)
        if isinstance(node, Comparison):
            return Translator.has_aggregate(node.left) or Translator.has_aggregate(node.right)
        if isinstance(node, BoolOp):
            return Translator.has_aggregate(node.left) or Translator.has_aggregate(node.right)
        if isinstance(node, Not):
            return Translator.has_aggregate(node.operand)
        if isinstance(node, UnaryOp):
            return Translator.has_aggregate(node.operand)
        if isinstance(node, IsNull):
            return Translator.has_aggregate(node.operand)
        return False

    # -------------------------------------------------------------- patterns

    def _edge_candidates(self, from_table: str, to_table: str, direction: str) -> List[Tuple[str, bool]]:
        """Edge tables that can join these two vertex tables.

        Returns (table, reversed) pairs: ``reversed`` means the edge's from_id
        points at the pattern's right-hand node, which is how an incoming
        relationship is satisfied by a table declared the other way round.
        """
        out = []
        for tbl, (ft, tt) in self.edge_schemas.items():
            if direction in ('out', 'both') and ft == from_table and tt == to_table:
                out.append((tbl, False))
            if direction in ('in', 'both') and ft == to_table and tt == from_table:
                out.append((tbl, True))
        return out

    def compile_pattern(self, pattern: PathPattern) -> Tuple[str, List[str]]:
        """Build the FROM/JOIN chain for one path pattern, plus its WHERE terms."""
        conditions: List[str] = []
        first = pattern.nodes[0]
        table = self._resolve_label(first.labels)
        alias = self._alias('n')
        self._register_node(first, alias, table, conditions)
        from_sql = f'{self._table_ref(table, self.realm)} AS {alias}'
        conditions.append(f'{alias}.realm = {self._bind(self.realm)}')

        prev_alias, prev_table = alias, table
        for rel, node in zip(pattern.rels, pattern.nodes[1:]):
            next_table = self._resolve_label(node.labels)
            next_alias = self._alias('n')
            if rel.min_hops is not None:
                from_sql += self._variable_length_join(
                    rel, prev_alias, prev_table, next_alias, next_table, conditions)
            else:
                from_sql += self._fixed_join(
                    rel, prev_alias, prev_table, next_alias, next_table, conditions)
            self._register_node(node, next_alias, next_table, conditions)
            conditions.append(f'{next_alias}.realm = {self._bind(self.realm)}')
            prev_alias, prev_table = next_alias, next_table
        return from_sql, conditions

    def _register_node(self, node: NodePattern, alias: str, table: str,
                       conditions: List[str]) -> None:
        promoted = self.promoted_columns.get(table, set())
        if node.variable:
            if node.variable in self.bindings:
                # Re-using a variable means the same row, e.g. (a)-[]->(b)-[]->(a).
                conditions.append(f'{alias}.id = {self.bindings[node.variable].alias}.id')
            else:
                self.bindings[node.variable] = Binding(alias, table, 'node', promoted)
        else:
            self.bindings[f'__anon_{alias}'] = Binding(alias, table, 'node', promoted)
        for key, value in node.properties.items():
            b = Binding(alias, table, 'node', promoted)
            conditions.append(self._inline_property_eq(b, key, value))

    def _inline_property_eq(self, binding: Binding, key: str, value: Any) -> str:
        saved, self.bindings['__tmp'] = self.bindings.get('__tmp'), binding
        try:
            left = self.property_sql(Property('__tmp', key))
        finally:
            if saved is None:
                self.bindings.pop('__tmp', None)
            else:
                self.bindings['__tmp'] = saved
        return f"({left} = ({self.expr(value)})::text)"

    def _rel_conditions(self, rel: RelPattern, edge_alias: str, table: str) -> List[str]:
        out = [f'{edge_alias}.realm = {self._bind(self.realm)}']
        if rel.types:
            out.append(f'{edge_alias}.relation_type = ANY({self._bind(list(rel.types))}::text[])')
        promoted = self.promoted_columns.get(table, set())
        for key, value in rel.properties.items():
            out.append(self._inline_property_eq(Binding(edge_alias, table, 'rel', promoted), key, value))
        return out

    def _fixed_join(self, rel, prev_alias, prev_table, next_alias, next_table,
                    conditions) -> str:
        cands = self._edge_candidates(prev_table, next_table, rel.direction)
        if not cands:
            raise CypherTranslationError(
                f"No edge table joins {prev_table} to {next_table} "
                f"in direction {rel.direction!r}")
        if len(cands) > 1:
            raise CypherTranslationError(
                f"Ambiguous relationship: {', '.join(t for t, _ in cands)} all join "
                f"{prev_table} to {next_table}. Name the table with a relationship type.")
        table, reversed_ = cands[0]
        edge_alias = self._alias('e')
        promoted = self.promoted_columns.get(table, set())
        if rel.variable:
            self.bindings[rel.variable] = Binding(edge_alias, table, 'rel', promoted)
        near, far = ('to_id', 'from_id') if reversed_ else ('from_id', 'to_id')
        conditions.extend(self._rel_conditions(rel, edge_alias, table))
        return (f' JOIN {self._table_ref(table, self.realm)} AS {edge_alias}'
                f' ON {edge_alias}.{near} = {prev_alias}.id AND {edge_alias}.realm = {prev_alias}.realm'
                f' JOIN {self._table_ref(next_table, self.realm)} AS {next_alias}'
                f' ON {next_alias}.id = {edge_alias}.{far} AND {next_alias}.realm = {edge_alias}.realm')

    def _variable_length_join(self, rel, prev_alias, prev_table, next_alias,
                              next_table, conditions) -> str:
        """Variable-length patterns become a recursive reachability CTE.

        Only same-table walks are supported: ``(a:Person)-[:KNOWS*1..3]->(b:Person)``
        is a walk within one vertex table, which is what a bounded-depth
        reachability query can express without knowing the shape of every
        intermediate hop.
        """
        if prev_table != next_table:
            raise CypherTranslationError(
                "A variable-length pattern must start and end on the same label")
        cands = self._edge_candidates(prev_table, next_table, rel.direction)
        if not cands:
            raise CypherTranslationError(
                f"No edge table joins {prev_table} to itself in direction {rel.direction!r}")
        table, reversed_ = cands[0]
        near, far = ('to_id', 'from_id') if reversed_ else ('from_id', 'to_id')
        min_hops = rel.min_hops if rel.min_hops is not None else 1
        max_hops = rel.max_hops if rel.max_hops is not None else 8   # bounded: no runaway walks
        if max_hops < min_hops:
            raise CypherTranslationError("Variable-length upper bound is below its lower bound")
        if rel.variable:
            raise CypherTranslationError(
                "A variable-length relationship cannot be bound to a variable in this dialect")

        cte = self._alias('vl')
        type_filter = ''
        if rel.types:
            type_filter = f' AND e.relation_type = ANY({self._bind(list(rel.types))}::text[])'
        realm_p = self._bind(self.realm)
        self._ctes.append(f"""{cte} AS (
            SELECT e.{near} AS src, e.{far} AS dst, 1 AS depth
            FROM {self._table_ref(table, self.realm)} e
            WHERE e.realm = {realm_p}{type_filter}
            UNION ALL
            SELECT w.src, e.{far}, w.depth + 1
            FROM {cte} w
            JOIN {self._table_ref(table, self.realm)} e
              ON e.{near} = w.dst AND e.realm = {realm_p}{type_filter}
            WHERE w.depth < {int(max_hops)}
        )""")
        walk_alias = self._alias('w')
        conditions.append(f'{walk_alias}.depth >= {int(min_hops)}')
        return (f' JOIN {cte} AS {walk_alias} ON {walk_alias}.src = {prev_alias}.id'
                f' JOIN {self._table_ref(next_table, self.realm)} AS {next_alias}'
                f' ON {next_alias}.id = {walk_alias}.dst')

    # ------------------------------------------------------- statement build

    def build_select(self, query: Query) -> Tuple[str, List[Any], List[str]]:
        """Assemble MATCH/WHERE/RETURN into one SELECT.

        Returns (sql, params, column_names).
        """
        matches = [c for c in query.clauses if isinstance(c, Match)]
        returns = [c for c in query.clauses if isinstance(c, Return)]
        withs = [c for c in query.clauses if isinstance(c, With)]
        if withs:
            raise CypherTranslationError(
                "WITH is parsed but not yet translated; express the query as a single MATCH/RETURN")
        if not matches:
            raise CypherTranslationError("A read query needs at least one MATCH")
        if len(returns) != 1:
            raise CypherTranslationError("A read query needs exactly one RETURN")
        ret = returns[0]

        from_parts: List[str] = []
        conditions: List[str] = []
        for m in matches:
            if m.optional:
                raise CypherTranslationError(
                    "OPTIONAL MATCH is parsed but not yet translated")
            for pattern in m.patterns:
                if pattern.path_variable:
                    raise CypherTranslationError("Path variables are not supported")
                frm, conds = self.compile_pattern(pattern)
                from_parts.append(frm)
                conditions.extend(conds)
            if m.where is not None:
                conditions.append(self.expr(m.where))

        select_items: List[str] = []
        columns: List[str] = []
        grouped: List[str] = []
        any_aggregate = any(self.has_aggregate(i.expression) for i in ret.items)

        for idx, item in enumerate(ret.items):
            if isinstance(item.expression, Variable) and item.expression.name == '*':
                raise CypherTranslationError(
                    "RETURN * is not supported; name the values you want")
            if isinstance(item.expression, Variable) and item.expression.name in self.bindings:
                # Returning a bare pattern variable yields the whole row as JSON,
                # which is the closest thing to Cypher's node value.
                b = self.bindings[item.expression.name]
                sql = (f"jsonb_build_object('id', {b.alias}.id, 'uuid', {b.alias}.uuid, "
                       f"'label', {self._bind(b.table)}::text, 'properties', {b.alias}.payload)")
                self.json_columns.append(item.alias or self._default_name(item.expression, idx))
            else:
                sql = self.expr(item.expression)
                if isinstance(item.expression, FunctionCall) and item.expression.name == 'properties':
                    self.json_columns.append(item.alias or self._default_name(item.expression, idx))
            name = item.alias or self._default_name(item.expression, idx)
            select_items.append(f'{sql} AS "{name}"')
            columns.append(name)
            if any_aggregate and not self.has_aggregate(item.expression):
                grouped.append(sql)

        sql = 'SELECT '
        if ret.distinct:
            sql += 'DISTINCT '
        sql += ', '.join(select_items)
        sql += ' FROM ' + ' CROSS JOIN '.join(from_parts)
        if conditions:
            sql += ' WHERE ' + ' AND '.join(conditions)
        if grouped:
            sql += ' GROUP BY ' + ', '.join(grouped)
        if ret.order_by:
            terms = []
            for o in ret.order_by:
                # ORDER BY may name a RETURN alias, which SQL resolves for us.
                if isinstance(o.expression, Variable) and o.expression.name in columns:
                    terms.append(f'"{o.expression.name}"{" DESC" if o.descending else ""}')
                else:
                    terms.append(f'{self.expr(o.expression)}{" DESC" if o.descending else ""}')
            sql += ' ORDER BY ' + ', '.join(terms)
        if ret.limit is not None:
            sql += f' LIMIT {self._row_count(ret.limit, "LIMIT")}'
        if ret.skip is not None:
            sql += f' OFFSET {self._row_count(ret.skip, "SKIP")}'
        if self._ctes:
            sql = 'WITH RECURSIVE ' + ', '.join(self._ctes) + ' ' + sql
        return sql, self.params, columns

    def _row_count(self, node: Any, clause: str) -> str:
        """SKIP/LIMIT take a non-negative integer.

        Cypher rejects a negative or fractional row count when the query is
        compiled. Passing it through would surface as a driver error naming a
        SQL clause the caller never wrote, so it is refused here instead.
        """
        value = None
        if isinstance(node, Literal):
            value = node.value
        elif isinstance(node, Param):
            value = self.parameters.get(node.name)
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise CypherTranslationError(f"{clause} requires an integer")
            if isinstance(value, float) and not value.is_integer():
                raise CypherTranslationError(f"{clause} requires an integer, not {value}")
            if value < 0:
                raise CypherTranslationError(f"{clause} must not be negative")
            return self._bind(int(value))
        return self.expr(node)

    @staticmethod
    def _default_name(expression: Any, idx: int) -> str:
        if isinstance(expression, Property):
            return f'{expression.variable}.{expression.key}'
        if isinstance(expression, Variable):
            return expression.name
        if isinstance(expression, FunctionCall):
            return f'{expression.name}(...)'
        return f'col{idx}'
