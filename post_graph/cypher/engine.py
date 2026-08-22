"""Execute Cypher against an AsyncPostGraph client.

Reads become one SQL statement (see translator.py). Writes go through the
client's own methods rather than generated SQL, so audit tables, triggers,
cycle checks and realm rules behave exactly as they do for any other caller —
a Cypher CREATE must not be a second, weaker way into the database.
"""
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .ast import (
    Create, Delete, Literal, Match, Merge, NodePattern, Param, PathPattern,
    Property, Query, Return, SetClause,
)
from .lexer import CypherSyntaxError
from .parser import parse
from .translator import CypherTranslationError, Translator


class CypherSession:
    """A Cypher entry point bound to one client and realm.

    The label→table and relationship→edge-table mapping is discovered from the
    database on first use, so a caller does not have to declare its schema
    twice.
    """

    def __init__(self, client, realm: str, vertex_tables: Optional[Sequence[str]] = None,
                 edge_tables: Optional[Sequence[str]] = None):
        self.client = client
        self.realm = realm
        self._vertex_tables = list(vertex_tables) if vertex_tables else None
        self._edge_tables = list(edge_tables) if edge_tables else None
        self._edge_schemas: Optional[Dict[str, Tuple[str, str]]] = None
        self._promoted: Dict[str, set] = {}

    # ------------------------------------------------------------ discovery

    async def _discover(self) -> None:
        if self._edge_schemas is not None:
            return
        schema = self.realm if self.client.schema_per_realm else 'public'
        rows = await self.client._fetch(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = $1 AND table_type = 'BASE TABLE'", schema)
        names = [r['table_name'] for r in rows]
        # _audit and _data are companions of a base table, not graph elements.
        base = [n for n in names if not n.endswith('_audit') and not n.endswith('_data')]

        edge_schemas: Dict[str, Tuple[str, str]] = {}
        vertex_tables: List[str] = []
        for name in base:
            try:
                s = await self.client.get_edge_schema(name, realm=self.realm)
            except Exception:
                s = None
            if s and 'from_id' in s and 'to_id' in s:
                edge_schemas[name] = (s['from_id'].split('.')[-1].strip('"'),
                                      s['to_id'].split('.')[-1].strip('"'))
            else:
                vertex_tables.append(name)

        if self._vertex_tables is None:
            self._vertex_tables = vertex_tables
        if self._edge_tables is None:
            self._edge_tables = list(edge_schemas)
        self._edge_schemas = {k: v for k, v in edge_schemas.items() if k in self._edge_tables}

        for table in list(self._vertex_tables) + list(self._edge_tables):
            self._promoted[table] = await self.client._promoted_columns(table, self.realm)

    # -------------------------------------------------------------- reading

    async def run(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Run a Cypher query and return rows as dicts."""
        await self._discover()
        ast = parse(query)
        kinds = {type(c).__name__ for c in ast.clauses}
        if kinds & {'Create', 'Merge', 'SetClause', 'Delete', 'Remove'}:
            return await self._run_write(ast, parameters or {})
        return await self._run_read(ast, parameters or {})

    async def explain(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> str:
        """What a query will do, without doing it.

        A read becomes one SQL statement and that statement is returned. A write
        does not: CREATE, MERGE, SET and DELETE are carried out through the
        client's own methods so audit tables and triggers behave, which means
        there is no single statement to show. For those, the plan of client
        operations is returned instead — labelled as operations, because
        presenting them as SQL would misrepresent what runs.
        """
        await self._discover()
        ast = parse(query)
        kinds = {type(c).__name__ for c in ast.clauses}
        if kinds & {'Create', 'Merge', 'SetClause', 'Delete', 'Remove'}:
            return self._explain_write(ast, parameters or {})
        sql, params, _ = self._translate(ast, parameters or {})
        return sql

    def _explain_write(self, ast: Query, parameters: Dict[str, Any]) -> str:
        """Describe the client calls a write query will make."""
        lines = ["-- write query: executed as client operations, not as one SQL statement"]
        for clause in ast.clauses:
            if isinstance(clause, Create):
                for pattern in clause.patterns:
                    for node in pattern.nodes:
                        table = self._label_table(node.labels)
                        payload = {k: self._static_value(v, parameters)
                                   for k, v in node.properties.items()}
                        name = node.variable or '_'
                        lines.append(f"add_vertex({table!r}, realm={self.realm!r}, "
                                     f"payload={payload!r})  -> {name}")
                    for rel, left, right in zip(pattern.rels, pattern.nodes, pattern.nodes[1:]):
                        a, b = left.variable or '_', right.variable or '_'
                        if rel.direction == 'in':
                            a, b = b, a
                        rtype = rel.types[0] if rel.types else '?'
                        payload = {k: self._static_value(v, parameters)
                                   for k, v in rel.properties.items()}
                        try:
                            table = self._edge_table_for(
                                self._label_table(left.labels),
                                self._label_table(right.labels), rel.direction)
                        except CypherTranslationError as exc:
                            table = f"<{exc}>"
                        lines.append(f"add_edge({table!r}, realm={self.realm!r}, "
                                     f"from={a}, to={b}, type={rtype!r}, payload={payload!r})")
            elif isinstance(clause, Merge):
                node = clause.pattern.nodes[0]
                table = self._label_table(node.labels)
                props = {k: self._static_value(v, parameters)
                         for k, v in node.properties.items()}
                lines.append(f"SELECT id FROM {table} WHERE payload matches {props!r}")
                lines.append(f"  if found: upsert_vertex({table!r}) applying ON MATCH SET "
                             f"{[p.key for p, _ in clause.on_match]!r}")
                lines.append(f"  if absent: add_vertex({table!r}, payload={props!r}) "
                             f"plus ON CREATE SET {[p.key for p, _ in clause.on_create]!r}")
            elif isinstance(clause, SetClause):
                for prop, value in clause.assignments:
                    lines.append(f"upsert_vertex(<{prop.variable}>) setting "
                                 f"{prop.key!r} = {self._static_value(value, parameters)!r}")
            elif isinstance(clause, Delete):
                for name in clause.variables:
                    lines.append(f"delete_vertex(<{name}>)"
                                 + ("  [DETACH]" if clause.detach else ""))
        return "\n".join(lines)

    def _translate(self, ast: Query, parameters: Dict[str, Any]):
        self._last_json_columns: List[str] = []
        tr = Translator(
            realm=self.realm,
            table_ref=self.client._get_table_ref,
            vertex_tables=self._vertex_tables,
            edge_tables=self._edge_tables,
            promoted_columns=self._promoted,
            schema_per_realm=self.client.schema_per_realm,
            parameters=parameters,
            edge_schemas=self._edge_schemas,
        )
        result = tr.build_select(ast)
        self._last_json_columns = tr.json_columns
        return result

    async def _run_read(self, ast: Query, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        sql, params, columns = self._translate(ast, parameters)
        rows = await self.client._fetch(sql, *params)
        out = []
        for row in rows:
            d = dict(row)
            # asyncpg hands back jsonb as text; a node value should arrive as a
            # dict, the way a Cypher caller expects.
            for col in self._last_json_columns:
                if isinstance(d.get(col), str):
                    try:
                        d[col] = json.loads(d[col])
                    except (TypeError, ValueError):
                        pass
            out.append(d)
        return out

    # -------------------------------------------------------------- writing

    @staticmethod
    def _static_value(node: Any, parameters: Dict[str, Any]) -> Any:
        if isinstance(node, Literal):
            return node.value
        if isinstance(node, Param):
            if node.name not in parameters:
                raise CypherTranslationError(f"Parameter ${node.name} was not supplied")
            return parameters[node.name]
        raise CypherTranslationError(
            "Only literals and parameters can be written; computed values are not supported")

    def _label_table(self, labels: List[str]) -> str:
        if len(labels) != 1:
            raise CypherTranslationError("A created node needs exactly one label")
        table = {t.lower(): t for t in self._vertex_tables}.get(labels[0].lower())
        if table is None:
            raise CypherTranslationError(f"Unknown label {labels[0]!r}")
        return table

    async def _run_write(self, ast: Query, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        created: Dict[str, Any] = {}
        results: List[Dict[str, Any]] = []

        for clause in ast.clauses:
            if isinstance(clause, Create):
                for pattern in clause.patterns:
                    await self._create_pattern(pattern, parameters, created)
            elif isinstance(clause, Merge):
                await self._merge_pattern(clause, parameters, created)
            elif isinstance(clause, Delete):
                await self._delete(clause, created)
            elif isinstance(clause, SetClause):
                await self._set(clause, parameters, created)
            elif isinstance(clause, Match):
                raise CypherTranslationError(
                    "MATCH combined with a write clause is not supported; "
                    "read first, then write with explicit ids")
            elif isinstance(clause, Return):
                for item in clause.items:
                    name = item.alias or getattr(item.expression, 'name', 'value')
                    var = getattr(item.expression, 'name', None)
                    node = created.get(var)
                    results.append({name: self._node_json(node)})
        return results

    @staticmethod
    def _node_json(node) -> Any:
        if node is None:
            return None
        return {'id': node.id, 'uuid': getattr(node, 'uuid', None),
                'properties': getattr(node, 'payload', None)}

    async def _create_pattern(self, pattern: PathPattern, parameters, created) -> None:
        for node in pattern.nodes:
            if node.variable and node.variable in created:
                continue
            table = self._label_table(node.labels)
            payload = {k: self._static_value(v, parameters) for k, v in node.properties.items()}
            vertex = await self.client.add_vertex(table, self.realm, payload=payload)
            if node.variable:
                created[node.variable] = vertex
            else:
                created[f'__anon{len(created)}'] = vertex

        for rel, left, right in zip(pattern.rels, pattern.nodes, pattern.nodes[1:]):
            if not rel.types:
                raise CypherTranslationError("A created relationship needs a type")
            a = created.get(left.variable) if left.variable else None
            b = created.get(right.variable) if right.variable else None
            if a is None or b is None:
                raise CypherTranslationError(
                    "Both endpoints of a created relationship must be named in the same CREATE")
            if rel.direction == 'in':
                a, b = b, a
            table = self._edge_table_for(
                self._label_table(left.labels), self._label_table(right.labels), rel.direction)
            payload = {k: self._static_value(v, parameters) for k, v in rel.properties.items()}
            edge = await self.client.add_edge(table, self.realm, a.id, b.id, rel.types[0],
                                              payload=payload)
            if rel.variable:
                created[rel.variable] = edge

    def _edge_table_for(self, from_table: str, to_table: str, direction: str) -> str:
        for tbl, (ft, tt) in (self._edge_schemas or {}).items():
            if ft == from_table and tt == to_table:
                return tbl
            if direction == 'in' and ft == to_table and tt == from_table:
                return tbl
        raise CypherTranslationError(f"No edge table joins {from_table} to {to_table}")

    async def _merge_pattern(self, clause: Merge, parameters, created) -> None:
        """MERGE is match-or-create on the pattern's stated properties."""
        pattern = clause.pattern
        if pattern.rels:
            raise CypherTranslationError("MERGE on a relationship pattern is not supported")
        node = pattern.nodes[0]
        table = self._label_table(node.labels)
        props = {k: self._static_value(v, parameters) for k, v in node.properties.items()}
        if not props:
            raise CypherTranslationError("MERGE needs at least one property to match on")
        ref = self.client._get_table_ref(table, self.realm)
        conds, params = ['realm = $1'], [self.realm]
        for key, value in props.items():
            params.append(str(value))
            conds.append(f"payload->>'{key}' = ${len(params)}")
        row = await self.client._fetchrow(
            f"SELECT id FROM {ref} WHERE {' AND '.join(conds)} LIMIT 1", *params)
        if row:
            existing = await self.client.get_vertex(table, self.realm, str(row['id']))
            for prop, value in clause.on_match:
                existing.payload[prop.key] = self._static_value(value, parameters)
            if clause.on_match:
                await self.client.upsert_vertex(table, self.realm, vertex_id=existing.id,
                                                payload=existing.payload)
            if node.variable:
                created[node.variable] = existing
            return
        for prop, value in clause.on_create:
            props[prop.key] = self._static_value(value, parameters)
        vertex = await self.client.add_vertex(table, self.realm, payload=props)
        if node.variable:
            created[node.variable] = vertex

    async def _set(self, clause: SetClause, parameters, created) -> None:
        for prop, value in clause.assignments:
            target = created.get(prop.variable)
            if target is None:
                raise CypherTranslationError(
                    f"SET target {prop.variable!r} must be created in the same query")
            payload = dict(getattr(target, 'payload', {}) or {})
            payload[prop.key] = self._static_value(value, parameters)
            table = getattr(target, 'table_name', None)
            await self.client.upsert_vertex(table, self.realm, vertex_id=target.id, payload=payload)
            target.payload = payload

    async def _delete(self, clause: Delete, created) -> None:
        for name in clause.variables:
            target = created.get(name)
            if target is None:
                raise CypherTranslationError(
                    f"DELETE target {name!r} must be created in the same query")
            table = getattr(target, 'table_name', None)
            await self.client.delete_vertex(table, self.realm, target.id)
