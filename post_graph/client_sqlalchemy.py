import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncConnection
from sqlalchemy.exc import IntegrityError, ProgrammingError

from post_graph.errors import (
    VertexNotFoundError,
    EdgeNotFoundError,
    TableExistsError,
    TableNotFoundError,
    PostGraphError,
)
from post_graph.models import Vertex, Edge, DataRecord

logger = logging.getLogger("post_graph")


class SQLAlchemyPostGraph:
    def __init__(self, engine_or_connection: Union[AsyncEngine, AsyncConnection], schema_per_realm: bool = False):
        self.engine_or_connection = engine_or_connection
        self.schema_per_realm = schema_per_realm
        self._schema_cache = {}  # Cache for edge metadata

    def _validate_identifier(self, identifier: str):
        """Ensure identifiers are safe and valid to prevent SQL injection."""
        if not identifier or not isinstance(identifier, str):
            raise ValueError("Identifier must be a non-empty string.")
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', identifier):
            raise ValueError(f"Invalid identifier name: '{identifier}'. Only alphanumeric characters and underscores are allowed.")

    def _get_table_ref(self, table_name: str, realm: Optional[str] = None) -> str:
        """Construct table reference string based on schema_per_realm setting."""
        if self.schema_per_realm:
            if not realm:
                raise PostGraphError("realm must be specified when schema_per_realm mode is enabled.")
            self._validate_identifier(realm)
            return f'"{realm}"."{table_name}"'
        else:
            return f'"{table_name}"'

    async def _execute(self, conn: AsyncConnection, query: str, **params) -> Any:
        return await conn.execute(text(query), params)

    async def _fetch(self, conn: AsyncConnection, query_str: str, **params) -> List[Dict[str, Any]]:
        """Fetch multiple rows as a list of mapping dictionaries."""
        result = await conn.execute(text(query_str), params)
        rows = result.mappings().all()
        return [dict(r) for r in rows]

    async def _fetchrow(self, conn: AsyncConnection, query_str: str, **params) -> Optional[Dict[str, Any]]:
        """Fetch a single row as a mapping dictionary."""
        result = await conn.execute(text(query_str), params)
        row = result.mappings().first()
        return dict(row) if row else None

    async def _run_in_tx(self, func, user_id: Optional[str] = None):
        """Helper to run a function block within a transaction and optional audit user context."""
        async def _execute_block(conn):
            if user_id:
                await conn.execute(text("SELECT set_config('app.current_user_id', :user_id, true)"), {"user_id": str(user_id)})
            else:
                await conn.execute(text("SELECT set_config('app.current_user_id', '', true)"))
            return await func(conn)

        if isinstance(self.engine_or_connection, AsyncConnection):
            if self.engine_or_connection.in_transaction():
                return await _execute_block(self.engine_or_connection)
            else:
                async with self.engine_or_connection.begin():
                    return await _execute_block(self.engine_or_connection)
        else:
            async with self.engine_or_connection.begin() as conn:
                return await _execute_block(conn)

    async def _table_exists(self, conn: AsyncConnection, table_name: str, realm: Optional[str] = None) -> bool:
        """Check if a table exists in PostgreSQL catalog."""
        self._validate_identifier(table_name)
        if self.schema_per_realm:
            if not realm:
                raise PostGraphError("realm must be specified when checking table existence in schema_per_realm mode.")
            self._validate_identifier(realm)
            query = """
            SELECT 1 
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = :realm AND c.relname = :table_name AND c.relkind = 'r'
            """
            row = await self._fetchrow(conn, query, realm=realm, table_name=table_name)
        else:
            query = """
            SELECT 1 
            FROM pg_class c
            WHERE c.relname = :table_name AND c.relkind = 'r' AND pg_table_is_visible(c.oid)
            """
            row = await self._fetchrow(conn, query, table_name=table_name)
        return row is not None

    async def create_vertex_table(
        self,
        table_name: str,
        realm: Optional[str] = None,
        vector_dim: Optional[int] = None
    ):
        """Create a new vertex table and its associated shadow audit table, indexes, and triggers."""
        self._validate_identifier(table_name)
        if self.schema_per_realm:
            if not realm:
                raise PostGraphError("realm must be specified in schema_per_realm mode when creating a table.")
            self._validate_identifier(realm)

        table_ref = self._get_table_ref(table_name, realm)
        audit_table_ref = self._get_table_ref(f"{table_name}_audit", realm)
        data_table_ref = self._get_table_ref(f"{table_name}_data", realm)

        # Resolve schema prefix for triggers/functions
        if self.schema_per_realm:
            schema_prefix = f'"{realm}".'
        else:
            schema_prefix = ""

        async def _op(conn):
            if self.schema_per_realm:
                await conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{realm}"'))

            # Create trigger function for updated_at
            await conn.execute(text(f"""
                CREATE OR REPLACE FUNCTION {schema_prefix}update_modified_column()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at = now();
                    RETURN NEW;
                END;
                $$ language 'plpgsql';
            """))

            # Create shared trigger function for auditing
            await conn.execute(text(f"""
                CREATE OR REPLACE FUNCTION {schema_prefix}audit_trigger_func()
                RETURNS TRIGGER AS $$
                DECLARE
                    v_user_id TEXT;
                    v_old_row JSONB := NULL;
                    v_new_row JSONB := NULL;
                    v_realm TEXT;
                BEGIN
                    v_user_id := NULLIF(current_setting('app.current_user_id', true), '');
                    
                    IF (TG_OP = 'DELETE') THEN
                        v_old_row := to_jsonb(OLD);
                        v_realm := OLD.realm;
                    ELSIF (TG_OP = 'UPDATE') THEN
                        v_old_row := to_jsonb(OLD);
                        v_new_row := to_jsonb(NEW);
                        v_realm := NEW.realm;
                    ELSIF (TG_OP = 'INSERT') THEN
                        v_new_row := to_jsonb(NEW);
                        v_realm := NEW.realm;
                    END IF;

                    EXECUTE format(
                        'INSERT INTO %I.%I (realm, action, changed_by, old_row, new_row) VALUES ($1, $2, $3, $4, $5)',
                        TG_TABLE_SCHEMA, TG_TABLE_NAME || '_audit'
                    ) USING v_realm, TG_OP, v_user_id, v_old_row, v_new_row;

                    IF (TG_OP = 'DELETE') THEN
                        RETURN OLD;
                    ELSE
                        RETURN NEW;
                    END IF;
                END;
                $$ LANGUAGE plpgsql;
            """))

            # 1. Create main table
            query = f"""
            CREATE TABLE IF NOT EXISTS {table_ref} (
                realm TEXT NOT NULL,
                id BIGSERIAL,
                fqid TEXT GENERATED ALWAYS AS (realm || '/' || '{table_name}' || '/' || id::text) STORED NOT NULL,
                payload JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (realm, id)
            );
            """
            await conn.execute(text(query))
            await conn.execute(text(f"ALTER TABLE {table_ref} ADD COLUMN IF NOT EXISTS fqid TEXT GENERATED ALWAYS AS (realm || '/' || '{table_name}' || '/' || id::text) STORED;"))

            if vector_dim and vector_dim > 0:
                try:
                    await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                    await conn.execute(text(f"ALTER TABLE {table_ref} ADD COLUMN IF NOT EXISTS embedding vector({vector_dim});"))
                    await conn.execute(text(f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_embedding" ON {table_ref} USING hnsw (embedding vector_cosine_ops);'))
                except Exception as e:
                    raise PostGraphError(f"Failed to initialize pgvector extension or embedding column for table '{table_name}': {e}")

            # 2. Create shadow audit table
            audit_query = f"""
            CREATE TABLE IF NOT EXISTS {audit_table_ref} (
                audit_id BIGSERIAL PRIMARY KEY,
                realm TEXT NOT NULL,
                action TEXT NOT NULL,
                changed_by TEXT,
                changed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                old_row JSONB,
                new_row JSONB
            );
            """
            await conn.execute(text(audit_query))

            await conn.execute(text(f'DROP TRIGGER IF EXISTS "update_{table_name}_modtime" ON {table_ref};'))
            await conn.execute(text(f"""
                CREATE TRIGGER "update_{table_name}_modtime"
                BEFORE UPDATE ON {table_ref}
                FOR EACH ROW
                EXECUTE FUNCTION {schema_prefix}update_modified_column();
            """))

            # 5. Create trigger for auditing
            await conn.execute(text(f'DROP TRIGGER IF EXISTS "audit_{table_name}_trigger" ON {table_ref};'))
            await conn.execute(text(f"""
                CREATE TRIGGER "audit_{table_name}_trigger"
                AFTER INSERT OR UPDATE OR DELETE ON {table_ref}
                FOR EACH ROW
                EXECUTE FUNCTION {schema_prefix}audit_trigger_func();
            """))

        if isinstance(self.engine_or_connection, AsyncConnection):
            await _op(self.engine_or_connection)
        else:
            async with self.engine_or_connection.begin() as conn:
                await _op(conn)

    async def create_edge_table(
        self,
        table_name: Optional[str] = None,
        *,
        from_vertex_table: str,
        to_vertex_table: str,
        cascade_delete_from: bool = False,
        cascade_delete_to: bool = False,
        realm: Optional[str] = None
    ):
        """Create a new edge table linking two vertex tables, plus shadow audit table and constraints."""
        # Validate vertex table identifiers
        self._validate_identifier(from_vertex_table)
        self._validate_identifier(to_vertex_table)

        # Determine effective edge table name: default is "{from}TO{to}" if not provided
        if not table_name:
            table_name = f"{from_vertex_table}TO{to_vertex_table}"
        else:
            self._validate_identifier(table_name)

        if self.schema_per_realm:
            if not realm:
                raise PostGraphError("realm must be specified in schema_per_realm mode when creating an edge table.")
            self._validate_identifier(realm)

        table_ref = self._get_table_ref(table_name, realm)
        audit_table_ref = self._get_table_ref(f"{table_name}_audit", realm)
        data_table_ref = self._get_table_ref(f"{table_name}_data", realm)
        from_vertex_ref = self._get_table_ref(from_vertex_table, realm)
        to_vertex_ref = self._get_table_ref(to_vertex_table, realm)

        # Resolve schema prefix for triggers/functions
        if self.schema_per_realm:
            schema_prefix = f'"{realm}".'
        else:
            schema_prefix = ""

        async def _op(conn):
            # Check that from_vertex_table and to_vertex_table exist
            if not await self._table_exists(conn, from_vertex_table, realm=realm):
                raise TableNotFoundError(f"Referenced from_vertex_table '{from_vertex_table}' does not exist.")
            if not await self._table_exists(conn, to_vertex_table, realm=realm):
                raise TableNotFoundError(f"Referenced to_vertex_table '{to_vertex_table}' does not exist.")

            # 1. Create main edge table
            query = f"""
            CREATE TABLE IF NOT EXISTS {table_ref} (
                realm TEXT NOT NULL,
                id BIGSERIAL,
                fqid TEXT GENERATED ALWAYS AS (realm || '/' || '{from_vertex_table}-{to_vertex_table}' || '/' || id::text) STORED NOT NULL,
                from_id BIGINT NOT NULL,
                to_id BIGINT NOT NULL,
                relation_type TEXT NOT NULL,
                payload JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (realm, id),
                FOREIGN KEY (realm, from_id) REFERENCES {from_vertex_ref}(realm, id) ON DELETE CASCADE,
                FOREIGN KEY (realm, to_id) REFERENCES {to_vertex_ref}(realm, id) ON DELETE CASCADE
            );
            """
            await conn.execute(text(query))
            await conn.execute(text(f"ALTER TABLE {table_ref} ADD COLUMN IF NOT EXISTS fqid TEXT GENERATED ALWAYS AS (realm || '/' || '{from_vertex_table}-{to_vertex_table}' || '/' || id::text) STORED;"))

            # 2. Create shadow audit table
            audit_query = f"""
            CREATE TABLE IF NOT EXISTS {audit_table_ref} (
                audit_id BIGSERIAL PRIMARY KEY,
                realm TEXT NOT NULL,
                action TEXT NOT NULL,
                changed_by TEXT,
                changed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                old_row JSONB,
                new_row JSONB
            );
            """
            await conn.execute(text(audit_query))

            # 3. Create append-only data table
            data_query = f"""
            CREATE TABLE IF NOT EXISTS {data_table_ref} (
                data_id BIGSERIAL PRIMARY KEY,
                realm TEXT NOT NULL,
                id BIGINT NOT NULL,
                payload JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (realm, id) REFERENCES {table_ref}(realm, id) ON DELETE CASCADE
            );
            """
            await conn.execute(text(data_query))
            await conn.execute(text(f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_data_id" ON {data_table_ref} (realm, id);'))
            await conn.execute(text(f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_data_payload" ON {data_table_ref} USING gin (payload);'))

            # 3. Create indexes
            await conn.execute(text(
                f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_to" ON {table_ref} (realm, to_id);'
            ))
            await conn.execute(text(
                f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_payload" ON {table_ref} USING gin (payload);'
            ))

            # 4. Create trigger for updated_at
            await conn.execute(text(f'DROP TRIGGER IF EXISTS "update_{table_name}_modtime" ON {table_ref};'))
            await conn.execute(text(f"""
                CREATE TRIGGER "update_{table_name}_modtime"
                BEFORE UPDATE ON {table_ref}
                FOR EACH ROW
                EXECUTE FUNCTION {schema_prefix}update_modified_column();
            """))

            # 5. Create trigger for auditing
            await conn.execute(text(f'DROP TRIGGER IF EXISTS "audit_{table_name}_trigger" ON {table_ref};'))
            await conn.execute(text(f"""
                CREATE TRIGGER "audit_{table_name}_trigger"
                AFTER INSERT OR UPDATE OR DELETE ON {table_ref}
                FOR EACH ROW
                EXECUTE FUNCTION {schema_prefix}audit_trigger_func();
            """))

            # 6. Create custom cascade delete trigger if either boolean is True
            await conn.execute(text(f'DROP TRIGGER IF EXISTS "cascade_delete_trigger_{table_name}" ON {table_ref};'))
            await conn.execute(text(f'DROP FUNCTION IF EXISTS {schema_prefix}"cascade_delete_func_{table_name}"();'))
            if cascade_delete_from or cascade_delete_to:
                from_clause = f'DELETE FROM {from_vertex_ref} WHERE realm = OLD.realm AND id = OLD.from_id;' if cascade_delete_from else ''
                to_clause = f'DELETE FROM {to_vertex_ref} WHERE realm = OLD.realm AND id = OLD.to_id;' if cascade_delete_to else ''
                
                await conn.execute(text(f"""
                    CREATE OR REPLACE FUNCTION {schema_prefix}"cascade_delete_func_{table_name}"()
                    RETURNS TRIGGER AS $$
                    BEGIN
                        {from_clause}
                        {to_clause}
                        RETURN OLD;
                    END;
                    $$ LANGUAGE plpgsql;
                """))
                await conn.execute(text(f"""
                    CREATE TRIGGER "cascade_delete_trigger_{table_name}"
                    AFTER DELETE ON {table_ref}
                    FOR EACH ROW
                    EXECUTE FUNCTION {schema_prefix}"cascade_delete_func_{table_name}"();
                """))

        if isinstance(self.engine_or_connection, AsyncConnection):
            await _op(self.engine_or_connection)
        else:
            async with self.engine_or_connection.begin() as conn:
                await _op(conn)

        # Clear cached schema as a new edge is declared
        cache_key = (realm, table_name) if self.schema_per_realm else table_name
        self._schema_cache.pop(cache_key, None)

    async def add_vertex(
        self,
        table_name: str,
        realm: str,
        vertex_id: Optional[Union[str, int]] = None,
        payload: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
        user_id: Optional[str] = None
    ) -> Vertex:
        """Add a new vertex. Raises TableExistsError if it already exists."""
        self._validate_identifier(table_name)
        payload_json = json.dumps(payload or {})
        table_ref = self._get_table_ref(table_name, realm)
        vec_str = f"[{','.join(str(x) for x in embedding)}]" if embedding else None

        async def _op(conn):
            nonlocal vertex_id
            table_ref_pg = f'"{realm}"."{table_name}"' if self.schema_per_realm else f'"{table_name}"'
            if vertex_id is None:
                seq_query = f"SELECT nextval(pg_get_serial_sequence('{table_ref_pg}', 'id'))"
                v_id_int = (await conn.execute(text(seq_query))).scalar()
            else:
                v_id_int = int(str(vertex_id).split('/')[-1]) if '/' in str(vertex_id) else int(vertex_id)

            v_id_str = str(v_id_int)

            if vec_str:
                query = f"""
                INSERT INTO {table_ref} (realm, id, payload, embedding)
                VALUES (:realm, :id, CAST(:payload AS JSONB), CAST(:vec AS vector))
                RETURNING realm, id, fqid, payload, created_at, updated_at, CAST(embedding AS TEXT) AS embedding_text
                """
                kwargs = {"realm": realm, "id": v_id_int, "payload": payload_json, "vec": vec_str}
            else:
                query = f"""
                INSERT INTO {table_ref} (realm, id, payload)
                VALUES (:realm, :id, CAST(:payload AS JSONB))
                RETURNING realm, id, fqid, payload, created_at, updated_at
                """
                kwargs = {"realm": realm, "id": v_id_int, "payload": payload_json}

            try:
                row = await self._fetchrow(conn, query, **kwargs)
                if vertex_id is not None:
                    await conn.execute(
                        text(f"SELECT setval(pg_get_serial_sequence('{table_ref_pg}', 'id'), (SELECT COALESCE(MAX(id), 1) FROM {table_ref}))")
                    )
                emb = None
                if 'embedding_text' in row and row['embedding_text']:
                    emb = [float(x) for x in row['embedding_text'].strip('[]').split(',') if x.strip()]
                return Vertex(
                    realm=row['realm'],
                    id=str(row['id']),
                    fqid=row['fqid'],
                    payload=row['payload'] if isinstance(row['payload'], dict) else json.loads(row['payload']),
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    table_name=table_name,
                    embedding=emb,
                    _client=self
                )
            except IntegrityError as e:
                if "unique" in str(e).lower() or "duplicate" in str(e).lower():
                    raise TableExistsError(
                        f"Vertex with ID '{v_id_str}' already exists in table '{table_name}' under realm '{realm}'."
                    )
                raise PostGraphError(f"Integrity error: {e}")
            except ProgrammingError as e:
                if "does not exist" in str(e).lower():
                    raise TableNotFoundError(f"Vertex table '{table_name}' does not exist.")
                raise PostGraphError(f"Programming error: {e}")

        return await self._run_in_tx(_op, user_id)

    async def upsert_vertex(
        self,
        table_name: str,
        realm: str,
        vertex_id: Optional[Union[str, int]] = None,
        payload: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
        user_id: Optional[str] = None
    ) -> Vertex:
        """Upsert a vertex (merges payload JSONB on conflict)."""
        self._validate_identifier(table_name)
        payload_json = json.dumps(payload or {})
        table_ref = self._get_table_ref(table_name, realm)
        vec_str = f"[{','.join(str(x) for x in embedding)}]" if embedding else None

        async def _op(conn):
            nonlocal vertex_id
            table_ref_pg = f'"{realm}"."{table_name}"' if self.schema_per_realm else f'"{table_name}"'
            if vertex_id is None:
                seq_query = f"SELECT nextval(pg_get_serial_sequence('{table_ref_pg}', 'id'))"
                v_id_int = (await conn.execute(text(seq_query))).scalar()
            else:
                v_id_int = int(str(vertex_id).split('/')[-1]) if '/' in str(vertex_id) else int(vertex_id)

            v_id_str = str(v_id_int)

            if vec_str:
                query = f"""
                INSERT INTO {table_ref} (realm, id, payload, embedding)
                VALUES (:realm, :id, CAST(:payload AS JSONB), CAST(:vec AS vector))
                ON CONFLICT (realm, id) DO UPDATE
                SET payload = {table_ref}.payload || EXCLUDED.payload,
                    embedding = EXCLUDED.embedding,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING realm, id, fqid, payload, created_at, updated_at, CAST(embedding AS TEXT) AS embedding_text
                """
                kwargs = {"realm": realm, "id": v_id_int, "payload": payload_json, "vec": vec_str}
            else:
                query = f"""
                INSERT INTO {table_ref} (realm, id, payload)
                VALUES (:realm, :id, CAST(:payload AS JSONB))
                ON CONFLICT (realm, id) DO UPDATE
                SET payload = {table_ref}.payload || EXCLUDED.payload,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING realm, id, fqid, payload, created_at, updated_at
                """
                kwargs = {"realm": realm, "id": v_id_int, "payload": payload_json}

            try:
                row = await self._fetchrow(conn, query, **kwargs)
                emb = None
                if 'embedding_text' in row and row['embedding_text']:
                    emb = [float(x) for x in row['embedding_text'].strip('[]').split(',') if x.strip()]
                return Vertex(
                    realm=row['realm'],
                    id=str(row['id']),
                    fqid=row['fqid'],
                    payload=row['payload'] if isinstance(row['payload'], dict) else json.loads(row['payload']),
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    table_name=table_name,
                    embedding=emb,
                    _client=self
                )
            except ProgrammingError as e:
                if "does not exist" in str(e).lower():
                    raise TableNotFoundError(f"Vertex table '{table_name}' does not exist.")
                raise PostGraphError(f"Programming error: {e}")

        return await self._run_in_tx(_op, user_id)

    async def get_vertex(self, table_name: str, realm: str, vertex_id: str) -> Optional[Vertex]:
        """Fetch a vertex by realm and id."""
        self._validate_identifier(table_name)
        table_ref = self._get_table_ref(table_name, realm)
        v_id_int = int(str(vertex_id).split('/')[-1]) if '/' in str(vertex_id) else int(vertex_id)

        async def _op(conn):
            query = f"""
            SELECT t.realm, t.id, t.fqid, t.payload, t.created_at, t.updated_at,
                   to_jsonb(t)->>'embedding' AS embedding_text
            FROM {table_ref} t
            WHERE t.realm = :realm AND t.id = :id
            """
            try:
                row = await self._fetchrow(conn, query, realm=realm, id=v_id_int)
                if not row:
                    return None

                emb = None
                if 'embedding_text' in row and row['embedding_text']:
                    emb = [float(x) for x in row['embedding_text'].strip('[]').split(',') if x.strip()]

                return Vertex(
                    realm=row['realm'],
                    id=str(row['id']),
                    fqid=row['fqid'],
                    payload=row['payload'] if isinstance(row['payload'], dict) else json.loads(row['payload']),
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    table_name=table_name,
                    embedding=emb,
                    _client=self
                )
            except ProgrammingError as e:
                if "does not exist" in str(e).lower():
                    raise TableNotFoundError(f"Vertex table '{table_name}' does not exist.")
                raise PostGraphError(f"Programming error: {e}")

        if isinstance(self.engine_or_connection, AsyncConnection):
            return await _op(self.engine_or_connection)
        else:
            async with self.engine_or_connection.connect() as conn:
                return await _op(conn)

    async def vector_search(
        self,
        table_name: str,
        realm: str,
        query_vector: List[float],
        top_k: int = 5,
        distance_metric: str = "cosine"
    ) -> List[Tuple[Vertex, float]]:
        """Perform vector similarity search on vertex embeddings using pgvector."""
        self._validate_identifier(table_name)
        table_ref = self._get_table_ref(table_name, realm)
        vec_str = f"[{','.join(str(x) for x in query_vector)}]"
        
        op = "<=>"
        if distance_metric == "l2":
            op = "<->"
        elif distance_metric == "inner_product":
            op = "<#>"

        query = f"""
        SELECT realm, id, fqid, payload, created_at, updated_at, CAST(embedding AS TEXT) AS embedding_text,
               (embedding {op} CAST(:vec AS vector)) AS distance
        FROM {table_ref}
        WHERE realm = :realm AND embedding IS NOT NULL
        ORDER BY embedding {op} CAST(:vec AS vector) ASC
        LIMIT :top_k
        """

        async def _op(conn):
            try:
                rows = await self._fetch(conn, query, realm=realm, vec=vec_str, top_k=top_k)
            except ProgrammingError as e:
                if "does not exist" in str(e).lower():
                    raise TableNotFoundError(f"Vertex table '{table_name}' does not exist.")
                logger.warning(f"Vector search failed: {e}")
                return []

            results = []
            for r in rows:
                emb = None
                if r['embedding_text']:
                    emb = [float(x) for x in r['embedding_text'].strip('[]').split(',') if x.strip()]
                v = Vertex(
                    realm=r['realm'],
                    id=str(r['id']),
                    fqid=r['fqid'],
                    payload=r['payload'] if isinstance(r['payload'], dict) else json.loads(r['payload']),
                    created_at=r['created_at'],
                    updated_at=r['updated_at'],
                    table_name=table_name,
                    embedding=emb,
                    _client=self
                )
                dist = float(r['distance'])
                results.append((v, dist))
            return results

        if isinstance(self.engine_or_connection, AsyncConnection):
            return await _op(self.engine_or_connection)
        else:
            async with self.engine_or_connection.connect() as conn:
                return await _op(conn)

    async def delete_vertex(self, table_name: str, realm: str, vertex_id: str, user_id: Optional[str] = None) -> bool:
        """Delete a vertex. Cascading foreign keys will automatically delete referencing edges."""
        self._validate_identifier(table_name)
        table_ref = self._get_table_ref(table_name, realm)
        v_str = str(vertex_id)

        async def _op(conn):
            query = f"""
            DELETE FROM {table_ref} 
            WHERE realm = :realm AND ((CASE WHEN :id ~ '^[0-9]+$' THEN id = CAST(:id AS BIGINT) ELSE FALSE END) OR fqid = :id)
            """
            try:
                res = await conn.execute(text(query), {"realm": realm, "id": v_str})
                return res.rowcount == 1
            except ProgrammingError as e:
                if "does not exist" in str(e).lower():
                    raise TableNotFoundError(f"Vertex table '{table_name}' does not exist.")
                raise PostGraphError(f"Programming error: {e}")

        return await self._run_in_tx(_op, user_id)

    async def add_edge(
        self,
        table_name: str,
        realm: str,
        from_id: str,
        to_id: str,
        relation_type: str,
        edge_id: Optional[Union[str, int]] = None,
        payload: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        check_cycle: Union[bool, List[str]] = False
    ) -> Edge:
        """Add a new edge. Raises TableExistsError if it already exists."""
        self._validate_identifier(table_name)
        payload_json = json.dumps(payload or {})
        table_ref = self._get_table_ref(table_name, realm)

        async def _op(conn):
            nonlocal edge_id
            schema = await self.get_edge_schema(conn, table_name, realm=realm)
            from_table = schema["from_id"]
            to_table = schema["to_id"]

            table_ref_pg = f'"{realm}"."{table_name}"' if self.schema_per_realm else f'"{table_name}"'
            if edge_id is None:
                seq_query = f"SELECT nextval(pg_get_serial_sequence('{table_ref_pg}', 'id'))"
                e_id_int = (await conn.execute(text(seq_query))).scalar()
            else:
                e_id_int = int(str(edge_id).split('/')[-1]) if '/' in str(edge_id) else int(edge_id)

            from_id_int = int(str(from_id).split('/')[-1]) if '/' in str(from_id) else int(from_id)
            to_id_int = int(str(to_id).split('/')[-1]) if '/' in str(to_id) else int(to_id)

            e_id_str = str(e_id_int)

            if check_cycle:
                cycle_tables = check_cycle if isinstance(check_cycle, list) else [table_name]
                
                path = await self.shortest_path(
                    realm=realm,
                    start_table=to_table,
                    start_id=str(to_id_int),
                    target_table=from_table,
                    target_id=str(from_id_int),
                    edge_tables=cycle_tables,
                    conn=conn
                )
                if path:
                    from post_graph.errors import CyclicReferenceError
                    raise CyclicReferenceError(
                        f"Adding edge '{e_id_str}' from '{from_id}' to '{to_id}' would create a cyclic reference. "
                        f"Existing path: {' -> '.join(path['path'])}"
                    )

            query = f"""
            INSERT INTO {table_ref} (realm, id, from_id, to_id, relation_type, payload)
            VALUES (:realm, :id, :from_id, :to_id, :relation_type, CAST(:payload AS JSONB))
            RETURNING realm, id, fqid, from_id, to_id, relation_type, payload, created_at, updated_at
            """
            try:
                row = await self._fetchrow(
                    conn, query,
                    realm=realm, id=e_id_int, from_id=from_id_int, to_id=to_id_int,
                    relation_type=relation_type, payload=payload_json
                )
                if edge_id is not None:
                    await conn.execute(
                        text(f"SELECT setval(pg_get_serial_sequence('{table_ref_pg}', 'id'), (SELECT COALESCE(MAX(id), 1) FROM {table_ref}))")
                    )
                return Edge(
                    realm=row['realm'],
                    id=str(row['id']),
                    fqid=row['fqid'],
                    from_id=str(row['from_id']),
                    to_id=str(row['to_id']),
                    relation_type=row['relation_type'],
                    payload=row['payload'] if isinstance(row['payload'], dict) else json.loads(row['payload']),
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    table_name=table_name,
                    _client=self
                )
            except IntegrityError as e:
                if "unique" in str(e).lower() or "duplicate" in str(e).lower():
                    raise TableExistsError(
                        f"Edge with ID '{e_id_str}' already exists in table '{table_name}' under realm '{realm}'."
                    )
                elif "foreign key" in str(e).lower():
                    raise VertexNotFoundError(f"Foreign key violation: referenced vertices do not exist. Details: {e}")
                raise PostGraphError(f"Integrity error: {e}")
            except ProgrammingError as e:
                if "does not exist" in str(e).lower():
                    raise TableNotFoundError(f"Edge table '{table_name}' does not exist.")
                raise PostGraphError(f"Programming error: {e}")

        return await self._run_in_tx(_op, user_id)

    async def upsert_edge(
        self,
        table_name: str,
        realm: str,
        from_id: str,
        to_id: str,
        relation_type: str,
        edge_id: Optional[Union[str, int]] = None,
        payload: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        check_cycle: Union[bool, List[str]] = False
    ) -> Edge:
        """Upsert an edge (merges payload JSONB on conflict)."""
        self._validate_identifier(table_name)
        payload_json = json.dumps(payload or {})
        table_ref = self._get_table_ref(table_name, realm)

        async def _op(conn):
            nonlocal edge_id
            schema = await self.get_edge_schema(conn, table_name, realm=realm)
            from_table = schema["from_id"]
            to_table = schema["to_id"]

            table_ref_pg = f'"{realm}"."{table_name}"' if self.schema_per_realm else f'"{table_name}"'
            if edge_id is None:
                seq_query = f"SELECT nextval(pg_get_serial_sequence('{table_ref_pg}', 'id'))"
                e_id_int = (await conn.execute(text(seq_query))).scalar()
            else:
                e_id_int = int(str(edge_id).split('/')[-1]) if '/' in str(edge_id) else int(edge_id)

            from_id_int = int(str(from_id).split('/')[-1]) if '/' in str(from_id) else int(from_id)
            to_id_int = int(str(to_id).split('/')[-1]) if '/' in str(to_id) else int(to_id)

            e_id_str = str(e_id_int)

            if check_cycle:
                cycle_tables = check_cycle if isinstance(check_cycle, list) else [table_name]
                
                path = await self.shortest_path(
                    realm=realm,
                    start_table=to_table,
                    start_id=str(to_id_int),
                    target_table=from_table,
                    target_id=str(from_id_int),
                    edge_tables=cycle_tables,
                    conn=conn
                )
                if path:
                    from post_graph.errors import CyclicReferenceError
                    raise CyclicReferenceError(
                        f"Upserting edge '{e_id_str}' from '{from_id}' to '{to_id}' would create a cyclic reference. "
                        f"Existing path: {' -> '.join(path['path'])}"
                    )

            query = f"""
            INSERT INTO {table_ref} (realm, id, from_id, to_id, relation_type, payload)
            VALUES (:realm, :id, :from_id, :to_id, :relation_type, CAST(:payload AS JSONB))
            ON CONFLICT (realm, id) DO UPDATE
            SET from_id = EXCLUDED.from_id,
                to_id = EXCLUDED.to_id,
                relation_type = EXCLUDED.relation_type,
                payload = {table_ref}.payload || EXCLUDED.payload,
                updated_at = CURRENT_TIMESTAMP
            RETURNING realm, id, fqid, from_id, to_id, relation_type, payload, created_at, updated_at
            """
            try:
                row = await self._fetchrow(
                    conn, query,
                    realm=realm, id=e_id_int, from_id=from_id_int, to_id=to_id_int,
                    relation_type=relation_type, payload=payload_json
                )
                return Edge(
                    realm=row['realm'],
                    id=str(row['id']),
                    fqid=row['fqid'],
                    from_id=str(row['from_id']),
                    to_id=str(row['to_id']),
                    relation_type=row['relation_type'],
                    payload=row['payload'] if isinstance(row['payload'], dict) else json.loads(row['payload']),
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    table_name=table_name,
                    _client=self
                )
            except IntegrityError as e:
                if "foreign key" in str(e).lower():
                    raise VertexNotFoundError(f"Foreign key violation: referenced vertices do not exist. Details: {e}")
                raise PostGraphError(f"Integrity error: {e}")
            except ProgrammingError as e:
                if "does not exist" in str(e).lower():
                    raise TableNotFoundError(f"Edge table '{table_name}' does not exist.")
                raise PostGraphError(f"Programming error: {e}")

        return await self._run_in_tx(_op, user_id)

    async def get_edge(self, table_name: str, realm: str, edge_id: str) -> Optional[Edge]:
        """Fetch an edge by realm and id."""
        self._validate_identifier(table_name)
        table_ref = self._get_table_ref(table_name, realm)
        e_str = str(edge_id)
        query = f"""
        SELECT realm, id, fqid, from_id, to_id, relation_type, payload, created_at, updated_at 
        FROM {table_ref} 
        WHERE realm = :realm AND ((CASE WHEN :id ~ '^[0-9]+$' THEN id = CAST(:id AS BIGINT) ELSE FALSE END) OR fqid = :id)
        """
        
        async def _op(conn):
            try:
                row = await self._fetchrow(conn, query, realm=realm, id=e_str)
                if not row:
                    return None
                return Edge(
                    realm=row['realm'],
                    id=str(row['id']),
                    fqid=row['fqid'],
                    from_id=str(row['from_id']),
                    to_id=str(row['to_id']),
                    relation_type=row['relation_type'],
                    payload=row['payload'] if isinstance(row['payload'], dict) else json.loads(row['payload']),
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    table_name=table_name,
                    _client=self
                )
            except ProgrammingError as e:
                if "does not exist" in str(e).lower():
                    raise TableNotFoundError(f"Edge table '{table_name}' does not exist.")
                raise PostGraphError(f"Programming error: {e}")

        if isinstance(self.engine_or_connection, AsyncConnection):
            return await _op(self.engine_or_connection)
        else:
            async with self.engine_or_connection.connect() as conn:
                return await _op(conn)

    async def delete_edge(self, table_name: str, realm: str, edge_id: str, user_id: Optional[str] = None) -> bool:
        """Delete an edge."""
        self._validate_identifier(table_name)
        table_ref = self._get_table_ref(table_name, realm)
        e_str = str(edge_id)

        async def _op(conn):
            query = f"""
            DELETE FROM {table_ref} 
            WHERE realm = :realm AND ((CASE WHEN :id ~ '^[0-9]+$' THEN id = CAST(:id AS BIGINT) ELSE FALSE END) OR fqid = :id)
            """
            try:
                res = await conn.execute(text(query), {"realm": realm, "id": e_str})
                return res.rowcount == 1
            except ProgrammingError as e:
                if "does not exist" in str(e).lower():
                    raise TableNotFoundError(f"Edge table '{table_name}' does not exist.")
                raise PostGraphError(f"Programming error: {e}")

        return await self._run_in_tx(_op, user_id)

    async def delete_realm(self, realm: str, user_id: Optional[str] = None) -> int:
        """Delete all rows belonging to a specific realm from all graph tables (vertices, edges, and audit tables)."""
        async def _op(conn):
            if self.schema_per_realm:
                query = """
                SELECT DISTINCT table_name 
                FROM information_schema.columns 
                WHERE column_name = 'realm' 
                  AND table_schema = :realm
                """
                rows = await self._fetch(conn, query, realm=realm)
            else:
                query = """
                SELECT DISTINCT table_name 
                FROM information_schema.columns 
                WHERE column_name = 'realm' 
                  AND table_schema = CURRENT_SCHEMA()
                """
                rows = await self._fetch(conn, query)

            tables = [row['table_name'] for row in rows]
            
            total_deleted = 0
            for table in tables:
                table_ref = self._get_table_ref(table, realm)
                delete_query = f'DELETE FROM {table_ref} WHERE realm = :realm'
                res = await conn.execute(text(delete_query), {"realm": realm})
                total_deleted += res.rowcount
            return total_deleted

        return await self._run_in_tx(_op, user_id)

    async def get_edge_schema(self, conn: AsyncConnection, edge_table: str, realm: Optional[str] = None) -> Dict[str, str]:
        """
        Query system catalog pg_constraint to discover referenced vertex tables
        for 'from_id' and 'to_id' keys of the edge table. Caches the schema.
        """
        cache_key = (realm, edge_table) if self.schema_per_realm else edge_table
        if cache_key in self._schema_cache:
            return self._schema_cache[cache_key]

        self._validate_identifier(edge_table)
        table_ref = self._get_table_ref(edge_table, realm)

        query = """
        SELECT 
            a.attname AS column_name,
            c.relname AS referenced_table
        FROM pg_constraint con
        JOIN pg_class c ON con.confrelid = c.oid
        JOIN pg_attribute a ON a.attnum = ANY(con.conkey) AND a.attrelid = con.conrelid
        WHERE con.conrelid = CAST(:table_ref AS regclass)
          AND con.contype = 'f'
          AND a.attname IN ('from_id', 'to_id');
        """

        try:
            rows = await self._fetch(conn, query, table_ref=table_ref)
        except ProgrammingError as e:
            if "does not exist" in str(e).lower():
                raise TableNotFoundError(f"Edge table '{edge_table}' does not exist.")
            raise PostGraphError(f"Error querying schema for '{edge_table}': {e}")

        schema = {}
        for row in rows:
            schema[row['column_name']] = row['referenced_table']

        if len(schema) < 2:
            raise PostGraphError(
                f"Edge table '{edge_table}' is missing proper foreign key constraints "
                f"on 'from_id' and/or 'to_id' referencing vertex tables."
            )

        self._schema_cache[cache_key] = schema
        return schema

    async def get_neighbors(
        self,
        realm: str,
        vertex_table: str,
        vertex_id: str,
        edge_tables: List[str],
        direction: str = 'out'
    ) -> List[Tuple[Vertex, Edge]]:
        """
        Get direct neighbor vertices and connecting edges.
        Returns a list of tuples: (NeighborVertex, ConnectingEdge)
        """
        self._validate_identifier(vertex_table)
        if direction not in ('out', 'in', 'both'):
            raise ValueError("Direction must be 'out', 'in', or 'both'")

        v_id_int = int(str(vertex_id).split('/')[-1]) if '/' in str(vertex_id) else int(vertex_id)

        async def _op(conn):
            results = []
            for edge_table in edge_tables:
                self._validate_identifier(edge_table)
                schema = await self.get_edge_schema(conn, edge_table, realm=realm)

                from_ref = schema['from_id']
                to_ref = schema['to_id']

                queries = []

                # 1. Outgoing paths: from_id is the start, to_id is the neighbor
                if direction in ('out', 'both') and from_ref == vertex_table:
                    edge_ref = self._get_table_ref(edge_table, realm)
                    to_ref_ref = self._get_table_ref(to_ref, realm)
                    queries.append((
                        to_ref,
                        f"""
                        SELECT 
                            v.id AS v_id, v.fqid AS v_fqid, v.payload AS v_payload, v.created_at AS v_created_at, v.updated_at AS v_updated_at,
                            e.id AS e_id, e.fqid AS e_fqid, e.from_id AS e_from, e.to_id AS e_to, e.relation_type AS e_rel, e.payload AS e_payload, e.created_at AS e_created_at, e.updated_at AS e_updated_at
                        FROM {edge_ref} e
                        JOIN {to_ref_ref} v ON e.realm = v.realm AND e.to_id = v.id
                        WHERE e.realm = :realm AND e.from_id = :vertex_id
                        """
                    ))

                # 2. Incoming paths: to_id is the start, from_id is the neighbor
                if direction in ('in', 'both') and to_ref == vertex_table:
                    edge_ref = self._get_table_ref(edge_table, realm)
                    from_ref_ref = self._get_table_ref(from_ref, realm)
                    queries.append((
                        from_ref,
                        f"""
                        SELECT 
                            v.id AS v_id, v.fqid AS v_fqid, v.payload AS v_payload, v.created_at AS v_created_at, v.updated_at AS v_updated_at,
                            e.id AS e_id, e.fqid AS e_fqid, e.from_id AS e_from, e.to_id AS e_to, e.relation_type AS e_rel, e.payload AS e_payload, e.created_at AS e_created_at, e.updated_at AS e_updated_at
                        FROM {edge_ref} e
                        JOIN {from_ref_ref} v ON e.realm = v.realm AND e.from_id = v.id
                        WHERE e.realm = :realm AND e.to_id = :vertex_id
                        """
                    ))

                for neighbor_table, sql in queries:
                    rows = await self._fetch(conn, sql, realm=realm, vertex_id=v_id_int)
                    for r in rows:
                        v = Vertex(
                            realm=realm,
                            id=str(r['v_id']),
                            fqid=r['v_fqid'],
                            payload=r['v_payload'] if isinstance(r['v_payload'], dict) else json.loads(r['v_payload']),
                            created_at=r['v_created_at'],
                            updated_at=r['v_updated_at'],
                            table_name=neighbor_table,
                            _client=self
                        )
                        e = Edge(
                            realm=realm,
                            id=str(r['e_id']),
                            fqid=r['e_fqid'],
                            from_id=str(r['e_from']),
                            to_id=str(r['e_to']),
                            relation_type=r['e_rel'],
                            payload=r['e_payload'] if isinstance(r['e_payload'], dict) else json.loads(r['e_payload']),
                            created_at=r['e_created_at'],
                            updated_at=r['e_updated_at'],
                            table_name=edge_table,
                            _client=self
                        )
                        results.append((v, e))

            return results

        if isinstance(self.engine_or_connection, AsyncConnection):
            return await _op(self.engine_or_connection)
        else:
            async with self.engine_or_connection.connect() as conn:
                return await _op(conn)

    async def traverse(
        self,
        realm: str,
        start_table: str,
        start_id: str,
        edge_tables: List[str],
        max_depth: int = 3,
        direction: str = 'out'
    ) -> List[Dict[str, Any]]:
        """
        Perform a dynamic graph traversal starting from (start_table, start_id)
        up to max_depth. scoped to a specific realm.
        """
        self._validate_identifier(start_table)
        if direction not in ('out', 'in', 'both'):
            raise ValueError("Direction must be 'out', 'in', or 'both'")

        start_id_str = str(start_id).split('/')[-1] if '/' in str(start_id) else str(start_id)

        async def _op(conn):
            subqueries = []
            for edge_table in edge_tables:
                schema = await self.get_edge_schema(conn, edge_table, realm=realm)
                from_ref = schema['from_id']
                to_ref = schema['to_id']
                edge_ref = self._get_table_ref(edge_table, realm)

                if direction in ('out', 'both'):
                    subqueries.append(f"""
                    SELECT 
                        CAST(to_id AS text) AS next_id, 
                        CAST('{to_ref}' AS text) AS next_table,
                        CAST(id AS text) AS edge_id,
                        relation_type,
                        payload,
                        CAST('{edge_table}' AS text) AS edge_table
                    FROM {edge_ref}
                    WHERE realm = :realm AND from_id = (CASE WHEN t.current_id ~ '^[0-9]+$' THEN CAST(t.current_id AS BIGINT) ELSE NULL END) AND t.current_table = '{from_ref}'
                    """)

                if direction in ('in', 'both'):
                    subqueries.append(f"""
                    SELECT 
                        CAST(from_id AS text) AS next_id, 
                        CAST('{from_ref}' AS text) AS next_table,
                        CAST(id AS text) AS edge_id,
                        relation_type,
                        payload,
                        CAST('{edge_table}' AS text) AS edge_table
                    FROM {edge_ref}
                    WHERE realm = :realm AND to_id = (CASE WHEN t.current_id ~ '^[0-9]+$' THEN CAST(t.current_id AS BIGINT) ELSE NULL END) AND t.current_table = '{to_ref}'
                    """)

            if not subqueries:
                return []

            union_all_steps = "\nUNION ALL\n".join(subqueries)

            cte_query = f"""
            WITH RECURSIVE graph_traversal AS (
                -- Anchor Member
                SELECT 
                    CAST(:start_id AS text) AS current_id,
                    CAST(:start_table AS text) AS current_table,
                    0 AS depth,
                    ARRAY[CAST(:start_table AS text) || ':' || CAST(:start_id AS text)]::text[] AS path,
                    ARRAY[]::text[] AS edge_path,
                    ARRAY[]::text[] AS edge_ids
                
                UNION ALL
                
                -- Recursive Member
                SELECT 
                    step.next_id,
                    step.next_table,
                    t.depth + 1,
                    t.path || (step.next_table || ':' || step.next_id),
                    t.edge_path || (step.edge_table || ':' || step.relation_type),
                    t.edge_ids || step.edge_id
                FROM graph_traversal t
                CROSS JOIN LATERAL (
                    {union_all_steps}
                ) step
                WHERE t.depth < :max_depth
                  AND NOT ((step.next_table || ':' || step.next_id) = ANY(t.path))
            )
            SELECT current_id, current_table, depth, path, edge_path, edge_ids FROM graph_traversal;
            """

            rows = await self._fetch(
                conn, cte_query,
                realm=realm, start_id=start_id_str, start_table=start_table, max_depth=max_depth
            )

            return [
                {
                    'id': row['current_id'],
                    'table_name': row['current_table'],
                    'depth': row['depth'],
                    'path': row['path'],
                    'edge_path': row['edge_path'],
                    'edge_ids': row['edge_ids']
                }
                for row in rows
            ]

        if isinstance(self.engine_or_connection, AsyncConnection):
            return await _op(self.engine_or_connection)
        else:
            async with self.engine_or_connection.connect() as conn:
                return await _op(conn)

    async def shortest_path(
        self,
        realm: str,
        start_table: str,
        start_id: str,
        target_table: str,
        target_id: str,
        edge_tables: List[str],
        max_depth: int = 5,
        direction: str = 'out',
        conn = None
    ) -> Optional[Dict[str, Any]]:
        """
        Find shortest path from (start_table, start_id) to (target_table, target_id)
        in a specific realm. Returns None if no path exists.
        """
        self._validate_identifier(start_table)
        self._validate_identifier(target_table)
        if direction not in ('out', 'in', 'both'):
            raise ValueError("Direction must be 'out', 'in', or 'both'")

        start_id_str = str(start_id).split('/')[-1] if '/' in str(start_id) else str(start_id)
        target_id_str = str(target_id).split('/')[-1] if '/' in str(target_id) else str(target_id)

        async def _op(c):
            subqueries = []
            for edge_table in edge_tables:
                schema = await self.get_edge_schema(c, edge_table, realm=realm)
                from_ref = schema['from_id']
                to_ref = schema['to_id']
                edge_ref = self._get_table_ref(edge_table, realm)

                if direction in ('out', 'both'):
                    subqueries.append(f"""
                    SELECT 
                        CAST(to_id AS text) AS next_id, 
                        CAST('{to_ref}' AS text) AS next_table,
                        CAST(id AS text) AS edge_id,
                        relation_type,
                        CAST('{edge_table}' AS text) AS edge_table
                    FROM {edge_ref}
                    WHERE realm = :realm AND from_id = (CASE WHEN t.current_id ~ '^[0-9]+$' THEN CAST(t.current_id AS BIGINT) ELSE NULL END) AND t.current_table = '{from_ref}'
                    """)

                if direction in ('in', 'both'):
                    subqueries.append(f"""
                    SELECT 
                        CAST(from_id AS text) AS next_id, 
                        CAST('{from_ref}' AS text) AS next_table,
                        CAST(id AS text) AS edge_id,
                        relation_type,
                        CAST('{edge_table}' AS text) AS edge_table
                    FROM {edge_ref}
                    WHERE realm = :realm AND to_id = (CASE WHEN t.current_id ~ '^[0-9]+$' THEN CAST(t.current_id AS BIGINT) ELSE NULL END) AND t.current_table = '{to_ref}'
                    """)

            if not subqueries:
                return None

            union_all_steps = "\nUNION ALL\n".join(subqueries)

            cte_query = f"""
            WITH RECURSIVE graph_traversal AS (
                -- Anchor Member
                SELECT 
                    CAST(:start_id AS text) AS current_id,
                    CAST(:start_table AS text) AS current_table,
                    0 AS depth,
                    ARRAY[CAST(:start_table AS text) || ':' || CAST(:start_id AS text)]::text[] AS path,
                    ARRAY[]::text[] AS edge_path,
                    ARRAY[]::text[] AS edge_ids
                
                UNION ALL
                
                -- Recursive Member
                SELECT 
                    step.next_id,
                    step.next_table,
                    t.depth + 1,
                    t.path || (step.next_table || ':' || step.next_id),
                    t.edge_path || (step.edge_table || ':' || step.relation_type),
                    t.edge_ids || step.edge_id
                FROM graph_traversal t
                CROSS JOIN LATERAL (
                    {union_all_steps}
                ) step
                WHERE t.depth < :max_depth
                  AND NOT ((step.next_table || ':' || step.next_id) = ANY(t.path))
                  AND NOT (t.current_id = :target_id AND t.current_table = :target_table)
            )
            SELECT depth, path, edge_path, edge_ids
            FROM graph_traversal 
            WHERE current_id = :target_id AND current_table = :target_table
            ORDER BY depth ASC
            LIMIT 1;
            """

            row = await self._fetchrow(
                c, cte_query,
                realm=realm, start_id=start_id_str, start_table=start_table, max_depth=max_depth,
                target_id=target_id_str, target_table=target_table
            )
                
            if not row:
                return None

            return {
                'depth': row['depth'],
                'path': row['path'],
                'edge_path': row['edge_path'],
                'edge_ids': row['edge_ids']
            }

        if conn:
            return await _op(conn)
        else:
            if isinstance(self.engine_or_connection, AsyncConnection):
                return await _op(self.engine_or_connection)
            else:
                async with self.engine_or_connection.connect() as conn:
                    return await _op(conn)

    async def add_vertex_data(
        self,
        table_name: str,
        realm: str,
        vertex_id: Union[str, int],
        payload: Dict[str, Any],
        timestamp: Optional[Any] = None,
        user_id: Optional[str] = None
    ) -> DataRecord:
        """Append a historical record to {table_name}_data table for a vertex."""
        self._validate_identifier(table_name)
        v_id_int = int(str(vertex_id).split('/')[-1]) if '/' in str(vertex_id) else int(vertex_id)
        payload_json = json.dumps(payload or {})
        data_table_ref = self._get_table_ref(f"{table_name}_data", realm)

        async def _op(conn):
            if timestamp:
                query = f"""
                INSERT INTO {data_table_ref} (realm, id, payload, timestamp)
                VALUES (:realm, :id, CAST(:payload AS JSONB), :timestamp)
                RETURNING data_id, realm, id, payload, timestamp
                """
                row = await self._fetchrow(conn, query, realm=realm, id=v_id_int, payload=payload_json, timestamp=timestamp)
            else:
                query = f"""
                INSERT INTO {data_table_ref} (realm, id, payload)
                VALUES (:realm, :id, CAST(:payload AS JSONB))
                RETURNING data_id, realm, id, payload, timestamp
                """
                row = await self._fetchrow(conn, query, realm=realm, id=v_id_int, payload=payload_json)

            return DataRecord(
                data_id=str(row['data_id']),
                realm=row['realm'],
                id=str(row['id']),
                payload=row['payload'] if isinstance(row['payload'], dict) else json.loads(row['payload']),
                timestamp=row['timestamp']
            )

        return await self._run_in_tx(_op, user_id)

    async def get_vertex_data(
        self,
        table_name: str,
        realm: str,
        vertex_id: Union[str, int],
        limit: Optional[int] = None
    ) -> List[DataRecord]:
        """Fetch append-only data records for a vertex sorted by timestamp descending."""
        self._validate_identifier(table_name)
        v_id_int = int(str(vertex_id).split('/')[-1]) if '/' in str(vertex_id) else int(vertex_id)
        data_table_ref = self._get_table_ref(f"{table_name}_data", realm)

        limit_clause = f"LIMIT {limit}" if limit and limit > 0 else ""
        query = f"""
        SELECT data_id, realm, id, payload, timestamp
        FROM {data_table_ref}
        WHERE realm = :realm AND id = :id
        ORDER BY timestamp DESC, data_id DESC
        {limit_clause}
        """

        async def _op(conn):
            rows = await self._fetch(conn, query, realm=realm, id=v_id_int)
            return [
                DataRecord(
                    data_id=str(r['data_id']),
                    realm=r['realm'],
                    id=str(r['id']),
                    payload=r['payload'] if isinstance(r['payload'], dict) else json.loads(r['payload']),
                    timestamp=r['timestamp']
                )
                for r in rows
            ]

        if isinstance(self.engine_or_connection, AsyncConnection):
            return await _op(self.engine_or_connection)
        else:
            async with self.engine_or_connection.connect() as conn:
                return await _op(conn)

    async def add_edge_data(
        self,
        table_name: str,
        realm: str,
        edge_id: Union[str, int],
        payload: Dict[str, Any],
        timestamp: Optional[Any] = None,
        user_id: Optional[str] = None
    ) -> DataRecord:
        """Append a historical record to {table_name}_data table for an edge."""
        return await self.add_vertex_data(table_name, realm, edge_id, payload, timestamp=timestamp, user_id=user_id)

    async def get_edge_data(
        self,
        table_name: str,
        realm: str,
        edge_id: Union[str, int],
        limit: Optional[int] = None
    ) -> List[DataRecord]:
        """Fetch append-only data records for an edge sorted by timestamp descending."""
        return await self.get_vertex_data(table_name, realm, edge_id, limit=limit)

