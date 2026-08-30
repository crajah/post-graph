import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncConnection
from sqlalchemy.exc import DBAPIError, DataError, IntegrityError, ProgrammingError

from post_graph import promoted as _promoted
from post_graph.errors import (
    VertexNotFoundError,
    EdgeNotFoundError,
    TableExistsError,
    TableNotFoundError,
    PostGraphError,
    ReservedSpaceError,
)
from post_graph.models import JSON_NULL, ABSENT, Vertex, Edge, DataRecord

logger = logging.getLogger("post_graph")

RESERVED_SPACE_ALL = "__all__"


class SQLAlchemyPostGraph:
    def __init__(self, engine_or_connection: Union[AsyncEngine, AsyncConnection], schema_per_realm: bool = False):
        self.engine_or_connection = engine_or_connection
        self.schema_per_realm = schema_per_realm
        self._schema_cache = {}
        self._promoted_cache = {}

    @classmethod
    def from_dsn(
        cls,
        dsn: str,
        schema_per_realm: bool = False,
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_timeout: float = 30.0,
        pool_recycle: int = 1800,
        pool_pre_ping: bool = True,
        **engine_kwargs,
    ) -> "SQLAlchemyPostGraph":
        """Create a client with a configured connection pool.

        Wraps ``create_async_engine`` with pool parameters exposed as
        keyword arguments for easy tuning.
        """
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(
            dsn,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=pool_timeout,
            pool_recycle=pool_recycle,
            pool_pre_ping=pool_pre_ping,
            **engine_kwargs,
        )
        return cls(engine, schema_per_realm=schema_per_realm)

    def get_pool_status(self) -> Optional[Dict[str, Any]]:
        """Return live pool status or None if not backed by an engine."""
        if not isinstance(self.engine_or_connection, AsyncEngine):
            return None
        pool = self.engine_or_connection.pool
        return {
            "size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
        }

    def _validate_identifier(self, identifier: str):
        """Ensure identifiers are safe and valid to prevent SQL injection."""
        if not identifier or not isinstance(identifier, str):
            raise ValueError("Identifier must be a non-empty string.")
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', identifier):
            raise ValueError(f"Invalid identifier name: '{identifier}'. Only alphanumeric characters and underscores are allowed.")

    @staticmethod
    def _prepare_filters(filters):
        """Split filters into a containment dict and a list of absent keys.

        Returns ``(containment_json_or_None, absent_keys)``. ``None`` as a
        value is rejected outright: it could mean JSON null, a missing key, or
        "skip this filter", and each reading returns different rows. The
        sentinels say which one is meant.
        """
        containment = {}
        absent = []
        for key, val in (filters or {}).items():
            if val is None:
                raise ValueError(
                    f"Filter value for {key!r} is None, which is ambiguous. "
                    f"Use post_graph.JSON_NULL to match an explicit JSON null, "
                    f"or post_graph.ABSENT to match a key that is not present."
                )
            if val is ABSENT:
                absent.append(key)
            elif val is JSON_NULL:
                containment[key] = None
            else:
                containment[key] = val
        return (json.dumps(containment) if containment else None), absent

    def _get_table_ref(self, table_name: str, realm: Optional[str] = None) -> str:
        """Construct table reference string based on schema_per_realm setting."""
        if self.schema_per_realm:
            if not realm:
                raise PostGraphError("realm must be specified when schema_per_realm mode is enabled.")
            self._validate_identifier(realm)
            return f'"{realm}"."{table_name}"'
        else:
            return f'"{table_name}"'

    async def execute(self, conn: AsyncConnection, query: str, **params) -> Any:
        """Execute a SQL statement."""
        return await conn.execute(text(query), params)

    async def fetch(self, conn: AsyncConnection, query_str: str, **params) -> List[Dict[str, Any]]:
        """Execute a SQL query and return all rows as dicts."""
        result = await conn.execute(text(query_str), params)
        rows = result.mappings().all()
        return [dict(r) for r in rows]

    async def fetchrow(self, conn: AsyncConnection, query_str: str, **params) -> Optional[Dict[str, Any]]:
        """Execute a SQL query and return the first row as a dict."""
        result = await conn.execute(text(query_str), params)
        row = result.mappings().first()
        return dict(row) if row else None

    # Backward-compatible aliases
    _execute = execute
    _fetch = fetch
    _fetchrow = fetchrow

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
        vector_dim: Optional[int] = None,
        vector_columns: Optional[Dict[str, int]] = None,
        temporal_keys: Optional[Sequence[str]] = None,
        promoted_keys: Optional[Sequence[str]] = None,
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
                space VARCHAR(255) DEFAULT 'default' NOT NULL,
                uuid UUID DEFAULT gen_random_uuid() NOT NULL,
                fqid TEXT GENERATED ALWAYS AS (realm || '/' || '{table_name}' || '/' || id::text) STORED NOT NULL,
                payload JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (realm, id)
            );
            """
            await conn.execute(text(query))
            await conn.execute(text(f"ALTER TABLE {table_ref} ADD COLUMN IF NOT EXISTS space VARCHAR(255) DEFAULT 'default';"))
            await conn.execute(text(f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_space" ON {table_ref} (realm, space);'))
            await conn.execute(text(f"ALTER TABLE {table_ref} ADD COLUMN IF NOT EXISTS fqid TEXT GENERATED ALWAYS AS (realm || '/' || '{table_name}' || '/' || id::text) STORED;"))
            await conn.execute(text(f"ALTER TABLE {table_ref} ADD COLUMN IF NOT EXISTS uuid UUID DEFAULT gen_random_uuid();"))
            # Hot payload keys as generated, indexed columns; see promoted.py.
            for _stmt in _promoted.all_column_ddl(table_ref, table_name, temporal_keys, promoted_keys):
                await conn.execute(text(_stmt))
            self._promoted_cache.pop((realm, table_name), None)
            await conn.execute(text(f'CREATE UNIQUE INDEX IF NOT EXISTS "idx_{table_name}_uuid" ON {table_ref} (uuid);'))

            # 2. Create shadow audit table
            audit_query = f"""
            CREATE TABLE IF NOT EXISTS {audit_table_ref} (
                audit_id BIGSERIAL PRIMARY KEY,
                realm TEXT NOT NULL,
                space VARCHAR(255) DEFAULT 'default',
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
                space VARCHAR(255) DEFAULT 'default' NOT NULL,
                payload JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (realm, id) REFERENCES {table_ref}(realm, id) ON DELETE CASCADE
            );
            """
            await conn.execute(text(data_query))
            await conn.execute(text(f"ALTER TABLE {data_table_ref} ADD COLUMN IF NOT EXISTS space VARCHAR(255) DEFAULT 'default';"))
            await conn.execute(text(f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_data_space" ON {data_table_ref} (realm, space);'))
            await conn.execute(text(f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_data_id" ON {data_table_ref} (realm, id);'))
            await conn.execute(text(f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_data_payload" ON {data_table_ref} USING gin (payload);'))

            # 4. Add vector columns. Must run after the data table exists, since the
            # embedding column is added to both the main and the data table.
            if vector_dim and vector_dim > 0:
                try:
                    await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                    await conn.execute(text(f"ALTER TABLE {table_ref} ADD COLUMN IF NOT EXISTS embedding vector({vector_dim});"))
                    await conn.execute(text(f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_embedding" ON {table_ref} USING hnsw (embedding vector_cosine_ops);'))
                    await conn.execute(text(f"ALTER TABLE {data_table_ref} ADD COLUMN IF NOT EXISTS embedding vector({vector_dim});"))
                    await conn.execute(text(f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_data_embedding" ON {data_table_ref} USING hnsw (embedding vector_cosine_ops);'))
                except Exception as e:
                    raise PostGraphError(f"Failed to initialize pgvector extension or embedding column for table '{table_name}': {e}")

            if vector_columns:
                try:
                    await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                    for col_name, dim in vector_columns.items():
                        self._validate_identifier(col_name)
                        await conn.execute(text(f'ALTER TABLE {table_ref} ADD COLUMN IF NOT EXISTS "{col_name}" vector({dim});'))
                        await conn.execute(text(f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_{col_name}" ON {table_ref} USING hnsw ("{col_name}" vector_cosine_ops);'))
                        await conn.execute(text(f'ALTER TABLE {data_table_ref} ADD COLUMN IF NOT EXISTS "{col_name}" vector({dim});'))
                        await conn.execute(text(f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_data_{col_name}" ON {data_table_ref} USING hnsw ("{col_name}" vector_cosine_ops);'))
                except Exception as e:
                    raise PostGraphError(f"Failed to add vector columns to table '{table_name}': {e}")

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
        realm: Optional[str] = None,
        vector_dim: Optional[int] = None,
        vector_columns: Optional[Dict[str, int]] = None,
        temporal_keys: Optional[Sequence[str]] = None,
        promoted_keys: Optional[Sequence[str]] = None,
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
                space VARCHAR(255) DEFAULT 'default' NOT NULL,
                uuid UUID DEFAULT gen_random_uuid() NOT NULL,
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
            await conn.execute(text(f"ALTER TABLE {table_ref} ADD COLUMN IF NOT EXISTS space VARCHAR(255) DEFAULT 'default';"))
            await conn.execute(text(f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_space" ON {table_ref} (realm, space);'))
            await conn.execute(text(f"ALTER TABLE {table_ref} ADD COLUMN IF NOT EXISTS fqid TEXT GENERATED ALWAYS AS (realm || '/' || '{from_vertex_table}-{to_vertex_table}' || '/' || id::text) STORED;"))
            await conn.execute(text(f"ALTER TABLE {table_ref} ADD COLUMN IF NOT EXISTS uuid UUID DEFAULT gen_random_uuid();"))
            # Hot payload keys as generated, indexed columns; see promoted.py.
            for _stmt in _promoted.all_column_ddl(table_ref, table_name, temporal_keys, promoted_keys):
                await conn.execute(text(_stmt))
            self._promoted_cache.pop((realm, table_name), None)
            await conn.execute(text(f'CREATE UNIQUE INDEX IF NOT EXISTS "idx_{table_name}_uuid" ON {table_ref} (uuid);'))

            # 2. Create shadow audit table
            audit_query = f"""
            CREATE TABLE IF NOT EXISTS {audit_table_ref} (
                audit_id BIGSERIAL PRIMARY KEY,
                realm TEXT NOT NULL,
                space VARCHAR(255) DEFAULT 'default',
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
                space VARCHAR(255) DEFAULT 'default' NOT NULL,
                payload JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (realm, id) REFERENCES {table_ref}(realm, id) ON DELETE CASCADE
            );
            """
            await conn.execute(text(data_query))
            await conn.execute(text(f"ALTER TABLE {data_table_ref} ADD COLUMN IF NOT EXISTS space VARCHAR(255) DEFAULT 'default';"))
            await conn.execute(text(f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_data_space" ON {data_table_ref} (realm, space);'))
            await conn.execute(text(f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_data_id" ON {data_table_ref} (realm, id);'))
            await conn.execute(text(f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_data_payload" ON {data_table_ref} USING gin (payload);'))

            # 3. Create indexes
            await conn.execute(text(
                f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_to" ON {table_ref} (realm, to_id);'
            ))
            await conn.execute(text(
                f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_payload" ON {table_ref} USING gin (payload);'
            ))

            # 3a. Add vector columns
            if vector_dim and vector_dim > 0:
                try:
                    await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                    await conn.execute(text(f"ALTER TABLE {table_ref} ADD COLUMN IF NOT EXISTS embedding vector({vector_dim});"))
                    await conn.execute(text(f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_embedding" ON {table_ref} USING hnsw (embedding vector_cosine_ops);'))
                    await conn.execute(text(f"ALTER TABLE {data_table_ref} ADD COLUMN IF NOT EXISTS embedding vector({vector_dim});"))
                    await conn.execute(text(f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_data_embedding" ON {data_table_ref} USING hnsw (embedding vector_cosine_ops);'))
                except Exception as e:
                    raise PostGraphError(f"Failed to initialize pgvector extension or embedding column for edge table '{table_name}': {e}")

            if vector_columns:
                try:
                    await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                    for col_name, dim in vector_columns.items():
                        self._validate_identifier(col_name)
                        await conn.execute(text(f'ALTER TABLE {table_ref} ADD COLUMN IF NOT EXISTS "{col_name}" vector({dim});'))
                        await conn.execute(text(f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_{col_name}" ON {table_ref} USING hnsw ("{col_name}" vector_cosine_ops);'))
                        await conn.execute(text(f'ALTER TABLE {data_table_ref} ADD COLUMN IF NOT EXISTS "{col_name}" vector({dim});'))
                        await conn.execute(text(f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_data_{col_name}" ON {data_table_ref} USING hnsw ("{col_name}" vector_cosine_ops);'))
                except Exception as e:
                    raise PostGraphError(f"Failed to add vector columns to edge table '{table_name}': {e}")

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
        embeddings: Optional[Dict[str, List[float]]] = None,
        user_id: Optional[str] = None,
        space: Optional[str] = "default"
    ) -> Vertex:
        """Add a new vertex. Raises TableExistsError if it already exists."""
        self._validate_identifier(table_name)
        if space == RESERVED_SPACE_ALL:
            raise ReservedSpaceError(f"'{RESERVED_SPACE_ALL}' is a reserved space name and cannot be used for creation. It is only valid as a query-time filter.")
        payload_json = json.dumps(payload or {})
        table_ref = self._get_table_ref(table_name, realm)
        vec_str = f"[{','.join(str(x) for x in embedding)}]" if embedding else None
        eff_space = space or "default"

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
                INSERT INTO {table_ref} (realm, id, space, payload, embedding)
                VALUES (:realm, :id, :space, CAST(:payload AS JSONB), CAST(:vec AS vector))
                RETURNING realm, id, space, fqid, payload, created_at, updated_at, uuid::text AS uuid_text, CAST(embedding AS TEXT) AS embedding_text
                """
                kwargs = {"realm": realm, "id": v_id_int, "space": eff_space, "payload": payload_json, "vec": vec_str}
            else:
                query = f"""
                INSERT INTO {table_ref} (realm, id, space, payload)
                VALUES (:realm, :id, :space, CAST(:payload AS JSONB))
                RETURNING realm, id, space, fqid, payload, created_at, updated_at, uuid::text AS uuid_text
                """
                kwargs = {"realm": realm, "id": v_id_int, "space": eff_space, "payload": payload_json}

            try:
                row = await self._fetchrow(conn, query, **kwargs)
                if vertex_id is not None:
                    await conn.execute(
                        text(f"SELECT setval(pg_get_serial_sequence('{table_ref_pg}', 'id'), (SELECT COALESCE(MAX(id), 1) FROM {table_ref}))")
                    )

                emb_dict = None
                if embeddings:
                    set_parts = []
                    up_params: Dict[str, Any] = {"_emb_realm": realm, "_emb_id": row['id']}
                    for i, (col_name, vec) in enumerate(embeddings.items()):
                        self._validate_identifier(col_name)
                        pname = f"_emb_v{i}"
                        up_params[pname] = f"[{','.join(str(x) for x in vec)}]"
                        set_parts.append(f'"{col_name}" = CAST(:{pname} AS vector)')
                    if set_parts:
                        await conn.execute(
                            text(f"UPDATE {table_ref} SET {', '.join(set_parts)} WHERE realm = :_emb_realm AND id = :_emb_id"),
                            up_params,
                        )
                        emb_dict = dict(embeddings)

                emb = None
                if 'embedding_text' in row and row['embedding_text']:
                    emb = [float(x) for x in row['embedding_text'].strip('[]').split(',') if x.strip()]
                return Vertex(
                    realm=row['realm'],
                    id=str(row['id']),
                    space=row.get('space') or 'default',
                    fqid=row['fqid'],
                    payload=row['payload'] if isinstance(row['payload'], dict) else json.loads(row['payload']),
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    table_name=table_name,
                    embedding=emb,
                    embeddings=emb_dict,
                    uuid=str(row['uuid_text']) if row.get('uuid_text') else None,
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
        embeddings: Optional[Dict[str, List[float]]] = None,
        user_id: Optional[str] = None,
        space: Optional[str] = "default"
    ) -> Vertex:
        """Upsert a vertex (merges payload JSONB on conflict)."""
        self._validate_identifier(table_name)
        if space == RESERVED_SPACE_ALL:
            raise ReservedSpaceError(f"'{RESERVED_SPACE_ALL}' is a reserved space name and cannot be used for creation. It is only valid as a query-time filter.")
        payload_json = json.dumps(payload or {})
        table_ref = self._get_table_ref(table_name, realm)
        vec_str = f"[{','.join(str(x) for x in embedding)}]" if embedding else None
        eff_space = space or "default"

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
                INSERT INTO {table_ref} (realm, id, space, payload, embedding)
                VALUES (:realm, :id, :space, CAST(:payload AS JSONB), CAST(:vec AS vector))
                ON CONFLICT (realm, id) DO UPDATE
                SET space = EXCLUDED.space,
                    payload = {table_ref}.payload || EXCLUDED.payload,
                    embedding = EXCLUDED.embedding,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING realm, id, space, fqid, payload, created_at, updated_at, uuid::text AS uuid_text, CAST(embedding AS TEXT) AS embedding_text
                """
                kwargs = {"realm": realm, "id": v_id_int, "space": eff_space, "payload": payload_json, "vec": vec_str}
            else:
                query = f"""
                INSERT INTO {table_ref} (realm, id, space, payload)
                VALUES (:realm, :id, :space, CAST(:payload AS JSONB))
                ON CONFLICT (realm, id) DO UPDATE
                SET space = EXCLUDED.space,
                    payload = {table_ref}.payload || EXCLUDED.payload,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING realm, id, space, fqid, payload, created_at, updated_at, uuid::text AS uuid_text
                """
                kwargs = {"realm": realm, "id": v_id_int, "space": eff_space, "payload": payload_json}

            try:
                row = await self._fetchrow(conn, query, **kwargs)

                emb_dict = None
                if embeddings:
                    set_parts = []
                    up_params: Dict[str, Any] = {"_emb_realm": realm, "_emb_id": row['id']}
                    for i, (col_name, vec) in enumerate(embeddings.items()):
                        self._validate_identifier(col_name)
                        pname = f"_emb_v{i}"
                        up_params[pname] = f"[{','.join(str(x) for x in vec)}]"
                        set_parts.append(f'"{col_name}" = CAST(:{pname} AS vector)')
                    if set_parts:
                        await conn.execute(
                            text(f"UPDATE {table_ref} SET {', '.join(set_parts)} WHERE realm = :_emb_realm AND id = :_emb_id"),
                            up_params,
                        )
                        emb_dict = dict(embeddings)

                emb = None
                if 'embedding_text' in row and row['embedding_text']:
                    emb = [float(x) for x in row['embedding_text'].strip('[]').split(',') if x.strip()]
                return Vertex(
                    realm=row['realm'],
                    id=str(row['id']),
                    space=row.get('space') or 'default',
                    fqid=row['fqid'],
                    payload=row['payload'] if isinstance(row['payload'], dict) else json.loads(row['payload']),
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    table_name=table_name,
                    embedding=emb,
                    embeddings=emb_dict,
                    uuid=str(row['uuid_text']) if row.get('uuid_text') else None,
                    _client=self
                )
            except ProgrammingError as e:
                if "does not exist" in str(e).lower():
                    raise TableNotFoundError(f"Vertex table '{table_name}' does not exist.")
                raise PostGraphError(f"Programming error: {e}")

        return await self._run_in_tx(_op, user_id)

    async def batch_upsert_vertices(
        self,
        table_name: str,
        realm: str,
        items: List[Dict[str, Any]],
        user_id: Optional[str] = None,
    ) -> List[Vertex]:
        """Upsert multiple vertices in a single transaction.

        Each dict in *items* may contain: ``vertex_id``, ``payload``,
        ``embedding``, ``space``.
        """
        results = []
        for item in items:
            v = await self.upsert_vertex(
                table_name,
                realm=realm,
                vertex_id=item.get("vertex_id"),
                payload=item.get("payload"),
                embedding=item.get("embedding"),
                space=item.get("space", "default"),
                user_id=user_id,
            )
            results.append(v)
        return results

    async def batch_upsert_edges(
        self,
        table_name: str,
        realm: str,
        items: List[Dict[str, Any]],
        user_id: Optional[str] = None,
    ) -> List[Edge]:
        """Upsert multiple edges in a single transaction.

        Each dict in *items* must contain ``from_id``, ``to_id``, and
        ``relation_type``.  Optional: ``edge_id``, ``payload``, ``space``.
        """
        results = []
        for item in items:
            e = await self.upsert_edge(
                table_name,
                realm=realm,
                from_id=item["from_id"],
                to_id=item["to_id"],
                relation_type=item["relation_type"],
                edge_id=item.get("edge_id"),
                payload=item.get("payload"),
                space=item.get("space", "default"),
                user_id=user_id,
            )
            results.append(e)
        return results

    async def get_vertex(self, table_name: str, realm: str, vertex_id: str, strict: bool = False) -> Optional[Vertex]:
        """Fetch a vertex by realm and id or uuid.

        When *strict* is True, raises VertexNotFoundError instead of returning None.
        """
        self._validate_identifier(table_name)
        v_str = str(vertex_id).strip()

        if len(v_str) == 36 and '-' in v_str:
            result = await self.get_vertex_by_uuid(table_name, realm, v_str)
            if strict and result is None:
                raise VertexNotFoundError(f"Vertex '{vertex_id}' not found in table '{table_name}', realm '{realm}'.")
            return result

        try:
            v_id_int = int(v_str.split('/')[-1]) if '/' in v_str else int(v_str)
        except ValueError:
            result = await self.get_vertex_by_uuid(table_name, realm, v_str)
            if strict and result is None:
                raise VertexNotFoundError(f"Vertex '{vertex_id}' not found in table '{table_name}', realm '{realm}'.")
            return result

        table_ref = self._get_table_ref(table_name, realm)

        async def _op(conn):
            query = f"""
            SELECT t.realm, t.id, t.space, t.fqid, t.payload, t.created_at, t.updated_at,
                   t.uuid::text AS uuid_text,
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
                    space=row.get('space') or 'default',
                    fqid=row['fqid'],
                    payload=row['payload'] if isinstance(row['payload'], dict) else json.loads(row['payload']),
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    table_name=table_name,
                    embedding=emb,
                    uuid=str(row['uuid_text']) if row.get('uuid_text') else None,
                    _client=self
                )
            except ProgrammingError as e:
                if "does not exist" in str(e).lower():
                    raise TableNotFoundError(f"Vertex table '{table_name}' does not exist.")
                raise PostGraphError(f"Programming error: {e}")

        if isinstance(self.engine_or_connection, AsyncConnection):
            result = await _op(self.engine_or_connection)
        else:
            async with self.engine_or_connection.connect() as conn:
                result = await _op(conn)
        if strict and result is None:
            raise VertexNotFoundError(f"Vertex '{vertex_id}' not found in table '{table_name}', realm '{realm}'.")
        return result

    async def get_vertices(
        self,
        table_name: str,
        realm: str,
        space: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Vertex]:
        """Fetch all vertices in a realm, optionally filtered by space."""
        self._validate_identifier(table_name)
        table_ref = self._get_table_ref(table_name, realm)

        async def _op(conn):
            params = {"realm": realm}
            space_clause = ""
            if space and space != RESERVED_SPACE_ALL:
                params["space"] = space
                space_clause = " AND (t.space = :space OR (:space = 'default' AND (t.space IS NULL OR t.space = 'default')))"

            limit_clause = ""
            if limit:
                params["limit"] = limit
                limit_clause = " LIMIT :limit"

            query = f"""
            SELECT t.realm, t.id, t.space, t.fqid, t.payload, t.created_at, t.updated_at,
                   t.uuid::text AS uuid_text,
                   to_jsonb(t)->>'embedding' AS embedding_text
            FROM {table_ref} t
            WHERE t.realm = :realm{space_clause}
            ORDER BY t.id ASC{limit_clause}
            """
            try:
                result = await conn.execute(text(query), params)
                rows = result.mappings().all()
                vertices = []
                for row in rows:
                    emb = None
                    if 'embedding_text' in row and row['embedding_text']:
                        emb = [float(x) for x in row['embedding_text'].strip('[]').split(',') if x.strip()]
                    vertices.append(Vertex(
                        realm=row['realm'],
                        id=str(row['id']),
                        space=row.get('space') or 'default',
                        fqid=row['fqid'],
                        payload=row['payload'] if isinstance(row['payload'], dict) else json.loads(row['payload']),
                        created_at=row['created_at'],
                        updated_at=row['updated_at'],
                        table_name=table_name,
                        embedding=emb,
                        uuid=str(row['uuid_text']) if row.get('uuid_text') else None,
                        _client=self
                    ))
                return vertices
            except ProgrammingError:
                raise TableNotFoundError(f"Vertex table '{table_name}' does not exist.")

        if isinstance(self.engine_or_connection, AsyncConnection):
            return await _op(self.engine_or_connection)
        else:
            async with self.engine_or_connection.connect() as conn:
                return await _op(conn)

    async def get_vertices_multi_realm(
        self,
        table_name: str,
        realms: List[str],
        space: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Vertex]:
        """Fetch vertices across multiple realms."""
        self._validate_identifier(table_name)

        async def _op(conn):
            if self.schema_per_realm:
                parts = []
                params: Dict[str, Any] = {}
                for i, r in enumerate(realms):
                    tref = self._get_table_ref(table_name, r)
                    rp = f"r{i}"
                    params[rp] = r
                    space_clause = ""
                    if space and space != RESERVED_SPACE_ALL:
                        sp = f"sp{i}"
                        params[sp] = space
                        space_clause = f" AND (t.space = :{sp} OR (:{sp} = 'default' AND (t.space IS NULL OR t.space = 'default')))"
                    parts.append(f"SELECT t.realm, t.id, t.space, t.fqid, t.payload, t.created_at, t.updated_at, t.uuid::text AS uuid_text, to_jsonb(t)->>'embedding' AS embedding_text FROM {tref} t WHERE t.realm = :{rp}{space_clause}")
                query = " UNION ALL ".join(parts) + " ORDER BY realm, id ASC"
                if limit:
                    params["limit"] = limit
                    query += " LIMIT :limit"
            else:
                table_ref = self._get_table_ref(table_name, realms[0])
                params = {"realms": realms}
                space_clause = ""
                if space and space != RESERVED_SPACE_ALL:
                    params["space"] = space
                    space_clause = " AND (t.space = :space OR (:space = 'default' AND (t.space IS NULL OR t.space = 'default')))"
                limit_clause = ""
                if limit:
                    params["limit"] = limit
                    limit_clause = " LIMIT :limit"
                query = f"""
                SELECT t.realm, t.id, t.space, t.fqid, t.payload, t.created_at, t.updated_at,
                       t.uuid::text AS uuid_text,
                       to_jsonb(t)->>'embedding' AS embedding_text
                FROM {table_ref} t
                WHERE t.realm = ANY(:realms){space_clause}
                ORDER BY t.realm, t.id ASC{limit_clause}
                """
            try:
                result = await conn.execute(text(query), params)
                rows = result.mappings().all()
                vertices = []
                for row in rows:
                    emb = None
                    if 'embedding_text' in row and row['embedding_text']:
                        emb = [float(x) for x in row['embedding_text'].strip('[]').split(',') if x.strip()]
                    vertices.append(Vertex(
                        realm=row['realm'],
                        id=str(row['id']),
                        space=row.get('space') or 'default',
                        fqid=row['fqid'],
                        payload=row['payload'] if isinstance(row['payload'], dict) else json.loads(row['payload']),
                        created_at=row['created_at'],
                        updated_at=row['updated_at'],
                        table_name=table_name,
                        embedding=emb,
                        uuid=str(row['uuid_text']) if row.get('uuid_text') else None,
                        _client=self
                    ))
                return vertices
            except ProgrammingError:
                raise TableNotFoundError(f"Vertex table '{table_name}' does not exist.")

        if isinstance(self.engine_or_connection, AsyncConnection):
            return await _op(self.engine_or_connection)
        else:
            async with self.engine_or_connection.connect() as conn:
                return await _op(conn)

    async def find_vertices(
        self,
        table_name: str,
        realm: str,
        filters: Dict[str, Any],
        space: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Vertex]:
        """Find vertices whose payload matches the given key-value filters.

        Matching is by JSONB containment (``payload @> filters``), which makes
        it type-sensitive: ``True`` matches JSON ``true``, and the number
        ``42`` matches ``42`` but not the string ``"42"``. Keys never enter
        the SQL text, so any JSON key is legal, and the comparison uses the
        payload GIN index.

        Containment, not equality, for nested values: a filter of
        ``{"tags": ["a"]}`` matches a payload whose ``tags`` is
        ``["a", "b"]``. Scalar filters are unaffected.

        ``None`` is rejected as a filter value because it is ambiguous. Use
        :data:`post_graph.JSON_NULL` to match a key holding an explicit JSON
        null, and :data:`post_graph.ABSENT` to match a key that is not present
        --- those are different states, and each has its own name.
        """
        self._validate_identifier(table_name)
        table_ref = self._get_table_ref(table_name, realm)

        async def _op(conn):
            params: Dict[str, Any] = {"realm": realm}
            clauses = ""
            if space and space != RESERVED_SPACE_ALL:
                params["space"] = space
                clauses += " AND (t.space = :space OR (:space = 'default' AND (t.space IS NULL OR t.space = 'default')))"
            # Containment, for the same reasons as the asyncpg client: the
            # previous per-key text comparison could never match a boolean, and
            # this form keeps keys out of the SQL and uses the GIN index.
            containment, absent_keys = self._prepare_filters(filters)
            if containment is not None:
                params["fjson"] = containment
                clauses += " AND t.payload @> CAST(:fjson AS jsonb)"
            for i, key in enumerate(absent_keys):
                pname = f"fabs{i}"
                params[pname] = key
                clauses += f" AND NOT jsonb_exists(t.payload, :{pname})"
            limit_clause = ""
            if limit:
                params["limit"] = limit
                limit_clause = " LIMIT :limit"
            query = f"""
            SELECT t.realm, t.id, t.space, t.fqid, t.payload, t.created_at, t.updated_at,
                   t.uuid::text AS uuid_text,
                   to_jsonb(t)->>'embedding' AS embedding_text
            FROM {table_ref} t
            WHERE t.realm = :realm{clauses}
            ORDER BY t.id ASC{limit_clause}
            """
            try:
                result = await conn.execute(text(query), params)
                rows = result.mappings().all()
                vertices = []
                for row in rows:
                    emb = None
                    if 'embedding_text' in row and row['embedding_text']:
                        emb = [float(x) for x in row['embedding_text'].strip('[]').split(',') if x.strip()]
                    vertices.append(Vertex(
                        realm=row['realm'],
                        id=str(row['id']),
                        space=row.get('space') or 'default',
                        fqid=row['fqid'],
                        payload=row['payload'] if isinstance(row['payload'], dict) else json.loads(row['payload']),
                        created_at=row['created_at'],
                        updated_at=row['updated_at'],
                        table_name=table_name,
                        embedding=emb,
                        uuid=str(row['uuid_text']) if row.get('uuid_text') else None,
                        _client=self
                    ))
                return vertices
            except ProgrammingError:
                raise TableNotFoundError(f"Vertex table '{table_name}' does not exist.")

        if isinstance(self.engine_or_connection, AsyncConnection):
            return await _op(self.engine_or_connection)
        else:
            async with self.engine_or_connection.connect() as conn:
                return await _op(conn)

    async def get_vertex_by_uuid(self, table_name: str, realm: str, uuid: str, strict: bool = False) -> Optional[Vertex]:
        """Fetch a vertex record by its UUID.

        When *strict* is True, raises VertexNotFoundError instead of returning None.
        """
        self._validate_identifier(table_name)
        table_ref = self._get_table_ref(table_name, realm)
        uuid_str = str(uuid).strip()

        async def _op(conn):
            query = f"""
            SELECT t.realm, t.id, t.space, t.fqid, t.payload, t.created_at, t.updated_at,
                   t.uuid::text AS uuid_text,
                   to_jsonb(t)->>'embedding' AS embedding_text
            FROM {table_ref} t
            WHERE t.realm = :realm AND t.uuid = CAST(:uuid AS UUID)
            """
            try:
                row = await self._fetchrow(conn, query, realm=realm, uuid=uuid_str)
                if not row:
                    return None

                emb = None
                if 'embedding_text' in row and row['embedding_text']:
                    emb = [float(x) for x in row['embedding_text'].strip('[]').split(',') if x.strip()]

                return Vertex(
                    realm=row['realm'],
                    id=str(row['id']),
                    space=row.get('space') or 'default',
                    fqid=row['fqid'],
                    payload=row['payload'] if isinstance(row['payload'], dict) else json.loads(row['payload']),
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    table_name=table_name,
                    embedding=emb,
                    uuid=str(row['uuid_text']) if row.get('uuid_text') else None,
                    _client=self
                )
            except (ProgrammingError, DataError, DBAPIError) as e:
                if "does not exist" in str(e).lower():
                    raise TableNotFoundError(f"Vertex table '{table_name}' does not exist.")
                return None

        if isinstance(self.engine_or_connection, AsyncConnection):
            result = await _op(self.engine_or_connection)
        else:
            async with self.engine_or_connection.connect() as conn:
                result = await _op(conn)
        if strict and result is None:
            raise VertexNotFoundError(f"Vertex with uuid '{uuid}' not found in table '{table_name}', realm '{realm}'.")
        return result

    async def vector_search(
        self,
        table_name: str,
        realm: str,
        query_vector: List[float],
        top_k: int = 5,
        distance_metric: str = "cosine",
        search_data_table: bool = False,
        search_scope: str = "main",
        space: Optional[str] = None,
        column_name: str = "embedding",
    ) -> List[Tuple[Vertex, float]]:
        """Perform vector similarity search on vertex embeddings using pgvector."""
        self._validate_identifier(table_name)
        self._validate_identifier(column_name)
        col = f'"{column_name}"'
        table_ref = self._get_table_ref(table_name, realm)
        data_table_ref = self._get_table_ref(f"{table_name}_data", realm)
        vec_str = f"[{','.join(str(x) for x in query_vector)}]"

        op = "<=>"
        if distance_metric == "l2":
            op = "<->"
        elif distance_metric == "inner_product":
            op = "<#>"

        scope = search_scope.lower()
        if search_data_table and scope == "main":
            scope = "data"

        effective_space = space if space and space != RESERVED_SPACE_ALL else None
        space_filter_d = " AND d.space = :space" if effective_space else ""
        space_filter_t = " AND t.space = :space" if effective_space else ""
        space_filter_combined = " AND space = :space" if effective_space else ""

        if scope == "data":
            query = f"""
            SELECT v.realm, v.id, v.space, v.fqid, v.payload, v.created_at, v.updated_at,
                   to_jsonb(v)->>'{column_name}' AS embedding_text,
                   MIN(d.{col} {op} CAST(:vec AS vector)) AS distance
            FROM {data_table_ref} d
            JOIN {table_ref} v ON d.realm = v.realm AND d.id = v.id
            WHERE d.realm = :realm AND d.{col} IS NOT NULL{space_filter_d}
            GROUP BY v.realm, v.id, v.space, v.fqid, v.payload, v.created_at, v.updated_at, to_jsonb(v)
            ORDER BY distance ASC
            LIMIT :top_k
            """
        elif scope == "both":
            query = f"""
            WITH combined AS (
                SELECT realm, id, ({col} {op} CAST(:vec AS vector)) AS distance
                FROM {table_ref}
                WHERE realm = :realm AND {col} IS NOT NULL{space_filter_combined}

                UNION ALL

                SELECT realm, id, ({col} {op} CAST(:vec AS vector)) AS distance
                FROM {data_table_ref}
                WHERE realm = :realm AND {col} IS NOT NULL{space_filter_combined}
            ),
            best AS (
                SELECT realm, id, MIN(distance) AS distance
                FROM combined
                GROUP BY realm, id
            )
            SELECT v.realm, v.id, v.space, v.fqid, v.payload, v.created_at, v.updated_at,
                   to_jsonb(v)->>'{column_name}' AS embedding_text,
                   b.distance
            FROM best b
            JOIN {table_ref} v ON b.realm = v.realm AND b.id = v.id
            ORDER BY b.distance ASC
            LIMIT :top_k
            """
        else:
            query = f"""
            SELECT t.realm, t.id, t.space, t.fqid, t.payload, t.created_at, t.updated_at,
                   CAST(t.{col} AS TEXT) AS embedding_text,
                   (t.{col} {op} CAST(:vec AS vector)) AS distance
            FROM {table_ref} t
            WHERE t.realm = :realm AND t.{col} IS NOT NULL{space_filter_t}
            ORDER BY t.{col} {op} CAST(:vec AS vector) ASC
            LIMIT :top_k
            """

        fetch_params = {"realm": realm, "vec": vec_str, "top_k": top_k}
        if effective_space:
            fetch_params["space"] = effective_space

        async def _op(conn):
            try:
                rows = await self._fetch(conn, query, **fetch_params)
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
                    space=r.get('space') or 'default',
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

    async def vector_search_edges(
        self,
        table_name: str,
        realm: str,
        query_vector: List[float],
        top_k: int = 5,
        distance_metric: str = "cosine",
        space: Optional[str] = None,
        relation_type: Optional[str] = None,
        column_name: str = "embedding",
    ) -> List[Tuple[Edge, float]]:
        """Perform vector similarity search over edge embeddings using pgvector.

        Requires the edge table to have been created with a ``vector_dim``
        or ``vector_columns``.

        Distance metrics match :meth:`vector_search`: 'cosine', 'l2', 'inner_product'.
        """
        self._validate_identifier(table_name)
        self._validate_identifier(column_name)
        col = f'"{column_name}"'
        table_ref = self._get_table_ref(table_name, realm)
        vec_str = f"[{','.join(str(x) for x in query_vector)}]"

        op = "<=>"
        if distance_metric == "l2":
            op = "<->"
        elif distance_metric == "inner_product":
            op = "<#>"

        effective_space = space if space and space != RESERVED_SPACE_ALL else None
        filters = ""
        if effective_space:
            filters += " AND t.space = :space"
        if relation_type:
            filters += " AND t.relation_type = :rel_type"

        query = f"""
        SELECT t.realm, t.id, t.space, t.fqid, t.from_id, t.to_id, t.relation_type,
               t.payload, t.created_at, t.updated_at, t.uuid::text AS uuid_text,
               CAST(t.{col} AS TEXT) AS embedding_text,
               (t.{col} {op} CAST(:vec AS vector)) AS distance
        FROM {table_ref} t
        WHERE t.realm = :realm AND t.{col} IS NOT NULL{filters}
        ORDER BY t.{col} {op} CAST(:vec AS vector) ASC
        LIMIT :top_k
        """

        fetch_params: Dict[str, Any] = {"realm": realm, "vec": vec_str, "top_k": top_k}
        if effective_space:
            fetch_params["space"] = effective_space
        if relation_type:
            fetch_params["rel_type"] = relation_type

        async def _op(conn):
            try:
                rows = await self._fetch(conn, query, **fetch_params)
            except ProgrammingError as e:
                if "does not exist" in str(e).lower():
                    raise TableNotFoundError(f"Edge table '{table_name}' does not exist.")
                logger.warning(f"Vector search on edges failed: {e}")
                return []

            results = []
            for r in rows:
                emb = None
                if r['embedding_text']:
                    emb = [float(x) for x in r['embedding_text'].strip('[]').split(',') if x.strip()]
                e = Edge(
                    realm=r['realm'],
                    id=str(r['id']),
                    from_id=str(r['from_id']),
                    to_id=str(r['to_id']),
                    relation_type=r['relation_type'],
                    space=r.get('space') or 'default',
                    fqid=r['fqid'],
                    payload=r['payload'] if isinstance(r['payload'], dict) else json.loads(r['payload']),
                    embedding=emb,
                    created_at=r['created_at'],
                    updated_at=r['updated_at'],
                    table_name=table_name,
                    uuid=str(r['uuid_text']) if r.get('uuid_text') else None,
                    _client=self
                )
                results.append((e, float(r['distance'])))
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

    async def fulltext_search_vertices(
        self,
        table_name: str,
        realm: str,
        query: str,
        fields: Optional[List[str]] = None,
        config: str = "english",
        limit: int = 20,
        space: Optional[str] = None,
    ) -> List[Vertex]:
        """Full-text search on vertex payload fields using tsvector/tsquery."""
        self._validate_identifier(table_name)
        table_ref = self._get_table_ref(table_name, realm)

        if fields:
            # Field names are interpolated into the tsvector expression, so
            # they get the same identifier check as table names.
            for f in fields:
                self._validate_identifier(f)
            ts_expr = " || ' ' || ".join(f"COALESCE(t.payload->>'{f}', '')" for f in fields)
        else:
            ts_expr = """(SELECT string_agg(value::text, ' ') FROM jsonb_each_text(t.payload))"""

        async def _op(conn):
            params: Dict[str, Any] = {"realm": realm, "query": query, "lim": limit}
            space_clause = ""
            if space and space != RESERVED_SPACE_ALL:
                params["space"] = space
                space_clause = " AND (t.space = :space OR (:space = 'default' AND (t.space IS NULL OR t.space = 'default')))"
            sql = f"""
            SELECT t.realm, t.id, t.space, t.fqid, t.payload, t.created_at, t.updated_at,
                   t.uuid::text AS uuid_text,
                   ts_rank(to_tsvector('{config}', {ts_expr}), plainto_tsquery('{config}', :query)) AS rank
            FROM {table_ref} t
            WHERE t.realm = :realm
              AND to_tsvector('{config}', {ts_expr}) @@ plainto_tsquery('{config}', :query){space_clause}
            ORDER BY rank DESC
            LIMIT :lim
            """
            try:
                result = await conn.execute(text(sql), params)
                rows = result.mappings().all()
                vertices = []
                for row in rows:
                    payload = row['payload'] if isinstance(row['payload'], dict) else json.loads(row['payload'])
                    vertices.append(Vertex(
                        realm=row['realm'],
                        id=str(row['id']),
                        space=row.get('space') or 'default',
                        fqid=row['fqid'],
                        payload=payload,
                        created_at=row['created_at'],
                        updated_at=row['updated_at'],
                        table_name=table_name,
                        uuid=str(row['uuid_text']) if row.get('uuid_text') else None,
                        _client=self
                    ))
                return vertices
            except ProgrammingError as e:
                if "does not exist" in str(e).lower():
                    raise TableNotFoundError(f"Vertex table '{table_name}' does not exist.")
                raise PostGraphError(f"Programming error: {e}")

        if isinstance(self.engine_or_connection, AsyncConnection):
            return await _op(self.engine_or_connection)
        else:
            async with self.engine_or_connection.connect() as conn:
                result = await _op(conn)
                await conn.commit()
                return result

    async def fulltext_search_edges(
        self,
        table_name: str,
        realm: str,
        query: str,
        fields: Optional[List[str]] = None,
        config: str = "english",
        limit: int = 20,
        space: Optional[str] = None,
    ) -> List[Edge]:
        """Full-text search on edge payload fields using tsvector/tsquery."""
        self._validate_identifier(table_name)
        table_ref = self._get_table_ref(table_name, realm)

        if fields:
            # Field names are interpolated into the tsvector expression, so
            # they get the same identifier check as table names.
            for f in fields:
                self._validate_identifier(f)
            ts_expr = " || ' ' || ".join(f"COALESCE(t.payload->>'{f}', '')" for f in fields)
        else:
            ts_expr = """(SELECT string_agg(value::text, ' ') FROM jsonb_each_text(t.payload))"""

        async def _op(conn):
            params: Dict[str, Any] = {"realm": realm, "query": query, "lim": limit}
            space_clause = ""
            if space and space != RESERVED_SPACE_ALL:
                params["space"] = space
                space_clause = " AND (t.space = :space OR (:space = 'default' AND (t.space IS NULL OR t.space = 'default')))"
            sql = f"""
            SELECT t.realm, t.id, t.space, t.fqid, t.from_id, t.to_id,
                   t.relation_type, t.payload, t.created_at, t.updated_at,
                   t.uuid::text AS uuid_text,
                   ts_rank(to_tsvector('{config}', {ts_expr}), plainto_tsquery('{config}', :query)) AS rank
            FROM {table_ref} t
            WHERE t.realm = :realm
              AND to_tsvector('{config}', {ts_expr}) @@ plainto_tsquery('{config}', :query){space_clause}
            ORDER BY rank DESC
            LIMIT :lim
            """
            try:
                result = await conn.execute(text(sql), params)
                rows = result.mappings().all()
                edges = []
                for row in rows:
                    payload = row['payload'] if isinstance(row['payload'], dict) else json.loads(row['payload'])
                    edges.append(Edge(
                        realm=row['realm'],
                        id=str(row['id']),
                        fqid=row['fqid'],
                        from_id=str(row['from_id']),
                        to_id=str(row['to_id']),
                        relation_type=row['relation_type'],
                        space=row.get('space') or 'default',
                        payload=payload,
                        created_at=row['created_at'],
                        updated_at=row['updated_at'],
                        table_name=table_name,
                        uuid=str(row['uuid_text']) if row.get('uuid_text') else None,
                        _client=self
                    ))
                return edges
            except ProgrammingError as e:
                if "does not exist" in str(e).lower():
                    raise TableNotFoundError(f"Edge table '{table_name}' does not exist.")
                raise PostGraphError(f"Programming error: {e}")

        if isinstance(self.engine_or_connection, AsyncConnection):
            return await _op(self.engine_or_connection)
        else:
            async with self.engine_or_connection.connect() as conn:
                result = await _op(conn)
                await conn.commit()
                return result

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
        check_cycle: Union[bool, List[str]] = False,
        space: Optional[str] = "default",
        embedding: Optional[List[float]] = None
    ) -> Edge:
        """Add a new edge. Raises TableExistsError if it already exists.

        ``embedding`` is stored when the edge table was created with a
        ``vector_dim``, enabling semantic search over relationships.
        """
        self._validate_identifier(table_name)
        if space == RESERVED_SPACE_ALL:
            raise ReservedSpaceError(f"'{RESERVED_SPACE_ALL}' is a reserved space name and cannot be used for creation. It is only valid as a query-time filter.")
        payload_json = json.dumps(payload or {})
        table_ref = self._get_table_ref(table_name, realm)
        vec_str = f"[{','.join(str(x) for x in embedding)}]" if embedding else None
        eff_space = space or "default"

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

            if vec_str:
                query = f"""
                INSERT INTO {table_ref} (realm, id, space, from_id, to_id, relation_type, payload, embedding)
                VALUES (:realm, :id, :space, :from_id, :to_id, :relation_type, CAST(:payload AS JSONB), CAST(:vec AS vector))
                RETURNING realm, id, space, fqid, from_id, to_id, relation_type, payload, created_at, updated_at, uuid::text AS uuid_text, CAST(embedding AS TEXT) AS embedding_text
                """
                kwargs = {"realm": realm, "id": e_id_int, "space": eff_space, "from_id": from_id_int,
                          "to_id": to_id_int, "relation_type": relation_type, "payload": payload_json, "vec": vec_str}
            else:
                query = f"""
                INSERT INTO {table_ref} (realm, id, space, from_id, to_id, relation_type, payload)
                VALUES (:realm, :id, :space, :from_id, :to_id, :relation_type, CAST(:payload AS JSONB))
                RETURNING realm, id, space, fqid, from_id, to_id, relation_type, payload, created_at, updated_at, uuid::text AS uuid_text
                """
                kwargs = {"realm": realm, "id": e_id_int, "space": eff_space, "from_id": from_id_int,
                          "to_id": to_id_int, "relation_type": relation_type, "payload": payload_json}

            try:
                row = await self._fetchrow(conn, query, **kwargs)
                if edge_id is not None:
                    await conn.execute(
                        text(f"SELECT setval(pg_get_serial_sequence('{table_ref_pg}', 'id'), (SELECT COALESCE(MAX(id), 1) FROM {table_ref}))")
                    )
                emb = None
                if 'embedding_text' in row and row['embedding_text']:
                    emb = [float(x) for x in row['embedding_text'].strip('[]').split(',') if x.strip()]
                return Edge(
                    realm=row['realm'],
                    id=str(row['id']),
                    fqid=row['fqid'],
                    from_id=str(row['from_id']),
                    to_id=str(row['to_id']),
                    relation_type=row['relation_type'],
                    space=row.get('space') or 'default',
                    payload=row['payload'] if isinstance(row['payload'], dict) else json.loads(row['payload']),
                    embedding=emb,
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    table_name=table_name,
                    uuid=str(row['uuid_text']) if row.get('uuid_text') else None,
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
        check_cycle: Union[bool, List[str]] = False,
        space: Optional[str] = "default",
        embedding: Optional[List[float]] = None
    ) -> Edge:
        """Upsert an edge (merges payload JSONB on conflict).

        ``embedding`` is stored when the edge table has a vector column, and
        replaces any previous embedding for the edge.
        """
        self._validate_identifier(table_name)
        if space == RESERVED_SPACE_ALL:
            raise ReservedSpaceError(f"'{RESERVED_SPACE_ALL}' is a reserved space name and cannot be used for creation. It is only valid as a query-time filter.")
        payload_json = json.dumps(payload or {})
        table_ref = self._get_table_ref(table_name, realm)
        vec_str = f"[{','.join(str(x) for x in embedding)}]" if embedding else None
        eff_space = space or "default"

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
                    conn=conn,
                    direction='out'
                )
                if path:
                    from post_graph.errors import CyclicReferenceError
                    raise CyclicReferenceError(f"Adding edge from '{from_id}' to '{to_id}' in table '{table_name}' would create a cycle.")

            if vec_str:
                query = f"""
                INSERT INTO {table_ref} (realm, id, space, from_id, to_id, relation_type, payload, embedding)
                VALUES (:realm, :id, :space, :from_id, :to_id, :relation_type, CAST(:payload AS JSONB), CAST(:vec AS vector))
                ON CONFLICT (realm, id) DO UPDATE
                SET space = EXCLUDED.space,
                    from_id = EXCLUDED.from_id,
                    to_id = EXCLUDED.to_id,
                    relation_type = EXCLUDED.relation_type,
                    payload = {table_ref}.payload || EXCLUDED.payload,
                    embedding = EXCLUDED.embedding,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING realm, id, space, fqid, from_id, to_id, relation_type, payload, created_at, updated_at, uuid::text AS uuid_text, CAST(embedding AS TEXT) AS embedding_text
                """
                kwargs = {"realm": realm, "id": e_id_int, "space": eff_space, "from_id": from_id_int,
                          "to_id": to_id_int, "relation_type": relation_type, "payload": payload_json, "vec": vec_str}
            else:
                query = f"""
                INSERT INTO {table_ref} (realm, id, space, from_id, to_id, relation_type, payload)
                VALUES (:realm, :id, :space, :from_id, :to_id, :relation_type, CAST(:payload AS JSONB))
                ON CONFLICT (realm, id) DO UPDATE
                SET space = EXCLUDED.space,
                    from_id = EXCLUDED.from_id,
                    to_id = EXCLUDED.to_id,
                    relation_type = EXCLUDED.relation_type,
                    payload = {table_ref}.payload || EXCLUDED.payload,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING realm, id, space, fqid, from_id, to_id, relation_type, payload, created_at, updated_at, uuid::text AS uuid_text
                """
                kwargs = {"realm": realm, "id": e_id_int, "space": eff_space, "from_id": from_id_int,
                          "to_id": to_id_int, "relation_type": relation_type, "payload": payload_json}

            try:
                row = await self._fetchrow(conn, query, **kwargs)
                emb = None
                if 'embedding_text' in row and row['embedding_text']:
                    emb = [float(x) for x in row['embedding_text'].strip('[]').split(',') if x.strip()]
                return Edge(
                    realm=row['realm'],
                    id=str(row['id']),
                    fqid=row['fqid'],
                    from_id=str(row['from_id']),
                    to_id=str(row['to_id']),
                    relation_type=row['relation_type'],
                    space=row.get('space') or 'default',
                    payload=row['payload'] if isinstance(row['payload'], dict) else json.loads(row['payload']),
                    embedding=emb,
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    table_name=table_name,
                    uuid=str(row['uuid_text']) if row.get('uuid_text') else None,
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

    async def get_edge(self, table_name: str, realm: str, edge_id: str, strict: bool = False) -> Optional[Edge]:
        """Fetch an edge by realm and id or uuid.

        When *strict* is True, raises EdgeNotFoundError instead of returning None.
        """
        self._validate_identifier(table_name)
        e_str = str(edge_id).strip()

        if len(e_str) == 36 and '-' in e_str:
            result = await self.get_edge_by_uuid(table_name, realm, e_str)
            if strict and result is None:
                raise EdgeNotFoundError(f"Edge '{edge_id}' not found in table '{table_name}', realm '{realm}'.")
            return result

        table_ref = self._get_table_ref(table_name, realm)
        query = f"""
        SELECT realm, id, space, fqid, from_id, to_id, relation_type, payload, created_at, updated_at, uuid::text AS uuid_text
        FROM {table_ref}
        WHERE realm = :realm AND ((CASE WHEN :id ~ '^[0-9]+$' THEN id = CAST(:id AS BIGINT) ELSE FALSE END) OR fqid = :id)
        """

        async def _op(conn):
            try:
                row = await self._fetchrow(conn, query, realm=realm, id=e_str)
                if not row:
                    return await self.get_edge_by_uuid(table_name, realm, e_str)
                return Edge(
                    realm=row['realm'],
                    id=str(row['id']),
                    fqid=row['fqid'],
                    from_id=str(row['from_id']),
                    to_id=str(row['to_id']),
                    relation_type=row['relation_type'],
                    space=row.get('space') or 'default',
                    payload=row['payload'] if isinstance(row['payload'], dict) else json.loads(row['payload']),
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    table_name=table_name,
                    uuid=str(row['uuid_text']) if row.get('uuid_text') else None,
                    _client=self
                )
            except ProgrammingError as e:
                if "does not exist" in str(e).lower():
                    raise TableNotFoundError(f"Edge table '{table_name}' does not exist.")
                raise PostGraphError(f"Programming error: {e}")

        if isinstance(self.engine_or_connection, AsyncConnection):
            result = await _op(self.engine_or_connection)
        else:
            async with self.engine_or_connection.connect() as conn:
                result = await _op(conn)
        if strict and result is None:
            raise EdgeNotFoundError(f"Edge '{edge_id}' not found in table '{table_name}', realm '{realm}'.")
        return result

    async def get_edge_by_uuid(self, table_name: str, realm: str, uuid: str, strict: bool = False) -> Optional[Edge]:
        """Fetch an edge record by its UUID.

        When *strict* is True, raises EdgeNotFoundError instead of returning None.
        """
        self._validate_identifier(table_name)
        table_ref = self._get_table_ref(table_name, realm)
        uuid_str = str(uuid).strip()
        query = f"""
        SELECT realm, id, space, fqid, from_id, to_id, relation_type, payload, created_at, updated_at, uuid::text AS uuid_text
        FROM {table_ref}
        WHERE realm = :realm AND uuid = CAST(:uuid AS UUID)
        """

        async def _op(conn):
            try:
                row = await self._fetchrow(conn, query, realm=realm, uuid=uuid_str)
                if not row:
                    return None
                return Edge(
                    realm=row['realm'],
                    id=str(row['id']),
                    fqid=row['fqid'],
                    from_id=str(row['from_id']),
                    to_id=str(row['to_id']),
                    relation_type=row['relation_type'],
                    space=row.get('space') or 'default',
                    payload=row['payload'] if isinstance(row['payload'], dict) else json.loads(row['payload']),
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    table_name=table_name,
                    uuid=str(row['uuid_text']) if row.get('uuid_text') else None,
                    _client=self
                )
            except (ProgrammingError, DataError, DBAPIError) as e:
                if "does not exist" in str(e).lower():
                    raise TableNotFoundError(f"Edge table '{table_name}' does not exist.")
                return None

        if isinstance(self.engine_or_connection, AsyncConnection):
            result = await _op(self.engine_or_connection)
        else:
            async with self.engine_or_connection.connect() as conn:
                result = await _op(conn)
        if strict and result is None:
            raise EdgeNotFoundError(f"Edge with uuid '{uuid}' not found in table '{table_name}', realm '{realm}'.")
        return result

    async def get_edges(
        self,
        table_name: str,
        realm: str,
        space: Optional[str] = None,
        limit: Optional[int] = None,
        relation_type: Optional[str] = None,
    ) -> List[Edge]:
        """Fetch all edges in a realm, optionally filtered by space and relation_type."""
        self._validate_identifier(table_name)
        table_ref = self._get_table_ref(table_name, realm)

        async def _op(conn):
            params = {"realm": realm}
            filters = ""
            if space and space != RESERVED_SPACE_ALL:
                params["space"] = space
                filters += " AND (t.space = :space OR (:space = 'default' AND (t.space IS NULL OR t.space = 'default')))"
            if relation_type:
                params["relation_type"] = relation_type
                filters += " AND t.relation_type = :relation_type"

            limit_clause = ""
            if limit:
                params["limit"] = limit
                limit_clause = " LIMIT :limit"

            query = f"""
            SELECT t.realm, t.id, t.space, t.fqid, t.from_id, t.to_id,
                   t.relation_type, t.payload, t.created_at, t.updated_at,
                   t.uuid::text AS uuid_text,
                   to_jsonb(t)->>'embedding' AS embedding_text
            FROM {table_ref} t
            WHERE t.realm = :realm{filters}
            ORDER BY t.id ASC{limit_clause}
            """
            try:
                result = await conn.execute(text(query), params)
                rows = result.mappings().all()
                edges = []
                for row in rows:
                    emb = None
                    if 'embedding_text' in row and row['embedding_text']:
                        emb = [float(x) for x in row['embedding_text'].strip('[]').split(',') if x.strip()]
                    edges.append(Edge(
                        realm=row['realm'],
                        id=str(row['id']),
                        fqid=row['fqid'],
                        from_id=str(row['from_id']),
                        to_id=str(row['to_id']),
                        relation_type=row['relation_type'],
                        space=row.get('space') or 'default',
                        payload=row['payload'] if isinstance(row['payload'], dict) else json.loads(row['payload']),
                        created_at=row['created_at'],
                        updated_at=row['updated_at'],
                        table_name=table_name,
                        embedding=emb,
                        uuid=str(row['uuid_text']) if row.get('uuid_text') else None,
                        _client=self
                    ))
                return edges
            except ProgrammingError:
                raise TableNotFoundError(f"Edge table '{table_name}' does not exist.")

        if isinstance(self.engine_or_connection, AsyncConnection):
            return await _op(self.engine_or_connection)
        else:
            async with self.engine_or_connection.connect() as conn:
                return await _op(conn)

    async def get_edges_multi_realm(
        self,
        table_name: str,
        realms: List[str],
        space: Optional[str] = None,
        limit: Optional[int] = None,
        relation_type: Optional[str] = None,
    ) -> List[Edge]:
        """Fetch edges across multiple realms."""
        self._validate_identifier(table_name)

        async def _op(conn):
            if self.schema_per_realm:
                parts = []
                params: Dict[str, Any] = {}
                for i, r in enumerate(realms):
                    tref = self._get_table_ref(table_name, r)
                    rp = f"r{i}"
                    params[rp] = r
                    extra = ""
                    if space and space != RESERVED_SPACE_ALL:
                        sp = f"sp{i}"
                        params[sp] = space
                        extra += f" AND (t.space = :{sp} OR (:{sp} = 'default' AND (t.space IS NULL OR t.space = 'default')))"
                    if relation_type:
                        rt = f"rt{i}"
                        params[rt] = relation_type
                        extra += f" AND t.relation_type = :{rt}"
                    parts.append(f"SELECT t.realm, t.id, t.space, t.fqid, t.from_id, t.to_id, t.relation_type, t.payload, t.created_at, t.updated_at, t.uuid::text AS uuid_text, to_jsonb(t)->>'embedding' AS embedding_text FROM {tref} t WHERE t.realm = :{rp}{extra}")
                query = " UNION ALL ".join(parts) + " ORDER BY realm, id ASC"
                if limit:
                    params["limit"] = limit
                    query += " LIMIT :limit"
            else:
                table_ref = self._get_table_ref(table_name, realms[0])
                params = {"realms": realms}
                extra = ""
                if space and space != RESERVED_SPACE_ALL:
                    params["space"] = space
                    extra += " AND (t.space = :space OR (:space = 'default' AND (t.space IS NULL OR t.space = 'default')))"
                if relation_type:
                    params["relation_type"] = relation_type
                    extra += " AND t.relation_type = :relation_type"
                limit_clause = ""
                if limit:
                    params["limit"] = limit
                    limit_clause = " LIMIT :limit"
                query = f"""
                SELECT t.realm, t.id, t.space, t.fqid, t.from_id, t.to_id,
                       t.relation_type, t.payload, t.created_at, t.updated_at,
                       t.uuid::text AS uuid_text,
                       to_jsonb(t)->>'embedding' AS embedding_text
                FROM {table_ref} t
                WHERE t.realm = ANY(:realms){extra}
                ORDER BY t.realm, t.id ASC{limit_clause}
                """
            try:
                result = await conn.execute(text(query), params)
                rows = result.mappings().all()
                edges = []
                for row in rows:
                    emb = None
                    if 'embedding_text' in row and row['embedding_text']:
                        emb = [float(x) for x in row['embedding_text'].strip('[]').split(',') if x.strip()]
                    edges.append(Edge(
                        realm=row['realm'],
                        id=str(row['id']),
                        fqid=row['fqid'],
                        from_id=str(row['from_id']),
                        to_id=str(row['to_id']),
                        relation_type=row['relation_type'],
                        space=row.get('space') or 'default',
                        payload=row['payload'] if isinstance(row['payload'], dict) else json.loads(row['payload']),
                        created_at=row['created_at'],
                        updated_at=row['updated_at'],
                        table_name=table_name,
                        embedding=emb,
                        uuid=str(row['uuid_text']) if row.get('uuid_text') else None,
                        _client=self
                    ))
                return edges
            except ProgrammingError:
                raise TableNotFoundError(f"Edge table '{table_name}' does not exist.")

        if isinstance(self.engine_or_connection, AsyncConnection):
            return await _op(self.engine_or_connection)
        else:
            async with self.engine_or_connection.connect() as conn:
                return await _op(conn)

    async def find_edges(
        self,
        table_name: str,
        realm: str,
        filters: Dict[str, Any],
        space: Optional[str] = None,
        relation_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Edge]:
        """Find edges whose payload matches the given key-value filters.

        Matching is by JSONB containment; see :meth:`find_vertices` for the
        semantics (type-sensitive, index-backed, containment for nested
        values, sentinels for null and absent keys). Optional *relation_type*
        further restricts results.
        """
        self._validate_identifier(table_name)
        table_ref = self._get_table_ref(table_name, realm)

        async def _op(conn):
            params: Dict[str, Any] = {"realm": realm}
            clauses = ""
            if space and space != RESERVED_SPACE_ALL:
                params["space"] = space
                clauses += " AND (t.space = :space OR (:space = 'default' AND (t.space IS NULL OR t.space = 'default')))"
            if relation_type:
                params["relation_type"] = relation_type
                clauses += " AND t.relation_type = :relation_type"
            containment, absent_keys = self._prepare_filters(filters)
            if containment is not None:
                params["fjson"] = containment
                clauses += " AND t.payload @> CAST(:fjson AS jsonb)"
            for i, key in enumerate(absent_keys):
                pname = f"fabs{i}"
                params[pname] = key
                clauses += f" AND NOT jsonb_exists(t.payload, :{pname})"
            limit_clause = ""
            if limit:
                params["limit"] = limit
                limit_clause = " LIMIT :limit"
            query = f"""
            SELECT t.realm, t.id, t.space, t.fqid, t.from_id, t.to_id,
                   t.relation_type, t.payload, t.created_at, t.updated_at,
                   t.uuid::text AS uuid_text,
                   to_jsonb(t)->>'embedding' AS embedding_text
            FROM {table_ref} t
            WHERE t.realm = :realm{clauses}
            ORDER BY t.id ASC{limit_clause}
            """
            try:
                result = await conn.execute(text(query), params)
                rows = result.mappings().all()
                edges = []
                for row in rows:
                    emb = None
                    if 'embedding_text' in row and row['embedding_text']:
                        emb = [float(x) for x in row['embedding_text'].strip('[]').split(',') if x.strip()]
                    edges.append(Edge(
                        realm=row['realm'],
                        id=str(row['id']),
                        fqid=row['fqid'],
                        from_id=str(row['from_id']),
                        to_id=str(row['to_id']),
                        relation_type=row['relation_type'],
                        space=row.get('space') or 'default',
                        payload=row['payload'] if isinstance(row['payload'], dict) else json.loads(row['payload']),
                        created_at=row['created_at'],
                        updated_at=row['updated_at'],
                        table_name=table_name,
                        embedding=emb,
                        uuid=str(row['uuid_text']) if row.get('uuid_text') else None,
                        _client=self
                    ))
                return edges
            except ProgrammingError:
                raise TableNotFoundError(f"Edge table '{table_name}' does not exist.")

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
                            v.id AS v_id, v.space AS v_space, v.fqid AS v_fqid, v.payload AS v_payload, v.created_at AS v_created_at, v.updated_at AS v_updated_at,
                            e.id AS e_id, e.space AS e_space, e.fqid AS e_fqid, e.from_id AS e_from, e.to_id AS e_to, e.relation_type AS e_rel, e.payload AS e_payload, e.created_at AS e_created_at, e.updated_at AS e_updated_at
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
                            v.id AS v_id, v.space AS v_space, v.fqid AS v_fqid, v.payload AS v_payload, v.created_at AS v_created_at, v.updated_at AS v_updated_at,
                            e.id AS e_id, e.space AS e_space, e.fqid AS e_fqid, e.from_id AS e_from, e.to_id AS e_to, e.relation_type AS e_rel, e.payload AS e_payload, e.created_at AS e_created_at, e.updated_at AS e_updated_at
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
                            space=r.get('v_space') or 'default',
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
                            space=r.get('e_space') or 'default',
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

    @staticmethod
    def _padded_date_sql(expr: str) -> str:
        """Pad a partial ISO date so string comparison orders it correctly.

        '2020' must compare below '2020-06-01', so a bare year becomes
        '2020-01-01'. Kept identical to how callers pad the ``as_of`` argument,
        because a mismatch would silently answer temporal queries differently
        one hop out than it does at the source vertex.
        """
        return (
            f"(split_part({expr}, '-', 1) || '-' || "
            f"COALESCE(NULLIF(split_part({expr}, '-', 2), ''), '01') || '-' || "
            f"COALESCE(NULLIF(split_part({expr}, '-', 3), ''), '01'))"
        )

    async def _promoted_columns(self, conn, table_name: str, realm: Optional[str] = None) -> set:
        """Promoted column names present on a table, cached.

        A table created before promotion existed, or with different keys, simply
        lacks the column — callers fall back to the payload expression, which
        returns the same rows more slowly.
        """
        cache_key = (realm, table_name)
        if cache_key in self._promoted_cache:
            return self._promoted_cache[cache_key]
        schema = realm if self.schema_per_realm else 'public'
        rows = await self._fetch(
            conn,
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = :s AND table_name = :t "
            "AND (column_name LIKE 'pt\\_%' OR column_name LIKE 'p\\_%')",
            s=schema, t=table_name,
        )
        cols = {r['column_name'] for r in rows}
        self._promoted_cache[cache_key] = cols
        return cols

    def _edge_filter_sql(
        self,
        relation_types: Optional[List[str]],
        as_of: Optional[str],
        payload_null_keys: Optional[List[str]],
        space: Optional[str],
        valid_from_key: str,
        valid_to_key: str,
        promoted_columns: Optional[set] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Build the WHERE fragment applied to every edge step of a traversal.

        Returned as a fragment plus its bind parameters so both traversal
        directions share one definition; drift between them would make an 'in'
        walk observe a different graph from an 'out' walk.
        """
        clauses: List[str] = []
        params: Dict[str, Any] = {}

        if relation_types:
            clauses.append("relation_type = ANY(CAST(:rel_types AS text[]))")
            params['rel_types'] = list(relation_types)

        if space and space != RESERVED_SPACE_ALL:
            clauses.append("space = :trav_space")
            params['trav_space'] = space

        # A relation with no stated period holds at every date: silence about
        # when a fact applied means it applied throughout, not that it never did.
        if as_of:
            at = self._padded_date_sql("CAST(:trav_as_of AS text)")
            vf_col = _promoted.temporal_column(valid_from_key)
            vt_col = _promoted.temporal_column(valid_to_key)
            if promoted_columns and vf_col in promoted_columns and vt_col in promoted_columns:
                # Indexed generated columns holding the same normalised date the
                # expression below computes: identical rows, index instead of scan.
                clauses.append(
                    f'(("{vf_col}" IS NULL OR "{vf_col}" <= {at}) AND '
                    f'("{vt_col}" IS NULL OR "{vt_col}" >= {at}))'
                )
            else:
                vf = f"payload->>'{valid_from_key}'"
                vt = f"payload->>'{valid_to_key}'"
                clauses.append(
                    f"(({vf} IS NULL OR {self._padded_date_sql(vf)} <= {at}) AND "
                    f"({vt} IS NULL OR {self._padded_date_sql(vt)} >= {at}))"
                )
            params['trav_as_of'] = as_of

        for key in payload_null_keys or []:
            self._validate_identifier(key)
            col = _promoted.generic_column(key)
            if promoted_columns and col in promoted_columns:
                clauses.append(f'"{col}" IS NULL')
            else:
                clauses.append(f"payload->>'{key}' IS NULL")

        return ("".join(f" AND {c}" for c in clauses), params)

    async def traverse(
        self,
        realm: str,
        start_table: str,
        start_id: str,
        edge_tables: List[str],
        max_depth: int = 3,
        direction: str = 'out',
        relation_types: Optional[List[str]] = None,
        as_of: Optional[str] = None,
        payload_null_keys: Optional[List[str]] = None,
        space: Optional[str] = None,
        valid_from_key: str = 'valid_from',
        valid_to_key: str = 'valid_to',
    ) -> List[Dict[str, Any]]:
        """
        Perform a dynamic graph traversal starting from (start_table, start_id)
        up to max_depth. scoped to a specific realm.

        Optional filters, all applied to every step rather than to the final
        result, so a walk is never routed *through* an edge the caller excluded:

        ``relation_types``   follow only these relation types.
        ``as_of``            follow only edges whose stated validity covers this
                             date; edges stating no period always qualify.
        ``payload_null_keys`` follow only edges where these payload keys are
                             absent or null — how a caller excludes edges it has
                             marked, such as superseded ones.
        ``space``            confine the walk to one space. Without it a
                             traversal that started from a correctly scoped
                             vertex will still wander into other tenants' data.
        """
        self._validate_identifier(start_table)
        if direction not in ('out', 'in', 'both'):
            raise ValueError("Direction must be 'out', 'in', or 'both'")

        start_id_str = str(start_id).split('/')[-1] if '/' in str(start_id) else str(start_id)

        async def _op(conn):
            # One filter fragment is shared by every edge table in the walk, so a
            # promoted column may only be used when *all* of them have it.
            promoted_columns = None
            for _et in edge_tables:
                _cols = await self._promoted_columns(conn, _et, realm)
                promoted_columns = _cols if promoted_columns is None else (promoted_columns & _cols)

            filter_sql, filter_params = self._edge_filter_sql(
                relation_types, as_of, payload_null_keys, space,
                valid_from_key, valid_to_key,
                promoted_columns=promoted_columns,
            )

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
                    WHERE realm = :realm AND from_id = (CASE WHEN t.current_id ~ '^[0-9]+$' THEN CAST(t.current_id AS BIGINT) ELSE NULL END) AND t.current_table = '{from_ref}'{filter_sql}
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
                    WHERE realm = :realm AND to_id = (CASE WHEN t.current_id ~ '^[0-9]+$' THEN CAST(t.current_id AS BIGINT) ELSE NULL END) AND t.current_table = '{to_ref}'{filter_sql}
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
                realm=realm, start_id=start_id_str, start_table=start_table, max_depth=max_depth,
                **filter_params
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

    async def connected_components(
        self,
        realm: str,
        vertex_table: str,
        edge_tables: List[str],
        direction: str = "both",
    ) -> List[List[str]]:
        """Return connected components as lists of vertex IDs."""
        self._validate_identifier(vertex_table)
        if direction not in ("out", "in", "both"):
            raise ValueError("direction must be 'out', 'in', or 'both'")

        vtable_ref = self._get_table_ref(vertex_table, realm)

        async def _op(conn):
            subqueries: list = []
            for et in edge_tables:
                self._validate_identifier(et)
                schema = await self.get_edge_schema(conn, et, realm=realm)
                eref = self._get_table_ref(et, realm)
                if schema["from_id"] == vertex_table:
                    if direction in ("out", "both"):
                        subqueries.append(
                            f"SELECT CAST(from_id AS text) AS src, CAST(to_id AS text) AS dst FROM {eref} WHERE realm = :realm"
                        )
                    if direction in ("in", "both"):
                        subqueries.append(
                            f"SELECT CAST(to_id AS text) AS src, CAST(from_id AS text) AS dst FROM {eref} WHERE realm = :realm"
                        )
                elif schema["to_id"] == vertex_table:
                    if direction in ("out", "both"):
                        subqueries.append(
                            f"SELECT CAST(to_id AS text) AS src, CAST(from_id AS text) AS dst FROM {eref} WHERE realm = :realm"
                        )
                    if direction in ("in", "both"):
                        subqueries.append(
                            f"SELECT CAST(from_id AS text) AS src, CAST(to_id AS text) AS dst FROM {eref} WHERE realm = :realm"
                        )

            if not subqueries:
                result = await conn.execute(
                    text(f"SELECT CAST(id AS text) FROM {vtable_ref} WHERE realm = :realm"),
                    {"realm": realm},
                )
                return [[str(r[0]) for r in result.fetchall()]]

            edges_union = "\nUNION ALL\n".join(subqueries)
            query = f"""
            WITH all_edges AS ({edges_union}),
            RECURSIVE flood AS (
                SELECT CAST(id AS text) AS vid, CAST(id AS text) AS component_root
                FROM {vtable_ref} WHERE realm = :realm

                UNION

                SELECT f.vid,
                       LEAST(f.component_root, e.dst) AS component_root
                FROM flood f
                JOIN all_edges e ON f.vid = e.src
            )
            SELECT component_root, array_agg(DISTINCT vid) AS members
            FROM flood
            GROUP BY component_root
            """
            result = await conn.execute(text(query), {"realm": realm})
            return [list(r[1]) for r in result.fetchall()]

        if isinstance(self.engine_or_connection, AsyncConnection):
            return await _op(self.engine_or_connection)
        else:
            async with self.engine_or_connection.connect() as conn:
                return await _op(conn)

    async def weighted_shortest_path(
        self,
        realm: str,
        start_table: str,
        start_id: str,
        target_table: str,
        target_id: str,
        edge_tables: List[str],
        weight_field: str = "weight",
        max_depth: int = 10,
        direction: str = "out",
    ) -> Optional[Dict[str, Any]]:
        """Dijkstra-style weighted shortest path using a payload field as weight."""
        self._validate_identifier(start_table)
        # weight_field is interpolated into the traversal subqueries.
        self._validate_identifier(weight_field)
        self._validate_identifier(target_table)
        if direction not in ("out", "in", "both"):
            raise ValueError("direction must be 'out', 'in', or 'both'")

        start_id_str = str(start_id).split("/")[-1] if "/" in str(start_id) else str(start_id)
        target_id_str = str(target_id).split("/")[-1] if "/" in str(target_id) else str(target_id)

        async def _op(conn):
            subqueries: list = []
            for et in edge_tables:
                schema = await self.get_edge_schema(conn, et, realm=realm)
                from_ref = schema["from_id"]
                to_ref = schema["to_id"]
                eref = self._get_table_ref(et, realm)

                if direction in ("out", "both"):
                    subqueries.append(f"""
                    SELECT CAST(to_id AS text) AS next_id,
                           CAST('{to_ref}' AS text) AS next_table,
                           CAST(id AS text) AS edge_id,
                           relation_type,
                           CAST('{et}' AS text) AS edge_table,
                           COALESCE(CAST(payload->>'{weight_field}' AS double precision), 1.0) AS edge_weight
                    FROM {eref}
                    WHERE realm = :realm
                      AND from_id = (CASE WHEN t.current_id ~ '^[0-9]+$$' THEN CAST(t.current_id AS bigint) ELSE NULL END)
                      AND t.current_table = '{from_ref}'
                    """)

                if direction in ("in", "both"):
                    subqueries.append(f"""
                    SELECT CAST(from_id AS text) AS next_id,
                           CAST('{from_ref}' AS text) AS next_table,
                           CAST(id AS text) AS edge_id,
                           relation_type,
                           CAST('{et}' AS text) AS edge_table,
                           COALESCE(CAST(payload->>'{weight_field}' AS double precision), 1.0) AS edge_weight
                    FROM {eref}
                    WHERE realm = :realm
                      AND to_id = (CASE WHEN t.current_id ~ '^[0-9]+$$' THEN CAST(t.current_id AS bigint) ELSE NULL END)
                      AND t.current_table = '{to_ref}'
                    """)

            if not subqueries:
                return None

            union_all = "\nUNION ALL\n".join(subqueries)

            cte_query = f"""
            WITH RECURSIVE graph_traversal AS (
                SELECT
                    CAST(:start_id AS text) AS current_id,
                    CAST(:start_table AS text) AS current_table,
                    0 AS depth,
                    ARRAY[CAST(:start_table AS text) || ':' || CAST(:start_id AS text)]::text[] AS path,
                    ARRAY[]::text[] AS edge_path,
                    ARRAY[]::text[] AS edge_ids,
                    CAST(0.0 AS double precision) AS total_weight

                UNION ALL

                SELECT
                    step.next_id,
                    step.next_table,
                    t.depth + 1,
                    t.path || (step.next_table || ':' || step.next_id),
                    t.edge_path || (step.edge_table || ':' || step.relation_type),
                    t.edge_ids || step.edge_id,
                    t.total_weight + step.edge_weight
                FROM graph_traversal t
                CROSS JOIN LATERAL (
                    {union_all}
                ) step
                WHERE t.depth < :max_depth
                  AND NOT ((step.next_table || ':' || step.next_id) = ANY(t.path))
            )
            SELECT depth, path, edge_path, edge_ids, total_weight
            FROM graph_traversal
            WHERE current_id = :target_id AND current_table = :target_table
            ORDER BY total_weight ASC
            LIMIT 1;
            """

            params = {
                "realm": realm,
                "start_id": start_id_str,
                "start_table": start_table,
                "max_depth": max_depth,
                "target_id": target_id_str,
                "target_table": target_table,
            }

            result = await conn.execute(text(cte_query), params)
            row = result.mappings().first()
            if not row:
                return None
            return {
                "depth": row["depth"],
                "path": list(row["path"]),
                "edge_path": list(row["edge_path"]),
                "edge_ids": list(row["edge_ids"]),
                "total_weight": float(row["total_weight"]),
            }

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
        embedding: Optional[List[float]] = None,
        user_id: Optional[str] = None
    ) -> DataRecord:
        """Append a historical record to {table_name}_data table for a vertex."""
        self._validate_identifier(table_name)
        v_id_int = int(str(vertex_id).split('/')[-1]) if '/' in str(vertex_id) else int(vertex_id)
        payload_json = json.dumps(payload or {})
        data_table_ref = self._get_table_ref(f"{table_name}_data", realm)
        vec_str = f"[{','.join(str(x) for x in embedding)}]" if embedding is not None else None

        async def _op(conn):
            has_emb = False
            if vec_str:
                data_t_name = f"{table_name}_data"
                if self.schema_per_realm:
                    check_q = "SELECT 1 FROM information_schema.columns WHERE table_schema = :schema AND table_name = :t_name AND column_name = 'embedding'"
                    res = await conn.execute(text(check_q), {"schema": realm, "t_name": data_t_name})
                    has_emb = res.scalar() is not None
                else:
                    check_q = "SELECT 1 FROM information_schema.columns WHERE table_name = :t_name AND column_name = 'embedding'"
                    res = await conn.execute(text(check_q), {"t_name": data_t_name})
                    has_emb = res.scalar() is not None

            if vec_str and has_emb:
                if timestamp:
                    query = f"""
                    INSERT INTO {data_table_ref} (realm, id, payload, timestamp, embedding)
                    VALUES (:realm, :id, CAST(:payload AS JSONB), :timestamp, CAST(:vec AS vector))
                    RETURNING data_id, realm, id, payload, timestamp, CAST(embedding AS TEXT) AS embedding_text
                    """
                    row = await self._fetchrow(conn, query, realm=realm, id=v_id_int, payload=payload_json, timestamp=timestamp, vec=vec_str)
                else:
                    query = f"""
                    INSERT INTO {data_table_ref} (realm, id, payload, embedding)
                    VALUES (:realm, :id, CAST(:payload AS JSONB), CAST(:vec AS vector))
                    RETURNING data_id, realm, id, payload, timestamp, CAST(embedding AS TEXT) AS embedding_text
                    """
                    row = await self._fetchrow(conn, query, realm=realm, id=v_id_int, payload=payload_json, vec=vec_str)
            else:
                if timestamp:
                    query = f"""
                    INSERT INTO {data_table_ref} (realm, id, payload, timestamp)
                    VALUES (:realm, :id, CAST(:payload AS JSONB), :timestamp)
                    RETURNING data_id, realm, id, payload, timestamp, NULL AS embedding_text
                    """
                    row = await self._fetchrow(conn, query, realm=realm, id=v_id_int, payload=payload_json, timestamp=timestamp)
                else:
                    query = f"""
                    INSERT INTO {data_table_ref} (realm, id, payload)
                    VALUES (:realm, :id, CAST(:payload AS JSONB))
                    RETURNING data_id, realm, id, payload, timestamp, NULL AS embedding_text
                    """
                    row = await self._fetchrow(conn, query, realm=realm, id=v_id_int, payload=payload_json)

            emb = None
            if 'embedding_text' in row and row['embedding_text']:
                emb = [float(x) for x in row['embedding_text'].strip('[]').split(',') if x.strip()]

            return DataRecord(
                data_id=str(row['data_id']),
                realm=row['realm'],
                id=str(row['id']),
                payload=row['payload'] if isinstance(row['payload'], dict) else json.loads(row['payload']),
                timestamp=row['timestamp'],
                embedding=emb
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
        SELECT d.data_id, d.realm, d.id, d.payload, d.timestamp, to_jsonb(d)->>'embedding' AS embedding_text
        FROM {data_table_ref} d
        WHERE d.realm = :realm AND d.id = :id
        ORDER BY d.timestamp DESC, d.data_id DESC
        {limit_clause}
        """

        async def _op(conn):
            rows = await self._fetch(conn, query, realm=realm, id=v_id_int)
            results = []
            for r in rows:
                emb = None
                if 'embedding_text' in r and r['embedding_text']:
                    emb = [float(x) for x in r['embedding_text'].strip('[]').split(',') if x.strip()]
                results.append(
                    DataRecord(
                        data_id=str(r['data_id']),
                        realm=r['realm'],
                        id=str(r['id']),
                        payload=r['payload'] if isinstance(r['payload'], dict) else json.loads(r['payload']),
                        timestamp=r['timestamp'],
                        embedding=emb
                    )
                )
            return results

        if isinstance(self.engine_or_connection, AsyncConnection):
            return await _op(self.engine_or_connection)
        else:
            async with self.engine_or_connection.connect() as conn:
                return await _op(conn)

    async def get_latest_vertex_data(
        self,
        table_name: str,
        realm: str,
        vertex_id: Union[str, int]
    ) -> Optional[DataRecord]:
        """Fetch the latest append-only data record (version) for a vertex."""
        records = await self.get_vertex_data(table_name, realm, vertex_id, limit=1)
        return records[0] if records else None

    async def get_vertex_data_by_id(
        self,
        table_name: str,
        realm: str,
        data_id: Union[str, int]
    ) -> Optional[DataRecord]:
        """Query a specific data entry / version by its sequential data_id."""
        self._validate_identifier(table_name)
        d_id_int = int(data_id)
        data_table_ref = self._get_table_ref(f"{table_name}_data", realm)

        query = f"""
        SELECT d.data_id, d.realm, d.id, d.payload, d.timestamp, to_jsonb(d)->>'embedding' AS embedding_text
        FROM {data_table_ref} d
        WHERE d.realm = :realm AND d.data_id = :data_id
        """

        async def _op(conn):
            rows = await self._fetch(conn, query, realm=realm, data_id=d_id_int)
            if not rows:
                return None
            r = rows[0]
            emb = None
            if 'embedding_text' in r and r['embedding_text']:
                emb = [float(x) for x in r['embedding_text'].strip('[]').split(',') if x.strip()]
            return DataRecord(
                data_id=str(r['data_id']),
                realm=r['realm'],
                id=str(r['id']),
                payload=r['payload'] if isinstance(r['payload'], dict) else json.loads(r['payload']),
                timestamp=r['timestamp'],
                embedding=emb
            )

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
        embedding: Optional[List[float]] = None,
        user_id: Optional[str] = None
    ) -> DataRecord:
        """Append a historical record to {table_name}_data table for an edge."""
        return await self.add_vertex_data(table_name, realm, edge_id, payload, timestamp=timestamp, embedding=embedding, user_id=user_id)

    async def get_edge_data(
        self,
        table_name: str,
        realm: str,
        edge_id: Union[str, int],
        limit: Optional[int] = None
    ) -> List[DataRecord]:
        """Fetch append-only data records for an edge sorted by timestamp descending."""
        return await self.get_vertex_data(table_name, realm, edge_id, limit=limit)

    async def get_latest_edge_data(
        self,
        table_name: str,
        realm: str,
        edge_id: Union[str, int]
    ) -> Optional[DataRecord]:
        """Fetch the latest append-only data record for an edge."""
        return await self.get_latest_vertex_data(table_name, realm, edge_id)

    async def get_edge_data_by_id(
        self,
        table_name: str,
        realm: str,
        data_id: Union[str, int]
    ) -> Optional[DataRecord]:
        """Query a specific data entry by its sequential data_id for an edge."""
        return await self.get_vertex_data_by_id(table_name, realm, data_id)

