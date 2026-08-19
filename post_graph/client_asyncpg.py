import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import asyncpg

from post_graph.errors import (
    VertexNotFoundError,
    EdgeNotFoundError,
    TableExistsError,
    TableNotFoundError,
    PostGraphError,
    ReservedSpaceError,
)
from post_graph.models import Vertex, Edge, DataRecord

logger = logging.getLogger("post_graph")


RESERVED_SPACE_ALL = "__all__"


class AsyncPostGraph:
    def __init__(
        self,
        connection_or_pool: Union[asyncpg.Connection, asyncpg.Pool, None] = None,
        dsn: Optional[str] = None,
        schema_per_realm: bool = False,
        pool_min_size: int = 10,
        pool_max_size: int = 10,
        pool_max_queries: int = 50000,
        pool_max_inactive_connection_lifetime: float = 300.0,
        **conn_kwargs
    ):
        self.connection = connection_or_pool
        self.dsn = dsn
        self.schema_per_realm = schema_per_realm
        self.conn_kwargs = conn_kwargs
        self._pool_config = {
            "min_size": pool_min_size,
            "max_size": pool_max_size,
            "max_queries": pool_max_queries,
            "max_inactive_connection_lifetime": pool_max_inactive_connection_lifetime,
        }
        self._pool = None
        self._schema_cache = {}

    async def connect(self):
        """Establish connection or connection pool to PostgreSQL."""
        if self.connection is None:
            kwargs = {**self._pool_config, **self.conn_kwargs}
            if self.dsn:
                self._pool = await asyncpg.create_pool(self.dsn, **kwargs)
            else:
                self._pool = await asyncpg.create_pool(**kwargs)
            self.connection = self._pool

    async def close(self):
        """Close connection pool if it was managed by this client."""
        if self._pool:
            await self._pool.close()

    def get_pool_config(self) -> Dict[str, Any]:
        """Return the current pool configuration."""
        return dict(self._pool_config)

    def get_pool_status(self) -> Optional[Dict[str, Any]]:
        """Return live pool status (sizes, free count) or None if not pooled."""
        if not self._pool:
            return None
        return {
            "min_size": self._pool.get_min_size(),
            "max_size": self._pool.get_max_size(),
            "size": self._pool.get_size(),
            "free_size": self._pool.get_idle_size(),
        }

    def _validate_identifier(self, identifier: str):
        """Ensure identifiers are safe and valid to prevent SQL injection."""
        if not identifier or not isinstance(identifier, str):
            raise ValueError("Identifier must be a non-empty string.")
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', identifier):
            raise ValueError(
                f"Invalid identifier: '{identifier}'. Must be alphanumeric and underscores only, "
                f"starting with a letter or underscore."
            )

    def _get_table_ref(self, table_name: str, realm: Optional[str] = None) -> str:
        """Get the table reference. Qualifies with schema (realm) if schema_per_realm is active."""
        self._validate_identifier(table_name)
        if self.schema_per_realm:
            if not realm:
                raise PostGraphError(f"Realm must be specified for table reference '{table_name}' under schema_per_realm mode.")
            self._validate_identifier(realm)
            return f'"{realm}"."{table_name}"'
        return f'"{table_name}"'

    async def execute(self, query: str, *args) -> str:
        """Execute a SQL statement and return the status string."""
        if isinstance(self.connection, asyncpg.Pool):
            async with self.connection.acquire() as conn:
                return await conn.execute(query, *args)
        else:
            return await self.connection.execute(query, *args)

    async def fetch(self, query: str, *args) -> List[asyncpg.Record]:
        """Execute a SQL query and return all rows."""
        if isinstance(self.connection, asyncpg.Pool):
            async with self.connection.acquire() as conn:
                return await conn.fetch(query, *args)
        else:
            return await self.connection.fetch(query, *args)

    async def fetchrow(self, query: str, *args) -> Optional[asyncpg.Record]:
        """Execute a SQL query and return the first row."""
        if isinstance(self.connection, asyncpg.Pool):
            async with self.connection.acquire() as conn:
                return await conn.fetchrow(query, *args)
        else:
            return await self.connection.fetchrow(query, *args)

    # Backward-compatible aliases
    _execute = execute
    _fetch = fetch
    _fetchrow = fetchrow

    async def _run_in_tx(self, func, user_id: Optional[str] = None):
        """Helper to run a block of operations inside a transaction, setting the user_id session context."""
        async def _execute_block(conn):
            if user_id:
                # Set transaction-local session variable for auditing
                await conn.execute("SELECT set_config('app.current_user_id', $1, true)", str(user_id))
            else:
                await conn.execute("SELECT set_config('app.current_user_id', '', true)")
            return await func(conn)

        if isinstance(self.connection, asyncpg.Pool):
            async with self.connection.acquire() as conn:
                async with conn.transaction():
                    return await _execute_block(conn)
        else:
            async with self.connection.transaction():
                return await _execute_block(self.connection)

    async def _table_exists(self, table_name: str, realm: Optional[str] = None) -> bool:
        """Check if a table exists in the database within the proper schema namespace."""
        self._validate_identifier(table_name)
        if self.schema_per_realm:
            if not realm:
                raise PostGraphError(f"Realm must be specified for _table_exists('{table_name}') in schema_per_realm mode.")
            self._validate_identifier(realm)
            query = """
            SELECT 1 
            FROM pg_class c
            JOIN pg_namespace n ON c.relnamespace = n.oid
            WHERE c.relname = $1 AND c.relkind = 'r' AND n.nspname = $2
            """
            row = await self._fetchrow(query, table_name, realm)
        else:
            query = """
            SELECT 1 
            FROM pg_class c
            WHERE c.relname = $1 AND c.relkind = 'r' AND pg_table_is_visible(c.oid)
            """
            row = await self._fetchrow(query, table_name)
        return row is not None

    async def _table_has_embedding(self, conn, table_name: str, realm: Optional[str] = None) -> bool:
        """Return True if the given table has an 'embedding' column."""
        if self.schema_per_realm:
            found = await conn.fetchval(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_schema = $1 AND table_name = $2 AND column_name = 'embedding'",
                realm, table_name
            )
        else:
            found = await conn.fetchval(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_name = $1 AND column_name = 'embedding'",
                table_name
            )
        return found is not None

    async def _add_vector_column(
        self,
        table_name: str,
        table_ref: str,
        data_table_ref: str,
        vector_dim: int
    ):
        """Add pgvector embedding columns and HNSW indexes to a table and its data table.

        Callers must invoke this only after both tables exist.
        """
        try:
            await self._execute("CREATE EXTENSION IF NOT EXISTS vector;")
            await self._execute(f"ALTER TABLE {table_ref} ADD COLUMN IF NOT EXISTS embedding vector({vector_dim});")
            await self._execute(f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_embedding" ON {table_ref} USING hnsw (embedding vector_cosine_ops);')
            await self._execute(f"ALTER TABLE {data_table_ref} ADD COLUMN IF NOT EXISTS embedding vector({vector_dim});")
            await self._execute(f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_data_embedding" ON {data_table_ref} USING hnsw (embedding vector_cosine_ops);')
        except Exception as e:
            raise PostGraphError(f"Failed to initialize pgvector extension or embedding column for table '{table_name}': {e}")

    async def _add_vector_columns(
        self,
        table_name: str,
        table_ref: str,
        data_table_ref: str,
        columns: Dict[str, int],
    ):
        """Add multiple named pgvector columns with HNSW indexes."""
        try:
            await self._execute("CREATE EXTENSION IF NOT EXISTS vector;")
            for col_name, dim in columns.items():
                self._validate_identifier(col_name)
                await self._execute(f'ALTER TABLE {table_ref} ADD COLUMN IF NOT EXISTS "{col_name}" vector({dim});')
                await self._execute(f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_{col_name}" ON {table_ref} USING hnsw ("{col_name}" vector_cosine_ops);')
                await self._execute(f'ALTER TABLE {data_table_ref} ADD COLUMN IF NOT EXISTS "{col_name}" vector({dim});')
                await self._execute(f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_data_{col_name}" ON {data_table_ref} USING hnsw ("{col_name}" vector_cosine_ops);')
        except Exception as e:
            raise PostGraphError(f"Failed to add vector columns to table '{table_name}': {e}")

    async def create_vertex_table(
        self,
        table_name: str,
        realm: Optional[str] = None,
        vector_dim: Optional[int] = None,
        vector_columns: Optional[Dict[str, int]] = None,
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
            await self._execute(f'CREATE SCHEMA IF NOT EXISTS "{realm}"')
        else:
            schema_prefix = ""

        # Create trigger function for updated_at
        await self._execute(f"""
            CREATE OR REPLACE FUNCTION {schema_prefix}update_modified_column()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = now();
                RETURN NEW;
            END;
            $$ language 'plpgsql';
        """)

        # Create shared trigger function for auditing
        await self._execute(f"""
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
        """)

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
        await self._execute(query)
        await self._execute(f"ALTER TABLE {table_ref} ADD COLUMN IF NOT EXISTS space VARCHAR(255) DEFAULT 'default';")
        await self._execute(f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_space" ON {table_ref} (realm, space);')
        await self._execute(f"ALTER TABLE {table_ref} ADD COLUMN IF NOT EXISTS fqid TEXT GENERATED ALWAYS AS (realm || '/' || '{table_name}' || '/' || id::text) STORED;")
        await self._execute(f"ALTER TABLE {table_ref} ADD COLUMN IF NOT EXISTS uuid UUID DEFAULT gen_random_uuid();")
        await self._execute(f'CREATE UNIQUE INDEX IF NOT EXISTS "idx_{table_name}_uuid" ON {table_ref} (uuid);')

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
        await self._execute(audit_query)

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
        await self._execute(data_query)
        await self._execute(f"ALTER TABLE {data_table_ref} ADD COLUMN IF NOT EXISTS space VARCHAR(255) DEFAULT 'default';")
        await self._execute(f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_data_space" ON {data_table_ref} (realm, space);')
        await self._execute(f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_data_id" ON {data_table_ref} (realm, id);')
        await self._execute(f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_data_payload" ON {data_table_ref} USING gin (payload);')

        # 4. Add vector columns. This must run after the data table exists, since
        # the embedding column is added to both the main and the data table.
        if vector_dim and vector_dim > 0:
            await self._add_vector_column(table_name, table_ref, data_table_ref, vector_dim)
        if vector_columns:
            await self._add_vector_columns(table_name, table_ref, data_table_ref, vector_columns)

        # 5. Create GIN index on payload
        await self._execute(
            f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_payload" ON {table_ref} USING gin (payload);'
        )

        # 5. Create trigger for updated_at
        await self._execute(f'DROP TRIGGER IF EXISTS "update_{table_name}_modtime" ON {table_ref};')
        await self._execute(f"""
            CREATE TRIGGER "update_{table_name}_modtime"
            BEFORE UPDATE ON {table_ref}
            FOR EACH ROW
            EXECUTE FUNCTION {schema_prefix}update_modified_column();
        """)

        # 6. Create trigger for auditing
        await self._execute(f'DROP TRIGGER IF EXISTS "audit_{table_name}_trigger" ON {table_ref};')
        await self._execute(f"""
            CREATE TRIGGER "audit_{table_name}_trigger"
            AFTER INSERT OR UPDATE OR DELETE ON {table_ref}
            FOR EACH ROW
            EXECUTE FUNCTION {schema_prefix}audit_trigger_func();
        """)

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
    ):
        """Create a new edge table linking two vertex tables, plus shadow audit table and constraints.

        When ``vector_dim`` is given, the edge table gains a pgvector ``embedding``
        column with an HNSW index, so relationships can be retrieved by semantic
        similarity the same way vertices are.
        """
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

        # Check that from_vertex_table and to_vertex_table exist
        if not await self._table_exists(from_vertex_table, realm=realm):
            raise TableNotFoundError(f"Referenced from_vertex_table '{from_vertex_table}' does not exist.")
        if not await self._table_exists(to_vertex_table, realm=realm):
            raise TableNotFoundError(f"Referenced to_vertex_table '{to_vertex_table}' does not exist.")

        # Resolve schema prefix for triggers/functions
        if self.schema_per_realm:
            schema_prefix = f'"{realm}".'
        else:
            schema_prefix = ""

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
        await self._execute(query)
        await self._execute(f"ALTER TABLE {table_ref} ADD COLUMN IF NOT EXISTS space VARCHAR(255) DEFAULT 'default';")
        await self._execute(f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_space" ON {table_ref} (realm, space);')
        await self._execute(f"ALTER TABLE {table_ref} ADD COLUMN IF NOT EXISTS fqid TEXT GENERATED ALWAYS AS (realm || '/' || '{from_vertex_table}-{to_vertex_table}' || '/' || id::text) STORED;")
        await self._execute(f"ALTER TABLE {table_ref} ADD COLUMN IF NOT EXISTS uuid UUID DEFAULT gen_random_uuid();")
        await self._execute(f'CREATE UNIQUE INDEX IF NOT EXISTS "idx_{table_name}_uuid" ON {table_ref} (uuid);')

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
        await self._execute(audit_query)

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
        await self._execute(data_query)
        await self._execute(f"ALTER TABLE {data_table_ref} ADD COLUMN IF NOT EXISTS space VARCHAR(255) DEFAULT 'default';")
        await self._execute(f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_data_space" ON {data_table_ref} (realm, space);')
        await self._execute(f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_data_id" ON {data_table_ref} (realm, id);')
        await self._execute(f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_data_payload" ON {data_table_ref} USING gin (payload);')

        # 3a. Add vector columns, after both the main and data tables exist.
        if vector_dim and vector_dim > 0:
            await self._add_vector_column(table_name, table_ref, data_table_ref, vector_dim)
        if vector_columns:
            await self._add_vector_columns(table_name, table_ref, data_table_ref, vector_columns)

        # 3. Create indexes
        await self._execute(
            f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_to" ON {table_ref} (realm, to_id);'
        )
        await self._execute(
            f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_payload" ON {table_ref} USING gin (payload);'
        )

        # 4. Create trigger for updated_at
        await self._execute(f'DROP TRIGGER IF EXISTS "update_{table_name}_modtime" ON {table_ref};')
        await self._execute(f"""
            CREATE TRIGGER "update_{table_name}_modtime"
            BEFORE UPDATE ON {table_ref}
            FOR EACH ROW
            EXECUTE FUNCTION {schema_prefix}update_modified_column();
        """)

        # 5. Create trigger for auditing
        await self._execute(f'DROP TRIGGER IF EXISTS "audit_{table_name}_trigger" ON {table_ref};')
        await self._execute(f"""
            CREATE TRIGGER "audit_{table_name}_trigger"
            AFTER INSERT OR UPDATE OR DELETE ON {table_ref}
            FOR EACH ROW
            EXECUTE FUNCTION {schema_prefix}audit_trigger_func();
        """)

        # 6. Create custom cascade delete trigger if either boolean is True
        await self._execute(f'DROP TRIGGER IF EXISTS "cascade_delete_trigger_{table_name}" ON {table_ref};')
        await self._execute(f'DROP FUNCTION IF EXISTS {schema_prefix}"cascade_delete_func_{table_name}"();')
        if cascade_delete_from or cascade_delete_to:
            from_clause = f'DELETE FROM {from_vertex_ref} WHERE realm = OLD.realm AND id = OLD.from_id;' if cascade_delete_from else ''
            to_clause = f'DELETE FROM {to_vertex_ref} WHERE realm = OLD.realm AND id = OLD.to_id;' if cascade_delete_to else ''
            
            await self._execute(f"""
                CREATE OR REPLACE FUNCTION {schema_prefix}"cascade_delete_func_{table_name}"()
                RETURNS TRIGGER AS $$
                BEGIN
                    {from_clause}
                    {to_clause}
                    RETURN OLD;
                END;
                $$ LANGUAGE plpgsql;
            """)
            await self._execute(f"""
                CREATE TRIGGER "cascade_delete_trigger_{table_name}"
                AFTER DELETE ON {table_ref}
                FOR EACH ROW
                EXECUTE FUNCTION {schema_prefix}"cascade_delete_func_{table_name}"();
            """)

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
                v_id_int = await conn.fetchval(seq_query)
            else:
                v_id_int = int(str(vertex_id).split('/')[-1]) if '/' in str(vertex_id) else int(vertex_id)

            v_id_str = str(v_id_int)

            has_emb = False
            if vec_str:
                if self.schema_per_realm:
                    has_emb = await conn.fetchval("SELECT 1 FROM information_schema.columns WHERE table_schema = $1 AND table_name = $2 AND column_name = 'embedding'", realm, table_name) is not None
                else:
                    has_emb = await conn.fetchval("SELECT 1 FROM information_schema.columns WHERE table_name = $1 AND column_name = 'embedding'", table_name) is not None

            if vec_str and has_emb:
                query = f"""
                INSERT INTO {table_ref} (realm, id, space, payload, embedding)
                VALUES ($1, $2, $3, $4::jsonb, $5::vector)
                RETURNING realm, id, space, fqid, payload, created_at, updated_at, uuid::text AS uuid_text, embedding::text AS embedding_text
                """
                row = await conn.fetchrow(query, realm, v_id_int, eff_space, payload_json, vec_str)
            else:
                query = f"""
                INSERT INTO {table_ref} (realm, id, space, payload)
                VALUES ($1, $2, $3, $4::jsonb)
                RETURNING realm, id, space, fqid, payload, created_at, updated_at, uuid::text AS uuid_text
                """
                row = await conn.fetchrow(query, realm, v_id_int, eff_space, payload_json)

            try:
                if vertex_id is not None:
                    await conn.execute(
                        f"SELECT setval(pg_get_serial_sequence('{table_ref_pg}', 'id'), (SELECT COALESCE(MAX(id), 1) FROM {table_ref}))"
                    )

                emb_dict = None
                if embeddings:
                    set_clauses = []
                    up_params: list = [realm, row['id']]
                    for col_name, vec in embeddings.items():
                        self._validate_identifier(col_name)
                        up_params.append(f"[{','.join(str(x) for x in vec)}]")
                        set_clauses.append(f'"{col_name}" = ${len(up_params)}::vector')
                    if set_clauses:
                        await conn.execute(
                            f"UPDATE {table_ref} SET {', '.join(set_clauses)} WHERE realm = $1 AND id = $2",
                            *up_params,
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
            except asyncpg.UniqueViolationError:
                raise TableExistsError(
                    f"Vertex with ID '{v_id_str}' already exists in table '{table_name}' under realm '{realm}'."
                )
            except asyncpg.UndefinedTableError:
                raise TableNotFoundError(f"Vertex table '{table_name}' does not exist.")

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
                v_id_int = await conn.fetchval(seq_query)
            else:
                v_id_int = int(str(vertex_id).split('/')[-1]) if '/' in str(vertex_id) else int(vertex_id)

            v_id_str = str(v_id_int)

            has_emb = False
            if vec_str:
                if self.schema_per_realm:
                    has_emb = await conn.fetchval("SELECT 1 FROM information_schema.columns WHERE table_schema = $1 AND table_name = $2 AND column_name = 'embedding'", realm, table_name) is not None
                else:
                    has_emb = await conn.fetchval("SELECT 1 FROM information_schema.columns WHERE table_name = $1 AND column_name = 'embedding'", table_name) is not None

            if vec_str and has_emb:
                query = f"""
                INSERT INTO {table_ref} (realm, id, space, payload, embedding)
                VALUES ($1, $2, $3, $4::jsonb, $5::vector)
                ON CONFLICT (realm, id) DO UPDATE
                SET space = EXCLUDED.space,
                    payload = {table_ref}.payload || EXCLUDED.payload,
                    embedding = EXCLUDED.embedding,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING realm, id, space, fqid, payload, created_at, updated_at, uuid::text AS uuid_text, embedding::text AS embedding_text
                """
                row = await conn.fetchrow(query, realm, v_id_int, eff_space, payload_json, vec_str)
            else:
                query = f"""
                INSERT INTO {table_ref} (realm, id, space, payload)
                VALUES ($1, $2, $3, $4::jsonb)
                ON CONFLICT (realm, id) DO UPDATE
                SET space = EXCLUDED.space,
                    payload = {table_ref}.payload || EXCLUDED.payload,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING realm, id, space, fqid, payload, created_at, updated_at, uuid::text AS uuid_text
                """
                row = await conn.fetchrow(query, realm, v_id_int, eff_space, payload_json)
            try:
                emb_dict = None
                if embeddings:
                    set_clauses = []
                    up_params: list = [realm, row['id']]
                    for col_name, vec in embeddings.items():
                        self._validate_identifier(col_name)
                        up_params.append(f"[{','.join(str(x) for x in vec)}]")
                        set_clauses.append(f'"{col_name}" = ${len(up_params)}::vector')
                    if set_clauses:
                        await conn.execute(
                            f"UPDATE {table_ref} SET {', '.join(set_clauses)} WHERE realm = $1 AND id = $2",
                            *up_params,
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
            except asyncpg.UndefinedTableError:
                raise TableNotFoundError(f"Vertex table '{table_name}' does not exist.")

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
        ``relation_type``.  Optional: ``edge_id``, ``payload``, ``embedding``,
        ``space``.
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
                embedding=item.get("embedding"),
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
            WHERE t.realm = $1 AND t.id = $2
            """
            try:
                row = await conn.fetchrow(query, realm, v_id_int)
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
            except asyncpg.UndefinedTableError:
                raise TableNotFoundError(f"Vertex table '{table_name}' does not exist.")

        if isinstance(self.connection, asyncpg.Pool):
            async with self.connection.acquire() as conn:
                result = await _op(conn)
        else:
            result = await _op(self.connection)
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
            params = [realm]
            space_clause = ""
            if space and space != RESERVED_SPACE_ALL:
                params.append(space)
                space_clause = f" AND (t.space = ${len(params)} OR (${len(params)} = 'default' AND (t.space IS NULL OR t.space = 'default')))"

            limit_clause = ""
            if limit:
                params.append(limit)
                limit_clause = f" LIMIT ${len(params)}"

            query = f"""
            SELECT t.realm, t.id, t.space, t.fqid, t.payload, t.created_at, t.updated_at,
                   t.uuid::text AS uuid_text,
                   to_jsonb(t)->>'embedding' AS embedding_text
            FROM {table_ref} t
            WHERE t.realm = $1{space_clause}
            ORDER BY t.id ASC{limit_clause}
            """
            try:
                rows = await conn.fetch(query, *params)
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
            except asyncpg.UndefinedTableError:
                raise TableNotFoundError(f"Vertex table '{table_name}' does not exist.")

        if isinstance(self.connection, asyncpg.Pool):
            async with self.connection.acquire() as conn:
                return await _op(conn)
        else:
            return await _op(self.connection)

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
                all_params: list = []
                for r in realms:
                    tref = self._get_table_ref(table_name, r)
                    base = len(all_params)
                    all_params.append(r)
                    space_clause = ""
                    if space and space != RESERVED_SPACE_ALL:
                        all_params.append(space)
                        space_clause = f" AND (t.space = ${base+2} OR (${base+2} = 'default' AND (t.space IS NULL OR t.space = 'default')))"
                    parts.append(f"SELECT t.realm, t.id, t.space, t.fqid, t.payload, t.created_at, t.updated_at, t.uuid::text AS uuid_text, to_jsonb(t)->>'embedding' AS embedding_text FROM {tref} t WHERE t.realm = ${base+1}{space_clause}")
                query = " UNION ALL ".join(parts) + " ORDER BY realm, id ASC"
                if limit:
                    all_params.append(limit)
                    query += f" LIMIT ${len(all_params)}"
                rows = await conn.fetch(query, *all_params)
            else:
                table_ref = self._get_table_ref(table_name, realms[0])
                params: list = [realms]
                space_clause = ""
                if space and space != RESERVED_SPACE_ALL:
                    params.append(space)
                    space_clause = f" AND (t.space = ${len(params)} OR (${len(params)} = 'default' AND (t.space IS NULL OR t.space = 'default')))"
                limit_clause = ""
                if limit:
                    params.append(limit)
                    limit_clause = f" LIMIT ${len(params)}"
                query = f"""
                SELECT t.realm, t.id, t.space, t.fqid, t.payload, t.created_at, t.updated_at,
                       t.uuid::text AS uuid_text,
                       to_jsonb(t)->>'embedding' AS embedding_text
                FROM {table_ref} t
                WHERE t.realm = ANY($1::text[]){space_clause}
                ORDER BY t.realm, t.id ASC{limit_clause}
                """
                rows = await conn.fetch(query, *params)
            try:
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
            except asyncpg.UndefinedTableError:
                raise TableNotFoundError(f"Vertex table '{table_name}' does not exist.")

        if isinstance(self.connection, asyncpg.Pool):
            async with self.connection.acquire() as conn:
                return await _op(conn)
        else:
            return await _op(self.connection)

    async def find_vertices(
        self,
        table_name: str,
        realm: str,
        filters: Dict[str, Any],
        space: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Vertex]:
        """Find vertices whose payload matches the given key-value filters.

        Each entry in *filters* becomes a ``payload->>'key' = value`` clause
        (all ANDed together).
        """
        self._validate_identifier(table_name)
        table_ref = self._get_table_ref(table_name, realm)

        async def _op(conn):
            params: list = [realm]
            clauses = ""
            if space and space != RESERVED_SPACE_ALL:
                params.append(space)
                clauses += f" AND (t.space = ${len(params)} OR (${len(params)} = 'default' AND (t.space IS NULL OR t.space = 'default')))"
            for key, val in filters.items():
                params.append(str(val))
                clauses += f" AND t.payload->>'{key}' = ${len(params)}"
            limit_clause = ""
            if limit:
                params.append(limit)
                limit_clause = f" LIMIT ${len(params)}"
            query = f"""
            SELECT t.realm, t.id, t.space, t.fqid, t.payload, t.created_at, t.updated_at,
                   t.uuid::text AS uuid_text,
                   to_jsonb(t)->>'embedding' AS embedding_text
            FROM {table_ref} t
            WHERE t.realm = $1{clauses}
            ORDER BY t.id ASC{limit_clause}
            """
            try:
                rows = await conn.fetch(query, *params)
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
            except asyncpg.UndefinedTableError:
                raise TableNotFoundError(f"Vertex table '{table_name}' does not exist.")

        if isinstance(self.connection, asyncpg.Pool):
            async with self.connection.acquire() as conn:
                return await _op(conn)
        else:
            return await _op(self.connection)

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
            WHERE t.realm = $1 AND t.uuid = $2::uuid
            """
            try:
                row = await conn.fetchrow(query, realm, uuid_str)
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
            except asyncpg.UndefinedTableError:
                raise TableNotFoundError(f"Vertex table '{table_name}' does not exist.")
            except (asyncpg.DataError, asyncpg.InvalidTextRepresentationError):
                return None

        if isinstance(self.connection, asyncpg.Pool):
            async with self.connection.acquire() as conn:
                result = await _op(conn)
        else:
            result = await _op(self.connection)
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
        """Perform vector similarity search on vertex embeddings using pgvector.
        
        Parameters:
          - search_data_table: If True and search_scope is 'main', searches the associated data table.
          - search_scope: 'main' (search main vertex table), 'data' (search associated data table), or 'both' (search both tables).
        
        Distance metrics:
          - 'cosine': <=> (cosine distance)
          - 'l2': <-> (Euclidean distance)
          - 'inner_product': <#> (negative inner product)
        """
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
        space_filter = ""
        if effective_space:
            space_filter = f" AND v.space = $4" if scope in ("data", "both") else f" AND t.space = $4"
        space_filter_d = f" AND d.space = $4" if effective_space else ""
        space_filter_combined = f" AND space = $4" if effective_space else ""

        if scope == "data":
            query = f"""
            SELECT v.realm, v.id, v.space, v.fqid, v.payload, v.created_at, v.updated_at,
                   to_jsonb(v)->>'{column_name}' AS embedding_text,
                   MIN(d.{col} {op} $2::vector) AS distance
            FROM {data_table_ref} d
            JOIN {table_ref} v ON d.realm = v.realm AND d.id = v.id
            WHERE d.realm = $1 AND d.{col} IS NOT NULL{space_filter_d}
            GROUP BY v.realm, v.id, v.space, v.fqid, v.payload, v.created_at, v.updated_at, to_jsonb(v)
            ORDER BY distance ASC
            LIMIT $3
            """
        elif scope == "both":
            query = f"""
            WITH combined AS (
                SELECT realm, id, ({col} {op} $2::vector) AS distance
                FROM {table_ref}
                WHERE realm = $1 AND {col} IS NOT NULL{space_filter_combined}

                UNION ALL

                SELECT realm, id, ({col} {op} $2::vector) AS distance
                FROM {data_table_ref}
                WHERE realm = $1 AND {col} IS NOT NULL{space_filter_combined}
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
            LIMIT $3
            """
        else:
            query = f"""
            SELECT t.realm, t.id, t.space, t.fqid, t.payload, t.created_at, t.updated_at,
                   to_jsonb(t)->>'{column_name}' AS embedding_text,
                   (t.{col} {op} $2::vector) AS distance
            FROM {table_ref} t
            WHERE t.realm = $1 AND t.{col} IS NOT NULL{space_filter}
            ORDER BY t.{col} {op} $2::vector ASC
            LIMIT $3
            """

        fetch_args = [realm, vec_str, top_k]
        if effective_space:
            fetch_args.append(effective_space)

        async def _op(conn):
            try:
                rows = await conn.fetch(query, *fetch_args)
            except asyncpg.UndefinedTableError:
                raise TableNotFoundError(f"Vertex table '{table_name}' does not exist.")
            except asyncpg.UndefinedColumnError:
                logger.warning(f"Table '{table_name}' or data table does not have a vector embedding column.")
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

        if isinstance(self.connection, asyncpg.Pool):
            async with self.connection.acquire() as conn:
                return await _op(conn)
        else:
            return await _op(self.connection)

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
        fetch_args: List[Any] = [realm, vec_str, top_k]
        filters = ""
        if effective_space:
            fetch_args.append(effective_space)
            filters += f" AND t.space = ${len(fetch_args)}"
        if relation_type:
            fetch_args.append(relation_type)
            filters += f" AND t.relation_type = ${len(fetch_args)}"

        query = f"""
        SELECT t.realm, t.id, t.space, t.fqid, t.from_id, t.to_id, t.relation_type,
               t.payload, t.created_at, t.updated_at, t.uuid::text AS uuid_text,
               to_jsonb(t)->>'{column_name}' AS embedding_text,
               (t.{col} {op} $2::vector) AS distance
        FROM {table_ref} t
        WHERE t.realm = $1 AND t.{col} IS NOT NULL{filters}
        ORDER BY t.{col} {op} $2::vector ASC
        LIMIT $3
        """

        async def _op(conn):
            try:
                rows = await conn.fetch(query, *fetch_args)
            except asyncpg.UndefinedTableError:
                raise TableNotFoundError(f"Edge table '{table_name}' does not exist.")
            except asyncpg.UndefinedColumnError:
                logger.warning(
                    f"Edge table '{table_name}' has no vector embedding column. "
                    f"Recreate it with create_edge_table(..., vector_dim=N)."
                )
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

        if isinstance(self.connection, asyncpg.Pool):
            async with self.connection.acquire() as conn:
                return await _op(conn)
        else:
            return await _op(self.connection)

    async def delete_vertex(self, table_name: str, realm: str, vertex_id: str, user_id: Optional[str] = None) -> bool:
        """Delete a vertex. Cascading foreign keys will automatically delete referencing edges."""
        self._validate_identifier(table_name)
        table_ref = self._get_table_ref(table_name, realm)
        v_str = str(vertex_id)

        async def _op(conn):
            query = f"""
            DELETE FROM {table_ref} 
            WHERE realm = $1 AND ((CASE WHEN $2 ~ '^[0-9]+$' THEN id = $2::bigint ELSE FALSE END) OR fqid = $2)
            """
            try:
                res = await conn.execute(query, realm, v_str)
                return res == "DELETE 1"
            except asyncpg.UndefinedTableError:
                raise TableNotFoundError(f"Vertex table '{table_name}' does not exist.")

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
        """Full-text search on vertex payload fields using tsvector/tsquery.

        *fields* lists payload keys to search (default: all text values).
        *config* is the PostgreSQL text search configuration.
        """
        self._validate_identifier(table_name)
        table_ref = self._get_table_ref(table_name, realm)

        if fields:
            ts_expr = " || ' ' || ".join(f"COALESCE(t.payload->>'{f}', '')" for f in fields)
        else:
            ts_expr = """(SELECT string_agg(value::text, ' ') FROM jsonb_each_text(t.payload))"""

        async def _op(conn):
            params: list = [realm, query, limit]
            space_clause = ""
            if space and space != RESERVED_SPACE_ALL:
                params.append(space)
                space_clause = f" AND (t.space = ${len(params)} OR (${len(params)} = 'default' AND (t.space IS NULL OR t.space = 'default')))"
            sql = f"""
            SELECT t.realm, t.id, t.space, t.fqid, t.payload, t.created_at, t.updated_at,
                   t.uuid::text AS uuid_text,
                   to_jsonb(t)->>'embedding' AS embedding_text,
                   ts_rank(to_tsvector('{config}', {ts_expr}), plainto_tsquery('{config}', $2)) AS rank
            FROM {table_ref} t
            WHERE t.realm = $1
              AND to_tsvector('{config}', {ts_expr}) @@ plainto_tsquery('{config}', $2){space_clause}
            ORDER BY rank DESC
            LIMIT $3
            """
            try:
                rows = await conn.fetch(sql, *params)
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
            except asyncpg.UndefinedTableError:
                raise TableNotFoundError(f"Vertex table '{table_name}' does not exist.")

        if isinstance(self.connection, asyncpg.Pool):
            async with self.connection.acquire() as conn:
                return await _op(conn)
        else:
            return await _op(self.connection)

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
        """Full-text search on edge payload fields using tsvector/tsquery.

        *fields* lists payload keys to search (default: all text values).
        *config* is the PostgreSQL text search configuration.
        """
        self._validate_identifier(table_name)
        table_ref = self._get_table_ref(table_name, realm)

        if fields:
            ts_expr = " || ' ' || ".join(f"COALESCE(t.payload->>'{f}', '')" for f in fields)
        else:
            ts_expr = """(SELECT string_agg(value::text, ' ') FROM jsonb_each_text(t.payload))"""

        async def _op(conn):
            params: list = [realm, query, limit]
            space_clause = ""
            if space and space != RESERVED_SPACE_ALL:
                params.append(space)
                space_clause = f" AND (t.space = ${len(params)} OR (${len(params)} = 'default' AND (t.space IS NULL OR t.space = 'default')))"
            sql = f"""
            SELECT t.realm, t.id, t.space, t.fqid, t.from_id, t.to_id,
                   t.relation_type, t.payload, t.created_at, t.updated_at,
                   t.uuid::text AS uuid_text,
                   to_jsonb(t)->>'embedding' AS embedding_text,
                   ts_rank(to_tsvector('{config}', {ts_expr}), plainto_tsquery('{config}', $2)) AS rank
            FROM {table_ref} t
            WHERE t.realm = $1
              AND to_tsvector('{config}', {ts_expr}) @@ plainto_tsquery('{config}', $2){space_clause}
            ORDER BY rank DESC
            LIMIT $3
            """
            try:
                rows = await conn.fetch(sql, *params)
                edges = []
                for row in rows:
                    emb = None
                    if row.get('embedding_text'):
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
            except asyncpg.UndefinedTableError:
                raise TableNotFoundError(f"Edge table '{table_name}' does not exist.")

        if isinstance(self.connection, asyncpg.Pool):
            async with self.connection.acquire() as conn:
                return await _op(conn)
        else:
            return await _op(self.connection)

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
            schema = await self.get_edge_schema(table_name, realm=realm, conn=conn)
            from_table = schema["from_id"]
            to_table = schema["to_id"]

            table_ref_pg = f'"{realm}"."{table_name}"' if self.schema_per_realm else f'"{table_name}"'
            if edge_id is None:
                seq_query = f"SELECT nextval(pg_get_serial_sequence('{table_ref_pg}', 'id'))"
                e_id_int = await conn.fetchval(seq_query)
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

            has_emb = False
            if vec_str:
                has_emb = await self._table_has_embedding(conn, table_name, realm)

            if vec_str and has_emb:
                query = f"""
                INSERT INTO {table_ref} (realm, id, space, from_id, to_id, relation_type, payload, embedding)
                VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8::vector)
                RETURNING realm, id, space, fqid, from_id, to_id, relation_type, payload, created_at, updated_at, uuid::text AS uuid_text, embedding::text AS embedding_text
                """
                args = (realm, e_id_int, eff_space, from_id_int, to_id_int, relation_type, payload_json, vec_str)
            else:
                if vec_str and not has_emb:
                    logger.warning(
                        f"Edge table '{table_name}' has no embedding column; the supplied embedding was not stored. "
                        f"Recreate the table with create_edge_table(..., vector_dim=N)."
                    )
                query = f"""
                INSERT INTO {table_ref} (realm, id, space, from_id, to_id, relation_type, payload)
                VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
                RETURNING realm, id, space, fqid, from_id, to_id, relation_type, payload, created_at, updated_at, uuid::text AS uuid_text
                """
                args = (realm, e_id_int, eff_space, from_id_int, to_id_int, relation_type, payload_json)

            try:
                row = await conn.fetchrow(query, *args)
                if edge_id is not None:
                    await conn.execute(
                        f"SELECT setval(pg_get_serial_sequence('{table_ref_pg}', 'id'), (SELECT COALESCE(MAX(id), 1) FROM {table_ref}))"
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
            except asyncpg.UniqueViolationError:
                raise TableExistsError(
                    f"Edge with ID '{e_id_str}' already exists in table '{table_name}' under realm '{realm}'."
                )
            except asyncpg.ForeignKeyViolationError as e:
                raise VertexNotFoundError(f"Foreign key violation: referenced vertices do not exist. Details: {e}")
            except asyncpg.UndefinedTableError:
                raise TableNotFoundError(f"Edge table '{table_name}' does not exist.")

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
            schema = await self.get_edge_schema(table_name, realm=realm, conn=conn)
            from_table = schema["from_id"]
            to_table = schema["to_id"]

            table_ref_pg = f'"{realm}"."{table_name}"' if self.schema_per_realm else f'"{table_name}"'
            if edge_id is None:
                seq_query = f"SELECT nextval(pg_get_serial_sequence('{table_ref_pg}', 'id'))"
                e_id_int = await conn.fetchval(seq_query)
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
                    raise CyclicReferenceError(
                        f"Upserting edge '{e_id_str}' from '{from_id}' to '{to_id}' would create a cyclic reference. "
                        f"Existing path: {' -> '.join(path['path'])}"
                    )

            has_emb = False
            if vec_str:
                has_emb = await self._table_has_embedding(conn, table_name, realm)

            if vec_str and has_emb:
                query = f"""
                INSERT INTO {table_ref} (realm, id, space, from_id, to_id, relation_type, payload, embedding)
                VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8::vector)
                ON CONFLICT (realm, id) DO UPDATE
                SET space = EXCLUDED.space,
                    from_id = EXCLUDED.from_id,
                    to_id = EXCLUDED.to_id,
                    relation_type = EXCLUDED.relation_type,
                    payload = {table_ref}.payload || EXCLUDED.payload,
                    embedding = EXCLUDED.embedding,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING realm, id, space, fqid, from_id, to_id, relation_type, payload, created_at, updated_at, uuid::text AS uuid_text, embedding::text AS embedding_text
                """
                args = (realm, e_id_int, eff_space, from_id_int, to_id_int, relation_type, payload_json, vec_str)
            else:
                if vec_str and not has_emb:
                    logger.warning(
                        f"Edge table '{table_name}' has no embedding column; the supplied embedding was not stored. "
                        f"Recreate the table with create_edge_table(..., vector_dim=N)."
                    )
                query = f"""
                INSERT INTO {table_ref} (realm, id, space, from_id, to_id, relation_type, payload)
                VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
                ON CONFLICT (realm, id) DO UPDATE
                SET space = EXCLUDED.space,
                    from_id = EXCLUDED.from_id,
                    to_id = EXCLUDED.to_id,
                    relation_type = EXCLUDED.relation_type,
                    payload = {table_ref}.payload || EXCLUDED.payload,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING realm, id, space, fqid, from_id, to_id, relation_type, payload, created_at, updated_at, uuid::text AS uuid_text
                """
                args = (realm, e_id_int, eff_space, from_id_int, to_id_int, relation_type, payload_json)

            try:
                row = await conn.fetchrow(query, *args)
                emb = None
                if 'embedding_text' in row and row['embedding_text']:
                    emb = [float(x) for x in row['embedding_text'].strip('[]').split(',') if x.strip()]
                return Edge(
                    embedding=emb,
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
            except asyncpg.ForeignKeyViolationError as e:
                raise VertexNotFoundError(f"Foreign key violation: referenced vertices do not exist. Details: {e}")
            except asyncpg.UndefinedTableError:
                raise TableNotFoundError(f"Edge table '{table_name}' does not exist.")

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
        WHERE realm = $1 AND ((CASE WHEN $2 ~ '^[0-9]+$' THEN id = $2::bigint ELSE FALSE END) OR fqid = $2)
        """
        try:
            row = await self._fetchrow(query, realm, e_str)
            if not row:
                result = await self.get_edge_by_uuid(table_name, realm, e_str)
                if strict and result is None:
                    raise EdgeNotFoundError(f"Edge '{edge_id}' not found in table '{table_name}', realm '{realm}'.")
                return result
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
        except asyncpg.UndefinedTableError:
            raise TableNotFoundError(f"Edge table '{table_name}' does not exist.")

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
        WHERE realm = $1 AND uuid = $2::uuid
        """
        try:
            row = await self._fetchrow(query, realm, uuid_str)
            if not row:
                if strict:
                    raise EdgeNotFoundError(f"Edge with uuid '{uuid}' not found in table '{table_name}', realm '{realm}'.")
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
        except asyncpg.UndefinedTableError:
            raise TableNotFoundError(f"Edge table '{table_name}' does not exist.")
        except (asyncpg.DataError, asyncpg.InvalidTextRepresentationError):
            if strict:
                raise EdgeNotFoundError(f"Edge with uuid '{uuid}' not found in table '{table_name}', realm '{realm}'.")
            return None

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
            params: list = [realm]
            filters = ""
            if space and space != RESERVED_SPACE_ALL:
                params.append(space)
                filters += f" AND (t.space = ${len(params)} OR (${len(params)} = 'default' AND (t.space IS NULL OR t.space = 'default')))"
            if relation_type:
                params.append(relation_type)
                filters += f" AND t.relation_type = ${len(params)}"

            limit_clause = ""
            if limit:
                params.append(limit)
                limit_clause = f" LIMIT ${len(params)}"

            query = f"""
            SELECT t.realm, t.id, t.space, t.fqid, t.from_id, t.to_id,
                   t.relation_type, t.payload, t.created_at, t.updated_at,
                   t.uuid::text AS uuid_text,
                   to_jsonb(t)->>'embedding' AS embedding_text
            FROM {table_ref} t
            WHERE t.realm = $1{filters}
            ORDER BY t.id ASC{limit_clause}
            """
            try:
                rows = await conn.fetch(query, *params)
                edges = []
                for row in rows:
                    emb = None
                    if row.get('embedding_text'):
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
            except asyncpg.UndefinedTableError:
                raise TableNotFoundError(f"Edge table '{table_name}' does not exist.")

        if isinstance(self.connection, asyncpg.Pool):
            async with self.connection.acquire() as conn:
                return await _op(conn)
        else:
            return await _op(self.connection)

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
                all_params: list = []
                for r in realms:
                    tref = self._get_table_ref(table_name, r)
                    base = len(all_params)
                    all_params.append(r)
                    extra = ""
                    if space and space != RESERVED_SPACE_ALL:
                        all_params.append(space)
                        extra += f" AND (t.space = ${base+2} OR (${base+2} = 'default' AND (t.space IS NULL OR t.space = 'default')))"
                    if relation_type:
                        all_params.append(relation_type)
                        extra += f" AND t.relation_type = ${len(all_params)}"
                    parts.append(f"SELECT t.realm, t.id, t.space, t.fqid, t.from_id, t.to_id, t.relation_type, t.payload, t.created_at, t.updated_at, t.uuid::text AS uuid_text, to_jsonb(t)->>'embedding' AS embedding_text FROM {tref} t WHERE t.realm = ${base+1}{extra}")
                query = " UNION ALL ".join(parts) + " ORDER BY realm, id ASC"
                if limit:
                    all_params.append(limit)
                    query += f" LIMIT ${len(all_params)}"
                rows = await conn.fetch(query, *all_params)
            else:
                table_ref = self._get_table_ref(table_name, realms[0])
                params: list = [realms]
                extra = ""
                if space and space != RESERVED_SPACE_ALL:
                    params.append(space)
                    extra += f" AND (t.space = ${len(params)} OR (${len(params)} = 'default' AND (t.space IS NULL OR t.space = 'default')))"
                if relation_type:
                    params.append(relation_type)
                    extra += f" AND t.relation_type = ${len(params)}"
                limit_clause = ""
                if limit:
                    params.append(limit)
                    limit_clause = f" LIMIT ${len(params)}"
                query = f"""
                SELECT t.realm, t.id, t.space, t.fqid, t.from_id, t.to_id,
                       t.relation_type, t.payload, t.created_at, t.updated_at,
                       t.uuid::text AS uuid_text,
                       to_jsonb(t)->>'embedding' AS embedding_text
                FROM {table_ref} t
                WHERE t.realm = ANY($1::text[]){extra}
                ORDER BY t.realm, t.id ASC{limit_clause}
                """
                rows = await conn.fetch(query, *params)
            try:
                edges = []
                for row in rows:
                    emb = None
                    if row.get('embedding_text'):
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
            except asyncpg.UndefinedTableError:
                raise TableNotFoundError(f"Edge table '{table_name}' does not exist.")

        if isinstance(self.connection, asyncpg.Pool):
            async with self.connection.acquire() as conn:
                return await _op(conn)
        else:
            return await _op(self.connection)

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

        Each entry in *filters* becomes a ``payload->>'key' = value`` clause
        (all ANDed together).  Optional *relation_type* further restricts results.
        """
        self._validate_identifier(table_name)
        table_ref = self._get_table_ref(table_name, realm)

        async def _op(conn):
            params: list = [realm]
            clauses = ""
            if space and space != RESERVED_SPACE_ALL:
                params.append(space)
                clauses += f" AND (t.space = ${len(params)} OR (${len(params)} = 'default' AND (t.space IS NULL OR t.space = 'default')))"
            if relation_type:
                params.append(relation_type)
                clauses += f" AND t.relation_type = ${len(params)}"
            for key, val in filters.items():
                params.append(str(val))
                clauses += f" AND t.payload->>'{key}' = ${len(params)}"
            limit_clause = ""
            if limit:
                params.append(limit)
                limit_clause = f" LIMIT ${len(params)}"
            query = f"""
            SELECT t.realm, t.id, t.space, t.fqid, t.from_id, t.to_id,
                   t.relation_type, t.payload, t.created_at, t.updated_at,
                   t.uuid::text AS uuid_text,
                   to_jsonb(t)->>'embedding' AS embedding_text
            FROM {table_ref} t
            WHERE t.realm = $1{clauses}
            ORDER BY t.id ASC{limit_clause}
            """
            try:
                rows = await conn.fetch(query, *params)
                edges = []
                for row in rows:
                    emb = None
                    if row.get('embedding_text'):
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
            except asyncpg.UndefinedTableError:
                raise TableNotFoundError(f"Edge table '{table_name}' does not exist.")

        if isinstance(self.connection, asyncpg.Pool):
            async with self.connection.acquire() as conn:
                return await _op(conn)
        else:
            return await _op(self.connection)

    async def delete_edge(self, table_name: str, realm: str, edge_id: str, user_id: Optional[str] = None) -> bool:
        """Delete an edge."""
        self._validate_identifier(table_name)
        table_ref = self._get_table_ref(table_name, realm)
        e_str = str(edge_id)

        async def _op(conn):
            query = f"""
            DELETE FROM {table_ref} 
            WHERE realm = $1 AND ((CASE WHEN $2 ~ '^[0-9]+$' THEN id = $2::bigint ELSE FALSE END) OR fqid = $2)
            """
            try:
                res = await conn.execute(query, realm, e_str)
                return res == "DELETE 1"
            except asyncpg.UndefinedTableError:
                raise TableNotFoundError(f"Edge table '{table_name}' does not exist.")

        return await self._run_in_tx(_op, user_id)

    async def delete_realm(self, realm: str, user_id: Optional[str] = None) -> int:
        """Delete all rows belonging to a specific realm from all graph tables (vertices, edges, and audit tables)."""
        async def _op(conn):
            if self.schema_per_realm:
                query = """
                SELECT DISTINCT table_name 
                FROM information_schema.columns 
                WHERE column_name = 'realm' 
                  AND table_schema = $1
                """
                rows = await conn.fetch(query, realm)
            else:
                query = """
                SELECT DISTINCT table_name 
                FROM information_schema.columns 
                WHERE column_name = 'realm' 
                  AND table_schema = CURRENT_SCHEMA()
                """
                rows = await conn.fetch(query)

            tables = [row['table_name'] for row in rows]
            
            total_deleted = 0
            for table in tables:
                table_ref = self._get_table_ref(table, realm)
                delete_query = f'DELETE FROM {table_ref} WHERE realm = $1'
                res = await conn.execute(delete_query, realm)
                if res.startswith("DELETE "):
                    total_deleted += int(res.split(" ")[1])
            return total_deleted

        return await self._run_in_tx(_op, user_id)

    async def get_edge_schema(self, edge_table: str, realm: Optional[str] = None, conn = None) -> Dict[str, str]:
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
        WHERE con.conrelid = $1::regclass
          AND con.contype = 'f'
          AND a.attname IN ('from_id', 'to_id');
        """

        try:
            if conn:
                rows = await conn.fetch(query, table_ref)
            else:
                rows = await self._fetch(query, table_ref)
        except asyncpg.UndefinedTableError:
            raise TableNotFoundError(f"Edge table '{edge_table}' does not exist.")
        except Exception as e:
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

        results = []
        for edge_table in edge_tables:
            self._validate_identifier(edge_table)
            schema = await self.get_edge_schema(edge_table, realm=realm)

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
                    WHERE e.realm = $1 AND e.from_id = $2
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
                    WHERE e.realm = $1 AND e.to_id = $2
                    """
                ))

            for neighbor_table, sql in queries:
                rows = await self._fetch(sql, realm, v_id_int)
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

    def _edge_filter_sql(
        self,
        relation_types: Optional[List[str]],
        as_of: Optional[str],
        payload_null_keys: Optional[List[str]],
        space: Optional[str],
        valid_from_key: str,
        valid_to_key: str,
        first_param: int,
    ) -> Tuple[str, List[Any]]:
        """Build the WHERE fragment applied to every edge step of a traversal.

        Returned as a fragment plus its parameters so both traversal directions
        share one definition; drift between them would make an 'in' walk observe
        a different graph from an 'out' walk.
        """
        clauses: List[str] = []
        params: List[Any] = []
        n = first_param

        if relation_types:
            clauses.append(f"relation_type = ANY(${n}::text[])")
            params.append(list(relation_types))
            n += 1

        if space and space != RESERVED_SPACE_ALL:
            clauses.append(f"space = ${n}")
            params.append(space)
            n += 1

        # A relation with no stated period holds at every date: silence about
        # when a fact applied means it applied throughout, not that it never did.
        if as_of:
            vf = f"payload->>'{valid_from_key}'"
            vt = f"payload->>'{valid_to_key}'"
            at = self._padded_date_sql(f"${n}::text")
            clauses.append(
                f"(({vf} IS NULL OR {self._padded_date_sql(vf)} <= {at}) AND "
                f"({vt} IS NULL OR {self._padded_date_sql(vt)} >= {at}))"
            )
            params.append(as_of)
            n += 1

        for key in payload_null_keys or []:
            self._validate_identifier(key)
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

        # $1..$4 are realm, start id, start table and depth; filters follow.
        filter_sql, filter_params = self._edge_filter_sql(
            relation_types, as_of, payload_null_keys, space,
            valid_from_key, valid_to_key, first_param=5,
        )

        subqueries = []
        for edge_table in edge_tables:
            schema = await self.get_edge_schema(edge_table, realm=realm)
            from_ref = schema['from_id']
            to_ref = schema['to_id']
            edge_ref = self._get_table_ref(edge_table, realm)

            if direction in ('out', 'both'):
                subqueries.append(f"""
                SELECT 
                    to_id::text AS next_id, 
                    '{to_ref}'::text AS next_table,
                    id::text AS edge_id,
                    relation_type,
                    payload,
                    '{edge_table}'::text AS edge_table
                FROM {edge_ref}
                WHERE realm = $1 AND from_id = (CASE WHEN t.current_id ~ '^[0-9]+$' THEN t.current_id::bigint ELSE NULL END) AND t.current_table = '{from_ref}'{filter_sql}
                """)

            if direction in ('in', 'both'):
                subqueries.append(f"""
                SELECT 
                    from_id::text AS next_id, 
                    '{from_ref}'::text AS next_table,
                    id::text AS edge_id,
                    relation_type,
                    payload,
                    '{edge_table}'::text AS edge_table
                FROM {edge_ref}
                WHERE realm = $1 AND to_id = (CASE WHEN t.current_id ~ '^[0-9]+$' THEN t.current_id::bigint ELSE NULL END) AND t.current_table = '{to_ref}'{filter_sql}
                """)

        if not subqueries:
            return []

        union_all_steps = "\nUNION ALL\n".join(subqueries)

        cte_query = f"""
        WITH RECURSIVE graph_traversal AS (
            -- Anchor Member
            SELECT 
                $2::text AS current_id,
                $3::text AS current_table,
                0 AS depth,
                ARRAY[$3::text || ':' || $2::text]::text[] AS path,
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
            WHERE t.depth < $4
              AND NOT ((step.next_table || ':' || step.next_id) = ANY(t.path))
        )
        SELECT current_id, current_table, depth, path, edge_path, edge_ids FROM graph_traversal;
        """

        rows = await self._fetch(cte_query, realm, start_id_str, start_table, max_depth, *filter_params)

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

        subqueries = []
        for edge_table in edge_tables:
            schema = await self.get_edge_schema(edge_table, realm=realm)
            from_ref = schema['from_id']
            to_ref = schema['to_id']
            edge_ref = self._get_table_ref(edge_table, realm)

            if direction in ('out', 'both'):
                subqueries.append(f"""
                SELECT 
                    to_id::text AS next_id, 
                    '{to_ref}'::text AS next_table,
                    id::text AS edge_id,
                    relation_type,
                    '{edge_table}'::text AS edge_table
                FROM {edge_ref}
                WHERE realm = $1 AND from_id = (CASE WHEN t.current_id ~ '^[0-9]+$' THEN t.current_id::bigint ELSE NULL END) AND t.current_table = '{from_ref}'
                """)

            if direction in ('in', 'both'):
                subqueries.append(f"""
                SELECT 
                    from_id::text AS next_id, 
                    '{from_ref}'::text AS next_table,
                    id::text AS edge_id,
                    relation_type,
                    '{edge_table}'::text AS edge_table
                FROM {edge_ref}
                WHERE realm = $1 AND to_id = (CASE WHEN t.current_id ~ '^[0-9]+$' THEN t.current_id::bigint ELSE NULL END) AND t.current_table = '{to_ref}'
                """)

        if not subqueries:
            return None

        union_all_steps = "\nUNION ALL\n".join(subqueries)

        cte_query = f"""
        WITH RECURSIVE graph_traversal AS (
            -- Anchor Member
            SELECT 
                $2::text AS current_id,
                $3::text AS current_table,
                0 AS depth,
                ARRAY[$3::text || ':' || $2::text]::text[] AS path,
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
            WHERE t.depth < $4
              AND NOT ((step.next_table || ':' || step.next_id) = ANY(t.path))
              AND NOT (t.current_id = $5 AND t.current_table = $6)
        )
        SELECT depth, path, edge_path, edge_ids
        FROM graph_traversal 
        WHERE current_id = $5 AND current_table = $6
        ORDER BY depth ASC
        LIMIT 1;
        """

        if conn:
            row = await conn.fetchrow(cte_query, realm, start_id_str, start_table, max_depth, target_id_str, target_table)
        else:
            row = await self._fetchrow(cte_query, realm, start_id_str, start_table, max_depth, target_id_str, target_table)
            
        if not row:
            return None

        return {
            'depth': row['depth'],
            'path': row['path'],
            'edge_path': row['edge_path'],
            'edge_ids': row['edge_ids']
        }

    async def connected_components(
        self,
        realm: str,
        vertex_table: str,
        edge_tables: List[str],
        direction: str = "both",
    ) -> List[List[str]]:
        """Return connected components as lists of vertex IDs.

        Uses a BFS/UF approach via recursive CTE.  *direction* controls
        which edges are considered reachable ('out', 'in', or 'both').
        """
        self._validate_identifier(vertex_table)
        if direction not in ("out", "in", "both"):
            raise ValueError("direction must be 'out', 'in', or 'both'")

        vtable_ref = self._get_table_ref(vertex_table, realm)

        subqueries: list = []
        for et in edge_tables:
            self._validate_identifier(et)
            schema = await self.get_edge_schema(et, realm=realm)
            eref = self._get_table_ref(et, realm)
            if schema["from_id"] == vertex_table:
                if direction in ("out", "both"):
                    subqueries.append(
                        f"SELECT from_id::text AS src, to_id::text AS dst FROM {eref} WHERE realm = $1"
                    )
                if direction in ("in", "both"):
                    subqueries.append(
                        f"SELECT to_id::text AS src, from_id::text AS dst FROM {eref} WHERE realm = $1"
                    )
            elif schema["to_id"] == vertex_table:
                if direction in ("out", "both"):
                    subqueries.append(
                        f"SELECT to_id::text AS src, from_id::text AS dst FROM {eref} WHERE realm = $1"
                    )
                if direction in ("in", "both"):
                    subqueries.append(
                        f"SELECT from_id::text AS src, to_id::text AS dst FROM {eref} WHERE realm = $1"
                    )

        if not subqueries:
            all_ids = await self._fetch(
                f"SELECT id::text FROM {vtable_ref} WHERE realm = $1", realm
            )
            return [[r["id"] for r in all_ids]]

        edges_union = "\nUNION ALL\n".join(subqueries)

        query = f"""
        WITH all_edges AS ({edges_union}),
        RECURSIVE flood AS (
            SELECT id::text AS vid, id::text AS component_root
            FROM {vtable_ref} WHERE realm = $1

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

        async def _op(conn):
            try:
                rows = await conn.fetch(query, realm)
                return [list(r["members"]) for r in rows]
            except asyncpg.UndefinedTableError:
                raise TableNotFoundError(f"Table not found during connected_components.")

        if isinstance(self.connection, asyncpg.Pool):
            async with self.connection.acquire() as conn:
                return await _op(conn)
        else:
            return await _op(self.connection)

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
        """Dijkstra-style weighted shortest path using a payload field as weight.

        *weight_field* names the key inside each edge's payload that holds the
        numeric weight (defaults to ``"weight"``).  Edges without the field are
        assigned weight 1.0.

        Returns ``None`` when no path exists, otherwise a dict with keys
        ``depth``, ``path``, ``edge_path``, ``edge_ids``, and ``total_weight``.
        """
        self._validate_identifier(start_table)
        self._validate_identifier(target_table)
        if direction not in ("out", "in", "both"):
            raise ValueError("direction must be 'out', 'in', or 'both'")

        start_id_str = str(start_id).split("/")[-1] if "/" in str(start_id) else str(start_id)
        target_id_str = str(target_id).split("/")[-1] if "/" in str(target_id) else str(target_id)

        subqueries: list = []
        for et in edge_tables:
            schema = await self.get_edge_schema(et, realm=realm)
            from_ref = schema["from_id"]
            to_ref = schema["to_id"]
            eref = self._get_table_ref(et, realm)

            if direction in ("out", "both"):
                subqueries.append(f"""
                SELECT to_id::text AS next_id,
                       '{to_ref}'::text AS next_table,
                       id::text AS edge_id,
                       relation_type,
                       '{et}'::text AS edge_table,
                       COALESCE((payload->>'{weight_field}')::double precision, 1.0) AS edge_weight
                FROM {eref}
                WHERE realm = $1
                  AND from_id = (CASE WHEN t.current_id ~ '^[0-9]+$' THEN t.current_id::bigint ELSE NULL END)
                  AND t.current_table = '{from_ref}'
                """)

            if direction in ("in", "both"):
                subqueries.append(f"""
                SELECT from_id::text AS next_id,
                       '{from_ref}'::text AS next_table,
                       id::text AS edge_id,
                       relation_type,
                       '{et}'::text AS edge_table,
                       COALESCE((payload->>'{weight_field}')::double precision, 1.0) AS edge_weight
                FROM {eref}
                WHERE realm = $1
                  AND to_id = (CASE WHEN t.current_id ~ '^[0-9]+$' THEN t.current_id::bigint ELSE NULL END)
                  AND t.current_table = '{to_ref}'
                """)

        if not subqueries:
            return None

        union_all = "\nUNION ALL\n".join(subqueries)

        cte_query = f"""
        WITH RECURSIVE graph_traversal AS (
            SELECT
                $2::text AS current_id,
                $3::text AS current_table,
                0 AS depth,
                ARRAY[$3::text || ':' || $2::text]::text[] AS path,
                ARRAY[]::text[] AS edge_path,
                ARRAY[]::text[] AS edge_ids,
                0.0::double precision AS total_weight

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
            WHERE t.depth < $4
              AND NOT ((step.next_table || ':' || step.next_id) = ANY(t.path))
        )
        SELECT depth, path, edge_path, edge_ids, total_weight
        FROM graph_traversal
        WHERE current_id = $5 AND current_table = $6
        ORDER BY total_weight ASC
        LIMIT 1;
        """

        async def _op(conn):
            row = await conn.fetchrow(
                cte_query, realm, start_id_str, start_table,
                max_depth, target_id_str, target_table,
            )
            if not row:
                return None
            return {
                "depth": row["depth"],
                "path": row["path"],
                "edge_path": row["edge_path"],
                "edge_ids": row["edge_ids"],
                "total_weight": float(row["total_weight"]),
            }

        if isinstance(self.connection, asyncpg.Pool):
            async with self.connection.acquire() as conn:
                return await _op(conn)
        else:
            return await _op(self.connection)

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
        # RETURNING can only name the target table unqualified: PostgreSQL
        # rejects to_jsonb("schema"."table") there with "missing FROM-clause
        # entry", which broke every add_vertex_data in schema_per_realm mode.
        data_table_bare = f'"{table_name}_data"'
        vec_str = f"[{','.join(str(x) for x in embedding)}]" if embedding is not None else None

        async def _op(conn):
            has_emb = False
            if vec_str:
                data_t_name = f"{table_name}_data"
                if self.schema_per_realm:
                    has_emb = await conn.fetchval("SELECT 1 FROM information_schema.columns WHERE table_schema = $1 AND table_name = $2 AND column_name = 'embedding'", realm, data_t_name) is not None
                else:
                    has_emb = await conn.fetchval("SELECT 1 FROM information_schema.columns WHERE table_name = $1 AND column_name = 'embedding'", data_t_name) is not None

            if vec_str and not has_emb:
                logger.warning(
                    f"Data table '{table_name}_data' has no embedding column; the supplied embedding "
                    f"was not stored. Recreate '{table_name}' with a vector_dim to enable it."
                )

            if vec_str and has_emb:
                if timestamp:
                    query = f"""
                    INSERT INTO {data_table_ref} (realm, id, payload, timestamp, embedding)
                    VALUES ($1, $2, $3::jsonb, $4, $5::vector)
                    RETURNING data_id, realm, id, payload, timestamp, to_jsonb({data_table_bare})->>'embedding' AS embedding_text
                    """
                    row = await conn.fetchrow(query, realm, v_id_int, payload_json, timestamp, vec_str)
                else:
                    query = f"""
                    INSERT INTO {data_table_ref} (realm, id, payload, embedding)
                    VALUES ($1, $2, $3::jsonb, $4::vector)
                    RETURNING data_id, realm, id, payload, timestamp, to_jsonb({data_table_bare})->>'embedding' AS embedding_text
                    """
                    row = await conn.fetchrow(query, realm, v_id_int, payload_json, vec_str)
            else:
                if timestamp:
                    query = f"""
                    INSERT INTO {data_table_ref} (realm, id, payload, timestamp)
                    VALUES ($1, $2, $3::jsonb, $4)
                    RETURNING data_id, realm, id, payload, timestamp, to_jsonb({data_table_bare})->>'embedding' AS embedding_text
                    """
                    row = await conn.fetchrow(query, realm, v_id_int, payload_json, timestamp)
                else:
                    query = f"""
                    INSERT INTO {data_table_ref} (realm, id, payload)
                    VALUES ($1, $2, $3::jsonb)
                    RETURNING data_id, realm, id, payload, timestamp, to_jsonb({data_table_bare})->>'embedding' AS embedding_text
                    """
                    row = await conn.fetchrow(query, realm, v_id_int, payload_json)

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
        WHERE d.realm = $1 AND d.id = $2
        ORDER BY d.timestamp DESC, d.data_id DESC
        {limit_clause}
        """
        rows = await self._fetch(query, realm, v_id_int)
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
        WHERE d.realm = $1 AND d.data_id = $2
        """
        rows = await self._fetch(query, realm, d_id_int)
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

