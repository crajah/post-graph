"""Promoted payload columns: hot JSONB keys lifted into indexed columns.

Properties live in ``payload`` JSONB, which is flexible but opaque to the
planner. A filter such as ``payload->>'valid_from' <= '2005-06-15'`` cannot use
an index, so every temporal traversal step degrades to a sequential scan — and
the as-of filter runs on *every* step of *every* walk, which is the hottest path
in the library.

A generated column fixes this without touching the write path: PostgreSQL
maintains it from ``payload`` on insert and update, so callers keep writing
plain JSON and the planner gets a real indexed column.

Two flavours:

``pt_<key>``  temporal. Holds the ISO date normalised to ``YYYY-MM-DD``, so a
              partial date ('2024' or '2024-06') still orders correctly against
              a full one. Text rather than DATE because casting text to date is
              only STABLE — PostgreSQL rejects it in a generated column — while
              split_part/COALESCE/NULLIF are IMMUTABLE and permitted. ISO-8601
              sorts lexically, so text comparison and date comparison agree.

``p_<key>``   generic. Holds ``payload->>'<key>'`` verbatim, for equality and
              presence tests. This is how a caller makes its own hot key fast
              without post-graph having to know what the key means.

Measured on 65k rows, the temporal filter goes from a 588-buffer sequential scan
to a 27-buffer bitmap index scan, and from ~31ms to ~2.5ms warm. The margin
grows with the table.
"""

import re
from typing import List, Optional, Sequence, Tuple

#: Payload keys promoted as temporal columns unless a caller overrides them.
#: These are post-graph's own vocabulary — ``traverse(as_of=...)`` already
#: defaults to them — so promoting them by default surprises nobody.
DEFAULT_TEMPORAL_KEYS: Tuple[str, str] = ('valid_from', 'valid_to')

_IDENT = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')


def validate_key(key: str) -> str:
    """Payload keys become column names, so they must be safe identifiers."""
    if not key or not isinstance(key, str) or not _IDENT.match(key):
        raise ValueError(
            f"Invalid payload key for promotion: {key!r}. Must be alphanumeric "
            f"and underscores only, starting with a letter or underscore."
        )
    return key


def temporal_column(key: str) -> str:
    """Column name holding the normalised ISO date for ``key``."""
    return f"pt_{validate_key(key)}"


def generic_column(key: str) -> str:
    """Column name holding the raw text value of ``key``."""
    return f"p_{validate_key(key)}"


def padded_date_sql(expr: str) -> str:
    """Normalise an ISO date expression to YYYY-MM-DD.

    '2024' becomes '2024-01-01' and '2024-06' becomes '2024-06-01', so a partial
    date compares correctly against a full one instead of sorting as a shorter
    string. Every function here is IMMUTABLE, which is what lets the result be
    used in a generated column.
    """
    return (
        f"(split_part({expr}, '-', 1) || '-' || "
        f"COALESCE(NULLIF(split_part({expr}, '-', 2), ''), '01') || '-' || "
        f"COALESCE(NULLIF(split_part({expr}, '-', 3), ''), '01'))"
    )


def temporal_generation_expr(key: str) -> str:
    """Generation expression for a temporal promoted column.

    NULL payload key yields NULL rather than '--', so 'no stated period' stays
    distinguishable from 'a period that failed to parse'.
    """
    src = f"payload->>'{validate_key(key)}'"
    return f"CASE WHEN {src} IS NULL THEN NULL ELSE {padded_date_sql(src)} END"


def generic_generation_expr(key: str) -> str:
    return f"payload->>'{validate_key(key)}'"


def column_ddl(table_ref: str, table_name: str, key: str, temporal: bool) -> List[str]:
    """ADD COLUMN plus index statements for one promoted key.

    Both are IF NOT EXISTS, so this is safe to re-run against a table that
    already has the column — which is how existing tables acquire it.
    """
    col = temporal_column(key) if temporal else generic_column(key)
    expr = temporal_generation_expr(key) if temporal else generic_generation_expr(key)
    return [
        f'ALTER TABLE {table_ref} ADD COLUMN IF NOT EXISTS "{col}" TEXT '
        f'GENERATED ALWAYS AS ({expr}) STORED;',
        f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_{col}" ON {table_ref} ("{col}");',
    ]


def all_column_ddl(
    table_ref: str,
    table_name: str,
    temporal_keys: Optional[Sequence[str]] = None,
    promoted_keys: Optional[Sequence[str]] = None,
) -> List[str]:
    """Every DDL statement needed to promote the requested keys on one table."""
    stmts: List[str] = []
    for key in (temporal_keys if temporal_keys is not None else DEFAULT_TEMPORAL_KEYS):
        stmts += column_ddl(table_ref, table_name, key, temporal=True)
    for key in (promoted_keys or []):
        stmts += column_ddl(table_ref, table_name, key, temporal=False)
    return stmts
