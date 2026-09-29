"""Telling a concurrent creator apart from a real DDL failure.

``IF NOT EXISTS`` is not atomic. It checks the catalog and then creates, and two
sessions can both pass the check before either inserts -- so one of them gets a
uniqueness violation from a system index rather than the silent no-op the clause
implies. A multi-replica deployment writing a new space hits this on the first
write, and with six replicas racing, five of them fail.

The rule here: a provisioning statement that lost a race has still achieved what
the caller wanted, because the object exists afterwards and that is the whole
contract. Anything else is a real failure and must surface -- a missing table or
index that was swallowed becomes a silent wrong answer later, which is far worse
than a loud failure now.
"""
from typing import Optional

# SQLSTATEs meaning "someone else already created this".
#
#   42P07 duplicate_table      CREATE TABLE lost the race
#   42P06 duplicate_schema     CREATE SCHEMA lost the race
#   42710 duplicate_object     a trigger, constraint or index name was taken
#   42701 duplicate_column     ALTER TABLE ... ADD COLUMN lost the race
#   42P16 invalid_table_definition, raised by some ALTERs re-adding a column
#   23505 unique_violation     the catalog's own index rejected the insert;
#                              this is what CREATE SCHEMA actually raises, on
#                              pg_namespace_nspname_index
_BENIGN_SQLSTATES = frozenset({"42P07", "42P06", "42710", "42701", "42P16", "23505"})

# Substrings for drivers and wrappers that do not surface a sqlstate. Kept
# narrow on purpose: matching on "already exists" alone would also swallow a
# genuine uniqueness violation from application data.
_BENIGN_TEXT = (
    "already exists",
    "duplicate key value violates unique constraint",
    "duplicate column",
    "tuple concurrently updated",       # CREATE OR REPLACE FUNCTION raced
    "tuple concurrently deleted",
)

# Catalog indexes. A violation against one of these is DDL racing; a violation
# against anything else is application data and must not be swallowed.
_CATALOG_INDEXES = (
    "pg_namespace_nspname_index",
    "pg_type_typname_nsp_index",
    "pg_class_relname_nsp_index",
    "pg_proc_proname_args_nsp_index",
    "pg_constraint_conrelid_contypid_conname_index",
    "pg_trigger_tgrelid_tgname_index",
    "pg_attribute_relid_attnam_index",
)


def sqlstate_of(exc: BaseException) -> Optional[str]:
    """The SQLSTATE, wherever the driver happens to have put it."""
    for candidate in (exc, getattr(exc, "orig", None), getattr(exc, "__cause__", None)):
        if candidate is None:
            continue
        code = getattr(candidate, "sqlstate", None) or getattr(candidate, "pgcode", None)
        if code:
            return str(code)
    return None


def is_concurrent_creation(exc: BaseException) -> bool:
    """True when this DDL failed only because another session got there first.

    A unique_violation is the ambiguous case: it is what CREATE SCHEMA raises
    when it loses, and also what an application insert raises on a duplicate
    key. It counts as a race only when the constraint named is one of
    PostgreSQL's own catalog indexes, so real data violations still surface.
    """
    text = str(exc).lower()
    code = sqlstate_of(exc)

    if code == "23505" or "duplicate key value violates unique constraint" in text:
        return any(idx in text for idx in _CATALOG_INDEXES)

    if code in _BENIGN_SQLSTATES:
        return True

    return any(fragment in text for fragment in _BENIGN_TEXT)


# Not a race to swallow, but not a defeat either. Concurrent DDL on the same
# objects genuinely deadlocks -- six replicas provisioning one realm take
# AccessExclusiveLock on the schema, the trigger function and the catalog in
# whatever order they arrive -- and PostgreSQL resolves it by killing all but
# one participant. The loser's work is still wanted, and PostgreSQL guarantees
# at least one winner, so retrying converges rather than spinning.
#
#   40P01 deadlock_detected
#   55P03 lock_not_available
#   XX000 "tuple concurrently updated", from CREATE OR REPLACE FUNCTION racing
_RETRYABLE_SQLSTATES = frozenset({"40P01", "55P03"})
_RETRYABLE_TEXT = ("deadlock detected", "tuple concurrently updated",
                   "tuple concurrently deleted", "could not obtain lock")


def is_retryable_ddl(exc: BaseException) -> bool:
    """True when this DDL should simply be attempted again."""
    if sqlstate_of(exc) in _RETRYABLE_SQLSTATES:
        return True
    text = str(exc).lower()
    return any(fragment in text for fragment in _RETRYABLE_TEXT)
