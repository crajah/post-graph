"""Pieces that both clients share: advisory lock keys, and orderable columns.

Advisory lock keys, derived so that no caller has to invent an integer.

``pg_advisory_xact_lock`` keys are database-wide: not per-table, not per-schema,
not per-database-user. Two unrelated callers that happen to pick the same
integer block each other for reasons neither can see. So post-graph derives the
key from the arguments a caller already has -- a realm and a table -- and never
exposes an integer to be collided with.

The two-argument form is used rather than the one-argument form, because
PostgreSQL keeps them in *separate* lock spaces. A lock taken as
``(classid, objid)`` cannot collide with anything that takes a single bigint
key, whatever integer it picks. Fixing ``classid`` to a constant of
post-graph's own therefore isolates every lock post-graph takes from every lock
anything else in the database takes, which a folded single bigint could not do.

The hash must be stable **across processes**, which rules out the obvious
choice. Python's ``hash()`` of a string is salted per interpreter by
PYTHONHASHSEED, so two replicas would derive different keys for the same table,
take different locks and serialise nothing -- the exact silent failure this
exists to prevent, reintroduced by the key derivation itself. A cryptographic
digest is used for stability, not for secrecy.
"""
import hashlib

# Both arguments of the two-argument form are int4, so keys are folded to a
# signed 32-bit range.
_INT4_BYTES = 4


def _stable_int4(*parts: str) -> int:
    """A process-stable, non-negative int4 derived from the given strings.

    Non-negative deliberately, at the cost of one bit. ``pg_advisory_xact_lock``
    accepts signed int4, but ``pg_locks`` reports the keys as unsigned, so a
    negative key shows up there as a large positive one and nothing the library
    printed matches what an operator reading pg_locks sees. Spending a bit to
    keep those two numbers identical is worth it the first time someone has to
    debug a lock.
    """
    joined = "\x00".join(parts).encode("utf-8")
    digest = hashlib.sha256(joined).digest()
    return int.from_bytes(digest[:_INT4_BYTES], "big", signed=False) & 0x7FFFFFFF


# post-graph's own class identifier. Every advisory lock this library takes
# carries it, so a lock held by post-graph is distinguishable in pg_locks from
# a lock held by anything else, and cannot collide with it.
ADVISORY_CLASSID = _stable_int4("post-graph.advisory")


def advisory_key(realm: str, table_name: str) -> int:
    """The objid for a (realm, table) pair.

    Folding 256 bits into 32 means two tables can in principle share a key. The
    consequence is over-serialisation -- two tables taking turns that did not
    have to -- and never two writers believing they hold different locks, which
    is the only direction that could corrupt anything.
    """
    return _stable_int4(realm, table_name)


# Real columns that may be ordered on. A whitelist rather than validation,
# because the value reaches the SQL text verbatim: there is no bind parameter
# for an ORDER BY target, so the only safe set is a closed one.
#
# Separate from `order_by`, which names a payload key. `id` is a perfectly
# legal JSON key, so overloading `order_by="id"` would silently change what an
# existing caller's query means.
VERTEX_ORDER_COLUMNS = frozenset({"id", "created_at", "updated_at", "space", "uuid", "fqid"})
EDGE_ORDER_COLUMNS = frozenset(
    {"id", "created_at", "updated_at", "space", "from_id", "to_id", "relation_type"}
)


def validate_order_column(column: str, allowed) -> str:
    """Admit *column* only if it is in *allowed*; return it unchanged."""
    if column not in allowed:
        raise ValueError(
            f"order_by_column={column!r} is not an orderable column. "
            f"Allowed: {', '.join(sorted(allowed))}. To order by a payload key, "
            f"use order_by= instead."
        )
    return column
