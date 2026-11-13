from typing import Optional


class PostGraphError(Exception):
    """Base exception class for all post-graph library errors."""
    pass


class VertexNotFoundError(PostGraphError):
    """Raised when a referenced vertex cannot be found."""
    pass


class EdgeNotFoundError(PostGraphError):
    """Raised when a referenced edge cannot be found."""
    pass


class TableExistsError(PostGraphError):
    """Raised when attempting to create a table that already exists or on uniqueness violation."""
    pass


class TableNotFoundError(PostGraphError):
    """Raised when an operation is performed on a non-existent table."""
    pass


class CyclicReferenceError(PostGraphError):
    """Raised when adding an edge would introduce a cyclic reference in the graph."""
    pass


class ReservedSpaceError(PostGraphError):
    """Raised when a reserved space name like '__all__' is used in a write operation."""
    pass


class UniqueConstraintError(TableExistsError):
    """A write violated a UNIQUE index or constraint.

    Subclasses ``TableExistsError`` so callers written against earlier releases
    -- which caught that on ``add_vertex`` for a duplicate id -- keep working,
    while ``constraint_name`` says *which* uniqueness was violated. Without it
    a caller distinguishing "this username is taken" from "this id is taken"
    had to read ``__cause__`` and import the driver's exception type, which is
    the thing a caller routing all its SQL through one library is avoiding.

    ``constraint_name`` is whatever PostgreSQL reported and may be ``None``
    when the driver did not surface it.
    """

    def __init__(self, message: str, *, constraint_name: Optional[str] = None) -> None:
        super().__init__(message)
        self.constraint_name = constraint_name


class ConcurrentModificationError(PostGraphError):
    """A compare-and-swap write found the row had moved on.

    Raised when ``expected_updated_at`` does not match the stored row, meaning
    somebody else wrote between the caller's read and its write. The caller's
    change was not applied.
    """
    pass


class NestedTransactionError(PostGraphError):
    """``transaction()`` was opened on a client already inside one.

    Not permitted rather than silently nested: a nested block would become a
    SAVEPOINT whose exit commits nothing, so a caller believing it had
    committed would be wrong in a way nothing reports.
    """
    pass


class CommitFailedError(PostGraphError):
    """The block succeeded and the COMMIT did not.

    Separate from an ordinary failure because the caller's situation differs:
    everything inside the block ran, so any external side effect it performed
    -- a charge, an email, a webhook -- has happened, while the database may
    or may not hold the writes. A pre-commit failure needs a retry; this needs
    reconciliation.
    """
    pass
