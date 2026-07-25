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
