"""Tests for the post_graph.errors module."""

from post_graph.errors import (
    PostGraphError,
    VertexNotFoundError,
    EdgeNotFoundError,
    TableExistsError,
    TableNotFoundError,
    CyclicReferenceError,
    ReservedSpaceError,
)


class TestErrorHierarchy:
    def test_base_is_exception(self):
        assert issubclass(PostGraphError, Exception)

    def test_vertex_not_found(self):
        assert issubclass(VertexNotFoundError, PostGraphError)
        err = VertexNotFoundError("gone")
        assert str(err) == "gone"

    def test_edge_not_found(self):
        assert issubclass(EdgeNotFoundError, PostGraphError)

    def test_table_exists(self):
        assert issubclass(TableExistsError, PostGraphError)

    def test_table_not_found(self):
        assert issubclass(TableNotFoundError, PostGraphError)

    def test_cyclic_reference(self):
        assert issubclass(CyclicReferenceError, PostGraphError)

    def test_reserved_space(self):
        assert issubclass(ReservedSpaceError, PostGraphError)
        err = ReservedSpaceError("__all__ is reserved")
        assert "__all__" in str(err)

    def test_catch_all_with_base(self):
        for cls in (
            VertexNotFoundError,
            EdgeNotFoundError,
            TableExistsError,
            TableNotFoundError,
            CyclicReferenceError,
            ReservedSpaceError,
        ):
            try:
                raise cls("test")
            except PostGraphError:
                pass
