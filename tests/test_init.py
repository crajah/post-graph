"""Tests for the post_graph package exports."""

import post_graph


class TestPackageExports:
    def test_version_is_string(self):
        assert isinstance(post_graph.__version__, str)
        parts = post_graph.__version__.split(".")
        assert len(parts) == 3

    def test_reserved_space_all(self):
        assert post_graph.RESERVED_SPACE_ALL == "__all__"

    def test_error_classes_exported(self):
        assert post_graph.PostGraphError is not None
        assert post_graph.VertexNotFoundError is not None
        assert post_graph.EdgeNotFoundError is not None
        assert post_graph.TableExistsError is not None
        assert post_graph.TableNotFoundError is not None
        assert post_graph.CyclicReferenceError is not None
        assert post_graph.ReservedSpaceError is not None

    def test_model_classes_exported(self):
        assert post_graph.Vertex is not None
        assert post_graph.Edge is not None
        assert post_graph.TraversalStep is not None

    def test_asyncpg_client_exported(self):
        assert post_graph.AsyncPostGraph is not None

    def test_sqlalchemy_client_exported_or_none(self):
        # SQLAlchemyPostGraph is None when sqlalchemy is not installed
        assert hasattr(post_graph, "SQLAlchemyPostGraph")

    def test_all_list(self):
        for name in post_graph.__all__:
            assert hasattr(post_graph, name), f"{name} in __all__ but not an attribute"
