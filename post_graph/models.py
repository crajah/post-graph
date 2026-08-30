from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, List


@dataclass
class Vertex:
    realm: str
    id: str
    space: Optional[str] = "default"
    payload: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    table_name: Optional[str] = None
    fqid: Optional[str] = None
    embedding: Optional[List[float]] = None
    embeddings: Optional[Dict[str, List[float]]] = None
    uuid: Optional[str] = None
    _client: Optional[Any] = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.fqid is None and self.realm and self.table_name and self.id:
            self.fqid = f"{self.realm}/{self.table_name}/{self.id}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert the Vertex object to a dictionary."""
        d: Dict[str, Any] = {
            "realm": self.realm,
            "id": self.id,
            "space": self.space,
            "uuid": self.uuid,
            "fqid": self.fqid,
            "payload": self.payload,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "table_name": self.table_name,
            "embedding": self.embedding,
        }
        if self.embeddings:
            d["embeddings"] = self.embeddings
        return d

    async def to(self, edge_table: str, direction: str = 'out') -> List['TraversalStep']:
        """Traverse to neighboring vertices via the specified edge table (defaults to outgoing)."""
        if not self._client:
            from post_graph.errors import PostGraphError
            raise PostGraphError("Vertex was not loaded with a database client reference. Traversal is unavailable.")
        
        neighbors = await self._client.get_neighbors(
            realm=self.realm,
            vertex_table=self.table_name,
            vertex_id=self.id,
            edge_tables=[edge_table],
            direction=direction
        )
        return [TraversalStep(edge=edge, neighbor_vertex=neighbor) for neighbor, edge in neighbors]

    async def from_(self, edge_table: str) -> List['TraversalStep']:
        """Traverse to neighboring vertices via incoming edges (reverse traversal)."""
        return await self.to(edge_table, direction='in')

    async def incoming(self, edge_table: str) -> List['TraversalStep']:
        """Traverse to neighboring vertices via incoming edges (synonym for from_)."""
        return await self.to(edge_table, direction='in')

    async def outgoing(self, edge_table: str) -> List['TraversalStep']:
        """Traverse to neighboring vertices via outgoing edges (synonym for to)."""
        return await self.to(edge_table, direction='out')

    async def add_edge_to(
        self,
        to_id: str,
        edge_table: str,
        edge_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        space: Optional[str] = None
    ) -> 'Edge':
        """Create a new outgoing edge from this vertex to another vertex."""
        if not self._client:
            from post_graph.errors import PostGraphError
            raise PostGraphError("Vertex was not loaded with a database client reference. Edge creation is unavailable.")

        return await self._client.add_edge(
            table_name=edge_table,
            realm=self.realm,
            edge_id=edge_id,
            from_id=self.id,
            to_id=to_id,
            relation_type=edge_table,  # Default relation_type to the edge table name
            payload=payload,
            user_id=user_id,
            space=space or self.space
        )

    async def add_edge_from(
        self,
        from_id: str,
        edge_table: str,
        edge_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        space: Optional[str] = None
    ) -> 'Edge':
        """Create a new incoming edge from another vertex to this vertex."""
        if not self._client:
            from post_graph.errors import PostGraphError
            raise PostGraphError("Vertex was not loaded with a database client reference. Edge creation is unavailable.")

        return await self._client.add_edge(
            table_name=edge_table,
            realm=self.realm,
            edge_id=edge_id,
            from_id=from_id,
            to_id=self.id,
            relation_type=edge_table,
            payload=payload,
            user_id=user_id,
            space=space or self.space
        )

    async def delete(self, user_id: Optional[str] = None) -> bool:
        """Delete this vertex. All referencing edges to and from it will be automatically deleted."""
        if not self._client:
            from post_graph.errors import PostGraphError
            raise PostGraphError("Vertex was not loaded with a database client reference. Deletion is unavailable.")
        return await self._client.delete_vertex(
            table_name=self.table_name,
            realm=self.realm,
            vertex_id=self.id,
            user_id=user_id
        )

    async def add_data(
        self,
        payload: Dict[str, Any],
        timestamp: Optional[datetime] = None,
        embedding: Optional[List[float]] = None,
        user_id: Optional[str] = None
    ) -> 'DataRecord':
        """Append a data record to this vertex's {table_name}_data table."""
        if not self._client:
            from post_graph.errors import PostGraphError
            raise PostGraphError("Vertex was not loaded with a database client reference.")
        return await self._client.add_vertex_data(
            table_name=self.table_name,
            realm=self.realm,
            vertex_id=self.id,
            payload=payload,
            timestamp=timestamp,
            embedding=embedding,
            user_id=user_id
        )

    async def get_data(self, limit: Optional[int] = None) -> List['DataRecord']:
        """Fetch append-only data records for this vertex."""
        if not self._client:
            from post_graph.errors import PostGraphError
            raise PostGraphError("Vertex was not loaded with a database client reference.")
        return await self._client.get_vertex_data(
            table_name=self.table_name,
            realm=self.realm,
            vertex_id=self.id,
            limit=limit
        )

    async def get_latest_data(self) -> Optional['DataRecord']:
        """Fetch the latest append-only data record (version) for this vertex."""
        if not self._client:
            from post_graph.errors import PostGraphError
            raise PostGraphError("Vertex was not loaded with a database client reference.")
        return await self._client.get_latest_vertex_data(
            table_name=self.table_name,
            realm=self.realm,
            vertex_id=self.id
        )

    async def get_data_by_id(self, data_id: Any) -> Optional['DataRecord']:
        """Fetch a specific append-only data record / version by sequential data_id."""
        if not self._client:
            from post_graph.errors import PostGraphError
            raise PostGraphError("Vertex was not loaded with a database client reference.")
        return await self._client.get_vertex_data_by_id(
            table_name=self.table_name,
            realm=self.realm,
            data_id=data_id
        )


@dataclass
class Edge:
    realm: str
    id: str
    from_id: str
    to_id: str
    relation_type: str
    space: Optional[str] = "default"
    payload: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    table_name: Optional[str] = None
    fqid: Optional[str] = None
    embedding: Optional[List[float]] = None
    embeddings: Optional[Dict[str, List[float]]] = None
    uuid: Optional[str] = None
    _client: Optional[Any] = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.fqid is None and self.realm and self.table_name and self.id:
            self.fqid = f"{self.realm}/{self.table_name}/{self.id}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert the Edge object to a dictionary."""
        d: Dict[str, Any] = {
            "realm": self.realm,
            "id": self.id,
            "space": self.space,
            "uuid": self.uuid,
            "fqid": self.fqid,
            "from_id": self.from_id,
            "to_id": self.to_id,
            "relation_type": self.relation_type,
            "payload": self.payload,
            "embedding": self.embedding,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "table_name": self.table_name,
        }
        if self.embeddings:
            d["embeddings"] = self.embeddings
        return d

    async def delete(self, user_id: Optional[str] = None) -> bool:
        """Delete this edge."""
        if not self._client:
            from post_graph.errors import PostGraphError
            raise PostGraphError("Edge was not loaded with a database client reference. Deletion is unavailable.")
        return await self._client.delete_edge(
            table_name=self.table_name,
            realm=self.realm,
            edge_id=self.id,
            user_id=user_id
        )

    async def add_data(
        self,
        payload: Dict[str, Any],
        timestamp: Optional[datetime] = None,
        embedding: Optional[List[float]] = None,
        user_id: Optional[str] = None
    ) -> 'DataRecord':
        """Append a data record to this edge's {table_name}_data table."""
        if not self._client:
            from post_graph.errors import PostGraphError
            raise PostGraphError("Edge was not loaded with a database client reference.")
        return await self._client.add_edge_data(
            table_name=self.table_name,
            realm=self.realm,
            edge_id=self.id,
            payload=payload,
            timestamp=timestamp,
            embedding=embedding,
            user_id=user_id
        )

    async def get_data(self, limit: Optional[int] = None) -> List['DataRecord']:
        """Fetch append-only data records for this edge."""
        if not self._client:
            from post_graph.errors import PostGraphError
            raise PostGraphError("Edge was not loaded with a database client reference.")
        return await self._client.get_edge_data(
            table_name=self.table_name,
            realm=self.realm,
            edge_id=self.id,
            limit=limit
        )

    async def get_latest_data(self) -> Optional['DataRecord']:
        """Fetch the latest append-only data record for this edge."""
        if not self._client:
            from post_graph.errors import PostGraphError
            raise PostGraphError("Edge was not loaded with a database client reference.")
        return await self._client.get_latest_edge_data(
            table_name=self.table_name,
            realm=self.realm,
            edge_id=self.id
        )

    async def get_data_by_id(self, data_id: Any) -> Optional['DataRecord']:
        """Fetch a specific append-only data record by sequential data_id for this edge."""
        if not self._client:
            from post_graph.errors import PostGraphError
            raise PostGraphError("Edge was not loaded with a database client reference.")
        return await self._client.get_edge_data_by_id(
            table_name=self.table_name,
            realm=self.realm,
            data_id=data_id
        )


@dataclass
class DataRecord:
    data_id: str
    realm: str
    id: str
    space: Optional[str] = "default"
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert DataRecord object to a dictionary."""
        return {
            "data_id": self.data_id,
            "realm": self.realm,
            "id": self.id,
            "space": self.space,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "embedding": self.embedding
        }


@dataclass
class TraversalStep:
    edge: Edge
    neighbor_vertex: Vertex

    def vertex(self) -> Vertex:
        """Get the neighboring vertex for this step."""
        return self.neighbor_vertex

    async def add_edge_to(
        self,
        to_id: str,
        edge_table: str,
        edge_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> Edge:
        """Create a new outgoing edge from this step's neighbor vertex to another vertex."""
        return await self.neighbor_vertex.add_edge_to(
            to_id=to_id,
            edge_table=edge_table,
            edge_id=edge_id,
            payload=payload,
            user_id=user_id
        )

    async def add_edge_from(
        self,
        from_id: str,
        edge_table: str,
        edge_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> Edge:
        """Create a new incoming edge from another vertex to this step's neighbor vertex."""
        return await self.neighbor_vertex.add_edge_from(
            from_id=from_id,
            edge_table=edge_table,
            edge_id=edge_id,
            payload=payload,
            user_id=user_id
        )


class _FilterSentinel:
    """A named marker for a filter state that no ordinary Python value can spell.

    ``None`` is deliberately not a filter value: it could mean "the key holds
    JSON null", "the key is missing", or "ignore this filter", and each reading
    silently produces different rows. The find methods reject it and take one
    of these instead.
    """
    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:
        return self._name


#: Matches a key that is present and holds an explicit JSON ``null``.
JSON_NULL = _FilterSentinel("JSON_NULL")

#: Matches a key that is not present in the payload at all.
ABSENT = _FilterSentinel("ABSENT")
