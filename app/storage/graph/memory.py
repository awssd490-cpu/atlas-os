"""In-memory graph store for testing and development.

Implements the ``GraphStore`` interface with adjacency-list-like
structures.  Suitable for unit tests and small-scale development.
"""

from __future__ import annotations

from typing import Any

from app.storage.interfaces import GraphNode, GraphRelationship, GraphStore


class InMemoryGraphStore(GraphStore):
    """Graph store backed by in-memory dicts.

    Nodes are stored by ID.  Relationships are stored in adjacency lists
    keyed by source and target node IDs.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, GraphNode] = {}
        self._relationships: dict[str, GraphRelationship] = {}
        # adjacency: source_id -> target_id -> [rel_ids]
        self._adj_out: dict[str, dict[str, list[str]]] = {}
        # reverse adjacency: target_id -> source_id -> [rel_ids]
        self._adj_in: dict[str, dict[str, list[str]]] = {}
        self._next_id = 0

    async def create_node(self, node: GraphNode) -> GraphNode:
        """Create a node.  Raises ``ValueError`` if the ID exists."""
        if node.id in self._nodes:
            raise ValueError(f"Node '{node.id}' already exists")
        self._nodes[node.id] = node
        return node

    async def get_node(self, id: str) -> GraphNode | None:
        """Retrieve a node by ID."""
        return self._nodes.get(id)

    async def update_node(self, node: GraphNode) -> GraphNode:
        """Replace properties and labels on an existing node."""
        if node.id not in self._nodes:
            raise ValueError(f"Node '{node.id}' not found")
        self._nodes[node.id] = node
        return node

    async def delete_node(self, id: str) -> None:
        """Delete a node and all its relationships."""
        self._nodes.pop(id, None)
        # Remove all relationships involving this node
        rel_ids: list[str] = []
        if id in self._adj_out:
            for target in self._adj_out[id]:
                rel_ids.extend(self._adj_out[id][target])
        if id in self._adj_in:
            for source in self._adj_in[id]:
                rel_ids.extend(self._adj_in[id][source])
        for rid in rel_ids:
            self._relationships.pop(rid, None)
        self._adj_out.pop(id, None)
        self._adj_in.pop(id, None)

    async def find_nodes(
        self,
        *,
        labels: list[str] | None = None,
        properties: dict[str, Any] | None = None,
    ) -> list[GraphNode]:
        """Find nodes by label and/or property filter."""
        results: list[GraphNode] = []
        for node in self._nodes.values():
            if labels and not any(lbl in node.labels for lbl in labels):
                continue
            if properties:
                if not all(node.properties.get(k) == v for k, v in properties.items()):
                    continue
            results.append(node)
        return results

    async def create_relationship(self, rel: GraphRelationship) -> GraphRelationship:
        """Create a relationship between two nodes."""
        rel_id = rel.id or f"rel_{self._next_id}"
        self._next_id += 1
        rel.id = rel_id
        self._relationships[rel_id] = rel

        # Build forward adjacency
        self._adj_out.setdefault(rel.source_id, {})
        self._adj_out[rel.source_id].setdefault(rel.target_id, []).append(rel_id)

        # Build reverse adjacency
        self._adj_in.setdefault(rel.target_id, {})
        self._adj_in[rel.target_id].setdefault(rel.source_id, []).append(rel_id)

        return rel

    async def get_relationships(
        self,
        *,
        source_id: str | None = None,
        target_id: str | None = None,
        type: str | None = None,
    ) -> list[GraphRelationship]:
        """Query relationships by source, target, and/or type."""
        results: list[GraphRelationship] = []
        for rel in self._relationships.values():
            if source_id is not None and rel.source_id != source_id:
                continue
            if target_id is not None and rel.target_id != target_id:
                continue
            if type is not None and rel.type != type:
                continue
            results.append(rel)
        return results

    async def traverse(
        self,
        start_id: str,
        *,
        direction: str = "outgoing",
        max_depth: int = 3,
        relationship_types: list[str] | None = None,
    ) -> list[GraphNode]:
        """BFS traversal of the graph starting from *start_id*."""
        visited: set[str] = set()
        queue: list[tuple[str, int]] = [(start_id, 0)]
        result: list[GraphNode] = []

        while queue:
            current_id, depth = queue.pop(0)
            if current_id in visited or depth > max_depth:
                continue
            visited.add(current_id)
            if current_id != start_id:
                node = self._nodes.get(current_id)
                if node:
                    result.append(node)

            if depth == max_depth:
                continue

            if direction in ("outgoing", "both"):
                adj = self._adj_out.get(current_id, {})
                for target_id in adj:
                    # Check relationship_types filter
                    if relationship_types:
                        for rid in adj[target_id]:
                            rel = self._relationships.get(rid)
                            if rel and rel.type not in relationship_types:
                                continue
                    queue.append((target_id, depth + 1))

            if direction in ("incoming", "both"):
                adj = self._adj_in.get(current_id, {})
                for source_id in adj:
                    if relationship_types:
                        for rid in adj[source_id]:
                            rel = self._relationships.get(rid)
                            if rel and rel.type not in relationship_types:
                                continue
                    queue.append((source_id, depth + 1))

        return result
