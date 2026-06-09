"""
Dual-adapter graph client.

Tries Neo4j first (if USE_NEO4J=true); falls back to NetworkX automatically.
Both adapters expose an identical interface so the rest of the system is unaware
of which backend is in use.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import networkx as nx
import structlog

from config import settings

log = structlog.get_logger()


# ─── Abstract Interface ──────────────────────────────────────────────────────


class GraphAdapter(ABC):
    @abstractmethod
    def upsert_node(self, label: str, node_id: str, properties: Dict[str, Any]) -> str: ...

    @abstractmethod
    def upsert_relationship(self, from_id: str, to_id: str, rel_type: str,
                             properties: Dict[str, Any] | None = None) -> None: ...

    @abstractmethod
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]: ...

    @abstractmethod
    def find_nodes(self, label: str, filters: Dict[str, Any] | None = None) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def get_neighbors(self, node_id: str, rel_type: str | None = None,
                      direction: str = "both") -> List[Dict[str, Any]]: ...

    @abstractmethod
    def get_all_nodes(self) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def get_all_relationships(self) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def shortest_path(self, from_id: str, to_id: str) -> List[str]: ...

    @abstractmethod
    def clear(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


# ─── NetworkX Adapter ────────────────────────────────────────────────────────


class NetworkXAdapter(GraphAdapter):
    """In-process graph backed by a pickle file for persistence across restarts."""

    def __init__(self):
        import os
        self._path = os.path.abspath(settings.GRAPH_PERSIST_PATH)
        self._g: nx.MultiDiGraph = self._load()
        log.info("graph_initialised_in_memory", nodes=self._g.number_of_nodes(),
                 persisted_to=self._path)

    def _load(self) -> nx.MultiDiGraph:
        import pickle, os
        try:
            with open(self._path, "rb") as f:
                g = pickle.load(f)
            log.info("graph_loaded_from_disk", nodes=g.number_of_nodes(),
                     relationships=g.number_of_edges())
            return g
        except FileNotFoundError:
            return nx.MultiDiGraph()
        except Exception as e:
            log.warning("graph_load_failed_starting_fresh", error=str(e))
            return nx.MultiDiGraph()

    def _save(self) -> None:
        import pickle, os, tempfile
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            tmp = self._path + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(self._g, f)
            os.replace(tmp, self._path)   # atomic on POSIX
        except Exception as e:
            log.warning("graph_save_failed", error=str(e))

    # ── write ────────────────────────────────────────────────────────────────

    def upsert_node(self, label: str, node_id: str, properties: Dict[str, Any]) -> str:
        props = {**properties, "label": label, "id": node_id}
        if self._g.has_node(node_id):
            self._g.nodes[node_id].update(props)
        else:
            self._g.add_node(node_id, **props)
        self._save()
        return node_id

    def upsert_relationship(self, from_id: str, to_id: str, rel_type: str,
                             properties: Dict[str, Any] | None = None) -> None:
        props = properties or {}
        self._g.add_edge(from_id, to_id, rel_type=rel_type, **props)
        self._save()

    # ── read ─────────────────────────────────────────────────────────────────

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        if self._g.has_node(node_id):
            return dict(self._g.nodes[node_id])
        return None

    def find_nodes(self, label: str, filters: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        results = []
        for nid, data in self._g.nodes(data=True):
            if data.get("label") != label:
                continue
            if filters:
                match = all(data.get(k) == v for k, v in filters.items())
                if not match:
                    continue
            results.append({"id": nid, **data})
        return results

    def get_neighbors(self, node_id: str, rel_type: str | None = None,
                      direction: str = "both") -> List[Dict[str, Any]]:
        neighbors = []
        if direction in ("out", "both"):
            for _, nbr, data in self._g.out_edges(node_id, data=True):
                if rel_type and data.get("rel_type") != rel_type:
                    continue
                nbr_data = dict(self._g.nodes.get(nbr, {}))
                neighbors.append({"id": nbr, "rel_type": data.get("rel_type"), **nbr_data})
        if direction in ("in", "both"):
            for src, _, data in self._g.in_edges(node_id, data=True):
                if rel_type and data.get("rel_type") != rel_type:
                    continue
                src_data = dict(self._g.nodes.get(src, {}))
                neighbors.append({"id": src, "rel_type": data.get("rel_type"), **src_data})
        return neighbors

    def get_all_nodes(self) -> List[Dict[str, Any]]:
        return [{"id": nid, **data} for nid, data in self._g.nodes(data=True)]

    def get_all_relationships(self) -> List[Dict[str, Any]]:
        rels = []
        for u, v, data in self._g.edges(data=True):
            rels.append({"from": u, "to": v, **data})
        return rels

    def shortest_path(self, from_id: str, to_id: str) -> List[str]:
        try:
            return nx.shortest_path(self._g, from_id, to_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def clear(self) -> None:
        import os
        self._g = nx.MultiDiGraph()
        self._save()
        try:
            os.remove(self._path)
        except FileNotFoundError:
            pass
        log.info("graph_cleared")

    def close(self) -> None:
        pass


# ─── Neo4j Adapter ───────────────────────────────────────────────────────────


class Neo4jAdapter(GraphAdapter):
    """Neo4j adapter – requires a running Neo4j instance."""

    def __init__(self):
        from neo4j import GraphDatabase
        self._driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )
        log.info("neo4j_connected", uri=settings.NEO4J_URI)

    def _run(self, cypher: str, **params) -> List[Dict[str, Any]]:
        with self._driver.session() as session:
            result = session.run(cypher, **params)
            return [dict(r) for r in result]

    def upsert_node(self, label: str, node_id: str, properties: Dict[str, Any]) -> str:
        props = {**properties, "id": node_id}
        cypher = f"MERGE (n:{label} {{id: $id}}) SET n += $props RETURN n.id AS id"
        self._run(cypher, id=node_id, props=props)
        return node_id

    def upsert_relationship(self, from_id: str, to_id: str, rel_type: str,
                             properties: Dict[str, Any] | None = None) -> None:
        props = properties or {}
        cypher = (
            f"MATCH (a {{id: $from_id}}), (b {{id: $to_id}}) "
            f"MERGE (a)-[r:{rel_type}]->(b) SET r += $props"
        )
        self._run(cypher, from_id=from_id, to_id=to_id, props=props)

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        rows = self._run("MATCH (n {id: $id}) RETURN properties(n) AS props", id=node_id)
        return rows[0]["props"] if rows else None

    def find_nodes(self, label: str, filters: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        where = ""
        if filters:
            conditions = " AND ".join(f"n.{k} = ${k}" for k in filters)
            where = f"WHERE {conditions}"
        cypher = f"MATCH (n:{label}) {where} RETURN properties(n) AS props"
        rows = self._run(cypher, **(filters or {}))
        return [r["props"] for r in rows]

    def get_neighbors(self, node_id: str, rel_type: str | None = None,
                      direction: str = "both") -> List[Dict[str, Any]]:
        rel = f":{rel_type}" if rel_type else ""
        if direction == "out":
            pattern = f"(a {{id: $id}})-[r{rel}]->(b)"
        elif direction == "in":
            pattern = f"(a {{id: $id}})<-[r{rel}]-(b)"
        else:
            pattern = f"(a {{id: $id}})-[r{rel}]-(b)"
        cypher = f"MATCH {pattern} RETURN properties(b) AS props, type(r) AS rel_type"
        rows = self._run(cypher, id=node_id)
        return [{"rel_type": r["rel_type"], **r["props"]} for r in rows]

    def get_all_nodes(self) -> List[Dict[str, Any]]:
        rows = self._run("MATCH (n) RETURN properties(n) AS props")
        return [r["props"] for r in rows]

    def get_all_relationships(self) -> List[Dict[str, Any]]:
        rows = self._run("MATCH (a)-[r]->(b) RETURN a.id AS from, type(r) AS rel_type, b.id AS to")
        return rows

    def shortest_path(self, from_id: str, to_id: str) -> List[str]:
        cypher = (
            "MATCH p=shortestPath((a {id: $from_id})-[*]-(b {id: $to_id})) "
            "RETURN [n IN nodes(p) | n.id] AS path"
        )
        rows = self._run(cypher, from_id=from_id, to_id=to_id)
        return rows[0]["path"] if rows else []

    def clear(self) -> None:
        self._run("MATCH (n) DETACH DELETE n")
        log.info("graph_cleared")

    def close(self) -> None:
        self._driver.close()


# ─── Factory ─────────────────────────────────────────────────────────────────


def create_graph_adapter() -> GraphAdapter:
    if settings.USE_NEO4J:
        try:
            adapter = Neo4jAdapter()
            return adapter
        except Exception as e:
            log.warning("neo4j_unavailable_falling_back", error=str(e))
    return NetworkXAdapter()


_graphs: Dict[str, GraphAdapter] = {}


def get_graph(project_id: str = "default") -> GraphAdapter:
    if project_id not in _graphs:
        if project_id == "default":
            _graphs[project_id] = create_graph_adapter()
        else:
            # Project-specific graph uses a separate pkl path
            adapter = NetworkXAdapter.__new__(NetworkXAdapter)
            import os
            graph_dir = os.path.dirname(os.path.abspath(settings.GRAPH_PERSIST_PATH))
            adapter._path = os.path.join(graph_dir, f"project_{project_id}.pkl")
            adapter._g = adapter._load()
            _graphs[project_id] = adapter
    return _graphs[project_id]
