"""Bounded, source-aware reads of the published KG projection.

The publisher owns constraints and writes. This module only reads one run_id.
"""
from __future__ import annotations

from collections.abc import Mapping
import time

RELATIONS = frozenset({"HAS_VERSION", "HAS_SOURCE", "HAS_ITEM", "RESOLVES_TO", "MEMBER_OF"})
LABELS = frozenset({"Paper", "PaperVersion", "Source", "SourceItem", "Contribution", "ResearchProblem", "Method", "Benchmark", "ResearchArea"})
MAX_IDS = 20
MAX_LIMIT = 50

# Explicit projections avoid retrieving PaperVersion.text or entire extraction payloads.
NODE_FIELDS = """n.node_id AS node_id, labels(n) AS labels,
 n.title AS title, n.name AS name, substring(coalesce(n.description,''),0,4000) AS description,
 n.kind AS kind, n.roles AS roles, n.source_url AS source_url,
 n.version_label AS version_label, n.name_origin AS name_origin"""
GET_NODE = f"MATCH (n:KGNode {{run_id:$run_id,node_id:$node_id}}) RETURN {NODE_FIELDS}"
EXPAND = f"""MATCH (n:KGNode {{run_id:$run_id,node_id:$node_id}})
 MATCH (n)PATTERN_PLACEHOLDER(m:KGNode {{run_id:$run_id}})
 RETURN type(r) AS relation, n.node_id AS node_id, m.node_id AS neighbor_id,
        labels(m) AS neighbor_labels, m.name AS neighbor_name, m.title AS neighbor_title,
        startNode(r).node_id AS edge_source_id, endNode(r).node_id AS edge_target_id
 LIMIT $limit"""
SHARED_PAPERS = """UNWIND $node_ids AS requested_id
 MATCH (entity:KGNode {run_id:$run_id,node_id:requested_id})
 MATCH (item:KGNode:SourceItem)-[:RESOLVES_TO]->(entity)
 MATCH (source:KGNode:Source)-[:HAS_ITEM]->(item)
 MATCH (version:KGNode:PaperVersion)-[:HAS_SOURCE]->(source)
 MATCH (paper:KGNode:Paper)-[:HAS_VERSION]->(version)
 WHERE item.run_id=$run_id AND source.run_id=$run_id AND version.run_id=$run_id AND paper.run_id=$run_id
 WITH paper,count(DISTINCT requested_id) AS matched WHERE matched=$target_count
 RETURN paper.node_id AS paper_id,paper.title AS title,matched
 ORDER BY paper_id LIMIT $limit"""
EVIDENCE_ENTITY = """UNWIND $node_ids AS requested_id
 MATCH (entity:KGNode {run_id:$run_id,node_id:requested_id})
 MATCH (item:KGNode:SourceItem)-[:RESOLVES_TO]->(entity)
 MATCH (source:KGNode:Source)-[:HAS_ITEM]->(item)
 MATCH (version:KGNode:PaperVersion)-[:HAS_SOURCE]->(source)
 MATCH (paper:KGNode:Paper)-[:HAS_VERSION]->(version)
 WHERE item.run_id=$run_id AND source.run_id=$run_id AND version.run_id=$run_id AND paper.run_id=$run_id
 RETURN requested_id,item.node_id AS item_id,item.kind AS kind,item.name AS name,
        item.roles AS roles,substring(coalesce(item.source_json,''),0,4000) AS source_json,
        source.node_id AS source_id,source.source_url AS source_url,
        version.node_id AS version_id,version.source_url AS version_url,
        paper.node_id AS paper_id,paper.title AS paper_title
 LIMIT $limit"""
EVIDENCE_ITEM = """UNWIND $node_ids AS requested_id
 MATCH (item:KGNode {run_id:$run_id,node_id:requested_id})
 MATCH (source:KGNode:Source)-[:HAS_ITEM]->(item)
 MATCH (version:KGNode:PaperVersion)-[:HAS_SOURCE]->(source)
 MATCH (paper:KGNode:Paper)-[:HAS_VERSION]->(version)
 WHERE source.run_id=$run_id AND version.run_id=$run_id AND paper.run_id=$run_id
 RETURN requested_id,item.node_id AS item_id,item.kind AS kind,item.name AS name,
        item.roles AS roles,substring(coalesce(item.source_json,''),0,4000) AS source_json,
        source.node_id AS source_id,source.source_url AS source_url,
        version.node_id AS version_id,version.source_url AS version_url,
        paper.node_id AS paper_id,paper.title AS paper_title
 LIMIT $limit"""


def _spec(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties,
                           "required": required, "additionalProperties": False}}}


ID = {"type": "string", "description": "Published node_id in the configured graph run"}
IDS = {"type": "array", "items": ID, "minItems": 1, "maxItems": MAX_IDS}
LIMIT = {"type": "integer", "minimum": 1, "maximum": MAX_LIMIT}
TOOL_SPECS = [
    _spec("get_node", "Get a published node by exact ID; long source text is omitted.", {"node_id": ID}, ["node_id"]),
    _spec("expand_neighbors", "Get directly connected nodes along published relation types only.",
          {"node_id": ID, "relation_types": {"type": "array", "items": {"type": "string", "enum": sorted(RELATIONS)}},
           "direction": {"type": "string", "enum": ["out", "in", "both"]}, "limit": LIMIT}, ["node_id"]),
    _spec("find_shared_papers", "Papers containing source mentions resolving to every supplied canonical entity; co-occurrence does not establish a scientific relation.",
          {"node_ids": IDS, "limit": LIMIT}, ["node_ids"]),
    _spec("get_evidence", "Source-item or contribution evidence and Paper→Version→Source lineage for exact IDs; roles and source attribution are preserved.",
          {"node_ids": IDS, "limit": LIMIT}, ["node_ids"]),
]


def _id(value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > 512:
        raise ValueError("node_id must be a nonempty string of at most 512 characters")
    return value


def _ids(value: object) -> list[str]:
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_IDS:
        raise ValueError(f"node_ids must contain 1 to {MAX_IDS} IDs")
    return list(dict.fromkeys(_id(item) for item in value))


def _limit(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_LIMIT:
        raise ValueError(f"limit must be an integer from 1 to {MAX_LIMIT}")
    return value


class GraphTools:
    """Official Neo4j driver read transactions, scoped to a configured run."""

    tools = TOOL_SPECS

    def __init__(self, driver, run_id: str, *, database: str = "neo4j") -> None:
        self.driver = driver
        self.run_id = _id(run_id)
        self.database = database

    async def _query(self, statement: str, parameters: dict, deadline: float) -> list[dict]:
        from neo4j import unit_of_work

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("agent time budget exhausted")

        @unit_of_work(timeout=min(30, remaining))
        async def read(tx):
            result = await tx.run(statement, **parameters)
            return [dict(record) async for record in result]

        async with self.driver.session(database=self.database) as session:
            return await session.execute_read(read)

    async def call(self, name: str, args: Mapping, *, deadline: float) -> object:
        if not isinstance(args, Mapping):
            raise ValueError("tool arguments must be an object")
        if name == "get_node":
            rows = await self._query(GET_NODE, {"run_id": self.run_id, "node_id": _id(args.get("node_id"))}, deadline)
            return rows[0] if rows else None
        if name == "expand_neighbors":
            node_id = _id(args.get("node_id"))
            relations = args.get("relation_types", sorted(RELATIONS))
            if not isinstance(relations, list) or not relations or any(r not in RELATIONS for r in relations):
                raise ValueError("relation_types must be a nonempty list of published relations")
            direction = args.get("direction", "both")
            if direction not in {"out", "in", "both"}:
                raise ValueError("direction must be out, in, or both")
            # Only validated, closed-set relation names and direction syntax enter Cypher.
            rel = "|".join(sorted(set(relations)))
            pattern = {"out": f"-[r:{rel}]->", "in": f"<-[r:{rel}]-", "both": f"-[r:{rel}]-"}[direction]
            statement = EXPAND.replace("PATTERN_PLACEHOLDER", pattern)
            rows = await self._query(statement,
                                     {"run_id": self.run_id, "node_id": node_id, "limit": _limit(args.get("limit", 20))}, deadline)
            return rows
        if name in {"find_shared_papers", "get_evidence"}:
            ids = _ids(args.get("node_ids"))
            limit = _limit(args.get("limit", 20))
            params = {"run_id": self.run_id, "node_ids": ids, "limit": limit}
            if name == "find_shared_papers":
                return await self._query(SHARED_PAPERS, {**params, "target_count": len(ids)}, deadline)
            entity_rows = await self._query(EVIDENCE_ENTITY, params, deadline)
            if len(entity_rows) >= limit:
                return entity_rows
            item_rows = await self._query(EVIDENCE_ITEM, {**params, "limit": limit - len(entity_rows)}, deadline)
            return entity_rows + item_rows
        raise ValueError(f"unknown graph tool: {name}")
