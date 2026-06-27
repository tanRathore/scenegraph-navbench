"""Export helpers for scene graphs."""

from __future__ import annotations

import json
from pathlib import Path

from scenegraph_navbench.models import Relation, SceneGraph


def scene_graph_to_dict(graph: SceneGraph) -> dict[str, object]:
    """Return a JSON-serializable scene graph dictionary."""
    return {
        "nodes": [
            {
                "id": node.id,
                "label": node.label,
                "bbox": list(node.bbox),
                "confidence": node.confidence,
                "depth": node.depth,
            }
            for node in graph.nodes
        ],
        "relations": [relation.model_dump() for relation in graph.relations],
    }


def relation_triples_to_dicts(relations: list[Relation]) -> list[dict[str, str]]:
    """Return source/relation/target triples as dictionaries."""
    return [
        {
            "source": relation.source,
            "relation": relation.relation,
            "target": relation.target,
        }
        for relation in relations
    ]


def write_scene_graph_json(graph: SceneGraph, path: str | Path) -> None:
    """Write a scene graph export as formatted JSON."""
    Path(path).write_text(
        json.dumps(scene_graph_to_dict(graph), indent=2) + "\n",
        encoding="utf-8",
    )


def scene_graph_to_dot(graph: SceneGraph) -> str:
    """Return a Graphviz DOT representation without external dependencies."""
    lines = [
        "digraph SceneGraph {",
        "  graph [rankdir=LR];",
        '  node [shape=box, style="rounded"];',
    ]

    for node in graph.nodes:
        label = f"{node.id}\\n{node.label}"
        lines.append(f'  "{_escape(node.id)}" [label="{_escape(label)}"];')

    for relation in graph.relations:
        label = f"{relation.relation}\\nscore={relation.score:.3f}"
        lines.append(
            f'  "{_escape(relation.source)}" -> "{_escape(relation.target)}" '
            f'[label="{_escape(label)}"];'
        )

    lines.append("}")
    return "\n".join(lines) + "\n"


def write_scene_graph_dot(graph: SceneGraph, path: str | Path) -> None:
    """Write a scene graph export as Graphviz DOT text."""
    Path(path).write_text(scene_graph_to_dot(graph), encoding="utf-8")


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
