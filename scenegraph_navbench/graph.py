"""Scene graph construction."""

from __future__ import annotations

from itertools import permutations

from scenegraph_navbench.models import Scene, SceneGraph
from scenegraph_navbench.spatial import SpatialConfig, infer_pairwise_relations


def build_scene_graph(
    scene: Scene,
    min_confidence: float | None = None,
    *,
    config: SpatialConfig | None = None,
) -> SceneGraph:
    """Build a deterministic scene graph from a validated scene."""
    config = config or SpatialConfig()
    confidence_threshold = (
        config.min_confidence if min_confidence is None else min_confidence
    )
    nodes = [
        detection
        for detection in scene.detections
        if detection.confidence >= confidence_threshold
    ]

    relations = []
    for source, target in permutations(nodes, 2):
        relations.extend(
            infer_pairwise_relations(
                source,
                target,
                scene.image_width,
                scene.image_height,
                config=config,
            )
        )

    return SceneGraph(nodes=nodes, relations=relations)
