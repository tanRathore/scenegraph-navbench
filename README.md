# SceneGraphNavBench

SceneGraphNavBench is a computer vision and robotics benchmark foundation for converting object detection outputs into deterministic navigable scene graphs. It operates on detection JSON, not raw images. Each JSON scene simulates the downstream output of a perception model: object ids, labels, bounding boxes, confidence scores, and optional depth estimates.

The benchmark tests spatial reasoning after perception. Given detections, it produces graph nodes for objects and directed spatial relations such as `left_of`, `near`, `overlaps`, `in_front_of`, and `behind`. Each relation includes a score and deterministic reason so later robot-agent queries can cite supporting evidence instead of inventing spatial claims.

The v0 scope intentionally stops before the robot query runtime. There is no robot context, planner, agent loop, or model adapter yet. Those are meant to build on this baseline in a later feature session.

## Why Deterministic Scene Graphs

Embodied AI systems often need to answer spatial questions such as "what is left of the door?" or "which object is closer to the robot?" If those answers are generated directly by a language model, small changes in wording or context can produce inconsistent spatial claims.

SceneGraphNavBench makes spatial reasoning explicit. Geometry rules derive relations from centers, normalized distances, overlap, and depth. This reduces hallucinated spatial reasoning by keeping graph facts reproducible and inspectable.

## Concepts

- Nodes are validated object detections that pass the configured confidence threshold.
- Relations are directed spatial facts between ordered object pairs.
- Reasons explain the deterministic measurement behind a relation, such as center coordinates, normalized distance thresholds, IoU, or depth comparison.
- Expected files can be exhaustive gold graphs or sparse must-include checklists.
- Benchmark manifests group multiple scene fixtures into a repeatable synthetic benchmark run.

## Repository Architecture

- `scenegraph_navbench/models.py` defines validated Pydantic models for detections, scenes, relations, and scene graphs.
- `scenegraph_navbench/spatial.py` contains geometry helpers, `SpatialConfig`, and pairwise relation inference.
- `scenegraph_navbench/graph.py` filters detections and builds scene graphs across all ordered object pairs.
- `scenegraph_navbench/evaluator.py` loads expected triples and computes mode-aware evaluation metrics.
- `scenegraph_navbench/benchmark.py` loads benchmark manifests and aggregates per-scene results.
- `scenegraph_navbench/export.py` exports scene graphs as JSON dictionaries or Graphviz DOT text.
- `scenegraph_navbench/cli.py` provides single-scene and benchmark commands.
- `samples/` contains synthetic detection-output fixtures and a benchmark manifest.
- `tests/` covers spatial inference, graph construction, evaluation, benchmark loading, exports, and CLI benchmark mode.

## Installation

```bash
python -m pip install -e ".[dev]"
```

## Running Tests

```bash
python -m pytest -q
```

## Single-Scene CLI

```bash
python -m scenegraph_navbench.cli samples/room_scene.json --expected samples/room_scene_expected.json
```

Export a graph for inspection or visualization:

```bash
python -m scenegraph_navbench.cli samples/room_scene.json \
  --expected samples/room_scene_expected.json \
  --export-json /tmp/scenegraph.json \
  --export-dot /tmp/scenegraph.dot
```

## Benchmark CLI

```bash
python -m scenegraph_navbench.cli --benchmark samples/benchmark_manifest.json
```

The manifest groups the synthetic scenes:

```json
{
  "name": "SceneGraphNavBench synthetic v0 benchmark",
  "scenes": [
    {
      "scene": "samples/room_scene.json",
      "expected": "samples/room_scene_expected.json"
    }
  ]
}
```

Benchmark output includes the benchmark name, scene count, per-scene node and relation counts, per-scene evaluation summaries, aggregate required-relation recall for `must_include` scenes, aggregate exact precision/recall/F1 if any `exact` scenes exist, and relation counts by type across the benchmark.

## Evaluation Modes

Expected relation files can describe either an exhaustive gold graph or a sparse set of required relations:

```json
{
  "mode": "must_include",
  "relations": [
    {"source": "chair_1", "relation": "left_of", "target": "door_1"}
  ]
}
```

- `exact` means the expected file is an exhaustive gold relation set. The evaluator reports precision, recall, and F1.
- `must_include` means the expected file is a sparse checklist of relations that should appear in the generated graph. The evaluator reports required relation recall and does not count extra deterministic relations as false positives.

The sample benchmark uses `must_include` because v0 deliberately generates broad deterministic graphs while the sample expected files only name reviewer-friendly required subsets.

## Example Single-Scene Output

```text
SceneGraphNavBench
Loaded scene: 1280x720 with 6 detections
Graph: 6 nodes, 106 deterministic relations

Generated relation counts
- above: 15
- behind: 15
- below: 15
- in_front_of: 15
- left_of: 15
- right_of: 15
- near: 12
- far: 2
- overlaps: 2

Evaluation
- mode: must_include (sparse required triples; extra generated relations are not counted as false positives)
- required relation recall: 1.000
- required triples matched: 12/12
- generated triples checked: 106
```

## Example Benchmark Output

```text
SceneGraphNavBench Benchmark
Benchmark: SceneGraphNavBench synthetic v0 benchmark
Scenes: 5

Aggregate evaluation
- must_include required relation recall: 1.000
- must_include required triples matched: 41/41
- exact scenes: none
```

## Configurable Thresholds

`SpatialConfig` controls deterministic inference thresholds:

- `min_confidence`
- `near_threshold`
- `far_threshold`
- `overlap_threshold`
- `depth_epsilon`
- `horizontal_margin`
- `vertical_margin`

Defaults preserve the v0 demo behavior. Tests cover threshold changes so later benchmark versions can tune relation density deliberately.

## Limitations And Next Steps

This v0 baseline uses deterministic image-space geometry over synthetic detection JSON. It does not yet model camera intrinsics, free-space topology, support surfaces, robot pose, object affordances, occlusion, temporal consistency, or natural-language robot queries. The next layer should add the robot query runtime and richer task-level evaluation on top of this benchmark foundation.
