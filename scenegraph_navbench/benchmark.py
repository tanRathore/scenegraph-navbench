"""Benchmark loading and aggregate evaluation."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from scenegraph_navbench.evaluator import (
    EvaluationMetrics,
    evaluate_relations,
    load_expected_relations,
)
from scenegraph_navbench.graph import build_scene_graph
from scenegraph_navbench.models import Scene, SceneGraph
from scenegraph_navbench.spatial import SpatialConfig


@dataclass(frozen=True)
class SceneBundle:
    """A scene file and optional expected relation file loaded together."""

    scene_path: Path
    expected_path: Path | None
    scene: Scene


@dataclass(frozen=True)
class BenchmarkScene:
    """One scene entry from a benchmark manifest."""

    scene_path: Path
    expected_path: Path | None = None


@dataclass(frozen=True)
class BenchmarkManifest:
    """A benchmark manifest with resolved scene paths."""

    name: str
    scenes: list[BenchmarkScene]
    path: Path


@dataclass(frozen=True)
class SceneBenchmarkResult:
    """Per-scene graph construction and evaluation result."""

    scene_path: Path
    expected_path: Path | None
    graph: SceneGraph
    metrics: EvaluationMetrics | None


@dataclass(frozen=True)
class AggregateBenchmarkMetrics:
    """Aggregate metrics across a benchmark run."""

    must_include_scenes: int
    must_include_matched: int
    must_include_expected: int
    must_include_required_recall: float | None
    exact_scenes: int
    exact_true_positives: int
    exact_predicted: int
    exact_expected: int
    exact_precision: float | None
    exact_recall: float | None
    exact_f1: float | None


@dataclass(frozen=True)
class BenchmarkResult:
    """Complete benchmark run result."""

    name: str
    scenes: list[SceneBenchmarkResult]
    aggregate: AggregateBenchmarkMetrics
    relation_counts: dict[str, int]


def load_scene_bundle(
    scene_path: str | Path,
    expected_path: str | Path | None = None,
) -> SceneBundle:
    """Load one detection JSON scene and keep its expected path, if provided."""
    resolved_scene_path = Path(scene_path)
    scene = Scene.model_validate_json(resolved_scene_path.read_text(encoding="utf-8"))
    return SceneBundle(
        scene_path=resolved_scene_path,
        expected_path=Path(expected_path) if expected_path is not None else None,
        scene=scene,
    )


def load_benchmark_manifest(path: str | Path) -> BenchmarkManifest:
    """Load a benchmark manifest and resolve scene paths."""
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    name = str(payload.get("name", manifest_path.stem))
    scene_entries = payload.get("scenes")
    if not isinstance(scene_entries, list):
        raise ValueError("benchmark manifest must contain a 'scenes' list")

    scenes: list[BenchmarkScene] = []
    for index, entry in enumerate(scene_entries):
        if not isinstance(entry, dict):
            raise ValueError(f"scene entry at index {index} must be an object")
        if "scene" not in entry:
            raise ValueError(f"scene entry at index {index} is missing 'scene'")

        scene_path = _resolve_manifest_path(manifest_path, str(entry["scene"]))
        expected_value = entry.get("expected")
        expected_path = (
            _resolve_manifest_path(manifest_path, str(expected_value))
            if expected_value is not None
            else None
        )
        scenes.append(BenchmarkScene(scene_path=scene_path, expected_path=expected_path))

    return BenchmarkManifest(name=name, scenes=scenes, path=manifest_path)


def run_scene_evaluation(
    scene_path: str | Path,
    expected_path: str | Path | None = None,
    *,
    config: SpatialConfig | None = None,
    min_confidence: float | None = None,
) -> SceneBenchmarkResult:
    """Build and evaluate one scene."""
    bundle = load_scene_bundle(scene_path, expected_path)
    graph = build_scene_graph(
        bundle.scene,
        min_confidence=min_confidence,
        config=config,
    )
    metrics = None
    if bundle.expected_path is not None:
        expected = load_expected_relations(bundle.expected_path)
        metrics = evaluate_relations(graph.relations, expected)

    return SceneBenchmarkResult(
        scene_path=bundle.scene_path,
        expected_path=bundle.expected_path,
        graph=graph,
        metrics=metrics,
    )


def run_benchmark(
    manifest_path: str | Path,
    *,
    config: SpatialConfig | None = None,
    min_confidence: float | None = None,
) -> BenchmarkResult:
    """Run graph construction and evaluation for every scene in a manifest."""
    manifest = load_benchmark_manifest(manifest_path)
    scene_results: list[SceneBenchmarkResult] = []
    relation_counts: Counter[str] = Counter()

    for scene_entry in manifest.scenes:
        result = run_scene_evaluation(
            scene_entry.scene_path,
            scene_entry.expected_path,
            config=config,
            min_confidence=min_confidence,
        )
        scene_results.append(result)
        relation_counts.update(relation.relation for relation in result.graph.relations)

    return BenchmarkResult(
        name=manifest.name,
        scenes=scene_results,
        aggregate=_aggregate_metrics(scene_results),
        relation_counts=dict(sorted(relation_counts.items())),
    )


def _aggregate_metrics(
    scene_results: list[SceneBenchmarkResult],
) -> AggregateBenchmarkMetrics:
    metrics = [result.metrics for result in scene_results if result.metrics is not None]
    must_include = [item for item in metrics if item.mode == "must_include"]
    exact = [item for item in metrics if item.mode == "exact"]

    must_include_matched = sum(item.true_positives for item in must_include)
    must_include_expected = sum(item.expected for item in must_include)
    exact_true_positives = sum(item.true_positives for item in exact)
    exact_predicted = sum(item.predicted for item in exact)
    exact_expected = sum(item.expected for item in exact)

    must_include_required_recall = (
        must_include_matched / must_include_expected
        if must_include_expected
        else 1.0 if must_include else None
    )
    exact_precision = None
    exact_recall = None
    if exact:
        exact_precision = (
            exact_true_positives / exact_predicted
            if exact_predicted
            else 1.0 if exact_expected == 0 else 0.0
        )
        exact_recall = (
            exact_true_positives / exact_expected
            if exact_expected
            else 1.0 if exact_predicted == 0 else 0.0
        )
    exact_f1 = (
        2 * exact_precision * exact_recall / (exact_precision + exact_recall)
        if exact_precision is not None
        and exact_recall is not None
        and exact_precision + exact_recall > 0.0
        else None
    )

    return AggregateBenchmarkMetrics(
        must_include_scenes=len(must_include),
        must_include_matched=must_include_matched,
        must_include_expected=must_include_expected,
        must_include_required_recall=must_include_required_recall,
        exact_scenes=len(exact),
        exact_true_positives=exact_true_positives,
        exact_predicted=exact_predicted,
        exact_expected=exact_expected,
        exact_precision=exact_precision,
        exact_recall=exact_recall,
        exact_f1=exact_f1,
    )


def _resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path

    manifest_relative = manifest_path.parent / path
    if manifest_relative.exists():
        return manifest_relative

    cwd_relative = Path.cwd() / path
    if cwd_relative.exists():
        return cwd_relative

    return manifest_relative
