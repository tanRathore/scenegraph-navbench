"""Command line interface for SceneGraphNavBench."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Sequence

from scenegraph_navbench.benchmark import BenchmarkResult, run_benchmark
from scenegraph_navbench.evaluator import EvaluationMetrics
from scenegraph_navbench.evaluator import evaluate_relations, load_expected_relations
from scenegraph_navbench.export import write_scene_graph_dot, write_scene_graph_json
from scenegraph_navbench.graph import build_scene_graph
from scenegraph_navbench.models import Relation, Scene


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)

    if args.benchmark is not None:
        result = run_benchmark(
            args.benchmark,
            min_confidence=args.min_confidence,
        )
        print_benchmark_result(result)
        return

    _run_scene(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and optionally evaluate a deterministic scene graph."
    )
    parser.add_argument(
        "scene",
        nargs="?",
        type=Path,
        help="Path to a scene JSON file.",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        help="Run a benchmark manifest instead of a single scene.",
    )
    parser.add_argument(
        "--expected",
        type=Path,
        help="Optional expected relation JSON file.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.25,
        help="Minimum detection confidence to include in the graph.",
    )
    parser.add_argument(
        "--relation-limit",
        type=int,
        default=12,
        help="Number of generated relation examples to display.",
    )
    parser.add_argument(
        "--export-json",
        type=Path,
        help="Write the single-scene graph as JSON.",
    )
    parser.add_argument(
        "--export-dot",
        type=Path,
        help="Write the single-scene graph as Graphviz DOT.",
    )
    return parser


def _validate_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if args.benchmark is not None:
        if args.scene is not None:
            parser.error("provide either a scene path or --benchmark, not both")
        if args.expected is not None:
            parser.error("--expected is only valid for single-scene mode")
        if args.export_json is not None or args.export_dot is not None:
            parser.error("export flags are only valid for single-scene mode")

    elif args.scene is None:
        parser.error("provide a scene path or --benchmark")


def _run_scene(args: argparse.Namespace) -> None:
    scene = Scene.model_validate_json(args.scene.read_text(encoding="utf-8"))
    graph = build_scene_graph(scene, min_confidence=args.min_confidence)

    print("SceneGraphNavBench")
    print(
        f"Loaded scene: {scene.image_width}x{scene.image_height} "
        f"with {len(scene.detections)} detections"
    )
    print(
        f"Graph: {len(graph.nodes)} nodes, "
        f"{len(graph.relations)} deterministic relations"
    )
    print()

    print("Objects")
    for node in graph.nodes:
        depth = f"{node.depth:.2f}" if node.depth is not None else "unknown"
        bbox = ", ".join(f"{value:.1f}" for value in node.bbox)
        print(
            f"- {node.id}: {node.label} bbox=({bbox}) "
            f"confidence={node.confidence:.2f} depth={depth}"
        )
    print()

    print_relation_summary(graph.relations, relation_limit=args.relation_limit)

    if args.expected:
        expected = load_expected_relations(args.expected)
        metrics = evaluate_relations(graph.relations, expected)
        print()
        print("Evaluation")
        print_evaluation(metrics)

    if args.export_json:
        write_scene_graph_json(graph, args.export_json)
        print()
        print(f"Exported JSON: {args.export_json}")
    if args.export_dot:
        write_scene_graph_dot(graph, args.export_dot)
        print()
        print(f"Exported DOT: {args.export_dot}")


def print_relation_summary(
    relations: list[Relation],
    *,
    relation_limit: int,
) -> None:
    print("Generated relation counts")
    counts = Counter(relation.relation for relation in relations)
    for relation, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"- {relation}: {count}")

    print()
    sample_size = max(0, min(relation_limit, len(relations)))
    print(f"Sample generated relations ({sample_size} of {len(relations)})")
    for relation in relations[:sample_size]:
        print(
            f"- {relation.source} {relation.relation} {relation.target} "
            f"(score={relation.score:.3f})"
        )


def print_evaluation(metrics: EvaluationMetrics) -> None:
    if metrics.mode == "must_include":
        print(
            "- mode: must_include "
            "(sparse required triples; extra generated relations are not "
            "counted as false positives)"
        )
        print(
            "- required relation recall: "
            f"{metrics.required_relation_recall:.3f}"
        )
        print(
            "- required triples matched: "
            f"{metrics.true_positives}/{metrics.expected}"
        )
        print(f"- generated triples checked: {metrics.predicted}")
    else:
        print("- mode: exact (expected file is an exhaustive gold set)")
        print(f"- precision: {metrics.precision:.3f}")
        print(f"- recall: {metrics.recall:.3f}")
        print(f"- f1: {metrics.f1:.3f}")
        print(f"- true positives: {metrics.true_positives}")
        print(f"- predicted triples: {metrics.predicted}")
        print(f"- expected triples: {metrics.expected}")


def print_benchmark_result(result: BenchmarkResult) -> None:
    print("SceneGraphNavBench Benchmark")
    print(f"Benchmark: {result.name}")
    print(f"Scenes: {len(result.scenes)}")
    print()

    print("Per-scene results")
    for scene_result in result.scenes:
        print(
            f"- {_display_path(scene_result.scene_path)}: "
            f"nodes={len(scene_result.graph.nodes)} "
            f"relations={len(scene_result.graph.relations)}"
        )
        if scene_result.metrics is None:
            print("  evaluation: none")
        elif scene_result.metrics.mode == "must_include":
            print(
                "  required relation recall="
                f"{scene_result.metrics.required_relation_recall:.3f} "
                f"({scene_result.metrics.true_positives}/"
                f"{scene_result.metrics.expected})"
            )
        else:
            print(
                "  exact precision/recall/f1="
                f"{scene_result.metrics.precision:.3f}/"
                f"{scene_result.metrics.recall:.3f}/"
                f"{scene_result.metrics.f1:.3f}"
            )

    print()
    print("Aggregate evaluation")
    aggregate = result.aggregate
    if aggregate.must_include_required_recall is None:
        print("- must_include scenes: none")
    else:
        print(
            "- must_include required relation recall: "
            f"{aggregate.must_include_required_recall:.3f}"
        )
        print(
            "- must_include required triples matched: "
            f"{aggregate.must_include_matched}/{aggregate.must_include_expected}"
        )

    if aggregate.exact_precision is None:
        print("- exact scenes: none")
    else:
        print(f"- exact precision: {aggregate.exact_precision:.3f}")
        print(f"- exact recall: {aggregate.exact_recall:.3f}")
        print(f"- exact f1: {aggregate.exact_f1:.3f}")

    print()
    print("Benchmark relation counts")
    for relation, count in sorted(
        result.relation_counts.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        print(f"- {relation}: {count}")


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
