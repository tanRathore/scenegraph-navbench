"""Command line interface for SceneGraphNavBench."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from scenegraph_navbench.benchmark import BenchmarkResult, run_benchmark
from scenegraph_navbench.evaluator import EvaluationMetrics
from scenegraph_navbench.evaluator import evaluate_relations, load_expected_relations
from scenegraph_navbench.export import write_scene_graph_dot, write_scene_graph_json
from scenegraph_navbench.graph import build_scene_graph
from scenegraph_navbench.models import Relation, Scene
from scenegraph_navbench.query_runtime import (
    QueryStatus,
    RobotQueryResult,
    RobotQueryRuntime,
)
from scenegraph_navbench.robot_context import NavigationContext


@dataclass(frozen=True)
class AgentDemoQuestion:
    """One CLI demo question and its deterministic runtime result."""

    name: str
    question: str
    result: RobotQueryResult | None
    skip_reason: str | None = None


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
    parser.add_argument(
        "--navigation",
        type=Path,
        help="World-frame navigation context JSON for the robot demo.",
    )
    parser.add_argument(
        "--show-agent-demo",
        action="store_true",
        help="Run and display deterministic robot queries.",
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
        if args.navigation is not None or args.show_agent_demo:
            parser.error("robot demo flags are only valid for single-scene mode")

    elif args.scene is None:
        parser.error("provide a scene path or --benchmark")
    elif args.show_agent_demo and args.navigation is None:
        parser.error("--show-agent-demo requires --navigation")
    elif args.navigation is not None and not args.show_agent_demo:
        parser.error("--navigation requires --show-agent-demo")


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

    if args.show_agent_demo:
        navigation_context = NavigationContext.model_validate_json(
            args.navigation.read_text(encoding="utf-8")
        )
        runtime = RobotQueryRuntime(graph, navigation_context)
        print()
        print_agent_demo(navigation_context, runtime)


def print_agent_demo(
    context: NavigationContext,
    runtime: RobotQueryRuntime,
) -> None:
    """Print robot context, fixed demo questions, answers, and evidence."""
    print("Robot context")
    print(f"- frame: {context.frame_id}")
    print(
        "- pose: "
        f"x={context.robot_pose.x:.3f}, "
        f"y={context.robot_pose.y:.3f}, "
        f"heading_radians={context.robot_pose.heading_radians:.3f}"
    )
    print(f"- robot radius: {context.robot_radius:.3f}")
    print(f"- navigation objects: {len(context.objects)}")
    for item in context.objects:
        position = (
            f"({item.x:.3f}, {item.y:.3f})"
            if item.x is not None and item.y is not None
            else "unknown"
        )
        print(
            f"  - {item.object_id}: position={position}, "
            f"radius={item.radius:.3f}, role={item.role or 'none'}, "
            f"obstacle={'yes' if item.is_obstacle else 'no'}"
        )

    questions = _agent_demo_questions(runtime)

    print()
    print("Questions")
    for index, item in enumerate(questions, start=1):
        print(f"{index}. {item.question}")

    print()
    print("Deterministic answers")
    for index, item in enumerate(questions, start=1):
        if item.result is None:
            print(f"{index}. {item.name}: not evaluated")
            print(f"   reason: {item.skip_reason}")
            continue

        object_ids = ", ".join(item.result.object_ids) or "none"
        print(f"{index}. {item.name}: {item.result.status.value}")
        print(f"   objects: {object_ids}")
        print(f"   reason: {item.result.reason}")

    print()
    print("Evidence traces")
    for index, item in enumerate(questions, start=1):
        print(f"{index}. {item.name}")
        if item.result is None:
            print(f"   - not evaluated: {item.skip_reason}")
            continue
        if not item.result.evidence:
            print("   - no evidence records")
            continue

        for evidence in item.result.evidence:
            measurements = ", ".join(
                f"{name}={_format_measurement(value)}"
                for name, value in evidence.measurements.items()
            )
            suffix = f" [{measurements}]" if measurements else ""
            print(
                f"   - {evidence.object_id}: {evidence.reason}{suffix}"
            )


def _agent_demo_questions(runtime: RobotQueryRuntime) -> list[AgentDemoQuestion]:
    exit_result = runtime.exit_target()
    questions = [
        AgentDemoQuestion(
            name="closest_object",
            question="What is the closest object to the robot?",
            result=runtime.closest_object(),
        ),
        AgentDemoQuestion(
            name="object_in_front",
            question="What object is in front of the robot?",
            result=runtime.object_in_front(),
        ),
        AgentDemoQuestion(
            name="exit_target",
            question="Which object should the robot move toward to exit?",
            result=exit_result,
        ),
    ]

    if exit_result.status == QueryStatus.OK and len(exit_result.object_ids) == 1:
        exit_id = exit_result.object_ids[0]
        near_result = runtime.objects_near(exit_id)
        blocker_result = runtime.blockers_for(exit_id)
        skip_reason = None
    else:
        near_result = None
        blocker_result = None
        skip_reason = "A single exit target was not resolved."

    questions.extend(
        [
            AgentDemoQuestion(
                name="objects_near_exit",
                question="Which objects are near the exit target?",
                result=near_result,
                skip_reason=skip_reason,
            ),
            AgentDemoQuestion(
                name="blockers_for_exit",
                question="Does anything block the corridor to the exit target?",
                result=blocker_result,
                skip_reason=skip_reason,
            ),
            AgentDemoQuestion(
                name="object_to_avoid",
                question="Which object in front should the robot avoid?",
                result=runtime.object_to_avoid(),
            ),
        ]
    )
    return questions


def _format_measurement(value: float | int | str | bool) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


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
