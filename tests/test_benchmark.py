from pathlib import Path

from scenegraph_navbench.benchmark import load_benchmark_manifest, run_benchmark
from scenegraph_navbench.graph import build_scene_graph
from scenegraph_navbench.models import Scene


MANIFEST_PATH = Path("samples/benchmark_manifest.json")


def test_load_benchmark_manifest_resolves_scene_paths() -> None:
    manifest = load_benchmark_manifest(MANIFEST_PATH)

    assert manifest.name == "SceneGraphNavBench synthetic v0 benchmark"
    assert len(manifest.scenes) == 5
    assert manifest.scenes[0].scene_path.exists()
    assert manifest.scenes[0].expected_path is not None
    assert manifest.scenes[0].expected_path.exists()


def test_run_benchmark_aggregates_must_include_metrics() -> None:
    result = run_benchmark(MANIFEST_PATH)

    assert len(result.scenes) == 5
    assert result.aggregate.must_include_required_recall == 1.0
    assert result.aggregate.must_include_matched == 41
    assert result.aggregate.must_include_expected == 41
    assert result.aggregate.exact_precision is None
    assert result.relation_counts["left_of"] > 0


def test_depth_missing_scene_does_not_hallucinate_depth_relations() -> None:
    scene = Scene.model_validate_json(
        Path("samples/depth_missing_scene.json").read_text(encoding="utf-8")
    )
    graph = build_scene_graph(scene)

    relation_names = {relation.relation for relation in graph.relations}

    assert "in_front_of" not in relation_names
    assert "behind" not in relation_names
