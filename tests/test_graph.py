from scenegraph_navbench.graph import build_scene_graph
from scenegraph_navbench.models import Detection, Scene


def make_scene() -> Scene:
    return Scene(
        image_width=1000,
        image_height=1000,
        detections=[
            Detection(
                id="chair_1",
                label="chair",
                bbox=(100, 500, 240, 820),
                confidence=0.95,
                depth=2.0,
            ),
            Detection(
                id="table_1",
                label="table",
                bbox=(320, 460, 620, 760),
                confidence=0.85,
                depth=2.5,
            ),
            Detection(
                id="ghost_1",
                label="chair",
                bbox=(700, 100, 780, 220),
                confidence=0.1,
                depth=5.0,
            ),
        ],
    )


def test_graph_construction_includes_high_confidence_detections() -> None:
    graph = build_scene_graph(make_scene(), min_confidence=0.25)

    assert {node.id for node in graph.nodes} == {"chair_1", "table_1"}


def test_low_confidence_detections_are_filtered_out() -> None:
    graph = build_scene_graph(make_scene(), min_confidence=0.25)

    assert graph.get_node("ghost_1") is None
    assert all(
        relation.source != "ghost_1" and relation.target != "ghost_1"
        for relation in graph.relations
    )


def test_graph_does_not_create_self_relations() -> None:
    graph = build_scene_graph(make_scene(), min_confidence=0.25)

    assert graph.relations
    assert all(relation.source != relation.target for relation in graph.relations)


def test_graph_helper_methods_work() -> None:
    graph = build_scene_graph(make_scene(), min_confidence=0.25)

    assert graph.get_node("chair_1") is not None
    assert graph.get_node("missing") is None
    assert [node.id for node in graph.find_by_label("chair")] == ["chair_1"]
    assert graph.relations_for("chair_1")
    assert all(
        relation.source == "chair_1" or relation.target == "chair_1"
        for relation in graph.relations_for("chair_1")
    )
