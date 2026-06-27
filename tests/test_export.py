from scenegraph_navbench.export import scene_graph_to_dict, scene_graph_to_dot
from scenegraph_navbench.graph import build_scene_graph
from scenegraph_navbench.models import Detection, Scene, SceneGraph


def make_graph() -> SceneGraph:
    scene = Scene(
        image_width=800,
        image_height=600,
        detections=[
            Detection(
                id="chair_1",
                label="chair",
                bbox=(100, 300, 260, 560),
                confidence=0.95,
                depth=2.0,
            ),
            Detection(
                id="door_1",
                label="door",
                bbox=(560, 120, 760, 580),
                confidence=0.98,
                depth=4.0,
            ),
        ],
    )
    return build_scene_graph(scene)


def test_scene_graph_to_dict_is_json_serializable_shape() -> None:
    graph_dict = scene_graph_to_dict(make_graph())

    assert graph_dict["nodes"][0]["id"] == "chair_1"
    assert graph_dict["nodes"][0]["bbox"] == [100.0, 300.0, 260.0, 560.0]
    assert graph_dict["relations"]
    assert {"source", "target", "relation", "score", "reason"} <= set(
        graph_dict["relations"][0]
    )


def test_scene_graph_to_dot_contains_nodes_and_edges() -> None:
    dot = scene_graph_to_dot(make_graph())

    assert dot.startswith("digraph SceneGraph")
    assert '"chair_1"' in dot
    assert '"door_1"' in dot
    assert "left_of" in dot
    assert "score=" in dot
