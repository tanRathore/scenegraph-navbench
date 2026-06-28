import math
from pathlib import Path

from scenegraph_navbench.graph import build_scene_graph
from scenegraph_navbench.models import Detection, Scene, SceneGraph
from scenegraph_navbench.query_runtime import RobotQueryConfig, RobotQueryRuntime
from scenegraph_navbench.robot_context import NavigationContext


def make_graph(
    *detections: Detection,
    image_width: int = 1000,
    image_height: int = 1000,
) -> SceneGraph:
    return build_scene_graph(
        Scene(
            image_width=image_width,
            image_height=image_height,
            detections=list(detections),
        )
    )


def detection(
    object_id: str,
    label: str,
    bbox: tuple[float, float, float, float],
    *,
    confidence: float = 0.9,
) -> Detection:
    return Detection(
        id=object_id,
        label=label,
        bbox=bbox,
        confidence=confidence,
    )


def navigation_context(
    objects: list[dict[str, object]],
    *,
    robot_x: float = 0.0,
    robot_y: float = 0.0,
    heading_radians: float = 0.0,
    robot_radius: float = 0.25,
) -> NavigationContext:
    return NavigationContext.model_validate(
        {
            "frame_id": "map",
            "robot_pose": {
                "x": robot_x,
                "y": robot_y,
                "heading_radians": heading_radians,
            },
            "robot_radius": robot_radius,
            "objects": objects,
        }
    )


def nav_object(
    object_id: str,
    *,
    x: float | None,
    y: float | None,
    radius: float = 0.25,
    role: str | None = None,
    is_obstacle: bool = False,
) -> dict[str, object]:
    return {
        "object_id": object_id,
        "x": x,
        "y": y,
        "radius": radius,
        "role": role,
        "is_obstacle": is_obstacle,
    }


def test_closest_object_uses_robot_world_position_not_image_geometry() -> None:
    graph = make_graph(
        detection("image_center_object", "chair", (450, 450, 550, 550)),
        detection("world_near_object", "cup", (0, 0, 20, 20)),
        detection("large_bbox_object", "table", (100, 100, 900, 900)),
    )
    context = navigation_context(
        [
            nav_object("image_center_object", x=30.0, y=30.0),
            nav_object("world_near_object", x=10.5, y=10.0),
            nav_object("large_bbox_object", x=25.0, y=25.0),
        ],
        robot_x=10.0,
        robot_y=10.0,
    )

    result = RobotQueryRuntime(graph, context).closest_object()

    assert result.status == "ok"
    assert result.object_ids == ["world_near_object"]


def test_object_in_front_respects_heading_and_field_of_view() -> None:
    graph = make_graph(
        detection("north", "chair", (850, 450, 950, 550)),
        detection("east", "table", (450, 450, 550, 550)),
        detection("northwest_outside_fov", "plant", (400, 100, 500, 200)),
    )
    context = navigation_context(
        [
            nav_object("north", x=0.0, y=4.0),
            nav_object("east", x=4.0, y=0.0),
            nav_object("northwest_outside_fov", x=-4.0, y=2.0),
        ],
        heading_radians=math.pi / 2.0,
    )
    runtime = RobotQueryRuntime(
        graph,
        context,
        config=RobotQueryConfig(front_field_of_view_degrees=60.0),
    )

    result = runtime.object_in_front()

    assert result.status == "ok"
    assert result.object_ids == ["north"]


def test_missing_object_position_returns_insufficient_data() -> None:
    graph = make_graph(
        detection("positioned", "chair", (100, 100, 200, 200)),
        detection("position_missing", "table", (700, 700, 800, 800)),
    )
    context = navigation_context(
        [
            nav_object("positioned", x=2.0, y=0.0),
            nav_object("position_missing", x=None, y=None),
        ]
    )

    result = RobotQueryRuntime(graph, context).closest_object()

    assert result.status == "insufficient_data"
    assert result.object_ids == []
    assert result.missing_object_ids == ["position_missing"]


def test_multiple_doors_are_ambiguous_unless_one_is_marked_as_exit() -> None:
    graph = make_graph(
        detection("door_a", "door", (100, 100, 200, 800)),
        detection("door_b", "door", (800, 100, 900, 800)),
    )
    ambiguous_context = navigation_context(
        [
            nav_object("door_a", x=5.0, y=-2.0),
            nav_object("door_b", x=5.0, y=2.0),
        ]
    )

    ambiguous = RobotQueryRuntime(graph, ambiguous_context).exit_target()

    assert ambiguous.status == "ambiguous"
    assert ambiguous.object_ids == ["door_a", "door_b"]

    explicit_exit_context = navigation_context(
        [
            nav_object("door_a", x=5.0, y=-2.0),
            nav_object("door_b", x=5.0, y=2.0, role="exit"),
        ]
    )

    selected = RobotQueryRuntime(graph, explicit_exit_context).exit_target()

    assert selected.status == "ok"
    assert selected.object_ids == ["door_b"]


def test_blocker_requires_footprint_intersection_with_target_corridor() -> None:
    graph = make_graph(
        detection("target", "door", (800, 100, 900, 800)),
        detection("intersects_corridor", "box", (100, 100, 200, 200)),
        detection("outside_corridor", "chair", (300, 300, 400, 400)),
        detection("behind_robot", "crate", (500, 500, 600, 600)),
        detection("non_obstacle", "floor_marker", (600, 600, 700, 700)),
    )
    context = navigation_context(
        [
            nav_object("target", x=10.0, y=0.0, role="exit"),
            nav_object(
                "intersects_corridor",
                x=5.0,
                y=0.6,
                radius=0.4,
                is_obstacle=True,
            ),
            nav_object(
                "outside_corridor",
                x=5.0,
                y=1.0,
                radius=0.4,
                is_obstacle=True,
            ),
            nav_object(
                "behind_robot",
                x=-1.0,
                y=0.0,
                radius=0.4,
                is_obstacle=True,
            ),
            nav_object(
                "non_obstacle",
                x=3.0,
                y=0.0,
                radius=0.4,
                is_obstacle=False,
            ),
        ],
        robot_radius=0.25,
    )

    result = RobotQueryRuntime(graph, context).blockers_for("target")

    assert result.status == "ok"
    assert result.object_ids == ["intersects_corridor"]


def test_filtered_low_confidence_detection_is_not_returned_by_robot_query() -> None:
    graph = make_graph(
        detection(
            "retained",
            "chair",
            (800, 800, 900, 900),
            confidence=0.95,
        ),
        detection(
            "filtered",
            "cup",
            (450, 450, 550, 550),
            confidence=0.1,
        ),
    )
    context = navigation_context(
        [
            nav_object("retained", x=3.0, y=0.0),
            nav_object("filtered", x=0.1, y=0.0),
        ]
    )

    assert graph.get_node("filtered") is None

    result = RobotQueryRuntime(graph, context).closest_object()

    assert result.status == "ok"
    assert result.object_ids == ["retained"]


def test_objects_near_returns_not_found_for_unknown_target() -> None:
    graph = make_graph(
        detection("chair", "chair", (100, 100, 200, 200)),
    )
    context = navigation_context(
        [
            nav_object("chair", x=1.0, y=0.0),
        ]
    )

    result = RobotQueryRuntime(graph, context).objects_near("missing")

    assert result.status == "not_found"
    assert result.object_ids == []


def test_objects_near_requires_target_world_position() -> None:
    graph = make_graph(
        detection("target", "door", (100, 100, 200, 800)),
        detection("chair", "chair", (300, 300, 400, 400)),
    )
    context = navigation_context(
        [
            nav_object("target", x=None, y=None),
            nav_object("chair", x=1.0, y=0.0),
        ]
    )

    result = RobotQueryRuntime(graph, context).objects_near("target")

    assert result.status == "insufficient_data"
    assert result.object_ids == []
    assert result.missing_object_ids == ["target"]


def test_objects_near_requires_candidate_world_positions() -> None:
    graph = make_graph(
        detection("target", "door", (100, 100, 200, 800)),
        detection("positioned", "chair", (300, 300, 400, 400)),
        detection("position_missing", "table", (500, 500, 600, 600)),
    )
    context = navigation_context(
        [
            nav_object("target", x=0.0, y=0.0),
            nav_object("positioned", x=1.0, y=0.0),
            nav_object("position_missing", x=None, y=None),
        ]
    )

    result = RobotQueryRuntime(graph, context).objects_near("target")

    assert result.status == "insufficient_data"
    assert result.object_ids == []
    assert result.missing_object_ids == ["position_missing"]


def test_objects_near_orders_by_world_distance_then_object_id() -> None:
    graph = make_graph(
        detection("target", "door", (450, 100, 550, 900)),
        detection("same_distance_b", "chair", (100, 100, 200, 200)),
        detection("same_distance_a", "table", (800, 100, 900, 200)),
        detection("nearest", "cup", (450, 450, 550, 550)),
        detection("outside_threshold", "plant", (300, 700, 400, 800)),
    )
    context = navigation_context(
        [
            nav_object("target", x=10.0, y=10.0),
            nav_object("same_distance_b", x=11.0, y=10.0),
            nav_object("same_distance_a", x=9.0, y=10.0),
            nav_object("nearest", x=10.0, y=10.5),
            nav_object("outside_threshold", x=13.0, y=10.0),
        ]
    )
    runtime = RobotQueryRuntime(
        graph,
        context,
        config=RobotQueryConfig(near_distance=2.0),
    )

    result = runtime.objects_near("target")

    assert result.status == "ok"
    assert result.object_ids == [
        "nearest",
        "same_distance_a",
        "same_distance_b",
    ]
    assert [item.object_id for item in result.evidence] == [
        "nearest",
        "same_distance_a",
        "same_distance_b",
        "outside_threshold",
    ]


def test_object_to_avoid_returns_nearest_obstacle_inside_heading_fov() -> None:
    graph = make_graph(
        detection("nearest_obstacle", "box", (800, 400, 900, 500)),
        detection("far_obstacle", "chair", (700, 400, 800, 500)),
        detection("near_non_obstacle", "floor_marker", (600, 400, 700, 500)),
        detection("behind_obstacle", "crate", (500, 400, 600, 500)),
        detection("outside_fov_obstacle", "plant", (400, 400, 500, 500)),
    )
    context = navigation_context(
        [
            nav_object(
                "nearest_obstacle",
                x=2.0,
                y=0.0,
                is_obstacle=True,
            ),
            nav_object("far_obstacle", x=4.0, y=0.0, is_obstacle=True),
            nav_object("near_non_obstacle", x=0.5, y=0.0),
            nav_object("behind_obstacle", x=-1.0, y=0.0, is_obstacle=True),
            nav_object(
                "outside_fov_obstacle",
                x=1.0,
                y=2.0,
                is_obstacle=True,
            ),
        ]
    )
    runtime = RobotQueryRuntime(
        graph,
        context,
        config=RobotQueryConfig(front_field_of_view_degrees=60.0),
    )

    result = runtime.object_to_avoid()

    assert result.status == "ok"
    assert result.object_ids == ["nearest_obstacle"]


def test_object_to_avoid_returns_not_found_when_no_obstacle_is_in_front() -> None:
    graph = make_graph(
        detection("front_non_obstacle", "floor_marker", (700, 400, 800, 500)),
        detection("behind_obstacle", "crate", (500, 400, 600, 500)),
        detection("outside_fov_obstacle", "plant", (400, 400, 500, 500)),
    )
    context = navigation_context(
        [
            nav_object("front_non_obstacle", x=1.0, y=0.0),
            nav_object("behind_obstacle", x=-1.0, y=0.0, is_obstacle=True),
            nav_object(
                "outside_fov_obstacle",
                x=1.0,
                y=2.0,
                is_obstacle=True,
            ),
        ]
    )
    runtime = RobotQueryRuntime(
        graph,
        context,
        config=RobotQueryConfig(front_field_of_view_degrees=60.0),
    )

    result = runtime.object_to_avoid()

    assert result.status == "not_found"
    assert result.object_ids == []


def test_navigation_context_fixture_is_separate_from_detection_fixture() -> None:
    context = NavigationContext.model_validate_json(
        Path("samples/room_scene_navigation.json").read_text(encoding="utf-8")
    )

    assert context.frame_id == "map"
    assert context.robot_pose.x == 0.0
    assert {item.object_id for item in context.objects} == {
        "door_1",
        "chair_1",
        "table_1",
        "backpack_1",
        "wall_art_1",
        "box_1",
    }
    door = next(item for item in context.objects if item.object_id == "door_1")
    assert door.role == "exit"
