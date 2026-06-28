from math import pi, sqrt

from scenegraph_navbench.robot_geometry import (
    footprint_intersects_corridor,
    is_within_heading_fov,
)


def test_exact_field_of_view_edge_counts_as_inside() -> None:
    assert is_within_heading_fov(
        origin=(0.0, 0.0),
        heading_radians=0.0,
        target=(sqrt(3.0), 1.0),
        field_of_view_radians=pi / 3.0,
    )


def test_target_behind_robot_is_outside_field_of_view() -> None:
    assert not is_within_heading_fov(
        origin=(0.0, 0.0),
        heading_radians=0.0,
        target=(-1.0, 0.0),
        field_of_view_radians=pi / 2.0,
    )


def test_tangent_footprint_counts_as_corridor_intersection() -> None:
    result = footprint_intersects_corridor(
        corridor_start=(0.0, 0.0),
        corridor_end=(10.0, 0.0),
        robot_radius=0.25,
        footprint_center=(5.0, 1.0),
        footprint_radius=0.75,
    )

    assert result.intersects
    assert result.distance_to_segment == 1.0
    assert result.clearance == 1.0


def test_footprint_beyond_target_does_not_intersect_corridor() -> None:
    result = footprint_intersects_corridor(
        corridor_start=(0.0, 0.0),
        corridor_end=(10.0, 0.0),
        robot_radius=0.25,
        footprint_center=(10.5, 0.0),
        footprint_radius=0.4,
    )

    assert result.segment_projection > 1.0
    assert not result.intersects
