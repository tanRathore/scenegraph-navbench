"""Pure planar geometry helpers for robot queries."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, hypot, isclose, pi

Point2D = tuple[float, float]


@dataclass(frozen=True)
class CorridorIntersection:
    """Measurements for a circular footprint against a swept robot corridor."""

    intersects: bool
    distance_to_segment: float
    clearance: float
    segment_projection: float


def distance(first: Point2D, second: Point2D) -> float:
    """Return Euclidean distance between two world-frame points."""
    return hypot(second[0] - first[0], second[1] - first[1])


def heading_offset_radians(
    origin: Point2D,
    heading_radians: float,
    target: Point2D,
) -> float | None:
    """Return the smallest absolute angle between a heading and target bearing."""
    dx = target[0] - origin[0]
    dy = target[1] - origin[1]
    if dx == 0.0 and dy == 0.0:
        return None

    target_bearing = atan2(dy, dx)
    signed_offset = (target_bearing - heading_radians + pi) % (2.0 * pi) - pi
    return abs(signed_offset)


def is_within_heading_fov(
    origin: Point2D,
    heading_radians: float,
    target: Point2D,
    field_of_view_radians: float,
) -> bool:
    """Return whether a target lies inside the robot's symmetric heading FOV."""
    if not 0.0 < field_of_view_radians <= 2.0 * pi:
        raise ValueError("field_of_view_radians must be in (0, 2*pi]")

    offset = heading_offset_radians(origin, heading_radians, target)
    half_field_of_view = field_of_view_radians / 2.0
    return offset is not None and (
        offset <= half_field_of_view
        or isclose(offset, half_field_of_view, rel_tol=1e-12, abs_tol=1e-12)
    )


def point_to_segment_distance(
    point: Point2D,
    segment_start: Point2D,
    segment_end: Point2D,
) -> tuple[float, float]:
    """Return distance to a segment and the clamped projection in [0, 1]."""
    segment_x = segment_end[0] - segment_start[0]
    segment_y = segment_end[1] - segment_start[1]
    length_squared = segment_x * segment_x + segment_y * segment_y

    if length_squared == 0.0:
        return distance(point, segment_start), 0.0

    projection = (
        (point[0] - segment_start[0]) * segment_x
        + (point[1] - segment_start[1]) * segment_y
    ) / length_squared
    clamped_projection = min(1.0, max(0.0, projection))
    closest = (
        segment_start[0] + clamped_projection * segment_x,
        segment_start[1] + clamped_projection * segment_y,
    )
    return distance(point, closest), clamped_projection


def footprint_intersects_corridor(
    corridor_start: Point2D,
    corridor_end: Point2D,
    robot_radius: float,
    footprint_center: Point2D,
    footprint_radius: float,
) -> CorridorIntersection:
    """Check a circular footprint against the robot's swept path segment."""
    if robot_radius < 0.0:
        raise ValueError("robot_radius must be non-negative")
    if footprint_radius < 0.0:
        raise ValueError("footprint_radius must be non-negative")

    segment_distance, _ = point_to_segment_distance(
        footprint_center,
        corridor_start,
        corridor_end,
    )
    segment_x = corridor_end[0] - corridor_start[0]
    segment_y = corridor_end[1] - corridor_start[1]
    length_squared = segment_x * segment_x + segment_y * segment_y
    projection = (
        (
            (footprint_center[0] - corridor_start[0]) * segment_x
            + (footprint_center[1] - corridor_start[1]) * segment_y
        )
        / length_squared
        if length_squared > 0.0
        else 0.0
    )
    clearance = robot_radius + footprint_radius
    return CorridorIntersection(
        intersects=(
            length_squared > 0.0
            and 0.0 <= projection <= 1.0
            and segment_distance <= clearance
        ),
        distance_to_segment=segment_distance,
        clearance=clearance,
        segment_projection=projection,
    )
