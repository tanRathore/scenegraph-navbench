"""Deterministic robot queries over explicit world-frame navigation metadata."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import degrees, isfinite, pi
from typing import TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from scenegraph_navbench.models import SceneGraph
from scenegraph_navbench.robot_context import NavigationContext, NavigationObject
from scenegraph_navbench.robot_geometry import (
    distance,
    footprint_intersects_corridor,
    heading_offset_radians,
    is_within_heading_fov,
)

EvidenceValue: TypeAlias = float | int | str | bool


class QueryStatus(str, Enum):
    """Explicit completion status for a deterministic robot query."""

    OK = "ok"
    AMBIGUOUS = "ambiguous"
    NOT_FOUND = "not_found"
    INSUFFICIENT_DATA = "insufficient_data"


class QueryEvidence(BaseModel):
    """Structured measurements explaining one evaluated object."""

    model_config = ConfigDict(extra="forbid")

    object_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    measurements: dict[str, EvidenceValue] = Field(default_factory=dict)


class RobotQueryResult(BaseModel):
    """Typed answer returned by every robot query."""

    model_config = ConfigDict(extra="forbid")

    status: QueryStatus
    object_ids: list[str]
    reason: str = Field(min_length=1)
    evidence: list[QueryEvidence] = Field(default_factory=list)
    missing_object_ids: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class RobotQueryConfig:
    """Thresholds controlling deterministic query behavior."""

    front_field_of_view_degrees: float = 90.0
    near_distance: float = 2.0

    def __post_init__(self) -> None:
        if not 0.0 < self.front_field_of_view_degrees <= 360.0:
            raise ValueError("front_field_of_view_degrees must be in (0, 360]")
        if not isfinite(self.near_distance) or self.near_distance < 0.0:
            raise ValueError("near_distance must be finite and non-negative")


class RobotQueryRuntime:
    """Answer robot-relative questions without using image-space geometry."""

    def __init__(
        self,
        graph: SceneGraph,
        context: NavigationContext,
        *,
        config: RobotQueryConfig | None = None,
    ) -> None:
        node_ids = [node.id for node in graph.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("scene graph node ids must be unique")

        self.graph = graph
        self.context = context
        self.config = config or RobotQueryConfig()
        self._nodes = {node.id: node for node in graph.nodes}
        self._navigation = {
            item.object_id: item
            for item in context.objects
            if item.object_id in self._nodes
        }

    def closest_object(self) -> RobotQueryResult:
        """Return the graph object nearest to the robot in world coordinates."""
        if not self._nodes:
            return _result(
                QueryStatus.NOT_FOUND,
                reason="The scene graph contains no queryable objects.",
            )

        positioned, missing = self._positioned_graph_objects()
        evidence = [
            QueryEvidence(
                object_id=object_id,
                reason="World-frame position is missing.",
                measurements={"frame_id": self.context.frame_id},
            )
            for object_id in missing
        ]
        evidence.extend(
            QueryEvidence(
                object_id=item.object_id,
                reason="Measured Euclidean distance from the robot world position.",
                measurements={
                    "distance": distance(
                        self.context.robot_pose.position,
                        item.position,
                    ),
                    "frame_id": self.context.frame_id,
                },
            )
            for item in positioned
            if item.position is not None
        )

        if missing:
            return _result(
                QueryStatus.INSUFFICIENT_DATA,
                reason=(
                    "Closest object cannot be determined while graph objects "
                    "lack world-frame positions."
                ),
                evidence=evidence,
                missing_object_ids=missing,
            )

        ranked = sorted(
            positioned,
            key=lambda item: (
                distance(self.context.robot_pose.position, _position(item)),
                item.object_id,
            ),
        )
        selected = ranked[0]
        selected_distance = distance(
            self.context.robot_pose.position,
            _position(selected),
        )
        return _result(
            QueryStatus.OK,
            object_ids=[selected.object_id],
            reason=(
                f"{selected.object_id} has the minimum world-frame distance "
                f"({selected_distance:.3f})."
            ),
            evidence=evidence,
        )

    def object_in_front(self) -> RobotQueryResult:
        """Return the nearest graph object inside the robot's heading FOV."""
        if not self._nodes:
            return _result(
                QueryStatus.NOT_FOUND,
                reason="The scene graph contains no queryable objects.",
            )

        positioned, missing = self._positioned_graph_objects()
        if missing:
            return _result(
                QueryStatus.INSUFFICIENT_DATA,
                reason=(
                    "The object in front cannot be determined while graph "
                    "objects lack world-frame positions."
                ),
                evidence=[
                    QueryEvidence(
                        object_id=object_id,
                        reason="World-frame position is missing.",
                        measurements={"frame_id": self.context.frame_id},
                    )
                    for object_id in missing
                ],
                missing_object_ids=missing,
            )

        field_of_view_radians = (
            self.config.front_field_of_view_degrees * pi / 180.0
        )
        candidates: list[NavigationObject] = []
        evidence: list[QueryEvidence] = []
        for item in positioned:
            position = _position(item)
            offset = heading_offset_radians(
                self.context.robot_pose.position,
                self.context.robot_pose.heading_radians,
                position,
            )
            within_fov = is_within_heading_fov(
                self.context.robot_pose.position,
                self.context.robot_pose.heading_radians,
                position,
                field_of_view_radians,
            )
            if within_fov:
                candidates.append(item)
            evidence.append(
                QueryEvidence(
                    object_id=item.object_id,
                    reason=(
                        "Compared the world-frame bearing with the robot heading."
                    ),
                    measurements={
                        "distance": distance(
                            self.context.robot_pose.position,
                            position,
                        ),
                        "heading_offset_degrees": (
                            degrees(offset) if offset is not None else 0.0
                        ),
                        "field_of_view_degrees": (
                            self.config.front_field_of_view_degrees
                        ),
                        "within_field_of_view": within_fov,
                        "frame_id": self.context.frame_id,
                    },
                )
            )

        if not candidates:
            return _result(
                QueryStatus.NOT_FOUND,
                reason="No graph object lies inside the robot's heading FOV.",
                evidence=evidence,
            )

        selected = min(
            candidates,
            key=lambda item: (
                distance(self.context.robot_pose.position, _position(item)),
                item.object_id,
            ),
        )
        return _result(
            QueryStatus.OK,
            object_ids=[selected.object_id],
            reason=(
                f"{selected.object_id} is the nearest object inside the "
                f"{self.config.front_field_of_view_degrees:.1f}-degree FOV."
            ),
            evidence=evidence,
        )

    def object_to_avoid(self) -> RobotQueryResult:
        """Return the nearest explicit obstacle inside the robot's heading FOV."""
        obstacles = sorted(
            (item for item in self._navigation.values() if item.is_obstacle),
            key=lambda item: item.object_id,
        )
        missing = sorted(
            item.object_id for item in obstacles if item.position is None
        )
        if missing:
            return _result(
                QueryStatus.INSUFFICIENT_DATA,
                reason=(
                    "The nearest obstacle in front cannot be determined while "
                    "explicit obstacles lack world-frame positions."
                ),
                missing_object_ids=missing,
                evidence=[
                    QueryEvidence(
                        object_id=object_id,
                        reason="World-frame obstacle position is missing.",
                        measurements={"frame_id": self.context.frame_id},
                    )
                    for object_id in missing
                ],
            )

        field_of_view_radians = (
            self.config.front_field_of_view_degrees * pi / 180.0
        )
        candidates: list[tuple[float, NavigationObject]] = []
        evidence: list[QueryEvidence] = []
        for item in obstacles:
            position = _position(item)
            measured_distance = distance(
                self.context.robot_pose.position,
                position,
            )
            offset = heading_offset_radians(
                self.context.robot_pose.position,
                self.context.robot_pose.heading_radians,
                position,
            )
            within_fov = is_within_heading_fov(
                self.context.robot_pose.position,
                self.context.robot_pose.heading_radians,
                position,
                field_of_view_radians,
            )
            if within_fov:
                candidates.append((measured_distance, item))
            evidence.append(
                QueryEvidence(
                    object_id=item.object_id,
                    reason=(
                        "Evaluated an explicit obstacle against the robot "
                        "heading and field of view."
                    ),
                    measurements={
                        "distance": measured_distance,
                        "heading_offset_degrees": (
                            degrees(offset) if offset is not None else 0.0
                        ),
                        "field_of_view_degrees": (
                            self.config.front_field_of_view_degrees
                        ),
                        "within_field_of_view": within_fov,
                        "is_obstacle": True,
                        "frame_id": self.context.frame_id,
                    },
                )
            )

        if not candidates:
            return _result(
                QueryStatus.NOT_FOUND,
                reason="No explicit obstacle lies inside the robot's heading FOV.",
                evidence=evidence,
            )

        measured_distance, selected = min(
            candidates,
            key=lambda pair: (pair[0], pair[1].object_id),
        )
        return _result(
            QueryStatus.OK,
            object_ids=[selected.object_id],
            reason=(
                f"{selected.object_id} is the nearest explicit obstacle inside "
                f"the heading FOV at world distance {measured_distance:.3f}."
            ),
            evidence=evidence,
        )

    def exit_target(self) -> RobotQueryResult:
        """Resolve an explicit exit role, or an unambiguous door fallback."""
        explicit_exits = sorted(
            (
                item.object_id
                for item in self._navigation.values()
                if item.role is not None and item.role.casefold() == "exit"
            )
        )
        if len(explicit_exits) == 1:
            object_id = explicit_exits[0]
            return _result(
                QueryStatus.OK,
                object_ids=[object_id],
                reason=f"{object_id} is explicitly marked with role=exit.",
                evidence=[
                    QueryEvidence(
                        object_id=object_id,
                        reason="Selected from explicit navigation semantics.",
                        measurements={"role": "exit"},
                    )
                ],
            )
        if len(explicit_exits) > 1:
            return _result(
                QueryStatus.AMBIGUOUS,
                object_ids=explicit_exits,
                reason="Multiple graph objects are explicitly marked as exits.",
                evidence=[
                    QueryEvidence(
                        object_id=object_id,
                        reason="Object is explicitly marked with role=exit.",
                        measurements={"role": "exit"},
                    )
                    for object_id in explicit_exits
                ],
            )

        doors = sorted(
            node.id for node in self._nodes.values() if node.label.casefold() == "door"
        )
        if len(doors) == 1:
            object_id = doors[0]
            return _result(
                QueryStatus.OK,
                object_ids=[object_id],
                reason=(
                    f"{object_id} is the only door in the filtered scene graph."
                ),
                evidence=[
                    QueryEvidence(
                        object_id=object_id,
                        reason="Selected as the sole graph node labeled door.",
                        measurements={"label": "door"},
                    )
                ],
            )
        if len(doors) > 1:
            return _result(
                QueryStatus.AMBIGUOUS,
                object_ids=doors,
                reason=(
                    "Multiple doors exist and no graph object is explicitly "
                    "marked with role=exit."
                ),
                evidence=[
                    QueryEvidence(
                        object_id=object_id,
                        reason="Object is a door without an explicit exit role.",
                        measurements={"label": "door"},
                    )
                    for object_id in doors
                ],
            )

        return _result(
            QueryStatus.NOT_FOUND,
            reason="No explicit exit or door exists in the filtered scene graph.",
        )

    def objects_near(self, target_id: str) -> RobotQueryResult:
        """Return graph objects within world distance of a target graph object."""
        if target_id not in self._nodes:
            return _result(
                QueryStatus.NOT_FOUND,
                reason=f"Target {target_id!r} is not in the filtered scene graph.",
            )

        target = self._navigation.get(target_id)
        if target is None or target.position is None:
            return _result(
                QueryStatus.INSUFFICIENT_DATA,
                reason=f"Target {target_id!r} lacks a world-frame position.",
                missing_object_ids=[target_id],
                evidence=[
                    QueryEvidence(
                        object_id=target_id,
                        reason="World-frame target position is missing.",
                        measurements={"frame_id": self.context.frame_id},
                    )
                ],
            )

        candidates: list[NavigationObject] = []
        missing: list[str] = []
        for object_id in sorted(self._nodes):
            if object_id == target_id:
                continue
            item = self._navigation.get(object_id)
            if item is None or item.position is None:
                missing.append(object_id)
            else:
                candidates.append(item)

        if missing:
            return _result(
                QueryStatus.INSUFFICIENT_DATA,
                reason=(
                    "Nearby objects cannot be determined while candidate graph "
                    "objects lack world-frame positions."
                ),
                missing_object_ids=missing,
                evidence=[
                    QueryEvidence(
                        object_id=object_id,
                        reason="World-frame candidate position is missing.",
                        measurements={"frame_id": self.context.frame_id},
                    )
                    for object_id in missing
                ],
            )

        measured = sorted(
            (
                (distance(target.position, _position(item)), item)
                for item in candidates
            ),
            key=lambda pair: (pair[0], pair[1].object_id),
        )
        object_ids = [
            item.object_id
            for measured_distance, item in measured
            if measured_distance <= self.config.near_distance
        ]
        evidence = [
            QueryEvidence(
                object_id=item.object_id,
                reason=(
                    "Compared world-frame object distance with the configured "
                    "near threshold."
                ),
                measurements={
                    "distance": measured_distance,
                    "near_distance": self.config.near_distance,
                    "within_near_distance": (
                        measured_distance <= self.config.near_distance
                    ),
                    "frame_id": self.context.frame_id,
                },
            )
            for measured_distance, item in measured
        ]
        return _result(
            QueryStatus.OK,
            object_ids=object_ids,
            reason=(
                f"{len(object_ids)} graph object(s) are within "
                f"{self.config.near_distance:.3f} world units of {target_id}."
            ),
            evidence=evidence,
        )

    def blockers_for(self, target_id: str) -> RobotQueryResult:
        """Return obstacle footprints intersecting the robot-to-target corridor."""
        if target_id not in self._nodes:
            return _result(
                QueryStatus.NOT_FOUND,
                reason=f"Target {target_id!r} is not in the filtered scene graph.",
            )

        target = self._navigation.get(target_id)
        if target is None or target.position is None:
            return _result(
                QueryStatus.INSUFFICIENT_DATA,
                reason=f"Target {target_id!r} lacks a world-frame position.",
                missing_object_ids=[target_id],
                evidence=[
                    QueryEvidence(
                        object_id=target_id,
                        reason="World-frame target position is missing.",
                        measurements={"frame_id": self.context.frame_id},
                    )
                ],
            )

        target_position = target.position
        if distance(self.context.robot_pose.position, target_position) == 0.0:
            return _result(
                QueryStatus.INSUFFICIENT_DATA,
                reason="A corridor is undefined when the target is at the robot pose.",
                missing_object_ids=[target_id],
            )

        obstacles = sorted(
            (
                item
                for item in self._navigation.values()
                if item.object_id != target_id and item.is_obstacle
            ),
            key=lambda item: item.object_id,
        )
        missing = sorted(
            item.object_id for item in obstacles if item.position is None
        )
        if missing:
            return _result(
                QueryStatus.INSUFFICIENT_DATA,
                reason=(
                    "Blockers cannot be determined while obstacle footprints "
                    "lack world-frame positions."
                ),
                missing_object_ids=missing,
                evidence=[
                    QueryEvidence(
                        object_id=object_id,
                        reason="World-frame obstacle position is missing.",
                        measurements={"frame_id": self.context.frame_id},
                    )
                    for object_id in missing
                ],
            )

        blockers: list[tuple[float, str]] = []
        evidence: list[QueryEvidence] = []
        for item in obstacles:
            check = footprint_intersects_corridor(
                self.context.robot_pose.position,
                target_position,
                self.context.robot_radius,
                _position(item),
                item.radius,
            )
            if check.intersects:
                blockers.append((check.segment_projection, item.object_id))
            evidence.append(
                QueryEvidence(
                    object_id=item.object_id,
                    reason=(
                        "Compared the obstacle footprint with the swept "
                        "robot-to-target corridor."
                    ),
                    measurements={
                        "distance_to_corridor": check.distance_to_segment,
                        "required_clearance": check.clearance,
                        "segment_projection": check.segment_projection,
                        "intersects": check.intersects,
                        "frame_id": self.context.frame_id,
                    },
                )
            )

        blocker_ids = [
            object_id for _, object_id in sorted(blockers, key=lambda item: item)
        ]
        return _result(
            QueryStatus.OK,
            object_ids=blocker_ids,
            reason=(
                f"{len(blocker_ids)} explicit obstacle footprint(s) intersect "
                f"the world-frame corridor to {target_id}."
            ),
            evidence=evidence,
        )

    def _positioned_graph_objects(
        self,
    ) -> tuple[list[NavigationObject], list[str]]:
        positioned: list[NavigationObject] = []
        missing: list[str] = []
        for object_id in sorted(self._nodes):
            item = self._navigation.get(object_id)
            if item is None or item.position is None:
                missing.append(object_id)
            else:
                positioned.append(item)
        return positioned, missing


def _position(item: NavigationObject) -> tuple[float, float]:
    position = item.position
    if position is None:
        raise ValueError(f"navigation object {item.object_id!r} has no position")
    return position


def _result(
    status: QueryStatus,
    *,
    object_ids: list[str] | None = None,
    reason: str,
    evidence: list[QueryEvidence] | None = None,
    missing_object_ids: list[str] | None = None,
) -> RobotQueryResult:
    return RobotQueryResult(
        status=status,
        object_ids=object_ids or [],
        reason=reason,
        evidence=evidence or [],
        missing_object_ids=missing_object_ids or [],
    )
