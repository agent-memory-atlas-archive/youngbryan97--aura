"""Object-relative geometry persists when viewpoint or visibility changes.

This is a declared rigid-body model, not a vision recognizer. Occlusion keeps
the last measured shape/landmarks and does not certify the current pose. An
explicit deformation/replacement observation, not a camera turn, changes
intrinsic geometry. Track identities reuse the existing TrackStore.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

from core.cognition.entity_track import TrackState
from core.governance_context import local_internal_governed_scope
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.state_ownership import state_root
from core.verify.invariants import invariant


def vector3(value):
    value = tuple(float(part) for part in value)
    if len(value) != 3 or not all(map(math.isfinite, value)):
        raise ValueError("rigid object coordinates must be finite three-vectors")
    return value


@dataclass(frozen=True)
class RigidPose:
    translation: tuple[float, float, float] = (0., 0., 0.)
    quaternion: tuple[float, float, float, float] = (0., 0., 0., 1.)

    def __post_init__(self):
        object.__setattr__(self, "translation", vector3(self.translation))
        quaternion = np.asarray(self.quaternion, dtype=float)
        if (quaternion.shape != (4,) or not np.isfinite(quaternion).all()
                or not math.isfinite(float(np.linalg.norm(quaternion))) or np.linalg.norm(quaternion) < 1e-12):
            raise ValueError("rigid object pose needs a nonzero finite rotation quaternion")
        object.__setattr__(self, "quaternion", tuple(Rotation.from_quat(quaternion).as_quat()))

    def apply(self, point):
        return Rotation.from_quat(self.quaternion).apply(vector3(point)) + np.asarray(self.translation)

    def inverse_apply(self, point):
        return Rotation.from_quat(self.quaternion).inv().apply(np.asarray(vector3(point)) - self.translation)

    def compose(self, local):
        rotation = Rotation.from_quat(self.quaternion) * Rotation.from_quat(local.quaternion)
        return RigidPose(tuple(self.apply(local.translation)), tuple(rotation.as_quat()))


@dataclass(frozen=True)
class CapsuleGeometry:
    """A segment swept by a radius; equal endpoints are an exact sphere.

    Geometry and pose translations use SI metres in a common world frame.
    """
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    radius: float

    def __post_init__(self):
        object.__setattr__(self, "start", vector3(self.start))
        object.__setattr__(self, "end", vector3(self.end))
        if not math.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("rigid geometry needs a positive finite radius")

    @property
    def length(self):
        return math.dist(self.start, self.end) + 2. * self.radius


@dataclass(frozen=True)
class RigidObjectBelief:
    identity: str
    geometry: CapsuleGeometry
    pose: RigidPose
    source: str
    observed_at: float
    landmarks: dict = field(default_factory=dict)
    surface_properties: dict = field(default_factory=dict)
    visible: bool = True
    pose_uncertainty: float = 0.
    pose_updated_at: float | None = None
    pose_prediction_source: str | None = None
    intrinsic_sources: dict = field(default_factory=dict)

    def __post_init__(self):
        if (not isinstance(self.identity, str) or not self.identity or not isinstance(self.source, str) or not self.source
                or not isinstance(self.geometry, CapsuleGeometry) or not isinstance(self.pose, RigidPose)
                or type(self.observed_at) not in {int, float} or type(self.pose_uncertainty) not in {int, float}
                or not math.isfinite(self.observed_at) or self.observed_at < 0
                or not math.isfinite(self.pose_uncertainty) or self.pose_uncertainty < 0
                or type(self.visible) is not bool):
            raise ValueError("rigid object belief needs measured identity and provenance")
        landmarks = {name: vector3(point) for name, point in self.landmarks.items()}
        properties = dict(self.surface_properties)
        if any(not isinstance(name, str) or not name for name in (*landmarks, *properties)):
            raise ValueError("rigid object surface identities differ")
        if any(not isinstance(value, str) for value in properties.values()):
            raise ValueError("rigid object surface properties must remain explicit observations")
        object.__setattr__(self, "landmarks", MappingProxyType(landmarks))
        object.__setattr__(self, "surface_properties", MappingProxyType(properties))
        names = {*(f"landmark:{name}" for name in landmarks), *(f"surface:{name}" for name in properties)}
        provenance = dict(self.intrinsic_sources)
        if not set(provenance) <= names:
            raise ValueError("intrinsic provenance must identify a retained property")
        for name in names:
            record = provenance.get(name, (self.source, self.observed_at))
            if (not isinstance(record, (tuple, list)) or len(record) != 2
                    or not isinstance(record[0], str) or not record[0]
                    or type(record[1]) not in {int, float} or not math.isfinite(record[1])
                    or not 0 <= record[1] <= self.observed_at):
                raise ValueError("intrinsic provenance needs a measured source and ordered time")
            provenance[name] = tuple(record)
        object.__setattr__(self, "intrinsic_sources", MappingProxyType(provenance))
        updated_at = self.observed_at if self.pose_updated_at is None else self.pose_updated_at
        if (type(updated_at) not in {int, float} or not math.isfinite(updated_at) or updated_at < self.observed_at
                or self.pose_prediction_source is not None and
                   (not isinstance(self.pose_prediction_source, str) or not self.pose_prediction_source)):
            raise ValueError("rigid pose update requires ordered time and explicit prediction provenance")
        object.__setattr__(self, "pose_updated_at", updated_at)

    def landmark_in_view(self, name, observer=None):
        observer = RigidPose() if observer is None else observer
        return observer.inverse_apply(self.pose.apply(self.landmarks[name]))

    def world_segment(self):
        return self.pose.apply(self.geometry.start), self.pose.apply(self.geometry.end)

    def to_dict(self):
        return {"identity": self.identity, "geometry": {"start": self.geometry.start,
            "end": self.geometry.end, "radius": self.geometry.radius},
            "pose": {"translation": self.pose.translation, "quaternion": self.pose.quaternion},
            "source": self.source, "observed_at": self.observed_at, "landmarks": dict(self.landmarks),
            "surface_properties": dict(self.surface_properties), "visible": self.visible,
            "pose_uncertainty": self.pose_uncertainty, "pose_updated_at": self.pose_updated_at,
            "pose_prediction_source": self.pose_prediction_source,
            "intrinsic_sources": dict(self.intrinsic_sources)}

    @classmethod
    def from_dict(cls, value):
        return cls(**{**value, "geometry": CapsuleGeometry(**value["geometry"]),
                      "pose": RigidPose(**value["pose"])})


def swept_capsule_contact(body, obstacle, displacement):
    """Continuous translation, retaining the entire hidden object extent.

    Minimize segment separation over both segment coordinates and time in
    [0, 1]. This does not approximate simultaneous rotations or deformation.
    Model contact is not a claim about an unmeasured real-world obstacle.
    """
    displacement = np.asarray(vector3(displacement))
    a, b = body.world_segment()
    c, d = obstacle.world_segment()
    matrix = np.stack((b - a, -(d - c), displacement), axis=1)
    result = lsq_linear(matrix, c - a, bounds=(0., 1.), tol=1e-12, max_iter=200)
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError("swept rigid contact solve did not converge")
    distance = float(np.linalg.norm(matrix @ result.x - (c - a)))
    # A feasible iterate bounds contact from above. Convexity bounds clearance
    # from below even when optimizer tolerances leave an unresolved gap.
    residual = matrix @ result.x - (c - a)
    gradient = matrix.T @ residual
    gap = float(np.maximum(gradient * result.x, gradient * (result.x - 1.)).sum())
    rounding = 64. * np.finfo(float).eps * (1. + np.linalg.norm(matrix) + np.linalg.norm(c - a)) ** 2
    lower_distance = math.sqrt(max(0., distance ** 2 - 2. * max(0., gap) - rounding))
    radius = body.geometry.radius + obstacle.geometry.radius
    uncertainty = body.pose_uncertainty + obstacle.pose_uncertainty
    tolerance = 1e-9 * (1. + radius + distance)
    status = ("contact" if distance + uncertainty <= radius + tolerance else
              "clear" if lower_distance - uncertainty > radius + tolerance else "possible_contact")
    return {"status": status, "minimum_axis_separation": distance,
            "axis_separation_lower_bound": lower_distance, "optimality_gap_upper_bound": max(0., gap),
            "combined_radius": radius, "pose_uncertainty": uncertainty,
            "nearest_time_fraction": float(result.x[2]), "scope": "declared_rigid_translation",
            "current_poses_observed": (body.visible and obstacle.visible
                and body.pose_prediction_source is None and obstacle.pose_prediction_source is None),
            "pose_sources": (body.pose_prediction_source or body.source,
                             obstacle.pose_prediction_source or obstacle.source)}


def swept_rigid_contact(body, obstacle, end_pose, *, obstacle_end_pose=None, intervals=64):
    """Certified clearance bound for shortest-rotation, linear-translation paths.

    Distance between two occupied sets is Lipschitz in their point motion.
    The translation plus angular arc-radius speed bounds every intervening
    point; uncertain or under-resolved contact stays possible, never clear.
    The trajectory is declared, not inferred from a single camera image.
    """
    if (not isinstance(end_pose, RigidPose) or type(intervals) is not int or not 1 <= intervals <= 256
            or obstacle_end_pose is not None and not isinstance(obstacle_end_pose, RigidPose)):
        raise ValueError("rigid sweep requires bounded resolution and explicit ending poses")
    obstacle_end_pose = obstacle.pose if obstacle_end_pose is None else obstacle_end_pose

    def trajectory(object_, end):
        start_rotation = Rotation.from_quat(object_.pose.quaternion)
        turn = (Rotation.from_quat(end.quaternion) * start_rotation.inv()).as_rotvec()
        displacement = np.asarray(end.translation) - object_.pose.translation
        lever = max(np.linalg.norm(object_.geometry.start), np.linalg.norm(object_.geometry.end))
        speed = float(np.linalg.norm(displacement) + np.linalg.norm(turn) * lever)

        def at(fraction):
            rotation = Rotation.from_rotvec(fraction * turn) * start_rotation
            pose = RigidPose(tuple(np.asarray(object_.pose.translation) + fraction * displacement),
                             tuple(rotation.as_quat()))
            return replace(object_, pose=pose)

        return at, speed

    body_at, body_speed = trajectory(body, end_pose)
    obstacle_at, obstacle_speed = trajectory(obstacle, obstacle_end_pose)
    samples = [swept_capsule_contact(body_at(index / intervals), obstacle_at(index / intervals), (0., 0., 0.))
               for index in range(intervals + 1)]
    best_index = min(range(len(samples)), key=lambda index: samples[index]["minimum_axis_separation"])
    distance = samples[best_index]["minimum_axis_separation"]
    lower_distance = min(sample["axis_separation_lower_bound"] for sample in samples)
    interpolation_bound = (body_speed + obstacle_speed) / (2. * intervals)
    uncertainty = body.pose_uncertainty + obstacle.pose_uncertainty
    radius = body.geometry.radius + obstacle.geometry.radius
    tolerance = 1e-9 * (1. + radius + distance)
    status = ("contact" if distance + uncertainty <= radius + tolerance else
              "clear" if lower_distance - interpolation_bound - uncertainty > radius + tolerance else "possible_contact")
    return {"status": status, "scope": "declared_rigid_slerp_and_linear_translation",
            "sampled_minimum_axis_separation": distance, "distance_interpolation_bound": interpolation_bound,
            "continuous_separation_lower_bound": max(0., lower_distance - interpolation_bound - uncertainty),
            "combined_radius": radius, "pose_uncertainty": uncertainty, "intervals": intervals,
            "nearest_sample_time_fraction": best_index / intervals,
            "trajectory_is_observation": False}


class RigidObjectMemory:
    def __init__(self, tracks, *, objects=(), snapshot_root=None):
        self.tracks = tracks
        self.snapshot_path = (state_root() if snapshot_root is None else Path(snapshot_root)) / "rigid-object-memory.json"
        self.objects = {}
        for body in objects:
            self.observe(body)

    def observe(self, body, *, replace_intrinsics=False):
        if not isinstance(body, RigidObjectBelief) or body.pose_prediction_source is not None:
            raise ValueError("rigid observation must be measured; use apply_motion for predictions")
        if type(replace_intrinsics) is not bool or body.pose_updated_at != body.observed_at:
            raise ValueError("rigid observation cannot certify an unobserved future pose")
        track = self.tracks.resolve(body.identity)
        if track is None:
            raise ValueError("rigid geometry requires an existing tracked identity")
        body = replace(body, identity=track.track_id)
        previous = self.objects.get(body.identity)
        if previous is not None and body.observed_at < previous.pose_updated_at:
            raise ValueError("older geometry cannot overwrite a newer tracked observation")
        retain = previous is not None and previous.geometry == body.geometry and not replace_intrinsics
        landmarks = {**(dict(previous.landmarks) if retain else {}), **dict(body.landmarks)}
        properties = {**(dict(previous.surface_properties) if retain else {}), **dict(body.surface_properties)}
        provenance = dict(previous.intrinsic_sources) if retain else {}
        for category, fields in (("landmark", body.landmarks), ("surface", body.surface_properties)):
            provenance.update({f"{category}:{name}": (body.source, body.observed_at) for name in fields})
        body = replace(body, landmarks=landmarks, surface_properties=properties, intrinsic_sources=provenance)
        self.objects[body.identity] = body

    def view_changed(self, identity, *, visible=False):
        """Visibility is not a deformation, disappearance or new identity."""
        track = self.tracks.resolve(identity)
        if track is None or track.track_id not in self.objects:
            raise KeyError(identity)
        body = replace(self.objects[track.track_id], visible=visible)
        self.objects[track.track_id] = body
        return body

    def apply_motion(self, identity, transform, *, at, source, frame="world", uncertainty_growth=0.):
        """Update a rigid pose without changing its identity or intrinsic frame."""
        body = self.get(identity)
        if (body is None or not isinstance(transform, RigidPose) or frame not in {"world", "object"}
                or type(at) not in {int, float} or not math.isfinite(at) or at < body.pose_updated_at
                or not isinstance(source, str) or not source
                or type(uncertainty_growth) not in {int, float} or not math.isfinite(uncertainty_growth)
                or uncertainty_growth < 0):
            raise ValueError("rigid motion needs a tracked object, ordered time and explicit uncertainty")
        pose = transform.compose(body.pose) if frame == "world" else body.pose.compose(transform)
        updated = replace(body, pose=pose, pose_updated_at=at, pose_prediction_source=source,
                          visible=False, pose_uncertainty=body.pose_uncertainty + uncertainty_growth)
        self.objects[updated.identity] = updated
        return updated

    def get(self, identity):
        track = self.tracks.resolve(identity)
        if track is None:
            body = self.objects.get(identity)
            return replace(body, visible=False) if body is not None else None
        bodies = [body for key, body in self.objects.items()
                  if (self.tracks.resolve(key) is not None
                      and self.tracks.resolve(key).track_id == track.track_id)]
        if not bodies:
            return None
        if len({body.geometry for body in bodies}) != 1:
            raise ValueError("merged tracks disagree on intrinsic geometry; reconcile observations first")
        body = replace(max(bodies, key=lambda value: value.pose_updated_at), identity=track.track_id)
        return replace(body, visible=False) if track.state != TrackState.VISIBLE else body

    def save(self, path=None):
        if path is not None and Path(path).absolute() != self.snapshot_path.absolute():
            raise ValueError("rigid memory owns only its configured snapshot filename")
        with local_internal_governed_scope("rigid_object_memory", domain="memory_write"):
            get_file_write_gateway().write_json(self.snapshot_path, [body.to_dict() for body in self.objects.values()],
                schema_version=1, schema_name="aura.rigid_object_memory", source="rigid_object_memory")

    @classmethod
    def load(cls, path, tracks):
        if Path(path).name != "rigid-object-memory.json":
            raise ValueError("rigid memory snapshot filename differs from its owner")
        envelope = json.loads(Path(path).read_text())
        if envelope.get("schema") != "aura.rigid_object_memory" or envelope.get("schema_version") != 1:
            raise ValueError("unknown rigid object memory")
        # Tracks are restored by their owner, not fabricated from positions.
        memory = cls(tracks, snapshot_root=Path(path).parent)
        for value in envelope["payload"]:
            body = RigidObjectBelief.from_dict(value)
            if body.identity in memory.objects:
                raise ValueError("rigid object snapshot repeats a tracked identity")
            memory.objects[body.identity] = body
        return memory

    def context_referents(self, *, namespace="physical-world", scope=()):
        from core.learning.semantic_context_binding import ContextReferent

        values = []
        seen = set()
        for identity in sorted(self.objects):
            body = self.get(identity)
            if body is not None and body.identity not in seen:
                seen.add(body.identity)
                values.append(ContextReferent(namespace, body.identity, "RigidObject", body.source,
                    scope=tuple(scope), attributes={**dict(body.surface_properties),
                    "visibility": "visible" if body.visible else "occluded",
                    "frame": "object-relative", "rigidity": "declared-model",
                    "pose_basis": "prediction" if body.pose_prediction_source else "last-observation"}))
        return tuple(values)


@invariant("cognition.rigid_memory_retains_occluded_intrinsics", scope="cognition", owner=__name__)
def rigid_permanence_contract():
    from core.cognition.entity_track import Observation, TrackStore

    tracks = TrackStore()
    identity = tracks.update([Observation(1., geometry=(0., 0., 0.))])[0].track_id
    shape = CapsuleGeometry((0., 0., 0.), (0., 0., 0.), 1.)
    first = RigidObjectBelief(identity, shape, RigidPose(), "observed:first", 1.,
                             landmarks={"dot": (1., 0., 0.)}, surface_properties={"side:a": "dark"})
    memory = RigidObjectMemory(tracks, objects=(first,))
    memory.view_changed(identity)
    turn = RigidPose(quaternion=(0., 1., 0., 0.))
    memory.observe(RigidObjectBelief(identity, shape, turn, "observed:second", 2.,
                                   surface_properties={"side:b": "light"}))
    value = memory.get(identity)
    assert value.geometry == shape and value.landmarks == first.landmarks
    assert value.surface_properties == {"side:a": "dark", "side:b": "light"}
    assert value.intrinsic_sources["surface:side:a"] == ("observed:first", 1.)
    assert np.allclose(value.pose.inverse_apply(value.pose.apply(value.landmarks["dot"])), (1., 0., 0.))
    return ()
