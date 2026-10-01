"""The object frame, not its visible projection, owns intrinsic geometry."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from core.cognition.entity_track import Observation, TrackStore
from core.cognition.rigid_object_memory import (
    CapsuleGeometry,
    RigidObjectBelief,
    RigidObjectMemory,
    RigidPose,
    rigid_permanence_contract,
    swept_capsule_contact,
    swept_rigid_contact,
)
from core.learning.semantic_context_binding import BindingContext, BindingRole, bind_context_roles


def body(identity, *, geometry=None, pose=None):
    return RigidObjectBelief(identity, geometry or CapsuleGeometry((0., 0., 0.), (0., 0., 0.), 1.),
        pose or RigidPose(), "camera:frame1", 1., landmarks={"dot": (1., 0., 0.)},
        surface_properties={"previously_seen_hemisphere": "dark"})


def test_dot_object_coordinate_does_not_change_under_rotation_or_viewpoint():
    sphere = body("sphere")
    rotation = RigidPose(quaternion=tuple(Rotation.from_euler("y", 180, degrees=True).as_quat()))
    turned = replace(sphere, pose=rotation)
    assert turned.landmarks["dot"] == sphere.landmarks["dot"]
    assert np.allclose(turned.pose.inverse_apply(turned.pose.apply(sphere.landmarks["dot"])), (1., 0., 0.))
    observer = RigidPose((5., 2., -3.), tuple(Rotation.from_euler("z", 90, degrees=True).as_quat()))
    assert np.allclose(observer.apply(turned.landmark_in_view("dot", observer)), turned.pose.apply((1., 0., 0.)))
    assert turned.geometry == sphere.geometry


def test_registered_contract_covers_retention_not_automatic_vision_recognition():
    assert rigid_permanence_contract() == ()


def test_occlusion_retains_shape_and_observed_appearance_without_certifying_pose():
    tracks = TrackStore()
    identity = tracks.update([Observation(1., geometry=(0., 0., 0.))])[0].track_id
    memory = RigidObjectMemory(tracks, objects=(body(identity),))
    hidden = memory.view_changed(identity)
    assert not hidden.visible and hidden.geometry == memory.objects[identity].geometry
    assert hidden.surface_properties["previously_seen_hemisphere"] == "dark"
    assert "unseen_hemisphere" not in hidden.surface_properties
    tracks.update([])
    referent = memory.context_referents(namespace="scene1")[0]
    assert referent.identity == identity and referent.attributes["visibility"] == "occluded"
    context = BindingContext((referent,), {"RigidObject": None})
    role = BindingRole("it", "theme", "RigidObject", referents=(referent.key,))
    assert bind_context_roles(context, (role,)).executable


def test_long_extent_still_contacts_tree_when_observer_sees_only_an_endpoint():
    pole = body("pole", geometry=CapsuleGeometry((-3., 0., 0.), (3., 0., 0.), .1))
    tree = body("tree", geometry=CapsuleGeometry((0., -2., 0.), (0., 2., 0.), .1),
                pose=RigidPose((-3.3, 0., 0.)))
    assert swept_capsule_contact(pole, tree, (0., 0., 0.))["status"] == "clear"
    pole = replace(pole, visible=False)
    receipt = swept_capsule_contact(pole, tree, (-.5, 0., 0.))
    assert receipt["status"] == "contact" and not receipt["current_poses_observed"]
    assert pole.geometry.length == pytest.approx(6.2)
    uncertain = replace(pole, pose_uncertainty=1.)
    assert swept_capsule_contact(uncertain, tree, (-.5, 0., 0.))["status"] == "possible_contact"


def test_snapshot_and_track_eviction_do_not_erase_measured_dimensions(tmp_path):
    tracks = TrackStore()
    identity = tracks.update([Observation(1., geometry=(0., 0., 0.))])[0].track_id
    memory = RigidObjectMemory(tracks, objects=(body(identity),), snapshot_root=tmp_path)
    path = tmp_path / "rigid-object-memory.json"
    memory.save(path)
    restored = RigidObjectMemory.load(path, TrackStore())
    assert restored.get(identity).geometry == memory.get(identity).geometry
    assert not restored.get(identity).visible


def test_camera_turn_is_not_deformation_but_new_observation_can_change_geometry():
    tracks = TrackStore()
    identity = tracks.update([Observation(1., geometry=(0., 0., 0.))])[0].track_id
    original = body(identity)
    memory = RigidObjectMemory(tracks, objects=(original,))
    memory.view_changed(identity)
    changed = replace(original, geometry=CapsuleGeometry((0., 0., 0.), (0., 0., 0.), .5),
                      observed_at=2., pose_updated_at=2., source="camera:frame2:deformation")
    memory.observe(changed)
    assert memory.get(identity).geometry.radius == .5
    with pytest.raises(ValueError, match="older"):
        memory.observe(original)


def test_motion_prediction_preserves_intrinsics_and_does_not_become_an_observation():
    tracks = TrackStore()
    identity = tracks.update([Observation(1., geometry=(0., 0., 0.))])[0].track_id
    original = body(identity)
    memory = RigidObjectMemory(tracks, objects=(original,))
    transform = RigidPose((0., 0., 3.), tuple(Rotation.from_euler("y", 180, degrees=True).as_quat()))
    predicted = memory.apply_motion(identity, transform, at=2., source="motor:declared-action", uncertainty_growth=.2)
    assert predicted.geometry == original.geometry and predicted.landmarks == original.landmarks
    assert predicted.observed_at == 1. and predicted.pose_updated_at == 2.
    assert not predicted.visible and predicted.pose_prediction_source
    referent = memory.context_referents()[0]
    assert referent.attributes["pose_basis"] == "prediction"
    with pytest.raises(ValueError, match="measured"):
        memory.observe(predicted)
    with pytest.raises(ValueError, match="older"):
        memory.observe(original)


def test_rotating_hidden_pole_cannot_tunnel_through_a_tree_between_samples():
    pole = body("pole", geometry=CapsuleGeometry((-3., 0., 0.), (3., 0., 0.), .1))
    tree = body("tree", geometry=CapsuleGeometry((0., -1., 0.), (0., 1., 0.), .1), pose=RigidPose((0., 0., 2.)))
    end = RigidPose(quaternion=tuple(Rotation.from_euler("y", 180, degrees=True).as_quat()))
    assert swept_rigid_contact(pole, tree, end, intervals=1)["status"] == "possible_contact"
    assert swept_rigid_contact(pole, tree, end, intervals=2)["status"] == "contact"
    far_tree = replace(tree, pose=RigidPose((0., 0., 20.)))
    assert swept_rigid_contact(pole, far_tree, end, intervals=2)["status"] == "clear"


def test_observed_property_cannot_overwrite_visibility_or_provenance_fields():
    tracks = TrackStore()
    identity = tracks.update([Observation(1., geometry=(0., 0., 0.))])[0].track_id
    value = replace(body(identity), surface_properties={"visibility": "visible", "pose_basis": "certainty"},
                    intrinsic_sources={})
    memory = RigidObjectMemory(tracks, objects=(value,))
    memory.view_changed(identity)
    reference = memory.context_referents()[0]
    assert reference.attributes["visibility"] == "occluded" and reference.attributes["pose_basis"] != "certainty"


def test_partial_new_view_retains_old_hemisphere_and_original_measurement(tmp_path):
    tracks = TrackStore()
    identity = tracks.update([Observation(1., geometry=(0., 0., 0.))])[0].track_id
    memory = RigidObjectMemory(tracks, objects=(body(identity),), snapshot_root=tmp_path)
    observation = RigidObjectBelief(identity, body(identity).geometry,
        RigidPose(quaternion=tuple(Rotation.from_euler("y", 180, degrees=True).as_quat())),
        "camera:frame2", 2., surface_properties={"opposite_hemisphere": "light"})
    memory.observe(observation)
    retained = memory.get(identity)
    assert retained.surface_properties == {"previously_seen_hemisphere": "dark", "opposite_hemisphere": "light"}
    assert retained.landmarks["dot"] == (1., 0., 0.)
    assert retained.intrinsic_sources["surface:previously_seen_hemisphere"] == ("camera:frame1", 1.)
    assert retained.intrinsic_sources["surface:opposite_hemisphere"] == ("camera:frame2", 2.)
    path = tmp_path / "rigid-object-memory.json"
    memory.save(path)
    assert RigidObjectMemory.load(path, tracks).get(identity).intrinsic_sources == retained.intrinsic_sources
    memory.observe(observation, replace_intrinsics=True)
    assert "previously_seen_hemisphere" not in memory.get(identity).surface_properties
    with pytest.raises(ValueError, match="configured snapshot"):
        memory.save(tmp_path / "unowned.json")


def test_measured_shape_change_does_not_copy_unknown_old_surface_coordinates():
    tracks = TrackStore()
    identity = tracks.update([Observation(1., geometry=(0., 0., 0.))])[0].track_id
    memory = RigidObjectMemory(tracks, objects=(body(identity),))
    changed = RigidObjectBelief(identity, CapsuleGeometry((0., 0., 0.), (0., 0., 0.), .5),
        RigidPose(), "camera:deformation", 2.)
    memory.observe(changed)
    assert not memory.get(identity).landmarks and not memory.get(identity).surface_properties
    with pytest.raises(ValueError, match="future pose"):
        memory.observe(replace(changed, pose_updated_at=3.))


def test_clearance_uses_dual_bound_not_only_feasible_solver_distance(monkeypatch):
    from types import SimpleNamespace

    import core.cognition.rigid_object_memory as geometry

    pole = body("pole", geometry=CapsuleGeometry((-3., 0., 0.), (3., 0., 0.), .1))
    tree = body("tree", geometry=CapsuleGeometry((0., -2., 0.), (0., 2., 0.), .1))
    monkeypatch.setattr(geometry, "lsq_linear", lambda *args, **kwargs:
        SimpleNamespace(success=True, x=np.zeros(3)))
    receipt = swept_capsule_contact(pole, tree, (0., 0., 0.))
    assert receipt["minimum_axis_separation"] > receipt["combined_radius"]
    assert receipt["status"] == "possible_contact"
    assert receipt["axis_separation_lower_bound"] == 0.
