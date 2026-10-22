"""Rigid alignment engine (module 3): scanner coordinates -> CAD coordinates.

Every operation returns a 4x4 homogeneous matrix T acting on column vectors
(x' = R x + t). The document bakes T into the mesh, features and picked points
and accumulates it into the overall scanner -> CAD transform.

3-2-1 sequence:
  1. snap_plane      a datum plane onto XY / XZ / YZ (3 DOF: 2 rotations + offset)
  2. center_on_axis  a cylinder axis through the origin  (2 DOF: in-plane translation)
  3. clock           a secondary feature onto an in-plane axis (1 DOF: rotation about the normal)
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation

from .features import CylinderFeature, Feature, PlaneFeature
from .mesh import MeshData

PLANE_NORMALS = {"XY": np.array([0.0, 0.0, 1.0]), "XZ": np.array([0.0, 1.0, 0.0]), "YZ": np.array([1.0, 0.0, 0.0])}
NORMAL_AXIS = {"XY": "Z", "XZ": "Y", "YZ": "X"}
AXES = {"X": np.array([1.0, 0.0, 0.0]), "Y": np.array([0.0, 1.0, 0.0]), "Z": np.array([0.0, 0.0, 1.0])}
MIN_PROJECTION = 0.05  # |sin| / |cos| below which a direction is considered degenerate


class AlignmentError(ValueError):
    """Raised when a constraint cannot be applied; message is user-facing."""


# ---------------------------------------------------------------------- matrix helpers


def rigid(R: np.ndarray, t=(0.0, 0.0, 0.0)) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def translation(t) -> np.ndarray:
    return rigid(np.eye(3), t)


def axis_angle(axis: np.ndarray, degrees: float) -> np.ndarray:
    """3x3 rotation about a unit axis (right-hand rule)."""
    return Rotation.from_rotvec(np.radians(degrees) * np.asarray(axis, float) / np.linalg.norm(axis)).as_matrix()


def rotation_about(axis: np.ndarray, degrees: float, pivot=(0.0, 0.0, 0.0)) -> np.ndarray:
    R = axis_angle(axis, degrees)
    p = np.asarray(pivot, float)
    return rigid(R, p - R @ p)


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal rotation taking unit vector a onto unit vector b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if c < -1.0 + 1e-12:  # antiparallel: half turn about any perpendicular
        perp = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
        return axis_angle(perp, 180.0)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def invert(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    return rigid(R.T, -R.T @ t)


def apply_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ T[:3, :3].T + T[:3, 3]


def parse_axis(name: str) -> np.ndarray:
    """'+X', '-Y', 'Z' -> signed unit vector."""
    sign = -1.0 if name.startswith("-") else 1.0
    return sign * AXES[name.lstrip("+-")]


def describe(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(translation, XYZ fixed-axis rotation angles in degrees) of a rigid transform."""
    return T[:3, 3].copy(), Rotation.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)


# ---------------------------------------------------------------------- applying transforms


def transform_mesh(mesh: MeshData, T: np.ndarray) -> MeshData:
    return MeshData(vertices=apply_points(T, mesh.vertices), faces=mesh.faces, source_path=mesh.source_path)


def transform_feature(feature: Feature, T: np.ndarray) -> Feature:
    R = T[:3, :3]
    if isinstance(feature, PlaneFeature):
        return replace(
            feature,
            point=apply_points(T, feature.point[None])[0],
            normal=R @ feature.normal,
            corners=apply_points(T, feature.corners),
        )
    axis = R @ feature.axis
    axis = axis if axis[np.argmax(np.abs(axis))] > 0 else -axis  # keep the canonical sense
    return replace(feature, center=apply_points(T, feature.center[None])[0], axis=axis)


# ---------------------------------------------------------------------- 3-2-1 constraints


def snap_plane(plane: PlaneFeature, target: str, flip: bool = False) -> np.ndarray:
    """Lay the plane onto a global plane with the material on its positive side.

    The plane's outward normal maps to -N (N = target normal), so the part sits
    "on top" of the target plane; flip=True maps it to +N instead. The rotation
    pivots on the plane centroid to keep the part near its current position.
    """
    N = PLANE_NORMALS[target]
    R = rotation_between(plane.normal, N if flip else -N)
    p = plane.point
    T = rigid(R, p - R @ p)
    return translation(-(p @ N) * N) @ T


def center_on_axis(cylinder: CylinderFeature, datum: str) -> tuple[np.ndarray, float]:
    """Translate in the datum plane so the cylinder axis pierces it at the origin.

    Returns (T, deviation of the axis from the datum normal in degrees).
    """
    N = PLANE_NORMALS[datum]
    cos = float(cylinder.axis @ N)
    if abs(cos) < MIN_PROJECTION:
        raise AlignmentError(f"{cylinder.name} lies in the {datum} plane; choose a hole perpendicular to the datum.")
    s = -(cylinder.center @ N) / cos
    pierce = cylinder.center + s * cylinder.axis
    pierce -= (pierce @ N) * N
    return translation(-pierce), float(np.degrees(np.arccos(min(1.0, abs(cos)))))


def feature_direction(feature: Feature, origin: np.ndarray) -> np.ndarray:
    """Direction a secondary feature defines: a plane's normal, or origin -> cylinder centre."""
    if isinstance(feature, PlaneFeature):
        return feature.normal
    return feature.center - origin


def clock(feature: Feature, datum: str, target_axis: str) -> tuple[np.ndarray, float]:
    """Rotate about the datum normal through the origin so the feature's in-plane
    direction points along target_axis. Returns (T, applied angle in degrees)."""
    N = PLANE_NORMALS[datum]
    target = parse_axis(target_axis)
    if abs(target @ N) > 1e-9:
        raise AlignmentError(f"{target_axis} is not in the {datum} plane.")
    full = feature_direction(feature, np.zeros(3))
    d = full - (full @ N) * N
    if np.linalg.norm(d) < max(MIN_PROJECTION * np.linalg.norm(full), 1e-9):
        raise AlignmentError(
            f"{feature.name} gives no direction within the {datum} plane "
            "(parallel to the datum normal, or the hole at the origin itself)."
        )
    d /= np.linalg.norm(d)
    angle = float(np.degrees(np.arctan2(N @ np.cross(d, target), d @ target)))
    return rotation_about(N, angle), angle
