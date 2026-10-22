"""Screen-space selection math (module 2). Pure NumPy: no VTK, no Qt.

Display coordinates follow VTK: physical pixels, origin at the bottom-left.
"""

from __future__ import annotations

import numpy as np
from matplotlib.path import Path as _MplPath

from .mesh import MeshData


def project_to_display(
    points: np.ndarray, mvp: np.ndarray, viewport: tuple[float, float, float, float], return_depth: bool = False
):
    """Project (N, 3) world points to (N, 2) display pixels.

    mvp is the camera's world -> NDC matrix; viewport is (x0, y0, width, height)
    in pixels. Returns (xy, valid) where valid excludes points behind the camera
    or outside the near/far clipping range; with return_depth, (xy, ndc_z, valid).
    """
    x0, y0, w, h = viewport
    clip = points @ mvp[:, :3].T + mvp[:, 3]
    cw = clip[:, 3]
    valid = cw > 1e-12
    ndc = clip[:, :3] / np.where(valid, cw, 1.0)[:, None]
    valid &= np.abs(ndc[:, 2]) <= 1.0
    xy = np.empty((len(points), 2))
    xy[:, 0] = x0 + (ndc[:, 0] + 1.0) * 0.5 * w
    xy[:, 1] = y0 + (ndc[:, 1] + 1.0) * 0.5 * h
    if return_depth:
        return xy, ndc[:, 2], valid
    return xy, valid


def eye_depth(ndc_z: np.ndarray, near: float, far: float, parallel: bool) -> np.ndarray:
    """Convert NDC depth in [-1, 1] to linear distance from the camera plane."""
    if parallel:
        return near + 0.5 * (ndc_z + 1.0) * (far - near)
    return 2.0 * near * far / ((far + near) - ndc_z * (far - near))


class Region:
    """A 2D screen region. bbox() bounds the pixel query, contains() refines it."""

    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def contains(self, xy: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class RectRegion(Region):
    def __init__(self, p0, p1) -> None:
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        self.lo, self.hi = np.minimum(p0, p1), np.maximum(p0, p1)

    def bbox(self):
        return self.lo, self.hi

    def contains(self, xy):
        return np.all((xy >= self.lo) & (xy <= self.hi), axis=1)


class CircleRegion(Region):
    def __init__(self, center, radius: float) -> None:
        self.center = np.asarray(center, float)
        self.radius = float(radius)

    def bbox(self):
        return self.center - self.radius, self.center + self.radius

    def contains(self, xy):
        d = xy - self.center
        return np.einsum("ij,ij->i", d, d) <= self.radius**2


class CapsuleRegion(Region):
    """Brush stroke segment a -> b swept by a circle; keeps fast drags gap-free."""

    def __init__(self, a, b, radius: float) -> None:
        self.a, self.b = np.asarray(a, float), np.asarray(b, float)
        self.radius = float(radius)

    def bbox(self):
        return np.minimum(self.a, self.b) - self.radius, np.maximum(self.a, self.b) + self.radius

    def contains(self, xy):
        ab = self.b - self.a
        l2 = ab @ ab
        if l2 < 1e-12:
            d = xy - self.a
        else:
            t = np.clip((xy - self.a) @ ab / l2, 0.0, 1.0)
            d = xy - (self.a + t[:, None] * ab)
        return np.einsum("ij,ij->i", d, d) <= self.radius**2


class PolygonRegion(Region):
    """Freeform lasso; implicitly closed."""

    def __init__(self, vertices) -> None:
        self.vertices = np.asarray(vertices, float)
        self._path = _MplPath(self.vertices)

    def bbox(self):
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def contains(self, xy):
        return self._path.contains_points(xy)


def delete_faces(mesh: MeshData, face_mask: np.ndarray) -> MeshData:
    """Return a new mesh without the masked faces; orphaned vertices are dropped."""
    faces = mesh.faces[~face_mask]
    used = np.zeros(mesh.n_vertices, dtype=bool)
    used[faces.ravel()] = True
    remap = np.cumsum(used) - 1
    return MeshData(vertices=mesh.vertices[used], faces=remap[faces], source_path=mesh.source_path)
