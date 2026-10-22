"""Triangle mesh container: plain NumPy arrays plus vectorised derived quantities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class MeshData:
    vertices: np.ndarray  # (N, 3) float64
    faces: np.ndarray  # (M, 3) int64, triangle vertex indices
    source_path: Path | None = None

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def mean_edge_length(self) -> float:
        tri = self.vertices[self.faces]
        return float(np.linalg.norm(tri - np.roll(tri, 1, axis=1), axis=2).mean())

    def face_centers(self) -> np.ndarray:
        return self.vertices[self.faces].mean(axis=1)

    def face_normals(self) -> np.ndarray:
        tri = self.vertices[self.faces]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        length = np.linalg.norm(n, axis=1, keepdims=True)
        return np.divide(n, length, out=np.zeros_like(n), where=length > 0)

    def smooth_face_normals(self) -> np.ndarray:
        """Face normals averaged over their vertices' area-weighted normals.

        Robust to scan noise and sliver triangles, whose raw normals are unreliable.
        """
        tri = self.vertices[self.faces]
        area_n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])  # length = 2 * area
        idx = self.faces.ravel()
        vn = np.column_stack(
            [np.bincount(idx, weights=np.repeat(area_n[:, k], 3), minlength=self.n_vertices) for k in range(3)]
        )
        n = vn[self.faces].sum(axis=1)
        length = np.linalg.norm(n, axis=1, keepdims=True)
        return np.divide(n, length, out=np.zeros_like(n), where=length > 0)
