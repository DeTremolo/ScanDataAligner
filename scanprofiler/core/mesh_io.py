"""Mesh import (module 1). Exporters will live beside this in core/exporters/."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from .mesh import MeshData

IMPORT_FILTER = "Mesh files (*.stl *.obj);;STL (*.stl);;Wavefront OBJ (*.obj);;All files (*)"


def load_mesh(path: str | Path) -> MeshData:
    """Load ASCII/binary STL or OBJ into contiguous NumPy arrays.

    trimesh auto-detects ASCII vs binary STL. force="mesh" flattens multi-body
    OBJ scenes into one mesh; process=True merges duplicate STL vertices so the
    mesh is connected (required later for boundary-loop sorting in module 4).
    """
    path = Path(path)
    tm = trimesh.load(path, force="mesh", process=True)
    if not isinstance(tm, trimesh.Trimesh) or len(tm.faces) == 0:
        raise ValueError(f"No triangle geometry found in {path.name}")

    # TODO(module 2): optional open3d cleanup pass (remove degenerate / duplicated
    # triangles, non-manifold edges) before fitting and slicing.
    return MeshData(
        vertices=np.ascontiguousarray(tm.vertices, dtype=np.float64),
        faces=np.ascontiguousarray(tm.faces, dtype=np.int64),
        source_path=path,
    )


# TODO(module 1): export_solidworks_macro(sections, path)  -> .vbs (SW2021+ compatible)
# TODO(module 1): export_step_curves(sections, path)       -> AP214 GEOMETRIC_CURVE_SET
# TODO(module 1): export_dxf_profiles(sections, path)      -> closed SPLINE/ARC/CIRCLE + center marks
