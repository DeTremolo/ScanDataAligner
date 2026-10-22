"""Shared setup for the test suites: import path, work folder and generated test parts.

Test parts are generated on first use (deterministic, a few seconds each) into
tests/_work/, which is git-ignored: the meshes are ~10-20 MB of STL each.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_API", "pyside6")

TESTS = ROOT / "tests"
WORK = TESTS / "_work"
WORK.mkdir(exist_ok=True)
HARNESS = TESTS / "sw_mock_harness.vbs"  # mock SolidWorks API for the VBS macro (cscript)


def _box_part(path: Path) -> None:
    """100 x 60 x 20 box, ~200k evenly sized triangles."""
    import trimesh

    part = trimesh.creation.box((100, 60, 20))
    while len(part.faces) < 150_000:
        part = part.subdivide()
    part.export(path)


def _plate(path: Path) -> None:
    """100 x 60 x 20 plate (z -10..10) with a d20 through-hole at the origin, ~356k triangles."""
    import trimesh
    from shapely.geometry import Point, Polygon

    poly = Polygon([(-50, -30), (50, -30), (50, 30), (-50, 30)], [list(Point(0, 0).buffer(10, 128).exterior.coords)])
    m = trimesh.creation.extrude_polygon(poly, 20)
    m.apply_translation([0, 0, -10])
    v, f = trimesh.remesh.subdivide_to_size(m.vertices, m.faces, max_edge=2.2)
    trimesh.Trimesh(v, f).export(path)


def scan_pose():
    """The arbitrary 'scanner' pose used by the alignment suites."""
    from scipy.spatial.transform import Rotation

    from scanprofiler.core import alignment as A

    return A.rigid(Rotation.from_euler("xyz", [23, -41, 117], degrees=True).as_matrix(), [312.5, -87.25, 45.0])


def _plate_scanpose(path: Path) -> None:
    import trimesh

    from scanprofiler.core import alignment as A
    from scanprofiler.core.mesh_io import load_mesh

    cad = load_mesh(fixture("test_plate.stl"))
    trimesh.Trimesh(A.apply_points(scan_pose(), cad.vertices), cad.faces, process=False).export(path)


def _fillet_plate(path: Path) -> None:
    """100 x 60 x 20 plate (z 0..20), R5 corners, d20 hole at (-20, 0), R4 slot at x 20..30."""
    import trimesh
    from shapely.geometry import LineString, Point, box

    shape = (
        box(-45, -25, 45, 25).buffer(5, quad_segs=32)
        .difference(Point(-20, 0).buffer(10, quad_segs=64))
        .difference(LineString([(20, 0), (30, 0)]).buffer(4, quad_segs=32))
    )
    m = trimesh.creation.extrude_polygon(shape, 20)
    v, f = trimesh.remesh.subdivide_to_size(m.vertices, m.faces, max_edge=2.0)
    trimesh.Trimesh(v, f).export(path)


GENERATORS = {
    "test_part.stl": _box_part,
    "test_plate.stl": _plate,
    "test_plate_scanpose.stl": _plate_scanpose,
    "test_fillet_plate.stl": _fillet_plate,
}


def fixture(name: str) -> str:
    """Path to a generated test part, building it on first use."""
    path = WORK / name
    if not path.exists():
        print(f"[fixtures] generating {name} ...", flush=True)
        GENERATORS[name](path)
    return str(path)


def ensure_all() -> None:
    for name in GENERATORS:
        fixture(name)
