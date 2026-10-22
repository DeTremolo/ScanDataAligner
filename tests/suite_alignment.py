"""Unit tests for core.alignment + Document undo, on the plate posed like a raw scan."""
import os, sys
import _common  # noqa: E402  repo import path, QT_API, generated test parts
import numpy as np
from scipy.spatial.transform import Rotation
from scanprofiler.core import alignment as A, fitting as F
from scanprofiler.core.document import Document
from scanprofiler.core.mesh_io import load_mesh

SCR = str(_common.WORK)  # generated parts + outputs (git-ignored)
_common.ensure_all()
out = []
def check(name, cond, info=""):
    out.append(f"{'PASS' if cond else 'FAIL'}  {name}  {info}")

# primitives
rng = np.random.default_rng(3)
for _ in range(50):
    a, b = rng.normal(size=3), rng.normal(size=3)
    a /= np.linalg.norm(a); b /= np.linalg.norm(b)
    R = A.rotation_between(a, b)
    assert np.allclose(R @ a, b) and np.allclose(R.T @ R, np.eye(3)) and np.isclose(np.linalg.det(R), 1)
R = A.rotation_between(np.array([0, 0, 1.0]), np.array([0, 0, -1.0]))
check("rotation_between incl. antiparallel", np.allclose(R @ [0, 0, 1], [0, 0, -1]) and np.isclose(np.linalg.det(R), 1))
T = A.rigid(Rotation.random(random_state=1).as_matrix(), [1, 2, 3])
check("invert", np.allclose(A.invert(T) @ T, np.eye(4)))
check("rotation_about pivot fixed", np.allclose(A.apply_points(A.rotation_about([0, 0, 1], 37, [5, 6, 7]), np.array([[5, 6, 7.0]])), [5, 6, 7]))

# Ground truth: plate in CAD pose has bottom at z=-10, hole axis = Z through origin, side faces at x=+-50.
cad = load_mesh(os.path.join(SCR, "test_plate.stl"))
POSE = A.rigid(Rotation.from_euler("xyz", [23, -41, 117], degrees=True).as_matrix(), [312.5, -87.25, 45.0])
doc = Document()
doc.set_mesh(A.transform_mesh(cad, POSE))  # "scanned" pose
c = cad.face_centers()  # face ids are unchanged by transforms, so CAD-pose masks select the same faces
rad = np.linalg.norm(c[:, :2], axis=1)

def fit_plane(mask):
    f, _ = F.fit_plane_region(doc.mesh, np.flatnonzero(mask), tol=0.2, rng=0)
    return doc.add_feature(f)

bottom = fit_plane(c[:, 2] < -9.99)
hole, _ = F.fit_cylinder_region(doc.mesh, np.flatnonzero((rad < 10.01) & (c[:, 1] > 0)), tol=0.2, rng=0)
hole = doc.add_feature(hole)
side = fit_plane(np.abs(c[:, 0] - 50) < 0.01)

# Step 1: bottom face on XY, material on +Z
T1 = A.snap_plane(doc.feature(bottom.id), "XY")
doc.apply_transform(T1, "Snap", datum="XY", step=("primary", "Plane 1 -> XY"))
b = doc.feature(bottom.id)
zmin, zmax = doc.mesh.bounds[0][2], doc.mesh.bounds[1][2]
check("step1: datum normal -Z, on z=0", np.allclose(b.normal, [0, 0, -1], atol=1e-9) and abs(b.point[2]) < 1e-9, f"n={np.round(b.normal, 9)}")
check("step1: part sits on +Z side", abs(zmin) < 1e-6 and abs(zmax - 20) < 1e-6, f"z {zmin:.6f}..{zmax:.6f}")
T1f = A.snap_plane(doc.feature(bottom.id), "XY", flip=True)
check("flip puts material on -Z", np.isclose(A.apply_points(T1f, doc.mesh.vertices)[:, 2].min(), -20, atol=1e-6))

# Step 2: hole axis through origin
T2, dev = A.center_on_axis(doc.feature(hole.id), "XY")
doc.apply_transform(T2, "Centre", step=("secondary", "Hole 1 at origin"))
h = doc.feature(hole.id)
check("step2: hole axis through origin", np.linalg.norm(h.center[:2]) < 1e-6 and dev < 1e-3, f"centre {np.round(h.center, 7)} dev {dev:.5f} deg")

# Step 3: clock the side face normal onto +X
T3, ang = A.clock(doc.feature(side.id), "XY", "+X")
doc.apply_transform(T3, "Clock", step=("tertiary", "Plane 2 -> +X"))
s = doc.feature(side.id)
check("step3: side normal = +X at x=50", np.allclose(s.normal, [1, 0, 0], atol=1e-9) and abs(s.point[0] - 50) < 1e-6, f"rotated {ang:.4f} deg")

# Final pose must equal ground truth shifted by +10 in Z
err = np.abs(doc.mesh.vertices - (cad.vertices + [0, 0, 10])).max()
check("3-2-1 recovers CAD pose exactly", err < 1e-6, f"max vertex error {err:.2e} mm")
dT = np.abs(doc.transform @ POSE - A.translation([0, 0, 10])).max()
check("accumulated transform = truth", dT < 1e-6, f"max element error {dT:.1e}")
check("alignment state", doc.alignment.datum == "XY" and list(doc.alignment.steps) == ["primary", "secondary", "tertiary"])

# Errors
try:
    A.clock(doc.feature(hole.id), "XY", "+X"); check("clock with primary hole raises", False)
except A.AlignmentError as e:
    check("clock with primary hole raises", True, str(e)[:60])
try:
    A.clock(doc.feature(bottom.id), "XY", "+X"); check("clock with datum plane raises", False)
except A.AlignmentError:
    check("clock with datum plane raises", True)
try:
    A.clock(doc.feature(side.id), "XY", "+Z"); check("out-of-plane target raises", False)
except A.AlignmentError:
    check("out-of-plane target raises", True)

# Incremental shifts + feature added after transforms survives undo
doc.apply_transform(A.translation([1.5, 0, 0]), "Move X")
doc.apply_transform(A.rotation_about([0, 0, 1], 90), "Rot Z")
probe = doc.add_feature(F.fit_plane_region(doc.mesh, np.flatnonzero(c[:, 2] > 9.99), tol=0.2, rng=0)[0])
check("shift + rot: side normal now +Y", np.allclose(doc.feature(side.id).normal, [0, 1, 0], atol=1e-9))
labels = []
while doc.can_undo:
    labels.append(doc.undo())
check("undo order", labels == ["Rot Z", "Move X", "Clock", "Centre", "Snap"], str(labels))
err = np.abs(doc.mesh.vertices - A.transform_mesh(cad, POSE).vertices).max()
check("undo restores scanner pose exactly", err == 0.0 and np.allclose(doc.transform, np.eye(4)) and doc.alignment.datum is None, f"{err}")
p = doc.feature(probe.id)
pose_top = A.apply_points(POSE, np.array([[0, 0, 10.0]]))[0]
check("later feature carried back through undo", p is not None and abs((pose_top - p.point) @ p.normal) < 1e-6)

# Reset = inverse of accumulated transform
doc.apply_transform(T1 := A.snap_plane(doc.feature(bottom.id), "XY"), "Snap", datum="XY")
doc.apply_transform(A.invert(doc.transform), "Reset", reset_alignment=True)
err = np.abs(doc.mesh.vertices - A.transform_mesh(cad, POSE).vertices).max()
check("reset returns to scanner coords", err < 1e-9 and doc.alignment.datum is None, f"{err:.1e}")

print("\n".join(out)); print(f"{sum(o.startswith('FAIL') for o in out)} failures")
