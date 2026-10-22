"""Synthetic accuracy tests for core.fitting (no GUI)."""
import sys, time
import _common  # noqa: E402  repo import path, QT_API, generated test parts
import numpy as np
import trimesh
from scanprofiler.core.document import MeshData
from scanprofiler.core import fitting as F

rng = np.random.default_rng(1)
out = []
def check(name, cond, info=""):
    out.append(f"{'PASS' if cond else 'FAIL'}  {name}  {info}")

def mesh_of(tm, noise=0.0):
    v = np.asarray(tm.vertices, float).copy()
    if noise:
        v += rng.normal(0, noise, v.shape)
    return MeshData(v, np.asarray(tm.faces, np.int64))

# --- hole: annulus (tube) R_in=10 R_out=25 h=30, axis tilted
T = trimesh.transformations.rotation_matrix(0.4, [1, 1, 0])
T[:3, 3] = [5, -3, 2]
# Evenly meshed tube (~0.3 mm triangles, like a scan): structured walls + caps
import pyvista as pv
grid = pv.CylinderStructured(radius=np.linspace(10, 25, 50), height=30, theta_resolution=210, z_resolution=100, direction=(0, 0, 1))
surf = grid.extract_surface(algorithm=None).triangulate().clean()
tube = trimesh.Trimesh(np.asarray(surf.points), surf.regular_faces)
tube.fix_normals()
print("tube faces", len(tube.faces), "mean edge", round(MeshData(np.asarray(tube.vertices), np.asarray(tube.faces)).mean_edge_length(), 3))
tube.apply_transform(T)
true_axis = T[:3, :3] @ [0, 0, 1]
true_c = T[:3, 3]
for noise in (0.0, 0.02):
    m = mesh_of(tube, noise)
    c = m.face_centers()
    rel = c - true_c
    rad = np.linalg.norm(rel - np.outer(rel @ true_axis, true_axis), axis=1)
    inner = np.flatnonzero((rad < 10.05) & (np.abs(rel @ true_axis) < 14.9))  # wall only
    # half the inner wall + overspray onto the top annular face near the edge
    ang = np.arctan2(*(rel[inner] @ np.c_[np.cross(true_axis, [1, 0, 0]), np.cross(true_axis, np.cross(true_axis, [1, 0, 0]))]).T)
    region = inner[ang > 0]
    top = np.flatnonzero((np.abs(rel @ true_axis - 15) < 0.01) & (rad < 12))
    region = np.concatenate([region, top])
    t = time.perf_counter()
    feat, inl = F.fit_cylinder_region(m, region, tol=0.1 + 3 * noise, rng=0)
    dt = time.perf_counter() - t
    ang_err = np.degrees(np.arccos(min(1, abs(feat.axis @ true_axis))))
    off = feat.center - true_c
    off_err = np.linalg.norm(off - (off @ true_axis) * true_axis)
    check(f"hole noise={noise}: radius", abs(feat.radius - 10) < 0.01 + noise, f"R={feat.radius:.5f}")
    check(f"hole noise={noise}: axis angle", ang_err < 0.05 + 2 * noise, f"{ang_err:.4f} deg")
    check(f"hole noise={noise}: axis offset", off_err < 0.01 + noise, f"{off_err:.5f}")
    check(f"hole noise={noise}: kind", feat.kind == "hole")
    check(f"hole noise={noise}: overspray rejected", not np.isin(top, inl).any() and len(inl) > 0.9 * (len(region) - len(top)),
          f"{len(inl)}/{len(region)-len(top)} wall faces, {np.isin(top, inl).sum()} top")
    check(f"hole noise={noise}: fast", dt < 2.0, f"{dt:.2f}s, {len(region)} faces")
    # boss: outer wall
    outer = np.flatnonzero(rad > 24)
    feat_b, _ = F.fit_cylinder_region(m, outer, tol=0.1 + 3 * noise, rng=0)
    check(f"boss noise={noise}: radius+kind", abs(feat_b.radius - 25) < 0.02 + noise and feat_b.kind == "boss",
          f"R={feat_b.radius:.5f} {feat_b.kind}")

# --- plane: top face of a subdivided box, with overspray onto a side and a stray island
box = trimesh.creation.box((100, 60, 20))
for _ in range(6):
    box = box.subdivide()
for noise in (0.0, 0.02):
    m = mesh_of(box, noise)
    c, n = m.face_centers(), m.face_normals()
    top = np.flatnonzero((c[:, 2] > 9.9) & (np.abs(c[:, 0]) < 30))
    side = np.flatnonzero((np.abs(c[:, 1] - 30) < 0.1) & (c[:, 2] > 7) & (np.abs(c[:, 0]) < 30))
    island = np.flatnonzero((c[:, 2] < -9.9) & (np.abs(c[:, 0] - 40) < 2) & (np.abs(c[:, 1]) < 2))
    region = np.concatenate([top, side, island])
    keep, n_clusters = F.largest_cluster(c[region], eps=3 * (m).mean_edge_length(), min_samples=10)
    check(f"dbscan noise={noise}: island removed", not np.isin(island, region[keep]).any() and np.isin(top, region[keep]).all(),
          f"{n_clusters} clusters, kept {keep.sum()}/{len(region)}")
    feat, inl = F.fit_plane_region(m, region[keep], tol=0.1 + 3 * noise, rng=0)
    check(f"plane noise={noise}: normal", feat.normal @ [0, 0, 1] > 0.99999, str(np.round(feat.normal, 6)))
    check(f"plane noise={noise}: offset", abs(feat.point[2] - 10) < 0.005, f"z={feat.point[2]:.5f}")
    check(f"plane noise={noise}: side overspray rejected", not np.isin(side, inl).any())
    check(f"plane noise={noise}: rms ~ noise", abs(feat.rms - noise) < 0.005 + 0.3 * noise, f"rms={feat.rms:.4f}")

# --- landmark plane: 4 points on a tilted plane, normal oriented by surface
m = mesh_of(box)
pts = np.array([[-40, -20, -10], [40, -20, -10], [40, 20, -10], [-40, 25, -10.0]])
f = F.fit_landmark_plane(pts, m)
check("landmark bottom plane points outward (-Z)", f.normal @ [0, 0, -1] > 0.9999 and f.rms < 1e-9, str(np.round(f.normal, 6)))
try:
    F.fit_landmark_plane(pts[:2], m)
    check("landmark <3 points raises", False)
except F.FitError:
    check("landmark <3 points raises", True)
try:
    F.fit_cylinder_region(m, np.flatnonzero(m.face_centers()[:, 2] > 9.9), tol=0.1, rng=0)
    check("cylinder on flat region raises FitError", False)
except F.FitError as e:
    check("cylinder on flat region raises FitError", True, str(e))

print("\n".join(out))
print(f"{sum(o.startswith('FAIL') for o in out)} failures")

