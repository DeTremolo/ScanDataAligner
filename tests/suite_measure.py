import sys
import _common  # noqa: E402  repo import path, QT_API, generated test parts
import numpy as np
from scanprofiler.core.features import CylinderFeature, PlaneFeature
from scanprofiler.core.measure import measure, snap

out = []
def check(name, cond, info=""):
    out.append(f"{'PASS' if cond else 'FAIL'}  {name}  {info}")

def plane(point, normal, name, size=50.0):
    n = np.asarray(normal, float); n /= np.linalg.norm(n)
    u = np.cross(n, [1, 0, 0] if abs(n[0]) < 0.9 else [0, 1, 0]); u /= np.linalg.norm(u); v = np.cross(n, u)
    p = np.asarray(point, float)
    corners = np.array([p + a * u + b * v for a, b in ((-size, -size), (size, -size), (size, size), (-size, size))])
    return PlaneFeature(p, n, corners, 0, 0, 100, "ransac", name=name)

def cyl(center, axis, r, name, length=20.0):
    a = np.asarray(axis, float); a /= np.linalg.norm(a)
    return CylinderFeature(np.asarray(center, float), a, r, length, 0, 100, "hole", name=name)

top = plane([0, 0, 10], [0, 0, 1], "Top")
bottom = plane([3, -2, -10], [0, 0, -1], "Bottom")
side = plane([50, 0, 0], [1, 0, 0], "Side")
h1 = cyl([0, 0, 0], [0, 0, 1], 10, "Hole 1")
h2 = cyl([30, 40, 2], [0, 0, 1], 5, "Hole 2")

r = measure(top, bottom)
check("plane-plane parallel gap 20", abs(r.distance - 20) < 1e-12 and np.allclose(r.deltas, [0, 0, 20]) and r.extras["angle deviation (deg)"] < 1e-9, r.label)
r = measure(top, side)
check("plane-plane perpendicular -> angle 90", abs(r.extras["angle (deg)"] - 90) < 1e-9, r.label)
r = measure(h1, h2)
check("hole-hole centre distance 50 (3-4-5)", abs(r.distance - 50) < 1e-12 and np.allclose(r.deltas, [30, 40, 0]), f"{r.distance} {r.deltas}")
check("  min wall clearance 35 / max 65", abs(r.extras["min wall clearance"] - 35) < 1e-12 and abs(r.extras["max wall distance"] - 65) < 1e-12)
r2 = measure(h2, h1)
check("order independent", abs(r2.distance - 50) < 1e-12)
r = measure(bottom, h2)
check("plane-hole: perpendicular to centre 12", abs(r.distance - 12) < 1e-12 and np.allclose(r.deltas, [0, 0, 12]), r.label)
r = measure(side, h1)
check("side plane to hole axis (parallel): 50, walls 40/60", abs(r.distance - 50) < 1e-12
      and abs(r.extras["min wall distance"] - 40) < 1e-12 and abs(r.extras["max wall distance"] - 60) < 1e-12, r.extras)
r = measure(np.array([1.0, 2, 3]), np.array([4.0, -2, 15]))
check("point-point: 3D = 13, deltas 3/4/12", abs(r.distance - 13) < 1e-12 and np.allclose(r.deltas, [3, 4, 12]))
r = measure(np.array([5.0, 5, 30]), top)
check("point-plane 20", abs(r.distance - 20) < 1e-12)
r = measure(np.array([20.0, 0, 3]), h1)
check("point-hole: 20 to axis, gap 10", abs(r.distance - 20) < 1e-12 and abs(r.extras["radial gap to surface"] - 10) < 1e-12)
skew = cyl([0, 0, 25], [1, 0, 0], 2, "Cross")
r = measure(h1, skew)
check("skew axes: common perpendicular 0 (intersecting), angle 90", r.distance < 1e-12 and abs(r.extras["axis angle (deg)"] - 90) < 1e-9)
skew2 = cyl([0, 7, 25], [1, 0, 0], 2, "Cross2")
r = measure(h1, skew2)
check("skew axes offset 7", abs(r.distance - 7) < 1e-12, f"{r.distance}")

# snapping
feats = [top, bottom, h1, h2]
check("snap: point on hole wall -> Hole 1", snap(np.array([10.0, 0, 5]), feats, 0.1) is h1)
check("snap: point on top face -> Top", snap(np.array([30.0, -30, 10.02]), feats, 0.1) is top)
check("snap: point off everything -> None", snap(np.array([30.0, -30, 5]), feats, 0.1) is None)
check("snap: beyond hole length -> not hole", snap(np.array([10.0, 0, 40]), [h1], 0.1) is None)
check("snap: outside plane patch -> None", snap(np.array([300.0, 0, 10]), [top], 0.1) is None)
print("\n".join(out)); print(f"{sum(o.startswith('FAIL') for o in out)} failures")
