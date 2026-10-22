"""Synthetic tests for core.profile segmentation."""
import sys, time
import _common  # noqa: E402  repo import path, QT_API, generated test parts
import numpy as np
from scanprofiler.core.profile import segment_profile

rng = np.random.default_rng(7)
out = []
def check(name, cond, info=""):
    out.append(f"{'PASS' if cond else 'FAIL'}  {name}  {info}")

def resample(poly_fn, total, spacing, jitter=0.3):
    """Irregular arc-length sampling of a parametric outline (like mesh-edge crossings)."""
    s, ts = 0.0, []
    while s < total:
        ts.append(s)
        s += spacing * (1 + jitter * rng.uniform(-1, 1))
    return np.array([poly_fn(t) for t in ts])

def rounded_rect(w, h, r):
    straight = [w - 2 * r, h - 2 * r, w - 2 * r, h - 2 * r]
    quarter = np.pi * r / 2
    total = sum(straight) + 4 * quarter
    corners = [(w / 2 - r, -h / 2 + r), (w / 2 - r, h / 2 - r), (-w / 2 + r, h / 2 - r), (-w / 2 + r, -h / 2 + r)]
    starts = [(-w / 2 + r, -h / 2), (w / 2, -h / 2 + r), (w / 2 - r, h / 2), (-w / 2, h / 2 - r)]
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    def f(t):
        for k in range(4):
            if t < straight[k]:
                return np.array(starts[k]) + t * np.array(dirs[k])
            t -= straight[k]
            if t < quarter:
                a = -np.pi / 2 + k * np.pi / 2 + t / r
                return np.array(corners[k]) + r * np.array([np.cos(a), np.sin(a)])
            t -= quarter
        return np.array(starts[0])
    return f, total

def g0(segs, closed):
    gaps = [np.linalg.norm(segs[k].end - segs[k + 1].start) for k in range(len(segs) - 1)]
    if closed:
        gaps.append(np.linalg.norm(segs[-1].end - segs[0].start))
    return max(gaps)

def kinds(segs):
    return "".join(s.kind[0].upper() for s in segs)

TOL = 0.02
for noise in (0.0, 0.004):
    f, total = rounded_rect(100, 60, 5)
    pts = resample(f, total, 0.8)
    pts = pts + rng.normal(0, noise, pts.shape)
    t = time.perf_counter()
    segs = segment_profile(pts, True, TOL)
    dt = time.perf_counter() - t
    lines = [s for s in segs if s.kind == "line"]
    arcs = [s for s in segs if s.kind == "arc"]
    check(f"rounded rect noise={noise}: 4 lines + 4 arcs", len(lines) == 4 and len(arcs) == 4 and len(segs) == 8, f"{kinds(segs)} in {dt*1000:.0f} ms ({len(pts)} pts)")
    if len(arcs) == 4:
        check(f"  fillet radii 5", all(abs(a.radius - 5) < 0.02 for a in arcs), str([round(a.radius, 4) for a in arcs]))
        sweep_tol = 0.5 if noise == 0 else 1.0  # noisy: tangent points of a 10-point arc
        check(f"  arc sweeps 90 deg", all(abs(a.sweep_deg - 90) < sweep_tol for a in arcs), str([round(a.sweep_deg, 2) for a in arcs]))
        L = sorted(round(float(np.linalg.norm(l.end - l.start)), 3) for l in lines)
        check(f"  line lengths 50,50,90,90", len(L) == 4 and np.allclose(L, [50, 50, 90, 90], atol=0.05), str(L))
    else:
        out.append("      detail: " + str([(s.kind, round(float(np.linalg.norm(s.end - s.start)), 3), s.n_points) for s in segs]))
    check(f"  G0 closed exactly", g0(segs, True) < 1e-9, f"{g0(segs, True):.1e}")
    check(f"  within tolerance", max(s.max_dev for s in segs) <= TOL * 1.5, f"{max(s.max_dev for s in segs):.4f}")

# Sharp rectangle: 4 lines, corners exactly at the intersections
f, total = rounded_rect(100, 60, 0.0)
segs = segment_profile(resample(f, total, 0.8), True, TOL)
corners = np.array([s.start for s in segs])
expected = np.array([[50, -30], [50, 30], [-50, 30], [-50, -30]])
ok = len(segs) == 4 and all(np.min(np.linalg.norm(expected - c, axis=1)) < 1e-6 for c in corners)
check("sharp rectangle: 4 lines meeting at exact corners", ok, f"{kinds(segs)} {np.round(corners, 6).tolist()}")

# Obround slot: 2 lines + 2 half circles R10
f, total = rounded_rect(80, 20, 10)
segs = segment_profile(resample(f, total, 0.6), True, TOL)
arcs = [s for s in segs if s.kind == "arc"]
check("slot: 2 lines + 2 semicircles", kinds(segs).count("L") == 2 and len(arcs) == 2 and all(abs(a.sweep_deg - 180) < 0.5 and abs(a.radius - 10) < 0.02 for a in arcs),
      f"{kinds(segs)} {[(round(a.radius,3), round(a.sweep_deg,1)) for a in arcs]}")

# Ellipse: free-form -> None (caller keeps a single periodic spline)
t = np.sort(rng.uniform(0, 2 * np.pi, 400))
ell = np.column_stack((30 * np.cos(t), 15 * np.sin(t)))
check("ellipse stays free-form", segment_profile(ell, True, TOL) is None)

# Mixed: straight base + sine-wave top -> lines + spline(s), G0 closed
xs = np.linspace(-40, 40, 300)
top = np.column_stack((xs, 20 + 3 * np.sin(xs / 6)))
right = np.column_stack((np.full(30, 40.0), np.linspace(20 + 3 * np.sin(40 / 6), -10, 30)))
base = np.column_stack((np.linspace(40, -40, 100), np.full(100, -10.0)))
left = np.column_stack((np.full(30, -40.0), np.linspace(-10, 20 + 3 * np.sin(-40 / 6), 30)))
mixed = np.vstack((top, right[1:], base[1:], left[1:-1]))
segs = segment_profile(mixed, True, TOL)
k = kinds(segs)
check("mixed: 3 lines + free-form top as spline", k.count("L") == 3 and k.count("S") == 1 and "A" not in k, k)
check("  G0 closed exactly", g0(segs, True) < 1e-9)
check("  within tolerance", max(s.max_dev for s in segs) <= TOL * 1.5, f"{max(s.max_dev for s in segs):.4f}")

# Open profile (scan gap): line - fillet - line, endpoints on the data ends
f, total = rounded_rect(100, 60, 5)
span = (90 - 45) + np.pi * 5 / 2 + 25  # bottom middle -> fillet -> right-side middle
open_pts = resample(lambda t: f(t + 45), span, 0.8)
segs = segment_profile(open_pts, False, TOL)
check("open line-arc-line", kinds(segs) == "LAL", kinds(segs))
check("  open ends on data", np.linalg.norm(segs[0].start - open_pts[0]) < 0.05 and np.linalg.norm(segs[-1].end - open_pts[-1]) < 0.05)

# Speed on a dense loop
f, total = rounded_rect(300, 200, 12)
dense = resample(f, total, 0.15)
t = time.perf_counter(); segs = segment_profile(dense, True, TOL); dt = time.perf_counter() - t
check("dense loop speed", dt < 0.5 and kinds(segs).count("A") == 4, f"{len(dense)} pts in {dt*1000:.0f} ms -> {kinds(segs)}")

# Stress: noise vs tolerance. tol ~ 4 sigma must segment cleanly; at tol = 3 sigma the
# noise peaks reach the tolerance itself (informational).
f, total = rounded_rect(100, 60, 5)
for div, required in ((4, True), (3, False)):
    hits = 0
    for trial in range(5):
        pts = resample(f, total, 0.8); pts = pts + rng.normal(0, TOL / div, pts.shape)
        segs = segment_profile(pts, True, TOL)
        k = kinds(segs) if segs else "None"
        hits += k.count("L") == 4 and k.count("A") == 4 and "S" not in k
    if required:
        check(f"noise = tol/{div}: 4 lines + 4 arcs in 5/5 trials", hits == 5, f"{hits}/5")
    else:
        out.append(f"INFO  noise = tol/{div}: clean in {hits}/5 trials (noise peaks ~ tol)")

# Hexagon: 6 lines, 120 deg corners exact
ang = np.radians(np.arange(6) * 60.0)
verts = 30 * np.column_stack((np.cos(ang), np.sin(ang)))
hexa = np.vstack([np.linspace(verts[i], verts[(i + 1) % 6], 38, endpoint=False) for i in range(6)])
hexa = hexa + rng.normal(0, 0.002, hexa.shape)
segs = segment_profile(hexa, True, TOL)
corner_err = max(np.min(np.linalg.norm(verts - s.start, axis=1)) for s in segs)
check("hexagon: 6 lines, exact corners", kinds(segs) == "LLLLLL" and corner_err < 0.01, f"{kinds(segs)} corner err {corner_err:.4f}")

# Coarse sampling of small fillets (R1 with 0.8 mm spacing: ~2 samples per fillet)
f, total = rounded_rect(100, 60, 1.0)
segs = segment_profile(resample(f, total, 0.8), True, TOL)
check("tiny fillets: G0 closed, within tol, no crash", g0(segs, True) < 1e-9 and max(s.max_dev for s in segs) <= 2 * TOL,
      f"{kinds(segs)} dev {max(s.max_dev for s in segs):.4f}")

print("\n".join(out)); print(f"{sum(o.startswith('FAIL') for o in out)} failures")
