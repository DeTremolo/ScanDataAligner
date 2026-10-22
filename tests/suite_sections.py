"""Unit tests for core.sectioning on the CAD-pose plate, a noisy tube and an open mesh."""
import os, sys, time
import _common  # noqa: E402  repo import path, QT_API, generated test parts
import numpy as np
import pyvista as pv
from scanprofiler.core import sectioning as S
from scanprofiler.core.mesh import MeshData
from scanprofiler.core.mesh_io import load_mesh
from scanprofiler.core.selection import delete_faces

SCR = str(_common.WORK)  # generated parts + outputs (git-ignored)
_common.ensure_all()
out = []
def check(name, cond, info=""):
    out.append(f"{'PASS' if cond else 'FAIL'}  {name}  {info}")

def area(uv):
    x, y = uv[:, 0], uv[:, 1]
    return 0.5 * abs(x @ np.roll(y, -1) - y @ np.roll(x, -1))

plate = load_mesh(os.path.join(SCR, "test_plate.stl"))  # 100 x 60 x 20, z in [-10, 10], hole R10 at origin
edge = plate.mean_edge_length()

t = time.perf_counter()
secs = S.section_array(plate, 2, -9, 9, 0.45, circle_tol=0.05, spline_tol=0.01)
dt = time.perf_counter() - t
check("Z array: 41 sections", len(secs) == 41, f"{dt:.2f}s on {plate.n_faces:,} faces")
check("Z array fast", dt < 3.0, f"{dt / len(secs) * 1000:.0f} ms/section")
ok = all(len(s.loops) == 2 and len(s.circles) == 1 for s in secs)
check("each Z section: rectangle + one circle", ok, str([len(s.loops) for s in secs[:5]]))
s = secs[20]
circ = s.circles[0]
rect = next(lp for lp in s.loops if lp.kind != "circle")
check("hole Ø20 at origin", abs(circ.radius - 10) < 0.001 and np.linalg.norm(circ.center) < 1e-3,
      f"Ø{2 * circ.radius:.5f} c={np.round(circ.center, 5)} dev {circ.max_dev:.5f}")
check("nesting: rectangle outer, circle hole", rect.outer and not circ.outer)
lines4 = rect.kind == "profile" and [s.kind for s in rect.segments] == ["line"] * 4
check("rectangle -> 4 exact lines", lines4 and rect.closed, rect.summary())
corner_err = max(np.min(np.linalg.norm(np.array([[50, 30], [-50, 30], [-50, -30], [50, -30]]) - s.start, axis=1)) for s in rect.segments) if lines4 else 1
check("rectangle corners exact", corner_err < 1e-6, f"{corner_err:.1e}")
check("rectangle within max tolerance 0.01", rect.max_dev <= 0.01, f"dev {rect.max_dev:.4f}")
r_spl = S.fit_loop(rect.points, True, 0.05, 0.01, segment=False)
check("segment=False keeps single spline", r_spl.kind == "spline" and r_spl.max_dev <= 0.01, r_spl.summary())
noisy_spline = [lp for lp in S.make_section(S.Slicer(MeshData(plate.vertices + np.random.default_rng(1).normal(0, 0.02, plate.vertices.shape), plate.faces)), 2, 0.0, 0.1, 0.1).loops if lp.kind == "profile"]
check("noisy rectangle (sigma 0.02, tol 0.1 = 5 sigma): 4 lines", noisy_spline and [s.kind for s in noisy_spline[0].segments] == ["line"] * 4 and noisy_spline[0].max_dev <= 0.1,
      noisy_spline[0].summary() if noisy_spline else "none")
check("rectangle area 6000", abs(area(rect.points) - 6000) < 1, f"{area(rect.points):.3f}")
steps = np.linalg.norm(np.diff(rect.points, axis=0), axis=1)
check("loop points ordered (no jumps)", steps.max() < 4 * edge, f"max step {steps.max():.3f} vs edge {edge:.3f}")
samp = rect.sample()
check("spline sample closes", np.linalg.norm(samp[0] - samp[-1]) < 1e-9)
check("3D lift", np.allclose(s.to_3d(circ.points)[:, 2], s.offset))

# X = 0 cuts through the hole: two 20 x 20 squares (y in +-[10, 30], z in [-10, 10])
sx = S.make_section(S.Slicer(plate), 0, 0.0, 0.05, 0.01)
areas = sorted(area(lp.points) for lp in sx.loops)
check("X=0: two 20x20 loops", len(sx.loops) == 2 and np.allclose(areas, 400, atol=0.5) and all(lp.outer for lp in sx.loops),
      f"areas {np.round(areas, 3)}")

# Plane exactly through the top face / vertices: no crash, an outline at most
for z in (10.0, -10.0, 0.0):
    try:
        sz = S.make_section(S.Slicer(plate), 2, z, 0.05, 0.01)
        check(f"slice exactly at z={z}", all(lp.closed for lp in sz.loops), f"{len(sz.loops)} loops")
    except Exception as e:
        check(f"slice exactly at z={z}", False, repr(e))
check("outside the part: no sections", S.section_array(plate, 2, 50, 60, 5, 0.05, 0.01) == [])

# Noisy tube: circles survive 0.02 mm noise with a matching tolerance
rng = np.random.default_rng(0)
grid = pv.CylinderStructured(radius=np.linspace(10, 25, 50), height=30, theta_resolution=210, z_resolution=100, direction=(0, 0, 1))
surf = grid.extract_surface(algorithm=None).triangulate().clean()
import trimesh
welded = trimesh.Trimesh(np.asarray(surf.points), surf.regular_faces, process=True)  # weld the +-180 deg seam
raw = MeshData(np.asarray(welded.vertices, float), np.asarray(welded.faces, np.int64))
cc, cn = raw.face_centers(), raw.face_normals()
rr = np.linalg.norm(cc[:, :2], axis=1)
raw = delete_faces(raw, (rr > 10.2) & (rr < 24.8) & (np.abs(cn[:, 2]) < 0.1))  # seam slit left by extract_surface
tube = MeshData(raw.vertices + rng.normal(0, 0.02, raw.vertices.shape), raw.faces)
st = S.make_section(S.Slicer(tube), 2, 3.3, circle_tol=0.12, spline_tol=0.02)
radii = sorted(lp.radius for lp in st.circles)
check("noisy tube: two circles", len(st.circles) == 2 and abs(radii[0] - 10) < 0.01 and abs(radii[1] - 25) < 0.01,
      f"radii {np.round(radii, 4)}")
check("tube nesting", sorted(lp.outer for lp in st.loops) == [False, True])
st_strict = S.make_section(S.Slicer(tube), 2, 3.3, circle_tol=0.001, spline_tol=0.02)
check("strict circle tol -> free-form or arcs, never circles", not st_strict.circles and all(lp.kind in ("spline", "profile") for lp in st_strict.loops), [lp.summary() for lp in st_strict.loops])

# Open mesh: punch a window in the plate's side wall -> open chains, no crash
c = plate.face_centers()
window = (np.abs(c[:, 1] + 30) < 0.01) & (np.abs(c[:, 0]) < 10) & (np.abs(c[:, 2]) < 5)
holed = delete_faces(plate, window)
so = S.make_section(S.Slicer(holed), 2, 0.0, 0.05, 0.01)
open_loops = [lp for lp in so.loops if not lp.closed]
check("open mesh -> open chain + closed hole", len(open_loops) == 1 and len(so.circles) == 1, [lp.summary() for lp in so.loops])
check("open chain spans the gap", np.linalg.norm(open_loops[0].points[0] - open_loops[0].points[-1]) > 15)

# Real faceted mesh: 100x60 plate with R5 corners, a hole and a slot, sliced end to end
from shapely.geometry import LineString, Point, box
outline = box(-45, -25, 45, 25).buffer(5, quad_segs=32)
hole = Point(-20, 0).buffer(10, quad_segs=64)
slot = LineString([(20, 0), (30, 0)]).buffer(4, quad_segs=32)
shape = outline.difference(hole).difference(slot)
fp = trimesh.creation.extrude_polygon(shape, 20)
fv, ff = trimesh.remesh.subdivide_to_size(fp.vertices, fp.faces, max_edge=2.0)
fillet_plate = MeshData(np.asarray(fv, float), np.asarray(ff, np.int64))
t = time.perf_counter()
sf = S.make_section(S.Slicer(fillet_plate), 2, 10.0, circle_tol=0.05, spline_tol=0.02)
dt = time.perf_counter() - t
by_kind = sorted((lp.kind, "".join(s.kind[0].upper() for s in lp.segments)) for lp in sf.loops)
check("filleted plate: circle + 2 profiles", [k for k, _ in by_kind] == ["circle", "profile", "profile"], f"{by_kind} {dt*1000:.0f} ms")
outer = next((lp for lp in sf.loops if lp.outer), None)
slot_lp = next((lp for lp in sf.loops if lp.kind == "profile" and not lp.outer), None)
if outer is not None and slot_lp is not None:
    oa = [s for s in outer.segments if s.kind == "arc"]
    check("outer: 4 lines + 4 R5 fillets", "".join(sorted(s.kind[0] for s in outer.segments)) == "aaaallll"
          and all(abs(a.radius - 5) < 0.01 and abs(a.sweep_deg - 90) < 0.5 for a in oa), outer.summary())
    sa = [s for s in slot_lp.segments if s.kind == "arc"]
    check("slot: 2 lines + 2 R4 semicircles", len(slot_lp.segments) == 4 and len(sa) == 2
          and all(abs(a.radius - 4) < 0.01 and abs(a.sweep_deg - 180) < 0.5 for a in sa), slot_lp.summary())
    for name, lp in (("outer", outer), ("slot", slot_lp)):
        gaps = [np.linalg.norm(lp.segments[k].end - lp.segments[(k + 1) % len(lp.segments)].start) for k in range(len(lp.segments))]
        check(f"{name}: G0-closed chain", max(gaps) < 1e-9, f"max gap {max(gaps):.1e}")
hole_lp = next(lp for lp in sf.loops if lp.kind == "circle")
check("hole circle Ø20 at (-20, 0)", abs(hole_lp.radius - 10) < 0.002 and np.linalg.norm(hole_lp.center - [-20, 0]) < 1e-3, hole_lp.summary())

print("\n".join(out)); print(f"{sum(o.startswith('FAIL') for o in out)} failures")
