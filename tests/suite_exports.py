"""Exporter tests: DXF read-back + closure, STEP parse + geometry, VBS executed by cscript against a mock."""
import os, re, subprocess, sys
import _common  # noqa: E402  repo import path, QT_API, generated test parts
SCR = str(_common.WORK)  # generated parts + outputs (git-ignored)
_common.ensure_all()
OUT = os.path.join(SCR, "export_out"); os.makedirs(OUT, exist_ok=True)

import numpy as np
import ezdxf
from scanprofiler.core import sectioning as S
from scanprofiler.core.exporters import ExportOptions, export_dxf, export_solidworks_vbs, export_step
from scanprofiler.core.exporters.common import loop_entities, _signed_area
from scanprofiler.core.mesh import MeshData
from scanprofiler.core.mesh_io import load_mesh

out = []
def check(name, cond, info=""):
    out.append(f"{'PASS' if cond else 'FAIL'}  {name}  {info}")

# Fixture: filleted plate (R5 corners, hole d20 at (-20,0), R4 slot) -> Z sections + an X and a Y section
plate = load_mesh(os.path.join(SCR, "test_fillet_plate.stl"))
sl = S.Slicer(plate)
z10 = S.make_section(sl, 2, 10.0, 0.05, 0.02)
x0 = S.make_section(sl, 0, 0.0, 0.05, 0.02)      # through the plate middle (x=0 misses hole & slot)
y0 = S.make_section(sl, 1, 0.0, 0.05, 0.02)      # through hole and slot
# a free-form loop: ellipse cylinder
import pyvista as pv, trimesh
ell = trimesh.creation.extrude_polygon(__import__("shapely").geometry.Polygon(np.column_stack((30*np.cos(np.linspace(0,2*np.pi,400,endpoint=False)), 15*np.sin(np.linspace(0,2*np.pi,400,endpoint=False))))), 10)
ellsec = S.make_section(S.Slicer(MeshData(np.asarray(ell.vertices, float), np.asarray(ell.faces, np.int64))), 2, 5.0, 0.05, 0.02)
sections = [z10, x0, y0]
kinds = [(s.name, [lp.kind for lp in s.loops]) for s in sections + [ellsec]]
print("fixture:", kinds)

# ---------------------------------------------------------------- orientation prep
for s in sections:
    for lp in s.loops:
        ent = loop_entities(lp)
        if ent.segments and ent.closed:
            area = _signed_area(np.vstack([g.sample() for g in ent.segments]))
            gaps = [np.linalg.norm(ent.segments[k].end - ent.segments[(k + 1) % len(ent.segments)].start) for k in range(len(ent.segments))]
            check(f"{s.name}: {'outer CCW' if lp.outer else 'hole CW'}, G0", (area > 0) == lp.outer and max(gaps) < 1e-9, f"area {area:.1f}")

# ---------------------------------------------------------------- DXF
def dxf_contours(path):
    doc = ezdxf.readfile(path)
    auditor = doc.audit()
    msp = doc.modelspace()
    ends = {}  # layer -> list of (p0, p1)
    counts = {}
    for e in msp:
        t = e.dxftype(); counts[t] = counts.get(t, 0) + 1
        layer = e.dxf.layer
        if t == "LINE":
            p0, p1 = np.array(e.dxf.start)[:2], np.array(e.dxf.end)[:2]
        elif t == "ARC":
            c, r = np.array(e.dxf.center)[:2], e.dxf.radius
            a0, a1 = np.radians(e.dxf.start_angle), np.radians(e.dxf.end_angle)
            p0, p1 = c + r * np.array([np.cos(a0), np.sin(a0)]), c + r * np.array([np.cos(a1), np.sin(a1)])
        elif t == "SPLINE":
            cps = np.array(e.control_points)[:, :2]; p0, p1 = cps[0], cps[-1]
        elif t == "LWPOLYLINE":
            pts = np.array(e.get_points("xy")); p0, p1 = pts[0], pts[-1]
        else:
            continue
        ends.setdefault(layer, []).append((p0, p1))
    return doc, auditor, counts, ends

def closed_chains(segs, eps=1e-9):
    """Every endpoint must meet exactly one other endpoint (degree 2)."""
    pts = np.array([p for s in segs for p in s])
    if not len(pts):
        return True
    d = np.linalg.norm(pts[:, None] - pts[None], axis=2)
    np.fill_diagonal(d, np.inf)
    return bool(np.all((d < eps).sum(axis=1) == 1))

p = os.path.join(OUT, "sections.dxf")
rep = export_dxf(sections, p)
doc, aud, counts, ends = dxf_contours(p)
check("DXF single file written + audit clean", os.path.exists(p) and not aud.has_errors, f"{counts} | {rep.summary()}")
check("DXF units mm", doc.units == ezdxf.units.MM)
contour_layers = [l for l in ends if not l.endswith("CENTER") and not l.endswith("OPEN")]
check("DXF every contour closed exactly (1e-9)", all(closed_chains(ends[l]) for l in contour_layers), contour_layers)
check("DXF Z section: 4 LINE + 4 ARC outer, hole CIRCLE, slot 2+2, 2 centre-mark lines",
      len(ends.get("Z+10.000_OUTER", [])) == 8 and len(ends.get("Z+10.000_HOLES", [])) == 4 and len(ends.get("Z+10.000_CENTER", [])) == 2,
      {k: len(v) for k, v in ends.items() if k.startswith("Z")})
circles = [e for e in doc.modelspace().query("CIRCLE")]
check("DXF hole circle d20 at (-20,0)", any(abs(c.dxf.radius - 10) < 0.002 and abs(c.dxf.center[0] + 20) < 1e-3 for c in circles))
arcs = [e for e in doc.modelspace().query('ARC[layer=="Z+10.000_OUTER"]')]
check("DXF R5 fillets sweep 90 deg", len(arcs) == 4 and all(abs(a.dxf.radius - 5) < 0.01 and abs(((a.dxf.end_angle - a.dxf.start_angle) % 360) - 90) < 0.5 for a in arcs),
      [(round(a.dxf.radius, 3), round((a.dxf.end_angle - a.dxf.start_angle) % 360, 2)) for a in arcs])
lay = {e.dxf.name: (e.dxf.color, e.dxf.linetype) for e in doc.layers}
check("DXF layer colours + CENTER linetype", lay["Z+10.000_OUTER"] == (7, "Continuous") and lay["Z+10.000_HOLES"][0] == 1
      and lay["Z+10.000_CENTER"] == (3, "CENTER") and lay["Z+10.000_OPEN"][0] == 6, {k: v for k, v in lay.items() if k.startswith("Z")})

# separate files + splines (free-form ellipse) + polyline mode
rep2 = export_dxf(sections + [ellsec], os.path.join(OUT, "part.dxf"), ExportOptions(separate_files=True))
check("DXF separate files per section", len(rep2.files) == 4 and all(os.path.exists(f) for f in rep2.files), [os.path.basename(f) for f in rep2.files])
ell_file = rep2.files[-1]
_, aud3, c3, e3 = dxf_contours(ell_file)
spl = [e for e in ezdxf.readfile(ell_file).modelspace().query("SPLINE")]
check("DXF ellipse as one closed SPLINE", len(spl) == 1 and closed_chains(e3["OUTER"]) and not aud3.has_errors, c3)
if spl:
    from ezdxf.math import BSpline
    bs = spl[0].construction_tool()
    samples = np.array(list(bs.approximate(2000)))[:, :2]
    r = np.sqrt((samples[:, 0] / 30) ** 2 + (samples[:, 1] / 15) ** 2)
    check("DXF spline evaluates onto the ellipse", np.abs(r - 1).max() * 15 < 0.05, f"max radial err ~{np.abs(r-1).max()*15:.4f} mm")
rep3 = export_dxf([ellsec], os.path.join(OUT, "ellipse_poly.dxf"), ExportOptions(splines_as_polylines=True))
_, _, c4, e4 = dxf_contours(rep3.files[0])
check("DXF splines-as-polylines option", c4.get("LWPOLYLINE") == 1 and "SPLINE" not in c4 and closed_chains(e4["OUTER"]), c4)

# ---------------------------------------------------------------- STEP
def parse_step(path):
    text = open(path, encoding="ascii").read()
    assert text.startswith("ISO-10303-21;") and text.rstrip().endswith("END-ISO-10303-21;")
    data = text.split("DATA;")[1].split("ENDSEC;")[0]
    ents = {}
    for m in re.finditer(r"#(\d+)=(.*?);\s*$", data, re.M):
        ents[int(m.group(1))] = m.group(2)
    return text, ents

p = os.path.join(OUT, "sections.step")
rep = export_step(sections + [ellsec], p)
text, ents = parse_step(p)
refs = {int(r) for body in ents.values() for r in re.findall(r"#(\d+)", body)}
check("STEP all references resolve", refs <= set(ents), f"{len(ents)} entities, {len(refs - set(ents))} dangling")
check("STEP balanced parentheses", all(b.count("(") == b.count(")") for b in ents.values()))
check("STEP AP214 schema + mm units", "AUTOMOTIVE_DESIGN" in text and "SI_UNIT(.MILLI.,.METRE.)" in text)
types = {}
for b in ents.values():
    t = re.match(r"\(?([A-Z_0-9]+)", b).group(1); types[t] = types.get(t, 0) + 1
check("STEP entity mix", types.get("GEOMETRIC_CURVE_SET", 0) >= 4 and types.get("B_SPLINE_CURVE_WITH_KNOTS", 0) >= 1
      and types.get("TRIMMED_CURVE", 0) > 0 and "GEOMETRICALLY_BOUNDED_WIREFRAME_SHAPE_REPRESENTATION" in types, types)
def pt(ref):
    body = ents[int(ref[1:])]
    assert body.startswith("CARTESIAN_POINT"), body
    return np.array([float(v) for v in re.search(r"\('[^']*',\(([^)]*)\)\)$", body).group(1).split(",")])
# Z section trimmed curves: endpoints all at z=10 and every endpoint matched (closed)
zset = next(b for b in ents.values() if b.startswith("GEOMETRIC_CURVE_SET('Z = 10.0000'"))
members = re.findall(r"#\d+", zset.split(",(", 1)[1])
endpoints = []
for m in members:
    b = ents[int(m[1:])]
    if b.startswith("TRIMMED_CURVE"):
        a, c = re.findall(r"\((#\d+)\)", b)
        endpoints.append((pt(a), pt(c)))
check("STEP Z section curves lie in z=10", all(abs(p[2] - 10) < 1e-9 for pair in endpoints for p in pair), f"{len(endpoints)} trimmed curves")
xyz = [(p0, p1) for p0, p1 in endpoints]
pts = np.array([p for pair in xyz for p in pair]); d = np.linalg.norm(pts[:, None] - pts[None], axis=2); np.fill_diagonal(d, np.inf)
check("STEP Z section contours closed", bool(np.all((d < 1e-9).sum(axis=1) == 1)))
yset = next(b for b in ents.values() if b.startswith("GEOMETRIC_CURVE_SET('Y = 0.0000'"))
ys = [pt(r)[1] for m in re.findall(r"#\d+", yset.split(",(", 1)[1]) if ents[int(m[1:])].startswith("TRIMMED_CURVE")
      for r in re.findall(r"\((#\d+)\)", ents[int(m[1:])])]
check("STEP Y section curves lie in y=0", np.allclose(ys, 0, atol=1e-9), f"{len(ys)} endpoints")

# ---------------------------------------------------------------- SolidWorks VBS under cscript + mock
p = os.path.join(OUT, "sections.vbs")
rep = export_solidworks_vbs(sections + [ellsec], p)
src = open(p, "rb").read()
check("VBS is ASCII with CRLF line ends", src.isascii() and b"\r\n" in src and b"\n" not in src.replace(b"\r\n", b""))
harness = str(_common.HARNESS)
res = subprocess.run(["cscript", "//nologo", harness, p], capture_output=True, text=True, timeout=120)
log = res.stdout.splitlines()
check("VBS runs to completion under cscript (mock SolidWorks)", res.returncode == 0 and any(l.startswith("NOTIFY ScanProfiler: 4 section") for l in log),
      (res.stderr.strip() or (log[-1] if log else "no output"))[:200])
calls = [l.split()[0] for l in log]
n = {k: calls.count(k) for k in ("InsertRefPlane", "InsertSketch", "CreateLine", "CreateArc", "CreateCircleByRadius", "CreateSpline2", "Construction", "Rename")}
expected_lines = sum(1 for s in sections + [ellsec] for lp in s.loops for g in loop_entities(lp).segments if g.kind == "line")
expected_marks = 2 * sum(len(s.circles) for s in sections + [ellsec])
check("VBS planes: 3 offset (Y=0 / X=0 reuse base plane) ", n["InsertRefPlane"] == 2, n)
check("VBS sketch open/close pairs", n["InsertSketch"] == 8)
check("VBS entity counts match export", n["CreateLine"] == expected_lines + expected_marks and n["Construction"] == expected_marks
      and n["CreateCircleByRadius"] == sum(len(s.circles) for s in sections + [ellsec]) and n["CreateSpline2"] >= 1, n)
# geometry check: Z=10 outer line endpoints map to sketch (X, Y) in metres on the Front-plane offset
zl = [l for l in log if l.startswith("CreateLine")][:4]
coords = np.array([[float(v) for v in l.split()[1:]] for l in zl])
check("VBS Z sketch coords in metres, on Front-plane axes", np.all(np.abs(coords[:, [0, 1, 3, 4]]) <= 0.0501) and np.all(coords[:, [2, 5]] == 0),
      np.round(coords[0], 4).tolist())
plane_lines = [l for l in log if l.startswith("InsertRefPlane")]
check("VBS Z plane offset 10 mm = 0.01 m", any(abs(float(l.split()[2]) - 0.01) < 1e-12 for l in plane_lines), plane_lines)
renames = [l for l in log if l.startswith("Rename")]
check("VBS planes + sketches named", any("section Z+10.000" in l for l in renames) and any("plane Z+10.000" in l for l in renames), renames[:3])

print("\n".join(out)); print(f"{sum(o.startswith('FAIL') for o in out)} failures")
