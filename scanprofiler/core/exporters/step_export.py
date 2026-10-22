"""STEP AP214 curve wireframe export (ISO 10303-21 text, written directly).

One GEOMETRIC_CURVE_SET per section inside a GEOMETRICALLY_BOUNDED_WIREFRAME_SHAPE_
REPRESENTATION, in millimetres, in the current (aligned) CAD coordinates:
  line   -> TRIMMED_CURVE on LINE
  arc    -> TRIMMED_CURVE on CIRCLE (sense .T. = CCW about the section normal)
  circle -> CIRCLE
  spline -> B_SPLINE_CURVE_WITH_KNOTS (exact knots / control points)
Centre marks are trimmed lines in a separate '<section> centre marks' curve set.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import numpy as np

from ... import __version__
from ..sectioning import Section
from .common import ExportOptions, ExportReport, arc_angles_ccw, center_mark, loop_entities

SCHEMA = "AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }"


def _real(x: float) -> str:
    """STEP REAL: always has a decimal point; 12 decimals is 1e-12 mm."""
    s = f"{float(x):.12f}".rstrip("0")  # keeps the "." ("12." is a valid STEP real)
    return "0." if s == "-0." else s


def _str(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


class _Writer:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def add(self, entity: str) -> str:
        ref = f"#{len(self.lines) + 1}"
        self.lines.append(f"{ref}={entity};")
        return ref

    def point(self, p) -> str:
        return self.add(f"CARTESIAN_POINT('',({','.join(_real(v) for v in p)}))")

    def direction(self, d) -> str:
        d = np.asarray(d, float) / np.linalg.norm(d)
        return self.add(f"DIRECTION('',({','.join(_real(v) for v in d)}))")

    def trimmed_line(self, a, b) -> str:
        a, b = np.asarray(a, float), np.asarray(b, float)
        pa = self.point(a)
        vec = self.add(f"VECTOR('',{self.direction(b - a)},{_real(np.linalg.norm(b - a))})")
        line = self.add(f"LINE('',{pa},{vec})")
        return self.add(f"TRIMMED_CURVE('',{line},({pa}),({self.point(b)}),.T.,.CARTESIAN.)")

    def circle(self, center, normal, ref, radius) -> str:
        axis = self.add(f"AXIS2_PLACEMENT_3D('',{self.point(center)},{self.direction(normal)},{self.direction(ref)})")
        return self.add(f"CIRCLE('',{axis},{_real(radius)})")


def _knots(t) -> tuple[list[float], list[int]]:
    t = np.asarray(t, float)
    values, mults = [], []
    for v in t:
        if values and abs(v - values[-1]) <= 1e-12:
            mults[-1] += 1
        else:
            values.append(float(v))
            mults.append(1)
    return values, mults


def _write_section(w: _Writer, section: Section, options: ExportOptions, report: ExportReport) -> list[str]:
    normal = np.eye(3)[section.axis]  # (u, v, normal) is right-handed: CCW in (u, v) = CCW about +normal

    def lift(uv) -> np.ndarray:
        return section.to_3d(np.atleast_2d(np.asarray(uv, float)))[0]

    curves, marks = [], []
    for loop in section.loops:
        ent = loop_entities(loop)
        if not ent.closed:
            report.warnings.append(f"{section.name}: open chain exported (not a closed contour)")
        if ent.circle is not None:
            center, radius = ent.circle
            ref = lift(center + np.array([1.0, 0.0])) - lift(center)
            curves.append(w.circle(lift(center), normal, ref, radius))
            report.count("CIRCLE")
            if options.center_marks:
                marks += [w.trimmed_line(lift(a), lift(b)) for a, b in center_mark(center, radius)]
                report.count("centre mark")
            continue
        for seg in ent.segments:
            if seg.kind == "line":
                curves.append(w.trimmed_line(lift(seg.start), lift(seg.end)))
                report.count("LINE")
            elif seg.kind == "arc":
                start, end, _, _ = arc_angles_ccw(seg)
                c3 = lift(seg.center)
                circ = w.circle(c3, normal, lift(start) - c3, seg.radius)
                curves.append(w.add(f"TRIMMED_CURVE('',{circ},({w.point(lift(start))}),({w.point(lift(end))}),.T.,.CARTESIAN.)"))
                report.count("ARC")
            else:
                t, (cu, cv), k = seg.tck
                cps = [w.point(lift((u, v))) for u, v in zip(cu, cv)]
                values, mults = _knots(t)
                curves.append(w.add(
                    f"B_SPLINE_CURVE_WITH_KNOTS('',{k},({','.join(cps)}),.UNSPECIFIED.,.F.,.F.,"
                    f"({','.join(str(m) for m in mults)}),({','.join(_real(v) for v in values)}),.UNSPECIFIED.)"
                ))
                report.count("SPLINE")
    sets = [w.add(f"GEOMETRIC_CURVE_SET({_str(section.name)},({','.join(curves)}))")] if curves else []
    if marks:
        sets.append(w.add(f"GEOMETRIC_CURVE_SET({_str(section.name + ' centre marks')},({','.join(marks)}))"))
    return sets


def export_step(sections: list[Section], path: str | Path, options: ExportOptions | None = None) -> ExportReport:
    options = options or ExportOptions()
    report = ExportReport()
    w = _Writer()

    app = w.add("APPLICATION_CONTEXT('automotive design')")
    w.add(f"APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,{app})")
    pctx = w.add(f"PRODUCT_CONTEXT('',{app},'mechanical')")
    product = w.add(f"PRODUCT('ScanProfiler_sections','ScanProfiler sections','',({pctx}))")
    pdf = w.add(f"PRODUCT_DEFINITION_FORMATION('','',{product})")
    pdctx = w.add(f"PRODUCT_DEFINITION_CONTEXT('part definition',{app},'design')")
    pd = w.add(f"PRODUCT_DEFINITION('design','',{pdf},{pdctx})")
    pds = w.add(f"PRODUCT_DEFINITION_SHAPE('','',{pd})")
    length = w.add("(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.))")
    angle = w.add("(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.))")
    solid = w.add("(NAMED_UNIT(*)SI_UNIT($,.STERADIAN.)SOLID_ANGLE_UNIT())")
    unc = w.add(f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-06),{length},'distance_accuracy_value','confusion accuracy')")
    ctx = w.add(
        f"(GEOMETRIC_REPRESENTATION_CONTEXT(3)GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT(({unc}))"
        f"GLOBAL_UNIT_ASSIGNED_CONTEXT(({length},{angle},{solid}))"
        "REPRESENTATION_CONTEXT('Context3D','3D Context with UNIT and UNCERTAINTY'))"
    )
    origin = w.add(f"AXIS2_PLACEMENT_3D('',{w.point((0, 0, 0))},{w.direction((0, 0, 1))},{w.direction((1, 0, 0))})")

    sets = []
    for section in sections:
        sets += _write_section(w, section, options, report)
    rep = w.add(f"GEOMETRICALLY_BOUNDED_WIREFRAME_SHAPE_REPRESENTATION('sections',({','.join([origin] + sets)}),{ctx})")
    w.add(f"SHAPE_DEFINITION_REPRESENTATION({pds},{rep})")

    stamp = _dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    name = Path(path).name
    text = "\n".join([
        "ISO-10303-21;",
        "HEADER;",
        "FILE_DESCRIPTION(('ScanProfiler cross-section curves'),'2;1');",
        f"FILE_NAME({_str(name)},{_str(stamp)},(''),(''),'ScanProfiler {__version__}','ScanProfiler','');",
        f"FILE_SCHEMA(('{SCHEMA}'));",
        "ENDSEC;",
        "DATA;",
        *w.lines,
        "ENDSEC;",
        "END-ISO-10303-21;",
        "",
    ])
    Path(path).write_text(text, encoding="ascii")
    report.files.append(str(path))
    return report
