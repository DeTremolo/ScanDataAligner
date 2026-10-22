"""SolidWorks macro export: a VBScript (.vbs) that builds the sections natively.

Run it while SolidWorks (2021 or later; API is late-bound) has the target part open:
double-click the file or `cscript macro.vbs`. For every section it creates an offset
reference plane from the matching default plane, then a named 2D sketch on it with
lines, arcs, circles, splines and centre marks (construction lines).

Robustness notes:
  * default planes are found by feature-tree order (Front, Top, Right = first three
    RefPlane features), so non-English SolidWorks installs work
  * all numbers are written as VBScript literals ('.' decimal), never parsed from strings,
    so a decimal-comma Windows locale cannot misread them
  * SolidWorks splines are created through fit points (CreateSpline2) sampled densely
    from the exact B-spline; DXF / STEP carry the exact control points
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ... import __version__
from ..sectioning import AXIS_NAMES, Section
from .common import ExportOptions, ExportReport, center_mark, loop_entities, section_tag, spline_samples

# Section normal axis -> (index of the default plane in feature-tree order, its English name).
BASE_PLANE = {2: (0, "Front"), 1: (1, "Top"), 0: (2, "Right")}

# Model -> sketch coordinates on each default plane (and its offsets), from SolidWorks'
# standard view orientations:  Front: (X, Y)   Top: (X, -Z)   Right: (-Z, Y).
# Each is a proper rotation of the section (u, v) axes, so CCW stays CCW.
# UNVERIFIED IN SOLIDWORKS: if a section lands rotated / mirrored, fix it here.
SKETCH_AXES = {2: ((0, 1.0), (1, 1.0)), 1: ((0, 1.0), (2, -1.0)), 0: ((2, -1.0), (1, 1.0))}

SPLINE_ARRAY_CHUNK = 6  # array assignments per VBScript line

HELPERS = r"""
Function ConnectSolidWorks()
    On Error Resume Next
    Set ConnectSolidWorks = GetObject(, "SldWorks.Application")
    If Err.Number <> 0 Then
        Err.Clear
        Set ConnectSolidWorks = CreateObject("SldWorks.Application")
    End If
    On Error GoTo 0
End Function

Sub Notify(msg)
    MsgBox msg, 64, "ScanProfiler"
End Sub

' Default planes by feature-tree order (0 Front, 1 Top, 2 Right): language independent.
Function BasePlane(part, idx)
    Dim f, n
    n = 0
    Set BasePlane = Nothing
    Set f = part.FirstFeature
    Do While Not f Is Nothing
        If f.GetTypeName2 = "RefPlane" Then
            If n = idx Then
                Set BasePlane = f
                Exit Function
            End If
            n = n + 1
        End If
        Set f = f.GetNextFeature
    Loop
End Function

' Offset plane (mm) from a default plane; zero offset reuses the default plane.
Function SectionPlane(part, baseIdx, offsetMm, planeName)
    Dim base, plane
    Set base = BasePlane(part, baseIdx)
    If Abs(offsetMm) < 0.000001 Then
        Set SectionPlane = base
        Exit Function
    End If
    part.ClearSelection2 True
    base.Select2 False, 0
    If offsetMm > 0 Then
        Set plane = part.FeatureManager.InsertRefPlane(REFPLANE_DISTANCE, offsetMm * MM, 0, 0, 0, 0)
    Else
        Set plane = part.FeatureManager.InsertRefPlane(REFPLANE_DISTANCE + REFPLANE_FLIP, -offsetMm * MM, 0, 0, 0, 0)
    End If
    If Not plane Is Nothing Then plane.Name = planeName
    Set SectionPlane = plane
End Function

Sub BeginSketch(part, plane)
    part.ClearSelection2 True
    plane.Select2 False, 0
    part.SketchManager.InsertSketch True
    part.SketchManager.AddToDB = True          ' no snapping / inference while drawing
    part.SketchManager.DisplayWhenAdded = False
End Sub

Sub EndSketch(part, sketchName)
    part.SketchManager.AddToDB = False
    part.SketchManager.DisplayWhenAdded = True
    part.SketchManager.InsertSketch True        ' close the sketch
    part.FeatureByPositionReverse(0).Name = sketchName
End Sub

Sub Ln(sm, x1, y1, x2, y2)
    sm.CreateLine x1 * MM, y1 * MM, 0, x2 * MM, y2 * MM, 0
End Sub

Sub Arc(sm, xc, yc, x1, y1, x2, y2, dirn)
    sm.CreateArc xc * MM, yc * MM, 0, x1 * MM, y1 * MM, 0, x2 * MM, y2 * MM, 0, dirn
End Sub

Sub Circ(sm, xc, yc, r)
    sm.CreateCircleByRadius xc * MM, yc * MM, 0, r * MM
End Sub

Sub Spl(sm, pts)
    Dim i
    For i = 0 To UBound(pts)
        pts(i) = pts(i) * MM
    Next
    sm.CreateSpline2 pts, True
End Sub

Sub Mark(sm, x1, y1, x2, y2)
    Dim seg
    Set seg = sm.CreateLine(x1 * MM, y1 * MM, 0, x2 * MM, y2 * MM, 0)
    If Not seg Is Nothing Then seg.ConstructionGeometry = True
End Sub
"""


def _num(x: float) -> str:
    """VBScript numeric literal (locale independent); float noise below 1e-9 mm is 0."""
    x = 0.0 if abs(float(x)) < 1e-9 else float(x)
    s = f"{x:.10g}"
    return "0" if s in ("-0", "0") else s


def _sketch_xy(section: Section, uv: np.ndarray) -> np.ndarray:
    p = section.to_3d(np.atleast_2d(np.asarray(uv, float)))
    (ia, sa), (ib, sb) = SKETCH_AXES[section.axis]
    return np.column_stack((sa * p[:, ia], sb * p[:, ib]))


def _section_block(section: Section, options: ExportOptions, report: ExportReport) -> list[str]:
    idx, base = BASE_PLANE[section.axis]
    tag = section_tag(section)
    out = [
        "",
        f"    ' ---- {section.name}  (offset from {base} plane)",
        f'    Set plane = SectionPlane(part, {idx}, {_num(section.offset)}, "ScanProfiler plane {tag}")',
        "    If plane Is Nothing Then",
        f'        Notify "Could not create the plane for {section.name}."',
        "        Exit Sub",
        "    End If",
        "    BeginSketch part, plane",
    ]

    def xy(uv) -> tuple[str, str]:
        x, y = _sketch_xy(section, uv)[0]
        return _num(x), _num(y)

    for loop in section.loops:
        ent = loop_entities(loop)
        if not ent.closed:
            report.warnings.append(f"{section.name}: open chain exported (not a closed contour)")
        if ent.circle is not None:
            center, radius = ent.circle
            out.append(f"    Circ sm, {', '.join(xy(center))}, {_num(radius)}")
            report.count("circle")
            if options.center_marks:
                for a, b in center_mark(center, radius):
                    out.append(f"    Mark sm, {', '.join(xy(a))}, {', '.join(xy(b))}")
                report.count("centre mark")
            continue
        for seg in ent.segments:
            if seg.kind == "line":
                out.append(f"    Ln sm, {', '.join(xy(seg.start))}, {', '.join(xy(seg.end))}")
                report.count("line")
            elif seg.kind == "arc":
                direction = 1 if seg.ccw else -1
                out.append(
                    f"    Arc sm, {', '.join(xy(seg.center))}, {', '.join(xy(seg.start))}, {', '.join(xy(seg.end))}, {direction}"
                )
                report.count("arc")
            else:
                pts = _sketch_xy(section, spline_samples(seg, loop.tol or 0.01))
                flat = [v for x, y in pts for v in (x, y, 0.0)]
                out.append(f"    ReDim pts({len(flat) - 1})")
                for start in range(0, len(flat), SPLINE_ARRAY_CHUNK):
                    chunk = flat[start : start + SPLINE_ARRAY_CHUNK]
                    out.append("    " + ": ".join(f"pts({start + k}) = {_num(v)}" for k, v in enumerate(chunk)))
                out.append("    Spl sm, pts")
                report.count("spline")
    out.append(f'    EndSketch part, "ScanProfiler section {tag}"')
    return out


def export_solidworks_vbs(sections: list[Section], path: str | Path, options: ExportOptions | None = None) -> ExportReport:
    options = options or ExportOptions()
    report = ExportReport()
    body = []
    for section in sections:
        body += _section_block(section, options, report)

    axes = sorted({AXIS_NAMES[s.axis] for s in sections})
    lines = [
        f"' ScanProfiler {__version__}: {len(sections)} cross-section(s) along {', '.join(axes)} as SolidWorks sketches.",
        "' Usage: open (or create) the target PART in SolidWorks, then double-click this file",
        "'        or run:  cscript //nologo <this file>",
        "' Coordinates are the aligned CAD coordinates exported by ScanProfiler, in millimetres.",
        "Option Explicit",
        "",
        "Const MM = 0.001                 ' SolidWorks API works in metres",
        "Const REFPLANE_DISTANCE = 8      ' swRefPlaneReferenceConstraint_Distance",
        "Const REFPLANE_FLIP = 256        ' swRefPlaneReferenceConstraint_OptionFlip",
        "Const SW_DOC_PART = 1            ' swDocPART",
        HELPERS,
        "Sub Main()",
        "    Dim swApp, part, sm, plane, pts",
        "    Set swApp = ConnectSolidWorks()",
        "    If swApp Is Nothing Then",
        '        Notify "SolidWorks is not running."',
        "        Exit Sub",
        "    End If",
        "    Set part = swApp.ActiveDoc",
        "    If part Is Nothing Then",
        '        Notify "Open a part document in SolidWorks first."',
        "        Exit Sub",
        "    End If",
        "    If part.GetType <> SW_DOC_PART Then",
        '        Notify "The active document is not a part."',
        "        Exit Sub",
        "    End If",
        "    Set sm = part.SketchManager",
        *body,
        "",
        "    part.ClearSelection2 True",
        "    part.EditRebuild3",
        "    part.ViewZoomtofit2",
        f'    Notify "ScanProfiler: {len(sections)} section sketch(es) created."',
        "End Sub",
        "",
        "Main",
        "",
    ]
    text = "\n".join(lines).replace("\r\n", "\n").replace("\n", "\r\n")
    Path(path).write_bytes(text.encode("ascii"))  # VBScript: ANSI, CRLF
    report.files.append(str(path))
    return report
