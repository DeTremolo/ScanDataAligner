"""DXF profile export (ezdxf): native LINE / ARC / CIRCLE / SPLINE, CNC-ready closed contours.

Layers per section (prefixed with the section tag when several share one file):
  OUTER   outer contours, CCW            colour 7
  HOLES   inner contours and holes, CW   colour 1
  CENTER  hole centre marks              colour 3, CENTER linetype
  OPEN    open chains (not cuttable)     colour 6
"""

from __future__ import annotations

from pathlib import Path

import ezdxf
from ezdxf import units

from ..sectioning import Section
from .common import (
    ExportOptions,
    ExportReport,
    arc_angles_ccw,
    center_mark,
    loop_entities,
    section_tag,
    spline_samples,
)

LAYERS = {"OUTER": 7, "HOLES": 1, "CENTER": 3, "OPEN": 6}


def _new_doc() -> ezdxf.document.Drawing:
    doc = ezdxf.new("R2010", setup=True)  # setup=True provides the CENTER linetype
    doc.units = units.MM
    doc.header["$MEASUREMENT"] = 1  # metric
    return doc


def _layers(doc, prefix: str) -> dict[str, str]:
    names = {}
    for role, colour in LAYERS.items():
        name = f"{prefix}{role}"
        if name not in doc.layers:
            # ezdxf: color / linetype keywords override dxfattribs, so pass them directly
            doc.layers.add(name, color=colour, linetype="CENTER" if role == "CENTER" else "Continuous")
        names[role] = name
    return names


def _write_section(msp, section: Section, layers: dict[str, str], options: ExportOptions, report: ExportReport) -> None:
    for loop in section.loops:
        ent = loop_entities(loop)
        layer = layers["OPEN"] if not ent.closed else layers["OUTER"] if ent.outer else layers["HOLES"]
        attribs = {"layer": layer}
        if not ent.closed:
            report.warnings.append(f"{section.name}: open chain exported on layer {layer} (not a closed contour)")
        if ent.circle is not None:
            center, radius = ent.circle
            msp.add_circle(tuple(center), radius, dxfattribs=attribs)
            report.count("CIRCLE")
            if options.center_marks:
                for p0, p1 in center_mark(center, radius):
                    msp.add_line(tuple(p0), tuple(p1), dxfattribs={"layer": layers["CENTER"]})
                report.count("centre mark")
            continue
        for seg in ent.segments:
            if seg.kind == "line":
                msp.add_line(tuple(seg.start), tuple(seg.end), dxfattribs=attribs)
                report.count("LINE")
            elif seg.kind == "arc":
                _, _, a0, a1 = arc_angles_ccw(seg)
                msp.add_arc(tuple(seg.center), seg.radius, a0, a1, dxfattribs=attribs)
                report.count("ARC")
            elif options.splines_as_polylines:
                pts = spline_samples(seg, loop.tol or 0.01)
                msp.add_lwpolyline([tuple(p) for p in pts], dxfattribs=attribs)
                report.count("LWPOLYLINE")
            else:
                t, (cu, cv), k = seg.tck
                msp.add_open_spline([(float(x), float(y), 0.0) for x, y in zip(cu, cv)], degree=k, knots=[float(v) for v in t],
                                    dxfattribs=attribs)
                report.count("SPLINE")


def export_dxf(sections: list[Section], path: str | Path, options: ExportOptions | None = None) -> ExportReport:
    """Write sections as DXF. separate_files: '<stem>_<tag>.dxf' per section, else one file."""
    options = options or ExportOptions()
    report = ExportReport()
    path = Path(path)
    if options.separate_files and len(sections) > 1:
        for section in sections:
            doc = _new_doc()
            _write_section(doc.modelspace(), section, _layers(doc, ""), options, report)
            target = path.with_name(f"{path.stem}_{section_tag(section)}{path.suffix or '.dxf'}")
            doc.saveas(target)
            report.files.append(str(target))
    else:
        doc = _new_doc()
        msp = doc.modelspace()
        for section in sections:
            prefix = f"{section_tag(section)}_" if len(sections) > 1 else ""
            _write_section(msp, section, _layers(doc, prefix), options, report)
        doc.saveas(path)
        report.files.append(str(path))
    return report
