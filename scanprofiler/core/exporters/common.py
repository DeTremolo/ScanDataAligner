"""Shared export preparation: every section loop as oriented, exactly connected entities.

Conventions used by all exporters:
  * coordinates are a section's in-plane (u, v) in mm (see sectioning.PLANE_UV); 3D
    exporters lift them with Section.to_3d
  * closed contours are G0-closed exactly (consecutive entities share endpoints)
  * outer contours run counter-clockwise, holes clockwise (CNC cutter-compensation
    convention); open chains are flagged, they are not cuttable contours
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ..profile import Segment, clamped_spline
from ..sectioning import AXIS_NAMES, ProfileLoop, Section

CENTER_MARK_RATIO = 0.35  # centre-mark arm length relative to hole radius


@dataclass
class ExportOptions:
    center_marks: bool = True
    splines_as_polylines: bool = False  # DXF: for CAM software without SPLINE support
    separate_files: bool = False  # DXF: one file per section instead of layers


@dataclass
class ExportReport:
    files: list[str] = field(default_factory=list)
    entities: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def count(self, kind: str, n: int = 1) -> None:
        self.entities[kind] = self.entities.get(kind, 0) + n

    def summary(self) -> str:
        parts = ", ".join(f"{n} {k}" for k, n in sorted(self.entities.items()))
        return f"{len(self.files)} file(s): {parts}" + (f"  ({len(self.warnings)} warning(s))" if self.warnings else "")


@dataclass
class LoopEntities:
    """One loop ready for export: either a full circle or a chain of segments."""

    segments: list[Segment]
    circle: tuple[np.ndarray, float] | None  # (centre, radius) for full holes
    closed: bool
    outer: bool


def section_tag(section: Section) -> str:
    """File / layer safe section name, e.g. 'Z+10.000'."""
    return f"{AXIS_NAMES[section.axis]}{section.offset:+.3f}"


def reverse_segment(seg: Segment) -> Segment:
    if seg.kind == "line":
        return replace(seg, start=seg.end, end=seg.start)
    if seg.kind == "arc":
        return replace(seg, start=seg.end, end=seg.start, ccw=not seg.ccw)
    t, (cu, cv), k = seg.tck
    t = np.asarray(t)
    return replace(
        seg,
        start=seg.end,
        end=seg.start,
        tck=((t[0] + t[-1]) - t[::-1], [np.asarray(cu)[::-1], np.asarray(cv)[::-1]], k),
    )


def _signed_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(x @ np.roll(y, -1) - y @ np.roll(x, -1))


def loop_entities(loop: ProfileLoop) -> LoopEntities:
    if loop.kind == "circle":
        return LoopEntities([], (loop.center, loop.radius), True, loop.outer)

    pts = loop.points
    if loop.kind == "profile":
        segments = list(loop.segments)
    elif loop.kind == "spline":
        # One free-form loop -> one clamped spline closed on itself: every CAD reader
        # evaluates clamped splines identically (periodic knot vectors are less portable).
        tol = loop.tol or max(loop.max_dev, 1e-3)
        if loop.closed:
            segments = [clamped_spline(pts[1:], pts[0], pts[0], tol)]
        else:
            segments = [clamped_spline(pts[1:-1], pts[0], pts[-1], tol)]
    else:  # polyline: too few points to fit anything
        ends = np.vstack((pts, pts[:1])) if loop.closed else pts
        segments = [Segment("line", a, b) for a, b in zip(ends[:-1], ends[1:]) if np.linalg.norm(b - a) > 0]

    if loop.closed and segments:
        area = _signed_area(np.vstack([s.sample() for s in segments]))
        if (area > 0) != loop.outer:  # outer CCW, holes CW
            segments = [reverse_segment(s) for s in reversed(segments)]
    return LoopEntities(segments, None, loop.closed, loop.outer)


def center_mark(center: np.ndarray, radius: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """Two crossing line segments through a hole centre."""
    arm = CENTER_MARK_RATIO * radius
    c = np.asarray(center, float)
    return [(c - [arm, 0.0], c + [arm, 0.0]), (c - [0.0, arm], c + [0.0, arm])]


def arc_angles_ccw(seg: Segment) -> tuple[np.ndarray, np.ndarray, float, float]:
    """(start, end, start_deg, end_deg) of an arc, swapped if needed so it runs CCW."""
    start, end = (seg.start, seg.end) if seg.ccw else (seg.end, seg.start)
    a0 = float(np.degrees(np.arctan2(*(start - seg.center)[::-1])))
    a1 = float(np.degrees(np.arctan2(*(end - seg.center)[::-1])))
    return start, end, a0 % 360.0, a1 % 360.0


def spline_samples(seg: Segment, tol: float) -> np.ndarray:
    """Dense points on a spline segment (for polyline / fit-point representations)."""
    n_cp = len(seg.tck[1][0])
    n = max(40, 6 * n_cp)
    pts = seg.sample(n)
    # refine until the chord error between samples is well inside tol
    for _ in range(4):
        mid = seg.sample(2 * n - 1)[1::2]
        chord_mid = 0.5 * (pts[:-1] + pts[1:])
        if np.linalg.norm(mid - chord_mid, axis=1).max() <= 0.25 * tol:
            break
        n *= 2
        pts = seg.sample(n)
    return pts
