"""Cross-section engine (module 4): planar slicing, loop ordering, curve fitting.

Slicing is edge-based: each mesh edge that crosses the plane yields exactly one
intersection point, shared by both adjacent triangles. A crossing triangle
therefore links two edge-points, and loops are chained through edge ids, which
is exact (no float tolerance matching of segment endpoints).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from matplotlib.path import Path as _MplPath
from scipy.interpolate import splev, splprep

from .mesh import MeshData
from .profile import Segment, fit_circle, segment_profile

__all__ = ["ProfileLoop", "Section", "Slicer", "fit_circle", "fit_loop", "make_section", "section_array"]

AXIS_NAMES = ("X", "Y", "Z")
# In-plane (u, v) axes per section normal, right-handed with the normal:
# X-normal -> (Y, Z), Y-normal -> (Z, X), Z-normal -> (X, Y).
PLANE_UV = {0: (1, 2), 1: (2, 0), 2: (0, 1)}
MIN_CIRCLE_POINTS = 8
SPLINE_ATTEMPTS = 6  # smoothing refinements before falling back to interpolation (s = 0)


@dataclass
class ProfileLoop:
    points: np.ndarray  # (n, 2) ordered section points in plane (u, v) coordinates
    closed: bool
    # "circle" | "profile" (lines / arcs / splines) | "spline" (one free-form curve) | "polyline"
    kind: str = "polyline"
    center: np.ndarray | None = None  # circle centre (u, v)
    radius: float = 0.0
    tck: tuple | None = None  # scipy B-spline (knots, [cu, cv], degree)
    segments: list[Segment] = field(default_factory=list)  # "profile": G0-connected entities
    max_dev: float = 0.0  # max distance from the section points to the fitted curve
    tol: float = 0.0  # profile tolerance the loop was fitted with (exporters re-fit within it)
    outer: bool = True  # outer boundary (even nesting depth) vs hole

    @property
    def n_control_points(self) -> int:
        return 0 if self.tck is None else len(self.tck[1][0])

    def sample(self, n: int | None = None) -> np.ndarray:
        """Points along the fitted curve (u, v), closed loops repeat the first point."""
        if self.kind == "circle":
            t = np.linspace(0.0, 2.0 * np.pi, n or 128)
            return self.center + self.radius * np.column_stack((np.cos(t), np.sin(t)))
        if self.kind == "spline":
            u = np.linspace(0.0, 1.0, n or max(200, 8 * self.n_control_points))
            return np.column_stack(splev(u, self.tck))
        if self.kind == "profile":
            parts = [seg.sample() for seg in self.segments]
            return np.vstack([parts[0]] + [p[1:] for p in parts[1:]])
        return np.vstack((self.points, self.points[:1])) if self.closed else self.points

    def summary(self) -> str:
        role = "outer" if self.outer else "hole"
        state = "closed" if self.closed else "OPEN"
        if self.kind == "circle":
            return f"○ Ø{2 * self.radius:.4f} at ({self.center[0]:.4f}, {self.center[1]:.4f})  {role}"
        if self.kind == "profile":
            counts = {k: sum(s.kind == k for s in self.segments) for k in ("line", "arc", "spline")}
            radii = sorted({round(s.radius, 3) for s in self.segments if s.kind == "arc"})
            arcs = f" (R {', '.join(f'{r:g}' for r in radii[:4])}{'…' if len(radii) > 4 else ''})" if radii else ""
            return (f"▭ {counts['line']} lines, {counts['arc']} arcs{arcs}, {counts['spline']} splines, "
                    f"dev {self.max_dev:.4f}  {state} {role}")
        if self.kind == "spline":
            return f"∿ spline {self.n_control_points} cp, dev {self.max_dev:.4f}  {state} {role}"
        return f"polyline {len(self.points)} pts"


@dataclass
class Section:
    axis: int  # 0 / 1 / 2 = section plane normal along X / Y / Z
    offset: float  # plane position along the axis
    loops: list[ProfileLoop] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"{AXIS_NAMES[self.axis]} = {self.offset:.4f}"

    def to_3d(self, uv: np.ndarray) -> np.ndarray:
        """Lift in-plane (u, v) coordinates back to world XYZ."""
        iu, iv = PLANE_UV[self.axis]
        out = np.empty((len(uv), 3))
        out[:, self.axis] = self.offset
        out[:, iu] = uv[:, 0]
        out[:, iv] = uv[:, 1]
        return out

    @property
    def circles(self) -> list[ProfileLoop]:
        return [lp for lp in self.loops if lp.kind == "circle"]


# ---------------------------------------------------------------------- slicing


class Slicer:
    """Precomputes the mesh edge topology once; slice() is then a few vector ops."""

    def __init__(self, mesh: MeshData) -> None:
        self.mesh = mesh
        f = mesh.faces
        e = np.stack((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]), axis=1)  # (M, 3, 2)
        e = np.sort(e, axis=2)
        key = e[..., 0] * np.int64(mesh.n_vertices) + e[..., 1]
        unique, inverse = np.unique(key.ravel(), return_inverse=True)
        self.edges = np.column_stack((unique // mesh.n_vertices, unique % mesh.n_vertices))  # (E, 2)
        self.face_edges = inverse.reshape(-1, 3)  # (M, 3) unique edge id per face side

    def slice(self, axis: int, offset: float) -> list[tuple[np.ndarray, bool]]:
        """Ordered intersection polylines as (points (n, 3), closed)."""
        v = self.mesh.vertices
        d = v[:, axis] - offset
        # Vertices on the plane are nudged to the + side: no degenerate crossings.
        scale = max(1.0, float(np.abs(v[:, axis]).max()))
        d[np.abs(d) < 1e-12 * scale] = 1e-12 * scale
        side = d > 0

        e = self.edges
        edge_cross = side[e[:, 0]] != side[e[:, 1]]
        fe = self.face_edges
        fcross = edge_cross[fe]  # (M, 3)
        faces = np.flatnonzero(fcross.sum(axis=1) == 2)
        if not faces.size:
            return []
        pairs = fe[faces][fcross[faces]].reshape(-1, 2)  # each crossing face links two edges

        nodes, local = np.unique(pairs, return_inverse=True)
        local = local.reshape(-1, 2)
        a, b = e[nodes, 0], e[nodes, 1]
        t = (d[a] / (d[a] - d[b]))[:, None]
        points = v[a] + t * (v[b] - v[a])
        points[:, axis] = offset
        return [(points[chain], closed) for chain, closed in _chain(local, len(nodes))]


def _chain(links: np.ndarray, n: int) -> list[tuple[np.ndarray, bool]]:
    """Walk a graph of max-degree-2 nodes into ordered chains (closed loops / open paths)."""
    # Neighbour table: every link in both directions, ranked within its source node.
    src = np.concatenate((links[:, 0], links[:, 1]))
    dst = np.concatenate((links[:, 1], links[:, 0]))
    order = np.argsort(src, kind="stable")
    src, dst = src[order], dst[order]
    group_start = np.searchsorted(src, src, side="left")
    rank = np.arange(len(src)) - group_start
    ok = rank < 2  # non-manifold extras beyond two neighbours are dropped
    nbr = np.full((n, 2), -1, dtype=np.int64)
    nbr[src[ok], rank[ok]] = dst[ok]
    deg = np.bincount(src, minlength=n)

    visited = np.zeros(n, dtype=bool)
    chains = []
    # Open paths first (start at degree-1 ends), then the remaining closed loops.
    starts = list(np.flatnonzero(deg == 1)) + list(range(n))
    for s in starts:
        if visited[s]:
            continue
        path = [s]
        visited[s] = True
        prev, cur = -1, s
        while True:
            nxt = nbr[cur, 0] if nbr[cur, 0] != prev else nbr[cur, 1]
            if nxt < 0 or visited[nxt]:
                closed = nxt == s and len(path) > 2
                break
            visited[nxt] = True
            path.append(nxt)
            prev, cur = cur, nxt
        chains.append((np.array(path), closed))
    return chains


# ---------------------------------------------------------------------- curve fitting


def _dedupe(uv: np.ndarray, closed: bool, eps: float) -> np.ndarray:
    keep = np.r_[True, np.linalg.norm(np.diff(uv, axis=0), axis=1) > eps]
    uv = uv[keep]
    if closed and len(uv) > 1 and np.linalg.norm(uv[0] - uv[-1]) <= eps:
        uv = uv[:-1]
    return uv


def fit_loop(uv: np.ndarray, closed: bool, circle_tol: float, spline_tol: float, segment: bool = True) -> ProfileLoop:
    """Circle if it is one; else lines / arcs / splines (segment=True); else one free-form B-spline.

    spline_tol is the profile tolerance: max deviation for every entity type.
    """
    uv = _dedupe(uv, closed, eps=1e-9 * max(1.0, float(np.abs(uv).max())))
    loop = ProfileLoop(points=uv, closed=closed, tol=spline_tol)
    if closed and len(uv) >= MIN_CIRCLE_POINTS:
        c, r, dev = fit_circle(uv)
        if dev <= circle_tol:
            loop.kind, loop.center, loop.radius, loop.max_dev = "circle", c, r, dev
            return loop
    if len(uv) < 4:
        return loop
    if segment:
        segments = segment_profile(uv, closed, spline_tol)
        if segments is not None:
            loop.kind, loop.segments = "profile", segments
            loop.max_dev = max(s.max_dev for s in segments)
            return loop
    pts = np.vstack((uv, uv[:1])) if closed else uv
    # FITPACK's smoothing factor bounds the sum of squared residuals (an RMS target), so
    # sharp corners can still overshoot. spline_tol is a MAX deviation: tighten s until met.
    s = len(pts) * (0.5 * spline_tol) ** 2
    for attempt in range(SPLINE_ATTEMPTS):
        final = attempt == SPLINE_ATTEMPTS - 1
        tck, u = splprep(pts.T, s=0.0 if final else s, per=1 if closed else 0, k=3, quiet=2)
        dev = float(np.linalg.norm(np.column_stack(splev(u, tck)) - pts, axis=1).max())
        if dev <= spline_tol:
            break
        s /= 8.0
    loop.kind, loop.tck, loop.max_dev = "spline", tck, dev
    return loop


def _mark_nesting(loops: list[ProfileLoop]) -> None:
    """Even containment depth = outer boundary, odd = hole (closed loops only)."""
    closed = [lp for lp in loops if lp.closed]
    paths = [_MplPath(lp.points) for lp in closed]
    for i, lp in enumerate(closed):
        probe = lp.points[:1]
        depth = sum(1 for j, p in enumerate(paths) if j != i and p.contains_points(probe)[0])
        lp.outer = depth % 2 == 0


# ---------------------------------------------------------------------- public API


def make_section(
    slicer: Slicer, axis: int, offset: float, circle_tol: float, spline_tol: float, segment: bool = True
) -> Section:
    section = Section(axis=axis, offset=float(offset))
    iu, iv = PLANE_UV[axis]
    for pts3, closed in slicer.slice(axis, offset):
        if len(pts3) < 2:
            continue
        section.loops.append(fit_loop(pts3[:, [iu, iv]], closed, circle_tol, spline_tol, segment))
    _mark_nesting(section.loops)
    return section


def section_offsets(start: float, end: float, interval: float) -> np.ndarray:
    if interval <= 0:
        raise ValueError("Interval must be positive.")
    lo, hi = min(start, end), max(start, end)
    n = int(np.floor((hi - lo) / interval + 1e-9)) + 1
    return lo + interval * np.arange(n)


def section_array(
    mesh: MeshData, axis: int, start: float, end: float, interval: float, circle_tol: float, spline_tol: float,
    segment: bool = True,
) -> list[Section]:
    slicer = Slicer(mesh)
    sections = [
        make_section(slicer, axis, z, circle_tol, spline_tol, segment) for z in section_offsets(start, end, interval)
    ]
    return [s for s in sections if s.loops]
