"""2D profile segmentation (module 4): ordered section points -> lines, arcs, splines.

Pipeline for one loop:
  1. growth: from each start, the longest run fitting a line or an arc within tol
     (doubling + binary search). Runs do not share points, so at a sharp corner
     each edge keeps only its own samples; unfittable stretches become splines.
  2. clean-up: merge compatible neighbours, re-assign boundary points to the better
     carrier, drop sub-MIN_PTS gaps between analytic neighbours.
  3. genuineness: a real line / arc leaves noise-like residuals (small, random
     signs); a piece merely approximating a free-form curve uses the tolerance
     band systematically (few sign runs). Those become splines.
  4. geometry: junctions come from the fitted carriers (line/line intersection,
     exact tangent points at fillets); arcs are rebuilt through their junctions and
     fitted midpoint, spline ends are snapped, so neighbours share endpoints exactly
     (G0-closed contours for CAD / CNC). Pieces that still miss 2x tol are demoted
     to splines and the loop is rebuilt.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import splev, splprep
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

MIN_PTS = 4  # points needed before a run counts as a line or an arc
MIN_SWEEP_DEG = 2.0  # flatter "arcs" are lines
QUIET_RMS = 0.1  # residual RMS below this fraction of tol: genuine without further tests
SYSTEMATIC_MIN_PTS = 12  # fewer points: no meaningful smoothing, use SHORT_RMS
SHORT_RMS = 0.25  # short pieces: genuine only if this quiet (approximations measure >= ~0.3)
SYSTEMATIC_TOL = 0.15  # smoothed residual allowed beyond noise, as a fraction of tol
NOISE_K = 5.0  # noise allowance in the moving average (generous: correlated section noise)
RUNS_TEST_MIN = 15  # "short" piece limit for leftover absorption
LEFTOVER_DEG = 15.0  # a short piece continuing a spline within this angle belongs to it
FILLET_TANGENCY_TOL = 3.0  # arc touching genuine neighbours within this many tol (+2% R) = fillet
FILLET_RMS = 0.5  # ...kept as an arc while its residual RMS stays within this fraction of tol
DEMOTE_DEV = 2.0  # analytic pieces missing tol by this factor after junction snapping -> spline
SPLINE_ATTEMPTS = 6


# ---------------------------------------------------------------------- primitive fits


def fit_line(p: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Total least squares line: (point, unit direction, max |distance|)."""
    c = p.mean(axis=0)
    q = p - c
    _, _, vt = np.linalg.svd(q, full_matrices=False)
    d = vt[0]
    return c, d, float(np.abs(q @ np.array([-d[1], d[0]])).max())


def fit_circle_kasa(p: np.ndarray) -> tuple[np.ndarray, float]:
    A = np.column_stack((p, np.ones(len(p))))
    sol = np.linalg.lstsq(A, (p**2).sum(axis=1), rcond=None)[0]
    c = sol[:2] / 2.0
    return c, float(np.sqrt(max(sol[2] + c @ c, 0.0)))


def fit_circle(p: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Kasa fit refined geometrically. Returns (centre, radius, max |radial residual|)."""
    c0, r0 = fit_circle_kasa(p)
    res = least_squares(lambda x: np.linalg.norm(p - x[:2], axis=1) - x[2], np.r_[c0, r0])
    c, r = res.x[:2], abs(res.x[2])
    return c, float(r), float(np.abs(np.linalg.norm(p - c, axis=1) - r).max())


def _angle(v: np.ndarray) -> float:
    return float(np.arctan2(v[1], v[0]))


def _sweep(p: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Unwrapped polar angles of p about c, in point order."""
    return np.unwrap(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]))


def _circle_3pt(a, b, c) -> tuple[np.ndarray, float] | None:
    d = 2.0 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    if abs(d) < 1e-12:
        return None
    sa, sb, sc = a @ a, b @ b, c @ c
    ux = (sa * (b[1] - c[1]) + sb * (c[1] - a[1]) + sc * (a[1] - b[1])) / d
    uy = (sa * (c[0] - b[0]) + sb * (a[0] - c[0]) + sc * (b[0] - a[0])) / d
    center = np.array([ux, uy])
    return center, float(np.linalg.norm(a - center))


# ---------------------------------------------------------------------- segments


@dataclass
class Segment:
    kind: str  # "line" | "arc" | "spline"
    start: np.ndarray  # (2,)
    end: np.ndarray  # (2,)
    center: np.ndarray | None = None  # arc
    radius: float = 0.0  # arc
    ccw: bool = True  # arc direction from start to end
    tck: tuple | None = None  # spline (clamped: first / last coefficient = start / end)
    n_points: int = 0
    max_dev: float = 0.0

    @property
    def sweep_deg(self) -> float:
        if self.kind != "arc":
            return 0.0
        a0, a1 = _angle(self.start - self.center), _angle(self.end - self.center)
        sweep = (a1 - a0) % (2 * np.pi) if self.ccw else (a0 - a1) % (2 * np.pi)
        return float(np.degrees(sweep))

    @property
    def length(self) -> float:
        if self.kind == "arc":
            return float(np.radians(self.sweep_deg) * self.radius)
        return float(np.linalg.norm(np.diff(self.sample(), axis=0), axis=1).sum())

    def sample(self, n: int | None = None) -> np.ndarray:
        if self.kind == "line":
            return np.vstack((self.start, self.end))
        if self.kind == "arc":
            a0 = _angle(self.start - self.center)
            sweep = np.radians(self.sweep_deg) * (1 if self.ccw else -1)
            t = a0 + np.linspace(0.0, sweep, n or max(8, int(abs(np.degrees(sweep)) / 3)))
            return self.center + self.radius * np.column_stack((np.cos(t), np.sin(t)))
        u = np.linspace(0.0, 1.0, n or max(40, 8 * len(self.tck[1][0])))
        return np.column_stack(splev(u, self.tck))

    def distance(self, p: np.ndarray) -> np.ndarray:
        """Distance from points to this (bounded) segment."""
        if self.kind == "line":
            d = self.end - self.start
            t = np.clip((p - self.start) @ d / max(d @ d, 1e-30), 0.0, 1.0)
            return np.linalg.norm(p - (self.start + t[:, None] * d), axis=1)
        if self.kind == "arc":
            return np.abs(np.linalg.norm(p - self.center, axis=1) - self.radius)
        return cKDTree(self.sample(max(400, 20 * len(p)))).query(p)[0]


# ---------------------------------------------------------------------- runs


@dataclass
class _Run:
    kind: str
    points: np.ndarray  # contiguous, not shared with neighbours
    c: np.ndarray | None = None  # carrier: line point / circle centre
    d: np.ndarray | None = None  # line direction (along the data)
    r: float = 0.0  # circle radius

    def fit(self) -> _Run:
        if self.kind == "line":
            self.c, self.d, _ = fit_line(self.points)
            if (self.points[-1] - self.points[0]) @ self.d < 0:
                self.d = -self.d
        elif self.kind == "arc":
            self.c, self.r, _ = fit_circle(self.points)
        return self

    def residuals(self, p: np.ndarray | None = None) -> np.ndarray:
        p = self.points if p is None else p
        if self.kind == "line":
            return (p - self.c) @ np.array([-self.d[1], self.d[0]])
        return np.linalg.norm(p - self.c, axis=1) - self.r


def _line_ok(p: np.ndarray, tol: float) -> bool:
    return fit_line(p)[2] <= tol


def _arc_ok(p: np.ndarray, tol: float, r_max: float) -> bool:
    # A window that is straight within tol does not rule an arc out: dense sections put
    # many collinear points on each facet chord, and a short stretch of any arc is
    # "straight". Let the arc keep growing; the line/arc tie-break still prefers the line.
    if _line_ok(p, tol):
        return True
    c, r = fit_circle_kasa(p)
    if not (0.0 < r <= r_max) or np.abs(np.linalg.norm(p - c, axis=1) - r).max() > tol:
        return False
    ang = _sweep(p, c)
    sweep = abs(ang[-1] - ang[0])
    if not np.radians(MIN_SWEEP_DEG) <= sweep < 2 * np.pi - 1e-6:
        return False
    steps = np.diff(ang) * np.sign(ang[-1] - ang[0])
    return np.mean(steps >= 0) >= 0.9  # angularly monotonic (noise tolerant)


def _extent(pts: np.ndarray, i: int, ok) -> int:
    """Largest j such that pts[i:j+1] passes ok (j = i if even MIN_PTS fails)."""
    last = len(pts) - 1
    j = i + MIN_PTS - 1
    if j > last or not ok(pts[i : j + 1]):
        return i
    good, step = j, MIN_PTS
    while good < last:  # doubling
        j = min(good + step, last)
        if not ok(pts[i : j + 1]):
            break
        good, step = j, step * 2
    else:
        return good
    lo, hi = good, j  # lo passes, hi fails
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if ok(pts[i : mid + 1]) else (lo, mid)
    return lo


def _grow(pts: np.ndarray, tol: float, r_max: float) -> list[_Run]:
    runs, i, spline_start = [], 0, None
    last = len(pts) - 1
    while i <= last:
        jl = _extent(pts, i, lambda p: _line_ok(p, tol))
        ja = _extent(pts, i, lambda p: _arc_ok(p, tol, r_max))
        kind, j = ("line", jl) if jl >= ja - 1 else ("arc", ja)  # simplicity bias
        if j - i + 1 >= MIN_PTS:
            if spline_start is not None:
                runs.append(_Run("spline", pts[spline_start:i]))
                spline_start = None
            runs.append(_Run(kind, pts[i : j + 1]))
            i = j + 1
        else:
            spline_start = i if spline_start is None else spline_start
            i += 1
    if spline_start is not None:
        runs.append(_Run("spline", pts[spline_start:]))
    return runs


def _fits(kind: str, p: np.ndarray, tol: float, r_max: float) -> bool:
    return kind == "spline" or (_line_ok(p, tol) if kind == "line" else _arc_ok(p, tol, r_max))


def _merge(runs: list[_Run], tol: float, r_max: float, closed: bool) -> list[_Run]:
    """Merge neighbours of the same kind whose union still fits (incl. wrap-around)."""
    merged: list[_Run] = []
    for run in runs:
        if merged and merged[-1].kind == run.kind:
            union = np.vstack((merged[-1].points, run.points))
            if _fits(run.kind, union, tol, r_max):
                merged[-1] = _Run(run.kind, union)
                continue
        merged.append(run)
    if closed and len(merged) > 1 and merged[0].kind == merged[-1].kind:
        union = np.vstack((merged[-1].points, merged[0].points))
        if _fits(merged[0].kind, union, tol, r_max):
            merged = [_Run(merged[0].kind, union)] + merged[1:-1]
    return merged


def _drop_gaps(runs: list[_Run], closed: bool) -> list[_Run]:
    """Tiny unfittable stretches between two analytic runs are just the junction area."""
    n = len(runs)
    keep = []
    for k, run in enumerate(runs):
        if run.kind == "spline" and len(run.points) < MIN_PTS and n > 2:
            prev, nxt = runs[k - 1] if (closed or k > 0) else None, runs[(k + 1) % n] if (closed or k < n - 1) else None
            if prev is not None and nxt is not None and prev.kind != "spline" and nxt.kind != "spline":
                continue
        keep.append(run)
    return keep


def _refine_boundaries(runs: list[_Run], closed: bool, passes: int = 2) -> None:
    """Move each analytic/analytic boundary to where the two carriers fit best.

    Greedy growth lets a run overshoot into its neighbour by every sample within tol of
    its carrier; on dense sections that is many points (e.g. ~sqrt(2 R tol) of a fillet).
    The split is searched over up to a third of each run, by prefix sums of |residual|,
    and repeated so the refitted carriers settle.
    """
    n = len(runs)
    for _ in range(passes):
        for k in range(n if closed else n - 1):
            a, b = runs[k], runs[(k + 1) % n]
            if a is b or "spline" in (a.kind, b.kind):
                continue
            wa = min(len(a.points) // 3, len(a.points) - MIN_PTS)
            wb = min(len(b.points) // 3, len(b.points) - MIN_PTS)
            if wa <= 0 and wb <= 0:
                continue
            wa, wb = max(wa, 0), max(wb, 0)
            zone = np.vstack((a.points[len(a.points) - wa :], b.points[:wb]))
            ra, rb = np.abs(a.residuals(zone)), np.abs(b.residuals(zone))
            costs = np.r_[0.0, np.cumsum(ra)] + np.r_[np.cumsum(rb[::-1])[::-1], 0.0]
            split = int(np.argmin(costs))
            a.points = np.vstack((a.points[: len(a.points) - wa], zone[:split]))
            b.points = np.vstack((zone[split:], b.points[wb:]))
            a.fit()
            b.fit()


def _genuine(run: _Run, tol: float) -> bool:
    """Real primitive vs approximation of a free-form stretch.

    An approximation leaves a *systematic* residual (it bows away and back within the
    tolerance band, over the whole piece); a real primitive leaves only noise. A wide
    moving average keeps the bow and averages the noise away. The noise level comes from
    second differences (MAD), which ignore any smooth bow, so noise-free approximations
    get no allowance. Section points share mesh vertices, so their noise is correlated over
    a few neighbours: the allowance for noise left in the average is generous (NOISE_K).
    """
    r = run.residuals()
    rms = np.sqrt(np.mean(r**2))
    if rms <= QUIET_RMS * tol:
        return True
    n = len(r)
    if n < SYSTEMATIC_MIN_PTS:
        return rms <= SHORT_RMS * tol
    w = max(5, n // 4) | 1  # odd window, a quarter of the piece
    systematic = float(np.abs(np.convolve(r, np.ones(w) / w, mode="valid")).max())
    noise = float(np.median(np.abs(np.diff(r, 2))) / (0.6745 * np.sqrt(6.0)))
    return systematic <= SYSTEMATIC_TOL * tol + NOISE_K * noise / np.sqrt(w)


def _tangency_gap(arc: _Run, other: _Run) -> float:
    """How far the arc's circle is from touching the neighbour's carrier (0 = tangent)."""
    if other.kind == "line":
        foot = other.c + ((arc.c - other.c) @ other.d) * other.d
        return abs(float(np.linalg.norm(arc.c - foot)) - arc.r)
    D = float(np.linalg.norm(arc.c - other.c))
    return min(abs(D - (arc.r + other.r)), abs(D - abs(arc.r - other.r)))


def _rescue_fillets(runs: list[_Run], genuine: list[bool], closed: bool, tol: float) -> None:
    """A short noisy arc tangent to genuine neighbours on both sides is a fillet, not free-form.

    Tangency is judged from the fitted carriers (centre-to-line distance vs radius), which is
    far steadier than the end-point tangent directions of a short noisy arc.
    """
    n = len(runs)
    for k, run in enumerate(runs):
        if run.kind != "arc" or genuine[k] or not (closed or 0 < k < n - 1):
            continue
        prev, nxt = runs[k - 1], runs[(k + 1) % n]
        if not (genuine[k - 1] and genuine[(k + 1) % n]) or "spline" in (prev.kind, nxt.kind):
            continue
        slack = FILLET_TANGENCY_TOL * tol + 0.02 * run.r
        r = run.residuals()
        if (_tangency_gap(run, prev) <= slack and _tangency_gap(run, nxt) <= slack
                and np.sqrt(np.mean(r**2)) <= FILLET_RMS * tol):
            genuine[k] = True


def _tangent(run: _Run, at_end: bool) -> np.ndarray:
    """Unit direction of travel at a run's end (data order)."""
    p = run.points
    if run.kind == "line":
        return run.d
    if run.kind == "arc":
        q = p[-1] if at_end else p[0]
        radial = q - run.c
        t = np.array([-radial[1], radial[0]]) / max(np.linalg.norm(radial), 1e-30)
        ang = _sweep(p, run.c)
        return t if ang[-1] >= ang[0] else -t
    k = min(3, len(p) - 1)
    v = p[-1] - p[-1 - k] if at_end else p[k] - p[0]
    return v / max(np.linalg.norm(v), 1e-30)


def _absorb_leftovers(runs: list[_Run], closed: bool, tol: float) -> None:
    """Analytic pieces that are really part of a neighbouring free-form stretch.

    Short pieces go if sandwiched between splines or continuing one tangentially. Longer
    pieces go if they continue a spline tangentially and are not exact (an exact arc next to
    a spline, e.g. a cam dwell, is design intent). Repeats until stable, since absorbing one
    piece can expose the next.
    """
    cos_max = np.cos(np.radians(LEFTOVER_DEG))
    changed = True
    while changed:
        changed = False
        n = len(runs)
        for k, run in enumerate(runs):
            if run.kind == "spline":
                continue
            prev = runs[k - 1] if (closed or k > 0) else None
            nxt = runs[(k + 1) % n] if (closed or k < n - 1) else None
            sandwiched = prev is not None and nxt is not None and prev.kind == nxt.kind == "spline"
            continues = any(
                nb is not None and nb is not run and nb.kind == "spline" and len(nb.points) > 1
                and _tangent(nb, at_end) @ _tangent(run, not at_end) > cos_max
                for nb, at_end in ((prev, True), (nxt, False))
            )
            quiet = np.sqrt(np.mean(run.residuals() ** 2)) <= QUIET_RMS * tol
            short = len(run.points) < RUNS_TEST_MIN
            if (short and (sandwiched or continues)) or (continues and not quiet):
                run.kind = "spline"
                changed = True


def _coalesce_splines(runs: list[_Run], closed: bool) -> list[_Run]:
    out: list[_Run] = []
    for run in runs:
        if out and out[-1].kind == run.kind == "spline":
            out[-1] = _Run("spline", np.vstack((out[-1].points, run.points)))
        else:
            out.append(run)
    if closed and len(out) > 1 and out[0].kind == out[-1].kind == "spline":
        out = [_Run("spline", np.vstack((out[-1].points, out[0].points)))] + out[1:-1]
    return out


# ---------------------------------------------------------------------- geometry


def _project(run: _Run, q: np.ndarray) -> np.ndarray:
    if run.kind == "line":
        return run.c + ((q - run.c) @ run.d) * run.d
    if run.kind == "arc":
        v = q - run.c
        return run.c + run.r * v / max(np.linalg.norm(v), 1e-30)
    return q


def _junction(a: _Run, b: _Run, tol: float) -> np.ndarray:
    """Shared endpoint of consecutive runs, from their carriers."""
    q = 0.5 * (a.points[-1] + b.points[0])  # the boundary between the two runs' samples
    if a.kind == "spline" and b.kind == "spline":
        return q
    if a.kind == "spline":  # the analytic piece owns the junction: its own end sample, on its
        return _project(b, b.points[0])  # carrier (the spline bends to meet it)
    if b.kind == "spline":
        return _project(a, a.points[-1])
    line, other = (a, b) if a.kind == "line" else (b, a)
    candidates = []
    if a.kind == "line" and b.kind == "line":
        cross = a.d[0] * b.d[1] - a.d[1] * b.d[0]
        if abs(cross) > np.sin(np.radians(0.5)):
            w = b.c - a.c
            candidates.append(a.c + ((w[0] * b.d[1] - w[1] * b.d[0]) / cross) * a.d)
    elif line.kind == "line":  # line + circle
        foot = line.c + ((other.c - line.c) @ line.d) * line.d
        dist = np.linalg.norm(other.c - foot)
        if dist >= other.r - tol:  # tangent (fillet) or just missing: the tangent point
            candidates.append(foot)
        else:
            h = np.sqrt(other.r**2 - dist**2)
            candidates += [foot + h * line.d, foot - h * line.d]
    else:  # circle + circle
        v = b.c - a.c
        D = np.linalg.norm(v)
        if D > 1e-12:
            u = v / D
            if abs(a.r - b.r) + tol < D < a.r + b.r - tol:  # proper crossing
                x = (D**2 + a.r**2 - b.r**2) / (2 * D)
                h = np.sqrt(max(a.r**2 - x**2, 0.0))
                base, perp = a.c + x * u, np.array([-u[1], u[0]])
                candidates += [base + h * perp, base - h * perp]
            else:  # tangent arcs (or just missing): touch point on the centre line
                candidates += [a.c + a.r * u, a.c - a.r * u]
    if candidates:
        best = min(candidates, key=lambda p: np.linalg.norm(p - q))
        gap = np.linalg.norm(a.points[-1] - b.points[0])
        if np.linalg.norm(best - q) <= 10.0 * tol + 2.0 * gap:
            return best
    return 0.5 * (_project(a, q) + _project(b, q))


def clamped_spline(p: np.ndarray, j0: np.ndarray, j1: np.ndarray, tol: float) -> Segment:
    """Cubic spline j0 -> p... -> j1 within tol, clamped so its end coefficients are exactly j0 / j1."""
    pts = np.vstack((j0, p, j1))
    pts = pts[np.r_[True, np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-12]]
    if len(pts) < 4:
        return Segment("line", j0, j1, n_points=len(p))
    s = len(pts) * (0.5 * tol) ** 2
    for attempt in range(SPLINE_ATTEMPTS):
        tck, u = splprep(pts.T, s=0.0 if attempt == SPLINE_ATTEMPTS - 1 else s, k=3, quiet=2)
        if np.linalg.norm(np.column_stack(splev(u, tck)) - pts, axis=1).max() <= tol:
            break
        s /= 8.0
    t, (cu, cv), k = tck
    cu, cv = np.array(cu), np.array(cv)
    cu[0], cv[0], cu[-1], cv[-1] = j0[0], j0[1], j1[0], j1[1]  # clamped ends -> exact junctions
    return Segment("spline", j0, j1, tck=(t, [cu, cv], k), n_points=len(p))


def _build(run: _Run, j0: np.ndarray, j1: np.ndarray, tol: float) -> Segment:
    if run.kind == "line":
        return Segment("line", j0, j1, n_points=len(run.points))
    if run.kind == "arc":
        # Sweep j0 -> j1 in the direction the data travels round the fitted centre;
        # rebuild the arc through j0, the fitted circle's midpoint of that sweep, and j1.
        ang = _sweep(run.points, run.c)
        direction = 1.0 if ang[-1] >= ang[0] else -1.0
        a0, a1 = _angle(j0 - run.c), _angle(j1 - run.c)
        mid_ang = a0 + direction * 0.5 * (((a1 - a0) * direction) % (2 * np.pi))
        mid = run.c + run.r * np.array([np.cos(mid_ang), np.sin(mid_ang)])
        circ = _circle_3pt(j0, mid, j1)
        if circ is None:
            return Segment("line", j0, j1, n_points=len(run.points))
        c, r = circ
        ccw = (mid - j0)[0] * (j1 - mid)[1] - (mid - j0)[1] * (j1 - mid)[0] > 0
        return Segment("arc", j0, j1, center=c, radius=r, ccw=bool(ccw), n_points=len(run.points))
    return clamped_spline(run.points, j0, j1, tol)


def _assemble(runs: list[_Run], closed: bool, tol: float) -> list[Segment]:
    m = len(runs)
    if closed:
        joints = [_junction(runs[k - 1], runs[k], tol) for k in range(m)]
        ends = [(joints[k], joints[(k + 1) % m]) for k in range(m)]
    else:
        inner = [_junction(runs[k - 1], runs[k], tol) for k in range(1, m)]
        joints = [_project(runs[0], runs[0].points[0])] + inner + [_project(runs[-1], runs[-1].points[-1])]
        ends = [(joints[k], joints[k + 1]) for k in range(m)]
    built = [(run, _build(run, j0, j1, tol)) for run, (j0, j1) in zip(runs, ends) if np.linalg.norm(j1 - j0) >= 1e-9]
    # Deviation = distance to the profile, not just to the run's own piece: near a tangent
    # junction a sample can sit just past the junction, on the neighbouring entity.
    n = len(built)
    for k, (run, seg) in enumerate(built):
        near = [seg] + [built[(k + o) % n][1] for o in (-1, 1) if closed or 0 <= k + o < n]
        seg.max_dev = float(np.min([s.distance(run.points) for s in near], axis=0).max())
    return [seg for _, seg in built]


# ---------------------------------------------------------------------- public API


def segment_profile(uv: np.ndarray, closed: bool, tol: float) -> list[Segment] | None:
    """Lines / arcs / splines for an ordered profile, or None if it is purely free-form."""
    if len(uv) < 2 * MIN_PTS:
        return None
    r_max = 100.0 * float(np.linalg.norm(uv.max(axis=0) - uv.min(axis=0)))
    pts = uv
    if closed:
        # A closed loop must start at a real primitive boundary, or the piece crossing the
        # start is split in two. Grow once, then restart at the beginning of the longest run.
        first = _grow(pts, tol, r_max)
        longest = int(np.argmax([len(r.points) for r in first]))
        pts = np.roll(pts, -sum(len(r.points) for r in first[:longest]), axis=0)

    runs = _merge(_grow(pts, tol, r_max), tol, r_max, closed)
    runs = _drop_gaps(runs, closed)
    for run in runs:
        run.fit()
    _refine_boundaries(runs, closed)
    genuine = [run.kind != "spline" and _genuine(run, tol) for run in runs]
    _rescue_fillets(runs, genuine, closed, tol)
    for run, ok in zip(runs, genuine):
        if run.kind != "spline" and not ok:
            run.kind = "spline"
    _absorb_leftovers(runs, closed, tol)
    runs = [r.fit() for r in _coalesce_splines(runs, closed)]

    for _ in range(2):  # build; demote analytic pieces that junction snapping pushed out of tol
        if not any(r.kind != "spline" for r in runs):
            return None
        segments = _assemble(runs, closed, tol)
        bad = {id(r) for r, s in zip(runs, segments) if s.kind != "spline" and s.max_dev > DEMOTE_DEV * tol}
        if not bad or len(segments) != len(runs):
            return segments or None
        for r in runs:
            if id(r) in bad:
                r.kind = "spline"
        runs = [r.fit() for r in _coalesce_splines(runs, closed)]
    return segments or None
