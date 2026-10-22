"""CAD-free measurement (module 5): feature-to-feature and point distances.

Every measurement is defined by two anchor points p_a, p_b whose separation IS the main
dimension (perpendicular feet, common perpendiculars, ...). So for any pair:
  3D distance = |p_b - p_a|,  dX / dY / dZ = |components of p_b - p_a|
and pair-specific extras (wall clearances, angles) are reported alongside.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .features import CylinderFeature, Feature, PlaneFeature

PARALLEL_DEG = 1.0  # axes / normals closer than this are treated as parallel


@dataclass
class MeasureResult:
    label: str  # e.g. "Hole 1 ↔ Hole 2  centre distance"
    p_a: np.ndarray
    p_b: np.ndarray
    extras: dict[str, float] = field(default_factory=dict)  # name -> value (mm or deg)

    @property
    def distance(self) -> float:
        return float(np.linalg.norm(self.p_b - self.p_a))

    @property
    def deltas(self) -> np.ndarray:
        return np.abs(self.p_b - self.p_a)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _angle_between_lines(a: np.ndarray, b: np.ndarray) -> float:
    """Acute angle between two undirected directions, degrees."""
    return float(np.degrees(np.arccos(min(1.0, abs(float(_unit(a) @ _unit(b)))))))


def _foot_on_line(p: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    d = _unit(d)
    return c + ((p - c) @ d) * d


def _foot_on_plane(p: np.ndarray, plane: PlaneFeature) -> np.ndarray:
    return p - ((p - plane.point) @ plane.normal) * plane.normal


def _name(entity) -> str:
    return entity.name if not isinstance(entity, np.ndarray) else "Point"


def measure(a: Feature | np.ndarray, b: Feature | np.ndarray) -> MeasureResult:
    """Measure between two entities: PlaneFeature, CylinderFeature or a 3D point (ndarray)."""
    # Normalise order: point < plane < cylinder, so each pair has one implementation.
    rank = {np.ndarray: 0, PlaneFeature: 1, CylinderFeature: 2}
    if rank[type(a)] > rank[type(b)]:
        a, b = b, a
    pair = f"{_name(a)} ↔ {_name(b)}"

    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return MeasureResult(f"{pair}  distance", a.copy(), b.copy())

    if isinstance(a, np.ndarray) and isinstance(b, PlaneFeature):
        return MeasureResult(f"{pair}  perpendicular", a.copy(), _foot_on_plane(a, b))

    if isinstance(a, np.ndarray) and isinstance(b, CylinderFeature):
        foot = _foot_on_line(a, b.center, b.axis)
        d = float(np.linalg.norm(a - foot))
        return MeasureResult(f"{pair}  to axis", a.copy(), foot, {"radial gap to surface": d - b.radius})

    if isinstance(a, PlaneFeature) and isinstance(b, PlaneFeature):
        angle = _angle_between_lines(a.normal, b.normal)
        if angle < PARALLEL_DEG:
            # Gap measured from each centroid to the other plane, reported as the mean.
            gap_ab = abs(float((a.point - b.point) @ b.normal))
            gap_ba = abs(float((b.point - a.point) @ a.normal))
            res = MeasureResult(f"{pair}  parallel gap", a.point.copy(), _foot_on_plane(a.point, b))
            res.extras = {"angle deviation (deg)": angle, "gap from other side": gap_ba, "gap difference": gap_ab - gap_ba}
            return res
        return MeasureResult(f"{pair}  angle {angle:.3f}°", a.point.copy(), b.point.copy(), {"angle (deg)": angle})

    if isinstance(a, PlaneFeature) and isinstance(b, CylinderFeature):
        foot = _foot_on_plane(b.center, a)
        res = MeasureResult(f"{pair}  plane to centre", foot, b.center.copy())
        tilt = 90.0 - _angle_between_lines(b.axis, a.normal)  # 0 = axis parallel to plane
        res.extras = {"axis to plane angle (deg)": tilt}
        if abs(tilt) < PARALLEL_DEG:  # axis lies parallel to the plane: wall distances make sense
            d = res.distance
            res.extras.update({"min wall distance": d - b.radius, "max wall distance": d + b.radius})
        else:
            res.extras["axis deviation from normal (deg)"] = _angle_between_lines(b.axis, a.normal)
        return res

    # cylinder - cylinder
    angle = _angle_between_lines(a.axis, b.axis)
    da, db = _unit(a.axis), _unit(b.axis)
    if angle < PARALLEL_DEG:
        # Parallel axes: centre distance = axis separation, measured from A's centre.
        p_a = a.center.copy()
        p_b = _foot_on_line(a.center, b.center, b.axis)
    else:  # skew axes: common perpendicular
        w0 = a.center - b.center
        bb, d, e = da @ db, da @ w0, db @ w0
        denom = 1.0 - bb * bb
        s = (bb * e - d) / denom
        t = (e - bb * d) / denom
        p_a, p_b = a.center + s * da, b.center + t * db
    d = float(np.linalg.norm(p_b - p_a))
    res = MeasureResult(f"{pair}  centre distance", p_a, p_b, {"axis angle (deg)": angle})
    res.extras["min wall clearance"] = d - a.radius - b.radius
    res.extras["max wall distance"] = d + a.radius + b.radius
    return res


# ---------------------------------------------------------------------- picking


def snap(point: np.ndarray, features: list[Feature], tol: float) -> Feature | None:
    """Feature whose surface passes within tol of a picked surface point (closest wins)."""
    best, best_d = None, tol
    for f in features:
        if isinstance(f, CylinderFeature):
            along = (point - f.center) @ f.axis
            if abs(along) > 0.5 * f.length + tol:
                continue
            d = abs(float(np.linalg.norm(point - _foot_on_line(point, f.center, f.axis))) - f.radius)
        else:
            d = abs(float((point - f.point) @ f.normal))
            if d > tol or not _inside_patch(point, f, tol):
                continue
        if d <= best_d:
            best, best_d = f, d
    return best


def _inside_patch(point: np.ndarray, plane: PlaneFeature, margin: float) -> bool:
    c = plane.corners
    u, v = c[1] - c[0], c[3] - c[0]
    q = point - c[0]
    su, sv = (q @ u) / (u @ u), (q @ v) / (v @ v)
    mu, mv = margin / max(np.linalg.norm(u), 1e-12), margin / max(np.linalg.norm(v), 1e-12)
    return -mu <= su <= 1 + mu and -mv <= sv <= 1 + mv
