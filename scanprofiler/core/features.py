"""Fitted geometric features: datum planes and cylinders (holes / bosses)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _vec(v: np.ndarray, digits: int = 4) -> str:
    return "(" + ", ".join(f"{x:.{digits}f}" for x in v) + ")"


@dataclass
class PlaneFeature:
    point: np.ndarray  # centroid of the fitted points, on the plane
    normal: np.ndarray  # unit, oriented out of the material
    corners: np.ndarray  # (4, 3) patch spanning the fitted points, for display
    rms: float  # RMS point-to-plane distance
    flatness: float  # peak-to-valley distance of the fitted points
    n_points: int
    source: str  # "landmark" | "ransac"
    id: int = 0
    name: str = ""

    def summary(self) -> str:
        return f"{self.name}  n={_vec(self.normal, 3)}  flat {self.flatness:.4f}"

    def details(self) -> str:
        return (
            f"{self.name}  ({self.source}, {self.n_points:,} pts)\n"
            f"Point   {_vec(self.point)}\n"
            f"Normal  {_vec(self.normal, 5)}\n"
            f"RMS {self.rms:.4f}   Flatness {self.flatness:.4f}"
        )


@dataclass
class CylinderFeature:
    center: np.ndarray  # on the axis, midway along the fitted length
    axis: np.ndarray  # unit direction
    radius: float
    length: float  # extent of the fitted points along the axis
    rms: float  # RMS radial deviation
    n_points: int
    kind: str  # "hole" (concave) | "boss" (convex)
    arc_deg: float = 360.0  # angular coverage of the fitted points
    id: int = 0
    name: str = ""

    @property
    def diameter(self) -> float:
        return 2.0 * self.radius

    def summary(self) -> str:
        return f"{self.name}  Ø{self.diameter:.4f}  axis {_vec(self.axis, 3)}"

    def details(self) -> str:
        return (
            f"{self.name}  ({self.kind}, {self.n_points:,} pts, {self.arc_deg:.0f}° arc)\n"
            f"Ø {self.diameter:.4f}   R {self.radius:.4f}   L {self.length:.3f}\n"
            f"Centre  {_vec(self.center)}\n"
            f"Axis    {_vec(self.axis, 5)}\n"
            f"RMS {self.rms:.4f}"
        )


Feature = PlaneFeature | CylinderFeature
