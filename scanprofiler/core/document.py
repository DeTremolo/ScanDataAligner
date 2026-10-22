"""Application data model: the loaded mesh plus everything derived from it.

All geometry lives in NumPy arrays so later stages (selection, fitting, slicing)
operate on whole arrays instead of per-triangle Python loops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from . import alignment
from .features import Feature, PlaneFeature
from .measure import MeasureResult, measure
from .mesh import MeshData
from .sectioning import Section

__all__ = ["AlignmentState", "Document", "MeasureRef", "Measurement", "MeshData", "ToolMode"]

HISTORY_LIMIT = 20  # undo depth; transforms share face arrays, ~5 MB each at 200k triangles


class ToolMode(Enum):
    """Active viewport tool. NEUTRAL means left-drag pans (SolidWorks paradigm)."""

    NEUTRAL = auto()
    POINT_PICK = auto()
    PAINT = auto()
    BOX_SELECT = auto()
    CIRCLE_SELECT = auto()
    LASSO_SELECT = auto()
    MEASURE = auto()


@dataclass
class AlignmentState:
    """Progress through the 3-2-1 sequence."""

    datum: str | None = None  # "XY" | "XZ" | "YZ" once step 1 is done
    steps: dict[str, str] = field(default_factory=dict)  # "primary" / "secondary" / "tertiary" -> description

    def copy(self) -> AlignmentState:
        return AlignmentState(self.datum, dict(self.steps))


@dataclass
class MeasureRef:
    """One end of a measurement: a fitted feature (by id) or a picked surface point."""

    feature_id: int = -1
    point: np.ndarray | None = None


@dataclass
class Measurement:
    id: int
    a: MeasureRef
    b: MeasureRef
    result: MeasureResult | None = None


@dataclass
class _MeshEdit:
    mesh: MeshData  # mesh before the edit
    label: str


@dataclass
class _TransformEdit:
    mesh: MeshData  # mesh before the transform, restored exactly on undo
    matrix: np.ndarray  # the applied transform
    transform: np.ndarray  # accumulated transform before
    alignment: AlignmentState
    label: str


@dataclass
class Document:
    """Single source of truth shared between the UI and the processing pipelines."""

    mesh: MeshData | None = None

    # Accumulated scanner -> CAD rigid transform (module 3).
    transform: np.ndarray = field(default_factory=lambda: np.eye(4))
    alignment: AlignmentState = field(default_factory=AlignmentState)

    # Per-face boolean mask for paint / box / circle / lasso selection (module 2).
    selection: np.ndarray | None = None

    # Surface landmarks picked in POINT_PICK mode, (K, 3) world coordinates.
    picked_points: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))

    # Fitted primitives (module 2): PlaneFeature / CylinderFeature.
    features: list[Feature] = field(default_factory=list)
    _next_feature_id: int = field(default=1, repr=False)
    _name_counts: dict[str, int] = field(default_factory=dict, repr=False)

    # Section results (module 4). Derived from the mesh: cleared whenever it changes.
    sections: list[Section] = field(default_factory=list)

    # Measurements (module 5): references, recomputed when features or the pose change.
    measurements: list[Measurement] = field(default_factory=list)
    _next_measure_id: int = field(default=1, repr=False)

    # Undo stack of mesh edits and transforms.
    _history: list[_MeshEdit | _TransformEdit] = field(default_factory=list, repr=False)

    def set_mesh(self, mesh: MeshData) -> None:
        self.mesh = mesh
        self.transform = np.eye(4)
        self.alignment = AlignmentState()
        self.selection = np.zeros(mesh.n_faces, dtype=bool)
        self.picked_points = np.empty((0, 3))
        self._history.clear()
        self.features.clear()
        self._name_counts.clear()
        self.sections.clear()
        self.measurements.clear()

    # ------------------------------------------------------------------ edits & undo

    def _push(self, entry) -> None:
        self._history.append(entry)
        del self._history[:-HISTORY_LIMIT]

    def edit_mesh(self, mesh: MeshData, label: str = "Delete faces") -> None:
        """Replace the mesh after a topology edit, keeping the old one for undo."""
        self._push(_MeshEdit(self.mesh, label))
        self.mesh = mesh
        self.selection = np.zeros(mesh.n_faces, dtype=bool)
        self.sections.clear()

    def apply_transform(
        self,
        T: np.ndarray,
        label: str,
        step: tuple[str, str] | None = None,
        datum: str | None = None,
        reset_alignment: bool = False,
    ) -> None:
        """Bake a rigid transform into the mesh, features and picked points.

        datum starts a new 3-2-1 sequence (step 1); step records a later step.
        """
        self._push(_TransformEdit(self.mesh, T, self.transform.copy(), self.alignment.copy(), label))
        self._apply(T)
        self.transform = T @ self.transform
        if reset_alignment:
            self.alignment = AlignmentState()
        if datum is not None:
            self.alignment = AlignmentState(datum=datum)
        if step is not None:
            self.alignment.steps[step[0]] = step[1]

    def _apply(self, T: np.ndarray, mesh: MeshData | None = None) -> None:
        self.mesh = mesh if mesh is not None else alignment.transform_mesh(self.mesh, T)
        self.features = [alignment.transform_feature(f, T) for f in self.features]
        self.picked_points = alignment.apply_points(T, self.picked_points)
        self.sections.clear()
        for m in self.measurements:
            for ref in (m.a, m.b):
                if ref.point is not None:
                    ref.point = alignment.apply_points(T, ref.point[None])[0]
        self.recompute_measurements()

    @property
    def can_undo(self) -> bool:
        return bool(self._history)

    @property
    def undo_label(self) -> str | None:
        return self._history[-1].label if self._history else None

    def undo(self) -> str | None:
        """Revert the last edit; returns its label. Features added since a transform
        are kept and moved back with the inverse transform."""
        if not self._history:
            return None
        entry = self._history.pop()
        if isinstance(entry, _MeshEdit):
            self.mesh = entry.mesh
            self.selection = np.zeros(self.mesh.n_faces, dtype=bool)
            self.sections.clear()
        else:
            self._apply(alignment.invert(entry.matrix), mesh=entry.mesh)
            self.transform = entry.transform
            self.alignment = entry.alignment
        return entry.label

    # ------------------------------------------------------------------ features

    def add_feature(self, feature: Feature) -> Feature:
        """Assign an id and a display name ("Plane 2", "Hole 1", ...) and store it."""
        if isinstance(feature, PlaneFeature):
            label = "Landmark plane" if feature.source == "landmark" else "Plane"
        else:
            label = feature.kind.title()
        self._name_counts[label] = self._name_counts.get(label, 0) + 1
        feature.id = self._next_feature_id
        feature.name = f"{label} {self._name_counts[label]}"
        self._next_feature_id += 1
        self.features.append(feature)
        return feature

    def remove_feature(self, feature_id: int) -> None:
        self.features = [f for f in self.features if f.id != feature_id]
        self.recompute_measurements()  # drops measurements that used it

    def feature(self, feature_id: int) -> Feature | None:
        return next((f for f in self.features if f.id == feature_id), None)

    # ------------------------------------------------------------------ measurements

    def entity(self, ref: MeasureRef) -> Feature | np.ndarray | None:
        return ref.point if ref.point is not None else self.feature(ref.feature_id)

    def add_measurement(self, a: MeasureRef, b: MeasureRef) -> Measurement:
        ea, eb = self.entity(a), self.entity(b)
        if ea is None or eb is None:
            raise ValueError("Measurement end no longer exists.")
        if a.point is None and b.point is None and a.feature_id == b.feature_id:
            raise ValueError("Pick two different entities.")
        m = Measurement(self._next_measure_id, a, b, measure(ea, eb))
        self._next_measure_id += 1
        self.measurements.append(m)
        return m

    def remove_measurement(self, measure_id: int) -> None:
        self.measurements = [m for m in self.measurements if m.id != measure_id]

    def recompute_measurements(self) -> None:
        kept = []
        for m in self.measurements:
            ea, eb = self.entity(m.a), self.entity(m.b)
            if ea is not None and eb is not None:
                m.result = measure(ea, eb)
                kept.append(m)
        self.measurements = kept

    def clear_selection(self) -> None:
        if self.selection is not None:
            self.selection[:] = False
