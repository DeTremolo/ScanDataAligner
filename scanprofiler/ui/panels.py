"""Dockable control panels. Each panel only emits signals; MainWindow routes them."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.alignment import describe
from ..core.features import CylinderFeature, PlaneFeature

PLANES = ("XY", "XZ", "YZ")
AXES = ("X", "Y", "Z")


def _spin(value: float, lo: float, hi: float, step: float, decimals: int = 3, suffix: str = "") -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setSingleStep(step)
    s.setValue(value)
    if suffix:
        s.setSuffix(suffix)
    return s


class SelectionPanel(QWidget):
    """Module 2: brush, selection cleanup, primitive fitting."""

    brush_size_changed = Signal(int)
    select_through_changed = Signal(bool)
    delete_selection_requested = Signal()
    clear_selection_requested = Signal()
    clear_points_requested = Signal()
    fit_landmark_plane_requested = Signal()
    map_plane_requested = Signal(str)  # "XY" | "XZ" | "YZ"
    dbscan_requested = Signal(float, int)  # eps, min_samples
    ransac_plane_requested = Signal()
    ransac_cylinder_requested = Signal()
    feature_selected = Signal(int)  # feature id, -1 for none
    remove_feature_requested = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        self._details: dict[int, str] = {}

        # Brush
        brush_box = QGroupBox("Brush radius  (LMB paint · RMB erase · [ / ])")
        bl = QHBoxLayout(brush_box)
        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setRange(1, 200)
        self.brush_slider.setValue(20)
        self.brush_spin = QSpinBox()
        self.brush_spin.setRange(1, 200)
        self.brush_spin.setValue(20)
        self.brush_spin.setSuffix(" px")
        self.brush_slider.valueChanged.connect(self.brush_spin.setValue)
        self.brush_spin.valueChanged.connect(self.brush_slider.setValue)
        self.brush_spin.valueChanged.connect(self.brush_size_changed)
        bl.addWidget(self.brush_slider, 1)
        bl.addWidget(self.brush_spin)
        root.addWidget(brush_box)

        # Selection editing
        sel_box = QGroupBox("Selection  (LMB add · RMB remove)")
        sl = QGridLayout(sel_box)
        self.selection_label = QLabel()
        self.through_check = QCheckBox("Select through (ignore occlusion)")
        self.through_check.toggled.connect(self.select_through_changed)
        delete_btn = QPushButton("Delete (Del)")
        delete_btn.clicked.connect(self.delete_selection_requested)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear_selection_requested)
        sl.addWidget(self.selection_label, 0, 0, 1, 2)
        sl.addWidget(self.through_check, 1, 0, 1, 2)
        sl.addWidget(delete_btn, 2, 0)
        sl.addWidget(clear_btn, 2, 1)
        root.addWidget(sel_box)

        # Landmark plane
        lm_box = QGroupBox("Landmark plane (≥ 3 picked points)")
        ll = QGridLayout(lm_box)
        self.points_label = QLabel()
        clear_pts_btn = QPushButton("Clear points")
        clear_pts_btn.clicked.connect(self.clear_points_requested)
        ll.addWidget(self.points_label, 2, 0)
        ll.addWidget(clear_pts_btn, 2, 1)
        fit_btn = QPushButton("Fit landmark plane")
        fit_btn.clicked.connect(self.fit_landmark_plane_requested)
        self.map_combo = QComboBox()
        self.map_combo.addItems(PLANES)
        map_btn = QPushButton("Approve → map to")
        map_btn.clicked.connect(lambda: self.map_plane_requested.emit(self.map_combo.currentText()))
        ll.addWidget(fit_btn, 0, 0, 1, 2)
        ll.addWidget(map_btn, 1, 0)
        ll.addWidget(self.map_combo, 1, 1)
        root.addWidget(lm_box)

        # Statistical refinement
        fit_box = QGroupBox("Refine && fit painted region")
        fl = QFormLayout(fit_box)
        self.eps_spin = _spin(0.5, 0.001, 100.0, 0.1, suffix=" mm")
        self.eps_spin.setToolTip("Neighbourhood radius; set to ~3x mesh edge length on import.")
        self.min_samples_spin = QSpinBox()
        self.min_samples_spin.setRange(1, 1000)
        self.min_samples_spin.setValue(10)
        dbscan_btn = QPushButton("DBSCAN: keep largest cluster")
        dbscan_btn.clicked.connect(
            lambda: self.dbscan_requested.emit(self.eps_spin.value(), self.min_samples_spin.value())
        )
        self.tol_spin = _spin(0.1, 0.0001, 10.0, 0.01, decimals=4, suffix=" mm")
        self.tol_spin.setToolTip("Max distance from the fitted surface for a face to count as inlier.")
        self.angle_spin = _spin(25.0, 1.0, 60.0, 1.0, decimals=1, suffix=" °")
        self.angle_spin.setToolTip("Max deviation of a face normal from the fitted surface normal.")
        self.dbscan_first_check = QCheckBox("DBSCAN before fitting")
        self.dbscan_first_check.setChecked(True)
        self.trim_check = QCheckBox("Trim selection to inliers")
        self.trim_check.setChecked(True)
        plane_btn = QPushButton("RANSAC plane")
        plane_btn.clicked.connect(self.ransac_plane_requested)
        cyl_btn = QPushButton("RANSAC cylinder / hole")
        cyl_btn.clicked.connect(self.ransac_cylinder_requested)
        fit_row = QHBoxLayout()
        fit_row.addWidget(plane_btn)
        fit_row.addWidget(cyl_btn)
        fl.addRow("DBSCAN eps", self.eps_spin)
        fl.addRow("Min samples", self.min_samples_spin)
        fl.addRow(dbscan_btn)
        fl.addRow("Fit tolerance", self.tol_spin)
        fl.addRow("Normal deviation", self.angle_spin)
        fl.addRow(self.dbscan_first_check)
        fl.addRow(self.trim_check)
        fl.addRow(fit_row)
        root.addWidget(fit_box)

        # Fitted features
        feat_box = QGroupBox("Features")
        vl = QVBoxLayout(feat_box)
        self.feature_list = QListWidget()
        self.feature_list.setMinimumHeight(90)
        self.feature_list.currentItemChanged.connect(self._on_feature_item)
        self.feature_details = QLabel()
        self.feature_details.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.feature_details.setStyleSheet("font-family: Consolas, monospace;")
        remove_btn = QPushButton("Remove feature")
        remove_btn.clicked.connect(self._on_remove_feature)
        vl.addWidget(self.feature_list)
        vl.addWidget(self.feature_details)
        vl.addWidget(remove_btn)
        root.addWidget(feat_box)

        root.addStretch(1)
        self.set_counts(0, 0, 0)

    def fit_settings(self) -> dict:
        return {
            "eps": self.eps_spin.value(),
            "min_samples": self.min_samples_spin.value(),
            "tol": self.tol_spin.value(),
            "max_angle_deg": self.angle_spin.value(),
            "dbscan_first": self.dbscan_first_check.isChecked(),
            "trim": self.trim_check.isChecked(),
        }

    def set_mesh_scale(self, edge_length: float) -> None:
        """Seed density-dependent defaults from the mesh's mean edge length."""
        self.eps_spin.setValue(3.0 * edge_length)
        self.tol_spin.setValue(max(0.25 * edge_length, 0.005))

    def set_features(self, features, current_id: int | None = None) -> None:
        self._details = {f.id: f.details() for f in features}
        self.feature_list.blockSignals(True)
        self.feature_list.clear()
        for f in features:
            item = QListWidgetItem(f.summary())
            item.setData(Qt.UserRole, f.id)
            self.feature_list.addItem(item)
            if f.id == current_id:
                self.feature_list.setCurrentItem(item)
        self.feature_list.blockSignals(False)
        self._on_feature_item(self.feature_list.currentItem())

    def _on_feature_item(self, item, _previous=None) -> None:
        fid = item.data(Qt.UserRole) if item is not None else -1
        self.feature_details.setText(self._details.get(fid, ""))
        self.feature_selected.emit(fid)

    def _on_remove_feature(self) -> None:
        item = self.feature_list.currentItem()
        if item is not None:
            self.remove_feature_requested.emit(item.data(Qt.UserRole))

    def step_brush(self, delta: int) -> None:
        self.brush_spin.setValue(self.brush_spin.value() + delta)

    def set_counts(self, selected: int, total: int, points: int) -> None:
        self.selection_label.setText(f"{selected:,} / {total:,} faces selected")
        self.points_label.setText(f"{points} point{'s' if points != 1 else ''} picked")


IN_PLANE_AXES = {"XY": ("+X", "-X", "+Y", "-Y"), "XZ": ("+X", "-X", "+Z", "-Z"), "YZ": ("+Y", "-Y", "+Z", "-Z")}


class _FeatureCombo(QComboBox):
    """Combo of document features of the accepted types; keeps its choice across refreshes."""

    def __init__(self, accept: tuple[type, ...]) -> None:
        super().__init__()
        self._accept = accept

    def set_features(self, features, prefer_id: int | None = None) -> None:
        keep = prefer_id if prefer_id is not None and any(f.id == prefer_id for f in features) else self.current_id()
        self.blockSignals(True)
        self.clear()
        for f in features:
            if isinstance(f, self._accept):
                self.addItem(f.name, f.id)
        index = self.findData(keep)
        self.setCurrentIndex(index if index >= 0 else self.count() - 1)
        self.blockSignals(False)

    def current_id(self) -> int:
        data = self.currentData()
        return -1 if data is None else int(data)


class AlignmentPanel(QWidget):
    """Module 3: 3-2-1 datum alignment and incremental shifts."""

    primary_datum_requested = Signal(int, str, bool)  # plane feature id, "XY" | "XZ" | "YZ", flip
    secondary_lock_requested = Signal(int)  # cylinder feature id
    tertiary_lock_requested = Signal(int, str)  # feature id, in-plane target axis ("+X", ...)
    reset_alignment_requested = Signal()
    translate_requested = Signal(int, float)  # axis index, distance
    rotate_requested = Signal(int, float)  # axis index, degrees

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)

        # Step 1
        s1 = QGroupBox("1 · Primary datum: plane → global plane")
        g1 = QGridLayout(s1)
        self.plane_combo = _FeatureCombo((PlaneFeature,))
        self.datum_combo = QComboBox()
        self.datum_combo.addItems(PLANES)
        self.flip_check = QCheckBox("Flip (material on − side)")
        snap_btn = QPushButton("Snap")
        snap_btn.clicked.connect(
            lambda: self.primary_datum_requested.emit(
                self.plane_combo.current_id(), self.datum_combo.currentText(), self.flip_check.isChecked()
            )
        )
        g1.addWidget(self.plane_combo, 0, 0)
        g1.addWidget(QLabel("→"), 0, 1)
        g1.addWidget(self.datum_combo, 0, 2)
        g1.addWidget(self.flip_check, 1, 0, 1, 2)
        g1.addWidget(snap_btn, 1, 2)
        g1.setColumnStretch(0, 3)
        g1.setColumnStretch(2, 1)
        root.addWidget(s1)

        # Step 2
        s2 = QGroupBox("2 · Centre: cylinder axis through origin")
        g2 = QGridLayout(s2)
        self.cylinder_combo = _FeatureCombo((CylinderFeature,))
        centre_btn = QPushButton("Lock")
        centre_btn.clicked.connect(lambda: self.secondary_lock_requested.emit(self.cylinder_combo.current_id()))
        g2.addWidget(self.cylinder_combo, 0, 0)
        g2.addWidget(centre_btn, 0, 1)
        g2.setColumnStretch(0, 3)
        root.addWidget(s2)

        # Step 3
        s3 = QGroupBox("3 · Clocking: feature direction → axis")
        g3 = QGridLayout(s3)
        self.clock_combo = _FeatureCombo((PlaneFeature, CylinderFeature))
        self.clock_combo.setToolTip("Plane: its normal. Cylinder: direction from the origin to its centre.")
        self.clock_axis_combo = QComboBox()
        clock_btn = QPushButton("Lock")
        clock_btn.clicked.connect(
            lambda: self.tertiary_lock_requested.emit(self.clock_combo.current_id(), self.clock_axis_combo.currentText())
        )
        g3.addWidget(self.clock_combo, 0, 0)
        g3.addWidget(QLabel("→"), 0, 1)
        g3.addWidget(self.clock_axis_combo, 0, 2)
        g3.addWidget(clock_btn, 1, 2)
        g3.setColumnStretch(0, 3)
        g3.setColumnStretch(2, 1)
        root.addWidget(s3)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        reset = QPushButton("Reset to scanner coordinates")
        reset.clicked.connect(self.reset_alignment_requested)
        root.addWidget(self.status_label)
        root.addWidget(reset)

        inc_box = QGroupBox("Incremental shifts (global axes, about origin)")
        il = QGridLayout(inc_box)
        self.trans_step = _spin(1.0, 0.001, 1000.0, 0.1, suffix=" mm")
        self.rot_step = _spin(1.0, 0.001, 180.0, 0.1, suffix=" °")
        il.addWidget(QLabel("Translate step"), 0, 0, 1, 2)
        il.addWidget(self.trans_step, 0, 2, 1, 3)
        il.addWidget(QLabel("Rotate step"), 1, 0, 1, 2)
        il.addWidget(self.rot_step, 1, 2, 1, 3)
        for row, name in enumerate(AXES, start=2):
            il.addWidget(QLabel(name), row, 0)
            for col, (label, sign, rotate) in enumerate(
                (("−T", -1, False), ("+T", 1, False), ("−R", -1, True), ("+R", 1, True)), start=1
            ):
                btn = QPushButton(label)
                btn.setFixedWidth(42)
                btn.clicked.connect(lambda _=False, a=row - 2, s=sign, r=rotate: self._emit_shift(a, s, r))
                il.addWidget(btn, row, col)
        root.addWidget(inc_box)

        tf_box = QGroupBox("Scanner → CAD transform")
        tl = QVBoxLayout(tf_box)
        self.transform_label = QLabel()
        self.transform_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.transform_label.setStyleSheet("font-family: Consolas, monospace;")
        tl.addWidget(self.transform_label)
        root.addWidget(tf_box)
        root.addStretch(1)

        self.set_alignment(None, {}, np.eye(4))

    def _emit_shift(self, axis: int, sign: int, rotate: bool) -> None:
        if rotate:
            self.rotate_requested.emit(axis, sign * self.rot_step.value())
        else:
            self.translate_requested.emit(axis, sign * self.trans_step.value())

    def set_features(self, features, prefer_id: int | None = None) -> None:
        for combo in (self.plane_combo, self.cylinder_combo, self.clock_combo):
            combo.set_features(features, prefer_id)

    def set_alignment(self, datum: str | None, steps: dict[str, str], transform: np.ndarray) -> None:
        current = self.clock_axis_combo.currentText()
        self.clock_axis_combo.clear()
        axes = IN_PLANE_AXES[datum] if datum else ("+X", "-X", "+Y", "-Y", "+Z", "-Z")
        self.clock_axis_combo.addItems(axes)
        if current in axes:
            self.clock_axis_combo.setCurrentText(current)

        lines = []
        for n, key in enumerate(("primary", "secondary", "tertiary"), start=1):
            lines.append(f"{'✓' if key in steps else '○'} {n}  {steps.get(key, '—')}")
        self.status_label.setText("\n".join(lines))

        t, euler = describe(transform)
        rows = "\n".join("  ".join(f"{v:+11.5f}" for v in transform[i]) for i in range(3))
        self.transform_label.setText(
            f"{rows}\n"
            f"T  ({t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f})\n"
            f"R  xyz ({euler[0]:+.4f}°, {euler[1]:+.4f}°, {euler[2]:+.4f}°)"
        )

class SectionPanel(QWidget):
    """Module 4: section array generation and CAD export triggers."""

    auto_range_requested = Signal(int)  # axis index
    section_array_requested = Signal(int, float, float, float)  # axis, start, end, interval
    single_section_requested = Signal(int, float)  # axis, offset
    clear_sections_requested = Signal()
    section_selected = Signal(int)  # row, -1 for none
    export_requested = Signal(str)  # "vbs" | "step" | "dxf"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        self._sections: list = []

        arr_box = QGroupBox("Section array")
        al = QFormLayout(arr_box)
        self.axis_combo = QComboBox()
        self.axis_combo.addItems(AXES)
        self.axis_combo.setCurrentIndex(2)
        self.start_spin = _spin(0.0, -1e6, 1e6, 1.0, suffix=" mm")
        self.end_spin = _spin(100.0, -1e6, 1e6, 1.0, suffix=" mm")
        self.interval_spin = _spin(10.0, 0.001, 1e6, 1.0, suffix=" mm")
        auto_btn = QPushButton("Range from mesh bounds")
        auto_btn.clicked.connect(lambda: self.auto_range_requested.emit(self.axis_combo.currentIndex()))
        gen_btn = QPushButton("Generate section array")
        gen_btn.clicked.connect(
            lambda: self.section_array_requested.emit(
                self.axis_combo.currentIndex(),
                self.start_spin.value(),
                self.end_spin.value(),
                self.interval_spin.value(),
            )
        )
        al.addRow("Normal axis", self.axis_combo)
        al.addRow("Start", self.start_spin)
        al.addRow("End", self.end_spin)
        al.addRow("Interval", self.interval_spin)
        al.addRow(auto_btn)
        al.addRow(gen_btn)
        root.addWidget(arr_box)

        single_box = QGroupBox("Single plane override")
        sl = QHBoxLayout(single_box)
        self.single_spin = _spin(0.0, -1e6, 1e6, 1.0, suffix=" mm")
        single_btn = QPushButton("Add section")
        single_btn.clicked.connect(
            lambda: self.single_section_requested.emit(self.axis_combo.currentIndex(), self.single_spin.value())
        )
        sl.addWidget(self.single_spin, 1)
        sl.addWidget(single_btn)
        root.addWidget(single_box)

        fit_box = QGroupBox("Curve fitting")
        fl = QFormLayout(fit_box)
        self.circle_tol_spin = _spin(0.1, 0.0001, 10.0, 0.01, decimals=4, suffix=" mm")
        self.circle_tol_spin.setToolTip("A closed loop within this max radial deviation becomes a true circle + centre mark.")
        self.spline_tol_spin = _spin(0.05, 0.0001, 10.0, 0.01, decimals=4, suffix=" mm")
        self.spline_tol_spin.setToolTip(
            "Max deviation of every line, arc and spline from the section points.\n"
            "Set to about 4x the scan noise: closer to the noise, fillets degrade to splines."
        )
        self.segment_check = QCheckBox("Lines / arcs / splines (else one spline per loop)")
        self.segment_check.setChecked(True)
        fl.addRow("Circle tolerance", self.circle_tol_spin)
        fl.addRow("Profile tolerance", self.spline_tol_spin)
        fl.addRow(self.segment_check)
        root.addWidget(fit_box)

        res_box = QGroupBox("Sections")
        rl = QVBoxLayout(res_box)
        self.summary_label = QLabel("No sections")
        self.summary_label.setWordWrap(True)
        self.summary_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.section_list = QListWidget()
        self.section_list.setMinimumHeight(110)
        self.section_list.currentRowChanged.connect(self._on_section_row)
        # Fixed-height read-only box: a section can have many loops, a label would grow unbounded.
        self.section_details = QPlainTextEdit()
        self.section_details.setReadOnly(True)
        self.section_details.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.section_details.setStyleSheet("font-family: Consolas, monospace;")
        self.section_details.setFixedHeight(6 * self.section_details.fontMetrics().lineSpacing())
        clear_btn = QPushButton("Clear sections")
        clear_btn.clicked.connect(self.clear_sections_requested)
        rl.addWidget(self.summary_label)
        rl.addWidget(self.section_list)
        rl.addWidget(self.section_details)
        rl.addWidget(clear_btn)
        root.addWidget(res_box)

        exp_box = QGroupBox("Export")
        el = QVBoxLayout(exp_box)
        for key, label in (("vbs", "SolidWorks macro (.vbs)"), ("step", "STEP curve wireframe"), ("dxf", "DXF profiles")):
            btn = QPushButton(label)
            btn.clicked.connect(lambda _=False, k=key: self.export_requested.emit(k))
            el.addWidget(btn)
        root.addWidget(exp_box)
        root.addStretch(1)

    def set_range(self, start: float, end: float) -> None:
        self.start_spin.setValue(start)
        self.end_spin.setValue(end)

    def fit_settings(self) -> tuple[float, float, bool]:
        return self.circle_tol_spin.value(), self.spline_tol_spin.value(), self.segment_check.isChecked()

    def set_sections(self, sections) -> None:
        self._sections = list(sections)
        self.section_list.blockSignals(True)
        self.section_list.clear()
        for s in self._sections:
            n_circ = len(s.circles)
            self.section_list.addItem(f"{s.name}   {len(s.loops)} loop(s), {n_circ} hole(s)")
        self.section_list.blockSignals(False)
        loops = [lp for s in self._sections for lp in s.loops]
        if loops:
            n_open = sum(not lp.closed for lp in loops)
            worst = max(lp.max_dev for lp in loops)
            segs = [seg for lp in loops for seg in lp.segments]
            n_spl = sum(lp.kind == "spline" for lp in loops) + sum(s.kind == "spline" for s in segs)
            self.summary_label.setText(
                f"{len(self._sections)} sections · {len(loops)} loops · "
                f"{sum(lp.kind == 'circle' for lp in loops)} circles · "
                f"{sum(s.kind == 'line' for s in segs)} lines · {sum(s.kind == 'arc' for s in segs)} arcs · "
                f"{n_spl} splines" + (f" · {n_open} OPEN" if n_open else "") + f" · max dev {worst:.4f}"
            )
        else:
            self.summary_label.setText("No sections")
        self.section_details.clear()

    def _on_section_row(self, row: int) -> None:
        if 0 <= row < len(self._sections):
            s = self._sections[row]
            self.section_details.setPlainText(s.name + "\n" + "\n".join(lp.summary() for lp in s.loops))
        else:
            self.section_details.clear()
        self.section_selected.emit(row)


class MeasurementPanel(QWidget):
    """Module 5: feature-to-feature / point measurements with Cartesian breakdown."""

    COLUMNS = ("#", "Pair", "3D", "ΔX", "ΔY", "ΔZ", "Details")

    measure_features_requested = Signal(int, int)  # feature ids
    remove_requested = Signal(int)  # measurement id
    clear_requested = Signal()
    measurement_selected = Signal(int)  # measurement id, -1 for none

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)

        bar = QHBoxLayout()
        self.a_combo = _FeatureCombo((PlaneFeature, CylinderFeature))
        self.b_combo = _FeatureCombo((PlaneFeature, CylinderFeature))
        measure_btn = QPushButton("Measure")
        measure_btn.clicked.connect(
            lambda: self.measure_features_requested.emit(self.a_combo.current_id(), self.b_combo.current_id())
        )
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self._on_remove)
        clear_btn = QPushButton("Clear all")
        clear_btn.clicked.connect(self.clear_requested)
        self.pending_label = QLabel("Measure tool (M): click two features or surface points")
        bar.addWidget(QLabel("Feature"))
        bar.addWidget(self.a_combo, 1)
        bar.addWidget(QLabel("↔"))
        bar.addWidget(self.b_combo, 1)
        bar.addWidget(measure_btn)
        bar.addSpacing(16)
        bar.addWidget(self.pending_label, 2)
        bar.addWidget(remove_btn)
        bar.addWidget(clear_btn)
        root.addLayout(bar)

        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(len(self.COLUMNS) - 1, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.itemSelectionChanged.connect(self._on_selection)
        root.addWidget(self.table)

    def set_features(self, features) -> None:
        self.a_combo.set_features(features)
        self.b_combo.set_features(features)
        if self.b_combo.count() > 1 and self.b_combo.currentIndex() == self.a_combo.currentIndex():
            self.b_combo.setCurrentIndex(0 if self.a_combo.currentIndex() else 1)

    def set_pending(self, text: str | None) -> None:
        self.pending_label.setText(text or "Measure tool (M): click two features or surface points")

    def set_measurements(self, measurements, current_id: int | None = None) -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(len(measurements))
        for row, m in enumerate(measurements):
            r = m.result
            details = "   ".join(f"{k} {v:.3f}" for k, v in r.extras.items())
            values = [str(m.id), r.label, f"{r.distance:.4f}", *(f"{d:.4f}" for d in r.deltas), details]
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setData(Qt.UserRole, m.id)
                if 2 <= col <= 5:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(row, col, item)
            if m.id == current_id:
                self.table.selectRow(row)
        self.table.blockSignals(False)

    def _current_id(self) -> int:
        items = self.table.selectedItems()
        return int(items[0].data(Qt.UserRole)) if items else -1

    def _on_selection(self) -> None:
        self.measurement_selected.emit(self._current_id())

    def _on_remove(self) -> None:
        mid = self._current_id()
        if mid >= 0:
            self.remove_requested.emit(mid)
