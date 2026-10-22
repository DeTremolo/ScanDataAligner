"""Main application window: menus, tool bar, docks and the 3D viewport."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QToolBar,
    QWidget,
)

from ..core import alignment, fitting, measure, sectioning
from ..core.document import Document, MeasureRef, ToolMode
from ..core.exporters import ExportOptions, export_dxf, export_solidworks_vbs, export_step
from ..core.features import CylinderFeature, Feature, PlaneFeature
from ..core.mesh_io import IMPORT_FILTER, load_mesh
from ..core.selection import delete_faces
from .export_dialog import ExportDialog
from .panels import AlignmentPanel, MeasurementPanel, SectionPanel, SelectionPanel
from .viewport import Viewport

log = logging.getLogger(__name__)

TOOL_ACTIONS = (
    (ToolMode.NEUTRAL, "Navigate", "Esc"),
    (ToolMode.POINT_PICK, "Pick points", "P"),
    (ToolMode.PAINT, "Paint", "B"),
    (ToolMode.BOX_SELECT, "Box", "R"),
    (ToolMode.CIRCLE_SELECT, "Circle", "C"),
    (ToolMode.LASSO_SELECT, "Lasso", "L"),
    (ToolMode.MEASURE, "Measure", "M"),
)

MAX_SECTIONS = 2000

EXPORTERS = {"dxf": export_dxf, "step": export_step, "vbs": export_solidworks_vbs}
EXPORT_SUFFIX = {"dxf": ".dxf", "step": ".step", "vbs": ".vbs"}
EXPORT_FILTER = {
    "dxf": "DXF drawing (*.dxf)",
    "step": "STEP AP214 (*.step *.stp)",
    "vbs": "SolidWorks macro, VBScript (*.vbs)",
}

TOOL_HINTS = {
    ToolMode.NEUTRAL: "LMB pan · MMB orbit · Ctrl+MMB pan · Shift+MMB zoom · wheel down = zoom in (at cursor)",
    ToolMode.POINT_PICK: "LMB pick point · RMB remove nearest point · MMB orbit",
    ToolMode.PAINT: "LMB paint · RMB erase · [ / ] brush radius · MMB orbit",
    ToolMode.BOX_SELECT: "LMB drag add box · RMB drag remove box · MMB orbit",
    ToolMode.CIRCLE_SELECT: "Drag from centre: LMB add · RMB remove · MMB orbit",
    ToolMode.LASSO_SELECT: "Draw outline: LMB add · RMB remove · MMB orbit",
    ToolMode.MEASURE: "LMB pick feature / surface point (snaps to fitted features) · RMB cancel · MMB orbit",
}


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ScanProfiler")
        self.resize(1600, 950)

        self.document = Document()
        self._current_feature_id = -1
        self._slicer: sectioning.Slicer | None = None
        self._export_dir: str | None = None
        self._pending_measure: MeasureRef | None = None
        self._snap_tol = 0.5

        self.viewport = Viewport(self)
        self.setCentralWidget(self.viewport)

        self.selection_panel = SelectionPanel()
        self.alignment_panel = AlignmentPanel()
        self.section_panel = SectionPanel()
        self.measurement_panel = MeasurementPanel()

        self._build_docks()
        self._build_actions()
        self._build_menus()
        self._build_toolbars()
        self._build_status_bar()
        self._build_shortcuts()
        self._connect_panels()

        self._set_tool_mode(ToolMode.NEUTRAL)

    # ================================================================ layout

    def _add_dock(self, widget: QWidget, title: str, area: Qt.DockWidgetArea, scroll: bool = True) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setObjectName(title)
        if scroll:
            # Panels are taller than many screens: scroll instead of squeezing widgets together.
            area_widget = QScrollArea()
            area_widget.setWidget(widget)
            area_widget.setWidgetResizable(True)
            area_widget.setFrameShape(QScrollArea.NoFrame)
            area_widget.setMinimumWidth(widget.sizeHint().width() + 20)
            widget = area_widget
        dock.setWidget(widget)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea)
        self.addDockWidget(area, dock)
        return dock

    def _build_docks(self) -> None:
        self.selection_dock = self._add_dock(self.selection_panel, "Selection / Fitting", Qt.LeftDockWidgetArea)
        self.alignment_dock = self._add_dock(self.alignment_panel, "Alignment", Qt.RightDockWidgetArea)
        self.section_dock = self._add_dock(self.section_panel, "Sections", Qt.RightDockWidgetArea)
        self.tabifyDockWidget(self.alignment_dock, self.section_dock)
        self.alignment_dock.raise_()
        self.measurement_dock = self._add_dock(self.measurement_panel, "Measurements", Qt.BottomDockWidgetArea, scroll=False)
        self.resizeDocks([self.measurement_dock], [160], Qt.Vertical)

    def _build_actions(self) -> None:
        self.act_open = QAction("&Import mesh…", self, shortcut=QKeySequence.Open, triggered=self.open_mesh_dialog)
        self.act_exit = QAction("E&xit", self, shortcut=QKeySequence.Quit, triggered=self.close)

        self.act_export = {
            key: QAction(label, self, enabled=False, triggered=lambda _=False, k=key: self.on_export(k))
            for key, label in (
                ("vbs", "SolidWorks macro (.vbs)…"),
                ("step", "STEP curve wireframe (.step)…"),
                ("dxf", "DXF profiles (.dxf)…"),
            )
        }

        self.act_views = {
            key: QAction(label, self, shortcut=sc, triggered=lambda _=False, k=key: self.viewport.set_view(k))
            for key, label, sc in (
                ("iso", "Isometric", "Ctrl+7"),
                ("front", "Front (XZ)", "Ctrl+1"),
                ("top", "Top (XY)", "Ctrl+5"),
                ("right", "Right (YZ)", "Ctrl+3"),
            )
        }
        self.act_fit = QAction("Zoom to fit", self, shortcut="F", triggered=self.viewport.reset_camera)

        self.tool_group = QActionGroup(self)
        self.tool_group.setExclusive(True)
        self.tool_actions: dict[ToolMode, QAction] = {}
        for mode, label, sc in TOOL_ACTIONS:
            act = QAction(label, self, checkable=True, shortcut=sc)
            act.triggered.connect(lambda _=False, m=mode: self._set_tool_mode(m))
            self.tool_group.addAction(act)
            self.tool_actions[mode] = act

        self.act_delete = QAction("Delete selection", self, triggered=self.delete_selection)
        self.act_undo = QAction("&Undo", self, shortcut=QKeySequence.Undo, enabled=False, triggered=self.undo)

    def _build_menus(self) -> None:
        mb = self.menuBar()
        file_menu = mb.addMenu("&File")
        file_menu.addAction(self.act_open)
        export_menu = file_menu.addMenu("&Export")
        export_menu.addActions(list(self.act_export.values()))
        file_menu.addSeparator()
        file_menu.addAction(self.act_exit)

        edit_menu = mb.addMenu("&Edit")
        edit_menu.addAction(self.act_undo)
        edit_menu.addAction(self.act_delete)

        view_menu = mb.addMenu("&View")
        view_menu.addActions(list(self.act_views.values()))
        view_menu.addAction(self.act_fit)
        view_menu.addSeparator()
        for dock in (self.selection_dock, self.alignment_dock, self.section_dock, self.measurement_dock):
            view_menu.addAction(dock.toggleViewAction())

        tools_menu = mb.addMenu("&Tools")
        tools_menu.addActions(list(self.tool_actions.values()))

    def _build_toolbars(self) -> None:
        file_tb = QToolBar("File", self)
        file_tb.setObjectName("FileToolbar")
        file_tb.addAction(self.act_open)
        self.addToolBar(file_tb)

        tool_tb = QToolBar("Tools", self)
        tool_tb.setObjectName("ToolsToolbar")
        tool_tb.addActions(list(self.tool_actions.values()))
        tool_tb.addSeparator()
        tool_tb.addAction(self.act_delete)
        tool_tb.addAction(self.act_undo)
        self.addToolBar(tool_tb)

        view_tb = QToolBar("View", self)
        view_tb.setObjectName("ViewToolbar")
        view_tb.addActions(list(self.act_views.values()))
        view_tb.addAction(self.act_fit)
        self.addToolBar(view_tb)

    def _build_status_bar(self) -> None:
        self.hint_label = QLabel()
        self.statusBar().addWidget(self.hint_label)
        self.mode_label = QLabel()
        self.mesh_label = QLabel("No mesh loaded")
        self.statusBar().addPermanentWidget(self.mode_label)
        self.statusBar().addPermanentWidget(self.mesh_label)

    def _build_shortcuts(self) -> None:
        # Scoped to the viewport so Delete still edits text inside spin boxes in the docks.
        scope = self.viewport
        QShortcut(QKeySequence.Delete, scope, self.delete_selection, context=Qt.WidgetWithChildrenShortcut)
        QShortcut(QKeySequence("["), scope, lambda: self.selection_panel.step_brush(-5), context=Qt.WidgetWithChildrenShortcut)
        QShortcut(QKeySequence("]"), scope, lambda: self.selection_panel.step_brush(5), context=Qt.WidgetWithChildrenShortcut)

    def _connect_panels(self) -> None:
        ix = self.viewport.interaction
        ix.faces_selected.connect(self.on_faces_selected)
        ix.point_picked.connect(self.on_point_picked)
        ix.point_remove_requested.connect(self.on_point_remove)
        ix.measure_picked.connect(self.on_measure_pick)
        ix.measure_cancelled.connect(self.cancel_measure_pick)

        mp = self.measurement_panel
        mp.measure_features_requested.connect(self.on_measure_features)
        mp.remove_requested.connect(self.on_remove_measurement)
        mp.clear_requested.connect(self.clear_measurements)
        mp.measurement_selected.connect(lambda mid: self._measurements_changed(mid if mid >= 0 else None, update_table=False))

        sp = self.selection_panel
        sp.delete_selection_requested.connect(self.delete_selection)
        sp.clear_selection_requested.connect(self.clear_selection)
        sp.clear_points_requested.connect(self.clear_points)
        sp.select_through_changed.connect(lambda on: setattr(ix, "select_through", on))
        sp.brush_size_changed.connect(self.on_brush_size_changed)
        sp.fit_landmark_plane_requested.connect(self.on_fit_landmark_plane)
        sp.map_plane_requested.connect(self.on_map_plane)
        sp.dbscan_requested.connect(self.on_dbscan)
        sp.ransac_plane_requested.connect(self.on_ransac_plane)
        sp.ransac_cylinder_requested.connect(self.on_ransac_cylinder)
        sp.feature_selected.connect(self.on_feature_selected)
        sp.remove_feature_requested.connect(self.on_remove_feature)

        ap = self.alignment_panel
        ap.primary_datum_requested.connect(self.on_primary_datum)
        ap.secondary_lock_requested.connect(self.on_secondary_lock)
        ap.tertiary_lock_requested.connect(self.on_tertiary_lock)
        ap.reset_alignment_requested.connect(self.on_reset_alignment)
        ap.translate_requested.connect(self.on_translate)
        ap.rotate_requested.connect(self.on_rotate)

        sc = self.section_panel
        sc.auto_range_requested.connect(self.on_auto_range)
        sc.section_array_requested.connect(self.on_section_array)
        sc.single_section_requested.connect(self.on_single_section)
        sc.section_selected.connect(self.on_section_selected)
        sc.clear_sections_requested.connect(self.clear_sections)
        sc.export_requested.connect(self.on_export)

    # ================================================================ helpers

    def _todo(self, message: str) -> None:
        log.info("TODO: %s", message)
        self.statusBar().showMessage(f"Not implemented yet: {message}", 4000)

    def _require_mesh(self) -> bool:
        if self.document.mesh is None:
            self.statusBar().showMessage("Import a mesh first.", 3000)
            return False
        return True

    def _set_tool_mode(self, mode: ToolMode) -> None:
        self.tool_actions[mode].setChecked(True)
        self.viewport.set_tool_mode(mode)
        if mode is ToolMode.MEASURE:
            self.measurement_dock.raise_()
        else:
            self.cancel_measure_pick()
        self.mode_label.setText(f"Mode: {mode.name.replace('_', ' ').title()}")
        self.hint_label.setText(TOOL_HINTS[mode])

    def _update_counts(self) -> None:
        doc = self.document
        n_sel = int(doc.selection.sum()) if doc.selection is not None else 0
        n_tot = doc.mesh.n_faces if doc.mesh is not None else 0
        self.selection_panel.set_counts(n_sel, n_tot, len(doc.picked_points))

    def _update_mesh_label(self) -> None:
        mesh = self.document.mesh
        lo, hi = mesh.bounds
        size = hi - lo
        self.mesh_label.setText(
            f"{mesh.source_path.name}  |  {mesh.n_faces:,} tris  {mesh.n_vertices:,} verts  |  "
            f"{size[0]:.2f} × {size[1]:.2f} × {size[2]:.2f}"
        )

    # ================================================================ module 1: I/O

    def open_mesh_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import mesh", "", IMPORT_FILTER)
        if path:
            self.load_mesh_file(path)

    def load_mesh_file(self, path: str | Path) -> None:
        # TODO(perf): move to a QThread worker if large OBJ parsing blocks the UI noticeably.
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            mesh = load_mesh(path)
        except Exception as exc:  # noqa: BLE001 - surface any parser error to the user
            QApplication.restoreOverrideCursor()
            log.exception("Failed to load %s", path)
            QMessageBox.critical(self, "Import failed", f"Could not load {path}:\n{exc}")
            return
        self.document.set_mesh(mesh)
        self.viewport.show_mesh(mesh)
        self.viewport.set_picked_points(self.document.picked_points)
        self.viewport.clear_features()
        self.viewport.show_origin(mesh)
        self._features_changed()
        self._sections_changed()
        self.selection_panel.set_mesh_scale(mesh.mean_edge_length())
        self._snap_tol = max(2.0 * mesh.mean_edge_length(), 0.05)
        self.cancel_measure_pick()
        self._measurements_changed()
        self.alignment_panel.set_alignment(None, {}, self.document.transform)
        QApplication.restoreOverrideCursor()

        self.act_undo.setEnabled(False)
        self.act_undo.setText("&Undo")
        self._update_mesh_label()
        self._update_counts()
        self.setWindowTitle(f"ScanProfiler — {mesh.source_path.name}")
        log.info("Loaded %s (%d faces)", mesh.source_path, mesh.n_faces)

    def on_export(self, kind: str) -> None:
        sections = self.document.sections
        if not sections:
            self.statusBar().showMessage("Generate sections before exporting.", 3000)
            return
        row = self.section_panel.section_list.currentRow()
        selected = sections[row] if 0 <= row < len(sections) else None
        dialog = ExportDialog(kind, len(sections), selected.name if selected else None, self)
        if dialog.exec() != ExportDialog.Accepted:
            return
        chosen = [selected] if dialog.selected_only and selected else list(sections)
        stem = self.document.mesh.source_path.stem if self.document.mesh.source_path else "sections"
        start = str(Path(self._export_dir or Path.home()) / f"{stem}_sections{EXPORT_SUFFIX[kind]}")
        path, _ = QFileDialog.getSaveFileName(self, f"Export {kind.upper()}", start, EXPORT_FILTER[kind])
        if not path:
            return
        self._export_dir = str(Path(path).parent)
        report = self.export_sections(kind, chosen, path, dialog.options())
        if report is not None and report.warnings:
            QMessageBox.warning(self, "Export warnings", "\n".join(report.warnings[:15]))

    def export_sections(self, kind: str, sections, path: str, options: ExportOptions):
        """Run an exporter; returns its report (None on failure). Separate from the dialogs for scripting."""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            report = EXPORTERS[kind](sections, path, options)
        except Exception as exc:  # noqa: BLE001 - surface any writer error to the user
            QApplication.restoreOverrideCursor()
            log.exception("Export failed")
            self.statusBar().showMessage(f"Export failed: {exc}", 8000)
            return None
        QApplication.restoreOverrideCursor()
        log.info("Exported %s: %s", kind, report.summary())
        self.statusBar().showMessage(f"Exported {report.summary()} -> {Path(report.files[0]).name}", 8000)
        return report

    # ================================================================ module 2: selection & fitting

    def on_brush_size_changed(self, size: int) -> None:
        self.viewport.interaction.brush_radius = size
        self.statusBar().showMessage(f"Brush radius: {size} px", 1500)

    def on_faces_selected(self, ids: np.ndarray, add: bool) -> None:
        self.document.selection[ids] = add
        self.viewport.update_selection(self.document.selection, render=False)
        self._update_counts()

    def on_point_picked(self, point: np.ndarray) -> None:
        doc = self.document
        doc.picked_points = np.vstack((doc.picked_points, point))
        self.viewport.set_picked_points(doc.picked_points)
        self._update_counts()
        x, y, z = point
        self.statusBar().showMessage(f"Point {len(doc.picked_points)}: ({x:.3f}, {y:.3f}, {z:.3f})", 4000)

    def on_point_remove(self, index: int) -> None:
        doc = self.document
        doc.picked_points = np.delete(doc.picked_points, index, axis=0)
        self.viewport.set_picked_points(doc.picked_points)
        self._update_counts()

    def clear_points(self) -> None:
        self.document.picked_points = np.empty((0, 3))
        self.viewport.set_picked_points(self.document.picked_points)
        self._update_counts()

    def delete_selection(self) -> None:
        if not self._require_mesh():
            return
        doc = self.document
        n = int(doc.selection.sum())
        if n == 0:
            self.statusBar().showMessage("Nothing selected.", 3000)
            return
        if n == doc.mesh.n_faces:
            self.statusBar().showMessage("Refusing to delete every face.", 3000)
            return
        doc.edit_mesh(delete_faces(doc.mesh, doc.selection), f"Delete {n:,} faces")
        self._refresh_scene(f"Deleted {n:,} faces (Ctrl+Z to undo)")

    def undo(self) -> None:
        label = self.document.undo()
        if label:
            self._refresh_scene(f"Undid: {label}")

    def _refresh_scene(self, message: str, reset_view: bool = False) -> None:
        """Redraw everything derived from the document after an edit, transform or undo."""
        doc = self.document
        self.viewport.show_mesh(doc.mesh, reset_camera=False)
        self.viewport.update_selection(doc.selection, render=False)
        self.viewport.show_features(doc.features)
        self.viewport.set_picked_points(doc.picked_points)
        self.viewport.show_origin(doc.mesh)
        if reset_view:
            self.viewport.reset_camera()
            self.viewport.set_view("iso")
        self._features_changed()
        self._sections_changed()  # the document clears sections whenever the mesh changes
        self.cancel_measure_pick()  # a pending pick was in the old pose
        self._measurements_changed()  # the document recomputed them in the new pose
        self.alignment_panel.set_alignment(doc.alignment.datum, doc.alignment.steps, doc.transform)
        self.act_undo.setEnabled(doc.can_undo)
        self.act_undo.setText(f"&Undo {doc.undo_label}" if doc.can_undo else "&Undo")
        self.act_undo.setIconText("Undo")  # toolbar keeps a short label; menu / tooltip name the step
        self.act_undo.setToolTip(f"Undo {doc.undo_label} (Ctrl+Z)" if doc.can_undo else "Undo (Ctrl+Z)")
        self._update_mesh_label()
        self._update_counts()
        self.statusBar().showMessage(message, 5000)

    def _features_changed(self, prefer_id: int | None = None) -> None:
        self.selection_panel.set_features(self.document.features, prefer_id)
        self.alignment_panel.set_features(self.document.features, prefer_id)
        self.measurement_panel.set_features(self.document.features)

    def clear_selection(self) -> None:
        if not self._require_mesh():
            return
        self.document.clear_selection()
        self.viewport.update_selection(self.document.selection)
        self._update_counts()

    def _selected_faces(self) -> np.ndarray | None:
        if not self._require_mesh():
            return None
        ids = np.flatnonzero(self.document.selection)
        if not ids.size:
            self.statusBar().showMessage("Paint or select a region first.", 3000)
            return None
        return ids

    def _set_selection(self, face_ids: np.ndarray) -> None:
        self.document.selection[:] = False
        self.document.selection[face_ids] = True
        self.viewport.update_selection(self.document.selection)
        self._update_counts()

    def _fit_failed(self, exc: Exception) -> None:
        log.info("Fit failed: %s", exc)
        self.statusBar().showMessage(f"Fit failed: {exc}", 8000)

    def _add_feature(self, feature: Feature) -> None:
        self.document.add_feature(feature)
        self.viewport.show_feature(feature)
        self._features_changed(feature.id)

    def on_fit_landmark_plane(self) -> None:
        if not self._require_mesh():
            return
        try:
            feature = fitting.fit_landmark_plane(self.document.picked_points, self.document.mesh)
        except fitting.FitError as exc:
            self._fit_failed(exc)
            return
        self._add_feature(feature)
        self.statusBar().showMessage(f"{feature.name}: RMS {feature.rms:.4f} over {feature.n_points} points", 5000)

    def on_map_plane(self, target: str) -> None:
        # TODO(module 3): rotation mapping the selected plane feature's normal onto the target plane normal.
        self._todo(f"map landmark plane to {target}")

    def on_dbscan(self, eps: float, min_samples: int) -> None:
        ids = self._selected_faces()
        if ids is None:
            return
        try:
            keep, n_clusters = fitting.largest_cluster(self.document.mesh.face_centers()[ids], eps, min_samples)
        except fitting.FitError as exc:
            self._fit_failed(exc)
            return
        self._set_selection(ids[keep])
        self.statusBar().showMessage(
            f"DBSCAN: kept {int(keep.sum()):,} of {len(ids):,} faces ({n_clusters} cluster(s) found)", 5000
        )

    def _run_fit(self, fit_region) -> None:
        ids = self._selected_faces()
        if ids is None:
            return
        s = self.selection_panel.fit_settings()
        mesh = self.document.mesh
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if s["dbscan_first"]:
                keep, _ = fitting.largest_cluster(mesh.face_centers()[ids], s["eps"], s["min_samples"])
                ids = ids[keep]
            feature, inliers = fit_region(mesh, ids, s["tol"], s["max_angle_deg"])
        except fitting.FitError as exc:
            self._fit_failed(exc)
            return
        finally:
            QApplication.restoreOverrideCursor()
        if s["trim"]:
            self._set_selection(inliers)
        self._add_feature(feature)
        self.statusBar().showMessage(f"{feature.name}: {len(inliers):,} of {len(ids):,} faces are inliers", 6000)

    def on_ransac_plane(self) -> None:
        self._run_fit(fitting.fit_plane_region)

    def on_ransac_cylinder(self) -> None:
        self._run_fit(fitting.fit_cylinder_region)

    def on_feature_selected(self, feature_id: int) -> None:
        self._current_feature_id = feature_id
        self.viewport.highlight_feature(feature_id if feature_id >= 0 else None)

    def on_remove_feature(self, feature_id: int) -> None:
        self.document.remove_feature(feature_id)  # also drops measurements that used it
        self.viewport.remove_feature(feature_id)
        self._features_changed()
        self._measurements_changed()

    # ================================================================ module 3: alignment

    def _feature_of(self, feature_id: int, kind: type, what: str):
        feature = self.document.feature(feature_id)
        if not isinstance(feature, kind):
            self.statusBar().showMessage(f"Fit a {what} first (Selection / Fitting panel).", 4000)
            return None
        return feature

    def _transform(self, T: np.ndarray, label: str, reset_view: bool = False, **state) -> None:
        self.document.apply_transform(T, label, **state)
        self._refresh_scene(label, reset_view=reset_view)

    def on_primary_datum(self, feature_id: int, target: str, flip: bool) -> None:
        if not self._require_mesh() or (plane := self._feature_of(feature_id, PlaneFeature, "plane")) is None:
            return
        side = "−" if flip else "+"
        T = alignment.snap_plane(plane, target, flip)
        text = f"{plane.name} → {target} (material {side}{alignment.NORMAL_AXIS[target]})"
        self._transform(T, f"Snap {plane.name} → {target}", reset_view=True, datum=target, step=("primary", text))

    def on_secondary_lock(self, feature_id: int) -> None:
        datum = self.document.alignment.datum
        if datum is None:
            self.statusBar().showMessage("Do step 1 (primary datum) first.", 4000)
            return
        if (cyl := self._feature_of(feature_id, CylinderFeature, "cylinder / hole")) is None:
            return
        try:
            T, deviation = alignment.center_on_axis(cyl, datum)
        except alignment.AlignmentError as exc:
            self.statusBar().showMessage(str(exc), 6000)
            return
        text = f"{cyl.name} axis at origin (⊥ deviation {deviation:.3f}°)"
        self._transform(T, f"Centre on {cyl.name}", step=("secondary", text))

    def on_tertiary_lock(self, feature_id: int, target_axis: str) -> None:
        datum = self.document.alignment.datum
        if datum is None:
            self.statusBar().showMessage("Do step 1 (primary datum) first.", 4000)
            return
        if (feature := self._feature_of(feature_id, (PlaneFeature, CylinderFeature), "feature")) is None:
            return
        try:
            T, angle = alignment.clock(feature, datum, target_axis)
        except alignment.AlignmentError as exc:
            self.statusBar().showMessage(str(exc), 6000)
            return
        text = f"{feature.name} → {target_axis} (rotated {angle:+.4f}°)"
        self._transform(T, f"Clock {feature.name} → {target_axis}", step=("tertiary", text))

    def on_map_plane(self, target: str) -> None:
        """Landmark 'Approve → map to': snap the selected plane feature, else the newest landmark plane."""
        feature = self.document.feature(self._current_feature_id)
        if not isinstance(feature, PlaneFeature):
            landmarks = [f for f in self.document.features if isinstance(f, PlaneFeature) and f.source == "landmark"]
            feature = landmarks[-1] if landmarks else None
        if feature is None:
            self.statusBar().showMessage("Fit a landmark plane first.", 4000)
            return
        self.on_primary_datum(feature.id, target, False)

    def on_reset_alignment(self) -> None:
        if not self._require_mesh():
            return
        if np.allclose(self.document.transform, np.eye(4)):
            self.statusBar().showMessage("Already in scanner coordinates.", 3000)
            return
        self._transform(alignment.invert(self.document.transform), "Reset alignment", reset_view=True, reset_alignment=True)

    def on_translate(self, axis: int, distance: float) -> None:
        if self._require_mesh():
            t = np.zeros(3)
            t[axis] = distance
            self._transform(alignment.translation(t), f"Move {'XYZ'[axis]} {distance:+g}")

    def on_rotate(self, axis: int, degrees: float) -> None:
        if self._require_mesh():
            self._transform(alignment.rotation_about(np.eye(3)[axis], degrees), f"Rotate {'XYZ'[axis]} {degrees:+g}°")

    # ================================================================ module 5: measurement

    def _ref_name(self, ref: MeasureRef) -> str:
        if ref.point is not None:
            x, y, z = ref.point
            return f"point ({x:.3f}, {y:.3f}, {z:.3f})"
        return self.document.feature(ref.feature_id).name

    def on_measure_pick(self, point: np.ndarray) -> None:
        """A click in MEASURE mode: snap to a fitted feature if one is under it, else a point."""
        doc = self.document
        feature = measure.snap(point, doc.features, self._snap_tol)
        ref = MeasureRef(feature_id=feature.id) if feature is not None else MeasureRef(point=point.copy())
        if self._pending_measure is None:
            self._pending_measure = ref
            self.viewport.show_measure_pending(point)
            text = f"First: {self._ref_name(ref)}. Pick the second (RMB cancels)."
            self.measurement_panel.set_pending(text)
            self.statusBar().showMessage(text, 6000)
            return
        first, self._pending_measure = self._pending_measure, None
        self.viewport.show_measure_pending(None)
        self.measurement_panel.set_pending(None)
        self._add_measurement(first, ref)

    def cancel_measure_pick(self) -> None:
        if self._pending_measure is not None:
            self._pending_measure = None
            self.viewport.show_measure_pending(None)
            self.measurement_panel.set_pending(None)

    def on_measure_features(self, a_id: int, b_id: int) -> None:
        if a_id < 0 or b_id < 0:
            self.statusBar().showMessage("Fit at least two features to measure between.", 4000)
            return
        self._add_measurement(MeasureRef(feature_id=a_id), MeasureRef(feature_id=b_id))

    def _add_measurement(self, a: MeasureRef, b: MeasureRef) -> None:
        try:
            m = self.document.add_measurement(a, b)
        except ValueError as exc:
            self.statusBar().showMessage(str(exc), 4000)
            return
        self._measurements_changed(m.id)
        self.measurement_dock.raise_()
        r = m.result
        dx, dy, dz = r.deltas
        self.statusBar().showMessage(
            f"#{m.id} {r.label}: {r.distance:.4f} mm  (ΔX {dx:.4f}  ΔY {dy:.4f}  ΔZ {dz:.4f})", 8000
        )

    def on_remove_measurement(self, measure_id: int) -> None:
        self.document.remove_measurement(measure_id)
        self._measurements_changed()

    def clear_measurements(self) -> None:
        self.document.measurements.clear()
        self._measurements_changed()

    def _measurements_changed(self, highlight: int | None = None, update_table: bool = True) -> None:
        self.viewport.show_measurements(self.document.measurements, highlight)
        if update_table:
            self.measurement_panel.set_measurements(self.document.measurements, highlight)

    # ================================================================ module 4: sections

    def on_auto_range(self, axis: int) -> None:
        if not self._require_mesh():
            return
        lo, hi = self.document.mesh.bounds
        self.section_panel.set_range(float(lo[axis]), float(hi[axis]))

    def _get_slicer(self) -> sectioning.Slicer:
        """Edge topology is built once per mesh; any edit or transform makes a new mesh."""
        if self._slicer is None or self._slicer.mesh is not self.document.mesh:
            self._slicer = sectioning.Slicer(self.document.mesh)
        return self._slicer

    def _make_sections(self, axis: int, offsets: np.ndarray) -> list[sectioning.Section]:
        circle_tol, spline_tol, segment = self.section_panel.fit_settings()
        slicer = self._get_slicer()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            sections = [sectioning.make_section(slicer, axis, z, circle_tol, spline_tol, segment) for z in offsets]
        finally:
            QApplication.restoreOverrideCursor()
        return [s for s in sections if s.loops]

    def _sections_changed(self, highlight: int | None = None) -> None:
        sections = self.document.sections
        self.viewport.show_sections(sections, highlight)
        self.section_panel.set_sections(sections)
        for act in self.act_export.values():
            act.setEnabled(bool(sections))

    def on_section_array(self, axis: int, start: float, end: float, interval: float) -> None:
        if not self._require_mesh():
            return
        try:
            offsets = sectioning.section_offsets(start, end, interval)
        except ValueError as exc:
            self.statusBar().showMessage(str(exc), 4000)
            return
        if len(offsets) > MAX_SECTIONS:
            self.statusBar().showMessage(f"{len(offsets)} sections requested; increase the interval (max {MAX_SECTIONS}).", 5000)
            return
        t0 = time.perf_counter()
        self.document.sections = self._make_sections(axis, offsets)
        self._sections_changed()
        self.section_dock.raise_()
        n = len(self.document.sections)
        self.statusBar().showMessage(
            f"{n} of {len(offsets)} planes along {'XYZ'[axis]} cut the part ({time.perf_counter() - t0:.2f}s)", 6000
        )

    def on_single_section(self, axis: int, offset: float) -> None:
        if not self._require_mesh():
            return
        new = self._make_sections(axis, np.array([offset]))
        if not new:
            self.statusBar().showMessage(f"{'XYZ'[axis]} = {offset:g} does not cut the part.", 4000)
            return
        self.document.sections.extend(new)
        self._sections_changed(highlight=len(self.document.sections) - 1)
        self.statusBar().showMessage(f"Added section {new[0].name}", 4000)

    def on_section_selected(self, row: int) -> None:
        self.viewport.show_sections(self.document.sections, row if row >= 0 else None)

    def clear_sections(self) -> None:
        self.document.sections.clear()
        self._sections_changed()

    # ================================================================ lifecycle

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        self.viewport.shutdown()
        super().closeEvent(event)
