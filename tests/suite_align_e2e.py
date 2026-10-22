"""End-to-end 3-2-1 alignment through the real window on a part in a random scanner pose."""
import os, sys, time, traceback
os.environ.setdefault("QT_API", "pyside6")
import _common  # noqa: E402  repo import path, QT_API, generated test parts
SCR = str(_common.WORK)  # generated parts + outputs (git-ignored)
_common.ensure_all()

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from PySide6.QtCore import QLocale, QTimer
from PySide6.QtWidgets import QApplication, QPushButton

from scanprofiler.core import alignment as A
from scanprofiler.core.features import CylinderFeature, PlaneFeature
from scanprofiler.core.mesh_io import load_mesh
from scanprofiler.ui.main_window import MainWindow

cad = load_mesh(os.path.join(SCR, "test_plate.stl"))
POSE = A.rigid(Rotation.from_euler("xyz", [23, -41, 117], degrees=True).as_matrix(), [312.5, -87.25, 45.0])
SCAN = os.path.join(SCR, "test_plate_scanpose.stl")
trimesh.Trimesh(A.apply_points(POSE, cad.vertices), cad.faces, process=False).export(SCAN)

errors, results = [], []
QLocale.setDefault(QLocale.c())
app = QApplication(sys.argv)
w = MainWindow(); w.show()
sys.excepthook = lambda *a: errors.append("".join(traceback.format_exception(*a)))

def check(name, cond, info=""):
    results.append(f"{'PASS' if cond else 'FAIL'}  {name}  {info}")

def run():
    try:
        w.load_mesh_file(SCAN)
        doc, ap, sp = w.document, w.alignment_panel, w.selection_panel
        loaded = doc.mesh.vertices.copy()  # float32-quantised by the STL round trip
        Q = 1e-4  # mm: float32 resolution at ~350 mm is ~3e-5
        # STL export/import may reorder nothing but can merge vertices: map faces by nearest CAD centre.
        c_cad = A.apply_points(A.invert(POSE), doc.mesh.face_centers())
        rad = np.linalg.norm(c_cad[:, :2], axis=1)

        def fit(mask, fn):
            w.clear_selection(); w.on_faces_selected(np.flatnonzero(mask), True); fn()
            return doc.features[-1]

        bottom = fit(c_cad[:, 2] < -9.99, w.on_ransac_plane)
        hole = fit((rad < 10.01) & (c_cad[:, 1] > 0), w.on_ransac_cylinder)
        side = fit(np.abs(c_cad[:, 0] - 50) < 0.01, w.on_ransac_plane)
        check("features fitted", [f.name for f in doc.features] == ["Plane 1", "Hole 1", "Plane 2"])
        check("combos populated", ap.plane_combo.count() == 2 and ap.cylinder_combo.count() == 1 and ap.clock_combo.count() == 3)

        # Step 1 via the panel widgets
        ap.plane_combo.setCurrentIndex(ap.plane_combo.findData(bottom.id))
        ap.datum_combo.setCurrentText("XY")
        t = time.perf_counter()
        next(b for b in ap.findChildren(QPushButton) if b.text() == "Snap").click()
        dt1 = time.perf_counter() - t
        lo, hi = doc.mesh.bounds
        check("snap: part on XY, material +Z", abs(lo[2]) < Q and abs(hi[2] - 20) < Q, f"z {lo[2]:.6f}..{hi[2]:.6f}, {dt1:.2f}s")
        check("clock axes restricted to XY", [ap.clock_axis_combo.itemText(i) for i in range(ap.clock_axis_combo.count())] == ["+X", "-X", "+Y", "-Y"])

        # Step 2 / 3 via signals with explicit ids
        ap.secondary_lock_requested.emit(hole.id)
        h = doc.feature(hole.id)
        check("centre: hole axis through origin", np.linalg.norm(h.center[:2]) < Q, str(np.round(h.center, 6)))
        ap.tertiary_lock_requested.emit(side.id, "+X")
        err = np.abs(doc.mesh.vertices - (cad.vertices + [0, 0, 10])).max()
        check("3-2-1 recovers CAD pose", err < Q, f"max vertex error {err:.1e} mm")
        status = ap.status_label.text()
        check("status shows 3 ticks", status.count("✓") == 3, status.replace("\n", " | "))
        check("transform readout", "R  xyz" in ap.transform_label.text())

        # Selection still works on the moved mesh (picker locator / depth buffer rebuilt)
        w.viewport.set_view("top"); app.processEvents()
        from scanprofiler.core import selection as sel
        W, H = w.viewport.plotter.ren_win.GetSize()
        ids = w.viewport.interaction.query_region(sel.RectRegion((0, 0), (W, H)))
        top_now = doc.mesh.face_centers()[:, 2] > 19.99
        check("visible selection after alignment", top_now[ids].sum() / top_now.sum() > 0.9999, f"{top_now[ids].sum()}/{top_now.sum()}")

        # Incremental shifts via the panel buttons
        ap.trans_step.setValue(2.5); ap.rot_step.setValue(90)
        ap.translate_requested.emit(0, 2.5)
        ap.rotate_requested.emit(2, 90.0)
        check("shift + rotate", np.allclose(doc.feature(side.id).normal, [0, 1, 0], atol=1e-6)
              and abs(doc.feature(hole.id).center[1] - 2.5) < 1e-6, w.act_undo.text())
        t = time.perf_counter(); ap.translate_requested.emit(1, 1.0); dt_shift = time.perf_counter() - t
        check("shift is interactive", dt_shift < 1.0, f"{dt_shift:.2f}s for 356k faces")
        w.undo()

        # A feature fitted after alignment survives undo of the alignment
        top = fit(doc.mesh.face_centers()[:, 2] > -1e9 if False else top_now, w.on_ransac_plane)
        n_undo = 0
        while doc.can_undo:
            w.undo(); n_undo += 1
        err = np.abs(doc.mesh.vertices - loaded).max()
        check("undo all -> scanner pose", err == 0.0 and n_undo == 5 and doc.alignment.datum is None, f"{n_undo} undos, err {err:.1e}")
        t_back = doc.feature(top.id)
        check("post-alignment feature carried back", t_back is not None and abs((A.apply_points(POSE, np.array([[0, 0, 10.0]]))[0] - t_back.point) @ t_back.normal) < 1e-5)
        check("status cleared", ap.status_label.text().count("✓") == 0 and not w.act_undo.isEnabled())

        # Reset alignment
        ap.primary_datum_requested.emit(bottom.id, "XZ", True)
        lo, hi = doc.mesh.bounds
        check("snap to XZ flipped: material on -Y", abs(hi[1]) < Q and abs(lo[1] + 20) < Q, f"y {lo[1]:.4f}..{hi[1]:.4f}")
        ap.reset_alignment_requested.emit()
        err = np.abs(doc.mesh.vertices - loaded).max()
        check("reset -> scanner pose", err < 1e-9 and np.allclose(doc.transform, np.eye(4)), f"{err:.1e}")

        # Error paths
        ap.secondary_lock_requested.emit(hole.id)
        check("step 2 before step 1 refused", "step 1" in w.statusBar().currentMessage(), w.statusBar().currentMessage())
        ap.primary_datum_requested.emit(-1, "XY", False)
        check("snap without plane refused", "Fit a plane" in w.statusBar().currentMessage())

        # Landmark 'Approve -> map to' with a landmark plane
        for p in ([-40, -20, -10], [40, -25, -10], [0, 25, -10]):
            w.on_point_picked(A.apply_points(POSE, np.array([p], float))[0])
        w.on_fit_landmark_plane()
        sp.map_plane_requested.emit("XY")
        lo, hi = doc.mesh.bounds
        check("landmark approve -> XY", abs(lo[2]) < Q and abs(hi[2] - 20) < Q and "Landmark plane 1" in ap.status_label.text())

        # final screenshot: aligned with all features
        ap.secondary_lock_requested.emit(hole.id)
        ap.tertiary_lock_requested.emit(side.id, "+X")
        w.viewport.set_view("iso"); app.processEvents()
        w.grab().save(os.path.join(SCR, "align_window.png"))
    except Exception:
        errors.append(traceback.format_exc())
    QTimer.singleShot(200, w.close)

QTimer.singleShot(800, run)
app.exec()
print("\n".join(results)); print("ERRORS:\n" + "\n".join(errors) if errors else "no exceptions")
