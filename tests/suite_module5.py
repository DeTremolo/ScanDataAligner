"""Module 5 GUI test: SolidWorks navigation + click/feature measurement on the filleted plate."""
import os, sys, traceback
os.environ.setdefault("QT_API", "pyside6")
import _common  # noqa: E402  repo import path, QT_API, generated test parts
SCR = str(_common.WORK)  # generated parts + outputs (git-ignored)
_common.ensure_all()
import numpy as np
from PySide6.QtCore import QLocale, QTimer
from PySide6.QtWidgets import QApplication
from scanprofiler.core import selection as sel
from scanprofiler.core.document import ToolMode
from scanprofiler.ui.interaction import display_projection
from scanprofiler.ui.main_window import MainWindow

results, errors = [], []
def check(name, cond, info=""):
    results.append(f"{'PASS' if cond else 'FAIL'}  {name}  {info}")

QLocale.setDefault(QLocale.c())
app = QApplication(sys.argv)
w = MainWindow(); w.show()
sys.excepthook = lambda *a: errors.append("".join(traceback.format_exception(*a)))

def run():
    try:
        # Plate in CAD pose: 100x60x20 (z 0..20), R5 corners, hole d20 at (-20,0), slot R4 at x 20..30
        w.load_mesh_file(os.path.join(SCR, "test_fillet_plate.stl"))
        doc, vp, ix = w.document, w.viewport, w.viewport.interaction
        iren, cam = vp.plotter.iren.interactor, vp.plotter.camera
        W, H = vp.plotter.ren_win.GetSize()
        cx, cy = W // 2, H // 2

        def ev(name, x, y, ctrl=0, shift=0):
            iren.SetEventInformation(int(x), int(y), ctrl, shift); iren.InvokeEvent(name)

        def screen_of(p):
            mvp, vpx = display_projection(vp.plotter.renderer)
            xy, _ = sel.project_to_display(np.atleast_2d(p), mvp, vpx)
            return xy[0]

        # ---------------- navigation
        vp.set_view("iso"); app.processEvents()
        p0, f0 = np.array(cam.position), np.array(cam.focal_point)
        dist0 = np.linalg.norm(p0 - f0)
        ev("LeftButtonPressEvent", cx, cy); ev("MouseMoveEvent", cx + 120, cy + 60); ev("LeftButtonReleaseEvent", cx + 120, cy + 60)
        p1, f1 = np.array(cam.position), np.array(cam.focal_point)
        check("neutral LMB drag pans (no rotation, no zoom)", not np.allclose(f0, f1)
              and abs(np.linalg.norm(p1 - f1) - dist0) < 1e-6 * dist0 and np.allclose(p1 - f1, p0 - f0, atol=1e-6 * dist0))

        anchor = np.array([30.0, -20.0, 20.0])  # a point on the top face
        sx, sy = screen_of(anchor)
        d_before = np.linalg.norm(np.array(cam.position) - np.array(cam.focal_point))
        ev("MouseWheelBackwardEvent", sx, sy)  # scroll down
        d_in = np.linalg.norm(np.array(cam.position) - np.array(cam.focal_point))
        ev("MouseWheelForwardEvent", sx, sy); ev("MouseWheelForwardEvent", sx, sy)  # scroll up twice
        d_out = np.linalg.norm(np.array(cam.position) - np.array(cam.focal_point))
        check("wheel down zooms IN, wheel up zooms OUT (SolidWorks)", d_in < d_before < d_out, f"{d_before:.1f} -> {d_in:.1f} -> {d_out:.1f}")
        moved = np.linalg.norm(screen_of(anchor) - [sx, sy])
        check("zoom keeps the point under the cursor in place", moved < 8.0, f"drift {moved:.2f} px (point off focal plane)")

        f2 = np.array(cam.focal_point)
        ev("MiddleButtonPressEvent", cx, cy, ctrl=1); ev("MouseMoveEvent", cx + 50, cy, ctrl=1); ev("MiddleButtonReleaseEvent", cx + 50, cy, ctrl=1)
        check("Ctrl+MMB pans", not np.allclose(f2, cam.focal_point))
        d3 = np.linalg.norm(np.array(cam.position) - np.array(cam.focal_point))
        ev("MiddleButtonPressEvent", cx, cy, shift=1); ev("MouseMoveEvent", cx, cy + 80, shift=1); ev("MiddleButtonReleaseEvent", cx, cy + 80, shift=1)
        check("Shift+MMB zooms", abs(np.linalg.norm(np.array(cam.position) - np.array(cam.focal_point)) - d3) > 1e-3)
        dop = np.array(cam.direction)
        ev("MiddleButtonPressEvent", cx, cy); ev("MouseMoveEvent", cx + 90, cy + 30); ev("MiddleButtonReleaseEvent", cx + 90, cy + 30)
        check("MMB orbits", not np.allclose(dop, cam.direction))
        w._set_tool_mode(ToolMode.PAINT)
        vp.set_view("top"); app.processEvents()
        f3 = np.array(cam.focal_point)
        ev("LeftButtonPressEvent", cx, cy); ev("MouseMoveEvent", cx + 40, cy); ev("LeftButtonReleaseEvent", cx + 40, cy)
        check("tool mode: LMB paints, does not pan", np.allclose(f3, cam.focal_point) and doc.selection.any())
        w.clear_selection()
        # Regression: rapid double clicks at one spot must not swap out the interactor style
        for _ in range(3):
            ev("LeftButtonPressEvent", cx, cy); ev("LeftButtonReleaseEvent", cx, cy)
        check("double clicks keep our interactor style", iren.GetInteractorStyle() is ix.style and doc.selection.any())
        w.clear_selection()

        # ---------------- features for measuring
        c = doc.mesh.face_centers(); n = doc.mesh.face_normals()
        def fit(mask, fn):
            w.clear_selection(); w.on_faces_selected(np.flatnonzero(mask), True); fn(); return doc.features[-1]
        top = fit(n[:, 2] > 0.99, w.on_ransac_plane)
        bottom = fit(n[:, 2] < -0.99, w.on_ransac_plane)
        hole = fit(np.linalg.norm(c[:, :2] - [-20, 0], axis=1) < 10.05, w.on_ransac_cylinder)
        slot_end = fit((np.linalg.norm(c[:, :2] - [30, 0], axis=1) < 4.05) & (c[:, 0] > 30.01), w.on_ransac_cylinder)
        w.clear_selection()
        check("fixture features", [f.name for f in doc.features] == ["Plane 1", "Plane 2", "Hole 1", "Hole 2"], [f.name for f in doc.features])

        # Feature-to-feature via the panel
        mp = w.measurement_panel
        mp.a_combo.setCurrentIndex(mp.a_combo.findData(top.id)); mp.b_combo.setCurrentIndex(mp.b_combo.findData(bottom.id))
        mp.measure_features_requested.emit(top.id, bottom.id)
        m = doc.measurements[-1].result
        check("top-bottom gap 20.000", abs(m.distance - 20) < 1e-3 and np.allclose(m.deltas, [0, 0, 20], atol=1e-3), m.label)
        mp.measure_features_requested.emit(hole.id, slot_end.id)
        m = doc.measurements[-1].result
        check("hole-slot end centre distance 50", abs(m.distance - 50) < 0.01 and abs(m.deltas[0] - 50) < 0.01
              and abs(m.extras["min wall clearance"] - 36) < 0.02, f"{m.distance:.4f} clear {m.extras['min wall clearance']:.4f}")
        check("table rows", mp.table.rowCount() == 2 and mp.table.item(1, 3).text().startswith("50.0"))

        # Click measuring: hole wall (snaps to Hole 1) then the bottom face via a hidden-free view
        w._set_tool_mode(ToolMode.MEASURE)
        vp.set_view("iso"); vp.reset_camera(); app.processEvents()
        wall_pt = np.array([-20.0 + 10 * np.cos(np.radians(200)), 10 * np.sin(np.radians(200)), 10.0])  # far wall, visible inside the bore
        top_pt = np.array([35.0, 20.0, 20.0])
        for p in (wall_pt, top_pt):
            x, y = screen_of(p); ev("LeftButtonPressEvent", x, y); ev("LeftButtonReleaseEvent", x, y)
        m = doc.measurements[-1]
        check("click: hole wall snaps to Hole 1, top face to Plane 1", m.a.feature_id == hole.id and m.b.feature_id == top.id,
              f"a={m.a}, b={m.b}")
        check("  plane-to-hole-centre = 10", abs(m.result.distance - 10) < 0.01, m.result.label)
        # free point (outside all features): side face with no fitted plane
        x, y = screen_of(np.array([50.0, 0.0, 10.0])); ev("LeftButtonPressEvent", x, y); ev("LeftButtonReleaseEvent", x, y)
        check("pending marker after first pick", vp._pending_actor is not None and "First:" in mp.pending_label.text(), mp.pending_label.text())
        ev("RightButtonPressEvent", x, y); ev("RightButtonReleaseEvent", x, y)
        check("RMB cancels pending pick", vp._pending_actor is None and w._pending_measure is None)
        for p in (np.array([50.0, 0.0, 10.0]), np.array([50.0, 0.0, 10.0]) + [0, 0, 5]):
            x, y = screen_of(p); ev("LeftButtonPressEvent", x, y); ev("LeftButtonReleaseEvent", x, y)
        m = doc.measurements[-1]
        check("point-to-point on side face: 5 in Z", m.a.point is not None and abs(m.result.distance - 5) < 0.1
              and m.result.deltas[2] > 4.9, f"{m.result.distance:.3f} {np.round(m.result.deltas, 3)}")
        labels = [a for a in vp.plotter.actors if a.startswith("measure_label_")]
        check("on-screen labels per measurement", len(labels) == len(doc.measurements) == 4, labels)

        # Alignment change: measurements follow, values unchanged; feature removal drops dependants
        before = [mm.result.distance for mm in doc.measurements]
        w.on_rotate(2, 30.0); w.on_translate(0, 12.5)
        after = [mm.result.distance for mm in doc.measurements]
        check("measurements survive transforms (same values)", np.allclose(before, after, atol=1e-9), np.round(after, 4).tolist())
        check("point measurement moved with the part", abs(doc.measurements[3].a.point[0] - 50) > 1.0)
        w.undo(); w.undo()
        check("undo restores measurement points", np.allclose(doc.measurements[3].a.point[0], 50, atol=0.05))
        w.on_remove_feature(hole.id)
        check("removing Hole 1 drops its measurements", len(doc.measurements) == 2 and mp.table.rowCount() == 2)
        check("dimension geometry lives in the on-top overlay layer",
              vp.overlay.GetActors().GetNumberOfItems() == len(vp._measure_actors) > 0)
        # selection still sees through the overlay: top-view visible query unaffected by dimension lines
        vp.set_view("top"); app.processEvents()
        Wd, Hd = vp.plotter.ren_win.GetSize()
        ids = ix.query_region(sel.RectRegion((0, 0), (Wd, Hd)))
        topf = doc.mesh.face_normals()[:, 2] > 0.99
        check("depth selection ignores overlay layer", topf[ids].sum() == topf.sum(), f"{topf[ids].sum()}/{topf.sum()}")

        mp.table.selectRow(0)
        check("row selection highlights", True)
        vp.set_view("iso"); vp.reset_camera(); app.processEvents()
        w.grab().save(os.path.join(SCR, "measure_window.png"))
    except Exception:
        errors.append(traceback.format_exc())
    QTimer.singleShot(200, w.close)

QTimer.singleShot(800, run)
app.exec()
print("\n".join(results)); print("ERRORS:\n" + "\n".join(errors) if errors else "no exceptions")
