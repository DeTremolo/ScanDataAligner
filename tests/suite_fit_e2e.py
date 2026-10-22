import os, sys, time, traceback
os.environ.setdefault("QT_API", "pyside6")
import _common  # noqa: E402  repo import path, QT_API, generated test parts
SCR = str(_common.WORK)  # generated parts + outputs (git-ignored)
_common.ensure_all()
STL = os.path.join(SCR, "test_plate.stl")

import numpy as np
from PySide6.QtCore import QLocale, QTimer
from PySide6.QtWidgets import QApplication

from scanprofiler.core.document import ToolMode
from scanprofiler.core.features import CylinderFeature, PlaneFeature
from scanprofiler.core import selection as sel
from scanprofiler.ui.main_window import MainWindow

errors, results = [], []
QLocale.setDefault(QLocale.c())
app = QApplication(sys.argv)
w = MainWindow()
w.show()
sys.excepthook = lambda *a: errors.append("".join(traceback.format_exception(*a)))


def check(name, cond, info=""):
    results.append(f"{'PASS' if cond else 'FAIL'}  {name}  {info}")


def run():
    try:
        w.load_mesh_file(STL)
        doc, vp, sp = w.document, w.viewport, w.selection_panel
        mesh = doc.mesh
        c = mesh.face_centers()
        rad = np.linalg.norm(c[:, :2], axis=1)
        s = sp.fit_settings()
        print(f"faces {mesh.n_faces}, defaults eps={s['eps']:.3f} tol={s['tol']:.4f}")

        def select(ids):
            w.clear_selection()
            w.on_faces_selected(np.asarray(ids), True)

        # 1. Hole: half the bore wall + overspray ring on the top face + a far island
        wall = np.flatnonzero((rad < 10.01) & (c[:, 1] > 0))
        ring = np.flatnonzero((c[:, 2] > 9.99) & (rad < 13))
        island = np.flatnonzero((c[:, 2] > 9.99) & (np.hypot(c[:, 0] - 40, c[:, 1] - 20) < 3))
        select(np.concatenate([wall, ring, island]))
        t = time.perf_counter()
        w.on_ransac_cylinder()
        dt = time.perf_counter() - t
        f = doc.features[-1] if doc.features else None
        ok = isinstance(f, CylinderFeature)
        check("cylinder feature created", ok, w.statusBar().currentMessage())
        if ok:
            check("hole Ø20", abs(f.diameter - 20) < 0.01 and f.kind == "hole", f"Ø{f.diameter:.5f} {f.kind} arc {f.arc_deg:.0f}°")
            check("hole axis +Z through origin", abs(f.axis @ [0, 0, 1]) > 0.99999 and np.linalg.norm(f.center[:2]) < 0.01,
                  f"axis {np.round(f.axis, 5)} centre {np.round(f.center, 4)}")
            check("hole length 20", abs(f.length - 20) < 0.05, f"L={f.length:.3f}")
            check("named Hole 1", f.name == "Hole 1")
        trimmed = np.flatnonzero(doc.selection)
        check("trim removed ring + island overspray", not np.isin(ring, trimmed).any() and not np.isin(island, trimmed).any()
              and np.isin(trimmed, wall).all(), f"{len(trimmed)} of {len(wall)} wall faces kept")
        check("cylinder fit time", dt < 3, f"{dt:.2f}s")

        # 2. RANSAC plane on top face + side overspray, DBSCAN removes an island on the bottom
        top = np.flatnonzero((c[:, 2] > 9.99) & (c[:, 0] < -20))
        side = np.flatnonzero((np.abs(c[:, 0] + 50) < 0.01) & (c[:, 2] > 6))
        bottom_island = np.flatnonzero((c[:, 2] < -9.99) & (np.hypot(c[:, 0] - 30, c[:, 1]) < 2))
        select(np.concatenate([top, side, bottom_island]))
        w.on_ransac_plane()
        f = doc.features[-1]
        check("plane feature", isinstance(f, PlaneFeature) and f.name == "Plane 1", f.summary())
        check("plane normal +Z at z=10", f.normal @ [0, 0, 1] > 0.99999 and abs(f.point[2] - 10) < 1e-4, f"z={f.point[2]:.6f}")
        trimmed = np.flatnonzero(doc.selection)
        check("plane trim drops side + island", np.isin(trimmed, top).all(), f"{len(trimmed)}/{len(top)}")

        # 3. DBSCAN button alone
        select(np.concatenate([top, bottom_island]))
        w.on_dbscan(*[sp.fit_settings()[k] for k in ("eps", "min_samples")])
        kept = np.flatnonzero(doc.selection)
        check("DBSCAN keeps top, drops island", np.array_equal(np.sort(kept), np.sort(top)), w.statusBar().currentMessage())

        # 4. Landmark plane from picked points on the bottom (outward normal -Z)
        for p in ([-40, -20, -10], [40, -25, -10], [0, 25, -10]):
            w.on_point_picked(np.array(p, float))
        w.on_fit_landmark_plane()
        f = doc.features[-1]
        check("landmark plane -Z", f.name == "Landmark plane 1" and f.normal @ [0, 0, -1] > 0.99999, f.summary())

        # 5. Fit failure is reported, not raised
        n_before = len(doc.features)
        select(top)
        w.on_ransac_cylinder()
        check("cylinder on flat face fails gracefully", len(doc.features) == n_before and "Fit failed" in w.statusBar().currentMessage(),
              w.statusBar().currentMessage())

        # 6. Feature list, highlight, remove
        check("feature list populated", sp.feature_list.count() == 3, [sp.feature_list.item(i).text() for i in range(sp.feature_list.count())])
        sp.feature_list.setCurrentRow(0)
        check("details shown for selected feature", "Ø 19.999" in sp.feature_details.text(), sp.feature_details.text().splitlines()[0])
        plane_id = doc.features[1].id
        w.on_remove_feature(plane_id)
        check("remove feature", len(doc.features) == 2 and plane_id not in vp._feature_actors and sp.feature_list.count() == 2)

        # 7. Feature overlays must not block selection (top view: landmark patch is on the bottom,
        #    re-add a top plane patch on top of the top face)
        select(top)
        w.on_ransac_plane()
        vp.set_view("top")
        app.processEvents()
        W, H = vp.plotter.ren_win.GetSize()
        ids = vp.interaction.query_region(sel.RectRegion((0, 0), (W, H)))
        top_all = np.flatnonzero(c[:, 2] > 9.99)
        check("overlays don't occlude selection", np.isin(top, ids).all(), f"{np.isin(top, ids).mean():.3f} of patch faces visible")

        # 8. screenshot
        sp.feature_list.setCurrentRow(0)
        vp.set_view("iso")
        vp.plotter.camera.zoom(1.6)
        w.clear_selection()
        w.on_faces_selected(np.flatnonzero((rad < 10.01) & (c[:, 1] > 0)), True)
        vp.plotter.render()
        app.processEvents()
        w.grab().save(os.path.join(SCR, "fit_window.png"))
    except Exception:
        errors.append(traceback.format_exc())
    QTimer.singleShot(200, w.close)


QTimer.singleShot(800, run)
app.exec()
print("\n".join(results))
print("ERRORS:\n" + "\n".join(errors) if errors else "no exceptions")
