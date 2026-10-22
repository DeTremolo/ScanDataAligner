import os, sys, traceback
os.environ.setdefault("QT_API", "pyside6")
import _common  # noqa: E402  repo import path, QT_API, generated test parts
SCR = str(_common.WORK)  # generated parts + outputs (git-ignored)
_common.ensure_all()
import numpy as np
from PySide6.QtCore import QLocale, QTimer
from PySide6.QtWidgets import QApplication
from scanprofiler.core.document import MeshData
from scanprofiler.core import selection as sel
from scanprofiler.core.mesh_io import load_mesh
from scanprofiler.ui.main_window import MainWindow

QLocale.setDefault(QLocale.c())
app = QApplication(sys.argv)
w = MainWindow(); w.show()
out, errors = [], []
sys.excepthook = lambda *a: errors.append("".join(traceback.format_exception(*a)))

def check(name, cond, info=""):
    out.append(f"{'PASS' if cond else 'FAIL'}  {name}  {info}")

def top_query(mesh, zoom=1.0):
    w.document.set_mesh(mesh)
    w.viewport.show_mesh(mesh)
    w.viewport.set_view("top")
    w.viewport.plotter.camera.zoom(zoom)
    w.viewport.plotter.render(); app.processEvents()
    W, H = w.viewport.plotter.ren_win.GetSize()
    return w.viewport.interaction.query_region(sel.RectRegion((0, 0), (W, H)))

def run():
    try:
        plate = load_mesh(os.path.join(SCR, "test_plate.stl"))
        c = plate.face_centers()
        top = c[:, 2] > 9.99
        bore = np.linalg.norm(c[:, :2], axis=1) < 10.01  # visible in perspective from above the hole
        expected = top | bore
        for zoom in (1.0, 0.25):
            ids = top_query(plate, zoom)
            got = np.zeros(plate.n_faces, bool); got[ids] = True
            bore_min = 0.99 if zoom == 1.0 else 0.8  # grazing bore wall squeezed into few pixels when zoomed out
            check(f"plate zoom {zoom}: all top incl. sub-pixel slivers, no hidden faces",
                  got[top].all() and not got[~expected].any() and got[bore].mean() >= bore_min,
                  f"top {got[top].sum()}/{top.sum()}, bore {got[bore].sum()}/{bore.sum()}, other {got[~expected].sum()}")
        flipped = MeshData(plate.vertices, plate.faces[:, ::-1].copy(), plate.source_path)
        ids = top_query(flipped)
        got = np.zeros(plate.n_faces, bool); got[ids] = True
        check("inverted winding self-calibrates", got[top].all() and not got[~expected].any(), f"{len(ids)} faces")
        # iso view: top + two side walls visible, bottom never
        w.document.set_mesh(plate); w.viewport.show_mesh(plate); app.processEvents()
        W, H = w.viewport.plotter.ren_win.GetSize()
        ids = w.viewport.interaction.query_region(sel.RectRegion((0, 0), (W, H)))
        n = plate.face_normals()[ids]
        check("iso: no bottom or hidden walls", (n[:, 2] > -0.5).all() and (n @ [1, 1, 0] < 0.5).sum() > 0,
              f"{len(ids)} faces, bottom {(n[:, 2] < -0.99).sum()}")
    except Exception:
        errors.append(traceback.format_exc())
    QTimer.singleShot(100, w.close)

QTimer.singleShot(600, run)
app.exec()
print("\n".join(out)); print("ERRORS:\n" + "\n".join(errors) if errors else "no exceptions")
