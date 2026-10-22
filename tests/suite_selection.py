import os, sys, time, traceback
os.environ.setdefault("QT_API", "pyside6")
import _common  # noqa: E402  repo import path, QT_API, generated test parts
SCR = str(_common.WORK)  # generated parts + outputs (git-ignored)
_common.ensure_all()
STL = os.path.join(SCR, "test_part.stl")

import numpy as np
from PySide6.QtCore import QLocale, QTimer
from PySide6.QtWidgets import QApplication

from scanprofiler.core.document import ToolMode
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
        doc, vp = w.document, w.viewport
        ix, iren = vp.interaction, vp.plotter.iren.interactor
        vp.set_view("top")
        app.processEvents()
        W, H = vp.plotter.ren_win.GetSize()
        cx, cy = W // 2, H // 2
        normals = doc.mesh.face_normals()
        top = normals[:, 2] > 0.99
        print(f"render window {W}x{H}, dpr {ix._dpr():.2f}, faces {doc.mesh.n_faces}, top faces {top.sum()}")

        def ev(name, x, y, ctrl=0, shift=0):
            iren.SetEventInformation(int(x), int(y), ctrl, shift)
            iren.InvokeEvent(name)

        def drag(btn, pts):
            ev(f"{btn}ButtonPressEvent", *pts[0])
            for p in pts[1:]:
                ev("MouseMoveEvent", *p)
            ev(f"{btn}ButtonReleaseEvent", *pts[-1])

        # A. visible-only full-screen query == exactly the top faces
        ids = ix.query_region(sel.RectRegion((0, 0), (W, H)))
        check("visible query returns only top faces", top[ids].all() and len(ids) > 0, f"{len(ids)} ids, {top[ids].mean():.3f} top")
        check("visible query finds ~all top faces", len(ids) >= 0.98 * top.sum(), f"{len(ids)}/{top.sum()}")

        # B. select-through picks hidden faces too
        ix.select_through = True
        ids_t = ix.query_region(sel.RectRegion((0, 0), (W, H)))
        ix.select_through = False
        check("select-through includes bottom faces", (normals[ids_t, 2] < -0.99).sum() > 0, f"{len(ids_t)} ids")

        # C. paint stroke via real event path
        w._set_tool_mode(ToolMode.PAINT)
        ix.brush_radius = 30
        drag("Left", [(cx - 200, cy), (cx - 100, cy), (cx + 200, cy)])
        painted = doc.selection.copy()
        check("paint stroke selects faces", painted.sum() > 0, f"{painted.sum()} faces")
        check("paint only hits visible top faces", top[painted].all())
        mvp, vpx = __import__("scanprofiler.ui.interaction", fromlist=["x"]).display_projection(vp.plotter.renderer)
        xy, _ = sel.project_to_display(doc.mesh.face_centers()[painted], mvp, vpx)
        r = 30 * ix._dpr()
        inside = sel.CapsuleRegion((cx - 200, cy), (cx + 200, cy), r + 1).contains(xy)
        check("painted faces lie inside the stroke capsule", inside.all())
        # gap-free: the jump from -100 to +200 must still be painted near cx+50
        near = sel.CircleRegion((cx + 50, cy), r * 0.5).contains(xy)
        check("fast drag leaves no gap", near.any())

        # timing: 60 paint moves
        ev("LeftButtonPressEvent", cx, cy - 300)
        t = time.perf_counter()
        for i in range(60):
            ev("MouseMoveEvent", cx - 300 + i * 10, cy - 300)
        dt = (time.perf_counter() - t) / 60
        ev("LeftButtonReleaseEvent", cx + 300, cy - 300)
        check("paint move cost < 40 ms", dt < 0.04, f"{dt*1000:.1f} ms/move")

        # D. RMB erase the first stroke
        before = doc.selection.sum()
        drag("Right", [(cx - 200, cy), (cx + 200, cy)])
        check("RMB erases", doc.selection.sum() < before and not doc.selection[painted].any(), f"{before} -> {doc.selection.sum()}")
        w.clear_selection()

        # E. box
        w._set_tool_mode(ToolMode.BOX_SELECT)
        drag("Left", [(cx - 100, cy - 100), (cx, cy), (cx + 100, cy + 100)])
        n_box = doc.selection.sum()
        xy, _ = sel.project_to_display(doc.mesh.face_centers()[doc.selection], mvp, vpx)
        check("box selects inside rect", n_box > 0 and sel.RectRegion((cx - 101, cy - 101), (cx + 101, cy + 101)).contains(xy).all(), f"{n_box}")
        drag("Right", [(cx - 50, cy - 50), (cx + 50, cy + 50)])
        check("RMB box removes", 0 < doc.selection.sum() < n_box)
        w.clear_selection()

        # F. circle
        w._set_tool_mode(ToolMode.CIRCLE_SELECT)
        drag("Left", [(cx, cy), (cx + 80, cy)])
        xy, _ = sel.project_to_display(doc.mesh.face_centers()[doc.selection], mvp, vpx)
        check("circle selects inside radius", doc.selection.sum() > 0 and (np.linalg.norm(xy - (cx, cy), axis=1) <= 81).all(), f"{doc.selection.sum()}")
        w.clear_selection()

        # G. lasso triangle
        w._set_tool_mode(ToolMode.LASSO_SELECT)
        tri = [(cx - 150, cy - 100), (cx + 150, cy - 100), (cx, cy + 150)]
        path = [tuple(np.array(tri[i]) + (np.array(tri[(i + 1) % 3]) - tri[i]) * t) for i in range(3) for t in np.linspace(0, 1, 20, endpoint=False)]
        drag("Left", path)
        xy, _ = sel.project_to_display(doc.mesh.face_centers()[doc.selection], mvp, vpx)
        check("lasso selects inside polygon", doc.selection.sum() > 0 and sel.PolygonRegion(np.array(tri)).contains(xy).mean() > 0.99, f"{doc.selection.sum()}")

        # H. delete + undo
        n_before, n_sel = doc.mesh.n_faces, int(doc.selection.sum())
        w.delete_selection()
        m = doc.mesh
        check("delete removes faces", m.n_faces == n_before - n_sel, f"{n_before} -> {m.n_faces}")
        check("delete keeps valid indices", m.faces.max() < m.n_vertices and np.unique(m.faces).size == m.n_vertices)
        check("selection reset after delete", doc.selection.sum() == 0 and len(doc.selection) == m.n_faces)
        # paint still works on the rebuilt actor
        w._set_tool_mode(ToolMode.PAINT)
        drag("Left", [(cx + 250, cy), (cx + 300, cy)])
        check("paint works after delete", doc.selection.sum() > 0)
        vp.plotter.screenshot(os.path.join(SCR, "after_delete.png"))
        w.undo()
        check("undo restores mesh", doc.mesh.n_faces == n_before and not w.act_undo.isEnabled())

        # I. point pick on top face (z = +10)
        w._set_tool_mode(ToolMode.POINT_PICK)
        for p in [(cx, cy), (cx + 150, cy + 50), (cx - 120, cy + 90)]:
            ev("LeftButtonPressEvent", *p); ev("LeftButtonReleaseEvent", *p)
        pts = doc.picked_points
        check("3 points picked on top surface", len(pts) == 3 and np.allclose(pts[:, 2], 10, atol=1e-3), str(np.round(pts, 2).tolist()))
        ev("RightButtonPressEvent", cx + 150, cy + 50); ev("RightButtonReleaseEvent", cx + 150, cy + 50)
        check("RMB removes nearest point", len(doc.picked_points) == 2)

        # J. MMB orbit and Ctrl+MMB pan
        cam = vp.plotter.camera
        p0 = np.array(cam.position)
        ev("MiddleButtonPressEvent", cx, cy); ev("MouseMoveEvent", cx + 80, cy + 40); ev("MiddleButtonReleaseEvent", cx + 80, cy + 40)
        p1 = np.array(cam.position); f1 = np.array(cam.focal_point)
        check("MMB orbits", not np.allclose(p0, p1))
        ev("MiddleButtonPressEvent", cx, cy, ctrl=1); ev("MouseMoveEvent", cx + 80, cy, ctrl=1); ev("MiddleButtonReleaseEvent", cx + 80, cy, ctrl=1)
        check("Ctrl+MMB pans", not np.allclose(f1, cam.focal_point))

        # K. iso screenshot with a painted patch
        vp.set_view("iso")
        w._set_tool_mode(ToolMode.PAINT)
        drag("Left", [(cx - 150, cy + 50), (cx + 100, cy - 60)])
        ev("MouseMoveEvent", cx + 100, cy - 60)
        app.processEvents()
        w.grab().save(os.path.join(SCR, "selection_window.png"))

        keys = getattr(vp.plotter.iren, "_key_press_event_callbacks", {})
        print("pyvista key bindings:", sorted(keys))
    except Exception:
        errors.append(traceback.format_exc())
    QTimer.singleShot(200, w.close)


QTimer.singleShot(800, run)
app.exec()
print("\n".join(results))
print("ERRORS:\n" + "\n".join(errors) if errors else "no exceptions")
