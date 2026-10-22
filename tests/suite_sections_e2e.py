"""Full pipeline through the window: scanner pose -> fit -> 3-2-1 -> section array."""
import os, sys, time, traceback
os.environ.setdefault("QT_API", "pyside6")
import _common  # noqa: E402  repo import path, QT_API, generated test parts
SCR = str(_common.WORK)  # generated parts + outputs (git-ignored)
_common.ensure_all()

import numpy as np
from scipy.spatial.transform import Rotation
from PySide6.QtCore import QLocale, QTimer
from PySide6.QtWidgets import QApplication, QPushButton

from scanprofiler.core import alignment as A
from scanprofiler.ui.main_window import MainWindow

POSE = A.rigid(Rotation.from_euler("xyz", [23, -41, 117], degrees=True).as_matrix(), [312.5, -87.25, 45.0])
errors, results = [], []
QLocale.setDefault(QLocale.c())
app = QApplication(sys.argv)
w = MainWindow(); w.show()
sys.excepthook = lambda *a: errors.append("".join(traceback.format_exception(*a)))

def check(name, cond, info=""):
    results.append(f"{'PASS' if cond else 'FAIL'}  {name}  {info}")

def button(panel, text):
    return next(b for b in panel.findChildren(QPushButton) if b.text() == text)

def run():
    try:
        w.load_mesh_file(os.path.join(SCR, "test_plate_scanpose.stl"))
        doc, ap, sc = w.document, w.alignment_panel, w.section_panel
        c = A.apply_points(A.invert(POSE), doc.mesh.face_centers())
        rad = np.linalg.norm(c[:, :2], axis=1)

        def fit(mask, fn):
            w.clear_selection(); w.on_faces_selected(np.flatnonzero(mask), True); fn()
            return doc.features[-1]

        bottom = fit(c[:, 2] < -9.99, w.on_ransac_plane)
        hole = fit((rad < 10.01) & (c[:, 1] > 0), w.on_ransac_cylinder)
        side = fit(np.abs(c[:, 0] - 50) < 0.01, w.on_ransac_plane)

        # Sections before alignment exist in scanner coords, then get cleared by the snap
        sc.axis_combo.setCurrentIndex(2)
        button(sc, "Range from mesh bounds").click()
        sc.interval_spin.setValue(10)
        button(sc, "Generate section array").click()
        check("sections in scanner pose", len(doc.sections) > 0 and w.act_export["dxf"].isEnabled(), f"{len(doc.sections)}")
        ap.primary_datum_requested.emit(bottom.id, "XY", False)
        check("transform clears stale sections", not doc.sections and sc.section_list.count() == 0
              and "sections_curves" not in w.viewport.plotter.actors and not w.act_export["dxf"].isEnabled())
        ap.secondary_lock_requested.emit(hole.id)
        ap.tertiary_lock_requested.emit(side.id, "+X")

        # Section array along Z through the aligned part (z in [0, 20])
        button(sc, "Range from mesh bounds").click()
        check("auto range = aligned bounds", abs(sc.start_spin.value()) < 1e-3 and abs(sc.end_spin.value() - 20) < 1e-3,
              f"{sc.start_spin.value():.4f}..{sc.end_spin.value():.4f}")
        sc.start_spin.setValue(1); sc.end_spin.setValue(19); sc.interval_spin.setValue(0.5)
        t = time.perf_counter()
        button(sc, "Generate section array").click()
        dt = time.perf_counter() - t
        secs = doc.sections
        check("37 sections generated", len(secs) == 37 and sc.section_list.count() == 37, f"{dt:.2f}s incl. drawing")
        holes = [lp for s in secs for lp in s.circles]
        centres = np.array([h.center for h in holes])
        check("every section: hole circle Ø20 at CAD origin", len(holes) == 37
              and np.abs(centres).max() < 2e-3 and all(abs(2 * h.radius - 20) < 2e-3 for h in holes),
              f"max centre offset {np.abs(centres).max():.5f}, Ø {min(2*h.radius for h in holes):.4f}..{max(2*h.radius for h in holes):.4f}")
        outer = [lp for s in secs for lp in s.loops if lp.kind == "profile"]
        check("outer profiles: 4 exact lines each", len(outer) == 37 and all(lp.closed and lp.outer and [g.kind for g in lp.segments] == ["line"] * 4 and lp.max_dev <= 0.05 for lp in outer),
              outer[0].summary())
        actors = w.viewport.plotter.actors
        check("drawn: lines, circles, centre marks", all(n in actors for n in ("sections_lines", "sections_circles", "sections_marks")))
        check("summary label", "37 sections" in sc.summary_label.text() and "37 circles" in sc.summary_label.text() and "148 lines" in sc.summary_label.text(), sc.summary_label.text())

        # Select a section -> highlight + details
        sc.section_list.setCurrentRow(10)
        check("highlight + details", "sections_highlight" in w.viewport.plotter.actors and "Ø19.999" in sc.section_details.toPlainText(),
              sc.section_details.toPlainText().splitlines()[0])

        # Single-plane override along X through the hole: two 20x20 squares
        sc.axis_combo.setCurrentIndex(0); sc.single_spin.setValue(0.0)
        button(sc, "Add section").click()
        last = doc.sections[-1]
        check("single X=0 override appended", len(doc.sections) == 38 and last.axis == 0 and len(last.loops) == 2, last.name)
        sc.single_spin.setValue(500.0)
        button(sc, "Add section").click()
        check("miss is reported", len(doc.sections) == 38 and "does not cut" in w.statusBar().currentMessage())

        # Too many planes refused
        sc.axis_combo.setCurrentIndex(2); sc.start_spin.setValue(0); sc.end_spin.setValue(20); sc.interval_spin.setValue(0.001)
        button(sc, "Generate section array").click()
        check("huge array refused", len(doc.sections) == 38 and "increase the interval" in w.statusBar().currentMessage())

        # Undo of the clocking clears sections; regenerate after that works
        w.undo()
        check("undo clears sections", not doc.sections)
        w.on_section_array(2, 5, 15, 5)
        check("regenerate after undo", len(doc.sections) == 3)
        w.on_tertiary_lock(side.id, "+X")
        w.on_section_array(2, 1, 19, 2)
        sc.section_list.setCurrentRow(4)
        w.viewport.set_view("iso"); app.processEvents()
        w.grab().save(os.path.join(SCR, "sections_window.png"))
    except Exception:
        errors.append(traceback.format_exc())
    QTimer.singleShot(200, w.close)

QTimer.singleShot(800, run)
app.exec()
print("\n".join(results)); print("ERRORS:\n" + "\n".join(errors) if errors else "no exceptions")
