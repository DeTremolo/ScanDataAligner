"""Export through the main window: real on_export path with the dialogs patched."""
import os, subprocess, sys, traceback
os.environ.setdefault("QT_API", "pyside6")
import _common  # noqa: E402  repo import path, QT_API, generated test parts
SCR = str(_common.WORK)  # generated parts + outputs (git-ignored)
_common.ensure_all()
OUT = os.path.join(SCR, "export_gui"); os.makedirs(OUT, exist_ok=True)
for f in os.listdir(OUT):
    os.remove(os.path.join(OUT, f))

import ezdxf
from PySide6.QtCore import QLocale, QTimer
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox
import scanprofiler.ui.main_window as mw
from scanprofiler.ui.export_dialog import ExportDialog

results, errors = [], []
def check(name, cond, info=""):
    results.append(f"{'PASS' if cond else 'FAIL'}  {name}  {info}")

# Patch modal UI: accept the options dialog with chosen settings, return a fixed save path.
choice = {"path": None, "separate": False, "selected": False}
def fake_exec(self):
    self.separate_check.setChecked(choice["separate"])
    self.selected_radio.setChecked(choice["selected"] and self.selected_radio.isEnabled())
    return ExportDialog.Accepted
ExportDialog.exec = fake_exec
QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (choice["path"], ""))
warned = []
QMessageBox.warning = staticmethod(lambda *a, **k: warned.append(a[2]))

QLocale.setDefault(QLocale.c())
app = QApplication(sys.argv)
w = mw.MainWindow(); w.show()
sys.excepthook = lambda *a: errors.append("".join(traceback.format_exception(*a)))

def run():
    try:
        w.load_mesh_file(os.path.join(SCR, "test_fillet_plate.stl"))
        w.on_export("dxf")
        check("no sections -> refused", "Generate sections" in w.statusBar().currentMessage())
        w.on_section_array(2, 2, 18, 8)  # 3 sections
        check("export actions enabled", all(a.isEnabled() for a in w.act_export.values()))

        choice.update(path=os.path.join(OUT, "plate.dxf"), separate=False)
        w.on_export("dxf")
        doc = ezdxf.readfile(choice["path"])
        layers = sorted(l.dxf.name for l in doc.layers if l.dxf.name.endswith("OUTER"))
        check("GUI DXF: one file, layer set per section", layers == ["Z+10.000_OUTER", "Z+18.000_OUTER", "Z+2.000_OUTER"],
              f"{layers} | {w.statusBar().currentMessage()}")

        choice.update(path=os.path.join(OUT, "plate.dxf"), separate=True)
        w.on_export("dxf")
        files = sorted(f for f in os.listdir(OUT) if f.startswith("plate_"))
        check("GUI DXF: separate files", files == ["plate_Z+10.000.dxf", "plate_Z+18.000.dxf", "plate_Z+2.000.dxf"], files)

        w.section_panel.section_list.setCurrentRow(1)
        choice.update(path=os.path.join(OUT, "one.step"), separate=False, selected=True)
        w.on_export("step")
        text = open(choice["path"], encoding="ascii").read()
        check("GUI STEP: selected section only", text.count("GEOMETRIC_CURVE_SET('Z = 10.0000'") == 1
              and "Z = 2.0000" not in text, w.statusBar().currentMessage())

        choice.update(path=os.path.join(OUT, "plate.vbs"), selected=False)
        w.on_export("vbs")
        res = subprocess.run(["cscript", "//nologo", str(_common.HARNESS), choice["path"]],
                             capture_output=True, text=True, timeout=120)
        check("GUI VBS runs under mock SolidWorks", res.returncode == 0 and "NOTIFY ScanProfiler: 3 section" in res.stdout,
              res.stdout.strip().splitlines()[-1] if res.stdout.strip() else res.stderr[:200])
        check("no warnings for closed contours", not warned, warned)

        # alignment change clears sections -> export refused (no stale geometry)
        w.on_translate(0, 1.0)
        w.on_export("dxf")
        check("stale sections not exportable after transform", "Generate sections" in w.statusBar().currentMessage()
              and not w.act_export["dxf"].isEnabled())

        # Failure is reported, not raised
        w.on_section_array(2, 10, 10, 1)
        report = w.export_sections("dxf", w.document.sections, os.path.join(OUT, "missing_dir", "x.dxf"), mw.ExportOptions())
        check("write failure reported in status bar", report is None and "Export failed" in w.statusBar().currentMessage(),
              w.statusBar().currentMessage()[:80])
    except Exception:
        errors.append(traceback.format_exc())
    QTimer.singleShot(200, w.close)

QTimer.singleShot(800, run)
app.exec()
print("\n".join(results)); print("ERRORS:\n" + "\n".join(errors) if errors else "no exceptions")
