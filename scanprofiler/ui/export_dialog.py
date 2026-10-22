"""Export options dialog (module 1)."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QLabel,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from ..core.exporters import ExportOptions

TITLES = {"dxf": "DXF profiles", "step": "STEP curve wireframe", "vbs": "SolidWorks macro (.vbs)"}
NOTES = {
    "dxf": "Native LINE / ARC / CIRCLE / SPLINE in section-plane coordinates (mm).\n"
           "Outer contours CCW, holes CW; open chains go to the OPEN layer.",
    "step": "AP214 wireframe in the aligned CAD coordinates (mm), one curve set per section.",
    "vbs": "Run with the target part open in SolidWorks: creates an offset plane and a\n"
           "named sketch per section. Splines are created through dense fit points.",
}


class ExportDialog(QDialog):
    def __init__(self, kind: str, n_sections: int, selected_name: str | None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Export {TITLES[kind]}")
        root = QVBoxLayout(self)
        root.addWidget(QLabel(NOTES[kind]))

        scope_box = QGroupBox("Sections")
        sl = QVBoxLayout(scope_box)
        self.all_radio = QRadioButton(f"All sections ({n_sections})")
        self.selected_radio = QRadioButton(f"Selected only ({selected_name})" if selected_name else "Selected only (none selected)")
        self.selected_radio.setEnabled(selected_name is not None)
        group = QButtonGroup(self)
        group.addButton(self.all_radio)
        group.addButton(self.selected_radio)
        self.all_radio.setChecked(True)
        sl.addWidget(self.all_radio)
        sl.addWidget(self.selected_radio)
        root.addWidget(scope_box)

        opt_box = QGroupBox("Options")
        ol = QVBoxLayout(opt_box)
        self.marks_check = QCheckBox("Hole centre marks")
        self.marks_check.setChecked(True)
        ol.addWidget(self.marks_check)
        self.separate_check = QCheckBox("One file per section (else one layer set per section)")
        self.polyline_check = QCheckBox("Splines as polylines (for CAM without SPLINE support)")
        if kind == "dxf":
            ol.addWidget(self.separate_check)
            ol.addWidget(self.polyline_check)
        root.addWidget(opt_box)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    @property
    def selected_only(self) -> bool:
        return self.selected_radio.isChecked()

    def options(self) -> ExportOptions:
        return ExportOptions(
            center_marks=self.marks_check.isChecked(),
            splines_as_polylines=self.polyline_check.isChecked(),
            separate_files=self.separate_check.isChecked(),
        )
