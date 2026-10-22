"""ScanProfiler entry point."""

import logging
import os
import sys

# pyvistaqt resolves its Qt binding through qtpy; pin it before anything imports Qt.
os.environ.setdefault("QT_API", "pyside6")

from PySide6.QtCore import QLocale  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from scanprofiler.ui.main_window import MainWindow  # noqa: E402


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Engineering input: always "." as decimal separator, no thousands grouping,
    # regardless of the Windows regional setting (e.g. Finnish "1,000" would read as 1000).
    QLocale.setDefault(QLocale.c())
    app = QApplication(sys.argv)
    app.setApplicationName("ScanProfiler")
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    # Optional: open a mesh passed on the command line (python main.py part.stl)
    if len(sys.argv) > 1:
        window.load_mesh_file(sys.argv[1])

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
