# ScanProfiler

Desktop application for 3D scan data: align a raw scan to CAD coordinates, fit datum
features, cut cross-sections and export them as native CAD geometry (DXF, STEP,
SolidWorks macro). Built with PySide6 + PyVista/VTK; all geometry is vectorised NumPy and
handles 100k–400k triangle meshes interactively.

## Workflow

1. **Import** an STL (ASCII / binary) or OBJ mesh.
2. **Select** surface regions: paint brush, box, circle, lasso, point picking; delete faces.
   Selection is occlusion-aware (depth buffer), including sub-pixel triangles.
3. **Fit features** on the selection: DBSCAN cleanup, RANSAC plane, RANSAC cylinder / hole,
   least-squares landmark plane from picked points.
4. **Align (3-2-1)**: snap a datum plane to XY / XZ / YZ, centre a hole axis on the origin,
   clock a secondary feature onto an axis; incremental shifts; full undo.
5. **Section** along X, Y or Z: arrays or single planes. Each loop becomes a true circle
   (with centre mark) or a chain of lines, arcs and splines with exactly shared endpoints;
   free-form stretches stay splines.
6. **Measure** feature-to-feature or point-to-point with ΔX / ΔY / ΔZ breakdown and
   on-screen dimensions.
7. **Export** sections: DXF (LINE / ARC / CIRCLE / SPLINE, CNC-ready closed contours, layers),
   STEP AP214 curve wireframe, SolidWorks VBScript macro (offset planes + named sketches).

## Navigation (SolidWorks style)

| Input | Action |
|---|---|
| Middle drag | orbit |
| Ctrl + middle drag | pan |
| Shift + middle drag | zoom |
| Wheel | zoom at cursor (scroll down = zoom in) |
| Left drag, no tool active | pan |

Tools: `P` pick points, `B` paint, `R` box, `C` circle, `L` lasso, `M` measure, `Esc` navigate,
`F` zoom to fit, `Del` delete selection, `Ctrl+Z` undo, `[` / `]` brush size.

## Install and run (Windows, Python 3.11–3.12)

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python main.py [optional_mesh.stl]
```

## Tests

```bash
.venv\Scripts\pip install -r requirements-dev.txt
.venv\Scripts\python tests\run_all.py
```

13 suites (core math and GUI end-to-end), ~260 checks. Test parts are generated on the
first run into `tests/_work/` (git-ignored). GUI suites briefly open a window. The
SolidWorks macro is executed by Windows `cscript` against a recording mock of the
SolidWorks API (`tests/sw_mock_harness.vbs`); no SolidWorks needed.

## Layout

```
main.py                      entry point
scanprofiler/core/           geometry, no Qt / VTK
  mesh.py mesh_io.py         mesh container, STL / OBJ import
  document.py                document model, undo
  selection.py               screen-space selection math
  fitting.py features.py     DBSCAN, RANSAC plane / cylinder, feature types
  alignment.py               3-2-1 rigid alignment
  sectioning.py profile.py   slicing, loop ordering, line / arc / spline segmentation
  measure.py                 measurements, feature snapping
  exporters/                 DXF, STEP, SolidWorks VBS
scanprofiler/ui/             PySide6 window, panels, viewport, mouse interaction
tests/                       suites + runner
```

## Status / known limits

- Not yet verified inside SolidWorks: macro sketch orientation on the Top / Right planes,
  negative plane offsets, and spline creation from VBScript; STEP import may need SolidWorks'
  "import 3D curves" option. DXF / STEP are validated by read-back and parsing.
- Profile tolerance should be about 4x the scan noise; closer to the noise, fillets degrade
  to splines.
- Measurements are not saved or exported yet.
