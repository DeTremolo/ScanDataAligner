# CLAUDE.md — ScanProfiler (repo: DeTremolo/ScanDataAligner)

Context for Claude sessions working on this repo. The owner communicates in **Finnish**:
reply in Finnish; code, comments, commit messages and docs stay in English.

## What this is

A Windows desktop app (PySide6 + PyVista/VTK) for 3D scan data, built step by step from a
5-module blueprint. All five modules exist and are tested (~260 checks):

| Module | Content | Main files |
|---|---|---|
| 1 I/O + exports | STL/OBJ import; DXF, STEP AP214 wireframe, SolidWorks VBS macro | `core/mesh_io.py`, `core/exporters/` |
| 2 Selection + fitting | paint / box / circle / lasso / point pick, delete + undo; DBSCAN, RANSAC plane & cylinder, landmark plane | `core/selection.py`, `core/fitting.py`, `ui/interaction.py` |
| 3 Alignment | 3-2-1: snap plane to XY/XZ/YZ, centre on hole axis, clock; incremental shifts; undo | `core/alignment.py`, `core/document.py` |
| 4 Sections | slicing, loop chaining, circles, line/arc/spline segmentation, centre marks | `core/sectioning.py`, `core/profile.py` |
| 5 Measure + navigation | feature/point measurements with dX/dY/dZ, on-screen dims; SolidWorks mouse | `core/measure.py`, `ui/viewport.py`, `ui/interaction.py` |

Not tried on a real scan yet: all validation is on generated parts (see Testing).

## Architecture rules

- `scanprofiler/core/` is pure NumPy/SciPy/shapely/trimesh/ezdxf: **never import Qt or VTK there**.
  `scanprofiler/ui/` holds widgets, the viewport and mouse handling; panels only emit signals,
  `ui/main_window.py` routes them.
- Vectorise over whole arrays (meshes are 100k–400k triangles); no per-triangle Python loops.
- `core/document.py` is the single source of truth. Rigid transforms are **baked** into the mesh,
  features, picked points and measurement points (`apply_transform`), accumulating `transform`
  (scanner -> CAD). Undo restores the previous mesh exactly and inverse-transforms features /
  points, so features fitted after a transform survive undoing it.
- Derived data: sections are cleared on any mesh change / transform / undo (stale geometry must not
  be exportable); measurements are stored as references (feature id or point) and recomputed.
- Match the surrounding style: short docstrings that explain *why*, module-level named constants
  for tunables, no lambdas assigned to names.

## Run and test

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements-dev.txt
.venv\Scripts\python main.py [mesh.stl]
.venv\Scripts\python tests\run_all.py            # all suites; or: run_all.py profile sections
```

- Suites are scripts `tests/suite_*.py` (not pytest) printing `PASS` / `FAIL` lines; `run_all.py`
  runs each in its own process. Add checks with the local `check(name, cond, info)` helper.
- `tests/_common.py` generates the test parts (box, plate with hole, plate in a "scanner" pose,
  filleted plate with hole + slot) into git-ignored `tests/_work/` on first run.
- GUI suites open a real window and drive VTK with synthetic events. On a headless Linux cloud
  box they need a virtual display (e.g. `xvfb-run`); the VBS macro checks need Windows `cscript`
  (`tests/sw_mock_harness.vbs` = recording mock of the SolidWorks API). Core suites (profile,
  measure, fitting, alignment, sections, exports minus VBS) run anywhere.
- Binary STL stores float32: tolerances around 1e-4 mm in tests that round-trip through STL are
  deliberate, not slack.
- Keep `run_all.py` green before committing.

## Decisions worth knowing (each was learned from a failing test)

- **Selection visibility** uses the scan's depth buffer + a facing test, not VTK's hardware
  selector: the id buffer misses sub-pixel triangles (dense scans zoomed out). Facing
  self-calibrates against inverted winding. Overlays (features, dimensions) are hidden during the
  depth capture; the on-top overlay renderer is switched off then too.
- **PyVista swaps the interactor style on a left double-click** (chart-interaction toggle). The
  controller untracks that click handler and installs its style through `plotter.iren.style`;
  otherwise two fast clicks silently disable every tool. Regression check in `suite_module5`.
- PyVista hotkeys `q` (closes render window), `C`, `b` are cleared in the viewport.
- **Fitting**: RANSAC hypotheses use smoothed (vertex-averaged) normals; inlier gating uses raw
  face normals (smoothed ones bleed across edges and admit rim faces). Final refits run on the
  inlier faces' *vertices* with an extra vertex-level tolerance filter. Cylinders must cover
  >= 20 deg of arc. DBSCAN is voxel-grid accelerated (eps/2 cells).
- **Profile segmentation** (`core/profile.py`): greedy line/arc growth with non-shared run
  boundaries; a straight window does not rule out an arc (dense sections are collinear per facet);
  boundaries re-split by residuals over up to a third of each run; "genuine" primitives are told
  from approximations of free-form curves by a smoothed-residual test whose noise estimate comes
  from second differences (section noise is correlated: shared mesh vertices). Short tangent arcs
  between genuine lines are kept as fillets. Junctions come from carriers (line/line intersection,
  tangent points); an analytic piece owns its junction against a spline. Arcs are rebuilt through
  their junctions, spline ends snapped, so contours are G0-closed exactly. Deviation is measured
  to the profile (own + neighbouring entity). Rule of thumb: profile tolerance ~4x scan noise.
- **Exports**: outer contours CCW, holes CW; free-form loops become one clamped spline (portable);
  DXF layers per section (OUTER 7, HOLES 1, CENTER 3 + CENTER linetype, OPEN 6). ezdxf: pass
  `color` / `linetype` as keywords to `layers.add` (dxfattribs are overridden).
- **Navigation**: wheel zooms at the cursor, scroll down = zoom in (`WHEEL_DOWN_ZOOMS_IN`);
  Ctrl+MMB pan, Shift+MMB zoom, MMB orbit; LMB pans only in Navigate mode.

## Known gaps / not verified

- **SolidWorks macro not yet run in real SolidWorks 2021**: sketch orientation on Top/Right planes
  (`SKETCH_AXES` in `core/exporters/solidworks_vbs.py`), the flip flag for negative plane offsets,
  `CreateSpline2` with a VBScript array. STEP import into SolidWorks may need its 3D-curve option.
- No project save/load; measurements are not saved or exported.
- `open3d` is in `requirements.txt` (blueprint) but unused.
- Fillets sampled by only 2–3 section points come out as sharp corners (data limit).

## Git

- Commit identity for this repo is the GitHub noreply address (`DeTremolo` /
  `DeTremolo@users.noreply.github.com`), set in the repo's local config; keep it that way.
- Do not commit meshes (`*.stl`, `*.obj`, `*.ply` are ignored) or `tests/_work/`.
- The repo is public and has no licence file yet (owner's decision).
