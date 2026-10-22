"""Embedded PyVista/VTK 3D canvas."""

from __future__ import annotations

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget
from vtkmodules.vtkRenderingCore import vtkRenderer

from ..core.document import MeshData, ToolMode
from ..core.exporters.common import CENTER_MARK_RATIO
from ..core.features import Feature, PlaneFeature
from .interaction import InteractionController

SCAN_COLOR = "#b8c4d0"
SELECTION_COLOR = "#ff7a1a"
PICK_COLOR = "#e5484d"
PLANE_COLOR = "#3e8ef7"
CYLINDER_COLOR = "#30c48d"
FEATURE_OPACITY = 0.3
FEATURE_OPACITY_HIGHLIGHT = 0.65
FEATURE_OFFSET = 0.004  # relative lift of feature overlays off the scan surface
ORIGIN_COLORS = ("#e5484d", "#30a46c", "#3e63dd")  # X, Y, Z
SECTION_COLOR = "#ffb224"  # lines
ARC_COLOR = "#00c2d7"
SPLINE_COLOR = "#b78cff"
CENTER_MARK_COLOR = "#ffffff"
HIGHLIGHT_COLOR = "#ff4fd8"
MEASURE_COLOR = "#f5f5f5"


def _polylines(polylines: list[np.ndarray]) -> pv.PolyData:
    """Pack many (n_i, 3) polylines into one PolyData (one actor, one draw call)."""
    points = np.vstack(polylines)
    sizes = np.array([len(p) for p in polylines])
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    cells = np.concatenate([np.r_[n, np.arange(s, s + n)] for s, n in zip(starts, sizes)])
    poly = pv.PolyData(points)
    poly.lines = cells
    return poly


class Viewport(QWidget):
    """Wraps a pyvistaqt QtInteractor and owns every actor drawn in the scene."""

    tool_mode_changed = Signal(object)  # ToolMode

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.plotter = QtInteractor(self)
        # Strong focus so the viewport receives Delete / [ / ] once clicked.
        self.plotter.interactor.setFocusPolicy(Qt.StrongFocus)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plotter.interactor)

        self.scan_actor = None
        self.scan_polydata: pv.PolyData | None = None
        self.tool_mode = ToolMode.NEUTRAL
        self._feature_actors: dict[int, list] = {}
        self._highlighted: int | None = None
        self._measure_actors: list = []
        self._measure_labels: list[str] = []
        self._pending_actor = None

        self._setup_scene()

        # On-top layer for dimensions: shares the camera, draws over the scan.
        self.overlay = vtkRenderer()
        self.overlay.SetLayer(1)
        self.overlay.InteractiveOff()
        self.overlay.SetActiveCamera(self.plotter.renderer.GetActiveCamera())
        ren_win = self.plotter.ren_win
        ren_win.SetNumberOfLayers(max(2, ren_win.GetNumberOfLayers()))
        ren_win.AddRenderer(self.overlay)

        # Mouse routing for tools and SolidWorks-style navigation.
        self.interaction = InteractionController(self.plotter, self)
        self.interaction.set_overlay_renderers([self.overlay])

    def _setup_scene(self) -> None:
        p = self.plotter
        p.set_background("#3b4252", top="#d8dee9")  # SolidWorks-like vertical gradient
        p.enable_anti_aliasing("fxaa")
        p.add_axes(interactive=False, line_width=2)
        p.enable_lightkit()
        # PyVista's own hotkeys: q closes the render window, C/b replace our interactor style.
        for key in ("q", "C", "b"):
            p.clear_events_for_key(key)

    # ------------------------------------------------------------------ mesh

    def show_mesh(self, mesh: MeshData, reset_camera: bool = True) -> None:
        """Upload a mesh to VTK, coloured per face by its selection state."""
        poly = pv.PolyData.from_regular_faces(mesh.vertices, mesh.faces)
        poly.cell_data["selected"] = np.zeros(mesh.n_faces, dtype=np.uint8)

        self.scan_actor = self.plotter.add_mesh(
            poly,
            name="scan",
            scalars="selected",
            cmap=[SCAN_COLOR, SELECTION_COLOR],
            clim=(0, 1),
            n_colors=2,
            show_scalar_bar=False,
            smooth_shading=False,
            specular=0.25,
            pickable=True,
            reset_camera=False,
        )
        # Keep the dataset the mapper actually renders so in-place updates reach the GPU.
        self.scan_polydata = self.scan_actor.mapper.dataset
        # Push the scan's polygons back in depth so section curves and centre marks drawn
        # exactly on the surface are not z-fought into dashes.
        self.scan_actor.mapper.SetResolveCoincidentTopologyToPolygonOffset()
        self.scan_actor.mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(1.0, 4.0)
        # Depth slack for "visible" selection: a face centre sits on the surface the depth
        # buffer recorded, so only noise / chord sag needs covering; larger slack lets
        # faces just behind an edge leak through.
        self.interaction.set_scan(
            self.scan_actor,
            self.scan_polydata,
            mesh.face_centers(),
            mesh.face_normals(),
            depth_tolerance=0.3 * mesh.mean_edge_length(),
        )

        if reset_camera:
            self.plotter.reset_camera()
            self.plotter.view_isometric()
        else:
            self.plotter.render()

    def update_selection(self, mask: np.ndarray, render: bool = True) -> None:
        """Push a per-face boolean mask to the GPU without rebuilding the actor."""
        if self.scan_polydata is None:
            return
        self.scan_polydata.cell_data["selected"][:] = mask
        self.scan_polydata.GetCellData().GetArray("selected").Modified()
        if render:
            self.plotter.render()

    def set_picked_points(self, points: np.ndarray) -> None:
        self.interaction.picked_points = points
        if len(points):
            self.plotter.add_points(
                points,
                name="picked_points",
                color=PICK_COLOR,
                point_size=14,
                render_points_as_spheres=True,
                pickable=False,
                reset_camera=False,
            )
        else:
            self.plotter.remove_actor("picked_points", render=False)
        self.plotter.render()

    def show_origin(self, mesh: MeshData) -> None:
        """Global CAD origin triad (X red, Y green, Z blue), sized to the part."""
        lo, hi = mesh.bounds
        size = 0.25 * float(np.linalg.norm(hi - lo))
        for axis, color in zip(np.eye(3), ORIGIN_COLORS):
            self.plotter.add_mesh(
                pv.Line((0.0, 0.0, 0.0), size * axis), name=f"origin_{color}", color=color, line_width=3,
                pickable=False, reset_camera=False,
            )
        self.plotter.add_mesh(
            pv.Sphere(radius=0.03 * size, center=(0.0, 0.0, 0.0)), name="origin_dot", color="#ffffff",
            pickable=False, reset_camera=False,
        )

    # ------------------------------------------------------------------ features

    def show_feature(self, feature: Feature) -> None:
        """Draw a fitted plane (patch + normal arrow) or cylinder (shell + axis line)."""
        self.remove_feature(feature.id, render=False)

        def add(mesh, **kw):
            return self.plotter.add_mesh(mesh, pickable=False, reset_camera=False, **kw)

        if isinstance(feature, PlaneFeature):
            size = float(np.linalg.norm(feature.corners[2] - feature.corners[0]))
            # Lift the patch off the surface it was fitted to, to avoid z-fighting.
            corners = feature.corners + feature.normal * FEATURE_OFFSET * size
            patch = pv.PolyData(corners, faces=np.array([4, 0, 1, 2, 3]))
            actors = [
                add(patch, color=PLANE_COLOR, opacity=FEATURE_OPACITY, show_edges=True, edge_color=PLANE_COLOR, line_width=2),
                add(pv.Arrow(start=feature.point, direction=feature.normal, scale=0.25 * size), color=PLANE_COLOR),
            ]
        else:
            r = feature.radius
            # Holes draw slightly inside the bore, bosses slightly outside the surface.
            shell_r = r * (1 - FEATURE_OFFSET) if feature.kind == "hole" else r * (1 + FEATURE_OFFSET)
            shell = pv.Cylinder(
                center=feature.center, direction=feature.axis, radius=shell_r, height=feature.length,
                resolution=96, capping=False,
            )
            half = 0.5 * feature.length + 0.5 * r
            axis = pv.Line(feature.center - half * feature.axis, feature.center + half * feature.axis)
            actors = [
                add(shell, color=CYLINDER_COLOR, opacity=FEATURE_OPACITY, smooth_shading=True),
                add(axis, color=CYLINDER_COLOR, line_width=3),
                add(pv.Sphere(radius=0.06 * r, center=feature.center), color=CYLINDER_COLOR),
            ]
        self._feature_actors[feature.id] = actors
        self.highlight_feature(self._highlighted)

    def remove_feature(self, feature_id: int, render: bool = True) -> None:
        for actor in self._feature_actors.pop(feature_id, []):
            self.plotter.remove_actor(actor, render=False)
        if render:
            self.plotter.render()

    def clear_features(self) -> None:
        for fid in list(self._feature_actors):
            self.remove_feature(fid, render=False)
        self.plotter.render()

    def show_features(self, features) -> None:
        """Redraw every feature (after a transform moved them all)."""
        for fid in list(self._feature_actors):
            self.remove_feature(fid, render=False)
        for f in features:
            self.show_feature(f)

    def highlight_feature(self, feature_id: int | None) -> None:
        self._highlighted = feature_id
        for fid, actors in self._feature_actors.items():
            actors[0].prop.opacity = FEATURE_OPACITY_HIGHLIGHT if fid == feature_id else FEATURE_OPACITY
        self.plotter.render()

    # ------------------------------------------------------------------ sections

    def show_sections(self, sections, highlight: int | None = None) -> None:
        """Section entities coloured by type (lines, arcs, splines, circles) plus centre marks."""
        lines, arcs, curves, circles, marks, hot = [], [], [], [], [], []
        for i, section in enumerate(sections):
            for loop in section.loops:
                if loop.kind == "profile":
                    for seg in loop.segments:
                        {"line": lines, "arc": arcs, "spline": curves}[seg.kind].append(section.to_3d(seg.sample()))
                else:
                    (circles if loop.kind == "circle" else curves).append(section.to_3d(loop.sample()))
                if i == highlight:
                    hot.append(section.to_3d(loop.sample()))
                if loop.kind == "circle":
                    arm = CENTER_MARK_RATIO * loop.radius
                    for d in ((arm, 0.0), (0.0, arm)):
                        d = np.array(d)
                        marks.append(section.to_3d(np.array([loop.center - d, loop.center + d])))
        for name, polylines, color, width in (
            ("sections_lines", lines, SECTION_COLOR, 2),
            ("sections_arcs", arcs, ARC_COLOR, 3),
            ("sections_curves", curves, SPLINE_COLOR, 2),
            ("sections_circles", circles, CYLINDER_COLOR, 2),
            ("sections_marks", marks, CENTER_MARK_COLOR, 2),
            ("sections_highlight", hot, HIGHLIGHT_COLOR, 4),
        ):
            if polylines:
                self.plotter.add_mesh(
                    _polylines(polylines), name=name, color=color, line_width=width, pickable=False, reset_camera=False
                )
            else:
                self.plotter.remove_actor(name, render=False)
        self.plotter.render()
    # ------------------------------------------------------------------ measurements

    def _overlay_actor(self, mesh: pv.PolyData, color: str, line_width: float = 2.0, point_size: float = 0.0) -> pv.Actor:
        """An actor in the on-top overlay layer (dimensions stay visible through the part)."""
        actor = pv.Actor(mapper=pv.DataSetMapper(mesh))
        prop = actor.prop
        prop.color = color
        prop.line_width = line_width
        if point_size:
            prop.point_size = point_size
            prop.render_points_as_spheres = True
        actor.PickableOff()
        self.overlay.SetActiveCamera(self.plotter.renderer.GetActiveCamera())
        self.overlay.AddActor(actor)
        return actor

    def show_measurements(self, measurements, highlight: int | None = None) -> None:
        """Dimension line, ΔX/ΔY/ΔZ step path and a floating label per measurement, drawn on top."""
        for actor in self._measure_actors:
            self.overlay.RemoveActor(actor)
        for name in self._measure_labels:
            self.plotter.remove_actor(name, render=False)
        self._measure_actors, self._measure_labels = [], []

        for m in measurements:
            r = m.result
            a, b = r.p_a, r.p_b
            hot = m.id == highlight
            self._measure_actors.append(
                self._overlay_actor(pv.Line(a, b), HIGHLIGHT_COLOR if hot else MEASURE_COLOR, 5 if hot else 3)
            )
            corners = [a, np.array([b[0], a[1], a[2]]), np.array([b[0], b[1], a[2]]), b]
            for k, color in enumerate(ORIGIN_COLORS):  # X, Y, Z legs
                if np.linalg.norm(corners[k + 1] - corners[k]) > 1e-9:
                    self._measure_actors.append(self._overlay_actor(pv.Line(corners[k], corners[k + 1]), color, 2))
            self._measure_actors.append(self._overlay_actor(pv.PolyData(np.vstack((a, b))), MEASURE_COLOR, point_size=10))
            dx, dy, dz = r.deltas
            text = f"#{m.id}  {r.distance:.3f} mm\nΔX {dx:.3f}  ΔY {dy:.3f}  ΔZ {dz:.3f}"
            name = f"measure_label_{m.id}"
            self.plotter.add_point_labels(
                [0.5 * (a + b)], [text], name=name, font_size=13, show_points=False, always_visible=True,
                shape_opacity=0.8, shape_color="#1f2430", text_color=HIGHLIGHT_COLOR if hot else "#ffffff",
                pickable=False, reset_camera=False,
            )
            self._measure_labels.append(name)
        self.plotter.render()

    def show_measure_pending(self, point: np.ndarray | None) -> None:
        """Marker on the first pick while waiting for the second."""
        if self._pending_actor is not None:
            self.overlay.RemoveActor(self._pending_actor)
            self._pending_actor = None
        if point is not None:
            self._pending_actor = self._overlay_actor(pv.PolyData(np.atleast_2d(point)), HIGHLIGHT_COLOR, point_size=16)
        self.plotter.render()

    # ------------------------------------------------------------------ camera / tools

    def set_view(self, name: str) -> None:
        views = {
            "iso": self.plotter.view_isometric,
            "top": self.plotter.view_xy,
            "front": self.plotter.view_xz,
            "right": self.plotter.view_yz,
        }
        views[name]()

    def reset_camera(self) -> None:
        self.plotter.reset_camera()

    def set_tool_mode(self, mode: ToolMode) -> None:
        self.tool_mode = mode
        self.interaction.set_mode(mode)
        self.tool_mode_changed.emit(mode)

    def shutdown(self) -> None:
        """Release the VTK render window before Qt destroys the widget."""
        self.plotter.close()
