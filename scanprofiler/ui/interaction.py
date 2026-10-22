"""VTK-side mouse interaction: tool gestures, visible-face queries and 2D overlays.

All positions here are VTK display coordinates (physical pixels, bottom-left
origin), so picking stays correct on high-DPI screens.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pyvista as pv
from PySide6.QtCore import QObject, Signal
from scipy.ndimage import maximum_filter
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkFloatArray
from vtkmodules.vtkCommonDataModel import vtkStaticCellLocator
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingCore import vtkActor2D, vtkCellPicker, vtkPolyDataMapper2D

from ..core import selection as sel
from ..core.document import ToolMode

ADD_COLOR = (1.0, 0.48, 0.1)
ERASE_COLOR = (0.25, 0.75, 1.0)
POINT_HIT_RADIUS = 20  # logical px for removing a picked point with RMB
LASSO_MIN_STEP = 3  # logical px between lasso vertices
WHEEL_ZOOM = 1.15  # zoom factor per wheel notch
WHEEL_DOWN_ZOOMS_IN = True  # SolidWorks default: scroll down (towards you) = zoom in

REGION_MODES = (ToolMode.PAINT, ToolMode.BOX_SELECT, ToolMode.CIRCLE_SELECT, ToolMode.LASSO_SELECT)


def display_projection(renderer) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Current camera as a (4, 4) world -> NDC matrix plus the renderer's pixel viewport."""
    m = renderer.GetActiveCamera().GetCompositeProjectionTransformMatrix(renderer.GetTiledAspectRatio(), -1, 1)
    mvp = np.array([[m.GetElement(i, j) for j in range(4)] for i in range(4)])
    w, h = renderer.GetRenderWindow().GetSize()
    vx0, vy0, vx1, vy1 = renderer.GetViewport()
    return mvp, (vx0 * w, vy0 * h, (vx1 - vx0) * w, (vy1 - vy0) * h)


def _circle(center: np.ndarray, radius: float, n: int = 64) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return center + radius * np.column_stack((np.cos(t), np.sin(t)))


class Overlay2D:
    """A screen-space polyline (brush ring, rubber band, lasso path)."""

    def __init__(self, renderer, line_width: float = 1.5) -> None:
        self.poly = pv.PolyData()
        mapper = vtkPolyDataMapper2D()
        mapper.SetInputData(self.poly)
        self.actor = vtkActor2D()
        self.actor.SetMapper(mapper)
        self.actor.PickableOff()
        self.actor.VisibilityOff()
        self.actor.GetProperty().SetLineWidth(line_width)
        renderer.AddViewProp(self.actor)

    def show(self, xy: np.ndarray, closed: bool, color) -> None:
        n = len(xy)
        if n < 2:
            self.hide()
            return
        self.poly.points = np.column_stack((xy, np.zeros(n)))
        idx = np.arange(n)
        if closed:
            idx = np.append(idx, 0)
        self.poly.lines = np.concatenate(([len(idx)], idx))
        self.actor.GetProperty().SetColor(color)
        self.actor.VisibilityOn()

    def hide(self) -> None:
        self.actor.VisibilityOff()


class DepthVisibility:
    """Occlusion test of surface points against the scan's own depth buffer.

    capture() renders the scan alone once and keeps a linear-depth image; a point
    is visible when it is not behind that surface by more than `tolerance`. Unlike
    an id buffer this also finds faces smaller than a pixel (dense scans zoomed out).
    """

    def __init__(self, renderer) -> None:
        self._renderer = renderer
        self.actor = None
        self.tolerance = 0.0  # world units, set per mesh
        self.overlays: list = []  # on-top layer renderers: they clear / write depth, so skip them
        self._depth: np.ndarray | None = None
        self._camera = (0.1, 1000.0, False)

    def capture(self) -> None:
        ren_win = self._renderer.GetRenderWindow()
        # Hide overlays (features, picked points, 2D rubber bands) so only the scan writes depth.
        props = self._renderer.GetViewProps()
        hidden = []
        for k in range(props.GetNumberOfItems()):
            prop = props.GetItemAsObject(k)
            if prop is not self.actor and prop.GetVisibility():
                prop.VisibilityOff()
                hidden.append(prop)
        for overlay in self.overlays:
            overlay.DrawOff()
        try:
            ren_win.Render()
            w, h = ren_win.GetSize()
            buf = vtkFloatArray()
            ren_win.GetZbufferData(0, 0, w - 1, h - 1, buf)
        finally:
            for prop in hidden:
                prop.VisibilityOn()
            for overlay in self.overlays:
                overlay.DrawOn()
        z = vtk_to_numpy(buf).reshape(h, w).astype(np.float64)
        cam = self._renderer.GetActiveCamera()
        near, far = cam.GetClippingRange()
        parallel = bool(cam.GetParallelProjection())
        self._camera = (near, far, parallel)
        depth = sel.eye_depth(2.0 * z - 1.0, near, far, parallel)
        depth[z >= 1.0] = -np.inf  # background never occludes
        # 3x3 max: lenient at grazing angles where depth changes fast per pixel.
        self._depth = maximum_filter(depth, size=3, mode="nearest")

    def visible(self, xy: np.ndarray, ndc_z: np.ndarray) -> np.ndarray:
        h, w = self._depth.shape
        px = np.clip(np.rint(xy[:, 0]).astype(np.int64), 0, w - 1)
        py = np.clip(np.rint(xy[:, 1]).astype(np.int64), 0, h - 1)
        return sel.eye_depth(ndc_z, *self._camera) <= self._depth[py, px] + self.tolerance

    def release(self) -> None:
        self._depth = None


@dataclass
class _Gesture:
    add: bool
    button: str
    path: list[np.ndarray]
    # Every face centre projected once per gesture (the camera is fixed meanwhile).
    ids: np.ndarray = field(repr=False)
    xy: np.ndarray = field(repr=False)
    visible: np.ndarray | None = field(default=None, repr=False)  # None = select through


class InteractionController(QObject):
    """Routes mouse events to the active tool; otherwise SolidWorks-style navigation.

    Tools own LMB / RMB (add / remove, or pick / cancel when measuring). Everywhere:
    MMB drag orbits, Ctrl+MMB pans, Shift+MMB zooms, the wheel zooms about the cursor
    (scroll down = zoom in, as in SolidWorks). With no tool active, LMB drag pans.
    """

    # The controller re-renders after emitting, so receivers should not render.
    faces_selected = Signal(object, bool)  # face indices, True = add / False = remove
    point_picked = Signal(object)  # (3,) world coordinate
    point_remove_requested = Signal(int)  # index into picked_points
    measure_picked = Signal(object)  # (3,) surface point clicked in MEASURE mode
    measure_cancelled = Signal()

    def __init__(self, plotter, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._plotter = plotter
        self._renderer = plotter.renderer
        self._iren = plotter.iren.interactor

        self.mode = ToolMode.NEUTRAL
        self.brush_radius = 20  # logical px
        self.select_through = False
        self.picked_points = np.empty((0, 3))

        self._actor = None
        self._face_centers: np.ndarray | None = None
        self._face_normals: np.ndarray | None = None
        self._visible = DepthVisibility(self._renderer)
        self._picker = vtkCellPicker()
        self._picker.SetTolerance(0.0005)
        self._picker.PickFromListOn()
        self._locator = None
        self._overlay = Overlay2D(self._renderer)
        self._gesture: _Gesture | None = None
        self._camera_op: str | None = None  # "pan" | "rotate" | "dolly" while a camera drag runs
        self._camera_button: str | None = None

        self.style = vtkInteractorStyleTrackballCamera()
        for event, handler in (
            ("LeftButtonPressEvent", self._on_left_down),
            ("LeftButtonReleaseEvent", self._on_left_up),
            ("RightButtonPressEvent", self._on_right_down),
            ("RightButtonReleaseEvent", self._on_right_up),
            ("MiddleButtonPressEvent", self._on_middle_down),
            ("MiddleButtonReleaseEvent", self._on_middle_up),
            ("MouseMoveEvent", self._on_mouse_move),
            ("MouseWheelForwardEvent", lambda *_: self._zoom_at_cursor(zoom_in=not WHEEL_DOWN_ZOOMS_IN)),
            ("MouseWheelBackwardEvent", lambda *_: self._zoom_at_cursor(zoom_in=WHEEL_DOWN_ZOOMS_IN)),
            ("LeaveEvent", self._on_leave),
            ("CharEvent", lambda *_: None),  # swallow VTK's w/s/e/q/f/p/3 key bindings
        ):
            self.style.AddObserver(event, handler)
        # Install through PyVista so its own update_style() calls keep this style, and drop
        # its left double-click chart toggle: that re-installs PyVista's default style on any
        # quick double click (e.g. two fast paint / pick clicks), silently disabling all tools.
        plotter.iren.untrack_click_position(side="left")
        plotter.iren.style = self.style

    # ------------------------------------------------------------------ setup

    def set_scan(
        self, actor, polydata: pv.PolyData, face_centers: np.ndarray, face_normals: np.ndarray, depth_tolerance: float
    ) -> None:
        """Called whenever the scan actor is (re)built."""
        self.cancel()
        self._actor = actor
        self._visible.actor = actor
        self._visible.tolerance = depth_tolerance
        self._face_centers = face_centers
        self._face_normals = face_normals
        self._picker.InitializePickList()
        self._picker.AddPickList(actor)
        self._locator = vtkStaticCellLocator()
        self._locator.SetDataSet(polydata)
        self._locator.BuildLocator()
        self._picker.RemoveAllLocators()
        self._picker.AddLocator(self._locator)

    def set_overlay_renderers(self, renderers: list) -> None:
        self._visible.overlays = list(renderers)

    def set_mode(self, mode: ToolMode) -> None:
        self.cancel()
        self.mode = mode
        self._overlay.hide()
        self._plotter.render()

    def cancel(self) -> None:
        if self._gesture is not None:
            self._end_gesture()

    def query_region(self, region: sel.Region) -> np.ndarray:
        """One-shot query of the faces inside a screen region (honours select_through)."""
        return self._query(region, self._new_gesture(True, "", region.bbox()[0]))

    # ------------------------------------------------------------------ helpers

    def _pos(self) -> np.ndarray:
        return np.array(self._iren.GetEventPosition(), dtype=float)

    def _dpr(self) -> float:
        """Physical pixels per logical pixel."""
        return self._renderer.GetRenderWindow().GetSize()[0] / max(1, self._plotter.width())

    def _brush_px(self) -> float:
        return self.brush_radius * self._dpr()

    def _new_gesture(self, add: bool, button: str, pos: np.ndarray) -> _Gesture:
        mvp, vp = display_projection(self._renderer)
        xy, ndc_z, valid = sel.project_to_display(self._face_centers, mvp, vp, return_depth=True)
        ids = np.flatnonzero(valid)
        g = _Gesture(add, button, [pos], ids, xy[valid])
        if not self.select_through:
            self._visible.capture()
            unoccluded = self._visible.visible(g.xy, ndc_z[valid])
            self._visible.release()
            front = self._front_facing(ids)
            # Self-calibrate against inverted winding: what we can see is mostly front-facing.
            if unoccluded.any() and front[unoccluded].mean() < 0.5:
                front = ~front
            # Depth alone cannot reject faces just behind an edge; facing can.
            g.visible = unoccluded & front
        return g

    def _front_facing(self, ids: np.ndarray) -> np.ndarray:
        cam = self._renderer.GetActiveCamera()
        normals = self._face_normals[ids]
        if cam.GetParallelProjection():
            return normals @ -np.array(cam.GetDirectionOfProjection()) > 0
        to_cam = np.array(cam.GetPosition()) - self._face_centers[ids]
        return np.einsum("ij,ij->i", normals, to_cam) > 0

    def _query(self, region: sel.Region, g: _Gesture) -> np.ndarray:
        lo, hi = region.bbox()
        in_box = np.all((g.xy >= lo) & (g.xy <= hi), axis=1)
        if g.visible is not None:
            in_box &= g.visible
        cand = np.flatnonzero(in_box)
        return g.ids[cand[region.contains(g.xy[cand])]]

    def _apply(self, region: sel.Region, g: _Gesture) -> None:
        ids = self._query(region, g)
        if ids.size:
            self.faces_selected.emit(ids, g.add)

    def _end_gesture(self) -> None:
        self._gesture = None

    def _draw(self, pos: np.ndarray) -> None:
        g = self._gesture
        color = ADD_COLOR if g is None or g.add else ERASE_COLOR
        if self.mode is ToolMode.PAINT:
            self._overlay.show(_circle(pos, self._brush_px()), True, color)
        elif g is None:
            self._overlay.hide()
        elif self.mode is ToolMode.BOX_SELECT:
            (x0, y0), (x1, y1) = g.path[0], pos
            self._overlay.show(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]), True, color)
        elif self.mode is ToolMode.CIRCLE_SELECT:
            self._overlay.show(_circle(g.path[0], np.linalg.norm(pos - g.path[0])), True, color)
        elif self.mode is ToolMode.LASSO_SELECT:
            self._overlay.show(np.array(g.path + [pos]), True, color)

    # ------------------------------------------------------------------ point picking

    def _pick_surface(self, pos: np.ndarray) -> np.ndarray | None:
        if self._picker.Pick(pos[0], pos[1], 0, self._renderer) and self._picker.GetCellId() >= 0:
            return np.array(self._picker.GetPickPosition())
        return None

    def _pick_point(self, pos: np.ndarray) -> None:
        point = self._pick_surface(pos)
        if point is not None:
            self.point_picked.emit(point)

    def _remove_nearest_point(self, pos: np.ndarray) -> None:
        if not len(self.picked_points):
            return
        mvp, vp = display_projection(self._renderer)
        xy, valid = sel.project_to_display(self.picked_points, mvp, vp)
        d = np.linalg.norm(xy - pos, axis=1)
        d[~valid] = np.inf
        i = int(np.argmin(d))
        if d[i] <= POINT_HIT_RADIUS * self._dpr():
            self.point_remove_requested.emit(i)

    # ------------------------------------------------------------------ tool gestures

    def _begin(self, add: bool, button: str) -> None:
        if self._actor is None or self._gesture is not None:
            return
        pos = self._pos()
        if self.mode is ToolMode.POINT_PICK:
            if add:
                self._pick_point(pos)
            else:
                self._remove_nearest_point(pos)
            return
        if self.mode is ToolMode.MEASURE:
            if not add:
                self.measure_cancelled.emit()
            elif (point := self._pick_surface(pos)) is not None:
                self.measure_picked.emit(point)
            return
        if self.mode not in REGION_MODES:
            return
        self._gesture = self._new_gesture(add, button, pos)
        if self.mode is ToolMode.PAINT:
            self._apply(sel.CapsuleRegion(pos, pos, self._brush_px()), self._gesture)
        self._draw(pos)
        self._plotter.render()

    def _finish(self, button: str) -> None:
        g = self._gesture
        if g is None or g.button != button:
            return
        pos = self._pos()
        region = None
        if self.mode is ToolMode.BOX_SELECT:
            region = sel.RectRegion(g.path[0], pos)
        elif self.mode is ToolMode.CIRCLE_SELECT:
            region = sel.CircleRegion(g.path[0], np.linalg.norm(pos - g.path[0]))
        elif self.mode is ToolMode.LASSO_SELECT and len(g.path) >= 2:
            region = sel.PolygonRegion(np.array(g.path + [pos]))
        if region is not None:
            self._apply(region, g)
        self._end_gesture()
        self._draw(pos)
        self._plotter.render()

    # ------------------------------------------------------------------ VTK observers

    def _on_left_down(self, *_):
        if self._camera_op:
            return
        if self.mode is ToolMode.NEUTRAL:
            self._start_camera("pan", "left")  # SolidWorks: LMB drag pans when no tool is active
        else:
            self._begin(True, "left")

    def _on_left_up(self, *_):
        if self._camera_button == "left":
            self._end_camera()
        elif self.mode is not ToolMode.NEUTRAL:
            self._finish("left")

    def _on_right_down(self, *_):
        if self._camera_op:
            return
        if self.mode is ToolMode.NEUTRAL:
            self.style.OnRightButtonDown()
        else:
            self._begin(False, "right")

    def _on_right_up(self, *_):
        if self.mode is ToolMode.NEUTRAL:
            self.style.OnRightButtonUp()
        else:
            self._finish("right")

    def _on_middle_down(self, *_):
        if self._gesture is not None or self._camera_op:
            return  # camera must stay fixed while a selection gesture is open
        if self._iren.GetControlKey():
            self._start_camera("pan", "middle")
        elif self._iren.GetShiftKey():
            self._start_camera("dolly", "middle")
        else:
            self._start_camera("rotate", "middle")

    def _on_middle_up(self, *_):
        if self._camera_button == "middle":
            self._end_camera()

    # ------------------------------------------------------------------ camera

    def _start_camera(self, op: str, button: str) -> None:
        x, y = self._iren.GetEventPosition()
        self.style.FindPokedRenderer(x, y)
        if self.style.GetCurrentRenderer() is None:
            return
        {"pan": self.style.StartPan, "rotate": self.style.StartRotate, "dolly": self.style.StartDolly}[op]()
        self._camera_op, self._camera_button = op, button

    def _end_camera(self) -> None:
        {"pan": self.style.EndPan, "rotate": self.style.EndRotate, "dolly": self.style.EndDolly}[self._camera_op]()
        self._camera_op = self._camera_button = None

    def _focal_plane_point(self, x: float, y: float) -> np.ndarray:
        """World point under display (x, y) at the depth of the camera's focal point."""
        ren = self._renderer
        ren.SetWorldPoint(*self._renderer.GetActiveCamera().GetFocalPoint(), 1.0)
        ren.WorldToDisplay()
        depth = ren.GetDisplayPoint()[2]
        ren.SetDisplayPoint(x, y, depth)
        ren.DisplayToWorld()
        w = ren.GetWorldPoint()
        return np.array(w[:3]) / w[3]

    def _zoom_at_cursor(self, zoom_in: bool) -> None:
        """Zoom by one wheel notch, keeping the point under the cursor fixed on screen."""
        if self._gesture is not None:
            return
        x, y = self._iren.GetEventPosition()
        cam = self._renderer.GetActiveCamera()
        before = self._focal_plane_point(x, y)
        factor = WHEEL_ZOOM if zoom_in else 1.0 / WHEEL_ZOOM
        if cam.GetParallelProjection():
            cam.SetParallelScale(cam.GetParallelScale() / factor)
        else:
            cam.Dolly(factor)
        shift = before - self._focal_plane_point(x, y)
        cam.SetFocalPoint(*(np.array(cam.GetFocalPoint()) + shift))
        cam.SetPosition(*(np.array(cam.GetPosition()) + shift))
        self._renderer.ResetCameraClippingRange()
        self._plotter.render()

    def _on_mouse_move(self, *_):
        g = self._gesture
        if self._camera_op or (g is None and self.mode is not ToolMode.PAINT):
            self.style.OnMouseMove()
            return
        pos = self._pos()
        if g is not None:
            if self.mode is ToolMode.PAINT:
                self._apply(sel.CapsuleRegion(g.path[-1], pos, self._brush_px()), g)
                g.path[-1] = pos
            elif self.mode is ToolMode.LASSO_SELECT:
                if np.linalg.norm(pos - g.path[-1]) >= LASSO_MIN_STEP * self._dpr():
                    g.path.append(pos)
        self._draw(pos)
        self._plotter.render()

    def _on_leave(self, *_):
        if self._gesture is None and self.mode is ToolMode.PAINT:
            self._overlay.hide()
            self._plotter.render()
