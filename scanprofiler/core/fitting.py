"""Primitive fitting (module 2): least-squares plane, DBSCAN cleanup, RANSAC plane / cylinder.

RANSAC hypotheses are generated and scored in vectorised batches against a random
subsample; the winner is then refined against all candidates and finally against
the mesh vertices of the inlier faces (vertices lie on the scanned surface, face
centres sit slightly inside curved surfaces and would bias the radius).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, QhullError, cKDTree

from .mesh import MeshData
from .features import CylinderFeature, PlaneFeature

BATCH = 128  # hypotheses scored per vectorised batch
SCORE_SAMPLE = 4000  # candidate points used to score hypotheses
ARC_BIN_DEG = 5.0
MIN_ARC_DEG = 20.0  # minimum angular coverage for a credible cylinder fit


class FitError(ValueError):
    """Raised when a region cannot support the requested fit; message is user-facing."""


# ---------------------------------------------------------------------- helpers


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors completing n to a right-handed orthonormal frame."""
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = _unit(np.cross(n, helper))
    return u, np.cross(n, u)


def _canonical(axis: np.ndarray) -> np.ndarray:
    """Flip an undirected axis so its dominant component is positive."""
    return axis if axis[np.argmax(np.abs(axis))] > 0 else -axis


def face_vertices(mesh: MeshData, face_ids: np.ndarray) -> np.ndarray:
    return mesh.vertices[np.unique(mesh.faces[face_ids])]


# ---------------------------------------------------------------------- least squares plane


def lstsq_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Total least-squares plane: centroid and the smallest singular vector."""
    if len(points) < 3:
        raise FitError("A plane needs at least 3 points.")
    centroid = points.mean(axis=0)
    _, s, vt = np.linalg.svd(points - centroid, full_matrices=False)
    if s[1] < 1e-9 * max(s[0], 1e-12):
        raise FitError("Points are collinear; pick points that span an area.")
    return centroid, vt[2]


def _plane_patch(points: np.ndarray, point: np.ndarray, normal: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """Minimum-area rectangle around the points in the plane.

    Rotating calipers over the 2D convex hull: the optimal rectangle has a side
    collinear with a hull edge, so it follows a face's edges regardless of how
    unevenly the mesh samples it.
    """
    u, v = _basis(normal)
    d = points - point
    uv = np.column_stack((d @ u, d @ v))
    try:
        hull = uv[ConvexHull(uv).vertices]
    except QhullError:  # degenerate (collinear / too few) points: keep the arbitrary basis
        hull = None
    if hull is not None:
        edges = np.roll(hull, -1, axis=0) - hull
        angles = np.arctan2(edges[:, 1], edges[:, 0])
        cos, sin = np.cos(angles), np.sin(angles)
        rx = hull @ np.vstack((cos, sin))  # (H, E): hull projected on each edge direction
        ry = hull @ np.vstack((-sin, cos))
        best = int(np.argmin(np.ptp(rx, axis=0) * np.ptp(ry, axis=0)))
        u, v = cos[best] * u + sin[best] * v, -sin[best] * u + cos[best] * v
    cu, cv = d @ u, d @ v
    pad_u = margin * (cu.max() - cu.min())
    pad_v = margin * (cv.max() - cv.min())
    u0, u1, v0, v1 = cu.min() - pad_u, cu.max() + pad_u, cv.min() - pad_v, cv.max() + pad_v
    return np.array([point + a * u + b * v for a, b in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))])


def _plane_feature(points, point, normal, source, outward_hint) -> PlaneFeature:
    if normal @ outward_hint < 0:
        normal = -normal
    dist = (points - point) @ normal
    return PlaneFeature(
        point=point,
        normal=normal,
        corners=_plane_patch(points, point, normal, margin=0.1 if source == "landmark" else 0.0),
        rms=float(np.sqrt(np.mean(dist**2))),
        flatness=float(dist.max() - dist.min()),
        n_points=len(points),
        source=source,
    )


def fit_landmark_plane(points: np.ndarray, mesh: MeshData) -> PlaneFeature:
    """Least-squares plane through picked points, normal oriented like the nearby surface."""
    point, normal = lstsq_plane(points)
    centers = mesh.face_centers()
    nearest = [int(np.argmin(np.einsum("ij,ij->i", centers - p, centers - p))) for p in points]
    hint = mesh.smooth_face_normals()[nearest].sum(axis=0)
    return _plane_feature(points, point, normal, "landmark", hint)


# ---------------------------------------------------------------------- DBSCAN


def dbscan(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Grid-accelerated DBSCAN. Returns labels per point, -1 for noise.

    Points are binned into eps/2 voxels and DBSCAN runs on voxel centroids with
    point counts as weights, so cost scales with surface area / eps^2 instead of
    the (dense) number of point pairs. Distances are approximate to within eps/2.
    """
    labels = np.full(len(points), -1, dtype=np.int64)
    if not len(points):
        return labels
    keys = np.floor(points / (0.5 * eps)).astype(np.int64)
    _, inverse, weight = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    m = len(weight)
    centroids = np.column_stack([np.bincount(inverse, weights=points[:, k], minlength=m) for k in range(3)]) / weight[:, None]

    pairs = cKDTree(centroids).query_pairs(eps, output_type="ndarray")
    i, j = pairs[:, 0], pairs[:, 1]
    density = weight + np.bincount(i, weights=weight[j], minlength=m) + np.bincount(j, weights=weight[i], minlength=m)
    core = density >= min_samples
    if not core.any():
        return labels

    cc = core[i] & core[j]
    graph = coo_matrix((np.ones(int(cc.sum())), (i[cc], j[cc])), shape=(m, m))
    _, comp = connected_components(graph, directed=False)
    voxel_label = np.where(core, comp, -1)
    # Border voxels join the cluster of any core neighbour.
    for a, b in ((i, j), (j, i)):
        attach = core[a] & ~core[b]
        voxel_label[b[attach]] = comp[a[attach]]
    valid = voxel_label >= 0
    voxel_label[valid] = np.unique(voxel_label[valid], return_inverse=True)[1].ravel()
    return voxel_label[inverse]


def largest_cluster(points: np.ndarray, eps: float, min_samples: int) -> tuple[np.ndarray, int]:
    """Mask of the most populated DBSCAN cluster, plus the number of clusters found."""
    labels = dbscan(points, eps, min_samples)
    valid = labels >= 0
    if not valid.any():
        raise FitError("DBSCAN found no dense region; increase eps or lower min samples.")
    counts = np.bincount(labels[valid])
    return labels == int(np.argmax(counts)), len(counts)


# ---------------------------------------------------------------------- RANSAC plane


def ransac_plane(
    points: np.ndarray,
    normals: np.ndarray,
    tol: float,
    max_angle_deg: float = 30.0,
    iterations: int = 1000,
    rng=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (point, normal, inlier mask). Inliers must also agree in normal direction."""
    n = len(points)
    if n < 3:
        raise FitError("Select at least 3 faces.")
    rng = np.random.default_rng(rng)
    cos_max = np.cos(np.radians(max_angle_deg))
    # Hypotheses are scored on distance only; normals gate the final inlier set.
    P = points[rng.choice(n, min(n, SCORE_SAMPLE), replace=False)]
    best, best_score = None, -1
    for start in range(0, iterations, BATCH):
        tri = rng.integers(0, n, size=(min(BATCH, iterations - start), 3))
        a, b, c = points[tri[:, 0]], points[tri[:, 1]], points[tri[:, 2]]
        nrm = np.cross(b - a, c - a)
        length = np.linalg.norm(nrm, axis=1)
        ok = length > 1e-12
        if not ok.any():
            continue
        a, nrm = a[ok], nrm[ok] / length[ok, None]
        dist = np.abs(P @ nrm.T - np.einsum("ij,ij->i", a, nrm))  # (M, k)
        score = (dist < tol).sum(axis=0)
        i = int(np.argmax(score))
        if score[i] > best_score:
            best_score, best = score[i], (a[i], nrm[i])
    if best is None:
        raise FitError("Selected points are degenerate (collinear).")

    point, normal = best
    for _ in range(3):  # re-estimate on all inliers
        inl = (np.abs((points - point) @ normal) < tol) & (np.abs(normals @ normal) > cos_max)
        if inl.sum() < 3:
            raise FitError("No consistent plane found; increase the fit tolerance.")
        point, normal = lstsq_plane(points[inl])
    inl = (np.abs((points - point) @ normal) < tol) & (np.abs(normals @ normal) > cos_max)
    return point, normal, inl


def fit_plane_region(mesh: MeshData, face_ids: np.ndarray, tol: float, max_angle_deg: float = 30.0, rng=None):
    """RANSAC plane on selected faces. Returns (PlaneFeature, inlier face ids).

    Raw face normals gate the inliers: smoothed normals bleed across edges and
    would admit faces of the adjacent surface.
    """
    centers = mesh.face_centers()[face_ids]
    normals = mesh.face_normals()[face_ids]
    _, _, inl = ransac_plane(centers, normals, tol, max_angle_deg, rng=rng)
    inlier_faces = face_ids[inl]
    verts = face_vertices(mesh, inlier_faces)
    point, normal = lstsq_plane(verts)
    verts = verts[np.abs((verts - point) @ normal) < tol]  # drop vertices shared with neighbours
    point, normal = lstsq_plane(verts)
    return _plane_feature(verts, point, normal, "ransac", normals[inl].sum(axis=0)), inlier_faces


# ---------------------------------------------------------------------- RANSAC cylinder


def _radial(points: np.ndarray, c: np.ndarray, a: np.ndarray) -> np.ndarray:
    w = points - c
    return w - np.outer(w @ a, a)


def ransac_cylinder(
    points: np.ndarray,
    normals: np.ndarray,
    tol: float,
    max_angle_deg: float = 25.0,
    iterations: int = 2000,
    rng=None,
    gate_normals: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Returns (axis point, axis, radius, inlier mask).

    Each hypothesis uses two oriented points: the axis is n1 x n2 and the axis
    point is where the two normal lines pass closest to each other. `normals`
    seed hypotheses (smoothed: robust to noise); `gate_normals` score and gate
    inliers (raw: sharp at edges). Defaults to `normals`.
    """
    n = len(points)
    if n < 10:
        raise FitError("Select more faces on the cylindrical surface.")
    gate = normals if gate_normals is None else gate_normals
    rng = np.random.default_rng(rng)
    cos_max = np.cos(np.radians(max_angle_deg))
    max_radius = 5.0 * np.linalg.norm(points.max(axis=0) - points.min(axis=0))
    sample = rng.choice(n, min(n, SCORE_SAMPLE), replace=False)
    P, PN = points[sample], gate[sample]

    best, best_score = None, -1
    for start in range(0, iterations, BATCH):
        i, j = rng.integers(0, n, size=(2, min(BATCH, iterations - start)))
        p1, n1, p2, n2 = points[i], normals[i], points[j], normals[j]
        axis = np.cross(n1, n2)
        s = np.linalg.norm(axis, axis=1)
        b = np.einsum("ij,ij->i", n1, n2)
        w0 = p1 - p2
        t = (b * np.einsum("ij,ij->i", n2, w0) - np.einsum("ij,ij->i", n1, w0)) / np.maximum(1.0 - b * b, 1e-12)
        radius = np.abs(t)
        ok = (s > np.sin(np.radians(3.0))) & (radius > 1e-9) & (radius < max_radius)
        if not ok.any():
            continue
        axis, c, radius = axis[ok] / s[ok, None], (p1 + t[:, None] * n1)[ok], radius[ok]

        w = P[None, :, :] - c[:, None, :]  # (k, M, 3)
        radial = w - np.einsum("kmj,kj->km", w, axis)[..., None] * axis[:, None, :]
        dist = np.linalg.norm(radial, axis=2)
        cos = np.abs(np.einsum("kmj,mj->km", radial, PN)) / np.maximum(dist, 1e-12)
        score = ((np.abs(dist - radius[:, None]) < tol) & (cos > cos_max)).sum(axis=1)
        k = int(np.argmax(score))
        if score[k] > best_score:
            best_score, best = score[k], (c[k], axis[k], radius[k])
    if best is None:
        raise FitError("No cylinder hypothesis possible; are the selected normals all parallel (flat region)?")

    c, a, r = best
    inl = _cylinder_inliers(points, gate, c, a, r, tol, cos_max)
    for _ in range(2):
        if inl.sum() < 10:
            raise FitError("No consistent cylinder found; increase the fit tolerance.")
        c, a, r = refine_cylinder(points[inl], c, a, r, tol)
        inl = _cylinder_inliers(points, gate, c, a, r, tol, cos_max)
    return c, a, r, inl


def _cylinder_inliers(points, normals, c, a, r, tol, cos_max) -> np.ndarray:
    radial = _radial(points, c, a)
    dist = np.linalg.norm(radial, axis=1)
    cos = np.abs(np.einsum("ij,ij->i", radial, normals)) / np.maximum(dist, 1e-12)
    return (np.abs(dist - r) < tol) & (cos > cos_max)


def refine_cylinder(points: np.ndarray, c: np.ndarray, a: np.ndarray, r: float, tol: float):
    """Robust nonlinear least squares over axis tilt (2), axis offset (2) and radius."""
    u, v = _basis(a)

    def unpack(x):
        axis = _unit(a + x[0] * u + x[1] * v)
        return c + x[2] * u + x[3] * v, axis, x[4]

    def residuals(x):
        cc, aa, rr = unpack(x)
        return np.linalg.norm(_radial(points, cc, aa), axis=1) - rr

    sol = least_squares(residuals, np.array([0.0, 0.0, 0.0, 0.0, r]), loss="soft_l1", f_scale=max(tol, 1e-9))
    cc, aa, rr = unpack(sol.x)
    return cc, aa, float(abs(rr))


def fit_cylinder_region(mesh: MeshData, face_ids: np.ndarray, tol: float, max_angle_deg: float = 25.0, rng=None):
    """RANSAC cylinder on selected faces. Returns (CylinderFeature, inlier face ids)."""
    centers = mesh.face_centers()[face_ids]
    normals = mesh.face_normals()[face_ids]
    smooth = mesh.smooth_face_normals()[face_ids]
    c, a, r, inl = ransac_cylinder(centers, smooth, tol, max_angle_deg, rng=rng, gate_normals=normals)
    inlier_faces = face_ids[inl]
    verts = face_vertices(mesh, inlier_faces)
    c, a, r = refine_cylinder(verts, c, a, r, tol)
    verts = verts[np.abs(np.linalg.norm(_radial(verts, c, a), axis=1) - r) < tol]  # drop rim vertices
    c, a, r = refine_cylinder(verts, c, a, r, tol)
    a = _canonical(a)

    along = (verts - c) @ a
    t0, t1 = along.min(), along.max()
    center = c + 0.5 * (t0 + t1) * a
    radial = _radial(verts, center, a)
    res = np.linalg.norm(radial, axis=1) - r

    # Arc coverage in 5-degree bins: a near-flat region fits a meaningless huge cylinder.
    u, v = _basis(a)
    bins = np.unique(np.floor(np.degrees(np.arctan2(radial @ v, radial @ u)) / ARC_BIN_DEG))
    arc = float(len(bins) * ARC_BIN_DEG)
    if arc < MIN_ARC_DEG:
        raise FitError(f"Region spans only ~{arc:.0f}° of arc and looks flat; select more of the cylindrical surface.")
    # Hole if surface normals point toward the axis.
    outward = np.einsum("ij,ij->i", _radial(centers[inl], center, a), normals[inl]).mean()
    feature = CylinderFeature(
        center=center,
        axis=a,
        radius=r,
        length=float(t1 - t0),
        rms=float(np.sqrt(np.mean(res**2))),
        n_points=len(verts),
        kind="boss" if outward > 0 else "hole",
        arc_deg=arc,
    )
    return feature, inlier_faces
