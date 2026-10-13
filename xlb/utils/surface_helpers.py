
import numpy as np
from scipy.spatial import cKDTree



def poisson_roots(vertices, faces, vertex_normals, spacing, rng, oversample=5.0, accept=None, existing=None, surface_cache=None):
    """Area-weighted random sequential (dart-throwing) sampling of the surface.

    Two points closer than ``spacing`` exclude each other only if their normals agree
    (dot > 0), so the two faces of a thin sheet can both carry tufts.
    ``accept(points, normals) -> bool mask`` filters the candidates BEFORE the thinning, so an unusable candidate
    never takes a slot from a good neighbour. ``existing`` = (points, normals) of roots already placed: candidates
    that one of them excludes are dropped (used to fill holes). Returns (points (T,3), normals (T,3)).
    """
    # surface_cache = (triangles, per-triangle areas, total area); used for refill passes.
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if surface_cache is None:
        tri = v[f]
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        area = 0.5 * np.linalg.norm(cross, axis=1)
        total = float(area.sum())
    else:
        tri, area, total = surface_cache
    if total <= 0.0:
        return np.empty((0, 3)), np.empty((0, 3))

    n_cand = int(oversample * total / spacing**2) + 1
    fi = rng.choice(len(f), size=n_cand, p=area / total)
    r1 = np.sqrt(rng.random(n_cand))
    r2 = rng.random(n_cand)
    w = np.stack([1.0 - r1, r1 * (1.0 - r2), r1 * r2], axis=1)
    pts = np.einsum("ni,nij->nj", w, tri[fi])
    nrm = np.einsum("ni,nij->nj", w, np.asarray(vertex_normals, dtype=np.float64)[f[fi]])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-20)
    if accept is not None:
        ok = np.asarray(accept(pts, nrm), dtype=bool)
        pts, nrm, n_cand = pts[ok], nrm[ok], int(ok.sum())
        if n_cand == 0:
            return np.empty((0, 3)), np.empty((0, 3))

    tree = cKDTree(pts)
    neigh = tree.query_ball_point(pts, r=spacing, workers=-1)
    alive = np.ones(n_cand, dtype=bool)
    if existing is not None and len(existing[0]):
        ex_p, ex_n = np.asarray(existing[0], dtype=np.float64), np.asarray(existing[1], dtype=np.float64)
        for i, nb in enumerate(cKDTree(ex_p).query_ball_point(pts, r=spacing, workers=-1)):
            if nb and (ex_n[nb] @ nrm[i] > 0.0).any():
                alive[i] = False
    keep = []
    for i in range(n_cand):  # candidates are already in random order
        if not alive[i]:
            continue
        keep.append(i)
        nb = np.asarray(neigh[i], dtype=np.int64)
        nb = nb[nrm[nb] @ nrm[i] > 0.0]
        alive[nb] = False
    keep = np.asarray(keep, dtype=np.int64)
    return pts[keep], nrm[keep]


def _unit(a, fallback=None):
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    out = a / np.maximum(n, 1e-20)
    if fallback is not None:
        bad = (n[..., 0] < 1e-12)
        out = np.where(bad[..., None], fallback, out)
    return out


def local_surface(tree, V, N, p, ref, same_skin=True, thin=0.001):
    """
    Nearest surface vertices to ``p`` that belong to the skin ``p`` is riding on. Returns (vi (T,4), w (T,4), v0 (T,)):
    indices, inverse-distance weights (0 for rejected) and the nearest accepted vertex.

    The 8 nearest vertices are screened and the 4 nearest accepted ones kept, so with a single skin nothing is
    rejected and the result equals a plain 4-nearest query. A vertex is rejected when
    it lies more than ``thin`` behind the nearest same-facing vertex's tangent plane AND its normal either faces
    away from ``ref`` (the other face of a thin sheet) or is nearly parallel to that vertex's normal (a second skin
    under this one). A crease that turns the normal is kept: its far side is behind the plane too, but its normal
    is neither opposed nor parallel.
    ``same_skin=False`` is the plain 4-nearest query.
    """
    if not same_skin:
        dv, vi = tree.query(p, k=4, workers=-1)
        return vi, 1.0 / np.maximum(dv, 1e-4), vi[:, 0]
    dv, vi = tree.query(p, k=8, workers=-1)
    rows = np.arange(len(p))
    agree = np.einsum("tkj,tj->tk", N[vi], ref) >= 0.0
    first = np.argmax(agree, axis=1)                       # nearest same-facing (0 if none)
    v0 = vi[rows, first]
    n0 = N[v0]
    behind = np.einsum("tkj,tj->tk", V[vi] - V[v0][:, None, :], n0) < -thin
    parallel = np.einsum("tkj,tj->tk", N[vi], n0) > 0.9
    ok = ~(behind & (~agree | parallel))
    ok[rows, first] = True
    ok &= (np.cumsum(ok, axis=1) <= 4)
    order = np.argsort(~ok, axis=1, kind="stable")[:, :4]
    vi, dv, ok = (np.take_along_axis(a, order, axis=1) for a in (vi, dv, ok))
    return vi, np.where(ok, 1.0 / np.maximum(dv, 1e-4), 0.0), v0
