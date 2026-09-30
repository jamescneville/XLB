"""
Uniform surface remeshing for the surface-field (USD/VTK) export.

The mesh that carries the mapped surface field only needs to be *close* to the
CAD (well inside a voxel); it does not have to be the CAD triangulation. This
module resamples an arbitrary triangle mesh to a target edge length so the
export mesh is consistent in density regardless of how the CAD was tessellated:
dense meshes get much smaller, coarse meshes get enough vertices to carry the
field.

Method: per connected component, uniform vertex clustering (ACVD via pyacvd),
then every new vertex is snapped back onto the ORIGINAL surface so the result
stays within sagitta distance (~e^2 * curvature / 8) of the input. Open shells
and gaps are left as they are; nothing is closed or made watertight.

Units: meters, same as the mesh handed to the solver. OBJ (Alias, cm) must be
scaled before calling; the CLI has ``--scale`` for that.
"""

import argparse
import time

import numpy as np
import trimesh

_EQUILATERAL = np.sqrt(3.0) / 2.0  # vertex area of an equilateral tiling = _EQUILATERAL * edge^2


def _face_components(faces, n_vertices):
    """Connected-component label per face (shared vertices), scipy only."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    f = np.asarray(faces)
    r = np.concatenate([f[:, 0], f[:, 1]])
    c = np.concatenate([f[:, 1], f[:, 2]])
    g = coo_matrix((np.ones(len(r), dtype=np.int8), (r, c)), shape=(n_vertices, n_vertices))
    _, vlab = connected_components(g, directed=False)
    return vlab[f[:, 0]]


def _to_polydata(vertices, faces):
    import pyvista as pv

    n = len(faces)
    cells = np.hstack([np.full((n, 1), 3, dtype=np.int64), faces.astype(np.int64)]).ravel()
    return pv.PolyData(np.asarray(vertices, dtype=np.float64), cells)


def _decimate(vertices, faces, target_faces):
    import open3d as o3d

    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32)),
    )
    m = m.simplify_quadric_decimation(int(target_faces))
    m.remove_degenerate_triangles()
    m.remove_unreferenced_vertices()
    return np.asarray(m.vertices), np.asarray(m.triangles)


def _cluster_component(vertices, faces, n_clusters):
    """Uniform ACVD resample of one connected component -> (vertices, faces)."""
    import pyacvd

    # Clustering wants many more input points than clusters; decimate huge input
    # first (cheap, projection recovers fidelity) and subdivide coarse input.
    if len(faces) > 16 * n_clusters:
        vertices, faces = _decimate(vertices, faces, 8 * n_clusters)
    clus = pyacvd.Clustering(_to_polydata(vertices, faces))
    while clus.mesh.n_points < 4 * n_clusters:
        clus.subdivide(1)
    clus.cluster(n_clusters)
    out = clus.create_mesh(moveclus=False)  # we project onto the source ourselves
    return np.asarray(out.points), np.asarray(out.faces).reshape(-1, 4)[:, 1:]


def _cluster_job(job):
    v, f, n_clusters = job
    return _cluster_component(v, f, n_clusters)


def _is_windows():
    import sys

    return sys.platform.startswith("win")


def _closest_point_scene(mesh):
    import open3d as o3d

    legacy = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy))
    return scene


def _closest(scene, points):
    import open3d as o3d

    ans = scene.compute_closest_points(o3d.core.Tensor(np.asarray(points, dtype=np.float32)))
    return ans["points"].numpy().astype(np.float64), ans["primitive_ids"].numpy()


def _block_ids(points, origin, block_len, dims):
    """Unique linear ids of the blocks that contain ``points``."""
    b = np.floor((points - origin) / block_len).astype(np.int64)
    b = np.clip(b, 0, np.asarray(dims) - 1)
    return np.unique((b[:, 0] * dims[1] + b[:, 1]) * dims[2] + b[:, 2])


def _dist_only(scene, points):
    import open3d as o3d

    d = scene.compute_distance(o3d.core.Tensor(np.asarray(points, dtype=np.float32))).numpy()
    return d.astype(np.float64)


def _keep_exterior_shells(m):
    """Orient outward and drop shells not reachable from outside (air pockets, nested contents)."""
    v, f = np.asarray(m.vertices), np.asarray(m.faces)
    lab = _face_components(f, len(v))
    o = v[f[:, 0]]
    vol6 = np.einsum("ij,ij->i", o, np.cross(v[f[:, 1]] - o, v[f[:, 2]] - o))  # 6 * signed tet volume
    vol = np.bincount(lab, weights=vol6)
    # marching-cubes winding is a library convention: fix it from the biggest shell
    if vol[np.argmax(np.abs(vol))] < 0:
        f = f[:, ::-1].copy()
        vol = -vol
    solid = vol > 0  # shells bounding solid; negative = enclosed air pocket
    m2 = trimesh.Trimesh(v, f[solid[lab]], process=False)
    m2.remove_unreferenced_vertices()

    pockets = ~solid[lab]
    if pockets.any() and solid.sum() > 1:
        # solid shells sitting inside an air pocket are hidden contents: drop them
        import open3d as o3d

        pm = trimesh.Trimesh(v, f[pockets], process=False)
        pm.remove_unreferenced_vertices()
        ps = o3d.t.geometry.RaycastingScene()
        ps.add_triangles(
            o3d.t.geometry.TriangleMesh.from_legacy(
                o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(pm.vertices), o3d.utility.Vector3iVector(pm.faces))
            )
        )
        k_lab = _face_components(m2.faces, len(m2.vertices))
        ids, first_face = np.unique(k_lab, return_index=True)
        pts = np.asarray(m2.vertices)[np.asarray(m2.faces)[first_face, 0]]
        inside = ps.compute_occupancy(o3d.core.Tensor(pts.astype(np.float32))).numpy() > 0.5
        drop = np.isin(k_lab, ids[inside])
        if drop.any():
            m2 = trimesh.Trimesh(np.asarray(m2.vertices), np.asarray(m2.faces)[~drop], process=False)
            m2.remove_unreferenced_vertices()
    return m2


def _sealed_classes(v, f, gap, verbose=True):
    """
    Coarse flood fill that decides which air is truly outside.

    Cells within ``gap`` of the soup are walls; the exterior is the air connected to the
    domain boundary through the remaining cells. Air that is only reachable through gaps
    narrower than ``2*gap`` is "sealed" (interior of a leaky body). Wall cells take the
    class of their nearest free cell. Returns (origin, cell, sealed_bool_grid).
    """
    from scipy import ndimage as ndi

    hc = 0.5 * float(gap)
    pad = gap + 3.0 * hc
    lo = v.min(axis=0) - pad
    hi = v.max(axis=0) + pad
    while np.prod(np.ceil((hi - lo) / hc) + 1) > 2.5e8:
        hc *= 1.25
    dims = (np.ceil((hi - lo) / hc).astype(np.int64) + 1)
    sv, _ = trimesh.remesh.subdivide_to_size(v, f, max_edge=0.5 * hc, max_iter=14)
    occ = np.zeros(tuple(dims), dtype=bool)
    ijk = np.clip(np.floor((sv - lo) / hc).astype(np.int64), 0, dims - 1)
    occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    blocked = ndi.distance_transform_edt(~occ) * hc <= gap
    del occ
    lab, n = ndi.label(~blocked)
    faces_lab = np.concatenate([lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(), lab[:, -1].ravel(),
                                lab[:, :, 0].ravel(), lab[:, :, -1].ravel()])
    is_ext = np.zeros(n + 1, dtype=bool)
    is_ext[np.unique(faces_lab)] = True
    is_ext[0] = False
    sealed_free = (~blocked) & ~is_ext[lab]
    del lab
    ind = ndi.distance_transform_edt(blocked, return_distances=False, return_indices=True)
    cls = sealed_free[ind[0], ind[1], ind[2]]
    if verbose:
        print(f"\tWrap gap closure: coarse {hc * 1000:.1f} mm grid {tuple(dims)}, "
              f"{100.0 * cls.mean():.1f}% of cells sealed")
    return lo, hc, cls


def wrap_surface(mesh, resolution, offset, gap_closure=0.0, verbose=True):
    """
    Boundary wrap: closed, oriented, outward-offset surface around a triangle soup.

    Level set {distance-to-soup = offset} on a sparse narrow band of a uniform grid
    (``resolution`` cell size), extracted with marching cubes. Any soup (open shells,
    gaps, overlaps, thin sheets, mixed winding) yields a closed manifold shell; sheets
    gain thickness ``2*offset`` and gaps under ``2*offset`` close. Shells enclosing
    pockets that cannot be reached from outside (internals of a closed body, ...) are
    discarded, so normals always point into the flow.

    Parameters
    ----------
    mesh : trimesh.Trimesh   input soup, meters
    resolution : float       grid cell size h (meters)
    offset : float           wrap offset (meters); ~0.5-0.75*h is the practical minimum for thin sheets
    gap_closure : float      seal leaks up to ~2*gap_closure wide when deciding what is interior
                             (0 = off; only shells fully enclosed by the wrap itself are dropped)

    Returns
    -------
    trimesh.Trimesh  (closed, outward normals)
    """
    import pyvista as pv
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    tic = time.perf_counter()
    h, off = float(resolution), float(offset)
    if h <= 0 or off <= 0:
        raise ValueError("wrap resolution and offset must be > 0")
    if off < 0.5 * h:
        print(f"\tWARNING wrap: offset {off * 1000:.2f} mm < 0.5*resolution; thin sheets may not close")

    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    scene = _closest_point_scene(mesh)

    sealed = _sealed_classes(v, f, float(gap_closure), verbose) if gap_closure and gap_closure > 0 else None
    t_seal = time.perf_counter()

    # Only nodes near the soup are evaluated: sub-blocks (Be nodes) touched by samples, +/-1.
    Be, Bm = 4, 16  # evaluation sub-block / marching block, in nodes
    sv, _ = trimesh.remesh.subdivide_to_size(v, f, max_edge=2.0 * h, max_iter=14)
    pad = off + 3.0 * h
    origin = v.min(axis=0) - pad
    extent = v.max(axis=0) + pad - origin
    dims_m = np.ceil(extent / (h * Bm)).astype(np.int64) + 1  # marching blocks per axis
    dims_e = dims_m * (Bm // Be)
    shape = tuple(int(d) * Bm + 1 for d in dims_m)
    n_nodes = shape[0] * shape[1] * shape[2]
    if n_nodes > 1.5e9:
        raise MemoryError(f"wrap grid would be {shape} ({n_nodes / 1e6:.0f}M nodes); use a larger wrap resolution")

    ids = _block_ids(sv, origin, h * Be, dims_e)
    bx, rem = np.divmod(ids, dims_e[1] * dims_e[2])
    by, bz = np.divmod(rem, dims_e[2])
    blocks = np.stack([bx, by, bz], axis=1)
    grown = np.concatenate([blocks + np.array([i, j, k]) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])
    grown = grown[((grown >= 0) & (grown < dims_e)).all(axis=1)]
    lin = np.unique((grown[:, 0] * dims_e[1] + grown[:, 1]) * dims_e[2] + grown[:, 2])
    bx, rem = np.divmod(lin, dims_e[1] * dims_e[2])
    by, bz = np.divmod(rem, dims_e[2])
    eval_blocks = np.stack([bx, by, bz], axis=1)

    field = np.full(shape, 4.0 * h, dtype=np.float32)  # "far outside"
    flat = field.reshape(-1)  # view
    loc = np.stack(np.meshgrid(*[np.arange(Be)] * 3, indexing="ij"), axis=-1).reshape(-1, 3)
    step = max(1, 4_000_000 // len(loc))
    for a_ in range(0, len(eval_blocks), step):
        idx = (eval_blocks[a_ : a_ + step, None, :] * Be + loc[None, :, :]).reshape(-1, 3)
        pos = origin + idx * h
        val = _dist_only(scene, pos) - off
        if sealed is not None:
            so, sc_, cls = sealed
            c = np.clip(np.floor((pos - so) / sc_).astype(np.int64), 0, np.asarray(cls.shape) - 1)
            val = np.where(cls[c[:, 0], c[:, 1], c[:, 2]], -(np.abs(val) + 1e-3 * h), val)
        flat[(idx[:, 0] * shape[1] + idx[:, 1]) * shape[2] + idx[:, 2]] = val
    blocks = np.unique(eval_blocks // (Bm // Be), axis=0)  # marching blocks that hold any evaluated node
    B = Bm
    t_field = time.perf_counter()

    all_v, all_f, all_b, base = [], [], [], 0
    grid = pv.ImageData(dimensions=(B + 1,) * 3, spacing=(h, h, h))  # block-local coords: exact in float32
    for bxyz in blocks:
        x0, y0, z0 = (int(c) * B for c in bxyz)
        vol = field[x0 : x0 + B + 1, y0 : y0 + B + 1, z0 : z0 + B + 1]
        if vol.min() >= 1e-4 * h or vol.max() <= 1e-4 * h:
            continue
        grid.point_data["f"] = vol.ravel(order="F")
        c = grid.contour([1e-4 * h], scalars="f", method="flying_edges", compute_normals=False, compute_gradients=False)
        if c.n_cells == 0:
            continue
        vv = np.asarray(c.points, dtype=np.float64)
        ff = np.asarray(c.faces).reshape(-1, 4)[:, 1:]
        t = vv / h
        on_plane = ((np.abs(t) < 1e-2) | (np.abs(t - B) < 1e-2)).any(axis=1)  # shared with a neighbour block
        all_v.append(vv + origin + np.array([x0, y0, z0]) * h)
        all_f.append(ff + base)
        all_b.append(on_plane)
        base += len(vv)
    if not all_v:
        raise ValueError("wrap_surface: empty iso-surface (offset too small for this resolution?)")
    av, af, ab = np.vstack(all_v), np.vstack(all_f), np.concatenate(all_b)

    t_mc = time.perf_counter()
    # weld only the vertices that sit on block-boundary planes
    keep = np.flatnonzero(~ab)
    bv = av[ab]
    pairs = cKDTree(bv).query_pairs(r=5e-3 * h, output_type="ndarray")
    g = coo_matrix((np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])), shape=(len(bv), len(bv)))
    ngrp, grp = connected_components(g, directed=False)
    rep = np.zeros(ngrp, dtype=np.int64)
    rep[grp[::-1]] = np.arange(len(bv))[::-1]  # first vertex of each group
    remap = np.empty(len(av), dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    remap[ab] = len(keep) + grp
    verts = np.vstack([av[keep], bv[rep]])
    wrapped = trimesh.Trimesh(verts, remap[af], process=False)
    wrapped.update_faces(wrapped.nondegenerate_faces())
    wrapped.remove_unreferenced_vertices()

    wrapped = _keep_exterior_shells(wrapped)
    if verbose:
        print(
            f"\tSurface wrap: {len(f):,} tris -> {len(wrapped.faces):,} tris "
            f"(h {h * 1000:.1f} mm, offset {off * 1000:.1f} mm, {len(blocks):,} blocks, "
            f"seal {t_seal - tic:0.1f} s, field {t_field - t_seal:0.1f} s, "
            f"mesh+weld {t_mc - t_field:0.1f} s, total {time.perf_counter() - tic:0.1f} s)"
        )
    return wrapped


def _snap_to_source(m, orig, max_dist, offset, verbose=True):
    """
    Pull the wrapped/resampled vertices back toward the ORIGINAL CAD to recover edge position.

    Each vertex moves to its closest point on the CAD, then backs off ``offset`` along the same
    line, so it stays on its own side. The two skins of a zero-thickness sheet therefore stay
    ``2*offset`` apart instead of collapsing onto one another. Vertices farther than
    ``max_dist`` from the CAD (bridged gaps) are left where the wrap put them.
    """
    scene = _closest_point_scene(orig)
    v = np.asarray(m.vertices)
    p, _ = _closest(scene, v)
    dv = p - v
    d = np.linalg.norm(dv, axis=1)
    ok = (d <= max_dist) & (d > 1e-12)
    back = np.minimum(float(offset), d)
    new = v.copy()
    new[ok] = p[ok] - dv[ok] / d[ok, None] * back[ok, None]
    if verbose:
        print(f"	Snap to source: {100.0 * ok.mean():.1f}% of vertices moved "
              f"(mean {d[ok].mean() * 1000:.2f} mm), left {float(offset) * 1000:.2f} mm off the CAD")
    return trimesh.Trimesh(new, np.asarray(m.faces), process=False)


def remesh_surface(mesh, target_edge=None, max_faces=None, project=True, workers=1, verbose=True,
                   wrap_resolution=None, wrap_offset=None, gap_closure=0.0, snap_offset=None):
    """
    Resample ``mesh`` to a roughly uniform ``target_edge`` (meters).

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input triangle mesh (meters). May be dirty: gaps, open shells, multiple
        bodies, non-manifold edges are all fine.
    target_edge : float
        Desired mean edge length in meters (~0.5-1x the finest voxel is a good
        starting point for a surface field).
    max_faces : int, optional
        Upper bound on output faces; the edge length is coarsened to honour it.
    project : bool
        Snap the new vertices onto the original surface (recommended).
    workers : int
        Processes used across connected components. Only use >1 from a script
        with an ``if __name__ == "__main__"`` guard and no live GPU context;
        :func:`remesh_surface_isolated` handles that for the solver.

    Returns
    -------
    trimesh.Trimesh
    """
    tic = time.perf_counter()
    if target_edge is None and not wrap_resolution:
        raise ValueError("need target_edge and/or wrap_resolution")
    if target_edge is not None:
        target_edge = float(target_edge)
        if target_edge <= 0:
            raise ValueError("target_edge must be > 0")

    src = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.faces), process=False)
    src.merge_vertices()  # weld split patches so connectivity is real
    src.update_faces(src.nondegenerate_faces())
    src.remove_unreferenced_vertices()
    if len(src.faces) == 0:
        raise ValueError("remesh_surface: input mesh has no valid faces")

    orig, wrap_off = None, None
    if wrap_resolution:
        # Closed, oriented shell just outside the soup; everything below works on that instead.
        orig = src
        wrap_off = wrap_offset if wrap_offset else 0.75 * float(wrap_resolution)
        src = wrap_surface(src, wrap_resolution, wrap_off, gap_closure=gap_closure, verbose=verbose)
        if target_edge is None:
            if snap_offset is not None:
                src = _snap_to_source(src, orig, 1.5 * wrap_off, snap_offset, verbose)
            return src

    edge = target_edge
    if max_faces is not None:
        # faces ~= 2 * vertices = 2 * area / (0.866 * e^2)
        edge = max(edge, float(np.sqrt(2.0 * src.area / (_EQUILATERAL * max_faces))))
        if edge > target_edge and verbose:
            print(f"\tRemesh: max_faces={max_faces:,} limits edge to {edge * 1000:.2f} mm")

    sv, sf = np.asarray(src.vertices), np.asarray(src.faces)
    labels = _face_components(sf, len(sv))
    order = np.argsort(labels, kind="stable")
    bounds = np.flatnonzero(np.diff(labels[order])) + 1
    tri_area = 0.5 * np.linalg.norm(np.cross(sv[sf[:, 1]] - sv[sf[:, 0]], sv[sf[:, 2]] - sv[sf[:, 0]]), axis=1)
    jobs = []
    for idx in np.split(order, bounds):
        used, inv = np.unique(sf[idx], return_inverse=True)
        n_clusters = int(tri_area[idx].sum() / (_EQUILATERAL * edge * edge))
        jobs.append((sv[used], inv.reshape(-1, 3), n_clusters))

    # Biggest first so a few large shells don't leave the pool idle at the end.
    big = sorted((i for i, j in enumerate(jobs) if j[2] >= 8 and len(j[1]) >= 4), key=lambda i: -len(jobs[i][1]))
    results = {}
    if workers > 1 and len(big) > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        ctx = mp.get_context("spawn" if mp.get_start_method(allow_none=True) == "spawn" or _is_windows() else "fork")
        with ProcessPoolExecutor(min(workers, len(big)), mp_context=ctx) as ex:
            for i, r in zip(big, ex.map(_cluster_job, [jobs[i] for i in big], chunksize=1)):
                results[i] = r
    else:
        for i in big:
            results[i] = _cluster_job(jobs[i])

    out_v, out_f, offset = [], [], 0
    for i, (v, f, _) in enumerate(jobs):  # pieces too small to resample are kept as-is
        v, f = results.get(i, (v, f))
        out_f.append(f + offset)
        out_v.append(v)
        offset += len(v)

    verts = np.vstack(out_v)
    faces = np.vstack(out_f)

    # pyacvd can emit NaNs (zero-length cluster normals) on degenerate patches.
    bad = ~np.isfinite(verts).all(axis=1)
    if bad.any():
        faces = faces[~bad[faces].any(axis=1)]
        verts = np.where(bad[:, None], 0.0, verts)

    if project:
        scene = _closest_point_scene(src)
        verts, _ = _closest(scene, verts)

    out = trimesh.Trimesh(verts, faces, process=False)
    out.update_faces(out.nondegenerate_faces())
    out.remove_unreferenced_vertices()

    # Match input winding: flip any component whose normals disagree with the source.
    if project:
        cen = out.triangles_center
        _, pid = _closest(scene, cen)
        agree = np.einsum("ij,ij->i", out.face_normals, src.face_normals[pid])
        labels = _face_components(out.faces, len(out.vertices))
        flip = np.zeros(len(out.faces), dtype=bool)
        for lab in np.unique(labels):
            sel = labels == lab
            if agree[sel].sum() < 0:
                flip[sel] = True
        if flip.any():
            f = out.faces.copy()
            f[flip] = f[flip][:, ::-1]
            out = trimesh.Trimesh(out.vertices, f, process=False)

    if snap_offset is not None and orig is not None:
        out = _snap_to_source(out, orig, 1.5 * wrap_off, snap_offset, verbose)

    if verbose:
        dev = ""
        if project:
            d = np.linalg.norm(_closest(scene, out.triangles_center)[0] - out.triangles_center, axis=1)
            dev = f", deviation mean {d.mean() * 1000:.3f} / p99 {np.percentile(d, 99) * 1000:.2f} / max {d.max() * 1000:.1f} mm"
        el = out.edges_unique_length
        print(
            f"\tSurface remesh: {len(src.faces):,} -> {len(out.faces):,} tris "
            f"(edge mean {el.mean() * 1000:.2f} mm, target {edge * 1000:.2f} mm{dev}) "
            f"in {time.perf_counter() - tic:0.1f} s"
        )
    return out


def remesh_surface_isolated(mesh, target_edge=None, max_faces=None, workers=None,
                            wrap_resolution=None, wrap_offset=None, gap_closure=0.0, snap_offset=None):
    """
    :func:`remesh_surface` in a clean child process, parallel across components.

    Safe to call from the solver: the child never touches CUDA/Warp, so forking
    workers is fine, and the parent's ``__main__`` is not re-imported.
    """
    import os
    import subprocess
    import sys
    import tempfile

    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    with tempfile.TemporaryDirectory() as tmp:
        fin, fout = os.path.join(tmp, "in.npz"), os.path.join(tmp, "out.npz")
        np.savez(fin, v=np.asarray(mesh.vertices, dtype=np.float64), f=np.asarray(mesh.faces, dtype=np.int64))
        cmd = [sys.executable, os.path.abspath(__file__), fin, fout, "--workers", str(workers)]
        if target_edge:
            cmd += ["--edge", repr(float(target_edge))]
        if wrap_resolution:
            cmd += ["--wrap-res", repr(float(wrap_resolution)), "--gap-closure", repr(float(gap_closure or 0.0))]
            if wrap_offset:
                cmd += ["--wrap-offset", repr(float(wrap_offset))]
            if snap_offset is not None:
                cmd += ["--snap-offset", repr(float(snap_offset))]
        if max_faces:
            cmd += ["--max-faces", str(int(max_faces))]
        subprocess.run(cmd, check=True)
        with np.load(fout) as d:  # close before the temp dir is removed (Windows)
            return trimesh.Trimesh(d["v"], d["f"], process=False)


def _main():
    ap = argparse.ArgumentParser(description="Uniformly remesh a surface for the USD/VTK surface-field export.")
    ap.add_argument("input")
    ap.add_argument("output", help="STL/OBJ/PLY, by extension")
    ap.add_argument("--edge", type=float, default=None, help="target edge length in METERS (after --scale)")
    ap.add_argument("--wrap-res", type=float, default=None, help="boundary-wrap grid size in meters (enables the wrap)")
    ap.add_argument("--wrap-offset", type=float, default=None, help="wrap offset in meters (default 0.75*wrap-res)")
    ap.add_argument("--snap-offset", type=float, default=None,
                    help="after the wrap, pull vertices back to within this distance (m) of the original CAD to recover edges")
    ap.add_argument("--gap-closure", type=float, default=0.0, help="seal leaks up to ~2x this when dropping interiors (m)")
    ap.add_argument("--scale", type=float, default=1.0, help="input scale to meters (0.01 for Alias OBJ in cm)")
    ap.add_argument("--max-faces", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()

    if a.input.endswith(".npz"):
        with np.load(a.input) as d:
            m = trimesh.Trimesh(d["v"], d["f"], process=False)
    else:
        m = trimesh.load(a.input, force="mesh", process=False)
    if a.scale != 1.0:
        m.apply_scale(a.scale)
    r = remesh_surface(m, a.edge, max_faces=a.max_faces, workers=a.workers,
                       wrap_resolution=a.wrap_res, wrap_offset=a.wrap_offset, gap_closure=a.gap_closure,
                       snap_offset=a.snap_offset)
    if a.output.endswith(".npz"):
        np.savez(a.output, v=np.asarray(r.vertices), f=np.asarray(r.faces))
    else:
        r.export(a.output)
    print(f"wrote {a.output}")


if __name__ == "__main__":
    _main()
