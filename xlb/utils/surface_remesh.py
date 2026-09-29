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


def remesh_surface(mesh, target_edge, max_faces=None, project=True, workers=1, verbose=True):
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
    target_edge = float(target_edge)
    if target_edge <= 0:
        raise ValueError("target_edge must be > 0")

    src = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.faces), process=False)
    src.merge_vertices()  # weld split patches so connectivity is real
    src.update_faces(src.nondegenerate_faces())
    src.remove_unreferenced_vertices()
    if len(src.faces) == 0:
        raise ValueError("remesh_surface: input mesh has no valid faces")

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


def remesh_surface_isolated(mesh, target_edge, max_faces=None, workers=None):
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
        cmd = [sys.executable, os.path.abspath(__file__), fin, fout, "--edge", repr(float(target_edge)), "--workers", str(workers)]
        if max_faces:
            cmd += ["--max-faces", str(int(max_faces))]
        subprocess.run(cmd, check=True)
        with np.load(fout) as d:  # close before the temp dir is removed (Windows)
            return trimesh.Trimesh(d["v"], d["f"], process=False)


def _main():
    ap = argparse.ArgumentParser(description="Uniformly remesh a surface for the USD/VTK surface-field export.")
    ap.add_argument("input")
    ap.add_argument("output", help="STL/OBJ/PLY, by extension")
    ap.add_argument("--edge", type=float, required=True, help="target edge length in METERS (after --scale)")
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
    r = remesh_surface(m, a.edge, max_faces=a.max_faces, workers=a.workers)
    if a.output.endswith(".npz"):
        np.savez(a.output, v=np.asarray(r.vertices), f=np.asarray(r.faces))
    else:
        r.export(a.output)
    print(f"wrote {a.output}")


if __name__ == "__main__":
    _main()
