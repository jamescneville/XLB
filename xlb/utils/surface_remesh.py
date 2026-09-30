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
    while clus.mesh.n_points < 2 * n_clusters:   # ACVD needs clearly more points than clusters; 2x is plenty
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


_GPU = {"checked": False, "ok": False}


def _make_kernels(st):
    wp = st["wp"]

    @wp.kernel
    def k_dist(mesh: wp.uint64, pts: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.float32), maxd: float):
        i = wp.tid()
        q = wp.mesh_query_point_no_sign(mesh, pts[i], maxd)
        if q.result:
            out[i] = wp.length(wp.mesh_eval_position(mesh, q.face, q.u, q.v) - pts[i])
        else:
            out[i] = maxd

    @wp.kernel
    def k_closest(
        mesh: wp.uint64,
        pts: wp.array(dtype=wp.vec3),
        out_p: wp.array(dtype=wp.vec3),
        out_f: wp.array(dtype=wp.int32),
        maxd: float,
    ):
        i = wp.tid()
        q = wp.mesh_query_point_no_sign(mesh, pts[i], maxd)
        if q.result:
            out_p[i] = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
            out_f[i] = q.face
        else:
            out_p[i] = pts[i]
            out_f[i] = -1

    @wp.kernel
    def k_field(
        mesh: wp.uint64,
        blocks: wp.array(dtype=wp.vec3i),
        origin: wp.vec3,
        h: float,
        off: float,
        far: float,
        be: int,
        ny: int,
        nz: int,
        field: wp.array(dtype=wp.float32),
        cls: wp.array3d(dtype=wp.uint8),
        use_cls: int,
        c_origin: wp.vec3,
        c_h: float,
    ):
        tid = wp.tid()
        n3 = be * be * be
        b = tid // n3
        l = tid - b * n3
        lx = l // (be * be)
        ly = (l // be) % be
        lz = l % be
        blk = blocks[b]
        ix = blk[0] * be + lx
        iy = blk[1] * be + ly
        iz = blk[2] * be + lz
        p = origin + wp.vec3(float(ix), float(iy), float(iz)) * h
        q = wp.mesh_query_point_no_sign(mesh, p, far)
        d = far
        if q.result:
            d = wp.length(wp.mesh_eval_position(mesh, q.face, q.u, q.v) - p)
        val = d - off
        if use_cls == 1:
            i0 = wp.clamp(int(wp.floor((p[0] - c_origin[0]) / c_h)), 0, cls.shape[0] - 1)
            i1 = wp.clamp(int(wp.floor((p[1] - c_origin[1]) / c_h)), 0, cls.shape[1] - 1)
            i2 = wp.clamp(int(wp.floor((p[2] - c_origin[2]) / c_h)), 0, cls.shape[2] - 1)
            if cls[i0, i1, i2] != wp.uint8(0):
                val = -(wp.abs(val) + 1.0e-3 * h)
        field[(ix * ny + iy) * nz + iz] = val

    @wp.kernel
    def k_blockmask(
        mesh: wp.uint64,
        origin: wp.vec3,
        h: float,
        be: int,
        ny: int,
        nz: int,
        thresh: float,
        out: wp.array(dtype=wp.uint8),
    ):
        tid = wp.tid()
        bx = tid // (ny * nz)
        rem = tid - bx * (ny * nz)
        by = rem // nz
        bz = rem - by * nz
        half = 0.5 * float(be - 1)
        c = origin + wp.vec3(float(bx * be) + half, float(by * be) + half, float(bz * be) + half) * h
        q = wp.mesh_query_point_no_sign(mesh, c, thresh)
        if q.result:
            out[tid] = wp.uint8(1)
        else:
            out[tid] = wp.uint8(0)

    st["k_dist"], st["k_closest"], st["k_field"], st["k_blockmask"] = k_dist, k_closest, k_field, k_blockmask


def _gpu_selftest():
    """Run every GPU kernel once on a 2-triangle mesh; any error or wrong answer disables the GPU path."""
    m = trimesh.Trimesh(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float), np.array([[0, 1, 2], [1, 3, 2]]), process=False
    )
    sc = _GpuScene(m)
    d = sc.distance(np.array([[0.25, 0.25, 0.5], [2.0, 0.0, 0.0]]), 10.0)
    assert abs(d[0] - 0.5) < 1e-4 and abs(d[1] - 1.0) < 1e-4, f"distance kernel returned {d}"
    p, f = sc.closest(np.array([[0.25, 0.25, 0.5]]))
    assert abs(p[0][2]) < 1e-4 and f[0] in (0, 1), f"closest-point kernel returned {p}, {f}"
    origin = np.array([-1.0, -1.0, -1.0])
    blocks = _gpu_block_list(sc, origin, 0.25, 4, np.array([4, 4, 4]), 0.05)
    assert len(blocks) > 0, "block-selection kernel selected nothing"
    fld = _gpu_field(sc, blocks, 4, origin, 0.25, 0.05, (17, 17, 17), None)
    assert fld.shape == (17, 17, 17) and np.isfinite(fld).all(), "field kernel returned bad values"


def _gpu_ok():
    """True if Warp + a CUDA device work. Set XLB_SURFACE_GPU=0 to force the CPU path; _GPU['reason'] says why not."""
    st = _GPU
    if st["checked"]:
        return st["ok"]
    st["checked"] = True
    import os

    st["reason"] = ""
    if os.environ.get("XLB_SURFACE_GPU", "1").strip().lower() in ("0", "false", "no", "off"):
        st["reason"] = "disabled (XLB_SURFACE_GPU=0 / useGpu false)"
        return False
    try:
        import warp as wp

        wp.config.quiet = True
        wp.init()
        if not wp.get_cuda_devices():
            st["reason"] = "no CUDA device"
            return False
        st["wp"], st["dev"] = wp, "cuda:0"
        _make_kernels(st)
        _gpu_selftest()
        st["ok"] = True
    except Exception as e:  # missing warp, old/forked warp without the needed API, driver problems, ...
        st["ok"] = False
        st["reason"] = f"{type(e).__name__}: {str(e)[:120]}"
    return st["ok"]


class _GpuScene:
    """Warp BVH on the GPU: exact point-to-triangle distance / closest point, ~100-200x open3d's CPU speed."""

    CHUNK = 16_000_000

    def __init__(self, mesh):
        wp, dev = _GPU["wp"], _GPU["dev"]
        self._verts = wp.array(np.ascontiguousarray(mesh.vertices, dtype=np.float32), dtype=wp.vec3, device=dev)
        self._idx = wp.array(np.ascontiguousarray(mesh.faces, dtype=np.int32).reshape(-1), dtype=wp.int32, device=dev)
        self.mesh = wp.Mesh(points=self._verts, indices=self._idx)

    def distance(self, points, maxd=1.0e3):
        wp, dev = _GPU["wp"], _GPU["dev"]
        pts = np.ascontiguousarray(points, dtype=np.float32)
        out = np.empty(len(pts), dtype=np.float32)
        for a in range(0, len(pts), self.CHUNK):
            c = pts[a : a + self.CHUNK]
            P = wp.array(c, dtype=wp.vec3, device=dev)
            O = wp.empty(len(c), dtype=wp.float32, device=dev)
            wp.launch(_GPU["k_dist"], dim=len(c), inputs=[self.mesh.id, P, O, float(maxd)], device=dev)
            out[a : a + len(c)] = O.numpy()
        return out.astype(np.float64)

    def closest(self, points):
        wp, dev = _GPU["wp"], _GPU["dev"]
        pts = np.ascontiguousarray(points, dtype=np.float32)
        op = np.empty((len(pts), 3), dtype=np.float32)
        of = np.empty(len(pts), dtype=np.int32)
        for a in range(0, len(pts), self.CHUNK):
            c = pts[a : a + self.CHUNK]
            P = wp.array(c, dtype=wp.vec3, device=dev)
            OP = wp.empty(len(c), dtype=wp.vec3, device=dev)
            OF = wp.empty(len(c), dtype=wp.int32, device=dev)
            wp.launch(_GPU["k_closest"], dim=len(c), inputs=[self.mesh.id, P, OP, OF, 1.0e3], device=dev)
            op[a : a + len(c)] = OP.numpy()
            of[a : a + len(c)] = OF.numpy()
        return op.astype(np.float64), of


def _gpu_block_list(scene, origin, h, be, dims_e, off):
    """
    Sub-blocks that contain a node within reach of the iso-surface, found by one distance query per block.

    A node matters only if it is within ``off + h*sqrt(3)`` of the soup (every marching-cubes cell corner
    around the iso-surface is), so a block is kept when its centre is within that plus the block's
    half-diagonal. Exact, unlike sampling the triangles, and ~5M queries take a fraction of a second.
    """
    wp, dev = _GPU["wp"], _GPU["dev"]
    n = int(dims_e[0]) * int(dims_e[1]) * int(dims_e[2])
    out = wp.zeros(n, dtype=wp.uint8, device=dev)
    thresh = float(off + 2.5 * np.sqrt(3.0) * h + 1.0e-3)
    wp.launch(
        _GPU["k_blockmask"],
        dim=n,
        inputs=[scene.mesh.id, wp.vec3(*[float(x) for x in origin]), float(h), int(be), int(dims_e[1]), int(dims_e[2]), thresh, out],
        device=dev,
    )
    lin = np.flatnonzero(out.numpy())
    bx, rem = np.divmod(lin, int(dims_e[1]) * int(dims_e[2]))
    by, bz = np.divmod(rem, int(dims_e[2]))
    return np.stack([bx, by, bz], axis=1)


def _gpu_field(scene, eval_blocks, be, origin, h, off, shape, sealed):
    """Distance-minus-offset field for the evaluated sub-blocks, computed and scattered entirely on the GPU."""
    wp, dev = _GPU["wp"], _GPU["dev"]
    far = 4.0 * h
    n_nodes = int(shape[0]) * int(shape[1]) * int(shape[2])
    field = wp.full(n_nodes, far, dtype=wp.float32, device=dev)
    blocks = wp.array(np.ascontiguousarray(eval_blocks, dtype=np.int32), dtype=wp.vec3i, device=dev)
    if sealed is not None:
        so, sc_, cls = sealed
        cls_d = wp.array(np.ascontiguousarray(cls, dtype=np.uint8), dtype=wp.uint8, device=dev)
        use_cls, c_origin, c_h = 1, wp.vec3(*[float(x) for x in so]), float(sc_)
    else:
        cls_d = wp.zeros((1, 1, 1), dtype=wp.uint8, device=dev)
        use_cls, c_origin, c_h = 0, wp.vec3(0.0, 0.0, 0.0), 1.0
    wp.launch(
        _GPU["k_field"],
        dim=len(eval_blocks) * be**3,
        inputs=[scene.mesh.id, blocks, wp.vec3(*[float(x) for x in origin]), float(h), float(off), float(far),
                int(be), int(shape[1]), int(shape[2]), field, cls_d, use_cls, c_origin, c_h],
        device=dev,
    )
    out = field.numpy().reshape(tuple(int(s) for s in shape))
    del field
    return out


def _closest_point_scene(mesh):
    if _gpu_ok():
        try:
            return _GpuScene(mesh)
        except Exception:
            pass
    import open3d as o3d

    legacy = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy))
    return scene


def _closest(scene, points):
    if isinstance(scene, _GpuScene):
        return scene.closest(points)
    import open3d as o3d

    ans = scene.compute_closest_points(o3d.core.Tensor(np.asarray(points, dtype=np.float32)))
    return ans["points"].numpy().astype(np.float64), ans["primitive_ids"].numpy()


def _dist_only(scene, points, maxd=None):
    if isinstance(scene, _GpuScene):
        return scene.distance(points, 1.0e3 if maxd is None else maxd)
    import open3d as o3d

    d = scene.compute_distance(o3d.core.Tensor(np.asarray(points, dtype=np.float32))).numpy().astype(np.float64)
    return d if maxd is None else np.minimum(d, maxd)


def _block_ids(points, origin, block_len, dims):
    """Unique linear ids of the blocks that contain ``points``."""
    b = np.floor((points - origin) / block_len).astype(np.int64)
    b = np.clip(b, 0, np.asarray(dims) - 1)
    return np.unique((b[:, 0] * dims[1] + b[:, 1]) * dims[2] + b[:, 2])


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
    _wl = [tic]

    def wtick(label):
        if verbose:
            now = time.perf_counter()
            print(f"{chr(9)}    wrap.{label}: {now - _wl[0]:.1f} s")
            _wl[0] = now

    h, off = float(resolution), float(offset)
    if h <= 0 or off <= 0:
        raise ValueError("wrap resolution and offset must be > 0")
    if off < 0.5 * h:
        print(f"\tWARNING wrap: offset {off * 1000:.2f} mm < 0.5*resolution; thin sheets may not close")

    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    scene = _closest_point_scene(mesh)
    wtick("BVH build")

    sealed = _sealed_classes(v, f, float(gap_closure), verbose) if gap_closure and gap_closure > 0 else None
    t_seal = time.perf_counter()

    # Only nodes near the soup are evaluated, in sub-blocks of Be^3 nodes.
    Be, Bm = 4, 16  # evaluation sub-block / marching block, in nodes
    pad = off + 3.0 * h
    origin = v.min(axis=0) - pad
    extent = v.max(axis=0) + pad - origin
    dims_m = np.ceil(extent / (h * Bm)).astype(np.int64) + 1  # marching blocks per axis
    dims_e = dims_m * (Bm // Be)
    shape = tuple(int(d) * Bm + 1 for d in dims_m)
    n_nodes = shape[0] * shape[1] * shape[2]
    if n_nodes > 1.5e9:
        raise MemoryError(f"wrap grid would be {shape} ({n_nodes / 1e6:.0f}M nodes); use a larger wrap resolution")

    eval_blocks = None
    if isinstance(scene, _GpuScene):
        try:
            eval_blocks = _gpu_block_list(scene, origin, h, Be, dims_e, off)
            wtick("block selection (GPU)")
        except Exception as e:
            if verbose:
                print(f"{chr(9)}GPU block selection failed ({e}); using the CPU path")
    if eval_blocks is None:
        # CPU: sub-blocks touched by samples of the soup, +/-1. Vertices, plus dense samples on the long faces.
        tri = v[f]
        longest = np.linalg.norm(tri[:, [1, 2, 0]] - tri, axis=2).max(axis=1)
        del tri
        sv = v
        if (longest > 2.0 * h).any():
            extra, _ = trimesh.remesh.subdivide_to_size(v, f[longest > 2.0 * h], max_edge=2.0 * h, max_iter=14)
            sv = np.vstack([v, extra])
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
        wtick("block selection (CPU)")

    field = None
    if isinstance(scene, _GpuScene):
        try:
            field = _gpu_field(scene, eval_blocks, Be, origin, h, off, shape, sealed)
        except Exception as e:  # e.g. out of GPU memory: fall back to the CPU path
            if verbose:
                print(f"{chr(9)}GPU distance field failed ({e}); using the CPU path")
    if field is None:
        if isinstance(scene, _GpuScene):
            _GPU["ok"] = False
            scene = _closest_point_scene(mesh)      # open3d scene from here on
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
    wtick("field eval (incl. GPU download)")
    mb = eval_blocks // (Bm // Be)                       # marching blocks that hold any evaluated node
    mlin = np.unique((mb[:, 0] * dims_m[1] + mb[:, 1]) * dims_m[2] + mb[:, 2])
    mx, mrem = np.divmod(mlin, dims_m[1] * dims_m[2])
    my, mz = np.divmod(mrem, dims_m[2])
    blocks = np.stack([mx, my, mz], axis=1)
    wtick("marching-block set")
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
    wtick("contour blocks")
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
    wrapped.update_faces(_nondegenerate_mask(wrapped))
    wrapped.remove_unreferenced_vertices()

    wtick("weld + assemble")
    wrapped = _keep_exterior_shells(wrapped)
    if verbose:
        print(
            f"\tSurface wrap: {len(f):,} tris -> {len(wrapped.faces):,} tris "
            f"(h {h * 1000:.1f} mm, offset {off * 1000:.1f} mm, {len(blocks):,} blocks, "
            f"seal {t_seal - tic:0.1f} s, field {t_field - t_seal:0.1f} s ({'GPU' if _GPU['ok'] else 'CPU'}), "
            f"mesh+weld {t_mc - t_field:0.1f} s, total {time.perf_counter() - tic:0.1f} s)"
        )
    return wrapped


def _feature_chains(src, angle_deg, min_len):
    """
    Sharp-edge polylines of the source CAD: mesh edges whose two faces differ by more than
    ``angle_deg``, chained through valence-2 vertices. Chains shorter than ``min_len`` are dropped
    (tessellation noise and details far below the export resolution).
    Returns a list of (n,3) ordered point arrays.
    """
    ang = src.face_adjacency_angles
    e = np.asarray(src.face_adjacency_edges)[ang > np.radians(angle_deg)]
    if len(e) == 0:
        return []
    e = np.unique(np.sort(e, axis=1), axis=0)
    nbr = {}
    for a, b in e.tolist():
        nbr.setdefault(a, []).append(b)
        nbr.setdefault(b, []).append(a)
    seen = set()
    chains = []

    def walk(start, nxt):
        path = [start, nxt]
        seen.add((min(start, nxt), max(start, nxt)))
        prev, cur = start, nxt
        while len(nbr[cur]) == 2:
            n2 = nbr[cur][0] if nbr[cur][0] != prev else nbr[cur][1]
            key = (min(cur, n2), max(cur, n2))
            if key in seen:
                break
            seen.add(key)
            path.append(n2)
            prev, cur = cur, n2
            if cur == start:
                break
        return path

    for v, ns in nbr.items():          # start at chain ends and junctions first
        if len(ns) != 2:
            for n in ns:
                if (min(v, n), max(v, n)) not in seen:
                    chains.append(walk(v, n))
    for v, ns in nbr.items():          # then closed loops
        for n in ns:
            if (min(v, n), max(v, n)) not in seen:
                chains.append(walk(v, n))
    V = np.asarray(src.vertices)
    out = []
    for c in chains:
        P = V[np.asarray(c)]
        if np.linalg.norm(np.diff(P, axis=0), axis=1).sum() >= min_len:
            out.append(P)
    return out


def _resample_polyline(P, step):
    """Points every ``step`` of arc length along polyline P (ends included)."""
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(int(np.ceil(s[-1] / step)), 1)
    t = np.linspace(0.0, s[-1], n + 1)
    return np.stack([np.interp(t, s, P[:, k]) for k in range(3)], axis=1)


def _smooth_polyline(P, edge, corner_deg=40.0):
    """
    Remove tessellation noise from a feature line while keeping its real corners.

    Resample every edge/4, split where the line turns by more than ``corner_deg`` over a
    +/-edge/2 window (a genuine corner), and moving-average each piece over ~1.25*edge with the
    piece ends held fixed. Feature lines that follow the CAD triangulation are jagged at the mm
    scale; snapping the mesh onto a jagged line makes the mesh jagged, so it is smoothed first.
    """
    ds = edge / 4.0
    Q = _resample_polyline(P, ds)
    n = len(Q)
    if n < 7:
        return Q
    k = 2                                   # +/- edge/2 window, in samples
    a = Q[k:] - Q[:-k]
    turn = np.zeros(n)
    u, w = a[:-k], a[k:]
    cosv = np.einsum("ij,ij->i", u, w) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1), 1e-20)
    turn[k : k + len(cosv)] = np.degrees(np.arccos(np.clip(cosv, -1, 1)))
    corners = [0] + [i for i in range(1, n - 1) if turn[i] > corner_deg and turn[i] >= turn[max(i - 2, 0) : i + 3].max()] + [n - 1]
    out = Q.copy()
    for a_, b_ in zip(corners[:-1], corners[1:]):
        if b_ - a_ < 5:
            continue
        seg = Q[a_ : b_ + 1]
        half = 2
        pad = np.vstack([np.repeat(seg[:1], half, axis=0), seg, np.repeat(seg[-1:], half, axis=0)])
        sm = np.stack([np.convolve(pad[:, c], np.ones(2 * half + 1) / (2 * half + 1), mode="valid") for c in range(3)], axis=1)
        sm[0], sm[-1] = seg[0], seg[-1]
        out[a_ : b_ + 1] = sm
    return out


def recover_features(mesh, src, edge, angle_deg=35.0, verbose=True, smooth=True,
                     relax_iters=3, project_to=None, offset=0.0):
    """
    Put the mesh's edges back on the CAD's sharp feature lines.

    1. Find the CAD's feature polylines (dihedral > ``angle_deg``).
    2. Every ``edge`` along a polyline, move the single nearest mesh vertex (within 0.75*edge)
       onto the polyline, so one row of vertices lies exactly on the crease.
    3. Consecutive snapped vertices are joined by real mesh edges via constrained edge flips
       (Sloan): without that, triangles still cut across the crease and the edge zig-zags.
    Only flips that keep the surface valid are made (convex quad, no normal reversal).
    4. Snapping leaves the triangles beside the line irregular, so the vertices within two edges of
       the snapped row are relaxed (``relax_iters`` Laplacian steps) and put back on ``project_to``
       (the surface the mesh lies on; ``offset`` is how far it stands off that surface).
    """
    from collections import deque
    from scipy.spatial import cKDTree

    tic = time.perf_counter()
    _fl = [tic]

    def ftick(label):
        if verbose:
            now = time.perf_counter()
            print(chr(9) + f"    features.{label}: {now - _fl[0]:.1f} s")
            _fl[0] = now

    # Moving a vertex only keeps the surface closed if every face that uses it shares it.
    mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.faces), process=False)
    mesh.merge_vertices()
    ftick("weld input")
    chains = _feature_chains(src, angle_deg, min_len=3.0 * edge)
    if smooth:
        chains = [_smooth_polyline(P, edge) for P in chains]
        ftick("feature lines + smoothing")
    if not chains:
        if verbose:
            print("\tFeature recovery: no feature lines found")
        return mesh
    V = np.array(mesh.vertices, dtype=np.float64)
    F = np.array(mesh.faces, dtype=np.int64)
    R = 0.75 * edge

    # 1. stations along every chain, nearest vertex each, unique assignment
    stations, st_chain, st_idx = [], [], []
    dense, dense_owner = [], []
    for ci, P in enumerate(chains):
        S = _resample_polyline(P, edge)
        stations.append(S)
        st_chain.extend([ci] * len(S))
        st_idx.extend(range(len(S)))
        D = _resample_polyline(P, 0.1 * edge)
        dense.append(D)
        dense_owner.extend([ci] * len(D))
    stations = np.vstack(stations)
    st_chain, st_idx = np.asarray(st_chain), np.asarray(st_idx)
    dense = np.vstack(dense)
    dtree = cKDTree(dense)

    vtree = cKDTree(V)
    ftick("vertex tree")
    d, vi = vtree.query(stations, distance_upper_bound=R)
    okst = np.isfinite(d)
    best = {}
    for s in np.flatnonzero(okst):
        v = int(vi[s])
        if v not in best or d[s] < best[v][0]:
            best[v] = (d[s], int(s))
    snapped = {}  # vertex -> station id
    for v, (_, s) in best.items():
        snapped[v] = s
    if not snapped:
        return mesh
    sv = np.fromiter(snapped.keys(), dtype=np.int64)
    # move onto the nearest point of the feature polyline
    _, di = dtree.query(V[sv])
    V[sv] = dense[di]

    # 2. constraints: snapped vertices on adjacent stations of the same chain
    by_station = {}
    for v, s in snapped.items():
        by_station[s] = v
    cons = []
    for s, v in by_station.items():
        s2 = s + 1
        if s2 in by_station and st_chain[s2] == st_chain[s] and st_idx[s2] == st_idx[s] + 1:
            if v != by_station[s2]:
                cons.append((v, by_station[s2]))

    # 3. constrained edge recovery by flipping
    # Plain-Python floats and cached tuples: this loop runs per constraint on 3-vectors, where numpy's
    # per-call overhead costs far more than the arithmetic (about 8x faster than the numpy version).
    from math import sqrt

    order = np.argsort(F.ravel(), kind="stable")
    counts = np.bincount(F.ravel(), minlength=len(V))
    starts = np.concatenate([[0], np.cumsum(counts)])
    vf = {}
    pc = {}   # vertex id -> (x, y, z); positions do not change during this stage
    fc = {}   # face id  -> [a, b, c]; mirrors F for the faces touched so far

    def vfaces(v):
        s = vf.get(v)
        if s is None:
            s = set((order[starts[v] : starts[v + 1]] // 3).tolist())
            vf[v] = s
        return s

    def edge_faces(u, v):
        return vfaces(u) & vfaces(v)

    def P(i):
        p = pc.get(i)
        if p is None:
            p = tuple(V[i].tolist())
            pc[i] = p
        return p

    def FT(f):
        t = fc.get(f)
        if t is None:
            t = F[f].tolist()
            fc[f] = t
        return t

    def sub(a, b):
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    def cross(a, b):
        return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])

    def fnormal(f):
        t = FT(f)
        a, b, c = P(t[0]), P(t[1]), P(t[2])
        n = cross(sub(b, a), sub(c, a))
        ln = max(sqrt(dot(n, n)), 1e-30)
        return (n[0] / ln, n[1] / ln, n[2] / ln)

    def directed(f, u, v):
        """third vertex if face f contains directed edge u->v, else None"""
        t = FT(f)
        for i in range(3):
            if t[i] == u and t[(i + 1) % 3] == v:
                return t[(i + 2) % 3]
        return None

    recovered = flips = failed = 0
    ftick("stations + constraints")
    for a, b in cons:
        if edge_faces(a, b):
            recovered += 1
            continue
        # local 2-D frame along a->b
        pa = P(a)
        tv = sub(P(b), pa)
        L = sqrt(dot(tv, tv))
        if L < 1e-12:
            continue
        tv = (tv[0] / L, tv[1] / L, tv[2] / L)
        nx = ny = nz = 0.0
        for f in list(vfaces(a)) + list(vfaces(b)):
            fn = fnormal(f)
            nx += fn[0]
            ny += fn[1]
            nz += fn[2]
        dn = nx * tv[0] + ny * tv[1] + nz * tv[2]
        nn = (nx - tv[0] * dn, ny - tv[1] * dn, nz - tv[2] * dn)
        ln = sqrt(dot(nn, nn))
        if ln < 1e-9:
            failed += 1
            continue
        nn = (nn[0] / ln, nn[1] / ln, nn[2] / ln)
        w = cross(nn, tv)

        def xy(v, pa=pa, tv=tv, w=w):
            r = sub(P(v), pa)
            return dot(r, tv), dot(r, w)

        def crosses(c, d, xy=xy, L=L):
            xc, yc = xy(c)
            xd, yd = xy(d)
            if yc * yd >= 0 or abs(yc) < 1e-9 or abs(yd) < 1e-9:
                return False
            x = xc + (xd - xc) * (0 - yc) / (yd - yc)
            return 1e-9 < x < L - 1e-9

        # first crossed edge: opposite edge of the face at a that the segment passes through
        q = deque()
        prev_face, cur = None, None
        for f in vfaces(a):
            tri = [x for x in FT(f) if x != a]
            if len(tri) == 2 and crosses(tri[0], tri[1]):
                cur, prev_face = tuple(tri), f
                break
        if cur is None:
            failed += 1
            continue
        ok = True
        for _ in range(64):
            q.append(cur)
            c, d = cur
            fs = [f for f in edge_faces(c, d) if f != prev_face]
            if len(fs) != 1:
                ok = False
                break
            f2 = fs[0]
            e3 = [x for x in FT(f2) if x != c and x != d]
            if len(e3) != 1:
                ok = False
                break
            e3 = e3[0]
            if e3 == b:
                break
            ye = xy(e3)[1]
            if abs(ye) < 1e-9:
                ok = False
                break
            cur = (e3, d) if (ye > 0) == (xy(c)[1] > 0) else (c, e3)
            prev_face = f2
        else:
            ok = False
        if not ok:
            failed += 1
            continue

        it = 0
        while q and it < 400:
            it += 1
            c, d = q.popleft()
            fs = list(edge_faces(c, d))
            if len(fs) != 2:
                continue
            f1, f2 = fs
            p = directed(f1, c, d)
            if p is not None:
                u, vv, ff1, ff2 = c, d, f1, f2
            else:
                p = directed(f2, c, d)
                if p is None:
                    continue
                u, vv, ff1, ff2 = c, d, f2, f1
            qv = directed(ff2, vv, u)
            if qv is None:
                continue
            # convex quad in the local frame?
            pu, pv_, pp, pq = xy(u), xy(vv), xy(p), xy(qv)

            def side(o, m, z):
                return (m[0] - o[0]) * (z[1] - o[1]) - (m[1] - o[1]) * (z[0] - o[0])

            if not (side(pp, pq, pu) * side(pp, pq, pv_) < 0 and side(pu, pv_, pp) * side(pu, pv_, pq) < 0):
                q.append((c, d))
                continue
            # keep the surface valid: new faces must not turn against the old ones
            n_old1, n_old2 = fnormal(ff1), fnormal(ff2)
            A, B = (u, qv, p), (qv, vv, p)
            nA = cross(sub(P(A[1]), P(A[0])), sub(P(A[2]), P(A[0])))
            nB = cross(sub(P(B[1]), P(B[0])), sub(P(B[2]), P(B[0])))
            if dot(nA, n_old1) <= 0 or dot(nB, n_old2) <= 0 or dot(nA, nB) <= 0:
                q.append((c, d))
                continue
            F[ff1] = A
            F[ff2] = B
            fc[ff1] = list(A)
            fc[ff2] = list(B)
            vfaces(vv).discard(ff1)
            vfaces(qv).add(ff1)
            vfaces(u).discard(ff2)
            vfaces(p).add(ff2)
            flips += 1
            if crosses(p, qv):
                q.append((p, qv))
        if edge_faces(a, b):
            recovered += 1
        else:
            failed += 1

    ftick("constrained flips")
    if relax_iters and project_to is not None:
        from scipy.sparse import coo_matrix

        n = len(V)
        rr = np.concatenate([F[:, 0], F[:, 1], F[:, 2], F[:, 1], F[:, 2], F[:, 0]])
        cc = np.concatenate([F[:, 1], F[:, 2], F[:, 0], F[:, 0], F[:, 1], F[:, 2]])
        A = coo_matrix((np.ones(len(rr)), (rr, cc)), shape=(n, n)).tocsr()
        A.data[:] = 1.0
        deg = np.maximum(np.asarray(A.sum(axis=1)).ravel(), 1)
        fixed = np.zeros(n)
        fixed[sv] = 1.0
        reach = fixed.copy()
        for _ in range(2):
            reach = np.minimum(reach + A @ reach, 1.0)
        region = (reach > 0) & (fixed == 0)
        pscene = _closest_point_scene(project_to)
        for _ in range(int(relax_iters)):
            avg = (A @ V) / deg[:, None]
            V[region] = 0.5 * V[region] + 0.5 * avg[region]
            p_, _ = _closest(pscene, V[region])
            dv = p_ - V[region]
            dd = np.linalg.norm(dv, axis=1)
            ok = dd > 1e-12
            nr = V[region].copy()
            nr[ok] = p_[ok] - dv[ok] / dd[ok, None] * np.minimum(float(offset), dd[ok])[:, None]
            V[region] = nr

    out = trimesh.Trimesh(V, F, process=False)
    ftick("relax")
    if verbose:
        tot = sum(float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()) for P in chains)
        print(
            f"\tFeature recovery: {len(chains):,} feature lines ({tot:.1f} m), {len(snapped):,} vertices moved onto them, "
            f"{recovered:,}/{len(cons):,} feature edges now real mesh edges ({flips:,} flips, {failed:,} not recovered) "
            f"in {time.perf_counter() - tic:0.1f} s"
        )
    return out


def _nondegenerate_mask(mesh):
    """Faces with area > 1e-13 m^2 (vectorised; trimesh's nondegenerate_faces is ~20x slower on millions of faces)."""
    v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    c = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    return np.einsum("ij,ij->i", c, c) > (2e-13) ** 2


def _relax_all(m, scene, iters, lam=0.5):
    """
    Global Laplacian relaxation with re-projection onto ``scene``: evens out triangle sizes/shapes after the
    resample without letting the surface drift (every step lands back on the source surface).
    """
    from scipy.sparse import coo_matrix

    V = np.array(m.vertices, dtype=np.float64)
    F = np.asarray(m.faces)
    n = len(V)
    rr = np.concatenate([F[:, 0], F[:, 1], F[:, 2], F[:, 1], F[:, 2], F[:, 0]])
    cc = np.concatenate([F[:, 1], F[:, 2], F[:, 0], F[:, 0], F[:, 1], F[:, 2]])
    A = coo_matrix((np.ones(len(rr)), (rr, cc)), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    deg = np.maximum(np.asarray(A.sum(axis=1)).ravel(), 1)
    for _ in range(int(iters)):
        V = (1.0 - lam) * V + lam * (A @ V) / deg[:, None]
        V, _ = _closest(scene, V)
    return trimesh.Trimesh(V, F, process=False)


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
                   wrap_resolution=None, wrap_offset=None, gap_closure=0.0, snap_offset=None, feature_angle=None, relax_iters=0):
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
    _last = [tic]
    if verbose:
        if _gpu_ok():
            print(f"{chr(9)}Distance queries: GPU (Warp {getattr(_GPU['wp'], '__version__', '?')}, {_GPU['dev']})")
        else:
            print(f"{chr(9)}Distance queries: CPU (open3d) [{_GPU.get('reason') or 'GPU unavailable'}]")

    def tick(label):
        if verbose:
            now = time.perf_counter()
            print(f"\t  [{label}: {now - _last[0]:.1f} s]")
            _last[0] = now

    if target_edge is None and not wrap_resolution:
        raise ValueError("need target_edge and/or wrap_resolution")
    if target_edge is not None:
        target_edge = float(target_edge)
        if target_edge <= 0:
            raise ValueError("target_edge must be > 0")

    src = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.faces), process=False)
    src.merge_vertices()  # weld split patches so connectivity is real
    src.update_faces(_nondegenerate_mask(src))
    src.remove_unreferenced_vertices()
    if len(src.faces) == 0:
        raise ValueError("remesh_surface: input mesh has no valid faces")
    tick("weld/clean input")

    orig, wrap_off = None, None
    if wrap_resolution:
        # Closed, oriented shell just outside the soup; everything below works on that instead.
        orig = src
        wrap_off = wrap_offset if wrap_offset else 0.75 * float(wrap_resolution)
        src = wrap_surface(src, wrap_resolution, wrap_off, gap_closure=gap_closure, verbose=verbose)
        tick("wrap (total)")
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

    tick("split shells")
    # Biggest first so a few large shells don't leave the pool idle at the end.
    big = sorted((i for i, j in enumerate(jobs) if j[2] >= 8 and len(j[1]) >= 4), key=lambda i: -len(jobs[i][1]))
    results = {}
    total_faces = sum(len(j[1]) for j in jobs)
    dominant = bool(big) and len(jobs[big[0]][1]) > 0.5 * total_faces   # e.g. the single closed shell a wrap produces
    if workers > 1 and len(big) > 1 and not dominant:            # a pool only helps with many comparable shells
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        ctx = mp.get_context("spawn" if mp.get_start_method(allow_none=True) == "spawn" or _is_windows() or _GPU["ok"] else "fork")
        with ProcessPoolExecutor(min(workers, len(big)), mp_context=ctx) as ex:
            for i, r in zip(big, ex.map(_cluster_job, [jobs[i] for i in big], chunksize=1)):
                results[i] = r
    else:
        for i in big:
            results[i] = _cluster_job(jobs[i])
    tick("cluster/resample (pyacvd)")

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
        tick("project onto source (incl. BVH build)")

    out = trimesh.Trimesh(verts, faces, process=False)
    out.update_faces(_nondegenerate_mask(out))
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
            tick("orientation fix")

    if relax_iters and project:
        out = _relax_all(out, scene, relax_iters)
        tick(f"global relax x{relax_iters}")
    tick("assemble/clean")
    if snap_offset is not None and orig is not None:
        out = _snap_to_source(out, orig, 1.5 * wrap_off, snap_offset, verbose)
        tick("snap to source (incl. BVH build)")

    if feature_angle is not None:
        if orig is not None:
            # the feature lines live on the CAD, so the mesh must already stand close to it
            if snap_offset is None:
                out = _snap_to_source(out, orig, 1.5 * wrap_off, 0.001, verbose)
                snap_offset = 0.001
                if verbose:
                    print("\tFeature recovery needs the mesh near the CAD: using snapOffset 1 mm")
            out = recover_features(out, orig, edge, feature_angle, verbose, project_to=orig, offset=snap_offset)
        else:
            out = recover_features(out, src, edge, feature_angle, verbose, project_to=src, offset=0.0)
        tick("feature recovery")

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
                            wrap_resolution=None, wrap_offset=None, gap_closure=0.0, snap_offset=None, feature_angle=None, relax_iters=0, use_gpu=True):
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
        if feature_angle is not None:
            cmd += ["--feature-angle", repr(float(feature_angle))]
        if relax_iters:
            cmd += ["--relax-iters", str(int(relax_iters))]
        if not use_gpu:
            cmd += ["--no-gpu"]
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
    ap.add_argument("--feature-angle", type=float, default=None,
                    help="recover the CAD's sharp edges (dihedral above this many degrees, e.g. 35) in the output mesh")
    ap.add_argument("--relax-iters", type=int, default=0, help="global Laplacian relaxation steps after the resample")
    ap.add_argument("--no-gpu", action="store_true", help="force the CPU (open3d) distance queries")
    ap.add_argument("--gap-closure", type=float, default=0.0, help="seal leaks up to ~2x this when dropping interiors (m)")
    ap.add_argument("--scale", type=float, default=1.0, help="input scale to meters (0.01 for Alias OBJ in cm)")
    ap.add_argument("--max-faces", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    if a.no_gpu:
        import os

        os.environ["XLB_SURFACE_GPU"] = "0"

    if a.input.endswith(".npz"):
        with np.load(a.input) as d:
            m = trimesh.Trimesh(d["v"], d["f"], process=False)
    else:
        m = trimesh.load(a.input, force="mesh", process=False)
    if a.scale != 1.0:
        m.apply_scale(a.scale)
    r = remesh_surface(m, a.edge, max_faces=a.max_faces, workers=a.workers,
                       wrap_resolution=a.wrap_res, wrap_offset=a.wrap_offset, gap_closure=a.gap_closure,
                       snap_offset=a.snap_offset, feature_angle=a.feature_angle, relax_iters=a.relax_iters)
    if a.output.endswith(".npz"):
        np.savez(a.output, v=np.asarray(r.vertices), f=np.asarray(r.faces))
    else:
        r.export(a.output)
    print(f"wrote {a.output}")


if __name__ == "__main__":
    _main()
