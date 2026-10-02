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

    @wp.kernel
    def k_closest_val(
        mesh: wp.uint64,
        pts: wp.array(dtype=wp.vec3),
        out_p: wp.array(dtype=wp.vec3),
        out_v: wp.array(dtype=wp.float32),
        maxd: float,
    ):
        # closest point on the mesh and a per-vertex value (stored in the mesh's "velocity" x) interpolated there
        i = wp.tid()
        q = wp.mesh_query_point_no_sign(mesh, pts[i], maxd)
        if q.result:
            out_p[i] = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
            vel = wp.mesh_eval_velocity(mesh, q.face, q.u, q.v)
            out_v[i] = vel[0]
        else:
            out_p[i] = pts[i]
            out_v[i] = 1.0e9

    @wp.kernel
    def k_curv(
        indptr: wp.array(dtype=wp.int32),
        indices: wp.array(dtype=wp.int32),
        verts: wp.array(dtype=wp.vec3),
        nrm: wp.array(dtype=wp.vec3),
        dmin: float,
        out: wp.array(dtype=wp.float32),
    ):
        # largest normal turn per unit length towards any neighbour (1/radius of the tightest local curvature);
        # the length is floored at dmin so the near-zero-length edges a wrap can contain do not read as huge curvature
        i = wp.tid()
        best = float(0.0)
        for e in range(indptr[i], indptr[i + 1]):
            j = indices[e]
            d = wp.max(wp.length(verts[j] - verts[i]), dmin)
            c = wp.clamp(wp.dot(nrm[i], nrm[j]), -1.0, 1.0)
            best = wp.max(best, wp.acos(c) / d)
        out[i] = best

    @wp.kernel
    def k_dilate(
        indptr: wp.array(dtype=wp.int32),
        indices: wp.array(dtype=wp.int32),
        vin: wp.array(dtype=wp.float32),
        vout: wp.array(dtype=wp.float32),
    ):
        i = wp.tid()
        best = float(vin[i])
        for e in range(indptr[i], indptr[i + 1]):
            best = wp.max(best, vin[indices[e]])
        vout[i] = best

    @wp.kernel
    def k_tosize(
        kappa: wp.array(dtype=wp.float32), tol: float, smin: float, smax: float, out: wp.array(dtype=wp.float32)
    ):
        # chord-error sizing: an edge of length L on radius R deviates L^2/(8R) from the surface
        i = wp.tid()
        out[i] = wp.clamp(wp.sqrt(8.0 * tol / wp.max(kappa[i], 1.0e-9)), smin, smax)

    @wp.kernel
    def k_grade(
        indptr: wp.array(dtype=wp.int32),
        indices: wp.array(dtype=wp.int32),
        verts: wp.array(dtype=wp.vec3),
        sin: wp.array(dtype=wp.float32),
        sout: wp.array(dtype=wp.float32),
        g: float,
    ):
        # size gradation: s_i <= s_j + g * |p_i - p_j|
        i = wp.tid()
        best = float(sin[i])
        for e in range(indptr[i], indptr[i + 1]):
            j = indices[e]
            best = wp.min(best, sin[j] + g * wp.length(verts[j] - verts[i]))
        sout[i] = best

    @wp.kernel
    def k_spring(
        indptr: wp.array(dtype=wp.int32),
        indices: wp.array(dtype=wp.int32),
        verts: wp.array(dtype=wp.vec3),
        size: wp.array(dtype=wp.float32),
        lam: float,
        out: wp.array(dtype=wp.vec3),
    ):
        # every edge is a spring whose rest length is the local target size; move to the mean spring force
        i = wp.tid()
        acc = wp.vec3()
        cnt = int(0)
        for e in range(indptr[i], indptr[i + 1]):
            j = indices[e]
            d = verts[j] - verts[i]
            L = wp.max(wp.length(d), 1.0e-9)
            acc = acc + d * ((L - 0.5 * (size[i] + size[j])) / L)
            cnt = cnt + 1
        if cnt > 0:
            out[i] = verts[i] + acc * (lam / float(cnt))
        else:
            out[i] = verts[i]

    st["k_dist"], st["k_closest"], st["k_field"], st["k_blockmask"] = k_dist, k_closest, k_field, k_blockmask
    st["k_closest_val"], st["k_curv"], st["k_dilate"] = k_closest_val, k_curv, k_dilate
    st["k_tosize"], st["k_grade"], st["k_spring"] = k_tosize, k_grade, k_spring


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
    # adaptive-remesh kernels: value interpolation (x coordinate is linear, so the interpolant must equal x)
    sc.set_values(m.vertices[:, 0])
    pv_, vv_ = sc.closest_val(np.array([[0.25, 0.25, 0.5], [0.8, 0.1, -0.3]]))
    assert abs(vv_[0] - 0.25) < 1e-4 and abs(vv_[1] - 0.8) < 1e-4, f"value-interpolation kernel returned {vv_}"
    ip, ix = _csr_adjacency(m.faces, 4)
    g = _gpu_grade(np.asarray(m.vertices), ip, ix, np.array([1.0, 9.0, 9.0, 9.0]), 0.5, 3)
    assert abs(g[1] - 1.5) < 1e-4 and abs(g[0] - 1.0) < 1e-4, f"gradation kernel returned {g}"
    sp = _gpu_spring(np.asarray(m.vertices), ip, ix, np.full(4, 1.0), 0.0)
    assert np.allclose(sp, m.vertices, atol=1e-5), "spring kernel moved a vertex with lam=0"


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

    def set_values(self, values):
        """Attach a per-vertex scalar so :meth:`closest_val` can interpolate it (stored as the mesh velocity x)."""
        wp, dev = _GPU["wp"], _GPU["dev"]
        vel = np.zeros((len(values), 3), dtype=np.float32)
        vel[:, 0] = values
        self._vel = wp.array(vel, dtype=wp.vec3, device=dev)
        self.mesh_v = wp.Mesh(points=self._verts, indices=self._idx, velocities=self._vel)

    def closest_val(self, points):
        """(closest point on the mesh, attached per-vertex scalar interpolated at that point)"""
        wp, dev = _GPU["wp"], _GPU["dev"]
        pts = np.ascontiguousarray(points, dtype=np.float32)
        op = np.empty((len(pts), 3), dtype=np.float32)
        ov = np.empty(len(pts), dtype=np.float32)
        for a in range(0, len(pts), self.CHUNK):
            c = pts[a : a + self.CHUNK]
            P = wp.array(c, dtype=wp.vec3, device=dev)
            OP = wp.empty(len(c), dtype=wp.vec3, device=dev)
            OV = wp.empty(len(c), dtype=wp.float32, device=dev)
            wp.launch(_GPU["k_closest_val"], dim=len(c), inputs=[self.mesh_v.id, P, OP, OV, 1.0e3], device=dev)
            op[a : a + len(c)] = OP.numpy()
            ov[a : a + len(c)] = OV.numpy()
        return op.astype(np.float64), ov.astype(np.float64)


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


def _split_marked(V, F, marked):
    """
    Split the edges whose keys (``min*n + max``, sorted and unique) are in ``marked`` at their midpoints.

    Each face is closed with a 1->2, 1->3 or 1->4 pattern depending on how many of its edges are split, so there
    are no hanging nodes and the winding is preserved. Returns (V2, F2); the midpoints are appended to V in the
    order of ``marked``.
    """
    n = len(V)
    F = np.asarray(F, dtype=np.int64)
    a, b, c = F[:, 0], F[:, 1], F[:, 2]
    lo = np.stack([np.minimum(a, b), np.minimum(b, c), np.minimum(c, a)], axis=1)
    hi = np.stack([np.maximum(a, b), np.maximum(b, c), np.maximum(c, a)], axis=1)
    key = lo * n + hi
    pos = np.minimum(np.searchsorted(marked, key), len(marked) - 1)
    mk = marked[pos] == key                                   # (F, 3): is this face edge split?
    mid = n + pos                                             # midpoint vertex id of each face edge
    V2 = np.vstack([V, 0.5 * (V[marked // n] + V[marked % n])])
    cnt = mk.sum(axis=1)

    out = [F[cnt == 0]]
    i3 = np.flatnonzero(cnt == 3)
    if len(i3):
        A, B, C = a[i3], b[i3], c[i3]
        m01, m12, m20 = mid[i3, 0], mid[i3, 1], mid[i3, 2]
        out += [np.stack([A, m01, m20], 1), np.stack([m01, B, m12], 1),
                np.stack([m20, m12, C], 1), np.stack([m01, m12, m20], 1)]
    i1 = np.flatnonzero(cnt == 1)
    if len(i1):
        k = np.argmax(mk[i1], axis=1)                          # which edge is split
        rows, ar = F[i1], np.arange(len(i1))
        p, q, r = rows[ar, k], rows[ar, (k + 1) % 3], rows[ar, (k + 2) % 3]
        m = mid[i1, k]
        out += [np.stack([p, m, r], 1), np.stack([m, q, r], 1)]
    i2 = np.flatnonzero(cnt == 2)
    if len(i2):
        u = np.argmin(mk[i2], axis=1)                          # the unsplit edge
        rows, ar = F[i2], np.arange(len(i2))
        A, B, C = rows[ar, (u + 1) % 3], rows[ar, (u + 2) % 3], rows[ar, u]   # (A,B) and (B,C) are split
        mab, mbc = mid[i2, (u + 1) % 3], mid[i2, (u + 2) % 3]
        d1 = np.linalg.norm(V2[A] - V2[mbc], axis=1)           # diagonal A-mbc
        d2 = np.linalg.norm(V2[mab] - V2[C], axis=1)           # diagonal mab-C
        s = d1 <= d2
        out += [np.stack([mab, B, mbc], 1),
                np.where(s[:, None], np.stack([A, mab, mbc], 1), np.stack([A, mab, C], 1)),
                np.where(s[:, None], np.stack([A, mbc, C], 1), np.stack([mab, mbc, C], 1))]
    return V2, np.vstack(out)


def _dbg_topology(label, mesh):
    """Opt-in (XLB_SURFACE_DEBUG=1): print boundary / non-manifold / winding-conflict / zero-area counts for a stage."""
    import os

    if os.environ.get("XLB_SURFACE_DEBUG", "0") != "1":
        return
    V, F = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    de = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = np.sort(de, axis=1)
    _, c = np.unique(e, axis=0, return_counts=True)
    k = de[:, 0].astype(np.int64) * (len(V) + 1) + de[:, 1]
    conf = int((np.unique(k, return_counts=True)[1] > 1).sum())
    a, b, cc = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    zero = int((0.5 * np.linalg.norm(np.cross(b - a, cc - a), axis=1) < 1e-12).sum())
    print(f"{chr(9)}  [topology after {label}: {len(F):,} faces, boundary {int((c == 1).sum())}, "
          f"non-manifold {int((c > 2).sum())}, winding conflicts {conf}, zero-area faces {zero}]")


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


# ---------------------------------------------------------------------------------------------------------------
# Curvature-adaptive refinement
#
# The resample above is uniform at the coarse edge length. Here the triangles are made smaller where the wrapped
# surface curves tightly, from a sizing field computed on the wrap alone (the CAD is not involved):
#   1. per-vertex curvature of the wrap -> chord-error edge length sqrt(8*tol*R), clamped to [min, max],
#      then graded so neighbouring sizes change by at most `grade` per unit distance;
#   2. edges longer than 4/3 of the local size are split (conforming 1->2/3/4), the new vertices go back onto the wrap;
#   3. Delaunay edge flips and size-aware spring smoothing (re-projected onto the wrap) restore the triangle shape.
# Distance/projection queries, the sizing field and the smoothing run on the GPU (Warp); the connectivity changes are
# vectorised numpy because they are irregular. Everything has a CPU fallback with the same result.
# ---------------------------------------------------------------------------------------------------------------


def _csr_adjacency(F, n):
    """Unique vertex neighbours of a triangle mesh in CSR form (int32 indptr, indices)."""
    from scipy.sparse import coo_matrix

    F = np.asarray(F, dtype=np.int32)
    r = np.concatenate([F[:, 0], F[:, 1], F[:, 2], F[:, 1], F[:, 2], F[:, 0]])
    c = np.concatenate([F[:, 1], F[:, 2], F[:, 0], F[:, 0], F[:, 1], F[:, 2]])
    A = coo_matrix((np.ones(len(r), dtype=np.int8), (r, c)), shape=(n, n)).tocsr()   # duplicates are summed away
    return A.indptr.astype(np.int32), A.indices.astype(np.int32)


def _seg_reduce(ufunc, vals, indptr, empty):
    """Per-row reduction of ``vals`` over CSR rows (``ufunc.reduceat`` that is safe for empty rows)."""
    n = len(indptr) - 1
    out = np.full((n,) + vals.shape[1:], empty, dtype=vals.dtype)
    ne = np.flatnonzero(np.diff(indptr) > 0)
    if len(ne):
        out[ne] = ufunc.reduceat(vals, indptr[ne].astype(np.int64), axis=0)
    return out


def _vertex_normals(V, F):
    """Area-weighted unit vertex normals and the area share of every vertex (a third of each adjacent face)."""
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    idx = F.ravel()
    out = np.empty((len(V), 3))
    for k in range(3):
        out[:, k] = np.bincount(idx, weights=np.repeat(fn[:, k], 3), minlength=len(V))
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-30)
    va = np.bincount(idx, weights=np.repeat(0.5 * np.linalg.norm(fn, axis=1) / 3.0, 3), minlength=len(V))
    return out, va


def _gpu_csr(indptr, indices):
    wp, dev = _GPU["wp"], _GPU["dev"]
    return (wp.array(np.ascontiguousarray(indptr, dtype=np.int32), dtype=wp.int32, device=dev),
            wp.array(np.ascontiguousarray(indices, dtype=np.int32), dtype=wp.int32, device=dev))


def _gpu_grade(V, indptr, indices, size, g, sweeps):
    wp, dev = _GPU["wp"], _GPU["dev"]
    ip, ix = _gpu_csr(indptr, indices)
    verts = wp.array(np.ascontiguousarray(V, dtype=np.float32), dtype=wp.vec3, device=dev)
    a = wp.array(np.ascontiguousarray(size, dtype=np.float32), dtype=wp.float32, device=dev)
    b = wp.empty_like(a)
    for _ in range(int(sweeps)):
        wp.launch(_GPU["k_grade"], dim=len(V), inputs=[ip, ix, verts, a, b, float(g)], device=dev)
        a, b = b, a
    return a.numpy().astype(np.float64)


def _gpu_spring(V, indptr, indices, size, lam):
    wp, dev = _GPU["wp"], _GPU["dev"]
    ip, ix = _gpu_csr(indptr, indices)
    verts = wp.array(np.ascontiguousarray(V, dtype=np.float32), dtype=wp.vec3, device=dev)
    sz = wp.array(np.ascontiguousarray(size, dtype=np.float32), dtype=wp.float32, device=dev)
    out = wp.empty(len(V), dtype=wp.vec3, device=dev)
    wp.launch(_GPU["k_spring"], dim=len(V), inputs=[ip, ix, verts, sz, float(lam), out], device=dev)
    return out.numpy().astype(np.float64)


class _WrapSizer:
    """
    Curvature of the wrap (computed once) and the target edge length derived from it for any
    (min, max, tolerance, gradation); the GPU keeps the curvature so re-sizing is a few kernel launches.
    """

    def __init__(self, V, F, verbose=True):
        self.V = np.asarray(V, dtype=np.float64)
        n = len(self.V)
        self.indptr, self.indices = _csr_adjacency(F, n)
        nrm, self.va = _vertex_normals(self.V, F)
        self.e_mean = float(np.sqrt(4.0 * self.va.sum() / len(F) / np.sqrt(3.0)))
        dmin = 0.5 * self.e_mean
        self.dev = None
        if _gpu_ok():
            try:
                wp, dev = _GPU["wp"], _GPU["dev"]
                ip, ix = _gpu_csr(self.indptr, self.indices)
                verts = wp.array(np.ascontiguousarray(self.V, dtype=np.float32), dtype=wp.vec3, device=dev)
                nr = wp.array(np.ascontiguousarray(nrm, dtype=np.float32), dtype=wp.vec3, device=dev)
                k = wp.empty(n, dtype=wp.float32, device=dev)
                wp.launch(_GPU["k_curv"], dim=n, inputs=[ip, ix, verts, nr, float(dmin), k], device=dev)
                k2 = wp.empty_like(k)
                for _ in range(2):                    # widen the curved zone by two rings so a thin crease is not missed
                    wp.launch(_GPU["k_dilate"], dim=n, inputs=[ip, ix, k, k2], device=dev)
                    k, k2 = k2, k
                self.dev = (ip, ix, verts, k)
            except Exception as e:
                self.dev = None
                if verbose:
                    print(f"{chr(9)}  sizing field: GPU failed ({type(e).__name__}: {str(e)[:80]}); using the CPU")
        if self.dev is None:
            self.rows = np.repeat(np.arange(n, dtype=np.int32), np.diff(self.indptr))
            self.d = np.linalg.norm(self.V[self.indices] - self.V[self.rows], axis=1)
            cosv = np.clip(np.einsum("ij,ij->i", nrm[self.rows], nrm[self.indices]), -1.0, 1.0)
            kap = _seg_reduce(np.maximum, np.arccos(cosv) / np.maximum(self.d, dmin), self.indptr, 0.0)
            for _ in range(2):
                kap = np.maximum(kap, _seg_reduce(np.maximum, kap[self.indices], self.indptr, 0.0))
            self.kap = kap

    def size(self, smin, smax, tol, grade):
        """Target edge length per wrap vertex: chord error <= tol, clamped to [smin, smax], graded."""
        sweeps = int(min(80, np.ceil((smax - smin) / (grade * self.e_mean)) + 2))
        if self.dev is not None:
            wp, dev = _GPU["wp"], _GPU["dev"]
            ip, ix, verts, k = self.dev
            a = wp.empty_like(k)
            wp.launch(_GPU["k_tosize"], dim=len(self.V), inputs=[k, float(tol), float(smin), float(smax), a], device=dev)
            b = wp.empty_like(a)
            for _ in range(sweeps):
                wp.launch(_GPU["k_grade"], dim=len(self.V), inputs=[ip, ix, verts, a, b, float(grade)], device=dev)
                a, b = b, a
            return a.numpy().astype(np.float64)
        sz = np.clip(np.sqrt(8.0 * tol / np.maximum(self.kap, 1e-9)), smin, smax)
        ge = grade * self.d
        for _ in range(sweeps):
            sz = np.minimum(sz, _seg_reduce(np.minimum, sz[self.indices] + ge, self.indptr, np.inf))
        return sz

    def estimate(self, size):
        """Triangle count if the whole surface were meshed at ``size`` (the real result is ~1.2x this)."""
        return float((self.va / (0.433 * size * size)).sum())


class _SizeField:
    """Sizing field defined on the wrap's vertices, evaluated (with the projection) at arbitrary points."""

    def __init__(self, scene, V, size):
        self.scene, self.gpu = scene, isinstance(scene, _GpuScene)
        if self.gpu:
            try:
                scene.set_values(size)
            except Exception:
                self.gpu = False
        if not self.gpu:
            from scipy.spatial import cKDTree

            self.tree, self.size = cKDTree(np.asarray(V)), np.asarray(size)

    def project(self, P):
        """(closest points on the wrap, target size there)"""
        if self.gpu:
            return self.scene.closest_val(P)
        p, _ = _closest(self.scene, P)
        _, i = self.tree.query(p, workers=-1)
        return p, self.size[i]


def _spring(V, indptr, indices, size, lam):
    """One size-aware smoothing step: each edge pulls/pushes its ends toward the local target length."""
    if _gpu_ok():
        try:
            return _gpu_spring(V, indptr, indices, size, lam)
        except Exception:
            pass
    deg = np.diff(indptr)
    rows = np.repeat(np.arange(len(V), dtype=np.int32), deg)
    d = V[indices] - V[rows]
    L = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
    f = d * ((L - 0.5 * (size[rows] + size[indices])) / L)[:, None]
    return V + _seg_reduce(np.add, f, indptr, 0.0) * (lam / np.maximum(deg, 1))[:, None]


def _split_long(V, F, s, ratio=4.0 / 3.0):
    """Split every edge longer than ``ratio`` x its local target size. Returns (V, F, s, number of edges split)."""
    n = len(V)
    Fi = np.asarray(F, dtype=np.int64)
    a, b = Fi.ravel(), Fi[:, [1, 2, 0]].ravel()
    mk = np.linalg.norm(V[a] - V[b], axis=1) > ratio * 0.5 * (s[a] + s[b])
    if not mk.any():
        return V, Fi, s, 0
    marked = np.unique((np.minimum(a, b) * n + np.maximum(a, b))[mk])
    V2, F2 = _split_marked(V, Fi, marked)
    return V2, F2, np.concatenate([s, 0.5 * (s[marked // n] + s[marked % n])]), len(marked)


def _flip_delaunay(V, F, seeds=None, min_cos=0.75, max_iter=4):
    """
    Edge flips toward the (cotangent) Delaunay criterion, many independent flips per round.

    Only the neighbourhood of ``seeds`` (face indices; None = the whole mesh) is examined in the first round, and
    only the neighbourhood of the faces the previous round changed afterwards, so the cost follows the amount of
    change and not the mesh size. A flip is refused if the new diagonal already exists, if it would leave a vertex
    with fewer than 3 neighbours, or if either new triangle turns more than ~40 degrees from the pair's mean normal
    (so creases are not cut across). Returns (F, total flips, indices of the faces that were changed).
    """
    n = len(V)
    F = np.array(F, dtype=np.int64)
    total = 0
    changed = []
    unit = lambda x: x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-30)

    def cot(u, w):
        return np.einsum("ij,ij->i", u, w) / np.maximum(np.linalg.norm(np.cross(u, w), axis=1), 1e-18)

    for _ in range(int(max_iter)):
        if seeds is None:
            S = np.arange(len(F))
            vm = np.ones(n, dtype=bool)
        else:
            if len(seeds) == 0:
                break
            vm = np.zeros(n, dtype=bool)                  # seed vertices: all faces around them are in the sub-mesh
            vm[F[seeds].ravel()] = True
            S = np.flatnonzero(vm[F].any(axis=1))
        sub = F[S]
        ha = sub.ravel()
        hb = sub[:, [1, 2, 0]].ravel()
        hc = sub[:, [2, 0, 1]].ravel()
        key = np.minimum(ha, hb) * n + np.maximum(ha, hb)
        order = np.argsort(key)
        ks = key[order]
        first = np.flatnonzero(ks[1:] == ks[:-1])
        bad = np.zeros(len(first), dtype=bool)
        i2 = first + 2
        ok2 = i2 < len(ks)
        bad[ok2] |= ks[i2[ok2]] == ks[first[ok2]]
        bad |= (first > 0) & (ks[np.maximum(first - 1, 0)] == ks[first])
        first = first[~bad]                                       # edges shared by exactly two half-edges
        h0, h1 = order[first], order[first + 1]
        good = (ha[h1] == hb[h0]) & (hb[h1] == ha[h0])            # consistently oriented
        h0, h1 = h0[good], h1[good]
        a, b, c, d = ha[h0], hb[h0], hc[h0], hc[h1]
        f0, f1 = S[h0 // 3], S[h1 // 3]
        ok = (c != d) & vm[c] & vm[d]                             # c, d interior to the sub-mesh: their edges are all visible
        idx = np.flatnonzero(ok)
        a, b, c, d, f0, f1 = a[idx], b[idx], c[idx], d[idx], f0[idx], f1[idx]
        Pa, Pb, Pc, Pd = V[a], V[b], V[c], V[d]
        gain = cot(Pa - Pc, Pb - Pc) + cot(Pb - Pd, Pa - Pd)      # < 0: the opposite angles sum to more than 180
        pre = np.flatnonzero(gain < -1e-3)                        # cheap test first; the rest only for survivors
        if len(pre) == 0:
            break
        a, b, c, d, f0, f1, gain = a[pre], b[pre], c[pre], d[pre], f0[pre], f1[pre], gain[pre]
        Pa, Pb, Pc, Pd = Pa[pre], Pb[pre], Pc[pre], Pd[pre]
        n0, n1 = np.cross(Pb - Pa, Pc - Pa), np.cross(Pa - Pb, Pd - Pb)
        t1, t2 = np.cross(Pa - Pc, Pd - Pc), np.cross(Pd - Pc, Pb - Pc)
        nav = unit(unit(n0) + unit(n1))
        sel = (np.einsum("ij,ij->i", unit(t1), nav) > min_cos) & (np.einsum("ij,ij->i", unit(t2), nav) > min_cos)
        idx = np.flatnonzero(sel)
        if len(idx) == 0:
            break
        a, b, c, d, f0, f1, gain = a[idx], b[idx], c[idx], d[idx], f0[idx], f1[idx], gain[idx]
        kcd = np.minimum(c, d) * n + np.maximum(c, d)             # the new diagonal must not exist yet
        pos = np.minimum(np.searchsorted(ks, kcd), len(ks) - 1)
        val = np.bincount(F.ravel(), minlength=n)
        idx = np.flatnonzero((ks[pos] != kcd) & (val[a] > 3) & (val[b] > 3))
        if len(idx) == 0:
            break
        a, b, c, d, f0, f1, gain = a[idx], b[idx], c[idx], d[idx], f0[idx], f1[idx], gain[idx]
        prio = np.empty(len(a), dtype=np.int64)
        prio[np.argsort(gain)] = np.arange(len(a))                # most negative first
        claim = np.full(len(F), np.iinfo(np.int64).max)
        np.minimum.at(claim, f0, prio)
        np.minimum.at(claim, f1, prio)
        idx = np.flatnonzero((claim[f0] == prio) & (claim[f1] == prio))     # each face takes part in one flip
        a, b, c, d, f0, f1 = a[idx], b[idx], c[idx], d[idx], f0[idx], f1[idx]
        dec = np.bincount(np.concatenate([a, b]), minlength=n)    # keep every valence >= 3
        idx = np.flatnonzero((val[a] - dec[a] >= 3) & (val[b] - dec[b] >= 3))
        if len(idx) == 0:
            break
        a, b, c, d, f0, f1 = a[idx], b[idx], c[idx], d[idx], f0[idx], f1[idx]
        F[f0] = np.stack([c, a, d], axis=1)
        F[f1] = np.stack([c, d, b], axis=1)
        total += len(a)
        seeds = np.concatenate([f0, f1])
        changed.append(seeds)
        if len(a) < 0.002 * len(h0):
            break
    return F, total, (np.unique(np.concatenate(changed)) if changed else np.zeros(0, dtype=np.int64))


def adaptive_refine(base, wrap, scene, min_edge, max_edge, tol=None, grade=0.4, verbose=True, max_faces=None):
    """
    Refine the uniform mesh ``base`` (edge ~``max_edge``, lying on ``wrap``) where ``wrap`` is tightly curved.

    ``tol`` is the chord error in meters (default min_edge/4): an edge on a radius R is kept below sqrt(8*tol*R).
    Edge lengths stay within [min_edge, max_edge] and change by at most ``grade`` per unit distance.
    ``max_faces`` (optional) is a budget: if the sizing field would give more triangles, ``tol`` is raised until it fits.
    ``scene`` is the closest-point scene of ``wrap``. Returns a trimesh.Trimesh.
    """
    tic = time.perf_counter()
    last = [tic]

    def tk(label):
        if verbose:
            now = time.perf_counter()
            print(f"{chr(9)}    adaptive.{label}: {now - last[0]:.1f} s")
            last[0] = now

    smin, smax = float(min_edge), float(max_edge)
    tol = float(tol) if tol else 0.25 * smin
    sizer = _WrapSizer(wrap.vertices, wrap.faces, verbose)
    size_w = sizer.size(smin, smax, tol, grade)
    tk("curvature + sizing field on the wrap")
    if max_faces:
        cap = float(max_faces) / 1.2                  # the sizing estimate runs ~20% below the real triangle count
        if sizer.estimate(size_w) > cap:
            lo, hi = tol, tol * 200.0
            for _ in range(16):
                mid = float(np.sqrt(lo * hi))
                if sizer.estimate(sizer.size(smin, smax, mid, grade)) > cap:
                    lo = mid
                else:
                    hi = mid
            if verbose:
                print(f"{chr(9)}  Curvature-adaptive: maxFaces {int(max_faces):,} raises the tolerance from "
                      f"{tol * 1000:.2f} to {hi * 1000:.2f} mm")
            tol = hi
            size_w = sizer.size(smin, smax, tol, grade)
    field = _SizeField(scene, np.asarray(wrap.vertices), size_w)
    V = np.array(base.vertices, dtype=np.float64)
    F = np.asarray(base.faces, dtype=np.int64)
    V, s = field.project(V)
    tk("sample sizes")
    refined = (s < 0.9 * smax)
    if verbose:
        print(f"{chr(9)}  Curvature-adaptive: {refined.mean() * 100:.1f}% of the base vertices want a finer mesh "
              f"(tol {tol * 1000:.2f} mm, size {smin * 1000:.1f}-{smax * 1000:.1f} mm, grade {grade:g})")

    n_pass = int(np.ceil(np.log2(smax / smin))) + 2
    n_base = len(V)
    for it in range(n_pass):
        V, F, s, ns = _split_long(V, F, s)
        if ns == 0:
            break
        n0 = len(V) - ns
        V[n0:], s[n0:] = field.project(V[n0:])
        tk(f"pass {it + 1}: split {ns:,} edges -> {len(F):,} faces")
        F, nflip, ch = _flip_delaunay(V, F, seeds=np.flatnonzero((F >= n0).any(axis=1)))
        tk(f"pass {it + 1}: {nflip:,} flips")
        ip, ix = _csr_adjacency(F, len(V))
        for _ in range(3):
            V, s = field.project(_spring(V, ip, ix, s, 0.5))
        tk(f"pass {it + 1}: smooth")
        if ns < 0.001 * len(F):                   # what is left is a handful of edges; not worth another pass
            break

    # final polish: shape first, then a little more smoothing; only around what was refined
    vm = np.zeros(len(V), dtype=bool)
    vm[n_base:] = True
    for rnd in range(2):
        F, nflip, ch = _flip_delaunay(V, F, seeds=np.flatnonzero(vm[F].any(axis=1)), max_iter=3 if rnd == 0 else 2)
        ip, ix = _csr_adjacency(F, len(V))
        for _ in range(3 if rnd == 0 else 2):
            V, s = field.project(_spring(V, ip, ix, s, 0.5))
    tk("final flips + smoothing")
    out = trimesh.Trimesh(V, F.astype(np.int64), process=False)
    keep = _nondegenerate_mask(out)
    if not keep.all():
        out.update_faces(keep)
        out.remove_unreferenced_vertices()
    if verbose:
        el = np.linalg.norm(V[F] - V[F[:, [1, 2, 0]]], axis=2)
        print(f"{chr(9)}  Curvature-adaptive: {len(base.faces):,} -> {len(out.faces):,} tris, edge "
              f"p5/p50/p95 = {np.percentile(el, 5) * 1000:.2f}/{np.percentile(el, 50) * 1000:.2f}/"
              f"{np.percentile(el, 95) * 1000:.2f} mm in {time.perf_counter() - tic:.1f} s")
    return out


def adaptive_kwargs(cfg):
    """
    Keyword arguments for :func:`remesh_surface` / :func:`remesh_surface_isolated` from the json block
    ``surfaceRemesh.curvatureAdaptive`` = {enabled, minEdge, maxEdge, tolerance, gradation} (meters); ``{}`` when off.
    ``maxEdge`` (default: the target edge) is the edge length on flat areas, ``minEdge`` (default maxEdge/2) the
    smallest edge at tight curvature, ``tolerance`` the chord error (default minEdge/4), ``gradation`` how fast the
    size may change with distance (default 0.4).
    """
    if not cfg or not cfg.get("enabled", False):
        return {}
    num = lambda v: float(v) if v is not None and v is not False and float(v) > 0 else None
    mx, mn = num(cfg.get("maxEdge")), num(cfg.get("minEdge"))
    return dict(adaptive_min_edge=mn if mn else (mx / 2.0 if mx else None), adaptive_max_edge=mx,
                adaptive_tol=num(cfg.get("tolerance")), adaptive_grade=num(cfg.get("gradation")) or 0.4)


def remesh_surface(mesh, target_edge=None, max_faces=None, project=True, workers=1, verbose=True,
                   wrap_resolution=None, wrap_offset=None, gap_closure=0.0, snap_offset=None, relax_iters=0,
                   adaptive_min_edge=None, adaptive_max_edge=None, adaptive_tol=None, adaptive_grade=0.4):
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
        _dbg_topology("wrap", src)
        if target_edge is None:
            if snap_offset is not None:
                src = _snap_to_source(src, orig, 1.5 * wrap_off, snap_offset, verbose)
            return src

    edge = target_edge
    adaptive = adaptive_min_edge is not None and float(adaptive_min_edge) > 0 and project
    if adaptive:
        if adaptive_max_edge:
            edge = float(adaptive_max_edge)           # the uniform base is made at the coarse end of the range
        if float(adaptive_min_edge) >= edge:
            adaptive = False
            if verbose:
                print(f"{chr(9)}Curvature-adaptive off: minEdge {float(adaptive_min_edge) * 1000:.2f} mm is not below "
                      f"the base edge {edge * 1000:.2f} mm")
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

    if adaptive:
        out = adaptive_refine(out, src, scene, float(adaptive_min_edge), edge, adaptive_tol, adaptive_grade, verbose,
                              max_faces=max_faces)
        tick("curvature-adaptive refinement (total)")
        _dbg_topology("curvature-adaptive", out)
    elif relax_iters and project:
        out = _relax_all(out, scene, relax_iters)
        tick(f"global relax x{relax_iters}")
    tick("assemble/clean")
    _dbg_topology("resample + project + relax", out)
    if snap_offset is not None and orig is not None:
        out = _snap_to_source(out, orig, 1.5 * wrap_off, snap_offset, verbose)
        tick("snap to source (incl. BVH build)")
        _dbg_topology("snap", out)

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
                            wrap_resolution=None, wrap_offset=None, gap_closure=0.0, snap_offset=None, relax_iters=0, use_gpu=True,
                            adaptive_min_edge=None, adaptive_max_edge=None, adaptive_tol=None, adaptive_grade=0.4):
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
        if adaptive_min_edge:
            cmd += ["--adaptive-min", repr(float(adaptive_min_edge)), "--adaptive-grade", repr(float(adaptive_grade))]
            if adaptive_max_edge:
                cmd += ["--adaptive-max", repr(float(adaptive_max_edge))]
            if adaptive_tol:
                cmd += ["--adaptive-tol", repr(float(adaptive_tol))]
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
    ap.add_argument("--adaptive-min", type=float, default=None,
                    help="curvature-adaptive refinement: smallest edge (m) at tight curvature; --edge/--adaptive-max is the largest")
    ap.add_argument("--adaptive-max", type=float, default=None, help="largest edge (m) of the adaptive range (default --edge)")
    ap.add_argument("--adaptive-tol", type=float, default=None, help="chord-error tolerance in meters (default adaptive-min/4)")
    ap.add_argument("--adaptive-grade", type=float, default=0.4, help="max size change per unit distance (default 0.4)")
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
                       snap_offset=a.snap_offset, relax_iters=a.relax_iters,
                       adaptive_min_edge=a.adaptive_min, adaptive_max_edge=a.adaptive_max, adaptive_tol=a.adaptive_tol,
                       adaptive_grade=a.adaptive_grade)
    if a.output.endswith(".npz"):
        np.savez(a.output, v=np.asarray(r.vertices), f=np.asarray(r.faces))
    else:
        r.export(a.output)
    print(f"wrote {a.output}")


if __name__ == "__main__":
    _main()
