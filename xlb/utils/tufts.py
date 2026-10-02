"""Aerodynamic flow "tufts" as plain triangle geometry.

Pure numpy/scipy (no Neon / Warp), so it can be tested on any machine with a synthetic flow.

A tuft is a short flexible filament fixed at a root on the surface. Here it is a chain of
``nseg`` segments marched away from the root: each segment direction comes from the
near-wall velocity sampled at the tuft's own position (tangential flow direction plus a
wall-normal lift), blended with the previous segment for stiffness. Tubes and ribbons are
plain ``Mesh`` triangles because that is what imports reliably into Alias / VRED
(``BasisCurves`` did not).

The caller supplies ``sample_velocity(points, base_points, normals) -> (vel (N,3), ok (N,))``.
"""

import numpy as np
from scipy.spatial import cKDTree


def poisson_roots(vertices, faces, vertex_normals, spacing, rng, oversample=5.0):
    """Area-weighted random sequential (dart-throwing) sampling of the surface.

    Two points closer than ``spacing`` exclude each other only if their normals agree
    (dot > 0), so the two faces of a thin sheet can both carry tufts.
    Returns (points (T,3), normals (T,3)).
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    tri = v[f]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    total = float(area.sum())
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

    tree = cKDTree(pts)
    neigh = tree.query_ball_point(pts, r=spacing, workers=-1)
    alive = np.ones(n_cand, dtype=bool)
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


def march_tufts(
    roots,
    normals,
    sample_velocity,
    *,
    length,
    width,
    nseg=6,
    probe_height,
    root_inset=0.0008,
    stiffness=0.4,
    surface_vertices=None,
    surface_normals=None,
    flow_dir=(1.0, 0.0, 0.0),
    max_elevation_deg=35.0,
    max_push=None,
):
    """
    March every tuft away from its root. Returns (P (T,nseg+1,3), keep (T,) bool, speed (T,nseg+1)).

    Each segment is kept within ``max_elevation_deg`` of the root's tangent plane. A tuft whose joint needs
    pushing out of the body by more than ``max_push`` (default half a segment) is dropped (``keep`` False)
    rather than shoved out, which is what made tufts stand up in narrow grooves.
    """
    roots = np.asarray(roots, dtype=np.float64)
    nrm = np.asarray(normals, dtype=np.float64)
    T = len(roots)
    ds = float(length) / nseg
    P = np.zeros((T, nseg + 1, 3))
    P[:, 0] = roots - nrm * root_inset
    keep = np.ones(T, dtype=bool)
    S = np.zeros((T, nseg + 1))  # sampled near-wall speed at each joint (last joint repeats the previous)

    surf_tree = None
    if surface_vertices is not None:
        surf_tree = cKDTree(np.asarray(surface_vertices, dtype=np.float64))
        surface_normals = np.asarray(surface_normals, dtype=np.float64)
    clearance = 0.5 * float(width) + 0.0003
    cap = np.radians(float(max_elevation_deg))
    sin_cap, cos_cap = np.sin(cap), np.cos(cap)
    max_push = 0.5 * ds if max_push is None else float(max_push)

    # Fallback heading where the tangential flow vanishes: the free-stream projected onto the wall.
    fd = np.asarray(flow_dir, dtype=np.float64)
    fallback0 = _unit(fd[None, :] - (nrm @ fd)[:, None] * nrm)
    fallback0 = np.where(np.linalg.norm(fallback0, axis=1, keepdims=True) < 1e-9,
                         _unit(np.cross(nrm, [0.0, 0.0, 1.0])), fallback0)

    prev_d = None
    prev_t = fallback0
    vref = None
    for i in range(nseg):
        p = P[:, i]
        h = np.einsum("ij,ij->i", p - roots, nrm)
        probe = p + nrm * np.maximum(0.0, probe_height - h)[:, None]
        vel, ok = sample_velocity(probe, roots, nrm)
        vel = np.asarray(vel, dtype=np.float64)
        speed = np.linalg.norm(vel, axis=1)
        S[:, i] = speed
        if vref is None:
            vref = float(np.percentile(speed[ok], 99)) if ok.any() else 1.0
            keep &= ok & (speed > 1e-3 * vref)
        tiny = 1e-3 * vref

        un = np.einsum("ij,ij->i", vel, nrm)
        ut = vel - un[:, None] * nrm
        ut_mag = np.linalg.norm(ut, axis=1)
        good = ok & (ut_mag > tiny)
        t_hat = np.where(good[:, None], ut / np.maximum(ut_mag, 1e-20)[:, None], prev_t)
        prev_t = t_hat

        # A tuft never lies perfectly flat: floor the lift (4 deg at the root to 15 deg at the tip).
        floor = min(np.radians(4.0 + 11.0 * i / max(nseg - 1, 1)), cap)
        theta = np.arctan2(un, np.maximum(ut_mag, 1e-20))
        theta = np.clip(theta, floor, cap)
        d_target = np.cos(theta)[:, None] * t_hat + np.sin(theta)[:, None] * nrm
        d = d_target if prev_d is None else _unit(stiffness * prev_d + (1.0 - stiffness) * d_target)
        # the blend can overshoot the cap; project back onto it
        e = np.einsum("ij,ij->i", d, nrm)
        over = e > sin_cap
        if over.any():
            tang = _unit(d - e[:, None] * nrm, fallback=t_hat)
            d = np.where(over[:, None], cos_cap * tang + sin_cap * nrm, d)
        prev_d = d

        q = p + d * ds
        if surf_tree is not None:  # keep the tuft out of the body (concave corners, neighbouring panels)
            _, vi = surf_tree.query(q, k=1, workers=-1)
            signed = np.einsum("ij,ij->i", q - surface_vertices[vi], surface_normals[vi])
            push = np.maximum(0.0, clearance - signed)
            keep &= push <= max_push
            q = q + surface_normals[vi] * np.minimum(push, max_push)[:, None]
        P[:, i + 1] = q

    S[:, nseg] = S[:, nseg - 1]
    return P, keep, S


def _taper(nseg):
    return 1.0 - 0.4 * np.arange(nseg + 1) / nseg


def _frames(P, nrm):
    """Per-joint tangent and the two cross-section axes (width axis first)."""
    t = np.empty_like(P)
    t[:, :-1] = P[:, 1:] - P[:, :-1]
    t[:, -1] = t[:, -2]
    t = _unit(t)
    n_w = nrm[:, None, :]
    a1 = np.cross(t, n_w)
    a1 = _unit(a1, fallback=_unit(np.cross(t, np.array([0.0, 0.0, 1.0]))))
    a2 = np.cross(t, a1)
    return t, a1, a2


def tube_mesh(P, nrm, width, sides=3):
    """Tapered tubes with a tip cap. Returns (verts (V,3), faces (F,3))."""
    T, J, _ = P.shape
    nseg = J - 1
    t, a1, a2 = _frames(P, nrm)
    r = 0.5 * float(width) * _taper(nseg)
    ang = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    ring = P[:, :, None, :] + r[None, :, None, None] * (
        np.cos(ang)[None, None, :, None] * a1[:, :, None, :] + np.sin(ang)[None, None, :, None] * a2[:, :, None, :]
    )
    tip = P[:, -1:, :] + t[:, -1:, :] * 0.0005
    verts = np.concatenate([ring.reshape(T, J * sides, 3), tip], axis=1)  # (T, J*S+1, 3)
    per = J * sides + 1

    local = []
    for i in range(nseg):
        for j in range(sides):
            a = i * sides + j
            b = i * sides + (j + 1) % sides
            c, d = a + sides, b + sides
            local += [(a, b, c), (b, d, c)]
    last = nseg * sides
    for j in range(sides):
        local.append((last + j, last + (j + 1) % sides, J * sides))
    local = np.asarray(local, dtype=np.int64)
    faces = (local[None] + (np.arange(T) * per)[:, None, None]).reshape(-1, 3)
    return verts.reshape(-1, 3), faces


def tube_vertex_scalar(S, sides=3):
    """Per-vertex copy of a per-joint scalar (T,J) in the vertex order of tube_mesh."""
    T, J = S.shape
    ring = np.repeat(S[:, :, None], sides, axis=2).reshape(T, J * sides)
    return np.concatenate([ring, S[:, -1:]], axis=1).reshape(-1)


def ribbon_vertex_scalar(S):
    return np.repeat(S[:, :, None], 2, axis=2).reshape(-1)


def ribbon_mesh(P, nrm, width):
    """Tapered flat ribbons lying along the wall. Returns (verts (V,3), faces (F,3))."""
    T, J, _ = P.shape
    nseg = J - 1
    _, a1, _ = _frames(P, nrm)
    half = 0.5 * float(width) * _taper(nseg)
    left = P + a1 * half[None, :, None]
    right = P - a1 * half[None, :, None]
    verts = np.stack([left, right], axis=2)  # (T, J, 2, 3)
    per = 2 * J
    local = []
    for i in range(nseg):
        a = 2 * i
        local += [(a, a + 1, a + 2), (a + 1, a + 3, a + 2)]
    local = np.asarray(local, dtype=np.int64)
    faces = (local[None] + (np.arange(T) * per)[:, None, None]).reshape(-1, 3)
    return verts.reshape(-1, 3), faces


def tape_mesh(roots, nrm, P, size, offset=0.0005):
    """
    One square patch of tape per tuft, lying on the surface (its normal is the surface normal at the root).
    Aligned with the first tuft segment, with the root a quarter of the way in from the downstream edge, so the
    tuft leaves the tape across its far edge. Returns (verts (4T,3), faces (2T,3)).
    """
    t0 = P[:, 1] - P[:, 0]
    d = t0 - np.einsum("ij,ij->i", t0, nrm)[:, None] * nrm
    d = _unit(d, fallback=_unit(np.cross(nrm, np.array([0.0, 0.0, 1.0]))))
    b = np.cross(nrm, d)
    c = roots + nrm * offset - d * (0.25 * size)
    h = 0.5 * size
    corners = np.stack([c - d * h - b * h, c + d * h - b * h, c + d * h + b * h, c - d * h + b * h], axis=1)
    local = np.array([(0, 1, 2), (0, 2, 3)], dtype=np.int64)
    faces = (local[None] + (np.arange(len(roots)) * 4)[:, None, None]).reshape(-1, 3)
    return corners.reshape(-1, 3), faces


def build_tufts(
    vertices,
    faces,
    vertex_normals,
    sample_velocity,
    *,
    spacing=0.03,
    length=0.025,
    width=0.0015,
    nseg=6,
    probe_height,
    shape="tube",
    root_inset=0.0008,
    max_tufts=150000,
    seed=0,
    flow_dir=(1.0, 0.0, 0.0),
    tape_size=0.0,
    tape_offset=0.0005,
    max_elevation_deg=35.0,
    max_push=None,
    log=print,
):
    """Roots on the surface -> marched chains -> tube/ribbon triangles.

    Returns (verts, faces, n_tufts, vertex_speed, tape): vertex_speed is the sampled near-wall speed per vertex;
    tape is (verts, faces) of the root tape patches when ``tape_size`` > 0, else None.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    vertex_normals = np.asarray(vertex_normals, dtype=np.float64)
    rng = np.random.default_rng(seed)

    tri = vertices[faces]
    area = float(0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum())
    est = 0.65 * area / spacing**2
    if est > max_tufts:
        spacing = float(np.sqrt(0.65 * area / max_tufts))
        log(f"\tTufts: ~{est:,.0f} at the requested spacing exceeds maxTufts={max_tufts:,}; "
            f"spacing raised to {spacing * 1000:.1f} mm")

    roots, nrm = poisson_roots(vertices, faces, vertex_normals, spacing, rng)
    log(f"\tTufts: {len(roots):,} roots on {area:.2f} m2 at {spacing * 1000:.1f} mm spacing")
    empty = (np.empty((0, 3)), np.empty((0, 3), dtype=np.int64), 0, np.empty(0), None)
    if len(roots) == 0:
        return empty

    P, keep, S = march_tufts(
        roots, nrm, sample_velocity,
        length=length, width=width, nseg=nseg, probe_height=probe_height, root_inset=root_inset,
        surface_vertices=vertices, surface_normals=vertex_normals, flow_dir=flow_dir,
        max_elevation_deg=max_elevation_deg, max_push=max_push,
    )
    log(f"\tTufts: dropped {int((~keep).sum()):,} (no usable flow, or pushed too far out of a groove); "
        f"{int(keep.sum()):,} kept")
    roots = roots[keep]
    P, nrm, S = P[keep], nrm[keep], S[keep]
    if len(P) == 0:
        return empty
    tape = None
    if tape_size > 0.0:
        tv, tf = tape_mesh(roots, nrm, P, float(tape_size), offset=float(tape_offset))
        tape = (tv.astype(np.float32), tf.astype(np.int32))

    if shape == "ribbon":
        v, f = ribbon_mesh(P, nrm, width)
        vs = ribbon_vertex_scalar(S)
    else:
        v, f = tube_mesh(P, nrm, width)
        vs = tube_vertex_scalar(S)
    return v.astype(np.float32), f.astype(np.int32), len(P), vs.astype(np.float32), tape
