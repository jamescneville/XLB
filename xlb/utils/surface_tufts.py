"""Aerodynamic flow "tufts" as plain triangle geometry.

Numerical geometry uses NumPy/SciPy. Shared helpers import from xlb.utils.utils, which also requires XLB runtime dependencies.

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
from xlb.utils.surface_helpers import _unit, local_surface, poisson_roots



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
    follow_surface=False,
    max_normal_turn_deg=30.0,
    same_skin=True,
    skin_gap=0.001,
    surface_tree=None,
):
    """
    March every tuft away from its root. Returns (P (T,nseg+1,3), keep (T,) bool, speed (T,nseg+1),
    normals (T,nseg+1,3)).

    Each segment is kept within ``max_elevation_deg`` of the wall. A tuft whose joint needs pushing out of the
    body by more than ``max_push`` (default half a segment) is dropped (``keep`` False) rather than shoved out,
    which is what made tufts stand up in narrow grooves.

    ``follow_surface``: instead of the root's normal for the whole chain, each joint uses the (inverse-distance
    smoothed, turn-limited to ``max_normal_turn_deg`` per segment) normal of the nearest surface vertices, so
    "along the wall", the elevation cap and the probe point all bend with the surface at lips and edges. A tuft
    that still ends up higher above the local surface than the cap allows has left the surface and is dropped.
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
        # Reuse the same tree across the initial pass, refills and diagnostics.
        surf_tree = surface_tree if surface_tree is not None else cKDTree(np.asarray(surface_vertices, dtype=np.float64))
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

    if follow_surface and surf_tree is None:
        raise ValueError("follow_surface needs surface_vertices / surface_normals")
    max_turn = np.radians(float(max_normal_turn_deg))
    NL = np.repeat(nrm[:, None, :], nseg + 1, axis=1)

    prev_d = None
    prev_t = fallback0
    vref = None
    for i in range(nseg):
        p = P[:, i]
        if follow_surface and i > 0:
            vi, w, v0 = local_surface(surf_tree, surface_vertices, surface_normals, p, nl, same_skin, skin_gap)
            raw = _unit((surface_normals[vi] * w[:, :, None]).sum(axis=1), fallback=nl)
            ang = np.arccos(np.clip(np.einsum("ij,ij->i", nl, raw), -1.0, 1.0))
            a = np.minimum(1.0, max_turn / np.maximum(ang, 1e-9))
            nl = _unit((1.0 - a)[:, None] * nl + a[:, None] * raw, fallback=nl)
            base = surface_vertices[v0]
            h = np.einsum("ij,ij->i", p - base, surface_normals[v0])
            # higher above the local surface than the elevation cap allows: it left the surface
            keep &= h <= i * ds * sin_cap + 2.0 * float(width) + 0.002
        else:
            nl = nrm
            base = roots
            h = np.einsum("ij,ij->i", p - roots, nrm)
        NL[:, i] = nl
        probe = p + nl * np.maximum(0.0, probe_height - h)[:, None]
        vel, ok = sample_velocity(probe, base, nl)
        vel = np.asarray(vel, dtype=np.float64)
        speed = np.linalg.norm(vel, axis=1)
        S[:, i] = speed
        if vref is None:
            vref = float(np.percentile(speed[ok], 99)) if ok.any() else 1.0
            keep &= ok & (speed > 1e-3 * vref)
        tiny = 1e-3 * vref

        un = np.einsum("ij,ij->i", vel, nl)
        ut = vel - un[:, None] * nl
        ut_mag = np.linalg.norm(ut, axis=1)
        good = ok & (ut_mag > tiny)
        t_hat = np.where(good[:, None], ut / np.maximum(ut_mag, 1e-20)[:, None], prev_t)
        prev_t = t_hat

        # A tuft never lies perfectly flat: floor the lift (4 deg at the root to 15 deg at the tip).
        floor = min(np.radians(4.0 + 11.0 * i / max(nseg - 1, 1)), cap)
        theta = np.arctan2(un, np.maximum(ut_mag, 1e-20))
        theta = np.clip(theta, floor, cap)
        d_target = np.cos(theta)[:, None] * t_hat + np.sin(theta)[:, None] * nl
        d = d_target if prev_d is None else _unit(stiffness * prev_d + (1.0 - stiffness) * d_target)
        # the blend can overshoot the cap; project back onto it
        e = np.einsum("ij,ij->i", d, nl)
        over = e > sin_cap
        if over.any():
            tang = _unit(d - e[:, None] * nl, fallback=t_hat)
            d = np.where(over[:, None], cos_cap * tang + sin_cap * nl, d)
        prev_d = d

        q = p + d * ds
        if surf_tree is not None:  # keep the tuft out of the body (concave corners, neighbouring panels)
            _, _, v0 = local_surface(surf_tree, surface_vertices, surface_normals, q, nl, same_skin, skin_gap)
            signed = np.einsum("ij,ij->i", q - surface_vertices[v0], surface_normals[v0])
            push = np.maximum(0.0, clearance - signed)
            keep &= push <= max_push
            q = q + surface_normals[v0] * np.minimum(push, max_push)[:, None]
        P[:, i + 1] = q

    S[:, nseg] = S[:, nseg - 1]
    NL[:, nseg] = NL[:, nseg - 1]
    return P, keep, S, NL


class TriIndex:
    """Triangle arrays and a KD tree on triangle centres, built once and shared by the seed and tuft crossing tests."""

    def __init__(self, vertices, faces):
        V = np.asarray(vertices, dtype=np.float64)
        F = np.asarray(faces, dtype=np.int64)
        tri = V[F]
        self.v0 = tri[:, 0]
        self.e1 = tri[:, 1] - tri[:, 0]
        self.e2 = tri[:, 2] - tri[:, 0]
        fn = np.cross(self.e1, self.e2)
        self.fn = fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-30)
        cen = tri.mean(axis=1)
        self.tri_r = float(np.linalg.norm(tri - cen[:, None, :], axis=2).max())
        self.tree = cKDTree(cen)


def crossing_tufts(P, roots, root_normals, vertices, faces, root_zone, chunk=3_000_000, index=None):
    """
    Which tufts pass through the surface? Tests every centre-line segment against the triangles near it
    (Moller-Trumbore, candidates from a KD tree on triangle centres). Returns (crossed (T,) bool, hits (T,) int).

    A tuft is rooted 0.8 mm inside its own skin, so its first segment legitimately pierces the triangles at its
    root: same-facing triangles (normal within ~70 deg of the root normal) whose plane passes within 2 mm of the root and whose hit point is within ``root_zone`` of the
    root are ignored. Everything else counts: a hit on a triangle facing the other way is the other skin
    of a panel (the tuft popped through); a same-facing hit away from the root is the tuft re-entering its own
    surface.
    """
    ix = index if index is not None else TriIndex(vertices, faces)
    v0, e1, e2, fn, tri_r, tree = ix.v0, ix.e1, ix.e2, ix.fn, ix.tri_r, ix.tree

    T, J, _ = P.shape
    a = P[:, :-1].reshape(-1, 3)
    b = P[:, 1:].reshape(-1, 3)
    seg_tuft = np.repeat(np.arange(T), J - 1)
    seg_first = np.tile(np.arange(J - 1) == 0, T)
    mid = 0.5 * (a + b)
    segment_vectors = b - a
    rad = 0.5 * np.linalg.norm(segment_vectors, axis=1) + tri_r + 1e-4
    # Preserve original candidate search to retain speed for uniform-length segments.
    neigh = tree.query_ball_point(mid, r=float(rad.max()), workers=-1)
    counts = np.fromiter((len(n) for n in neigh), dtype=np.int64, count=len(neigh))
    seg_idx = np.repeat(np.arange(len(a)), counts)
    tri_idx = np.concatenate([np.asarray(n, dtype=np.int64) for n in neigh]) if counts.sum() else np.empty(0, np.int64)

    hits = np.zeros(T, dtype=np.int64)
    for s in range(0, len(seg_idx), chunk):
        si, ti = seg_idx[s:s + chunk], tri_idx[s:s + chunk]
        o, d = a[si], segment_vectors[si]
        pv = np.cross(d, e2[ti])
        det = np.einsum("ij,ij->i", e1[ti], pv)
        ok = np.abs(det) > 1e-18
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        tv = o - v0[ti]
        u = np.einsum("ij,ij->i", tv, pv) * inv
        qv = np.cross(tv, e1[ti])
        w = np.einsum("ij,ij->i", d, qv) * inv
        t = np.einsum("ij,ij->i", e2[ti], qv) * inv
        hit = ok & (u >= 0.0) & (w >= 0.0) & (u + w <= 1.0) & (t >= 0.0) & (t <= 1.0)
        if not hit.any():
            continue
        hs, ht = si[hit], ti[hit]
        tuft = seg_tuft[hs]
        hp = a[hs] + segment_vectors[hs] * t[hit][:, None]
        near_root = np.linalg.norm(hp - roots[tuft], axis=1) <= root_zone
        same = np.einsum("ij,ij->i", fn[ht], root_normals[tuft]) > 0.34
        # the root sits ~1 mm inside its skin: only a triangle whose plane passes within ~2 mm of the root is that skin
        # (a second skin 4+ mm away, even facing the same way, is a pop-through)
        own_plane = np.abs(np.einsum("ij,ij->i", roots[tuft] - v0[ht], fn[ht])) <= 0.002
        count = ~(near_root & same & own_plane & seg_first[hs])      # only the first segment may pierce its own root skin
        np.add.at(hits, tuft[count], 1)
    return hits > 0, hits


def _taper(nseg):
    return 1.0 - 0.4 * np.arange(nseg + 1) / nseg


def _frames(P, nrm):
    """Per-joint tangent and the two cross-section axes (width axis first)."""
    t = np.empty_like(P)
    t[:, :-1] = P[:, 1:] - P[:, :-1]
    t[:, -1] = t[:, -2]
    t = _unit(t)
    n_w = nrm[:, None, :] if nrm.ndim == 2 else nrm  # (T,3) root normal or (T,J,3) per-joint normals
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
    follow_surface=False,
    max_normal_turn_deg=30.0,
    drop_crossing=True,
    seed_clearance=0.6,
    same_skin=True,
    refill_passes=2,
    skin_gap=0.001,
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
    twice_area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    area = float(0.5 * twice_area.sum())  # preserve original estimate reduction order
    area_per_face = 0.5 * twice_area
    surface_cache = (tri, area_per_face, float(area_per_face.sum()))
    est = 0.65 * area / spacing**2
    if est > max_tufts:
        spacing = float(np.sqrt(0.65 * area / max_tufts))
        log(f"\tTufts: ~{est:,.0f} at the requested spacing exceeds maxTufts={max_tufts:,}; "
            f"spacing raised to {spacing * 1000:.1f} mm")

    tri_index = TriIndex(vertices, faces) if (drop_crossing or seed_clearance > 0) else None
    accept = None
    if seed_clearance > 0:
        reach = float(seed_clearance) * float(length)

        def accept(pts, nr):
            # a seed whose outward normal meets the surface within the tuft's reach cannot host a tuft: it sits on an
            # inner skin / mis-oriented patch or in a slot. Tested before thinning so it never takes a slot from a good seed.
            seg = np.stack([pts + nr * 0.0015, pts + nr * reach], axis=1)
            blocked, _ = crossing_tufts(seg, pts, nr, vertices, faces, root_zone=0.0, index=tri_index)
            return ~blocked

    roots, nrm = poisson_roots(vertices, faces, vertex_normals, spacing, rng, accept=accept,
                               surface_cache=surface_cache)
    log(f"\tTufts: {len(roots):,} roots on {area:.2f} m2 at {spacing * 1000:.1f} mm spacing")
    empty = (np.empty((0, 3)), np.empty((0, 3), dtype=np.int64), 0, np.empty(0), None)
    if len(roots) == 0:
        return empty

    surface_tree = cKDTree(vertices)

    def grow(r, n, tag=""):
        P, keep, S, NL = march_tufts(
            r, n, sample_velocity,
            length=length, width=width, nseg=nseg, probe_height=probe_height, root_inset=root_inset,
            surface_vertices=vertices, surface_normals=vertex_normals, flow_dir=flow_dir,
            max_elevation_deg=max_elevation_deg, max_push=max_push,
            follow_surface=follow_surface, max_normal_turn_deg=max_normal_turn_deg,
            same_skin=same_skin, skin_gap=skin_gap, surface_tree=surface_tree,
        )
        log(f"\tTufts{tag}: dropped {int((~keep).sum()):,} (no usable flow, pushed too far out of a groove"
            f"{', or left the surface' if follow_surface else ''}); {int(keep.sum()):,} kept")
        if drop_crossing and keep.any():
            cr, _ = crossing_tufts(P[keep], r[keep], n[keep], vertices, faces, root_zone=3.0 * float(width) + 0.004,
                                   index=tri_index)
            idx = np.flatnonzero(keep)
            keep[idx[cr]] = False
            log(f"\tTufts{tag}: dropped {int(cr.sum()):,} that pass through the surface (popped through a panel or re-entered it); "
                f"{int(keep.sum()):,} kept")
        return P, keep, S, NL

    P, keep, S, NL = grow(roots, nrm)
    kept = [(roots[keep], nrm[keep], P[keep], S[keep], NL[keep])]
    lost_p, lost_n = roots[~keep], nrm[~keep]
    for it in range(int(refill_passes)):
        if len(lost_p) == 0:
            break
        lost_tree = cKDTree(lost_p)

        def refill_accept(pts, nr):
            # only candidates in a hole, and not on the same skin patch as a root that already failed there
            d, ii = lost_tree.query(pts, k=4, distance_upper_bound=spacing, workers=-1)
            ok = np.isfinite(d[:, 0])
            idx = np.flatnonzero(ok)
            if len(idx):
                ii = np.minimum(ii[idx], len(lost_p) - 1)
                diff = pts[idx, None, :] - lost_p[ii]
                same_patch = (np.isfinite(d[idx]) & (np.linalg.norm(diff, axis=2) < 0.5 * spacing)
                              & (np.abs(np.einsum("nkj,nkj->nk", diff, lost_n[ii])) < 0.002)
                              & (np.einsum("nj,nkj->nk", nr[idx], lost_n[ii]) > 0.9))
                ok[idx] = ~same_patch.any(axis=1)
            sub = np.flatnonzero(ok)
            if accept is not None and len(sub):
                ok[sub] = accept(pts[sub], nr[sub])
            return ok

        have = (np.concatenate([k[0] for k in kept]), np.concatenate([k[1] for k in kept]))
        r2, n2 = poisson_roots(vertices, faces, vertex_normals, spacing, rng, accept=refill_accept,
                               existing=have, surface_cache=surface_cache)
        if len(r2) == 0:
            break
        P2, k2, S2, NL2 = grow(r2, n2, tag=f" (refill {it + 1})")
        if k2.any():
            kept.append((r2[k2], n2[k2], P2[k2], S2[k2], NL2[k2]))
        lost_p = np.concatenate([lost_p, r2[~k2]])
        lost_n = np.concatenate([lost_n, n2[~k2]])
    roots, nrm, P, S, NL = (np.concatenate([k[j] for k in kept]) for j in range(5))
    if len(P) == 0:
        return empty
    # how far the tips ended up above the local surface (a tuft lying at the elevation cap is length*sin(cap) up)
    dv, vi = surface_tree.query(P[:, -1], k=1, workers=-1)
    tip_h = np.einsum("ij,ij->i", P[:, -1] - vertices[vi], vertex_normals[vi])
    log(f"\tTufts: tip height above the surface: median {np.median(tip_h) * 1000:.1f} mm, "
        f"p99 {np.percentile(tip_h, 99) * 1000:.1f} mm, max {tip_h.max() * 1000:.1f} mm; "
        f"{100.0 * float(np.mean(tip_h > 0.6 * length)):.2f}% above 0.6 x length")
    frame_n = NL if follow_surface else nrm
    tape = None
    if tape_size > 0.0:
        tv, tf = tape_mesh(roots, nrm, P, float(tape_size), offset=float(tape_offset))
        tape = (tv.astype(np.float32), tf.astype(np.int32))

    if shape == "ribbon":
        v, f = ribbon_mesh(P, frame_n, width)
        vs = ribbon_vertex_scalar(S)
    else:
        v, f = tube_mesh(P, frame_n, width)
        vs = tube_vertex_scalar(S)
    return v.astype(np.float32), f.astype(np.int32), len(P), vs.astype(np.float32), tape
