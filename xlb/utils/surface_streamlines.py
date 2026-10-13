"""Near-wall streamlines extracted from time-averaged CFD velocity.

Pure NumPy/SciPy geometry code. No NEON, Warp, USD or JSON dependencies.

Public functions
----------------
wall_streamlines(...): trace forward/backward polylines on the surface.
polyline_tubes(...): convert masked polylines into independent tube triangles.
build_wall_streamlines(...): convenience entry point for MultiresIO.

The caller supplies a vectorized callback:
    sample_velocity(points, base_points, normals) -> (velocity[N,3], valid[N])

Coordinates and all lengths are in meters; velocities are in m/s when the
MultiresIO unit converter is configured. Vertex normals MUST be consistently
outward-facing. This is a surface-constrained visualization: it traces the
near-wall tangential velocity direction and projects joints back onto a
piecewise-normal approximation of the surface. It is not a no-slip CFD wall
shear-stress/skin-friction streamline, nor a geodesic/face-walking integrator.

Based on the wall-streamline algorithm in the original surface_maps.py, with:
* shared poisson_roots, local_surface, _unit from surface_helpers;
* active-only velocity sampling (stopped lines cost no more sampler queries);
* robust empty/stagnant/invalid-field handling;
* a bounded seed-candidate budget and a strict maximum line count;
* direct compact tube indexing, avoiding np.unique(faces).

These changes are intentional behavioral improvements, NOT a bitwise-equivalent
refactor of the original surface_maps.py.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

from xlb.utils.surface_helpers import _unit, local_surface, poisson_roots


def _smooth_normals(vertices, faces, normals, iterations):
    """Normalize and smooth steering normals over connected mesh vertices.

    This ONLY influences streamline tracing. It does not change surface mesh
    coordinates or winding. The original outward normals are still used for
    sidedness checks and surface-height evaluation.
    """
    normals = _unit(np.asarray(normals, dtype=np.float64))
    if iterations <= 0:
        return normals

    faces = np.asarray(faces, dtype=np.int64)
    nv = len(vertices)
    src = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2],
                          faces[:, 1], faces[:, 2], faces[:, 0]])
    dst = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0],
                          faces[:, 0], faces[:, 1], faces[:, 2]])
    adj = coo_matrix((np.ones(len(src), dtype=np.float64), (src, dst)),
                     shape=(nv, nv)).tocsr()
    adj.data[:] = 1.0  # unique neighbors, not triangle multiplicity
    degrees = np.asarray(adj.sum(axis=1)).ravel()
    adj = adj.multiply(1.0 / np.maximum(degrees, 1.0)[:, None]).tocsr()

    out = normals.copy()
    for _ in range(int(iterations)):
        out = _unit(0.5 * out + 0.5 * (adj @ out), fallback=normals)
    return out


def _trace(
    roots, root_normals, sample_velocity, tree, vertices, normals, *,
    sign, nsteps, step, probe_height, lift, turn_cos, steer,
    min_speed_fraction=1e-3, vref=None, same_skin=True,
    max_step_turn=None, skin_gap=0.001, surface_distance_limit=None,
):
    """Integrate one direction (+/- projected flow) from every seed.

    Outputs (points, normals, speeds, valid_joints, reference_speed). Valid
    joints start at the root and stay contiguous. Invalid positions remain
    NaN and are never exported. Each step samples ONLY still-alive lines.
    """
    n_lines = len(roots)
    points = np.full((n_lines, nsteps + 1, 3), np.nan, dtype=np.float64)
    joint_normals = np.zeros((n_lines, nsteps + 1, 3), dtype=np.float64)
    speeds = np.zeros((n_lines, nsteps + 1), dtype=np.float64)
    valid = np.zeros((n_lines, nsteps + 1), dtype=bool)

    current = roots + root_normals * lift
    current_normals = root_normals.copy()
    base = roots.copy()
    points[:, 0] = current
    joint_normals[:, 0] = current_normals
    valid[:, 0] = True

    active = np.ones(n_lines, dtype=bool)
    previous_direction = np.zeros_like(roots)
    has_previous = np.zeros(n_lines, dtype=bool)

    for s in range(nsteps):
        ids = np.flatnonzero(active)
        if ids.size == 0:
            break

        p = current[ids]
        nl = current_normals[ids]
        probe = p + nl * max(float(probe_height) - float(lift), 0.0)
        vel, ok = sample_velocity(probe, base[ids], nl)
        vel = np.asarray(vel, dtype=np.float64)
        ok = np.asarray(ok, dtype=bool)
        if vel.shape != (len(ids), 3) or ok.shape != (len(ids),):
            raise ValueError("sample_velocity must return (N,3) velocities and (N,) validity")

        finite = np.isfinite(vel).all(axis=1)
        ok &= finite
        vel = np.where(finite[:, None], vel, 0.0)
        speed = np.linalg.norm(vel, axis=1)
        # Keep the previous finite speed at a terminal point if the *next*
        # velocity lookup fails; never bake an invalid velocity into the line.
        speeds[ids, s] = np.where(ok, speed, speeds[ids, s])

        if vref is None:
            vref = float(np.percentile(speed[ok], 99)) if ok.any() else 1.0
        speed_floor = float(min_speed_fraction) * max(float(vref), 1e-12)

        vn = np.einsum("ij,ij->i", vel, nl)
        tangent = vel - vn[:, None] * nl
        tangent_speed = np.linalg.norm(tangent, axis=1)
        direction = float(sign) * tangent / np.maximum(tangent_speed, 1e-20)[:, None]

        stop = ~ok | (tangent_speed < speed_floor)
        previous = has_previous[ids]
        if previous.any():
            dot = np.einsum("ij,ij->i", direction[previous], previous_direction[ids[previous]])
            stop[previous] |= dot < turn_cos

            if max_step_turn is not None:
                # Preserve the original style of direction smoothing (blend
                # toward the new tangent, then normalize). This is a smoothing
                # control, not an exact spherical angular clamp.
                angle = np.arccos(np.clip(dot, -1.0, 1.0))
                frac = np.minimum(1.0, max_step_turn / np.maximum(angle, 1e-9))
                mixed = _unit(
                    (1.0 - frac)[:, None] * previous_direction[ids[previous]]
                    + frac[:, None] * direction[previous],
                    fallback=direction[previous],
                )
                direction[previous] = mixed

        # Do geometry queries only for samples still eligible to advance.
        good = ~stop
        good_ids = ids[good]
        active[ids[~good]] = False
        if good_ids.size == 0:
            continue

        trial = p[good] + direction[good] * float(step)
        vi, w, v0 = local_surface(
            tree, vertices, normals, trial, nl[good], same_skin, skin_gap
        )
        local_normal = _unit(
            (steer[vi] * w[:, :, None]).sum(axis=1),
            fallback=nl[good],
        )

        # Project back to the specified height above the local tangent plane.
        # Divide by the normal alignment so the resulting signed height is
        # actually 'lift' even when the smoothed steering normal differs from
        # the raw vertex normal. Reject reversed/near-perpendicular normals.
        face_normal = normals[v0]
        normal_dot = np.einsum("ij,ij->i", local_normal, face_normal)
        acceptable_normal = normal_dot > 0.25
        height = np.einsum("ij,ij->i", trial - vertices[v0], face_normal)
        delta = (height - float(lift)) / np.maximum(normal_dot, 0.25)
        newp = trial - local_normal * delta[:, None]

        advance = np.linalg.norm(newp - p[good], axis=1)
        keep_step = (
            acceptable_normal
            & np.isfinite(newp).all(axis=1)
            & (advance <= 2.0 * float(step))
        )
        if surface_distance_limit is not None:
            # Without this check, a nearest-vertex projection continues drawing
            # lines indefinitely beyond open surface edges on a flat panel.
            keep_step &= (
                np.linalg.norm(newp - vertices[v0], axis=1)
                <= float(surface_distance_limit)
            )
        accepted = good_ids[keep_step]
        active[good_ids[~keep_step]] = False
        if accepted.size == 0:
            continue

        newp = newp[keep_step]
        new_normals = local_normal[keep_step]
        points[accepted, s + 1] = newp
        joint_normals[accepted, s + 1] = new_normals
        speeds[accepted, s + 1] = speed[good][keep_step]
        valid[accepted, s + 1] = True

        current[accepted] = newp
        current_normals[accepted] = new_normals
        base[accepted] = vertices[v0[keep_step]]
        previous_direction[accepted] = direction[good][keep_step]
        has_previous[accepted] = True

    return points, joint_normals, speeds, valid, (1.0 if vref is None else vref)


def wall_streamlines(
    vertices, faces, vertex_normals, sample_velocity, *,
    spacing=0.06, length=0.30, step=0.012, probe_height=0.014,
    lift=0.001, turn_limit_deg=100.0, seed=0, max_lines=10000,
    min_joints=4, same_skin=True, smooth_normals=3,
    max_step_turn_deg=30.0, skin_gap=0.001, surface_distance_limit=None,
    log=print,
):
    """Trace lines forward and backward from area-weighted surface roots.

    Returns (P, joint_normals, speeds, valid_mask), each with shape
    (n_lines, 2*nsteps+1, ...). The root is the central joint. Joints are
    contiguous around it; outside points are NaN and masked out.

    'length' is the *total* maximum two-direction trace distance, not the
    distance on each side. 'step' is an upper bound on spatial step size;
    actual_step = (length/2)/ceil((length/2)/step). The streamline can end
    sooner at stagnation, a sharp reversal, or a geometric jump.
    """
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    N = _unit(np.asarray(vertex_normals, dtype=np.float64))
    if V.ndim != 2 or V.shape[1] != 3 or F.ndim != 2 or F.shape[1] != 3:
        raise ValueError("vertices and faces must have shapes (V,3) and (F,3)")
    if N.shape != V.shape:
        raise ValueError("vertex_normals must have shape (V,3)")
    if not len(F) or len(V) < 8:
        raise ValueError("surface requires >=8 vertices and at least one triangle (shared local_surface uses k=8)")
    if not np.isfinite(V).all() or not np.isfinite(N).all():
        raise ValueError("vertices and normals must be finite")
    if F.min() < 0 or F.max() >= len(V):
        raise ValueError("faces contain invalid vertex indices")
    if any(not np.isfinite(x) or x <= 0 for x in (spacing, length, step, probe_height)):
        raise ValueError("spacing, length, step and probe_height must be positive and finite")
    if not np.isfinite(lift) or lift < 0:
        raise ValueError("lift must be nonnegative and finite")
    if not np.isfinite(skin_gap) or skin_gap < 0:
        raise ValueError("skin_gap must be nonnegative and finite")
    if max_lines < 1 or min_joints < 2 or smooth_normals < 0:
        raise ValueError("max_lines>=1, min_joints>=2 and smooth_normals>=0 are required")
    if not (0.0 <= turn_limit_deg <= 180.0):
        raise ValueError("turn_limit_deg must be in [0,180]")
    if max_step_turn_deg is not None and not (0.0 <= max_step_turn_deg <= 180.0):
        raise ValueError("max_step_turn_deg must be in [0,180] or None")

    nsteps = max(1, int(np.ceil(0.5 * float(length) / float(step))))
    if nsteps > 2048:
        raise ValueError("too many steps per side; increase 'step' or decrease 'length'")
    if min_joints > 2 * nsteps + 1:
        raise ValueError("min_joints exceeds the maximum number of joints for length/step")
    actual_step = 0.5 * float(length) / nsteps

    if surface_distance_limit is None:
        # Heuristic boundary guard: estimate typical triangle edge length from
        # a bounded, spatially distributed sample of faces. A vertex projection
        # is not a closest-triangle query, so allow several edge lengths. This
        # stops indefinite extrapolation past an open surface edge while
        # tolerating normal finite-element vertex spacing within a face.
        stride = max(1, len(F) // 20000)
        tr = V[F[::stride][:20000]]
        edges = np.concatenate((
            np.linalg.norm(tr[:, 1] - tr[:, 0], axis=1),
            np.linalg.norm(tr[:, 2] - tr[:, 1], axis=1),
            np.linalg.norm(tr[:, 0] - tr[:, 2], axis=1),
        ))
        edges = edges[edges > 0.0]
        typical_edge = float(np.median(edges)) if edges.size else actual_step
        surface_distance_limit = max(2.0 * actual_step, 3.0 * typical_edge, 2.0 * lift)
    if not np.isfinite(surface_distance_limit) or surface_distance_limit <= 0:
        raise ValueError("surface_distance_limit must be positive and finite")

    # Bound the Poisson dart-throwing memory cost BEFORE constructing candidate
    # lists. poisson_roots uses oversample=5 and O(area/spacing^2) candidates.
    area_tri = V[F]
    tri_cross = np.cross(area_tri[:, 1] - area_tri[:, 0], area_tri[:, 2] - area_tri[:, 0])
    area = 0.5 * float(np.linalg.norm(tri_cross, axis=1).sum())
    del area_tri, tri_cross
    budget = max(256, 8 * int(max_lines))
    estimated = 5.0 * area / float(spacing)**2 + 1.0
    if estimated > budget:
        spacing = float(np.sqrt(5.0 * area / max(budget - 1, 1))) * (1.0 + 1e-12)
        log(f"\tWall streamlines: spacing increased to {1000*spacing:.1f} mm to limit seed candidates")

    rng = np.random.default_rng(seed)
    roots, nrm = poisson_roots(V, F, N, spacing, rng)
    if len(roots) > max_lines:
        select = rng.choice(len(roots), size=int(max_lines), replace=False)
        roots, nrm = roots[select], nrm[select]
        log(f"\tWall streamlines: capped at {max_lines:,} roots")
    log(f"\tWall streamlines: {len(roots):,} seeds at {spacing*1000:.1f} mm spacing; "
        f"step {actual_step*1000:.2f} mm; <= {length*1000:.0f} mm full length")

    num_joints = 2 * nsteps + 1
    # Protect against allocating several gigabytes of dense position/normal
    # arrays if a user combines a very small step with a huge seed count.
    if len(roots) * num_joints > 3_000_000:
        raise ValueError(
            "streamline trace requires over 3 million joint slots; "
            "decrease max_lines/length or increase step"
        )
    if len(roots) == 0:
        empty_pts = np.empty((0, num_joints, 3), dtype=np.float64)
        empty_spd = np.empty((0, num_joints), dtype=np.float64)
        empty_valid = np.empty((0, num_joints), dtype=bool)
        log("\tWall streamlines: no roots on this surface")
        return empty_pts, empty_pts.copy(), empty_spd, empty_valid

    surface_tree = cKDTree(V)
    steer = _smooth_normals(V, F, N, smooth_normals)
    opts = dict(
        nsteps=nsteps, step=actual_step, probe_height=float(probe_height),
        lift=float(lift), turn_cos=float(np.cos(np.radians(turn_limit_deg))),
        steer=steer, same_skin=bool(same_skin), skin_gap=float(skin_gap),
        max_step_turn=(None if max_step_turn_deg is None else np.radians(max_step_turn_deg)),
        surface_distance_limit=float(surface_distance_limit),
    )
    forward = _trace(roots, nrm, sample_velocity, surface_tree, V, N, sign=1.0, **opts)
    backward = _trace(roots, nrm, sample_velocity, surface_tree, V, N,
                      sign=-1.0, vref=forward[-1], **opts)
    Pf, Nf, Sf, Vf, _ = forward
    Pb, Nb, Sb, Vb, _ = backward
    P = np.concatenate((Pb[:, :0:-1], Pf), axis=1)
    NL = np.concatenate((Nb[:, :0:-1], Nf), axis=1)
    SP = np.concatenate((Sb[:, :0:-1], Sf), axis=1)
    valid = np.concatenate((Vb[:, :0:-1], Vf), axis=1)

    lengths = valid.sum(axis=1)
    keep = lengths >= int(min_joints)
    if keep.any():
        log(f"\tWall streamlines: {int(keep.sum()):,} kept; "
            f"median represented length {(np.median(lengths[keep])-1)*actual_step*1000:.0f} mm")
    else:
        log("\tWall streamlines: no lines survived flow and geometric checks")
    return P[keep], NL[keep], SP[keep], valid[keep]


def polyline_tubes(P, NL, valid, speed, width, sides=3):
    """Build triangle tubes from variable-length, contiguous masked polylines.

    Returns (vertices float32 [N,3], faces int32 [M,3], speed float32 [N]).
    Outputs only the rings referenced by valid segments; no np.unique() or
    giant temporary indexed face list is required for vertex compaction.
    """
    P = np.asarray(P, dtype=np.float64)
    NL = np.asarray(NL, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    speed = np.asarray(speed, dtype=np.float64)
    if P.ndim != 3 or P.shape[2] != 3:
        raise ValueError("P must have shape (L,J,3)")
    L, J, _ = P.shape
    if NL.shape != P.shape or speed.shape != (L, J) or valid.shape != (L, J):
        raise ValueError("NL, valid, speed must match P joint dimensions")
    if J < 2 or sides < 3 or width <= 0:
        raise ValueError("at least two joints, sides>=3 and width>0 required")
    if L == 0:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.int32), np.empty(0, np.float32)
    if not valid.any(axis=1).all():
        raise ValueError("each input line needs at least one valid joint")

    # A valid streamline is one contiguous run, including the center/root.
    first = np.argmax(valid, axis=1)
    last = J - 1 - np.argmax(valid[:, ::-1], axis=1)
    if np.any(valid.sum(axis=1) != last - first + 1):
        raise ValueError("valid joints must be contiguous on each streamline")

    indices = np.clip(np.arange(J)[None, :], first[:, None], last[:, None])
    Pf = np.take_along_axis(P, indices[:, :, None], axis=1)
    Nf = np.take_along_axis(NL, indices[:, :, None], axis=1)

    tangent = np.empty_like(Pf)
    tangent[:, 1:-1] = Pf[:, 2:] - Pf[:, :-2]
    tangent[:, 0] = Pf[:, 1] - Pf[:, 0]
    tangent[:, -1] = Pf[:, -1] - Pf[:, -2]
    tangent = _unit(tangent, fallback=np.array([1.0, 0.0, 0.0]))

    a1 = np.cross(tangent, Nf)
    fallback = _unit(np.cross(tangent, np.array([0.0, 0.0, 1.0])))
    needs_alternate = np.linalg.norm(fallback, axis=2) < 1e-12
    if needs_alternate.any():
        fallback[needs_alternate] = _unit(
            np.cross(tangent[needs_alternate], np.array([0.0, 1.0, 0.0]))
        )
    a1 = _unit(a1, fallback=fallback)
    a2 = np.cross(tangent, a1)

    theta = np.linspace(0.0, 2.0*np.pi, int(sides), endpoint=False)
    ring = Pf[:, :, None, :] + 0.5 * float(width) * (
        np.cos(theta)[None, None, :, None] * a1[:, :, None, :]
        + np.sin(theta)[None, None, :, None] * a2[:, :, None, :]
    )

    # A joint is used if at least one adjacent segment exists. This covers
    # single-segment polylines too; the builder normally filters them earlier.
    seg_ok = valid[:, :-1] & valid[:, 1:]
    joint_ok = np.zeros_like(valid)
    joint_ok[:, :-1] |= seg_ok
    joint_ok[:, 1:] |= seg_ok
    if not joint_ok.any():
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.int32), np.empty(0, np.float32)

    # Contiguous ring IDs assigned in row-major order, like sorted np.unique()
    # indices, but without allocating/sorting the complete face index stream.
    ring_id = np.full((L, J), -1, dtype=np.int64)
    ring_id[joint_ok] = np.arange(int(joint_ok.sum()), dtype=np.int64)
    line, seg = np.nonzero(seg_ok)
    start = ring_id[line, seg] * int(sides)
    end = ring_id[line, seg + 1] * int(sides)
    around = np.arange(int(sides), dtype=np.int64)[None, :]
    after = (around + 1) % int(sides)
    a = start[:, None] + around
    b = start[:, None] + after
    c = end[:, None] + around
    d = end[:, None] + after
    faces = np.stack((
        np.stack((a, b, c), axis=-1),
        np.stack((b, d, c), axis=-1),
    ), axis=2).reshape(-1, 3)

    out_vertices = ring[joint_ok].reshape(-1, 3).astype(np.float32)
    out_speed = np.repeat(np.take_along_axis(speed, indices, axis=1)[joint_ok], int(sides)).astype(np.float32)
    if len(out_vertices) > np.iinfo(np.int32).max:
        raise OverflowError("streamline tube vertex count exceeds int32 USD index budget")
    return out_vertices, faces.astype(np.int32), out_speed


def build_wall_streamlines(
    vertices, faces, vertex_normals, sample_velocity, *,
    spacing=0.06, length=0.30, step=0.012, probe_height=0.014,
    width=0.002, sides=3, lift=0.001, turn_limit_deg=100.0,
    seed=0, max_lines=10000, min_joints=4, same_skin=True,
    smooth_normals=3, max_step_turn_deg=30.0, skin_gap=0.001,
    surface_distance_limit=None, log=print,
):
    """Top-level streamline builder for MultiresIO.

    Returns (tube_vertices, triangle_faces, line_count, vertex_speeds), with
    NumPy types matching build_tufts (except no tape object). Does not write USD.
    """
    if not np.isfinite(width) or width <= 0 or int(sides) < 3:
        raise ValueError("width must be positive and sides>=3")
    P, NL, speeds, valid = wall_streamlines(
        vertices, faces, vertex_normals, sample_velocity,
        spacing=spacing, length=length, step=step, probe_height=probe_height,
        lift=lift, turn_limit_deg=turn_limit_deg, seed=seed,
        max_lines=max_lines, min_joints=min_joints, same_skin=same_skin,
        smooth_normals=smooth_normals, max_step_turn_deg=max_step_turn_deg,
        skin_gap=skin_gap, surface_distance_limit=surface_distance_limit,
        log=log,
    )
    if len(P) == 0:
        return (np.empty((0, 3), np.float32), np.empty((0, 3), np.int32),
                0, np.empty(0, np.float32))
    V, F, VS = polyline_tubes(P, NL, valid, speeds, width=float(width), sides=int(sides))
    return V, F, len(P), VS
