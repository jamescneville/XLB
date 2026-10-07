"""Surface visualisations derived from the time-averaged flow: drag/lift contribution, separation index, wall streamlines.

Pure numpy/scipy (no Neon / Warp). Flow is +x, up is +z. The surface is the remeshed export mesh with outward
vertex normals. ``sample_velocity(points, base_points, normals) -> (vel (N,3), ok (N,))`` is supplied by the caller.
"""

import numpy as np
from scipy.spatial import cKDTree

from xlb.utils.tufts import poisson_roots, local_surface, _unit


# --------------------------------------------------------------------------------------------------------------
# Drag / lift contribution
# --------------------------------------------------------------------------------------------------------------
_AXIS = {"drag": 0, "lift": 2}


def force_density(cp, normals, component="drag"):
    """
    Pressure force per unit area (in units of q_inf) along the flow (``"drag"``, +x) or up (``"lift"``, +z).

    The body feels F = -p n dA with n pointing out of the body, so the density is -Cp * n_component:
    positive = pushes the car back (drag) / up (lift); negative = pulls it forward / presses it down.
    Front-facing high-pressure faces and rear-facing suction faces both read as drag.
    """
    if component not in _AXIS:
        raise ValueError(f"component must be 'drag' or 'lift', got {component!r}")
    n = np.asarray(normals, dtype=np.float64)
    return (-np.asarray(cp, dtype=np.float64) * n[:, _AXIS[component]]).astype(np.float32)


def integrate_over_surface(density, vertices, faces):
    """Area integral of a per-vertex density (mean of the three corners per triangle). Returns (integral, area)."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    tri = v[f]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    mean = np.asarray(density, dtype=np.float64)[f].mean(axis=1)
    return float((area * mean).sum()), float(area.sum())


# --------------------------------------------------------------------------------------------------------------
# Separation index
# --------------------------------------------------------------------------------------------------------------
def separation_index(v_near, v_outer, slow_fraction=0.05):
    """
    Backflow indicator in [-1, 1] from the near-wall velocity against the velocity a little further out:
    cos(angle between them), faded to 0 where the near-wall flow is almost stagnant (direction meaningless).
    +1 = attached (near-wall flow follows the outer flow), 0 = dead / stagnant, -1 = reversed (separated).
    """
    a = np.asarray(v_near, dtype=np.float64)
    b = np.asarray(v_outer, dtype=np.float64)
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    cos = np.einsum("ij,ij->i", a, b) / np.maximum(na * nb, 1e-20)
    fade = np.clip(na / np.maximum(slow_fraction * nb, 1e-20), 0.0, 1.0)
    out = np.where(nb > 1e-9, cos * fade, 0.0)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------------------------------------------
# Wall streamlines
# --------------------------------------------------------------------------------------------------------------
def _smooth_normals(V, F, N, iterations):
    """Steering normals: vertex normals averaged with their mesh neighbours ``iterations`` times (facet noise, creases)."""
    if iterations <= 0:
        return N
    from scipy.sparse import coo_matrix

    n = len(V)
    i = np.concatenate([F[:, 0], F[:, 1], F[:, 2], F[:, 1], F[:, 2], F[:, 0]])
    j = np.concatenate([F[:, 1], F[:, 2], F[:, 0], F[:, 0], F[:, 1], F[:, 2]])
    A = coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)).tocsr()
    A.data[:] = 1.0                                    # duplicates summed above: make it a 0/1 adjacency
    A = A.multiply(1.0 / np.maximum(np.asarray(A.sum(axis=1)).ravel(), 1.0)[:, None]).tocsr()
    out = N.copy()
    for _ in range(int(iterations)):
        out = _unit(0.5 * out + 0.5 * (A @ out), fallback=N)
    return out


def _trace(roots, nrm, sample_velocity, tree, V, N, *, sign, nsteps, step, probe_height, lift, turn_cos, vref=None,
           same_skin=True, steer=None, max_step_turn=None, skin_gap=0.001):
    T = len(roots)
    P = np.full((T, nsteps + 1, 3), np.nan)
    NL = np.zeros((T, nsteps + 1, 3))
    SP = np.zeros((T, nsteps + 1))
    valid = np.zeros((T, nsteps + 1), dtype=bool)
    p = roots + nrm * lift
    nl = nrm.copy()
    base = roots
    P[:, 0], NL[:, 0], valid[:, 0] = p, nl, True
    alive = np.ones(T, dtype=bool)
    prev = None
    steer = N if steer is None else steer
    for s in range(nsteps):
        probe = p + nl * max(probe_height - lift, 0.0)
        vel, ok = sample_velocity(probe, base, nl)
        vel = np.asarray(vel, dtype=np.float64)
        speed = np.linalg.norm(vel, axis=1)
        SP[:, s] = speed
        if vref is None:
            vref = float(np.percentile(speed[ok], 99)) if ok.any() else 1.0
        un = np.einsum("ij,ij->i", vel, nl)
        ut = vel - un[:, None] * nl
        um = np.linalg.norm(ut, axis=1)
        d = sign * ut / np.maximum(um, 1e-20)[:, None]
        stop = ~ok | (um < 1e-3 * vref)
        if prev is not None:
            stop |= np.einsum("ij,ij->i", d, prev) < turn_cos       # sharp reversal: convergence / stagnation line
        if prev is not None and max_step_turn is not None:
            # a joint may not swing the line more than max_step_turn from the last step (kills one-step sideways jogs)
            ang = np.arccos(np.clip(np.einsum("ij,ij->i", d, prev), -1.0, 1.0))
            a = np.minimum(1.0, max_step_turn / np.maximum(ang, 1e-9))
            d = _unit((1.0 - a)[:, None] * prev + a[:, None] * d, fallback=d)
        newp = p + d * step
        vi, w, v0 = local_surface(tree, V, N, newp, nl, same_skin, skin_gap)
        nl_new = _unit((steer[vi] * w[:, :, None]).sum(axis=1), fallback=nl)
        h = np.einsum("ij,ij->i", newp - V[v0], N[v0])
        newp = newp - nl_new * (h - lift)[:, None]
        stop |= np.linalg.norm(newp - p, axis=1) > 2.0 * step      # jumped onto another surface
        alive &= ~stop
        P[alive, s + 1] = newp[alive]
        NL[alive, s + 1] = nl_new[alive]
        valid[alive, s + 1] = True
        p = np.where(alive[:, None], newp, p)
        nl = np.where(alive[:, None], nl_new, nl)
        base = np.where(alive[:, None], V[v0], base)
        prev = np.where(alive[:, None], d, prev if prev is not None else d)
        if not alive.any():
            break
    SP[:, nsteps] = SP[:, nsteps - 1]
    return P, NL, SP, valid, vref


def wall_streamlines(vertices, faces, vertex_normals, sample_velocity, *, spacing, length, step, probe_height,
                     lift=0.001, turn_limit_deg=100.0, seed=0, max_lines=60000, min_joints=4, same_skin=True,
                     smooth_normals=3, max_step_turn_deg=30.0, skin_gap=0.001, log=print):
    """
    Streamlines that hug the surface and follow the near-wall flow, traced both ways from Poisson-spaced seeds.
    Returns (P (L,J,3), NL (L,J,3), speed (L,J), valid (L,J)); valid joints are contiguous round the seed.

    ``smooth_normals``: smoothing passes on the normals that steer the line (the tangent plane the flow is projected
    onto). ``max_step_turn_deg``: most a joint may turn the line from the previous step (None = unlimited).
    """
    V = np.asarray(vertices, dtype=np.float64)
    N = np.asarray(vertex_normals, dtype=np.float64)
    tree = cKDTree(V)
    rng = np.random.default_rng(seed)
    roots, nrm = poisson_roots(V, np.asarray(faces, dtype=np.int64), N, spacing, rng)
    if len(roots) > max_lines:
        sel = rng.choice(len(roots), size=max_lines, replace=False)
        roots, nrm = roots[sel], nrm[sel]
        log(f"\tWall streamlines: capped at {max_lines:,} seeds")
    log(f"\tWall streamlines: {len(roots):,} seeds at {spacing * 1000:.0f} mm spacing")
    nsteps = max(1, int(round(0.5 * length / step)))
    steer = _smooth_normals(V, np.asarray(faces, dtype=np.int64), N, smooth_normals)
    kw = dict(nsteps=nsteps, step=step, probe_height=probe_height, lift=lift, turn_cos=np.cos(np.radians(turn_limit_deg)),
              same_skin=same_skin, steer=steer, skin_gap=float(skin_gap),
              max_step_turn=None if max_step_turn_deg is None else np.radians(float(max_step_turn_deg)))
    Pf, NLf, SPf, vf, vref = _trace(roots, nrm, sample_velocity, tree, V, N, sign=+1.0, **kw)
    Pb, NLb, SPb, vb, _ = _trace(roots, nrm, sample_velocity, tree, V, N, sign=-1.0, vref=vref, **kw)
    P = np.concatenate([Pb[:, :0:-1], Pf], axis=1)
    NL = np.concatenate([NLb[:, :0:-1], NLf], axis=1)
    SP = np.concatenate([SPb[:, :0:-1], SPf], axis=1)
    valid = np.concatenate([vb[:, :0:-1], vf], axis=1)
    keep = valid.sum(axis=1) >= min_joints
    log(f"\tWall streamlines: {int(keep.sum()):,} kept (others stopped immediately: stagnant or reversing flow); "
        f"median {np.median(valid[keep].sum(axis=1)) * step * 1000:.0f} mm long")
    return P[keep], NL[keep], SP[keep], valid[keep]


def polyline_tubes(P, NL, valid, speed, width, sides=3):
    """
    Tubes along polylines with a per-joint validity mask. Returns (verts (V,3), faces (F,3), vertex_speed (V,)).
    Only vertices used by a valid segment are kept.
    """
    L, J, _ = P.shape
    Pf = P.copy()
    # invalid joints: copy the nearest valid joint so frames stay finite (their faces are dropped below)
    first = np.argmax(valid, axis=1)
    last = J - 1 - np.argmax(valid[:, ::-1], axis=1)
    ji = np.clip(np.arange(J)[None, :], first[:, None], last[:, None])
    Pf = np.take_along_axis(P, ji[:, :, None], axis=1)
    NLf = np.take_along_axis(NL, ji[:, :, None], axis=1)
    t = np.empty_like(Pf)
    t[:, 1:-1] = Pf[:, 2:] - Pf[:, :-2]
    t[:, 0] = Pf[:, 1] - Pf[:, 0]
    t[:, -1] = Pf[:, -1] - Pf[:, -2]
    t = _unit(t, fallback=np.array([1.0, 0.0, 0.0]))
    a1 = np.cross(t, NLf)
    a1 = _unit(a1, fallback=_unit(np.cross(t, np.array([0.0, 0.0, 1.0]))))
    a2 = np.cross(t, a1)
    r = 0.5 * float(width)
    ang = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    ring = Pf[:, :, None, :] + r * (np.cos(ang)[None, None, :, None] * a1[:, :, None, :]
                                    + np.sin(ang)[None, None, :, None] * a2[:, :, None, :])
    verts = ring.reshape(-1, 3)
    vspeed = np.repeat(np.take_along_axis(speed, ji, axis=1)[:, :, None], sides, axis=2).reshape(-1)

    local = []
    for j in range(sides):
        a, b = j, (j + 1) % sides
        local += [(a, b, a + sides), (b, b + sides, a + sides)]
    local = np.asarray(local, dtype=np.int64)                                     # one segment: joint 0 -> joint 1
    seg_ok = valid[:, :-1] & valid[:, 1:]                                         # (L, J-1)
    li, si = np.nonzero(seg_ok)
    faces = (local[None] + ((li * J + si) * sides)[:, None, None]).reshape(-1, 3)
    used, inv = np.unique(faces, return_inverse=True)
    return verts[used].astype(np.float32), inv.reshape(-1, 3).astype(np.int32), vspeed[used].astype(np.float32)
