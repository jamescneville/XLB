"""
FluidX3D-style realtime visualization for XLB multires (Neon) simulations.

Design
------
All rendering happens in Warp compute kernels operating on data that is already
in VRAM; the only host transfer per frame is the finished RGB image (a few MB).
Nothing here touches a graphics API from the GPU side, which matters because
CUDA/OpenGL interop is *not supported on WSL* -- any renderer that pushes
geometry or buffers through GL there silently falls back to host copies.

Per frame the pipeline is:

  1. ``sim.macro(...)``                  Neon f -> Neon rho/u   (callers operator)
  2. staging containers, coarse->fine   Neon u/bc_mask -> one uniform dense grid
  3. ``_q_kernel``                      dense u -> (Q, |omega|) on the same grid
  4. ``_brick_kernel``                  8^3 max-of-Q grid for empty-space skipping
  5. ``_render_kernel``                 two-level DDA raymarch of the Q isosurface,
                                        composited against the body triangle mesh
  6. ``wp.copy`` to a pinned host image, handed to the display thread

Step 2 flattens the Neon levels into a single uniform grid rather than one dense
box per level. Warp kernels cannot take a ragged list of arrays, and the scatter
direction makes the level merge trivial: a cell at level ``l`` covers ``2**l``
finest-grid units, so it writes the render cells it overlaps, and running the
levels coarsest-first lets finer data overwrite coarser data.

Two consequences shape what you see:

* Q is evaluated at render resolution, so vortices thinner than a render cell
  are decimated away. This is a live monitor, not an analysis tool.
* Velocity is discontinuous across a refinement interface, and a central
  difference taken across that seam manufactures enormous artificial Q -- it
  renders as a clean plane at every refinement box boundary. So each render cell
  records which level supplied it (see the ``flags`` layout below) and Q is only
  evaluated where the whole stencil came from one level.
* A level coarser than the render grid is replicated across a block of render
  cells, so an adjacent-cell difference would be zero inside a block and the
  whole jump concentrated (and amplified) at block faces -- grid-aligned slabs.
  ``_q_kernel`` therefore strides its stencil by the source cell size, landing
  on genuinely adjacent source cells. Every level is rendered; the coarse ones
  simply resolve less. Note the fine levels are the *small* boxes at the body,
  so excluding coarse levels would leave almost nothing to look at.

Render grid resolution is chosen from a memory budget; see ``_pick_render_level``.

Usage
-----
    view = LiveView(
        grid_shape_finest=grid_shape_zip,
        num_levels=actual_num_levels,
        body_vertices=[body_vertices, *wheel_vertices],
        u_ref=ulb,
    )
    ...
    for step in range(num_steps):
        sim.step()
        view.maybe_render(sim, step=step, mlups=MLUPS)
    view.close()

``maybe_render`` is wall-clock gated so the solver overhead is a fixed fraction
of runtime rather than something that scales with grid size.

Cost
----
Measured on an RTX A6000, 500x200x150 render grid (what the sizer picks for a
2000x800x600 car domain at the default 512 MB budget), with 1.2% of the volume
above the iso level:

    1280x720   q 2.0 ms | brick 0.3 ms | raymarch 2.7 ms | image copy 0.4 ms
    1920x1080  q 1.9 ms | brick 0.3 ms | raymarch 4.8 ms | image copy 0.7 ms

So roughly 5 ms and 8 ms per frame respectively, excluding the Neon staging
containers and ``sim.macro``. Overhead is then simply ``fps * frame_cost``: at
the default 10 fps and 720p that is about 5% of GPU time. Raise ``fps`` for a
smoother picture, lower it when you care about wall clock.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import queue
import threading
import time
from typing import Any, Optional, Sequence

import numpy as np
import warp as wp

# Sentinel written into the Q field wherever Q cannot be evaluated (inactive
# cell, boundary-adjacent cell, or grid edge). Any comparison against a real
# iso level fails, so these samples are simply never crossed.
_Q_INVALID = -1.0e30

# Render cells per brick, per axis, for the empty-space-skipping grid.
_BRICK = 8

# Layout of the per-render-cell ``flags`` byte.
#
# 0 means the cell was never written this frame. Otherwise the low nibble holds
# (source Neon level + 1) and bit 7 marks a boundary cell. Recording *which*
# level supplied each cell is what lets the Q kernel refuse to difference across
# a refinement interface: the velocity is discontinuous there, and differencing
# across the seam manufactures enormous artificial Q that renders as a perfect
# plane at every level boundary.
_LEVEL_MASK = 0x0F
_BOUNDARY_BIT = 0x80
_MAX_ENCODABLE_LEVELS = _LEVEL_MASK - 1  # level+1 must fit the nibble

# The only bc_mask value that means "not fluid". Mirrored from xlb.cell_type so
# this module stays importable without pulling in the package (the tests load it
# by path); asserted against the real one in LiveView.__init__.
_BC_SOLID = 255


# ---------------------------------------------------------------------------
# Colormap
# ---------------------------------------------------------------------------


@wp.func
def _turbo(t: float) -> wp.vec3:
    """The turbo colormap, 5th-order polynomial fit. ``t`` is clamped to [0,1].

    The fit tracks turbo closely across the middle of the range but drifts at the
    endpoints -- at t=0 it returns a dark grey rather than turbo's dark blue. That
    is invisible here because the isosurface is only ever coloured well inside the
    range, and it buys a colormap with no texture lookup and no host-side table.
    """
    x = wp.clamp(t, 0.0, 1.0)
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x4 * x

    r = 0.13572138 + 4.61539260 * x - 42.66032258 * x2 + 132.13108234 * x3 - 152.94239396 * x4 + 59.28637943 * x5
    g = 0.09140261 + 2.19418839 * x + 4.84296658 * x2 - 14.18503333 * x3 + 4.27729857 * x4 + 2.82956604 * x5
    b = 0.10667330 + 12.64194608 * x - 60.58204836 * x2 + 110.36276771 * x3 - 89.90310912 * x4 + 27.34824973 * x5

    return wp.vec3(wp.clamp(r, 0.0, 1.0), wp.clamp(g, 0.0, 1.0), wp.clamp(b, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Field kernels
# ---------------------------------------------------------------------------


@wp.kernel
def _q_kernel(
    vel: wp.array4d(dtype=wp.float32),
    flags: wp.array3d(dtype=wp.uint8),
    qv: wp.array4d(dtype=wp.float32),
    inv_u_ref: float,
    render_exp: int,
):
    """Q-criterion and vorticity magnitude from the staged velocity grid.

    Both are computed from velocity normalised by the reference (inlet) speed and
    differenced per render cell, so they are dimensionless and the iso level is
    comparable across levels, grid resolutions and unit systems.

    The stencil is **strided by the source cell size**. A cell from a level
    coarser than the render grid was replicated across a block of ``d`` render
    cells per axis, so an adjacent-cell difference reads two copies of the same
    value: zero everywhere inside a block, and the entire jump concentrated at
    block faces, amplified by ``d``. That is what renders as grid-aligned slabs.
    Stepping ``d`` cells instead lands on genuinely adjacent source cells, giving
    the correct coarse gradient smoothly across the whole block. Dividing by
    ``2*d`` keeps the result per render-cell-length, so one iso level stays
    meaningful across every level at once.
    """
    i, j, k = wp.tid()

    nx = flags.shape[0]
    ny = flags.shape[1]
    nz = flags.shape[2]

    # Q is only meaningful in fluid whose whole stencil is also fluid and came
    # from the same refinement level.
    #
    # Differencing across a wall or into an unwritten cell manufactures huge
    # spurious vorticity that would shroud the body in noise. Differencing across
    # a *level interface* is just as bad and less obvious: velocity is
    # discontinuous between levels, so the seam renders as a clean plane at every
    # refinement box boundary.
    c = flags[i, j, k]
    if c == wp.uint8(0) or (c & wp.uint8(_BOUNDARY_BIT)) != wp.uint8(0):
        qv[0, i, j, k] = float(_Q_INVALID)
        qv[1, i, j, k] = 0.0
        return

    lvl = c & wp.uint8(_LEVEL_MASK)

    # Nibble holds level+1, so the source level is lvl-1. Levels at or finer than
    # the render grid were decimated onto single cells and stride 1.
    d = 1 << wp.max(int(lvl) - 1 - render_exp, 0)

    if i - d < 0 or j - d < 0 or k - d < 0 or i + d >= nx or j + d >= ny or k + d >= nz:
        qv[0, i, j, k] = float(_Q_INVALID)
        qv[1, i, j, k] = 0.0
        return

    n_xp = flags[i + d, j, k]
    n_xm = flags[i - d, j, k]
    n_yp = flags[i, j + d, k]
    n_ym = flags[i, j - d, k]
    n_zp = flags[i, j, k + d]
    n_zm = flags[i, j, k - d]

    if n_xp != lvl or n_xm != lvl or n_yp != lvl or n_ym != lvl or n_zp != lvl or n_zm != lvl:
        # An exact match also rules out unwritten (0) and boundary (bit 7 set)
        # neighbours, since neither can equal a bare level nibble.
        qv[0, i, j, k] = float(_Q_INVALID)
        qv[1, i, j, k] = 0.0
        return

    s = 0.5 * inv_u_ref / float(d)

    dudx = (vel[0, i + d, j, k] - vel[0, i - d, j, k]) * s
    dudy = (vel[0, i, j + d, k] - vel[0, i, j - d, k]) * s
    dudz = (vel[0, i, j, k + d] - vel[0, i, j, k - d]) * s
    dvdx = (vel[1, i + d, j, k] - vel[1, i - d, j, k]) * s
    dvdy = (vel[1, i, j + d, k] - vel[1, i, j - d, k]) * s
    dvdz = (vel[1, i, j, k + d] - vel[1, i, j, k - d]) * s
    dwdx = (vel[2, i + d, j, k] - vel[2, i - d, j, k]) * s
    dwdy = (vel[2, i, j + d, k] - vel[2, i, j - d, k]) * s
    dwdz = (vel[2, i, j, k + d] - vel[2, i, j, k - d]) * s

    # Vorticity magnitude, used for colouring the isosurface.
    wx = dwdy - dvdz
    wy = dudz - dwdx
    wz = dvdx - dudy
    qv[1, i, j, k] = wp.sqrt(wx * wx + wy * wy + wz * wz)

    # Q = 0.5 * (|Omega|^2 - |S|^2), with both Frobenius norms written out from
    # the symmetric / antisymmetric parts of the velocity gradient.
    sxy = 0.5 * (dudy + dvdx)
    sxz = 0.5 * (dudz + dwdx)
    syz = 0.5 * (dvdz + dwdy)
    oxy = 0.5 * (dudy - dvdx)
    oxz = 0.5 * (dudz - dwdx)
    oyz = 0.5 * (dvdz - dwdy)

    s_norm = dudx * dudx + dvdy * dvdy + dwdz * dwdz + 2.0 * (sxy * sxy + sxz * sxz + syz * syz)
    o_norm = 2.0 * (oxy * oxy + oxz * oxz + oyz * oyz)

    qv[0, i, j, k] = 0.5 * (o_norm - s_norm)


@wp.kernel
def _brick_kernel(
    qv: wp.array4d(dtype=wp.float32),
    brick: wp.array3d(dtype=wp.float32),
):
    """Per-brick maximum of Q, for empty-space skipping in the raymarcher.

    A brick whose maximum is below the iso level cannot contain the isosurface,
    so the ray steps straight over it. Vortex cores occupy a small fraction of a
    windtunnel domain, which is where nearly all of the raymarch speedup comes
    from.
    """
    bi, bj, bk = wp.tid()

    nx = qv.shape[1]
    ny = qv.shape[2]
    nz = qv.shape[3]

    i0 = bi * _BRICK
    j0 = bj * _BRICK
    k0 = bk * _BRICK

    # Overlap neighbouring bricks by one cell: a trilinear sample taken inside
    # this brick reads cells from the next one, so a non-overlapping max could
    # report "empty" for a brick whose interpolated field does cross the iso.
    i1 = wp.min(i0 + _BRICK + 1, nx)
    j1 = wp.min(j0 + _BRICK + 1, ny)
    k1 = wp.min(k0 + _BRICK + 1, nz)

    m = float(_Q_INVALID)
    for i in range(i0, i1):
        for j in range(j0, j1):
            for k in range(k0, k1):
                m = wp.max(m, qv[0, i, j, k])

    brick[bi, bj, bk] = m


# ---------------------------------------------------------------------------
# Sampling / geometry helpers
# ---------------------------------------------------------------------------


@wp.func
def _sample(qv: wp.array4d(dtype=wp.float32), card: int, p: wp.vec3) -> float:
    """Trilinear sample of ``qv[card]`` at render-cell coordinate ``p``.

    Cell centres sit at integer+0.5, so the interpolation lattice is offset by
    half a cell. For ``card == 0`` (Q) the sample returns ``_Q_INVALID`` if any
    of the eight corners is invalid, which keeps the isosurface from leaking
    into masked regions.
    """
    nx = qv.shape[1]
    ny = qv.shape[2]
    nz = qv.shape[3]

    x = p[0] - 0.5
    y = p[1] - 0.5
    z = p[2] - 0.5

    i0 = int(wp.floor(x))
    j0 = int(wp.floor(y))
    k0 = int(wp.floor(z))

    if i0 < 0 or j0 < 0 or k0 < 0 or i0 >= nx - 1 or j0 >= ny - 1 or k0 >= nz - 1:
        return float(_Q_INVALID)

    fx = x - float(i0)
    fy = y - float(j0)
    fz = z - float(k0)

    c000 = qv[card, i0, j0, k0]
    c100 = qv[card, i0 + 1, j0, k0]
    c010 = qv[card, i0, j0 + 1, k0]
    c110 = qv[card, i0 + 1, j0 + 1, k0]
    c001 = qv[card, i0, j0, k0 + 1]
    c101 = qv[card, i0 + 1, j0, k0 + 1]
    c011 = qv[card, i0, j0 + 1, k0 + 1]
    c111 = qv[card, i0 + 1, j0 + 1, k0 + 1]

    if card == 0:
        lo = wp.min(
            wp.min(wp.min(c000, c100), wp.min(c010, c110)),
            wp.min(wp.min(c001, c101), wp.min(c011, c111)),
        )
        if lo <= float(_Q_INVALID):
            return float(_Q_INVALID)

    c00 = c000 + (c100 - c000) * fx
    c10 = c010 + (c110 - c010) * fx
    c01 = c001 + (c101 - c001) * fx
    c11 = c011 + (c111 - c011) * fx
    c0 = c00 + (c10 - c00) * fy
    c1 = c01 + (c11 - c01) * fy

    return c0 + (c1 - c0) * fz


@wp.func
def _q_normal(qv: wp.array4d(dtype=wp.float32), p: wp.vec3) -> wp.vec3:
    """Isosurface normal as the negated gradient of Q (Q decreases outward).

    Where a neighbour sample falls outside the valid region the difference
    degrades to one-sided rather than differencing against the invalid
    sentinel, which would otherwise dominate the gradient completely.
    """
    h = 1.0
    guard = float(_Q_INVALID) * 0.5

    c = _sample(qv, 0, p)

    xp = _sample(qv, 0, p + wp.vec3(h, 0.0, 0.0))
    xm = _sample(qv, 0, p - wp.vec3(h, 0.0, 0.0))
    yp = _sample(qv, 0, p + wp.vec3(0.0, h, 0.0))
    ym = _sample(qv, 0, p - wp.vec3(0.0, h, 0.0))
    zp = _sample(qv, 0, p + wp.vec3(0.0, 0.0, h))
    zm = _sample(qv, 0, p - wp.vec3(0.0, 0.0, h))

    if xp <= guard:
        xp = c
    if xm <= guard:
        xm = c
    if yp <= guard:
        yp = c
    if ym <= guard:
        ym = c
    if zp <= guard:
        zp = c
    if zm <= guard:
        zm = c

    g = wp.vec3(-(xp - xm), -(yp - ym), -(zp - zm))
    n = wp.length(g)
    if n < 1.0e-20:
        return wp.vec3(0.0, 0.0, 1.0)
    return g / n


@wp.func
def _slab(o: wp.vec3, d: wp.vec3, lo: wp.vec3, hi: wp.vec3) -> wp.vec2:
    """Ray/AABB intersection. Returns (t_enter, t_exit); empty when t_enter > t_exit.

    Written with a single exit rather than an early return inside the loop,
    which warp code generation handles more reliably.
    """
    # Explicit constructors: warp treats a bare literal as a compile-time
    # constant and refuses to let one be mutated inside a loop.
    t0 = float(0.0)
    t1 = float(1.0e30)
    miss = bool(False)

    for a in range(3):
        if wp.abs(d[a]) < 1.0e-12:
            # Ray is parallel to this pair of planes: either always inside the
            # slab or never, with no t interval to intersect.
            if o[a] < lo[a] or o[a] > hi[a]:
                miss = True
        else:
            inv = 1.0 / d[a]
            ta = (lo[a] - o[a]) * inv
            tb = (hi[a] - o[a]) * inv
            if ta > tb:
                tmp = ta
                ta = tb
                tb = tmp
            t0 = wp.max(t0, ta)
            t1 = wp.min(t1, tb)

    if miss:
        return wp.vec2(1.0, -1.0)
    return wp.vec2(t0, t1)


@wp.func
def _shade(n: wp.vec3, view: wp.vec3, base: wp.vec3, ambient: float, specular: float) -> wp.vec3:
    """Headlight Blinn-Phong: the light rides with the camera, so nothing the
    camera can see is ever unlit. Two-sided, since isosurface normals may point
    either way relative to the ray."""
    nn = wp.normalize(n)
    v = -wp.normalize(view)
    if wp.dot(nn, v) < 0.0:
        nn = -nn

    ndv = wp.max(wp.dot(nn, v), 0.0)
    spec = specular * wp.pow(ndv, 32.0)

    lit = wp.cw_mul(base, wp.vec3(ambient + (1.0 - ambient) * ndv))
    return lit + wp.vec3(spec, spec, spec)


# ---------------------------------------------------------------------------
# Raymarch kernel
# ---------------------------------------------------------------------------


@wp.kernel
def _render_kernel(
    qv: wp.array4d(dtype=wp.float32),
    brick: wp.array3d(dtype=wp.float32),
    fb: wp.array3d(dtype=wp.uint8),
    mesh_id: wp.uint64,
    has_mesh: int,
    eye: wp.vec3,
    fwd: wp.vec3,
    right: wp.vec3,
    up: wp.vec3,
    tan_half_fov: float,
    q_iso: float,
    w_max: float,
    step_cells: float,
    body_color: wp.vec3,
    bg_top: wp.vec3,
    bg_bottom: wp.vec3,
    ambient: float,
    specular: float,
    gamma: float,
):
    py, px = wp.tid()

    height = fb.shape[0]
    width = fb.shape[1]

    # --- primary ray ---------------------------------------------------------
    aspect = float(width) / float(height)
    sx = (2.0 * (float(px) + 0.5) / float(width) - 1.0) * aspect * tan_half_fov
    sy = (1.0 - 2.0 * (float(py) + 0.5) / float(height)) * tan_half_fov

    rd = wp.normalize(fwd + right * sx + up * sy)
    ro = eye

    # Background gradient, used wherever nothing is hit.
    bg_t = 1.0 - (float(py) + 0.5) / float(height)
    color = bg_bottom + (bg_top - bg_bottom) * bg_t
    hit = False

    # --- static body geometry ------------------------------------------------
    # The BVH is built once at startup, so this is just a traversal. Its hit
    # distance also caps the volume march below, which is what makes the
    # isosurface correctly disappear behind the body.
    t_far = 1.0e30
    if has_mesh != 0:
        query = wp.mesh_query_ray(mesh_id, ro, rd, t_far)
        if query.result:
            t_far = query.t
            color = _shade(query.normal, rd, body_color, ambient, specular)
            hit = True

    # --- volume bounds -------------------------------------------------------
    nx = float(qv.shape[1])
    ny = float(qv.shape[2])
    nz = float(qv.shape[3])

    span = _slab(ro, rd, wp.vec3(0.0, 0.0, 0.0), wp.vec3(nx, ny, nz))
    t_enter = wp.max(span[0], 0.0)
    t_exit = wp.min(span[1], t_far)

    if t_enter <= t_exit:
        bnx = brick.shape[0]
        bny = brick.shape[1]
        bnz = brick.shape[2]
        inv_b = 1.0 / float(_BRICK)

        # --- brick-level DDA (Amanatides & Woo) ------------------------------
        t = t_enter + 1.0e-4
        p = ro + rd * t

        bi = wp.clamp(int(wp.floor(p[0] * inv_b)), 0, bnx - 1)
        bj = wp.clamp(int(wp.floor(p[1] * inv_b)), 0, bny - 1)
        bk = wp.clamp(int(wp.floor(p[2] * inv_b)), 0, bnz - 1)

        step_i = int(1)
        step_j = int(1)
        step_k = int(1)
        if rd[0] < 0.0:
            step_i = int(-1)
        if rd[1] < 0.0:
            step_j = int(-1)
        if rd[2] < 0.0:
            step_k = int(-1)

        # Explicit constructors again: next_* are advanced inside the DDA loop,
        # so they must not start life as compile-time constants. An axis the ray
        # is parallel to keeps its huge value and is never selected as nearest.
        dt_i = float(1.0e30)
        dt_j = float(1.0e30)
        dt_k = float(1.0e30)
        next_i = float(1.0e30)
        next_j = float(1.0e30)
        next_k = float(1.0e30)

        # Distance along the ray to the far side of the current brick, per axis.
        # ``bi + max(step,0)`` picks the +x face when travelling in +x and the
        # -x face when travelling in -x.
        if wp.abs(rd[0]) > 1.0e-12:
            dt_i = float(_BRICK) / wp.abs(rd[0])
            next_i = t + (float(bi + wp.max(step_i, 0)) * float(_BRICK) - p[0]) / rd[0]
        if wp.abs(rd[1]) > 1.0e-12:
            dt_j = float(_BRICK) / wp.abs(rd[1])
            next_j = t + (float(bj + wp.max(step_j, 0)) * float(_BRICK) - p[1]) / rd[1]
        if wp.abs(rd[2]) > 1.0e-12:
            dt_k = float(_BRICK) / wp.abs(rd[2])
            next_k = t + (float(bk + wp.max(step_k, 0)) * float(_BRICK) - p[2]) / rd[2]

        found = bool(False)
        t_surf = float(0.0)

        while t < t_exit and not found:
            t_leave = wp.min(wp.min(next_i, next_j), next_k)
            seg_end = wp.min(t_leave, t_exit)

            if brick[bi, bj, bk] >= q_iso:
                # --- fine march inside an occupied brick ---------------------
                ts = t
                prev = _sample(qv, 0, ro + rd * ts)
                while ts < seg_end and not found:
                    tn = wp.min(ts + step_cells, seg_end)
                    cur = _sample(qv, 0, ro + rd * tn)

                    # ``prev`` must be valid, not just below the iso level: the
                    # invalid sentinel is below every threshold, so accepting it
                    # would paint a hard surface along every mask edge (body
                    # skin, solid interior, domain rim) instead of only where
                    # the flow genuinely crosses the isosurface.
                    if prev > float(_Q_INVALID) * 0.5 and prev < q_iso and cur >= q_iso:
                        # Bisect the bracketed crossing. Four iterations put the
                        # surface within step/16 of a cell, well below what a
                        # pixel can resolve.
                        a = ts
                        b = tn
                        for _ in range(4):
                            m = 0.5 * (a + b)
                            if _sample(qv, 0, ro + rd * m) >= q_iso:
                                b = m
                            else:
                                a = m
                        t_surf = 0.5 * (a + b)
                        found = True

                    prev = cur
                    ts = tn

            if not found:
                # Advance to the next brick along the smallest crossing distance.
                if t_leave >= t_exit:
                    t = t_exit
                elif next_i <= next_j and next_i <= next_k:
                    t = next_i
                    bi += step_i
                    next_i += dt_i
                    if bi < 0 or bi >= bnx:
                        t = t_exit
                elif next_j <= next_k:
                    t = next_j
                    bj += step_j
                    next_j += dt_j
                    if bj < 0 or bj >= bny:
                        t = t_exit
                else:
                    t = next_k
                    bk += step_k
                    next_k += dt_k
                    if bk < 0 or bk >= bnz:
                        t = t_exit

        if found:
            p_surf = ro + rd * t_surf
            n = _q_normal(qv, p_surf)
            w = _sample(qv, 1, p_surf)
            base = _turbo(w / wp.max(w_max, 1.0e-20))
            color = _shade(n, rd, base, ambient, specular)
            hit = True

    if hit:
        inv_gamma = 1.0 / gamma
        color = wp.vec3(
            wp.pow(wp.clamp(color[0], 0.0, 1.0), inv_gamma),
            wp.pow(wp.clamp(color[1], 0.0, 1.0), inv_gamma),
            wp.pow(wp.clamp(color[2], 0.0, 1.0), inv_gamma),
        )

    fb[py, px, 0] = wp.uint8(wp.clamp(color[0], 0.0, 1.0) * 255.0)
    fb[py, px, 1] = wp.uint8(wp.clamp(color[1], 0.0, 1.0) * 255.0)
    fb[py, px, 2] = wp.uint8(wp.clamp(color[2], 0.0, 1.0) * 255.0)


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------


class Camera:
    """Orbit camera in render-cell coordinates.

    Azimuth is measured from +x (streamwise) toward +y (lateral); elevation is
    measured from the xy-plane toward +z (up), matching the XLB box convention
    where left/right is x, front/back is y and bottom/top is z.
    """

    def __init__(self, target, distance, azimuth=35.0, elevation=20.0, fov=45.0):
        self.target = np.asarray(target, dtype=np.float64)
        self.distance = float(distance)
        self.azimuth = float(azimuth)
        self.elevation = float(elevation)
        self.fov = float(fov)

    def basis(self):
        """Return (eye, forward, right, up) as float32 vec3s."""
        az = math.radians(self.azimuth)
        # Clamped so right = cross(forward, world_up) never degenerates.
        el = math.radians(max(-85.0, min(85.0, self.elevation)))

        d = np.array(
            [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)],
            dtype=np.float64,
        )
        eye = self.target + self.distance * d
        fwd = -d

        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(fwd, world_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)

        return (
            wp.vec3(*eye.astype(np.float32)),
            wp.vec3(*fwd.astype(np.float32)),
            wp.vec3(*right.astype(np.float32)),
            wp.vec3(*up.astype(np.float32)),
        )


# ---------------------------------------------------------------------------
# Display backends
# ---------------------------------------------------------------------------


class _ViewerState:
    """Camera / iso state shared between the solver thread and the display thread.

    The solver thread only ever reads it and the display thread only ever writes
    it, so a plain lock around small assignments is sufficient.
    """

    def __init__(self, camera):
        self.lock = threading.Lock()
        self.camera = camera
        self.iso_scale = 1.0
        self.recalibrate = False
        self.save_png = False
        self.closed = False
        self.hud = ""


class _PygletDisplay:
    """Threaded pyglet window that blits the RGB framebuffer.

    Only the finished image is uploaded, so this needs nothing more than a
    texture blit -- no CUDA/OpenGL interop, which is what makes it work under
    WSLg where interop is unavailable.
    """

    def __init__(self, width, height, title, state):
        import pyglet

        self._pyglet = pyglet
        self.width = width
        self.height = height
        self.title = title
        self.state = state

        self._frame = None
        self._frame_lock = threading.Lock()
        self._ready = threading.Event()
        self._error = None

        self._thread = threading.Thread(target=self._run, name="xlb-live-view", daemon=True)
        self._thread.start()

        # Surface a window-creation failure to the caller rather than leaving a
        # dead thread and a silently black viewer.
        self._ready.wait(timeout=20.0)
        if self._error is not None:
            raise self._error

    def _run(self):
        pyglet = self._pyglet
        try:
            window = pyglet.window.Window(
                width=self.width,
                height=self.height,
                caption=self.title,
                resizable=False,
                vsync=False,
            )
            label = pyglet.text.Label(
                "",
                font_name="monospace",
                font_size=11,
                x=8,
                y=self.height - 8,
                anchor_x="left",
                anchor_y="top",
                color=(255, 255, 255, 220),
                multiline=True,
                width=self.width - 16,
            )
        except Exception as exc:  # noqa: BLE001
            self._error = exc
            self._ready.set()
            return

        self._ready.set()
        drag = {"active": False}

        @window.event
        def on_draw():
            window.clear()
            with self._frame_lock:
                buf = self._frame
            if buf is not None:
                # pitch is negative because the framebuffer is stored top-down
                # while OpenGL expects bottom-up rows.
                img = pyglet.image.ImageData(self.width, self.height, "RGB", buf, pitch=-self.width * 3)
                img.blit(0, 0)
            with self.state.lock:
                label.text = self.state.hud
            label.draw()

        @window.event
        def on_mouse_press(x, y, button, modifiers):
            drag["active"] = True

        @window.event
        def on_mouse_release(x, y, button, modifiers):
            drag["active"] = False

        @window.event
        def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
            with self.state.lock:
                self.state.camera.azimuth -= dx * 0.4
                self.state.camera.elevation += dy * 0.4

        @window.event
        def on_mouse_scroll(x, y, sx, sy):
            with self.state.lock:
                self.state.camera.distance *= 0.9**sy

        @window.event
        def on_key_press(symbol, modifiers):
            key = pyglet.window.key
            with self.state.lock:
                if symbol == key.BRACKETLEFT:
                    self.state.iso_scale /= 1.3
                elif symbol == key.BRACKETRIGHT:
                    self.state.iso_scale *= 1.3
                elif symbol == key.R:
                    self.state.recalibrate = True
                elif symbol == key.P:
                    self.state.save_png = True
                elif symbol == key.ESCAPE:
                    self.state.closed = True

        @window.event
        def on_close():
            with self.state.lock:
                self.state.closed = True

        # A no-op tick keeps the window responsive and repainting even while the
        # solver thread is busy and no new frame has arrived.
        def _tick(_dt):
            window.invalid = True

        pyglet.clock.schedule_interval(_tick, 1.0 / 60.0)
        pyglet.app.run()

        with self.state.lock:
            self.state.closed = True

    def show(self, frame_bytes):
        with self._frame_lock:
            self._frame = frame_bytes

    def close(self):
        # The window may already be gone (user closed it), which is not an error.
        with contextlib.suppress(Exception):
            self._pyglet.app.exit()


class _PngDisplay:
    """Fallback that writes a numbered PNG per frame.

    Used when no window can be opened (headless node, container without a
    display, no pyglet). The GPU pipeline is identical; only the transport
    differs.

    Compression runs on worker threads. PNG-encoding a 720p frame costs tens of
    milliseconds, which is several times the whole GPU render -- doing it inline
    would make the *viewer* the dominant cost of the run. The queue is short and
    drops frames rather than growing, so a slow disk throttles the movie instead
    of the solve.

    Several workers because an interactive draw rate outruns a single encoder.
    They finish out of order, so ``completed`` tracks the newest index that is
    actually on disk -- a viewer told to load a frame still being written would
    just flicker. ``keep_frames`` prunes behind them: at 30 fps a numbered
    sequence reaches gigabytes within minutes.
    """

    def __init__(self, width, height, output_dir, state, queue_depth=4, workers=3, keep_frames=900):
        from PIL import Image

        self._Image = Image
        self.width = width
        self.height = height
        self.output_dir = output_dir
        self.state = state
        self.index = 0
        self.dropped = 0
        self.completed = -1
        self.keep_frames = int(keep_frames)

        os.makedirs(output_dir, exist_ok=True)

        self._done_lock = threading.Lock()
        self._queue = queue.Queue(maxsize=queue_depth)
        self._threads = [threading.Thread(target=self._worker, name=f"xlb-live-view-png-{i}", daemon=True) for i in range(max(1, workers))]
        for t in self._threads:
            t.start()

    def _path(self, index):
        return os.path.join(self.output_dir, f"live_{index:06d}.png")

    def _worker(self):
        while True:
            item = self._queue.get()
            if item is None:
                return

            index, buf = item
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(self.height, self.width, 3)
            path = self._path(index)
            try:
                # compress_level=1 rather than the default 6: at 720p that is
                # roughly a 4x faster encode for ~20% larger files, which is the
                # right trade for a frame sequence nobody archives.
                self._Image.fromarray(arr).save(path, compress_level=1)
            except Exception as exc:  # noqa: BLE001
                print(f"[live_view] failed to write {path}: {exc}")
                continue

            with self._done_lock:
                # Workers finish out of order; only ever advance.
                if index > self.completed:
                    self.completed = index
                stale = index - self.keep_frames

            if self.keep_frames > 0 and stale >= 0:
                with contextlib.suppress(OSError):
                    os.remove(self._path(stale))

    def show(self, frame_bytes):
        try:
            self._queue.put_nowait((self.index, frame_bytes))
            self.index += 1
        except queue.Full:
            self.dropped += 1

    def close(self):
        for _ in self._threads:
            self._queue.put(None)
        for t in self._threads:
            t.join(timeout=30.0)
        if self.dropped:
            print(f"[live_view] wrote {self.index} PNG frames, dropped {self.dropped} (disk could not keep up)")
        else:
            print(f"[live_view] wrote {self.index} PNG frames to {self.output_dir}/")


# ---------------------------------------------------------------------------
# Render-grid sizing
# ---------------------------------------------------------------------------

# vel (3 x f32) + qv (2 x f32) + flags (1 x u8) per render cell.
_BYTES_PER_RENDER_CELL = 3 * 4 + 2 * 4 + 1


def _pick_render_level(grid_shape_finest, budget_bytes):
    """Smallest power-of-two decimation ``r`` whose render grid fits the budget.

    ``r`` must be a power of two so that mapping a Neon cell at level ``l`` onto
    render cells stays exact integer arithmetic in both directions.
    """
    shape = np.asarray(grid_shape_finest, dtype=np.int64)
    for r in range(0, 12):
        rshape = np.maximum((shape + (1 << r) - 1) >> r, 1)
        if int(np.prod(rshape)) * _BYTES_PER_RENDER_CELL <= budget_bytes:
            return r, tuple(int(v) for v in rshape)
    raise ValueError(f"Cannot fit a render grid for {tuple(shape)} into {budget_bytes} bytes")


# ---------------------------------------------------------------------------
# LiveView
# ---------------------------------------------------------------------------


class LiveView:
    """Realtime Q-criterion isosurface viewer for a multires Neon simulation.

    Parameters
    ----------
    grid_shape_finest
        Full domain shape in finest-level cells (``grid_shape_zip`` in the
        windtunnel examples). Neon global indices are expressed in these units.
    num_levels
        Number of refinement levels; level 0 is the finest.
    body_vertices
        One or more triangle-soup vertex arrays of shape ``(3*ntri, 3)`` in
        finest-lattice units -- exactly what the mesh boundary conditions are
        given. Concatenated into a single static ``wp.Mesh``.
    u_ref
        Reference (inlet) lattice velocity, used to non-dimensionalise Q.
    memory_budget_mb
        Cap on render-grid VRAM. Drives the render resolution via
        ``_pick_render_level``; the default keeps a car-scale domain at roughly
        the coarsest level.
    fps
        Wall-clock cap on the *data* refresh rate: macro + staging + Q, which
        needs the solver's current state and so drains its dispatch queue. This
        is what bounds the viewer's cost to the solve.
    view_fps
        Wall-clock cap on camera-only redraws, which reuse the resident Q field
        and run on a private stream. These cost a couple of milliseconds and do
        not interact with the solver, so this can be high. 0 means unthrottled.
    heartbeat
        Redraw at least this often (seconds) even with no camera movement, so the
        published frame and its reported age stay current. Cheap: one draw.
    max_lag_steps
        Cap on how many solver steps the host may run ahead of the device, by
        syncing every N steps. This is the main lever on how quickly the
        isosurface can update: without it the host fills CUDA's launch queue and
        a data refresh has to wait out seconds of queued work. Lower is more
        responsive and slightly slower in MLUPS; 0 disables the sync entirely and
        leaves the solver's dispatch exactly as it was.
    min_step_interval
        Never render more often than this many solver steps, whatever fps says.
    max_level
        Coarsest Neon level to stage; default is all of them. Lower it only to
        deliberately crop to the finer levels near the body -- excluded levels
        render as empty space.
    mask_solid
        Read ``bc_mask`` and refuse to evaluate Q on solid cells. Set False to
        skip the field entirely: Q is then evaluated everywhere data exists,
        which is a useful escape hatch if the coverage line reports an
        implausible solid percentage. The body mesh occludes its own interior
        anyway, so the main loss is some noise at walls.
    iso_brick_fraction
        Target fraction of occupied bricks left above the iso level, in
        (0, 0.5]. The level is set to the matching percentile of the per-brick
        max-Q distribution, so this directly controls how much surface is drawn
        -- and therefore how fast the raymarch is. Lower shows only the
        strongest cores; higher shows more structure and costs more.
    recalibrate_every
        Re-derive the iso level every N frames. Must be >0 for a startup
        transient: Q at step 0 is orders of magnitude below a developed wake, so
        a threshold frozen there saturates as the flow spins up. Set to 0 to
        calibrate once and hold (then press R in the viewer to redo it).
    iso_smoothing
        Exponential smoothing on tracked iso updates, in [0,1]. Low values
        follow the flow slowly and steadily; 1.0 snaps to each new estimate.
    output_dir
        Where the PNG fallback and manual snapshots are written.
    """

    def __init__(
        self,
        grid_shape_finest: Sequence[int],
        num_levels: int,
        body_vertices: Optional[Sequence[np.ndarray]] = None,
        u_ref: float = 0.05,
        width: int = 1280,
        height: int = 720,
        fps: float = 10.0,
        view_fps: float = 30.0,
        heartbeat: float = 0.5,
        max_lag_steps: int = 8,
        min_step_interval: int = 1,
        memory_budget_mb: float = 512.0,
        max_level: Optional[int] = None,
        mask_solid: bool = True,
        iso_brick_fraction: float = 0.05,
        recalibrate_every: int = 1,
        iso_smoothing: float = 0.2,
        step_cells: float = 0.75,
        fov: float = 45.0,
        backend: str = "auto",
        output_dir: str = "live_view",
        title: str = "XLB live view - Q criterion",
        body_color: Sequence[float] = (0.72, 0.72, 0.75),
        bg_top: Sequence[float] = (0.05, 0.06, 0.09),
        bg_bottom: Sequence[float] = (0.16, 0.17, 0.20),
        ambient: float = 0.18,
        specular: float = 0.25,
        gamma: float = 2.2,
        verbose: bool = True,
    ):
        self.grid_shape_finest = tuple(int(v) for v in grid_shape_finest)
        self.num_levels = int(num_levels)
        self.u_ref = float(u_ref)
        self.width = int(width)
        self.height = int(height)
        self.min_interval = 1.0 / float(fps) if fps > 0 else 0.0
        self.min_view_interval = 1.0 / float(view_fps) if view_fps > 0 else 0.0
        self.heartbeat = float(heartbeat)
        self.max_lag_steps = int(max_lag_steps)
        self.min_step_interval = max(1, int(min_step_interval))
        self.iso_brick_fraction = float(np.clip(iso_brick_fraction, 1.0e-4, 0.5))
        self.recalibrate_every = int(recalibrate_every)
        self.iso_smoothing = float(np.clip(iso_smoothing, 0.0, 1.0))
        self.step_cells = float(step_cells)
        self.output_dir = output_dir
        self.title = title
        self.verbose = bool(verbose)

        self.body_color = wp.vec3(*[float(v) for v in body_color])
        self.bg_top = wp.vec3(*[float(v) for v in bg_top])
        self.bg_bottom = wp.vec3(*[float(v) for v in bg_bottom])
        self.ambient = float(ambient)
        self.specular = float(specular)
        self.gamma = float(gamma)

        # --- render grid -----------------------------------------------------
        self.r, self.rshape = _pick_render_level(self.grid_shape_finest, memory_budget_mb * 1024 * 1024)
        self.rdiv = 1 << self.r
        nx, ny, nz = self.rshape

        self._vel = wp.zeros((3, nx, ny, nz), dtype=wp.float32)
        self._qv = wp.zeros((2, nx, ny, nz), dtype=wp.float32)
        self._flags = wp.zeros((nx, ny, nz), dtype=wp.uint8)

        # Stage every level by default. Coarse levels carry the wake and far
        # field -- in a 6-level car case the *finest* levels are tiny boxes right
        # at the body, so excluding coarse ones leaves almost nothing to render.
        # Their blockiness is handled in _q_kernel by striding the stencil to the
        # source cell size rather than by dropping them.
        self.mask_solid = bool(mask_solid)
        self.max_level = (self.num_levels - 1) if max_level is None else int(max_level)
        if self.num_levels > _MAX_ENCODABLE_LEVELS:
            # The flags nibble holds level+1, so very deep hierarchies would
            # alias distinct levels onto the same code and defeat the interface
            # masking. Clamp rather than silently mis-mask.
            self.max_level = min(self.max_level, _MAX_ENCODABLE_LEVELS - 1)

        self.bshape = tuple((v + _BRICK - 1) // _BRICK for v in self.rshape)
        self._brick = wp.zeros(self.bshape, dtype=wp.float32)

        self._fb = wp.zeros((self.height, self.width, 3), dtype=wp.uint8)
        self._host = wp.zeros((self.height, self.width, 3), dtype=wp.uint8, device="cpu")

        # --- static body mesh ------------------------------------------------
        self._mesh = self._build_mesh(body_vertices)
        self._has_mesh = 1 if self._mesh is not None else 0
        self._mesh_id = self._mesh.id if self._mesh is not None else wp.uint64(0)

        # --- camera ----------------------------------------------------------
        lo, hi = self._body_bounds(body_vertices)
        camera = Camera(
            target=0.5 * (lo + hi),
            # Frame the body plus roughly a body length of wake behind it.
            distance=2.5 * float(np.linalg.norm(hi - lo)),
            fov=fov,
        )
        self.state = _ViewerState(camera)

        # --- iso level -------------------------------------------------------
        self.q_iso = 0.0
        self.w_max = 1.0
        self._needs_calibration = True

        # --- lazily built Neon plumbing --------------------------------------
        # Deferred so importing this module does not require neon, and so the
        # staging kernels compile against the fields the first frame actually
        # hands us rather than ones guessed at construction time.
        self._containers = None
        self._neon = None
        self._scatter_n = None

        # --- display ---------------------------------------------------------
        self.display = self._make_display(backend)

        # --- filesystem control bridge ---------------------------------------
        # When frames go to disk rather than a window, the same directory doubles
        # as a two-way channel: we publish status.json and read camera.json back.
        # A viewer running outside this process -- crucially, outside this
        # *container* -- can then drive the camera with no network, no X11 and no
        # OpenGL, as long as it can see the directory. See tools/live_view_viewer.py.
        if isinstance(self.display, _PngDisplay):
            self._camera_path = os.path.join(output_dir, "camera.json")
            self._status_path = os.path.join(output_dir, "status.json")
            if self.verbose:
                print(f"[live_view] control bridge: {self._status_path} / {self._camera_path}")
        else:
            self._camera_path = None
            self._status_path = None

        self._camera_mtime = None
        self._recalibrate_seq = 0

        # Cheap guard against _BC_SOLID drifting from the real definition. Import
        # here rather than at module scope so this file stays loadable by path.
        from xlb.cell_type import BC_SOLID

        assert int(BC_SOLID) == _BC_SOLID, f"_BC_SOLID ({_BC_SOLID}) no longer matches xlb.cell_type.BC_SOLID ({BC_SOLID})"

        self._last_render = 0.0
        self._last_data = 0.0
        self._last_step = -(10**9)
        self._last_sync = -(10**9)
        self._frames = 0
        self._refreshes = 0
        self._render_ms = 0.0
        self._draw_ms = 0.0
        self._data_ms = 0.0
        self._drain_ms = 0.0
        self._macro_ms = 0.0
        self._stage_ms = 0.0
        self._q_ms = 0.0
        self.enabled = True

        # Private stream for camera-only redraws. It reads only arrays this class
        # owns, so it can overlap solver work instead of queueing behind it --
        # that independence is what makes dragging feel immediate. CPU-only warp
        # builds have no streams, hence the guard.
        try:
            self._stream = wp.Stream() if wp.get_device().is_cuda else None
        except Exception as exc:  # noqa: BLE001
            self._stream = None
            if self.verbose:
                print(f"[live_view] no private stream ({exc}); draws will share the default stream")

        if self.verbose:
            cells = nx * ny * nz
            mb = cells * _BYTES_PER_RENDER_CELL / (1024.0 * 1024.0)
            staged = min(self.max_level, self.num_levels - 1)
            print(
                f"[live_view] render grid {nx}x{ny}x{nz} "
                f"(decimation 2^{self.r} of {self.grid_shape_finest}), "
                f"{cells / 1e6:.1f}M cells, {mb:.0f} MB, "
                f"levels 0-{staged} of 0-{self.num_levels - 1} staged, "
                f"display={type(self.display).__name__}"
            )
            if staged < self.num_levels - 1:
                print(f"[live_view] levels {staged + 1}-{self.num_levels - 1} excluded by max_level; those regions will be blank")

    @property
    def last_render_seconds(self):
        """Wall-clock cost of the most recent frame, excluding solver backlog.

        Callers that measure solver throughput can add this to the start of
        their timing interval so viewer overhead is not charged to MLUPS. It is
        safe to do so precisely because render() drains the async solver queue
        before it starts timing -- see the comment there.
        """
        return self._render_ms / 1000.0

    # -- setup helpers -------------------------------------------------------

    def _build_mesh(self, body_vertices):
        """Concatenate the triangle soups into one static ``wp.Mesh``.

        The BVH is built here, once. Vertices arrive in finest-lattice units and
        are scaled into render-cell space so the mesh and the volume share a
        single coordinate system.
        """
        soups = self._as_soups(body_vertices)
        if not soups:
            return None

        verts = np.concatenate([s.astype(np.float32) for s in soups], axis=0) / np.float32(self.rdiv)

        # Trailing vertices that do not complete a triangle would index past the
        # end of the soup, so drop them rather than let wp.Mesh read garbage.
        ntri = verts.shape[0] // 3
        verts = verts[: ntri * 3]

        return wp.Mesh(
            points=wp.array(verts, dtype=wp.vec3),
            indices=wp.array(np.arange(ntri * 3, dtype=np.int32), dtype=wp.int32),
        )

    @staticmethod
    def _as_soups(body_vertices):
        """Normalise the body_vertices argument to a list of non-empty arrays."""
        if body_vertices is None:
            return []
        if isinstance(body_vertices, np.ndarray):
            body_vertices = [body_vertices]
        return [np.asarray(v) for v in body_vertices if v is not None and len(v) > 0]

    def _body_bounds(self, body_vertices):
        """Body bounding box in render-cell coordinates, or the domain box."""
        soups = self._as_soups(body_vertices)
        if soups:
            allv = np.concatenate(soups, axis=0) / float(self.rdiv)
            return allv.min(axis=0), allv.max(axis=0)

        return np.zeros(3), np.asarray(self.rshape, dtype=np.float64)

    def _make_display(self, backend):
        if backend in ("auto", "pyglet"):
            try:
                return _PygletDisplay(self.width, self.height, self.title, self.state)
            except Exception as exc:  # noqa: BLE001
                if backend == "pyglet":
                    raise
                if self.verbose:
                    print(f"[live_view] no window ({exc}); writing PNG frames to {self.output_dir}/ instead")

        return _PngDisplay(self.width, self.height, self.output_dir, self.state)

    # -- Neon staging --------------------------------------------------------

    def _build_containers(self, u_field, bc_mask_field):
        """Build one staging container per level, ordered coarsest to finest.

        Building a container compiles a kernel, so these are built once here and
        re-launched every frame. The coarsest-first ordering is what makes finer
        levels win wherever they overlap coarser ones.
        """
        import neon

        self._neon = neon
        nx, ny, nz = self.rshape
        rdiv = self.rdiv
        use_mask = bc_mask_field is not None
        # Captured python ints are compile-time constants to warp, so this
        # branch is resolved during codegen rather than per cell.
        mask_flag = 1 if use_mask else 0

        @neon.Container.factory(name="LiveViewStage")
        def factory(u_neon: Any, m_neon: Any, dst_vel: Any, dst_flags: Any, level: Any, cells_per_axis: Any, level_code: Any):
            def launcher(loader: neon.Loader):
                loader.set_mres_grid(u_neon.get_grid(), level)
                u_hdl = loader.get_mres_read_handle(u_neon)
                m_hdl = loader.get_mres_read_handle(m_neon)

                # The scatter bound must be read from a device array, not captured
                # as a python int: warp fully unrolls range() over a compile-time
                # constant, and a coarse level can cover 16 or 32 render cells per
                # axis -- a 4096- or 32768-body unrolled triple loop. Reading it
                # back forces a real dynamic loop instead.
                n_arr = cells_per_axis

                @wp.func
                def kernel(index: Any):
                    n = n_arr[0]
                    cIdx = wp.neon_global_idx(u_hdl, index)

                    # Global indices are in finest-level units; a cell at level l
                    # covers [g, g + 2^l), which maps onto n render cells per axis.
                    rx = wp.neon_get_x(cIdx) // rdiv
                    ry = wp.neon_get_y(cIdx) // rdiv
                    rz = wp.neon_get_z(cIdx) // rdiv

                    ux = wp.float32(wp.neon_read(u_hdl, index, 0))
                    uy = wp.float32(wp.neon_read(u_hdl, index, 1))
                    uz = wp.float32(wp.neon_read(u_hdl, index, 2))

                    # Low nibble = source level + 1, bit 7 = solid cell.
                    #
                    # Only BC_SOLID counts. Every other nonzero bc_mask value is a
                    # boundary-condition *id* sitting on a perfectly good fluid
                    # cell -- inlet, outlet, moving walls, ground, body surface --
                    # and the solver itself only ever tests against BC_SOLID
                    # (see nse_multires_stepper.prepare_coalescence_count, and
                    # MultiresIO._solid_mask_from_fields_data). Treating nonzero
                    # as "boundary" rejected 86% of the domain and left Q valid
                    # nowhere at all.
                    fl = level_code
                    if mask_flag == 1:
                        if wp.neon_read(m_hdl, index, 0) == wp.uint8(_BC_SOLID):
                            fl = level_code | wp.uint8(_BOUNDARY_BIT)

                    for a in range(n):
                        x = rx + a
                        if x < nx:
                            for b in range(n):
                                y = ry + b
                                if y < ny:
                                    for c in range(n):
                                        z = rz + c
                                        if z < nz:
                                            dst_vel[0, x, y, z] = ux
                                            dst_vel[1, x, y, z] = uy
                                            dst_vel[2, x, y, z] = uz
                                            dst_flags[x, y, z] = fl

                loader.declare_kernel(kernel)

            return launcher

        containers = []
        # Kept alive on self: these back the dynamic loop bounds above, so they
        # must outlive this function for as long as the containers are launched.
        self._scatter_n = []

        # Coarsest first, so finer levels overwrite where boxes overlap.
        #
        # Levels coarser than the render grid are skipped entirely: scattering a
        # coarse cell over a block of render cells gives a piecewise-constant
        # field whose gradient is meaningless, and it renders as grid-aligned
        # blocks and stair-steps rather than flow structure. Rendering Q only
        # where the simulation actually resolves the render grid also means the
        # picture covers the refined region -- body and wake -- and leaves the
        # under-resolved far field empty, which is what you want to look at.
        for level in range(min(self.max_level, self.num_levels - 1), -1, -1):
            n = 1 << max(0, level - self.r)
            n_arr = wp.array([n], dtype=wp.int32)
            self._scatter_n.append(n_arr)

            containers.append(
                factory(
                    u_field,
                    bc_mask_field if use_mask else u_field,
                    self._vel,
                    self._flags,
                    level,
                    n_arr,
                    wp.uint8(level + 1),
                )
            )

        return containers

    # -- filesystem control bridge -------------------------------------------

    def _poll_camera(self):
        """Apply camera state written by an external viewer, if it changed.

        Gated on mtime so the common case is a single stat() per frame. A partial
        read is expected rather than exceptional -- the viewer replaces the file
        atomically, but a bind mount to another OS does not always make that
        visible atomically -- so a parse failure just resets the mtime and waits
        for the next frame instead of propagating.
        """
        if self._camera_path is None:
            return False

        try:
            mtime = os.path.getmtime(self._camera_path)
        except OSError:
            return False

        if mtime == self._camera_mtime:
            return False

        try:
            with open(self._camera_path) as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            self._camera_mtime = None
            return False

        self._camera_mtime = mtime

        with self.state.lock:
            camera = self.state.camera
            for key in ("azimuth", "elevation", "distance", "fov"):
                if key in data:
                    setattr(camera, key, float(data[key]))

            if "iso_scale" in data:
                self.state.iso_scale = float(data["iso_scale"])

            # A sequence number rather than a boolean: the viewer has no way to
            # observe when a flag was consumed, so it counts up instead.
            seq = int(data.get("recalibrate_seq", 0))
            if seq != self._recalibrate_seq:
                self._recalibrate_seq = seq
                self.state.recalibrate = True

            if data.get("closed"):
                self.state.closed = True

        return True

    def _write_status(self, step, mlups, iso_scale):
        """Publish what the viewer needs: newest frame, HUD numbers, camera echo.

        Written via a temp file plus replace so a viewer never reads a half-file.
        """
        if self._status_path is None:
            return

        camera = self.state.camera
        payload = {
            "frame": self.display.completed,
            "frame_file": f"live_{self.display.completed:06d}.png",
            "dropped": self.display.dropped,
            "step": int(step),
            "mlups": float(mlups),
            "render_ms": round(self._render_ms, 2),
            "draw_ms": round(self._draw_ms, 2),
            "data_ms": round(self._data_ms, 2),
            "drain_ms": round(self._drain_ms, 2),
            "q_iso": self.q_iso * iso_scale,
            "iso_scale": iso_scale,
            "width": self.width,
            "height": self.height,
            "render_shape": list(self.rshape),
            "azimuth": camera.azimuth,
            "elevation": camera.elevation,
            "distance": camera.distance,
            "fov": camera.fov,
            "time": time.time(),
        }

        tmp = self._status_path + ".tmp"
        try:
            with open(tmp, "w") as fh:
                json.dump(payload, fh)
            os.replace(tmp, self._status_path)
        except OSError as exc:
            # Losing a status update is cosmetic; never let it break the solve.
            if self.verbose:
                print(f"[live_view] could not write status: {exc}")

    # -- iso calibration -----------------------------------------------------

    def _calibrate(self, force=False):
        """Track the iso level from the brick-max grid.

        The brick grid is a 512x reduction of the render grid, so this host copy
        is tiny -- a few thousand floats -- and gives a robust picture of the Q
        distribution without needing a device-side reduction (Neon exposes none).

        The level is a **percentile** of that distribution, chosen so only
        ``iso_brick_fraction`` of occupied bricks sit above it. Scaling some
        fraction of the *peak* instead does not work: nothing ties the peak to
        where the field's mass lies, so the level can land below the field floor
        entirely. That is the pathological case -- the crossing test in the
        raymarcher needs a sample *below* the level to bracket against, so a
        too-low level draws nothing at all while making every ray traverse the
        whole volume. Blank screen, worst-case cost. Anchoring to a percentile
        makes occupancy the controlled quantity, which bounds both.

        This has to keep tracking rather than calibrate once. Q at step 0 is the
        initial transient, orders of magnitude below a developed wake, so a
        one-shot threshold taken there is wrong within a few hundred steps.
        Updates are smoothed so the surface breathes instead of flickering;
        ``force`` (the R key, or the first frame) snaps straight to the target.
        """
        b = self._brick.numpy().ravel()

        # Only rotation-dominated bricks are candidates. Q <= 0 means strain
        # dominates, so those bricks hold no vortex and including them just drags
        # the percentile down into a region the isosurface can never occupy --
        # in a windtunnel most of the domain is quiet freestream, so they would
        # otherwise dominate the distribution completely.
        valid = b[(b > _Q_INVALID * 0.5) & (b > 0.0)]
        if valid.size == 0:
            # No rotation anywhere yet (very early in a startup). Hold the
            # previous level rather than set one that cannot bracket a crossing.
            return

        pct = 100.0 * (1.0 - self.iso_brick_fraction)
        target = float(np.percentile(valid, pct))

        if not np.isfinite(target) or target <= 0.0:
            return

        if force or self._needs_calibration:
            self.q_iso = target
        else:
            a = self.iso_smoothing
            self.q_iso = (1.0 - a) * self.q_iso + a * target

        # In a coherent vortex Q ~ 0.5|omega|^2, so sqrt(2*Q_iso) is the vorticity
        # at the isosurface itself; a few times that spans the visible range.
        self.w_max = 4.0 * math.sqrt(2.0 * max(self.q_iso, 0.0))

        first = self._needs_calibration
        self._needs_calibration = False

        # Only announce the initial lock-on and explicit recalibrations; the
        # per-frame tracking updates would otherwise flood the solver log.
        if self.verbose and (first or force):
            occupied = float(np.count_nonzero(valid >= self.q_iso)) / float(valid.size)
            print(
                f"[live_view] calibrated q_iso={self.q_iso:.4g} "
                f"(P{pct:.1f} of {valid.size} valid bricks, peak {valid.max():.4g}), "
                f"{occupied * 100.0:.1f}% of bricks occupied, w_max={self.w_max:.4g}"
            )

    def _report_coverage(self):
        """One-shot breakdown of which levels actually reached the render grid.

        Worth the single 25 MB readback: if a level contributes nothing, or Q
        ends up valid almost nowhere, the picture is empty and every other
        number in the log still looks reasonable. That failure is otherwise
        invisible -- it cost a couple of runs to spot.
        """
        flags = self._flags.numpy().ravel()
        total = flags.size

        written = np.count_nonzero(flags)
        solid = np.count_nonzero(flags & _BOUNDARY_BIT)
        levels = (flags & _LEVEL_MASK).astype(np.int32) - 1

        parts = []
        for level in range(self.num_levels):
            n = int(np.count_nonzero((levels == level) & (flags != 0)))
            if n:
                parts.append(f"L{level}:{n / 1e3:.0f}k")

        q = self._qv.numpy()[0].ravel()
        valid_q = int(np.count_nonzero(q > _Q_INVALID * 0.5))

        print(
            f"[live_view] coverage: {written / total * 100.0:.1f}% of render cells written "
            f"({', '.join(parts) if parts else 'none'}), {solid / max(written, 1) * 100.0:.1f}% solid, "
            f"Q valid in {valid_q / total * 100.0:.1f}% of cells"
        )
        if valid_q * 200 < total:
            print(
                "[live_view] WARNING: Q is valid in under 0.5% of the grid, so the isosurface will be "
                "nearly empty. A high solid percentage above means bc_mask is rejecting the domain; "
                "otherwise check max_level and memory_budget_mb."
            )

    # -- rendering -----------------------------------------------------------

    def maybe_render(self, sim, step: int, mlups: float = 0.0, force: bool = False, **hud):
        """Advance the viewer, doing as little as the situation needs.

        Two rates, because the two halves of a frame cost wildly different
        amounts:

        * **Data refresh** (``fps``) runs ``sim.macro`` plus the Neon staging
          containers and recomputes Q. It needs the solver's current state, so it
          must drain the async dispatch queue -- typically the dominant cost, and
          the reason a naive design updates roughly once a second.
        * **Draw** (``view_fps``) only re-runs the raymarch over the Q field that
          is already resident, on a private CUDA stream. It touches nothing the
          solver owns, so it neither waits for the queue nor blocks it.

        Dragging the camera therefore costs a couple of milliseconds and shows up
        immediately, instead of waiting out a queue drain. Called every step, so
        the effective draw rate is bounded by how often the host loops.

        The draw is checked **first**, and deliberately so. A refresh can easily
        cost more than its own requested period, which makes it perpetually
        "due"; testing the refresh first then starves the cheap path entirely and
        the camera goes back to feeling like it updates once a second.
        """
        if not self.enabled:
            return False

        with self.state.lock:
            if self.state.closed:
                self._shutdown()
                return False

        # Cheap: one stat() unless the file actually changed.
        camera_moved = self._poll_camera()

        now = time.perf_counter()
        acted = False

        # 1) Camera-only redraw, before anything that can block. A heartbeat
        #    keeps the published frame (and its age) current even when nobody is
        #    dragging, for a couple of milliseconds a second.
        since_draw = now - self._last_render
        if since_draw >= self.min_view_interval and (camera_moved or since_draw >= self.heartbeat):
            self._draw(step, mlups, hud)
            acted = True

        # 2) Bound how far the host runs ahead of the device.
        #
        #    sim.step() never syncs, so the host enqueues until CUDA's launch
        #    queue is full -- hundreds of steps, seconds of work. A data refresh
        #    needs current state and so must wait that out, which is what sets
        #    the floor on how often the isosurface can update. Syncing every few
        #    steps keeps the queue shallow, trading a little pipelining for a
        #    much shorter drain. Set max_lag_steps=0 to leave dispatch untouched.
        if self.max_lag_steps > 0 and (step - self._last_sync) >= self.max_lag_steps:
            wp.synchronize()
            self._last_sync = step

        # 3) Data refresh: new physics, and the expensive half.
        if force or ((now - self._last_data) >= self.min_interval and (step - self._last_step) >= self.min_step_interval):
            self._refresh_data(sim)
            self._last_data = time.perf_counter()
            self._last_step = step
            self._draw(step, mlups, hud)
            acted = True

        return acted

    def render(self, sim, step: int = 0, mlups: float = 0.0, **hud):
        """Force a full data refresh and draw. Kept for explicit one-shot use."""
        self._refresh_data(sim)
        self._last_data = time.perf_counter()
        self._last_step = step
        self._draw(step, mlups, hud)
        return True

    def _refresh_data(self, sim):
        """Pull current state from the solver into the Q field. Expensive half."""
        if self._containers is None:
            bc_mask = getattr(sim, "bc_mask", None) if self.mask_solid else None
            self._containers = self._build_containers(sim.u, bc_mask)

        # Drain the solver backlog *before* starting the clock.
        #
        # sim.step() dispatches asynchronously and deliberately does not sync, so
        # the host runs far ahead of the device -- by seconds on a large case.
        # Everything below queues behind that backlog. Timing from here would
        # report mostly solver time as "render time", which is not just a
        # cosmetic lie: callers subtract last_render_seconds from their
        # throughput interval, so it would inflate their reported MLUPS.
        #
        # The drain itself is unavoidable -- rendering the current state means
        # waiting for the current state -- but it belongs to the solver, not here.
        t_pre = time.perf_counter()
        wp.synchronize()
        t0 = time.perf_counter()

        sim.macro(sim.f_0, sim.bc_mask, sim.rho, sim.u, streamId=0)
        wp.synchronize()
        t_macro = time.perf_counter()

        # Cells not written this frame must read as invalid, otherwise the
        # isosurface would be built partly from stale data.
        self._flags.zero_()

        for container in self._containers:
            container.run(0, container_runtime=self._neon.Container.ContainerRuntime.neon)
        wp.synchronize()
        t_stage = time.perf_counter()

        wp.launch(_q_kernel, dim=self.rshape, inputs=[self._vel, self._flags, self._qv, 1.0 / self.u_ref, self.r])
        wp.launch(_brick_kernel, dim=self.bshape, inputs=[self._qv, self._brick])
        wp.synchronize()
        t_q = time.perf_counter()

        self._drain_ms = (t0 - t_pre) * 1000.0
        self._macro_ms = (t_macro - t0) * 1000.0
        self._stage_ms = (t_stage - t_macro) * 1000.0
        self._q_ms = (t_q - t_stage) * 1000.0
        # Includes the drain: that wait is what sets the refresh period, even
        # though the work itself belongs to the solver.
        self._data_ms = (t_q - t_pre) * 1000.0

        with self.state.lock:
            recalibrate = self.state.recalibrate
            self.state.recalibrate = False

        first = self._needs_calibration
        due = self.recalibrate_every > 0 and self._refreshes % self.recalibrate_every == 0
        if first or recalibrate or due:
            self._calibrate(force=recalibrate)

        self._refreshes += 1

        if first and self.verbose:
            self._report_coverage()

        # Report the breakdown on the first refresh *and* a few later ones. The
        # first is dominated by warp/Neon JIT compilation -- it read 13 s once,
        # almost all of it compiling, which says nothing about where the
        # steady-state cost is and sent an optimisation effort the wrong way.
        if self.verbose and self._refreshes in (1, 5, 25):
            label = "first, includes JIT compile" if self._refreshes == 1 else "steady state"
            print(
                f"[live_view] data refresh {self._data_ms:.0f} ms "
                f"(queue drain {self._drain_ms:.0f}, macro {self._macro_ms:.0f}, "
                f"stage {self._stage_ms:.0f}, Q {self._q_ms:.0f}) [{label}]"
            )

    def _draw(self, step, mlups, hud):
        """Raymarch the resident Q field and publish. Cheap half.

        Runs on a private stream so it is independent of the solver's queue in
        both directions: it does not wait behind pending solver work, and
        synchronising it does not force that work to complete early.
        """
        t0 = time.perf_counter()

        with self.state.lock:
            camera = self.state.camera
            iso_scale = self.state.iso_scale
            save_png = self.state.save_png
            self.state.save_png = False

        eye, fwd, right, up = camera.basis()

        inputs = [
            self._qv,
            self._brick,
            self._fb,
            self._mesh_id,
            self._has_mesh,
            eye,
            fwd,
            right,
            up,
            math.tan(math.radians(camera.fov) * 0.5),
            self.q_iso * iso_scale,
            self.w_max,
            self.step_cells,
            self.body_color,
            self.bg_top,
            self.bg_bottom,
            self.ambient,
            self.specular,
            self.gamma,
        ]

        if self._stream is not None:
            # Note: pass stream= explicitly rather than using wp.ScopedStream.
            # ScopedStream synchronizes on *entry* by default, which makes the
            # new stream wait for everything already queued on the default one --
            # exactly the solver backlog we are trying to step around. Measured:
            # 293 ms via ScopedStream vs 3.6 ms with an explicit stream, behind
            # an identical 400-launch backlog.
            wp.launch(_render_kernel, dim=(self.height, self.width), inputs=inputs, stream=self._stream)
            # The only per-frame device->host traffic: one RGB image.
            wp.copy(self._host, self._fb, stream=self._stream)
            wp.synchronize_stream(self._stream)
        else:
            wp.launch(_render_kernel, dim=(self.height, self.width), inputs=inputs)
            wp.copy(self._host, self._fb)
            wp.synchronize()

        frame = self._host.numpy().tobytes()

        self._frames += 1
        self._draw_ms = (time.perf_counter() - t0) * 1000.0
        self._render_ms = self._draw_ms
        self._last_render = time.perf_counter()

        with self.state.lock:
            self.state.hud = self._hud_text(step, mlups, iso_scale, hud)

        self.display.show(frame)

        if save_png and not isinstance(self.display, _PngDisplay):
            self._snapshot(frame, step)

        # After show(), so frame_file names a frame that has at least been queued.
        self._write_status(step, mlups, iso_scale)

    def _hud_text(self, step, mlups, iso_scale, extra):
        lines = [
            f"step {step}   MLUPS {mlups:.1f}   draw {self._draw_ms:.0f} ms   data {self._data_ms:.0f} ms",
            f"q_iso {self.q_iso * iso_scale:.4g}   grid {self.rshape[0]}x{self.rshape[1]}x{self.rshape[2]}",
        ]
        if extra:
            lines.append("   ".join(f"{k} {v}" for k, v in extra.items()))
        lines.append("drag orbit | wheel zoom | [ ] iso | R recalibrate | P snapshot | Esc close")
        return "\n".join(lines)

    def _snapshot(self, frame, step):
        from PIL import Image

        os.makedirs(self.output_dir, exist_ok=True)
        arr = np.frombuffer(frame, dtype=np.uint8).reshape(self.height, self.width, 3)
        path = os.path.join(self.output_dir, f"snapshot_{step:07d}.png")
        Image.fromarray(arr).save(path)
        if self.verbose:
            print(f"[live_view] wrote {path}")

    # -- teardown ------------------------------------------------------------

    def _shutdown(self):
        """Release the render grids so the solver gets the VRAM and full GPU back."""
        if not self.enabled:
            return

        self.enabled = False
        self.display.close()

        self._vel = None
        self._qv = None
        self._flags = None
        self._brick = None
        self._fb = None
        self._host = None
        self._containers = None
        self._scatter_n = None
        self._stream = None

        if self.verbose:
            print(f"[live_view] closed after {self._frames} frames, {self._refreshes} data refreshes")

    def close(self):
        with self.state.lock:
            self.state.closed = True
        self._shutdown()
