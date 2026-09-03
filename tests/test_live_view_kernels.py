"""Codegen/behaviour tests for the xlb.utils.live_view Warp kernels.

These run on the CPU device and do not need Neon, CUDA or a display, so they
cover the part of the live viewer that is pure Warp: Q evaluation, the brick
reduction, and the two-level DDA raymarch. The Neon staging containers are not
exercised here -- they need a real mGrid.

The module is loaded directly from its path rather than via ``import xlb`` so
the test does not drag in jax/neon.
"""

import importlib.util
import math
import pathlib
import queue
import sys

import numpy as np
import pytest

wp = pytest.importorskip("warp")

_MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / "xlb" / "utils" / "live_view.py"


@pytest.fixture(scope="module")
def lv():
    spec = importlib.util.spec_from_file_location("xlb_live_view_under_test", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _devices():
    """Every device warp can reach. CUDA and CPU go through different compilers,
    so kernel codegen is worth checking on both when a GPU is present."""
    wp.init()
    return [str(d) for d in wp.get_devices()]


@pytest.fixture(scope="module", params=_devices())
def device(request):
    return request.param


def _solid_body_rotation(shape, omega):
    """Velocity field u = omega x r about the z axis through the grid centre.

    Solid-body rotation has zero strain, so Q = 0.5*|Omega|^2 = omega^2 and the
    vorticity magnitude is 2*omega everywhere -- an exact target to check the
    Q kernel against.
    """
    nx, ny, nz = shape
    cx, cy = 0.5 * nx, 0.5 * ny

    i, j, k = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    x = i + 0.5 - cx
    y = j + 0.5 - cy

    vel = np.zeros((3, nx, ny, nz), dtype=np.float32)
    vel[0] = -omega * y
    vel[1] = omega * x
    return vel


def _fluid_flags(shape, level=0, device=None):
    """flags byte for uniform fluid sourced from a single Neon level."""
    return wp.full(shape, wp.uint8(level + 1), dtype=wp.uint8, device=device)


def test_q_kernel_matches_solid_body_rotation(lv, device):
    shape = (24, 24, 12)
    omega = 0.01
    u_ref = 1.0  # so the normalisation is a no-op and we can compare directly

    vel = wp.array(_solid_body_rotation(shape, omega), dtype=wp.float32, device=device)
    flags = _fluid_flags(shape, level=0, device=device)
    qv = wp.zeros((2, *shape), dtype=wp.float32, device=device)

    wp.launch(lv._q_kernel, dim=shape, inputs=[vel, flags, qv, 1.0 / u_ref, 0], device=device)

    out = qv.numpy()
    interior = (slice(1, -1),) * 3

    np.testing.assert_allclose(out[0][interior], omega**2, rtol=1e-4)
    np.testing.assert_allclose(out[1][interior], 2.0 * omega, rtol=1e-4)


def test_q_kernel_strided_stencil_recovers_the_coarse_gradient(lv, device):
    """A coarse level replicated over blocks must still give the right gradient.

    This is the grid-aligned-slab bug. With a stride-1 stencil, a block-constant
    field differences to zero inside each block and dumps the whole jump onto
    block faces, amplified by the block size. Striding by the source cell size
    must instead reproduce the same Q as the unreplicated field.
    """
    omega = 0.01
    render_exp = 0
    source_level = 2
    d = 1 << (source_level - render_exp)  # 4 render cells per source cell

    fine_shape = (24, 24, 24)
    fine = _solid_body_rotation(fine_shape, omega)

    # Replicate each source cell across a d^3 block, exactly as the staging
    # container does for a level coarser than the render grid.
    coarse = np.repeat(np.repeat(np.repeat(fine, d, axis=1), d, axis=2), d, axis=3)
    shape = coarse.shape[1:]

    vel = wp.array(np.ascontiguousarray(coarse), dtype=wp.float32, device=device)
    flags = _fluid_flags(shape, level=source_level, device=device)
    qv = wp.zeros((2, *shape), dtype=wp.float32, device=device)

    wp.launch(lv._q_kernel, dim=shape, inputs=[vel, flags, qv, 1.0, render_exp], device=device)

    out = qv.numpy()
    guard = lv._Q_INVALID * 0.5
    interior = (slice(2 * d, -2 * d),) * 3

    q_in = out[0][interior]
    w_in = out[1][interior]
    assert np.all(q_in > guard), "strided stencil should leave the block interiors valid"

    # Solid-body rotation differenced at the source spacing, expressed per
    # render-cell length: omega is per source cell, so per render cell it is
    # omega/d, giving Q = (omega/d)^2 and |w| = 2*omega/d.
    np.testing.assert_allclose(q_in, (omega / d) ** 2, rtol=2e-3)
    np.testing.assert_allclose(w_in, 2.0 * omega / d, rtol=2e-3)

    # The decisive part: no grid-aligned structure. A stride-1 stencil would make
    # Q vary hugely between block interiors and block faces.
    assert q_in.std() / q_in.mean() < 1e-2, "Q should be smooth across blocks, not concentrated at faces"


def test_q_kernel_invalidates_boundary_neighbourhoods(lv, device):
    """A single non-fluid cell must invalidate its whole 6-neighbourhood."""
    shape = (12, 12, 12)

    vel = wp.array(_solid_body_rotation(shape, 0.01), dtype=wp.float32, device=device)

    flags_np = np.ones(shape, dtype=np.uint8)  # level 0 fluid everywhere
    flags_np[6, 6, 6] = 1 | lv._BOUNDARY_BIT  # boundary cell, still level 0
    flags_np[3, 3, 3] = 0  # never written
    flags = wp.array(flags_np, dtype=wp.uint8, device=device)

    qv = wp.zeros((2, *shape), dtype=wp.float32, device=device)
    wp.launch(lv._q_kernel, dim=shape, inputs=[vel, flags, qv, 1.0, 0], device=device)

    q = qv.numpy()[0]
    guard = lv._Q_INVALID * 0.5

    for cx, cy, cz in ((6, 6, 6), (3, 3, 3)):
        assert q[cx, cy, cz] < guard
        for d in range(3):
            for s in (-1, 1):
                idx = [cx, cy, cz]
                idx[d] += s
                assert q[tuple(idx)] < guard, f"neighbour {idx} of {(cx, cy, cz)} should be invalid"

    # The grid rim is invalid too, since central differences need a border.
    assert q[0, 5, 5] < guard
    assert q[-1, 5, 5] < guard


def test_q_kernel_refuses_to_difference_across_a_level_interface(lv, device):
    """The refinement-interface artifact: a seam between two levels renders as a
    perfect plane unless Q is masked where the stencil spans levels.

    Velocity here is a single smooth field, so any Q the kernel reports at the
    seam would be real. The point is that the *source data* is discontinuous
    across a real interface, so the kernel must refuse to difference there at
    all -- which is a property of the flags, not of the velocity.
    """
    shape = (16, 16, 16)
    vel = wp.array(_solid_body_rotation(shape, 0.01), dtype=wp.float32, device=device)

    # Half the grid comes from level 0 (stride 1 at render_exp=0), half from
    # level 1 (stride 2), so the masked shell is thicker on the coarser side.
    flags_np = np.ones(shape, dtype=np.uint8)
    flags_np[8:, :, :] = 2  # level 1
    flags = wp.array(flags_np, dtype=wp.uint8, device=device)

    qv = wp.zeros((2, *shape), dtype=wp.float32, device=device)
    wp.launch(lv._q_kernel, dim=shape, inputs=[vel, flags, qv, 1.0, 0], device=device)

    q = qv.numpy()[0]
    guard = lv._Q_INVALID * 0.5

    # i=7 reaches i+1=8 (level 1); i=8 and i=9 reach i-2=6,7 (level 0).
    assert np.all(q[7, 1:-1, 1:-1] < guard), "level-0 side of the seam must be masked"
    assert np.all(q[8, 2:-2, 2:-2] < guard), "level-1 side of the seam must be masked"
    assert np.all(q[9, 2:-2, 2:-2] < guard), "stride-2 reaches two cells across the seam"

    # Well away from the seam, both levels still produce valid Q. The level-1
    # slice starts two cells in because its stencil strides by two.
    assert np.all(q[3, 1:-1, 1:-1] > guard), "level-0 interior should be unaffected"
    assert np.all(q[6, 1:-1, 1:-1] > guard), "level-0 cells not reaching the seam stay valid"
    assert np.all(q[12, 2:-2, 2:-2] > guard), "level-1 interior should be unaffected"
    assert np.all(q[10, 2:-2, 2:-2] > guard), "first level-1 cell whose stencil clears the seam"


def test_q_is_valid_across_a_full_nested_hierarchy(lv, device):
    """Regression for the empty-picture bug: 4 valid bricks out of 50,337.

    A 6-level car case has *small* boxes at the fine levels and the wake out in
    the coarse ones. Excluding levels coarser than the render grid therefore left
    almost nothing valid, and the isosurface was blank while every other number
    in the log looked plausible. With all levels staged and the stencil strided
    per level, most of the grid must carry usable Q.
    """
    n = 128
    shape = (n, n, n)
    render_exp = 2
    num_levels = 6

    # Nested boxes, coarsest covering everything, each finer one half the width
    # -- the same shape as a windtunnel refinement stack.
    flags_np = np.full(shape, num_levels, dtype=np.uint8)  # level 5 everywhere
    for level in range(num_levels - 2, -1, -1):
        half = n // (2 ** (num_levels - 1 - level) * 2)
        lo = [n // 2 - half] * 3
        hi = [n // 2 + half] * 3
        flags_np[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] = level + 1

    present = {int(v) - 1 for v in np.unique(flags_np)}
    assert present == set(range(num_levels)), f"test setup should exercise every level, got {sorted(present)}"

    # Velocity replicated at each level's own block granularity, as staging does.
    base = _solid_body_rotation(shape, 0.01)
    vel_np = np.empty_like(base)
    for level in range(num_levels):
        d = 1 << max(level - render_exp, 0)
        mask = flags_np == (level + 1)
        if d == 1:
            blocky = base
        else:
            # Snap each component to its block origin value.
            idx = (np.arange(n) // d) * d
            blocky = base[:, idx][:, :, idx][:, :, :, idx]
        vel_np[:, mask] = blocky[:, mask]

    vel = wp.array(np.ascontiguousarray(vel_np), dtype=wp.float32, device=device)
    flags = wp.array(flags_np, dtype=wp.uint8, device=device)
    qv = wp.zeros((2, *shape), dtype=wp.float32, device=device)

    wp.launch(lv._q_kernel, dim=shape, inputs=[vel, flags, qv, 1.0, render_exp], device=device)

    q = qv.numpy()[0]
    valid_frac = float(np.count_nonzero(q > lv._Q_INVALID * 0.5)) / q.size

    assert valid_frac > 0.5, f"only {valid_frac * 100:.1f}% of cells carry valid Q; the picture would be near-empty"

    # Every level must contribute some valid cells, not just the 1:1 one.
    for level in sorted(present):
        sel = (flags_np == (level + 1)) & (q > lv._Q_INVALID * 0.5)
        assert sel.any(), f"level {level} contributed no valid Q at all"


def test_flags_encoding_round_trips_all_levels(lv):
    """level+1 must fit the nibble without colliding with the boundary bit."""
    for level in range(lv._MAX_ENCODABLE_LEVELS):
        code = level + 1
        assert code & lv._LEVEL_MASK == code, f"level {level} overflows the nibble"
        assert code & lv._BOUNDARY_BIT == 0, f"level {level} collides with the boundary bit"
        assert code != 0, "no level may encode as unwritten"

        marked = code | lv._BOUNDARY_BIT
        assert marked & lv._LEVEL_MASK == code, "boundary bit must not corrupt the level"
        assert marked != code, "a boundary cell must be distinguishable from fluid"


def test_brick_kernel_is_overlapping_max(lv, device):
    """Bricks must overlap by one cell so trilinear samples cannot fall in a gap."""
    shape = (16, 16, 16)
    q = np.full((2, *shape), -1.0, dtype=np.float32)

    # Peak sits in the first cell of brick 1, so brick 0 must also see it.
    q[0, 8, 4, 4] = 5.0

    qv = wp.array(q, dtype=wp.float32, device=device)
    bshape = tuple((v + lv._BRICK - 1) // lv._BRICK for v in shape)
    brick = wp.zeros(bshape, dtype=wp.float32, device=device)

    wp.launch(lv._brick_kernel, dim=bshape, inputs=[qv, brick], device=device)

    b = brick.numpy()
    assert b[1, 0, 0] == pytest.approx(5.0)
    assert b[0, 0, 0] == pytest.approx(5.0), "brick 0 must include the overlap cell at i=8"
    assert b[1, 1, 1] == pytest.approx(-1.0)


def _render_sphere(lv, device, radius=8.0, width=64, height=64, iso=1.0):
    """Raymarch a synthetic Q field whose iso level is an exact sphere.

    Q = radius^2 - |r|^2 + iso puts the Q = iso surface at |r| = radius, so the
    rendered silhouette has a known analytic size.
    """
    shape = (40, 40, 40)
    centre = np.array([20.0, 20.0, 20.0])

    i, j, k = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    r2 = (i + 0.5 - centre[0]) ** 2 + (j + 0.5 - centre[1]) ** 2 + (k + 0.5 - centre[2]) ** 2

    q = np.zeros((2, *shape), dtype=np.float32)
    q[0] = radius**2 - r2 + iso
    q[1] = 1.0

    qv = wp.array(q, dtype=wp.float32, device=device)
    bshape = tuple((v + lv._BRICK - 1) // lv._BRICK for v in shape)
    brick = wp.zeros(bshape, dtype=wp.float32, device=device)
    wp.launch(lv._brick_kernel, dim=bshape, inputs=[qv, brick], device=device)

    fb = wp.zeros((height, width, 3), dtype=wp.uint8, device=device)

    camera = lv.Camera(target=centre, distance=60.0, azimuth=0.0, elevation=0.0, fov=45.0)
    eye, fwd, right, up = camera.basis()

    wp.launch(
        lv._render_kernel,
        dim=(height, width),
        inputs=[
            qv,
            brick,
            fb,
            wp.uint64(0),
            0,  # no body mesh
            eye,
            fwd,
            right,
            up,
            math.tan(math.radians(camera.fov) * 0.5),
            iso,
            1.0,
            0.5,
            wp.vec3(1.0, 1.0, 1.0),
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(0.0, 0.0, 0.0),
            0.2,
            0.0,
            1.0,
        ],
        device=device,
    )

    return fb.numpy(), camera, radius


def test_render_kernel_silhouette_matches_analytic_sphere(lv, device):
    """The isosurface silhouette must land within a pixel of the analytic one.

    This is the end-to-end check on the two-level DDA, the bisection refinement
    and the camera basis all agreeing on one coordinate system.
    """
    img, camera, radius = _render_sphere(lv, device)

    # Background is pure black in this setup, so anything lit is a hit.
    hit = img.sum(axis=2) > 0
    assert hit.any(), "raymarch found no isosurface at all"

    height, width = hit.shape
    expected_px = 2.0 * math.atan(radius / camera.distance) / (2.0 * math.tan(math.radians(camera.fov) * 0.5)) * height

    rows = np.flatnonzero(hit.any(axis=1))
    cols = np.flatnonzero(hit.any(axis=0))
    measured_v = rows[-1] - rows[0] + 1
    measured_h = cols[-1] - cols[0] + 1

    assert abs(measured_v - expected_px) <= 2.0, f"vertical extent {measured_v}px vs analytic {expected_px:.1f}px"
    assert abs(measured_h - expected_px) <= 2.0, f"horizontal extent {measured_h}px vs analytic {expected_px:.1f}px"

    # A centred sphere must be centred in frame.
    assert abs(0.5 * (rows[0] + rows[-1]) - 0.5 * height) <= 1.5
    assert abs(0.5 * (cols[0] + cols[-1]) - 0.5 * width) <= 1.5


def test_render_kernel_skips_field_below_iso(lv, device):
    """With the iso level above the field peak, every ray must miss."""
    shape = (32, 32, 32)
    q = np.zeros((2, *shape), dtype=np.float32)
    q[0] = 1.0

    qv = wp.array(q, dtype=wp.float32, device=device)
    bshape = tuple((v + lv._BRICK - 1) // lv._BRICK for v in shape)
    brick = wp.zeros(bshape, dtype=wp.float32, device=device)
    wp.launch(lv._brick_kernel, dim=bshape, inputs=[qv, brick], device=device)

    fb = wp.zeros((32, 32, 3), dtype=wp.uint8, device=device)
    camera = lv.Camera(target=np.array([16.0, 16.0, 16.0]), distance=50.0)
    eye, fwd, right, up = camera.basis()

    wp.launch(
        lv._render_kernel,
        dim=(32, 32),
        inputs=[
            qv,
            brick,
            fb,
            wp.uint64(0),
            0,
            eye,
            fwd,
            right,
            up,
            math.tan(math.radians(45.0) * 0.5),
            10.0,  # iso well above the field
            1.0,
            0.5,
            wp.vec3(1.0, 1.0, 1.0),
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(0.0, 0.0, 0.0),
            0.2,
            0.0,
            1.0,
        ],
        device=device,
    )

    assert fb.numpy().sum() == 0


def test_turbo_endpoints_and_clamping(lv, device):
    """Guard the polynomial fit: it must stay in gamut and clamp outside [0,1]."""

    @wp.kernel
    def probe(ts: wp.array(dtype=wp.float32), out: wp.array2d(dtype=wp.float32)):
        i = wp.tid()
        c = lv._turbo(ts[i])
        out[i, 0] = c[0]
        out[i, 1] = c[1]
        out[i, 2] = c[2]

    ts = np.array([-1.0, 0.0, 0.15, 0.5, 0.9, 1.0, 2.0], dtype=np.float32)
    ts_wp = wp.array(ts, dtype=wp.float32, device=device)
    out = wp.zeros((ts.size, 3), dtype=wp.float32, device=device)

    wp.launch(probe, dim=ts.size, inputs=[ts_wp, out], device=device)
    c = out.numpy()

    assert np.all(c >= 0.0) and np.all(c <= 1.0)
    np.testing.assert_allclose(c[0], c[1], atol=1e-6)  # t=-1 clamps to t=0
    np.testing.assert_allclose(c[-1], c[-2], atol=1e-6)  # t=2 clamps to t=1

    # Turbo runs blue -> green -> red. Sampled inside the range rather than at
    # the endpoints, where the polynomial fit is known to deviate from the real
    # colormap (see the _turbo docstring).
    assert c[2][2] > c[2][0], "blue should dominate low values"
    assert c[3][1] > c[3][2], "green should dominate mid values"
    assert c[4][0] > c[4][2], "red should dominate high values"


def test_scatter_loop_bound_from_array_is_not_unrolled(device):
    """Mirror of the Neon staging kernel's scatter loop.

    The staging container cannot capture its loop bound as a python int: warp
    fully unrolls range() over a compile-time constant, and a coarse level in a
    6-level case covers 32 render cells per axis -- a 32768-body unrolled triple
    loop. Reading the bound back from a device array forces a dynamic loop.

    The bound is varied at *launch* time here, over a single compiled kernel,
    which is only possible if the loop really is dynamic.
    """

    @wp.kernel
    def scatter(
        n_arr: wp.array(dtype=wp.int32),
        origin: wp.array2d(dtype=wp.int32),
        dst: wp.array3d(dtype=wp.uint8),
    ):
        c = wp.tid()
        n = n_arr[0]

        rx = origin[c, 0]
        ry = origin[c, 1]
        rz = origin[c, 2]

        nx = dst.shape[0]
        ny = dst.shape[1]
        nz = dst.shape[2]

        for a in range(n):
            x = rx + a
            if x < nx:
                for b in range(n):
                    y = ry + b
                    if y < ny:
                        for d in range(n):
                            z = rz + d
                            if z < nz:
                                dst[x, y, z] = wp.uint8(1)

    shape = (64, 64, 64)

    for n in (1, 4, 32):
        n_arr = wp.array([n], dtype=wp.int32, device=device)
        # One coarse cell at the origin, one deliberately overhanging the far
        # corner so the bounds guards are exercised.
        origins = np.array([[0, 0, 0], [shape[0] - 2, shape[1] - 2, shape[2] - 2]], dtype=np.int32)
        origin = wp.array(origins, dtype=wp.int32, device=device)
        dst = wp.zeros(shape, dtype=wp.uint8, device=device)

        wp.launch(scatter, dim=origins.shape[0], inputs=[n_arr, origin, dst], device=device)

        out = dst.numpy()
        assert out[:n, :n, :n].all(), f"n={n}: scatter block not filled"

        # The overhanging cell must clip to the grid rather than wrap or corrupt,
        # and nothing outside the two blocks may be touched. The blocks are
        # disjoint for every n tested here (n <= 32 < 62). Note the second block
        # is anchored at 62 and extends forward, so it is clipped to min(n, 2).
        o = shape[0] - 2
        clipped = min(n, 2)
        assert out[o : o + clipped, o : o + clipped, o : o + clipped].all(), f"n={n}: clipped scatter lost its in-bounds part"
        assert out.sum() == n**3 + clipped**3, f"n={n}: wrote outside the scatter blocks"


def test_pick_render_level_respects_budget(lv):
    shape = (2000, 800, 600)
    budget = 512 * 1024 * 1024

    r, rshape = lv._pick_render_level(shape, budget)

    assert int(np.prod(rshape)) * lv._BYTES_PER_RENDER_CELL <= budget
    # Must be the *smallest* decimation that fits, i.e. one step finer overflows.
    if r > 0:
        finer = tuple(max((s + (1 << (r - 1)) - 1) >> (r - 1), 1) for s in shape)
        assert int(np.prod(finer)) * lv._BYTES_PER_RENDER_CELL > budget

    # Decimation must cover the whole domain.
    for s, rs in zip(shape, rshape):
        assert rs * (1 << r) >= s


def test_camera_basis_is_orthonormal_and_looks_at_target(lv):
    camera = lv.Camera(target=np.array([10.0, 20.0, 30.0]), distance=50.0, azimuth=35.0, elevation=20.0)
    eye, fwd, right, up = camera.basis()

    e = np.array([eye[0], eye[1], eye[2]])
    f = np.array([fwd[0], fwd[1], fwd[2]])
    rt = np.array([right[0], right[1], right[2]])
    u = np.array([up[0], up[1], up[2]])

    for v in (f, rt, u):
        np.testing.assert_allclose(np.linalg.norm(v), 1.0, atol=1e-5)
    for a, b in ((f, rt), (f, u), (rt, u)):
        np.testing.assert_allclose(np.dot(a, b), 0.0, atol=1e-5)

    np.testing.assert_allclose(np.linalg.norm(e - camera.target), camera.distance, rtol=1e-5)
    np.testing.assert_allclose(f, (camera.target - e) / camera.distance, atol=1e-5)

    # Elevation is measured toward +z (up), so a positive elevation looks down.
    assert f[2] < 0.0


def test_camera_elevation_is_clamped(lv):
    """Elevation past the pole would make right = cross(fwd, world_up) degenerate."""
    camera = lv.Camera(target=np.zeros(3), distance=10.0, elevation=89.9)
    _, fwd, right, up = camera.basis()

    for v in (fwd, right, up):
        assert np.all(np.isfinite([v[0], v[1], v[2]]))
    np.testing.assert_allclose(np.linalg.norm([right[0], right[1], right[2]]), 1.0, atol=1e-5)


class _FakeBrick:
    """Stands in for the device brick array; only ``numpy()`` is used."""

    def __init__(self, values):
        self._values = np.asarray(values, dtype=np.float32)

    def numpy(self):
        return self._values

    def set(self, values):
        self._values = np.asarray(values, dtype=np.float32)


def _calib_stub(lv, **overrides):
    """Minimal duck-typed object for exercising LiveView._calibrate in isolation."""

    class Stub:
        pass

    s = Stub()
    s._brick = _FakeBrick(np.full(64, 1.0))
    s.iso_brick_fraction = 0.05
    s.iso_smoothing = 0.2
    s.q_iso = 0.0
    s.w_max = 1.0
    s._needs_calibration = True
    s.verbose = False
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def test_calibrate_leaves_the_target_brick_fraction_above_the_iso_level(lv):
    """The level must be a percentile of the field, not a fraction of its peak.

    This is the bug that produced a blank screen at 2.4 s/frame: scaling the
    peak put the level *below* the field floor, so no sample could bracket a
    crossing and every ray traversed the whole volume. Occupancy is the quantity
    that has to be controlled, because it bounds both cost and visibility.
    """
    s = _calib_stub(lv, iso_brick_fraction=0.05)

    # A field whose peak is orders of magnitude above its bulk -- exactly the
    # shape that breaks a peak-scaled threshold. Log-spread so there are no
    # exact ties, as in a real field.
    rng = np.random.default_rng(0)
    values = np.concatenate([
        10.0 ** rng.uniform(-10.0, -8.0, 1000),  # quiet freestream
        10.0 ** rng.uniform(-6.0, -3.0, 200),  # wake structure
    ])
    s._brick.set(values)

    lv.LiveView._calibrate(s, force=True)

    positive = values[values > 0.0]
    occupied = np.count_nonzero(positive >= s.q_iso) / positive.size
    assert occupied == pytest.approx(0.05, abs=0.01), f"occupancy {occupied:.3f} should track iso_brick_fraction"

    # And crucially the level sits inside the field, so a crossing can bracket.
    # A peak-scaled level (0.02 * max) would fall far below the field floor here,
    # which is precisely the blank-and-slow failure.
    assert values.min() < s.q_iso < values.max()
    assert s.q_iso > 0.02 * values.max() * 1.0e-3, "sanity: level is nowhere near a peak-scaled one"


def test_calibrate_ignores_strain_dominated_bricks(lv):
    """Q <= 0 bricks hold no vortex and must not drag the percentile down.

    Most of a windtunnel domain is quiet freestream, so if non-positive bricks
    counted they would dominate the distribution and pull the level below
    anything the isosurface could occupy.
    """
    s = _calib_stub(lv, iso_brick_fraction=0.1)

    hot = np.linspace(1.0, 2.0, 100)
    values = np.concatenate([np.full(5000, -3.0), np.zeros(2000), hot])
    s._brick.set(values)

    lv.LiveView._calibrate(s, force=True)

    # The level must be set from the 100 positive bricks alone.
    assert s.q_iso == pytest.approx(np.percentile(hot, 90.0), rel=1e-6)
    assert s.q_iso > 1.0


def test_calibrate_holds_level_when_no_rotation_exists_yet(lv):
    """All-negative field: hold the previous level rather than set an unusable one."""
    s = _calib_stub(lv, q_iso=0.25, _needs_calibration=False)
    s._brick.set(np.full(64, -1.0))

    lv.LiveView._calibrate(s)
    assert s.q_iso == pytest.approx(0.25)
    # Still uncalibrated in the sense that it never locked on to a real field.
    assert s._needs_calibration is False


def test_calibrate_occupancy_follows_the_requested_fraction(lv):
    """Lower fraction -> higher level -> less surface drawn and cheaper march."""
    values = np.linspace(1.0e-6, 1.0, 2000)

    levels = {}
    for frac in (0.01, 0.1, 0.4):
        s = _calib_stub(lv, iso_brick_fraction=frac)
        s._brick.set(values)
        lv.LiveView._calibrate(s, force=True)
        levels[frac] = s.q_iso

        occupied = np.count_nonzero(values >= s.q_iso) / values.size
        assert occupied == pytest.approx(frac, abs=0.01)

    assert levels[0.01] > levels[0.1] > levels[0.4], "a smaller fraction must give a higher iso level"


def test_calibrate_first_call_snaps_and_later_calls_smooth(lv):
    """The startup transient bug: a threshold frozen at step 0 is wrong later.

    Q at step 0 is orders of magnitude below a developed wake, so calibration
    has to keep tracking. The first call must snap (nothing to smooth against),
    subsequent calls must move toward the new target without jumping to it.
    """
    s = _calib_stub(lv, iso_brick_fraction=0.05)

    # Frame 1: undeveloped flow, tiny Q -- roughly what the R2 run reported.
    early = np.full(64, 6.131e-05)
    s._brick.set(early)
    lv.LiveView._calibrate(s)
    first = s.q_iso
    assert first == pytest.approx(6.131e-05)
    assert not s._needs_calibration

    # Flow develops: the distribution climbs four orders of magnitude.
    late = np.full(64, 0.5)
    s._brick.set(late)
    target = 0.5

    lv.LiveView._calibrate(s)
    assert first < s.q_iso < target, "second call should move toward the target, not snap to it"
    np.testing.assert_allclose(s.q_iso, 0.8 * first + 0.2 * target, rtol=1e-6)

    # Repeated tracking must converge rather than stall or overshoot.
    for _ in range(200):
        lv.LiveView._calibrate(s)
    np.testing.assert_allclose(s.q_iso, target, rtol=1e-3)


def test_calibrate_force_snaps_immediately(lv):
    """The R key must bypass smoothing rather than crawl to the new level."""
    s = _calib_stub(lv)
    s._brick.set(np.full(64, 1.0e-6))
    lv.LiveView._calibrate(s)

    s._brick.set(np.full(64, 1.0))
    lv.LiveView._calibrate(s, force=True)
    assert s.q_iso == pytest.approx(1.0)


def test_calibrate_ignores_invalid_bricks_and_degenerate_peaks(lv):
    """All-invalid or non-positive fields must leave the previous level intact."""
    s = _calib_stub(lv, q_iso=0.25, _needs_calibration=False)

    s._brick.set(np.full(64, lv._Q_INVALID))
    lv.LiveView._calibrate(s)
    assert s.q_iso == pytest.approx(0.25), "all-invalid brick grid should not move the iso level"

    # A field that is valid but everywhere <= 0 has no isosurface to find.
    s._brick.set(np.full(64, -3.0))
    lv.LiveView._calibrate(s)
    assert s.q_iso == pytest.approx(0.25)

    # w_max must stay consistent with q_iso once a real peak arrives.
    s._brick.set(np.full(64, 1.0))
    lv.LiveView._calibrate(s, force=True)
    np.testing.assert_allclose(s.w_max, 4.0 * math.sqrt(2.0 * s.q_iso), rtol=1e-6)


def test_png_display_writes_off_the_calling_thread(lv, tmp_path):
    """PNG encode must not run inline: it costs several times the GPU render."""
    width, height = 64, 32
    state = lv._ViewerState(lv.Camera(target=np.zeros(3), distance=1.0))
    display = lv._PngDisplay(width, height, str(tmp_path), state)

    frame = (np.random.default_rng(0).integers(0, 256, (height, width, 3), dtype=np.uint8)).tobytes()
    for _ in range(3):
        display.show(frame)

    # close() drains the queue and joins the worker.
    display.close()

    written = sorted(tmp_path.glob("live_*.png"))
    assert len(written) + display.dropped == 3, "every frame must be written or explicitly counted as dropped"
    assert written, "no frames survived the queue"

    from PIL import Image

    img = np.asarray(Image.open(written[0]))
    assert img.shape == (height, width, 3)
    np.testing.assert_array_equal(img, np.frombuffer(frame, dtype=np.uint8).reshape(height, width, 3))


def test_png_display_drops_rather_than_blocks_when_saturated(lv, tmp_path):
    """A slow disk must throttle the movie, never the solver."""
    width, height = 32, 16
    state = lv._ViewerState(lv.Camera(target=np.zeros(3), distance=1.0))
    display = lv._PngDisplay(width, height, str(tmp_path), state, queue_depth=1)

    # Swap in a queue with no consumer, so saturation is deterministic rather
    # than a race against however fast the worker happens to encode. The worker
    # keeps its reference to the original queue and stays parked on it.
    original = display._queue
    display._queue = queue.Queue(maxsize=1)

    frame = b"\x01" * (width * height * 3)
    for _ in range(50):
        display.show(frame)  # must return immediately, never block

    assert display.index == 1, "exactly one frame should have fit in the queue"
    assert display.dropped == 49, "every other frame should have been dropped, not blocked"

    # Restore the real queue so close() can drain it and join the worker.
    display._queue = original
    display.close()
