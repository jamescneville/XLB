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


def test_q_kernel_matches_solid_body_rotation(lv, device):
    shape = (24, 24, 12)
    omega = 0.01
    u_ref = 1.0  # so the normalisation is a no-op and we can compare directly

    vel = wp.array(_solid_body_rotation(shape, omega), dtype=wp.float32, device=device)
    flags = wp.full(shape, wp.uint8(1), dtype=wp.uint8, device=device)
    qv = wp.zeros((2, *shape), dtype=wp.float32, device=device)

    wp.launch(lv._q_kernel, dim=shape, inputs=[vel, flags, qv, 1.0 / u_ref], device=device)

    out = qv.numpy()
    interior = (slice(1, -1),) * 3

    np.testing.assert_allclose(out[0][interior], omega**2, rtol=1e-4)
    np.testing.assert_allclose(out[1][interior], 2.0 * omega, rtol=1e-4)


def test_q_kernel_invalidates_boundary_neighbourhoods(lv, device):
    """A single non-fluid cell must invalidate its whole 6-neighbourhood."""
    shape = (12, 12, 12)

    vel = wp.array(_solid_body_rotation(shape, 0.01), dtype=wp.float32, device=device)

    flags_np = np.ones(shape, dtype=np.uint8)
    flags_np[6, 6, 6] = 2  # boundary cell
    flags_np[3, 3, 3] = 0  # never written
    flags = wp.array(flags_np, dtype=wp.uint8, device=device)

    qv = wp.zeros((2, *shape), dtype=wp.float32, device=device)
    wp.launch(lv._q_kernel, dim=shape, inputs=[vel, flags, qv, 1.0], device=device)

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
