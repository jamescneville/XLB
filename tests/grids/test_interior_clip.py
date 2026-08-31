"""
Invariant tests for interior-cavity clipping of the multi-resolution mesh.

Clipping the wake padding inside a vehicle shrinks refinement regions, which
risks two things Neon's mGrid depends on: nesting of consecutive levels, and
strong balance (no level touching a level two steps coarser). These tests
replicate the level loop of `makemesh.makeMesh` on the CPU -- using the real
kernel, schedule and mask code -- and assert those invariants hold.

Run with: pytest tests/grids/test_interior_clip.py
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name, relpath):
    """Import a module by path, avoiding the warp/neon imports in `xlb.__init__`."""
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


idet = _load("interior_detect", "xlb/utils/interior_detect.py")


def calculate_kernel(padding_values):
    """Copy of `makemesh.calculate_kernel`, which cannot be imported without warp."""
    kernels = []
    for values in padding_values:
        xn, xp, yn, yp, zn, zp = values
        dims = [max(p, n) * 2 + 1 for n, p in ((xn, xp), (yn, yp), (zn, zp))]
        kernel = np.zeros(tuple(dims), bool)
        slices = []
        for dim, (n, p) in zip(dims, ((xn, xp), (yn, yp), (zn, zp))):
            mid = (dim - 1) // 2
            slices.append(slice(mid - p, dim - (mid - n)))
        kernel[tuple(slices)] = True
        kernels.append(kernel)
    return kernels


def grow_cpu(matrix, vox, origin, kernel, kernel_iso=None, interior=None):
    """CPU mirror of `makemesh.grow_gpu`, including the split/clipped dilation."""
    from scipy import ndimage

    kernels = [kernel] if kernel_iso is None else [kernel_iso, kernel]
    pad = np.max([(np.array(k.shape) * 0.5).astype(int) for k in kernels], axis=0)
    padded_shape = tuple(np.array(matrix.shape) + 2 * pad)
    origin_pad = origin - pad * vox

    def place(seed):
        buf = np.zeros(padded_shape, bool)
        buf[pad[0] : pad[0] + matrix.shape[0], pad[1] : pad[1] + matrix.shape[1], pad[2] : pad[2] + matrix.shape[2]] = seed
        return buf

    out = np.zeros(padded_shape, bool)
    if kernel_iso is not None:
        out |= ndimage.binary_dilation(place(matrix), structure=kernel_iso)
        exterior_seed = matrix & ~interior.seed.resample(matrix.shape, origin, vox)
        wake = ndimage.binary_dilation(place(exterior_seed), structure=kernel)
        out |= wake & ~interior.clip.resample(padded_shape, origin_pad, vox)
    else:
        out |= ndimage.binary_dilation(place(matrix), structure=kernel)
    return out, origin_pad


def fill_cpu(matrix, vox, origin, close):
    """CPU mirror of `makemesh.fill_gpu`: octree block alignment plus optional closing."""
    from scipy import ndimage

    a = (origin / vox) % 2
    close_to_int = np.isclose(a, np.round(a), atol=1e-8)
    a[close_to_int] = np.round(a[close_to_int])
    pad_lo = np.floor(a).astype(int)
    pad_hi = np.round((np.array(matrix.shape) + pad_lo) % 2).astype(int)
    padded = np.zeros(tuple(np.array(matrix.shape) + pad_lo + pad_hi), bool)
    padded[pad_lo[0] : pad_lo[0] + matrix.shape[0], pad_lo[1] : pad_lo[1] + matrix.shape[1], pad_lo[2] : pad_lo[2] + matrix.shape[2]] = matrix

    blocks = padded.reshape(padded.shape[0] // 2, 2, padded.shape[1] // 2, 2, padded.shape[2] // 2, 2).any(axis=(1, 3, 5))
    if close:
        cross = ndimage.generate_binary_structure(3, 1)
        blocks = ndimage.binary_erosion(ndimage.binary_dilation(blocks, structure=cross), structure=cross, border_value=1)
    return np.repeat(np.repeat(np.repeat(blocks, 2, 0), 2, 1), 2, 2), origin - pad_lo * vox


def crop_cpu(matrix, origin, domain_min, domain_max, vox):
    lo = np.maximum(np.round((domain_min - origin) / vox).astype(int), 0)
    hi = np.minimum(np.round((domain_max - origin) / vox).astype(int), matrix.shape)
    return matrix[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]], origin + lo * vox


def pad_to_even(grid):
    return np.pad(grid, [(0, s % 2) for s in grid.shape], constant_values=False)


def build_levels(solid, solid_origin, base_vox, domain_min, domain_max, kernels, kernels_iso=None, clips=None, levels=4):
    """Replicate the `makeMesh` level loop and return the per-level regions."""
    regions, origins = [], []
    matrix, origin, vox, df = solid, np.array(solid_origin, float), base_vox, None

    for level in range(levels):
        if level:
            matrix = df[::2, ::2, ::2]
            vox *= 2
        extra = {}
        if kernels_iso is not None and clips is not None and clips[level] is not None:
            extra = {"kernel_iso": kernels_iso[level], "interior": clips[level]}

        if level == levels - 1:
            shape = np.round((domain_max - domain_min) / vox).astype(int)
            df, origin = np.ones(tuple(shape), bool), domain_min.copy()
        else:
            g, origin = grow_cpu(matrix, vox, origin, kernels[level], **extra)
            f, origin = fill_cpu(g, vox, origin, True)
            df, origin = crop_cpu(f, origin, domain_min, domain_max, vox)
            df = pad_to_even(df)

        regions.append(df)
        origins.append(origin.copy())
    return regions, origins


def _to_finest(region, origin, vox, base_vox, domain_min, domain_shape):
    """Upsample a level region onto the finest lattice covering the whole domain."""
    ratio = int(round(vox / base_vox))
    full = np.zeros(tuple(domain_shape), bool)
    up = np.repeat(np.repeat(np.repeat(region, ratio, 0), ratio, 1), ratio, 2)
    off = np.round((origin - domain_min) / base_vox).astype(int)
    lo = np.maximum(off, 0)
    hi = np.minimum(off + np.array(up.shape), domain_shape)
    if np.all(hi > lo):
        full[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] = up[
            lo[0] - off[0] : hi[0] - off[0], lo[1] - off[1] : hi[1] - off[1], lo[2] - off[2] : hi[2] - off[2]
        ]
    return full


def _hollow_box(shape, wall, hole=None):
    """A shell with an optional hole in the +x face, mimicking an open vehicle body."""
    solid = np.ones(shape, bool)
    solid[wall:-wall, wall:-wall, wall:-wall] = False
    if hole is not None:
        cy, cz = shape[1] // 2, shape[2] // 2
        solid[-wall:, cy - hole : cy + hole, cz - hole : cz + hole] = False
    return solid


@pytest.fixture(scope="module")
def setup():
    base_vox = 0.05
    shape = (60, 30, 24)
    solid = _hollow_box(shape, wall=2, hole=2)
    solid_origin = np.array([1.0, 1.0, 1.0])

    max_vox = base_vox * 2 ** (4 - 1)
    domain_min = np.zeros(3)
    domain_max = np.ceil((solid_origin + np.array(shape) * base_vox + 2.0) / max_vox) * max_vox

    mask = idet.InteriorMask(*_interior_of(solid, solid_origin, base_vox))
    return base_vox, solid, solid_origin, domain_min, domain_max, mask


SEAL_CELLS = 2  # bridges the 4-cell hole in the shell


def _interior_of(solid, origin, vox):
    """Interior cavity of an already-voxelized shell, via the module's own detection."""
    padded = np.pad(solid, 4, constant_values=False)
    return idet.interior_from_solid(padded, SEAL_CELLS), np.array(origin, float) - 4 * vox, vox


PADDING = [[4, 20, 4, 4, 4, 4], [4, 20, 4, 4, 4, 4], [4, 12, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4]]
PADDING_INTERIOR = [[2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2], [4, 4, 4, 4, 4, 4]]


def test_interior_detected_through_hole(setup):
    """Sealing lets a shell with an open face still be recognised as enclosed."""
    _, solid, _, _, _, mask = setup
    cavity = (56, 26, 20)  # hollow interior of the 60x30x24 shell with 2-cell walls
    assert mask.mask.sum() > 0.8 * np.prod(cavity)
    # The cavity is strictly inside the shell, not leaking out through the hole.
    idx = np.argwhere(mask.mask)
    assert idx[:, 0].min() > 0 and idx[:, 0].max() < mask.mask.shape[0] - 1


def test_no_interior_without_sealing(setup):
    """The hole is a real leak: with sealing off, nothing is enclosed."""
    _, solid, _, _, _, _ = setup
    assert idet.interior_from_solid(np.pad(solid, 4, constant_values=False), 0).sum() == 0


# Voxel sizes need not be powers of two or round numbers, so the schedule is
# checked across the whole plausible range rather than at one size.
@pytest.mark.parametrize("base_mm", [4, 8, 9, 10, 12, 16, 20, 25, 32])
@pytest.mark.parametrize("resolution_level", [0, 1, 2, 3])
def test_clip_schedule_is_monotone_and_buffered(base_mm, resolution_level):
    """Nesting needs a shrinking clip; balance needs it to shrink fast enough."""
    base_vox = base_mm / 1000.0
    class_vox = idet.level_voxel_size(base_vox, resolution_level)

    # A level is a power-of-two multiple of the finest voxel, so one
    # classification cell is a whole number of voxels at every level and mask
    # resampling gives uniform steps.
    ratio = np.log2(class_vox / base_vox)
    assert abs(ratio - round(ratio)) < 1e-12

    cells = idet.clip_schedule(7, base_vox, class_vox, verbose=False)
    for level in range(1, len(cells)):
        needed = 2.0 * (2**level) * base_vox
        assert cells[level] > cells[level - 1], "clip must shrink at each coarser level (nesting)"
        assert (cells[level] - cells[level - 1]) * class_vox >= needed - 1e-12, "stagger too small for strong balance"


def _flatten(regions, origins, base_vox, domain_min, domain_shape):
    return [_to_finest(r, o, base_vox * 2**level, base_vox, domain_min, domain_shape) for level, (r, o) in enumerate(zip(regions, origins))]


def _max_level_jump(full, domain_shape):
    """Largest refinement-level difference between face-adjacent cells."""
    owner = np.full(domain_shape, len(full) - 1, np.int8)
    for level in reversed(range(len(full))):
        owner[full[level]] = level
    worst = 0
    for axis in range(3):
        a = np.take(owner, np.arange(domain_shape[axis] - 1), axis=axis).astype(int)
        b = np.take(owner, np.arange(1, domain_shape[axis]), axis=axis).astype(int)
        worst = max(worst, int(np.abs(a - b).max()))
    return worst


@pytest.mark.parametrize("clipped", [False, True])
def test_levels_are_nested_and_strongly_balanced(setup, clipped):
    base_vox, solid, solid_origin, domain_min, domain_max, mask = setup
    kernels = calculate_kernel(PADDING)
    kwargs = {}
    if clipped:
        kwargs = {
            "kernels_iso": calculate_kernel(PADDING_INTERIOR),
            "clips": idet.build_level_clips(mask, len(kernels), base_vox, 3, verbose=False),
        }

    regions, origins = build_levels(solid, solid_origin, base_vox, domain_min, domain_max, kernels, levels=4, **kwargs)
    domain_shape = np.round((domain_max - domain_min) / base_vox).astype(int)
    full = _flatten(regions, origins, base_vox, domain_min, domain_shape)

    # Nesting: each level's region contains every finer level's region.
    for level in range(1, len(full)):
        assert not (full[level - 1] & ~full[level]).any(), f"level {level} does not contain level {level - 1}"

    # Coverage: the coarsest level spans the whole domain, so no cell is orphaned.
    assert full[-1].all()

    # Every solid voxel is resolved at the finest level.
    solid_full = _to_finest(solid, solid_origin, base_vox, base_vox, domain_min, domain_shape)
    assert not (solid_full & ~full[0]).any(), "solid voxel missing from the finest level"

    assert _max_level_jump(full, domain_shape) <= 1, "level jump > 1: strong balance violated"


# Clipping the wake threatens strong balance by collapsing the buffer between
# consecutive levels at the clip face. Two independent mechanisms prevent it: the
# never-clipped isotropic pass, which always grows past the finer level's
# boundary, and the staggered clip schedule. Either alone is sufficient; the
# failure case documents what the schedule is actually for.
@pytest.mark.parametrize(
    "iso_padding, stagger, expect_jump",
    [
        (2, True, 1),
        (2, False, 1),  # isotropic pass alone preserves balance
        (0, True, 1),  # staggered schedule alone preserves balance
        (0, False, 2),  # neither: the buffer collapses, as predicted
    ],
)
def test_balance_guards(setup, iso_padding, stagger, expect_jump):
    base_vox, solid, solid_origin, domain_min, domain_max, mask = setup
    kernels = calculate_kernel(PADDING)
    n_clipped = len(PADDING) - 1

    if stagger:
        clips = idet.build_level_clips(mask, len(kernels), base_vox, n_clipped, verbose=False)
    else:
        seed, flat = mask.dilated(1), mask.eroded(2)
        clips = [idet.InteriorClip(seed=seed, clip=flat) for _ in range(n_clipped)] + [None]

    regions, origins = build_levels(
        solid,
        solid_origin,
        base_vox,
        domain_min,
        domain_max,
        kernels,
        kernels_iso=calculate_kernel([[iso_padding] * 6] * n_clipped + [[4] * 6]),
        clips=clips,
        levels=4,
    )
    domain_shape = np.round((domain_max - domain_min) / base_vox).astype(int)
    assert _max_level_jump(_flatten(regions, origins, base_vox, domain_min, domain_shape), domain_shape) == expect_jump


def test_clipping_actually_saves_voxels(setup):
    """The interior clip must reduce the fine-level count, or it is pointless."""
    base_vox, solid, solid_origin, domain_min, domain_max, mask = setup
    kernels = calculate_kernel(PADDING)
    plain, _ = build_levels(solid, solid_origin, base_vox, domain_min, domain_max, kernels, levels=4)
    clipped, _ = build_levels(
        solid,
        solid_origin,
        base_vox,
        domain_min,
        domain_max,
        kernels,
        kernels_iso=calculate_kernel(PADDING_INTERIOR),
        clips=idet.build_level_clips(mask, len(kernels), base_vox, 3, verbose=False),
        levels=4,
    )
    assert clipped[0].sum() < plain[0].sum()
