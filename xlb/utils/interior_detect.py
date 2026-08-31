"""
Interior-cavity detection for dirty, non-watertight vehicle meshes.

Vehicle STLs are rarely watertight, so winding-number and ray-parity tests are
unreliable: open windows, zero-thickness body panels, duplicated triangles and
flipped normals all break them. This module instead classifies space
morphologically, on a throwaway coarse grid:

    voxelize -> close (seal narrow gaps) -> flood fill from the outside ->
    whatever free space the flood could not reach is an interior cavity

The result is used only to build masks that steer voxel refinement. The STL
itself is never modified and boundary-condition voxelization never sees this
grid, so sealing a grille slot here cannot coarsen or remove the grille in the
simulation -- at worst it costs that cavity its long anisotropic wake padding
while leaving its isotropic refinement band untouched.

Settings follow the same convention as ground refinement -- a level says which
voxel size to work at, and voxel counts say how much -- so everything scales
with the configured voxel size:

  resolution_level  Detection runs on cells of this refinement level: level 1 is
                    16 mm when level 0 is 8 mm. Sets the step size of the
                    refinement transitions inside the body, and the cost of
                    detection.
  seal_voxels       The widest gap to treat as closed, in cells of
                    `resolution_level`. Keep it comfortably below the narrowest
                    passage that carries real flow -- ground clearance,
                    typically ~150 mm.
  smooth_voxels     Rounds the detected cavity by this many cells, which smooths
                    the refinement transitions inside the body.

Note that `resolution_level` seals too: voxelization marks a cell solid if any
triangle touches it, so any gap narrower than one cell is already closed before
`seal_voxels` is applied. The effective sealing is therefore the larger of the
two, and it is reported in millimetres on every run so the interaction is
visible rather than implied.
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy import ndimage
from tabulate import tabulate

# 26-connectivity for the exterior flood fill. This is the accuracy-safe
# direction: a cavity is only called interior when not even a diagonal path
# escapes it, so we under-report interiors rather than over-report them.
_FLOOD_STRUCT = np.ones((3, 3, 3), bool)

# Cube structuring element; iterating it N times gives a cube of side 2N+1.
_BOX_STRUCT = np.ones((3, 3, 3), bool)


def _dilate(mask, cells):
    if cells <= 0:
        return mask
    return ndimage.binary_dilation(mask, structure=_BOX_STRUCT, iterations=int(cells))


def _erode(mask, cells):
    if cells <= 0:
        return mask
    # border_value=0 treats outside the array as empty, so erosion never grows a
    # clip region at the array edge. Under-clipping only costs voxel savings.
    return ndimage.binary_erosion(mask, structure=_BOX_STRUCT, iterations=int(cells), border_value=0)


class InteriorMask:
    """An interior-cavity mask living on its own coarse classification grid."""

    def __init__(self, mask, origin, vox):
        self.mask = mask
        self.origin = np.asarray(origin, float)
        self.vox = float(vox)

    def resample(self, shape, origin, vox):
        """
        Nearest-neighbour sample onto another axis-aligned grid.

        Cells whose centre falls outside the classification grid sample as False,
        which is correct for both uses here: outside the vehicle bounding box
        there is no interior cavity.
        """
        origin = np.asarray(origin, float)
        idx, valid = [], []
        for ax in range(3):
            a = (origin[ax] - self.origin[ax]) / self.vox + 0.5 * vox / self.vox
            f = np.floor(a + np.arange(shape[ax]) * (vox / self.vox)).astype(np.int64)
            valid.append((f >= 0) & (f < self.mask.shape[ax]))
            idx.append(np.clip(f, 0, self.mask.shape[ax] - 1))
        out = self.mask[np.ix_(idx[0], idx[1], idx[2])]
        out &= valid[0][:, None, None]
        out &= valid[1][None, :, None]
        out &= valid[2][None, None, :]
        return out

    def eroded(self, cells):
        return InteriorMask(_erode(self.mask, cells), self.origin, self.vox)

    def dilated(self, cells):
        return InteriorMask(_dilate(self.mask, cells), self.origin, self.vox)


class InteriorClip:
    """
    Per-level masks steering the split dilation in `makemesh.grow_gpu`.

    seed  Solid voxels overlapping this are not allowed to seed the long
          anisotropic wake kernel (they still seed the isotropic kernel).
    clip  The wake kernel may not grow into this region at all. Retracted
          further at each coarser level to preserve strong balance -- see
          `clip_schedule`.
    """

    def __init__(self, seed, clip):
        self.seed = seed
        self.clip = clip


def clip_schedule(n_levels, base_vox, class_vox, verbose=True):
    """
    Retraction of the interior clip mask per level, in classification cells.

    Clipping the same interior surface at every level would collapse the buffer
    between consecutive refinement regions at the clip face, which can put level
    l face-to-face with level l+2 and break the strong balance that Neon's mGrid
    requires. Retracting the clip further at each coarser level restores it.

    Two conditions are enforced:
      nesting  clip[l+1] subset of clip[l], so a coarse level never removes a
               cell that the finer level below it kept.
      balance  the clip boundaries of consecutive levels are at least two cells
               of the coarser level apart, which is also enough slack to absorb
               the block-OR realignment and closing inside `fill_gpu`.
    """
    cells, prev, d = [], 0, 2.0 * base_vox
    for level in range(n_levels):
        buffer_m = 2.0 * (2**level) * base_vox
        if level > 0:
            d += buffer_m
        c = int(np.ceil(d / class_vox))
        if level > 0:
            c = max(c, prev + max(int(np.ceil(buffer_m / class_vox)), 1))
        cells.append(c)
        prev = c

    if verbose:
        rows = [[level, f"{c * class_vox * 1000:.0f}", c] for level, c in enumerate(cells)]
        print("    Clip retraction schedule (staggered to preserve strong balance):")
        print(tabulate(rows, headers=["Level", "Retraction (mm)", "Class cells"], tablefmt="grid"))
    return cells


def _voxelize(stl_path, vox, margin_cells):
    """
    Voxelize the mesh at the classification resolution.

    Indices come back relative to the voxel grid's own origin, so the mesh is
    never translated -- that both avoids mutating a large mesh and sidesteps a
    segfault in Open3D 0.18's `translate` on high-vertex-count meshes.
    """
    mesh = o3d.io.read_triangle_mesh(str(stl_path))
    if len(mesh.vertices) == 0:
        raise ValueError(f"Mesh is empty or invalid: {stl_path}")

    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=vox)
    voxels = voxel_grid.get_voxels()
    indices = np.array([v.grid_index for v in voxels], dtype=int) if voxels else np.empty((0, 3), int)
    if not indices.size:
        raise ValueError(f"Mesh produced no voxels at {vox} m: {stl_path}")

    # Indices can be negative relative to the reported origin, so rebase on the
    # actual minimum. An empty margin on all sides then guarantees the array
    # border is free space, giving the flood fill a valid exterior seed.
    lo = indices.min(axis=0)
    indices = indices - lo + margin_cells
    shape = indices.max(axis=0) + 1 + margin_cells

    solid = np.zeros(tuple(shape), bool)
    solid[indices[:, 0], indices[:, 1], indices[:, 2]] = True
    origin = np.asarray(voxel_grid.origin, float) + (lo - margin_cells) * vox
    return solid, origin


def _flood_exterior(free):
    """Label free space and return everything connected to the array border."""
    labels, _ = ndimage.label(free, structure=_FLOOD_STRUCT)
    border = np.concatenate(
        [
            labels[0].ravel(),
            labels[-1].ravel(),
            labels[:, 0].ravel(),
            labels[:, -1].ravel(),
            labels[:, :, 0].ravel(),
            labels[:, :, -1].ravel(),
        ]
    )
    outside = np.unique(border)
    outside = outside[outside > 0]
    return np.isin(labels, outside)


def mirror_match(mask, axis=1):
    """
    Fraction of set cells that survive mirroring about the mask's own mid-plane.

    A diagnostic, not a correction: it separates asymmetry that comes from the
    geometry (an exhaust, a tunnel, a fuel tank) from asymmetry introduced by
    classification, which is the question that matters when a supposedly
    symmetric run comes out lopsided.
    """
    idx = np.argwhere(mask)
    if not idx.size:
        return float("nan")
    lo, hi = idx[:, axis].min(), idx[:, axis].max()
    band = np.take(mask, np.arange(lo, hi + 1), axis=axis)
    return 100.0 * (band & np.flip(band, axis=axis)).sum() / band.sum()


def interior_from_solid(solid, seal_cells, smooth_cells=0):
    """
    Interior cavity of an already-voxelized solid, as a boolean array.

    Closing seals narrow openings so the flood fill cannot leak into cavities
    through door gaps, panel shut lines or grille slots. Because closing adds
    solid near the openings it bridged, the cavity it finds is slightly
    undersized, so it is grown back afterwards and trimmed to the true free
    space.

    `smooth_cells` then opens and closes the cavity to drop single-cell spurs
    and fill single-cell nicks. The cavity boundary becomes the shape of the
    refinement transitions inside the body, so its raggedness is visible in the
    final mesh.
    """
    if seal_cells:
        sealed = _erode(_dilate(solid, seal_cells), seal_cells) | solid
    else:
        sealed = solid

    interior = ~sealed & ~_flood_exterior(~sealed)

    if seal_cells:
        interior = _dilate(interior, seal_cells) & ~solid

    if smooth_cells:
        opened = _dilate(_erode(interior, smooth_cells), smooth_cells)
        interior = _erode(_dilate(opened, smooth_cells), smooth_cells) & ~solid
    return interior


def _apply_boxes(mask, origin, vox, boxes, value):
    """Force axis-aligned world-space boxes to `value` in the mask."""
    for box in boxes or []:
        lo = (np.asarray(box[:3], float) - origin) / vox
        hi = (np.asarray(box[3:], float) - origin) / vox
        lo = np.maximum(np.floor(lo).astype(int), 0)
        hi = np.minimum(np.ceil(hi).astype(int), mask.shape)
        if np.all(hi > lo):
            mask[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] = value
    return mask


def _cache_key(stl_path, params):
    h = hashlib.blake2b(digest_size=16)
    with open(stl_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    h.update(json.dumps(params, sort_keys=True).encode())
    return h.hexdigest()


def level_voxel_size(base_vox, level):
    """
    Cell size of a refinement level, matching the mesh's own level numbering.

    Because a level is by definition a power-of-two multiple of the finest
    voxel, one classification cell is always a whole number of voxels at every
    level -- so resampling the mask onto a level grid gives uniform steps with
    no rounding.
    """
    if level < 0:
        raise ValueError(f"resolution_level must be >= 0, got {level}")
    return base_vox * 2**int(level)


def compute_interior_mask(
    stl_path,
    base_vox,
    resolution_level=1,
    seal_voxels=4,
    smooth_voxels=1,
    margin_cells=4,
    force_exterior=None,
    force_interior=None,
    cache=True,
    export_vti=False,
    verbose=True,
):
    """
    Locate free space enclosed by the vehicle body.

    Returns an `InteriorMask` over the free voxels that the exterior flood fill
    could not reach. See the module docstring for the millimetre knobs.
    """
    tic = time.perf_counter()
    stl_path = Path(stl_path)
    class_vox = level_voxel_size(base_vox, resolution_level)

    # Closing with a cube of radius r bridges a slot roughly 2r cells wide.
    seal_cells = int(np.ceil(seal_voxels / 2.0))
    smooth_cells = int(smooth_voxels)

    params = {
        "class_vox": class_vox,
        "seal_cells": seal_cells,
        "smooth_cells": smooth_cells,
        "margin_cells": margin_cells,
        "force_exterior": force_exterior or [],
        "force_interior": force_interior or [],
    }
    cache_path = None
    if cache:
        cache_dir = stl_path.parent / ".interior_cache"
        cache_path = cache_dir / f"{_cache_key(stl_path, params)}.npz"
    if cache_path is not None and cache_path.exists():
        data = np.load(cache_path)
        solid, interior, origin = data["solid"], data["mask"], data["origin"]
        if verbose:
            print(f"    Interior mask loaded from cache ({time.perf_counter() - tic:0.1f}s)")
    else:
        solid, origin = _voxelize(stl_path, class_vox, margin_cells)
        interior = interior_from_solid(solid, seal_cells, smooth_cells)

        interior = _apply_boxes(interior, origin, class_vox, force_exterior, False)
        if force_interior:
            forced = _apply_boxes(np.zeros_like(interior), origin, class_vox, force_interior, True)
            interior |= forced & ~solid

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, mask=interior, solid=solid, origin=origin, vox=class_vox)

    if verbose:
        free = int((~solid).sum())
        frac = 100.0 * interior.sum() / free if free else 0.0
        effective_seal = max(class_vox, 2 * seal_cells * class_vox) * 1000
        print(f"    Grid: {tuple(solid.shape)} at level {resolution_level} ({class_vox * 1000:.0f} mm cells)")
        print(f"    Seals gaps up to ~{effective_seal:.0f} mm (grid {class_vox * 1000:.0f} mm, closing {seal_cells} cell)")
        print(f"    Cavity smoothing: {smooth_cells} cell ({smooth_cells * class_vox * 1000:.0f} mm)")
        print(f"    Interior cavity: {interior.sum():,} cells ({frac:.1f}% of free space)")
        print(f"    Lateral symmetry: geometry {mirror_match(solid):.1f}%, cavity {mirror_match(interior):.1f}%")
        print(f"    Interior detection in {time.perf_counter() - tic:0.1f} seconds")

    result = InteriorMask(interior, origin, class_vox)

    if export_vti:
        # Debug aid only: never let a missing writer or optional dependency take
        # down a mesh run.
        path = stl_path.with_name(f"{stl_path.stem}_interior.vti")
        try:
            _export_vti(result, solid, path)
            print(f"    Interior mask written to {path}")
        except Exception as exc:
            print(f"    Interior mask VTI export skipped: {exc}")

    return result


def _export_vti(interior_mask, solid, path):
    """Write the classification grid for visual inspection (0 solid, 1 exterior, 2 interior)."""
    import pyvista as pv

    field = np.where(solid, 0, np.where(interior_mask.mask, 2, 1)).astype(np.uint8)
    grid = pv.ImageData(dimensions=np.array(field.shape) + 1, spacing=(interior_mask.vox,) * 3, origin=interior_mask.origin)
    grid.cell_data["region"] = field.flatten(order="F")
    grid.save(str(path))


def build_level_clips(interior_mask, n_levels, base_vox, n_clipped_levels, verbose=True):
    """
    Build per-level `InteriorClip` masks.

    Levels at or beyond `n_clipped_levels` get no clip at all, which is safe:
    dropping a clip only ever enlarges a region, and those levels grow from an
    already-clipped finer level.
    """
    schedule = clip_schedule(n_clipped_levels, base_vox, interior_mask.vox, verbose=verbose) if n_clipped_levels else []

    # Solid voxels within one classification cell of a cavity are treated as
    # facing it, and so do not seed the wake kernel.
    seed_mask = interior_mask.dilated(1)

    # Erode incrementally rather than re-eroding the full mask per level, and
    # stop once the cavity is consumed: retraction grows as 2^level, so a deep
    # table would otherwise spend hundreds of passes producing empty masks.
    clips = []
    current, done = interior_mask, 0
    for level in range(n_levels):
        if level >= len(schedule):
            clips.append(None)
            continue
        if current.mask.any():
            current = current.eroded(schedule[level] - done)
            done = schedule[level]
        clips.append(InteriorClip(seed=seed_mask, clip=current))

    if verbose:
        empty = [level for level, clip in enumerate(clips) if clip is not None and not clip.clip.mask.any()]
        if empty:
            print(f"    Retraction consumes the cavity at level {empty[0]}; levels {empty[0]}+ are effectively unclipped")
    return clips
