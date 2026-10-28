import numpy as np
import os
import open3d as o3d
from typing import Any
import time
from pathlib import Path
from tabulate import tabulate

import neon
import warp as wp

DEVICE = "cuda"

def generate_mesh(
    levels,
    stl_name,
    voxSize,
    padding_table=None,
    domainMultiplier=None,
    close=True,
    ground_refinement_level=-1,
    ground_voxel_height=4,
    downsample=-1,
):
    """
    Generate a multi-resolution voxel grid based on an STL file.

    This function serves as a high-level interface to create a multi-resolution voxel grid by
    voxelizing an STL file and processing it across multiple resolution levels.

    Parameters
    ----------
    levels : int
        The number of resolution levels in the multi-resolution grid. Must be a positive integer.
    stl_name : str
        The file path or name of the STL file to be voxelized (e.g., "sphere.stl").
    voxSize : float
        The voxel size at the finest resolution level, in meters. Must be positive.
    padding_table : list of lists
        A list where each inner list contains six integers [xn, xp, yn, yp, zn, zp]
        representing padding in negative and positive x, y, z directions for each level.
    domainMultiplier : dictionary
        A 6-element dictionary array specifying scale in a given bounding box direction 'x', '-x', etc
    close : bool, optional
        If True, applies a closing operation to fill gaps or islands in the voxel grid at each level.
        Default is True.
    ground_refinement_level : int, optional
        The level at which to apply ground refinement (e.g., adding a solid ground layer).
        If -1, no ground refinement is performed. Default is -1.
    ground_voxel_height : int, optional
        The number of voxels to use during ground_refinement.
    downsample : int, optional
        The highest level (inclusive) to downsample when saving data, doubling voxel sizes for levels
        0 to `downsample`. If -1, no downsampling is applied. Default is -1.

    Returns
    -------
    level_data : list of tuples
        A list where each tuple corresponds to a resolution level and contains:
        - dr : numpy.ndarray
            The voxel matrix (3D boolean array) for the level.
        - v : float
            The voxel size for the level, in meters (adjusted for level).
        - dOrigin : numpy.ndarray
            The origin coordinates (x, y, z) of the voxel grid for the level, in meters.
        - l : int
            The level number (0 to `levels - 1`).
    """
    if not padding_table:
        padding_table = [
            [2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2],
        ]
    if not domainMultiplier:
        domainMultiplier = {
            "-x": 0,
            "x": 0,
            "-y": 0,
            "y": 0,
            "z": 0,
            "-z": 0,
        }
    kernel = calculate_kernel(padding_table)
    return makeMesh(
        levels,
        stl_name,
        voxSize,
        kernel,
        domainMultiplier,
        close=close,
        ground_refinement_level=ground_refinement_level,
        ground_voxel_height=ground_voxel_height,
        downsample=downsample,
    )


# Warp kernels for voxel operations: grow, fill, remove, and crop


# Copy input matrix into a padded matrix on GPU
@wp.kernel
def copy_to_padded_kernel(input: wp.array3d(dtype=wp.uint8), padded: wp.array3d(dtype=wp.uint8), pad_x: int, pad_y: int, pad_z: int):
    i, j, k = wp.tid()
    if i < input.shape[0] and j < input.shape[1] and k < input.shape[2]:
        padded[i + pad_x, j + pad_y, k + pad_z] = input[i, j, k]


# Apply convolution to a padded matrix on GPU
@wp.kernel
def convolution_kernel(
    padded: wp.array3d(dtype=wp.uint8), kernel: wp.array3d(dtype=wp.uint8), output: wp.array3d(dtype=wp.uint8), kx: int, ky: int, kz: int
):
    i, j, k = wp.tid()
    if i >= padded.shape[0] or j >= padded.shape[1] or k >= padded.shape[2]:
        return
    sum_val = wp.uint8(0)
    for di in range(kx):
        for dj in range(ky):
            for dk in range(kz):
                if kernel[di, dj, dk] != 0:
                    ii = i + di - kx // 2
                    jj = j + dj - ky // 2
                    kk = k + dk - kz // 2
                    if 0 <= ii < padded.shape[0] and 0 <= jj < padded.shape[1] and 0 <= kk < padded.shape[2]:
                        sum_val += padded[ii, jj, kk]
    if sum_val > 0:
        output[i, j, k] = wp.uint8(1)
    else:
        output[i, j, k] = wp.uint8(0)


# Expand a voxel matrix using convolution-based growth on GPU
def grow_gpu(matrix, voxSize, origin, kernel):
    pad = (np.array(kernel.shape) * 0.5).astype(int)
    print("    Grow padding: ", pad)
    wp_matrix = wp.array(matrix.astype(np.uint8), dtype=wp.uint8, device=DEVICE)
    wp_kernel = wp.array(kernel.astype(np.uint8), dtype=wp.uint8, device=DEVICE)
    padded_shape = tuple(np.array(matrix.shape) + 2 * pad)
    padded = wp.zeros(padded_shape, dtype=wp.uint8, device=DEVICE)
    nx, ny, nz = matrix.shape
    wp.launch(kernel=copy_to_padded_kernel, dim=(nx, ny, nz), inputs=[wp_matrix, padded, pad[0], pad[1], pad[2]], device=DEVICE)
    output = wp.zeros(padded_shape, dtype=wp.uint8, device=DEVICE)
    nx, ny, nz = padded_shape
    kx, ky, kz = kernel.shape
    wp.launch(kernel=convolution_kernel, dim=(nx, ny, nz), inputs=[padded, wp_kernel, output, kx, ky, kz], device=DEVICE)
    wp.synchronize()
    r = output.numpy().astype(bool)
    kernel_shape = np.array(kernel.shape)
    originPad = origin - (kernel_shape - 1) * voxSize * 0.5
    return r, originPad


# Compute OR of 2x2x2 blocks in a padded matrix on GPU
@wp.kernel
def compute_or_blocks(padded: wp.array3d(dtype=wp.uint8), or_results: wp.array3d(dtype=wp.uint8)):
    bx, by, bz = wp.tid()
    if bx >= or_results.shape[0] or by >= or_results.shape[1] or bz >= or_results.shape[2]:
        return
    result = wp.uint8(0)
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                if padded[bx * 2 + di, by * 2 + dj, bz * 2 + dk] != 0:
                    result = wp.uint8(1)
                    break
            if result != 0:
                break
        if result != 0:
            break
    or_results[bx, by, bz] = result


# Perform binary dilation with a cross-shaped element on GPU
@wp.kernel
def binary_dilation_cross(input: wp.array3d(dtype=wp.uint8), output: wp.array3d(dtype=wp.uint8)):
    i, j, k = wp.tid()
    if i >= input.shape[0] or j >= input.shape[1] or k >= input.shape[2]:
        return
    result = wp.uint8(0)
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                if (dx == 0 and dy == 0 and dz == 0) or (abs(dx) + abs(dy) + abs(dz) == 1):
                    ii = wp.clamp(i + dx, 0, input.shape[0] - 1)
                    jj = wp.clamp(j + dy, 0, input.shape[1] - 1)
                    kk = wp.clamp(k + dz, 0, input.shape[2] - 1)
                    if input[ii, jj, kk] != 0:
                        result = wp.uint8(1)
                        break
            if result != 0:
                break
        if result != 0:
            break
    output[i, j, k] = result


# Perform binary erosion with a cross-shaped element on GPU
@wp.kernel
def binary_erosion_cross(input: wp.array3d(dtype=wp.uint8), output: wp.array3d(dtype=wp.uint8)):
    i, j, k = wp.tid()
    if i >= input.shape[0] or j >= input.shape[1] or k >= input.shape[2]:
        return
    result = wp.uint8(1)
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                if (dx == 0 and dy == 0 and dz == 0) or (abs(dx) + abs(dy) + abs(dz) == 1):
                    ii = wp.clamp(i + dx, 0, input.shape[0] - 1)
                    jj = wp.clamp(j + dy, 0, input.shape[1] - 1)
                    kk = wp.clamp(k + dz, 0, input.shape[2] - 1)
                    if input[ii, jj, kk] == 0:
                        result = wp.uint8(0)
                        break
            if result == 0:
                break
        if result == 0:
            break
    output[i, j, k] = result


# Expand OR results back to the padded matrix size on GPU
@wp.kernel
def expand_or_results(or_results: wp.array3d(dtype=wp.uint8), padded: wp.array3d(dtype=wp.uint8)):
    i, j, k = wp.tid()
    if i >= padded.shape[0] or j >= padded.shape[1] or k >= padded.shape[2]:
        return
    bx = i // 2
    by = j // 2
    bz = k // 2
    padded[i, j, k] = or_results[bx, by, bz]


# Extract the central region of a padded matrix on GPU
@wp.kernel
def extract_center(input: wp.array3d(dtype=wp.uint8), output: wp.array3d(dtype=wp.uint8), px: int, py: int, pz: int):
    i, j, k = wp.tid()
    if i < output.shape[0] and j < output.shape[1] and k < output.shape[2]:
        output[i, j, k] = input[i + px, j + py, k + pz]


# Pad an array with zeros on GPU
def pad_array_zeros_gpu(input_arr, pad_width):
    px, py, pz = pad_width
    shape = input_arr.shape
    padded_shape = (shape[0] + 2 * px, shape[1] + 2 * py, shape[2] + 2 * pz)
    padded = wp.zeros(padded_shape, dtype=wp.uint8, device=DEVICE)
    nx, ny, nz = shape
    wp.launch(kernel=copy_to_padded_kernel, dim=(nx, ny, nz), inputs=[input_arr, padded, px, py, pz], device=DEVICE)
    return padded


# Fill a voxel matrix with optional closing operation on GPU
def fill_gpu(matrix, voxSize, origin, close):
    a = (origin / voxSize) % 2
    inds = np.isclose(a, np.round(a), atol=1e-8)
    a[inds] = np.round(a[inds])
    paddingLo = np.floor(a).astype(int)
    paddingHi = np.round((matrix.shape + paddingLo) % 2).astype(int)
    originPad = origin - paddingLo * voxSize
    wp_matrix = wp.array(matrix.astype(np.uint8), dtype=wp.uint8, device=DEVICE)
    padded_shape = tuple(np.array(matrix.shape) + paddingLo + paddingHi)
    padded = wp.zeros(padded_shape, dtype=wp.uint8, device=DEVICE)
    nx, ny, nz = matrix.shape
    wp.launch(kernel=copy_to_padded_kernel, dim=(nx, ny, nz), inputs=[wp_matrix, padded, paddingLo[0], paddingLo[1], paddingLo[2]], device=DEVICE)
    or_results_shape = tuple(np.array(padded_shape) // 2)
    or_results = wp.zeros(or_results_shape, dtype=wp.uint8, device=DEVICE)
    nx, ny, nz = or_results_shape
    wp.launch(kernel=compute_or_blocks, dim=(nx, ny, nz), inputs=[padded, or_results], device=DEVICE)
    if close:
        padded_or_results = pad_array_zeros_gpu(or_results, (2, 2, 2))
        nx, ny, nz = padded_or_results.shape
        dilated = wp.zeros(padded_or_results.shape, dtype=wp.uint8, device=DEVICE)
        wp.launch(kernel=binary_dilation_cross, dim=(nx, ny, nz), inputs=[padded_or_results, dilated], device=DEVICE)
        eroded = wp.zeros(padded_or_results.shape, dtype=wp.uint8, device=DEVICE)
        wp.launch(kernel=binary_erosion_cross, dim=(nx, ny, nz), inputs=[dilated, eroded], device=DEVICE)
        nx, ny, nz = or_results_shape
        new_or_results = wp.zeros(or_results_shape, dtype=wp.uint8, device=DEVICE)
        wp.launch(kernel=extract_center, dim=(nx, ny, nz), inputs=[eroded, new_or_results, 2, 2, 2], device=DEVICE)
        or_results = new_or_results
    nx, ny, nz = padded_shape
    wp.launch(kernel=expand_or_results, dim=(nx, ny, nz), inputs=[or_results, padded], device=DEVICE)
    wp.synchronize()
    m = padded.numpy().astype(bool)
    return m, originPad


# Set specified voxel indices to False on GPU
@wp.kernel
def set_false_kernel(matrix: wp.array3d(dtype=wp.uint8), indices: wp.array(dtype=wp.int32, ndim=2), offset: wp.array(dtype=wp.int32, ndim=1)):
    tid = wp.tid()
    if tid >= indices.shape[0]:
        return
    ix = indices[tid, 0] + offset[0]
    iy = indices[tid, 1] + offset[1]
    iz = indices[tid, 2] + offset[2]
    if 0 <= ix < matrix.shape[0] and 0 <= iy < matrix.shape[1] and 0 <= iz < matrix.shape[2]:
        matrix[ix, iy, iz] = wp.uint8(0)


# Remove specified voxels from a matrix on GPU
def remove_gpu(matrix, origin, removeMat, removeOrigin, voxSize):
    offset = np.round((removeOrigin - origin) / voxSize).astype(int)
    removeIndices = np.argwhere(removeMat)
    if len(removeIndices) == 0:
        return np.copy(matrix)
    wp_matrix = wp.array(matrix.astype(np.uint8), dtype=wp.uint8, device=DEVICE)
    wp_indices = wp.array(removeIndices, dtype=wp.int32, device=DEVICE)
    wp_offset = wp.array(offset, dtype=wp.int32, device=DEVICE)
    wp.launch(kernel=set_false_kernel, dim=len(removeIndices), inputs=[wp_matrix, wp_indices, wp_offset], device=DEVICE)
    wp.synchronize()
    mat = wp_matrix.numpy().astype(bool)
    return mat


# Copy a cropped region from a matrix on GPU
@wp.kernel
def copy_cropped(mat: wp.array3d(dtype=wp.uint8), cropped: wp.array3d(dtype=wp.uint8), offset_x: int, offset_y: int, offset_z: int):
    i, j, k = wp.tid()
    if i < cropped.shape[0] and j < cropped.shape[1] and k < cropped.shape[2]:
        cropped[i, j, k] = mat[i + offset_x, j + offset_y, k + offset_z]


# Crop a voxel matrix to a specified domain on GPU
def crop_gpu(mat, origin, domainMin, domainMax, v):
    cMin = np.round((domainMin - origin) / v).astype(int)
    cMax = np.round((domainMax - origin) / v).astype(int)
    cropMin = np.maximum(cMin, 0)
    cropMax = np.minimum(cMax, mat.shape)
    cropped_shape = tuple(cropMax - cropMin)
    if any(s <= 0 for s in cropped_shape):
        return np.empty((0, 0, 0), dtype=bool), origin
    wp_mat = wp.array(mat.astype(np.uint8), dtype=wp.uint8, device=DEVICE)
    wp_cropped = wp.zeros(cropped_shape, dtype=wp.uint8, device=DEVICE)
    nx, ny, nz = cropped_shape
    wp.launch(kernel=copy_cropped, dim=(nx, ny, nz), inputs=[wp_mat, wp_cropped, cropMin[0], cropMin[1], cropMin[2]], device=DEVICE)
    wp.synchronize()
    cropMat = wp_cropped.numpy().astype(bool)
    origin = origin + cropMin * v
    return cropMat, origin


def pad_to_even(grid):
    shape = grid.shape
    # Calculate padding: 0 if even, 1 if odd, added to the upper side
    pad_width = [(0, 1 if s % 2 != 0 else 0) for s in shape]
    padded_grid = np.pad(grid, pad_width, mode="constant", constant_values=False)
    # Check if any padding was added
    padding_added = [pw[1] for pw in pad_width]  # Extract upper padding for each dimension
    if any(p > 0 for p in padding_added):
        # Map axis indices to x, y, z and report only dimensions with padding
        padding_str = ", ".join([f"{'xyz'[i]} by {p}" for i, p in enumerate(padding_added) if p > 0])
        print(f"    Padded {padding_str}")
    return padded_grid


# Voxelize an STL file using Open3D
def voxelize_stl_open3d(stl_filename, length_lbm_unit):
    """
    Voxelize an STL file using Open3D.

    Args:
        stl_filename (str): Path to the STL file.
        length_lbm_unit (float): Voxel size in meters.

    Returns:
        tuple: (voxel_matrix, origin, partDomain)
            - voxel_matrix: Boolean 3D array representing the voxel grid.
            - origin: Coordinates of the grid origin.
            - partDomain: List of [max_bound, min_bound].
    """
    tic = time.perf_counter()
    mesh = o3d.io.read_triangle_mesh(stl_filename)
    if len(mesh.vertices) == 0:
        raise ValueError("The mesh is empty or invalid.")
    print(f"    Number of vertices: {len(mesh.vertices):,}")
    print(f"    Number of triangles: {len(mesh.triangles):,}")
    toc = time.perf_counter()
    print(f"    Model read in {toc - tic:0.1f} seconds")
    
    tic = time.perf_counter()
    
    # Compute the bounds of the mesh
    min_bound = np.asarray(mesh.get_min_bound())
    max_bound = np.asarray(mesh.get_max_bound())
    
    # Snap bounds to voxel grid
    min_bound = np.floor(min_bound / length_lbm_unit) * length_lbm_unit
    max_bound = np.ceil(max_bound / length_lbm_unit) * length_lbm_unit
    
    # Compute grid size with padding to avoid index errors
    grid_size = np.ceil((max_bound - min_bound) / length_lbm_unit).astype(int) + 1  # Add 1 for padding
    
    # Translate mesh to align min_bound with origin
    mesh.translate(-min_bound)
    
    # Voxelize the mesh
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=length_lbm_unit)
    voxels = voxel_grid.get_voxels()
    voxel_indices = np.array([v.grid_index for v in voxels], dtype=int) if voxels else np.empty((0, 3), dtype=int)
    
    # Ensure indices are within bounds
    voxel_indices = np.clip(
        voxel_indices,
        [0, 0, 0],
        grid_size - 1  # Clip to max valid index
    )
    
    # Create the voxel matrix
    voxel_matrix = np.zeros(grid_size, dtype=bool)
    if voxel_indices.size > 0:
        voxel_matrix[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = True
    
    origin = min_bound
    partDomain = [max_bound, min_bound]
    
    toc = time.perf_counter()
    print(f"    Grid created in {toc - tic:0.1f} seconds")
    
    return voxel_matrix, origin, partDomain


# Print a table of padding values for each level
def print_padding_table(padding_values):
    headers = ["Level", "X-", "X+", "Y-", "Y+", "Z-", "Z+"]
    table = [[level] + values for level, values in enumerate(padding_values)]
    print(tabulate(table, headers=headers, tablefmt="grid"))


# SWT RC1: optional refinement topology cleanup. All functions are contained here;
# no changes to the Neon solver, JSON schema or existing Warp voxelization path.
# Enable with SWT_MRES_TOPOLOGY=apply, or inspect without modifications with =audit.
# Default off preserves the original makeMesh output exactly.
class _SWTMresTopologyError(RuntimeError):
    pass


def _swt_mres_origin(record):
    """Origin in this level's integer lattice coordinates (pre-conversion data)."""
    mask, spacing, origin, level = record
    if np.ndim(mask) != 3 or np.ndim(origin) != 1 or len(origin) != 3:
        raise _SWTMresTopologyError('Each level needs a 3D mask and a 3-vector origin')
    if not np.isfinite(spacing) or spacing <= 0:
        raise _SWTMresTopologyError('Invalid level spacing')
    lattice = np.asarray(origin, dtype=np.float64) / float(spacing)
    if not np.all(np.isfinite(lattice)) or not np.allclose(lattice, np.rint(lattice), rtol=0, atol=1e-5):
        raise _SWTMresTopologyError(f'L{level} origin is not aligned to its lattice')
    return np.rint(lattice).astype(np.int64)


def _swt_mres_children_count(fine, fine_origin, coarse_origin, coarse_shape):
    """Count L0 children for each L1 lattice site, including a 1-cell halo.

    A global L1 index (x,y,z) corresponds to L0 indices (2*x+a,2*y+b,2*z+c)
    for child parities a,b,c in {0,1}. Both origins are in level-local units.
    No 3D repeated/upscaled L0 array is allocated.
    """
    halo = np.asarray(coarse_shape, dtype=np.int64) + 2
    first = coarse_origin - 1
    beyond = first + halo
    fine_shape = np.asarray(fine.shape, dtype=np.int64)
    counts = np.zeros(tuple(int(x) for x in halo), dtype=np.uint8)
    for px in (0, 1):
        for py in (0, 1):
            for pz in (0, 1):
                parity = np.asarray((px, py, pz), dtype=np.int64)
                lower = np.maximum(first, -np.floor_divide(-(fine_origin - parity), 2))
                upper = np.minimum(beyond, np.floor_divide(fine_origin + fine_shape - 1 - parity, 2) + 1)
                if np.any(lower >= upper):
                    continue
                extent = upper - lower
                src0 = 2 * lower + parity - fine_origin
                dst0 = lower - first
                source_slice = tuple(slice(int(src0[i]), int(src0[i] + 2 * extent[i]), 2) for i in range(3))
                target_slice = tuple(slice(int(dst0[i]), int(dst0[i] + extent[i])) for i in range(3))
                part = fine[source_slice]
                if part.shape != tuple(int(x) for x in extent):
                    raise _SWTMresTopologyError('Invalid L0/L1 origin/shape mapping')
                counts[target_slice] += part.astype(np.uint8, copy=False)
    return counts


def _swt_mres_bad_l1(coarse, fine_child_counts):
    """Mark active L1 cells with opposite fine faces or >=3 fine-facing faces.

    A face is classified as L0-facing only if its neighboring L1-volume cube
    has all eight L0 children active. Partial cubes are NOT presumed to be L0.
    """
    sx, sy, sz = coarse.shape
    p = fine_child_counts
    # Keep only one pair of face slabs in memory at a time. Large SWT masks
    # should not allocate six additional full-domain boolean arrays here.
    opposite = np.zeros(coarse.shape, dtype=bool)
    face_count = np.zeros(coarse.shape, dtype=np.uint8)
    for minus, plus in (
        (p[:-2, 1:-1, 1:-1], p[2:, 1:-1, 1:-1]),
        (p[1:-1, :-2, 1:-1], p[1:-1, 2:, 1:-1]),
        (p[1:-1, 1:-1, :-2], p[1:-1, 1:-1, 2:]),
    ):
        low_fine = minus == 8
        high_fine = plus == 8
        opposite |= low_fine & high_fine
        face_count += low_fine
        face_count += high_fine
    opposite &= coarse
    multiple = coarse & (face_count >= 3)
    return opposite, multiple


def _swt_mres_expand_within_l1(coarse, seed, rings):
    """Six-connected one-L1-cell buffer, never crossing an inactive L1 cell."""
    chosen = seed.copy()
    frontier = seed.copy()
    for _ in range(rings):
        p = np.pad(frontier, 1, constant_values=False)
        neighborhood = (p[:-2, 1:-1, 1:-1] | p[2:, 1:-1, 1:-1] |
                        p[1:-1, :-2, 1:-1] | p[1:-1, 2:, 1:-1] |
                        p[1:-1, 1:-1, :-2] | p[1:-1, 1:-1, 2:])
        frontier = neighborhood & coarse & ~chosen
        chosen |= frontier
        if not np.any(frontier):
            break
    return chosen


def _swt_mres_safe_selection(coarse, fine, counts, chosen, fine_origin, coarse_origin):
    """Filter promotions to avoid L0/L1 overlap, invalid child bounds or L0/L2
    face/edge/corner contacts. Return safe L1 coordinates and blocked count.

    This is deliberately conservative: unsafe candidates remain L1. They are
    reported, never silently written into an incomplete fine-grid stencil.
    """
    coords = np.argwhere(chosen)
    if not len(coords):
        return coords, 0
    starts = 2 * (coords + coarse_origin) - fine_origin
    good = np.all((starts >= 0) & (starts + 2 <= np.asarray(fine.shape)), axis=1)
    center_counts = counts[tuple((coords + 1).T)]
    good &= center_counts == 0
    # Every D3Q27 neighboring L1-volume cube must be covered by L1 or
    # a COMPLETE 2x2x2 L0 cube, including across the coarse-mask boundary.
    supported = np.pad(coarse, 1, constant_values=False) | (counts == 8)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                good &= supported[coords[:, 0] + 1 + dx,
                                  coords[:, 1] + 1 + dy,
                                  coords[:, 2] + 1 + dz]
    return coords[good], int(np.count_nonzero(~good))


def _swt_mres_presence_at_level(fine, fine_origin, coarse_origin, coarse_shape, ratio):
    """OR-reduce active occupancy through successive 2:1 lattices.

    Complexity is proportional to the source mask size, rather than ratio^3;
    global lattice indices (including negative origins) remain aligned.
    Returns a one-cell halo around the target coarse mask.
    """
    if ratio < 2 or ratio & (ratio - 1):
        raise _SWTMresTopologyError('Level ratio must be a power of two')
    data = np.asarray(fine, dtype=bool)
    base = np.asarray(fine_origin, dtype=np.int64).copy()
    remaining = ratio
    while remaining > 1:
        low = np.mod(base, 2).astype(int)
        high = (-(np.asarray(data.shape) + low)) % 2
        padded = np.pad(data, tuple((int(low[i]), int(high[i])) for i in range(3)),
                        mode='constant', constant_values=False)
        data = (padded[0::2,0::2,0::2] | padded[0::2,0::2,1::2] |
                padded[0::2,1::2,0::2] | padded[0::2,1::2,1::2] |
                padded[1::2,0::2,0::2] | padded[1::2,0::2,1::2] |
                padded[1::2,1::2,0::2] | padded[1::2,1::2,1::2])
        base = np.floor_divide(base, 2)
        remaining //= 2
    result_shape = np.asarray(coarse_shape, dtype=np.int64) + 2
    result = np.zeros(tuple(result_shape), dtype=bool)
    first = np.asarray(coarse_origin, dtype=np.int64) - 1
    lower = np.maximum(base, first)
    upper = np.minimum(base + np.asarray(data.shape), first + result_shape)
    if np.all(lower < upper):
        src = lower - base
        dst = lower - first
        extent = upper - lower
        ss = tuple(slice(int(src[i]), int(src[i]+extent[i])) for i in range(3))
        ds = tuple(slice(int(dst[i]), int(dst[i]+extent[i])) for i in range(3))
        result[ds] = data[ss]
    return result


def _swt_mres_neighbor26(presence):
    """Coarse cells neighboring a finer occupied cube (face, edge, corner)."""
    result = np.zeros(tuple(np.asarray(presence.shape) - 2), dtype=bool)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if (dx, dy, dz) != (0, 0, 0):
                    result |= presence[1+dx:presence.shape[0]-1+dx,
                                       1+dy:presence.shape[1]-1+dy,
                                       1+dz:presence.shape[2]-1+dz]
    return result


def _swt_mres_repair_hierarchy(level_data, *, mode='apply'):
    """Closure of refinement hierarchy under two safety rules at EVERY level.

    1. A level L cell cannot have level L-1 on opposing faces or >=4 faces.
    2. No level L cell may have a face/edge/corner contact with a cell
       at level L-2 or finer (one-level transition / D3Q27 guard).

    A violating L cell is replaced with eight L-1 children. Changes can
    propagate through all levels until the global fixed point is reached.
    There are no pass/cell caps. Since every operation replaces one coarse
    leaf with eight finer leaves, termination is bounded by the finite domain
    at the finest resolution, subject to physical memory availability.

    Existing active cells are never deleted except through replacement by
    their exact children. Unrelated levels and all existing parameters remain.
    """
    if mode not in ('audit', 'apply'):
        raise _SWTMresTopologyError('mode must be audit or apply')
    if len(level_data) < 2:
        return level_data
    masks = [np.asarray(rec[0], dtype=bool).copy() for rec in level_data]
    origin = [_swt_mres_origin(rec) for rec in level_data]
    num_levels = len(masks)
    for i in range(1, num_levels):
        if not np.isclose(float(level_data[i][1]), 2 * float(level_data[i-1][1])):
            raise _SWTMresTopologyError('Level spacings must be 2:1, in finest-first order')
    total_by_level = [0] * num_levels
    iteration = 0
    while True:
        # Decide the entire current wave from one consistent snapshot.
        planned = [np.zeros(m.shape, dtype=bool) for m in masks]
        bad_by_level = []
        gap_by_level = []
        for c in range(1, num_levels):
            counts = _swt_mres_children_count(masks[c-1], origin[c-1], origin[c], masks[c].shape)
            forbidden = _swt_mres_forbidden_l1(masks[c], counts)
            planned[c] |= forbidden
            bad_by_level.append(int(forbidden.sum()))
            gaps = np.zeros(masks[c].shape, dtype=bool)
            for f in range(c-2, -1, -1):
                presence = _swt_mres_presence_at_level(
                    masks[f], origin[f], origin[c], masks[c].shape, 1 << (c-f))
                gaps |= _swt_mres_neighbor26(presence)
            gaps &= masks[c]
            planned[c] |= gaps
            gap_by_level.append(int(gaps.sum()))
        count = sum(int(p.sum()) for p in planned)
        if iteration == 0:
            print('[SWT topology] %s initial forbidden per L1..LN=%s '
                  'skipped-level guards=%s total_promotions=%d' %
                  (mode.upper(), bad_by_level, gap_by_level, count), flush=True)
        if mode == 'audit':
            print('[SWT topology] AUDIT: no meshes modified', flush=True)
            return level_data
        if count == 0:
            break
        # Coarsest first so newly generated children can themselves be refined
        # on the next pass. Every source cell must have no existing immediate
        # child; otherwise the input hierarchy already overlaps.
        for c in range(num_levels-1, 0, -1):
            coords = np.argwhere(planned[c])
            if not len(coords):
                continue
            fine = masks[c-1]
            coarse = masks[c]
            children = _swt_mres_children_count(fine, origin[c-1], origin[c], coarse.shape)
            if np.any(children[tuple((coords + 1).T)]):
                raise _SWTMresTopologyError(
                    'Existing overlapping level masks at L%d/L%d; refusing to replace occupied cells' %
                    (c-1,c))
            starts_global = (coords + origin[c]) * 2
            lo = np.minimum(origin[c-1], starts_global.min(axis=0))
            hi = np.maximum(origin[c-1] + np.asarray(fine.shape),
                            (starts_global + 2).max(axis=0))
            lo = 2 * np.floor_divide(lo, 2)
            hi = 2 * np.floor_divide(hi + 1, 2)
            if np.any(lo != origin[c-1]) or np.any(hi != origin[c-1] + np.asarray(fine.shape)):
                expanded = np.zeros(tuple(hi-lo), dtype=bool)
                offset = origin[c-1] - lo
                sl = tuple(slice(int(offset[k]), int(offset[k]+fine.shape[k])) for k in range(3))
                expanded[sl] = fine
                fine = expanded
                origin[c-1] = lo
                masks[c-1] = fine
            starts = starts_global - origin[c-1]
            for i in (0,1):
                for j in (0,1):
                    for k in (0,1):
                        fine[starts[:,0]+i, starts[:,1]+j, starts[:,2]+k] = True
            coarse[tuple(coords.T)] = False
            total_by_level[c] += len(coords)
        iteration += 1
        if iteration <= 5 or iteration % 10 == 0:
            print('[SWT topology] wave=%d promotions=%d cumulative=%d levels=%s' %
                  (iteration, count, sum(total_by_level), total_by_level[1:]), flush=True)

    # Independent final verification: both conditions on every pair.
    for c in range(1, num_levels):
        counts = _swt_mres_children_count(masks[c-1], origin[c-1], origin[c], masks[c].shape)
        if np.any(_swt_mres_forbidden_l1(masks[c], counts)):
            raise _SWTMresTopologyError('Nonconvergence at L%d/L%d' % (c-1,c))
        for f in range(c-2, -1, -1):
            presence = _swt_mres_presence_at_level(
                masks[f], origin[f], origin[c], masks[c].shape, 1 << (c-f))
            if np.any(masks[c] & _swt_mres_neighbor26(presence)):
                raise _SWTMresTopologyError('Remaining skipped-level contact L%d/L%d' % (f,c))
    output = []
    for i, rec in enumerate(level_data):
        output.append((masks[i].astype(rec[0].dtype, copy=False), rec[1],
                       origin[i].astype(float)*float(rec[1]), rec[3]))
    print('[SWT topology] CONVERGED waves=%d total_promoted=%d '
          'promoted_by_level=%s remaining_forbidden=0 skipped_level_contacts=0' %
          (iteration, sum(total_by_level), total_by_level[1:]), flush=True)
    return output


def _swt_mres_topology_cleanup(level_data, *, mode='audit', rings=0,
                               max_fraction=None, max_cells=None,
                               max_passes=None, strict=True):
    """Backward-compatible entry point; applies recursive repair at all levels."""
    return _swt_mres_repair_hierarchy(level_data, mode=mode)


def _swt_mres_forbidden_l1(coarse, counts):
    """Opposite fine faces OR >=4 fine-facing faces at any adjacent pair."""
    faces = np.zeros(coarse.shape, dtype=np.uint8)
    opposite = np.zeros(coarse.shape, dtype=bool)
    for low, high in (
        (counts[:-2,1:-1,1:-1] == 8, counts[2:,1:-1,1:-1] == 8),
        (counts[1:-1,:-2,1:-1] == 8, counts[1:-1,2:,1:-1] == 8),
        (counts[1:-1,1:-1,:-2] == 8, counts[1:-1,1:-1,2:] == 8),
    ):
        faces += low
        faces += high
        opposite |= low & high
    return coarse & (opposite | (faces >= 4))


def _swt_mres_topology_selftest():
    """CPU-only regression tests for recursive, all-level closure."""
    z=np.zeros(3)
    coarse = np.ones((7,7,7),bool)
    fine = np.zeros((14,14,14),bool)
    for idx in ((3,3,2),(3,3,4)):
        coarse[idx]=False
        fine[2*idx[0]:2*idx[0]+2, 2*idx[1]:2*idx[1]+2, 2*idx[2]:2*idx[2]+2]=True
    old=[(fine,1.,z,0),(coarse,2.,z,1)]
    assert _swt_mres_topology_cleanup(old,mode='audit') is old
    out=_swt_mres_topology_cleanup(old,mode='apply')
    counts=_swt_mres_children_count(out[0][0],_swt_mres_origin(out[0]),_swt_mres_origin(out[1]),out[1][0].shape)
    assert not np.any(_swt_mres_forbidden_l1(out[1][0],counts))
    assert np.array_equal(old[0][0],fine)
    # A level-2 cube two L2-index steps from a level-0 cube; refinement
    # must provide an L1 buffer without touching existing fine cells.
    a=np.zeros((24,24,24),bool)
    b=np.zeros((12,12,12),bool)
    c=np.zeros((6,6,6),bool)
    a[8:10,8:10,8:10]=True
    c[3,2,2]=True
    multi=[(a,1.,z,0),(b,2.,z,1),(c,4.,z,2)]
    fixed=_swt_mres_topology_cleanup(multi,mode='apply')
    assert not fixed[2][0][3,2,2]
    assert fixed[1][0].sum() >= 8
    for l in (1,2):
        cnt=_swt_mres_children_count(fixed[l-1][0],_swt_mres_origin(fixed[l-1]),_swt_mres_origin(fixed[l]),fixed[l][0].shape)
        assert not np.any(_swt_mres_forbidden_l1(fixed[l][0],cnt))
    print('[SWT topology] SELFTEST PASS all levels')

# Generate a multi-level voxel mesh from an STL file
def makeMesh(levels, filename, voxSize, kernel, domainMultiplier, close=True, ground_refinement_level=-1, ground_voxel_height=4, downsample=-1):
    stem = Path(filename).stem
    tic = time.perf_counter()

    matrix, origin, partDomain = voxelize_stl_open3d(filename, voxSize)
    partSize = partDomain[0] - partDomain[1]

    domainMin = np.array([0, 0, 0], float)
    domainMax = np.array(
        [
            partDomain[0][0] + (domainMultiplier["x"] * partSize[0]),
            partDomain[0][1] + (domainMultiplier["y"] * partSize[1]),
            partDomain[0][2] + (domainMultiplier["z"] * partSize[2]),
        ],
        float,
    )

    # Store original bounds
    orig_domainMin = domainMin.copy()
    orig_domainMax = domainMax.copy()

    domainSize = domainMax - domainMin

    # Calculate the smallest domain dimension
    min_domain_size = np.min(domainSize)
    threshold = min_domain_size / 8.0  # 1/8th of the smallest dimension
    ratio = threshold / voxSize

    # Check if the base voxel size is valid
    if ratio <= 0:
        raise ValueError("Voxel size is larger than 1/8th of the smallest domain dimension.")

    # Calculate maximum allowable levels
    max_levels_allowed = max(1, int(np.floor(np.log2(ratio))) + 1)

    # Adjust levels if necessary
    if levels > max_levels_allowed:
        print(f"Reducing levels from {levels} to {max_levels_allowed} to satisfy voxel size constraint.")
        levels = max_levels_allowed

    maxVoxSize = voxSize * pow(2, levels - 1)
    domainMin = np.round(domainMin / maxVoxSize) * maxVoxSize
    domainMax = np.ceil(domainMax / maxVoxSize) * maxVoxSize

    print("\n" + "=" * 100 + "\n")
    print("Meshing Configuration:")
    print(f"Model: {filename}")
    print(f"Finest level: {voxSize} meters")
    print(f"Number of levels: {levels}")
    print(f"Close islands: {close}")
    if ground_refinement_level != -1:
        print(f"Ground refinement level: {ground_refinement_level}")
    print("Adjusted domain coordinates: ", domainMin, ", ", domainMax)
    print("Voxel growth strategy:")
    #print_padding_table(padding_values)
    print("\n" + "=" * 100 + "\n")

    domainSize = domainMax - domainMin
    print("/// Make Mesh started... " + stem)
    v = voxSize

    level_data = []
    print("/// Level 0 voxel size: ", v)
    ticLevel = time.perf_counter()
    if levels == 1:
        # Calculate full domain shape
        full_shape = np.round(domainSize / voxSize).astype(int)
        df = np.ones(full_shape, dtype=bool)
        df = pad_to_even(df)  # Pad to ensure even shape
        dr = df.copy()
        dOrigin = domainMin
        level_data.append((dr, v, dOrigin, 0))
    else:
        g, origin = grow_gpu(matrix, voxSize, origin, kernel[0])
        f, origin = fill_gpu(g, voxSize, origin, close)
        df, origin = crop_gpu(f, origin, domainMin, domainMax, v)
        df = pad_to_even(df)
        dOrigin = np.copy(origin)
        dr = df.copy()  # dr is the final matrix for this level

        level_data.append((dr, v, dOrigin, 0))
        tocLevel = time.perf_counter()
        print(f"    Level defined in {tocLevel - ticLevel:0.1f} seconds")

        ground_z = domainMin[2]

        for l in range(1, levels):
            ticLevel = time.perf_counter()
            d = df[::2, ::2, ::2]
            v = v * 2
            print("/// Level", l, "voxel size:", v)
            full_shape = np.round(domainSize / v).astype(int)

            if l < levels - 1:
                dg, dOrigin = grow_gpu(d, v, dOrigin, kernel[l])
                df_natural, dOrigin = fill_gpu(dg, v, dOrigin, close)
                df_natural, dOrigin = crop_gpu(df_natural, dOrigin, domainMin, domainMax, v)
                df_natural = pad_to_even(df_natural)
            else:
                df_natural = np.ones(tuple(full_shape), bool)
                dOrigin = domainMin

            if ground_refinement_level != -1 and l == ground_refinement_level:
                df_ground = np.zeros(tuple(full_shape), bool)
                dOrigin_ground = domainMin
                ground_z_index = int(np.round((ground_z - dOrigin_ground[2]) / v))
                # Ground Voxel Thickness
                n_thick = ground_voxel_height
                if 0 <= ground_z_index < full_shape[2]:
                    end_z = min(ground_z_index + n_thick, full_shape[2])
                    df_ground[:, :, ground_z_index:end_z] = True

                offset = np.round((dOrigin - dOrigin_ground) / v).astype(int)
                x0, y0, z0 = offset
                x1, y1, z1 = x0 + df_natural.shape[0], y0 + df_natural.shape[1], z0 + df_natural.shape[2]
                x0, y0, z0 = np.maximum([x0, y0, z0], 0)
                x1, y1, z1 = np.minimum([x1, y1, z1], full_shape)
                df = np.zeros(tuple(full_shape), bool)
                if x1 > x0 and y1 > y0 and z1 > z0:
                    df[x0:x1, y0:y1, z0:z1] = df_natural[: (x1 - x0), : (y1 - y0), : (z1 - z0)]
                df |= df_ground
                dOrigin = dOrigin_ground
            else:
                df = df_natural

            dr = remove_gpu(df, dOrigin, d, origin, v)
            level_data.append((dr, v, dOrigin, l))
            tocLevel = time.perf_counter()
            print(f"    Level defined in {tocLevel - ticLevel:0.1f} seconds")
            origin = np.copy(dOrigin)

    # Optional SWT topology stage: after the ORIGINAL L0..LN grow/fill/remove,
    # before the ORIGINAL statistics, downsample and level-index conversion.
    # Default off: the original mesh is bit-for-bit unchanged.
    swt_mode = os.environ.get('SWT_MRES_TOPOLOGY', 'off').strip().lower()
    if swt_mode not in ('', 'off', 'audit', 'apply'):
        raise _SWTMresTopologyError('SWT_MRES_TOPOLOGY must be off, audit or apply')
    if swt_mode in ('audit', 'apply'):
        if downsample >= 0:
            raise _SWTMresTopologyError('Topology cleanup cannot be combined with downsample export')
        level_data = _swt_mres_topology_cleanup(
            level_data, mode=swt_mode,
            rings=0,
            max_fraction=None,
            max_cells=None,
            max_passes=None,
            strict=True,
        )

    toc = time.perf_counter()

    print()
    print("/// Mesh Data Report")
    finest_possible_voxels = int(np.prod(domainSize / voxSize))
    total_voxels_billions = finest_possible_voxels / 1e9
    print(f"    Total domain size: {total_voxels_billions:.2f} billion voxels (full dense at {voxSize} m)")

    total_voxel_count = sum(np.sum(dr) for dr, _, _, _ in level_data)
    total_voxel_count_millions = total_voxel_count / 1e6
    print(f"    Total voxel count: {total_voxel_count_millions:.2f} million")

    percentage_reduction = ((finest_possible_voxels - total_voxel_count) / finest_possible_voxels) * 100 if finest_possible_voxels > 0 else 0
    print(f"    Percentage reduction: {percentage_reduction:.4f}% (vs. uniform dense grid)")

    print("    Voxel distribution per level:")
    headers = ["Level", "Voxel Size (m)", "Voxels (M)", "Percentage (%)", "Computation (%)"]
    table_data = []
    # Calculate computational work
    num_levels = len(level_data)
    comp_work = []
    for l, (dr, v, _, _) in enumerate(level_data):
        voxel_count = np.sum(dr)
        # Inner iterations: finest level (l=0) has 2^(num_levels-1), coarsest (l=num_levels-1) has 2^0=1
        inner_iterations = 2 ** (num_levels - 1 - l)
        work = voxel_count * inner_iterations
        comp_work.append(work)
    total_work = sum(comp_work)
    for l, (dr, v, _, _) in enumerate(level_data):
        voxel_count = np.sum(dr)
        voxel_count_millions = voxel_count / 1e6
        percentage = (voxel_count / total_voxel_count) * 100 if total_voxel_count > 0 else 0
        comp_percentage = (comp_work[l] / total_work) * 100 if total_work > 0 else 0
        table_data.append([l, v, f"{voxel_count_millions:.2f}", f"{percentage:.2f}", f"{comp_percentage:.2f}"])
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print()

    # Downsample levels up to (and including) the given 'downsample' threshold.
    if downsample >= 0:
        print(f"Downsampling levels 0 to {downsample} for file saving; doubling their voxel sizes.")
        for i, (dr, v, dOrigin, lev) in enumerate(level_data):
            if lev <= downsample:
                dr_down = dr[::2, ::2, ::2]
                v_down = v * 2
                level_data[i] = (dr_down, v_down, dOrigin, lev)

    ##NEW SHIFT TO FIRST OCTANT
    level_data = [(dr, int(v / voxSize), np.round(dOrigin / v).astype(int), l) for dr, v, dOrigin, l in level_data]

    # Domain Adjustment Report
    print("/// Domain Extension Report")
    print(f"    Original domain: {orig_domainMin} to {orig_domainMax}")
    print(f"    Adjusted domain: {domainMin} to {domainMax}")

    # Calculate extensions in terms of finest voxels
    extension_pos = (domainMax - orig_domainMax) / voxSize
    extension_neg = (orig_domainMin - domainMin) / voxSize

    print("    Extension in positive directions (in finest voxels):")
    print(f"     +x: {extension_pos[0]:.2f}")
    print(f"     +y: {extension_pos[1]:.2f}")
    print(f"     +z: {extension_pos[2]:.2f}")
    print("    Extension in negative directions (in finest voxels):")
    print(f"     -x: {extension_neg[0]:.2f}")
    print(f"     -y: {extension_neg[1]:.2f}")
    print(f"     -z: {extension_neg[2]:.2f}")
    print()

    print("/// Level Shapes Report")
    headers = ["Level", "Shape", "Origin"]
    table_data = []
    for l, (dr, _, dOrigin, _) in enumerate(level_data):
        shape_str = f"({dr.shape[0]}, {dr.shape[1]}, {dr.shape[2]})"
        origin_str = f"({dOrigin[0]}, {dOrigin[1]}, {dOrigin[2]})"
        table_data.append([l, shape_str, origin_str])
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    print(f"/// Mesh Generation Complete in {toc - tic:0.2f} seconds")
    print()        
    
    return level_data

# Calculate convolution kernels based on padding values
def calculate_kernel(padding_values):
    kernels = []
    for level, values in enumerate(padding_values):
        xn, xp, yn, yp, zn, zp = values
        x_dim = max(xp, xn) * 2 + 1
        y_dim = max(yp, yn) * 2 + 1
        z_dim = max(zp, zn) * 2 + 1
        ones_x = xp + xn + 1
        ones_y = yp + yn + 1
        ones_z = zp + zn + 1
        mid_x = (x_dim - 1) // 2
        x1 = mid_x - xp
        x2 = x_dim - (mid_x - xn)
        mid_y = (y_dim - 1) // 2
        y1 = mid_y - yp
        y2 = y_dim - (mid_y - yn)
        mid_z = (z_dim - 1) // 2
        z1 = mid_z - zp
        z2 = z_dim - (mid_z - zn)
        kernel = np.zeros((x_dim, y_dim, z_dim), bool)
        kernel[x1:x2, y1:y2, z1:z2] = np.ones((x2 - x1, y2 - y1, z2 - z1), bool)
        kernels.append(kernel)
    return kernels