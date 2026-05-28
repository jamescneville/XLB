import numpy as np
import trimesh
from typing import Any, Optional, Dict
import time
import neon
import warp as wp
from xlb.utils.utils import UnitConvertor
from scipy.spatial import cKDTree


def adjust_bbox(cuboid_max, cuboid_min, voxel_size_up):
    """
    Adjust the bounding box to the nearest points of one level finer grid that encloses the desired region.

    Args:
        cuboid_min (np.ndarray): Desired minimum coordinates of the bounding box.
        cuboid_max (np.ndarray): Desired maximum coordinates of the bounding box.
        voxel_size_up (float): Voxel size of one level higher (finer) grid.

    Returns:
        tuple: (adjusted_min, adjusted_max) snapped to grid points of one level higher.
    """
    adjusted_min = np.round(cuboid_min / voxel_size_up) * voxel_size_up
    adjusted_max = np.round(cuboid_max / voxel_size_up) * voxel_size_up
    return adjusted_min, adjusted_max


def prepare_sparsity_pattern(level_data):
    """
    Prepare the sparsity pattern for the multiresolution grid based on the level data. "level_data" is expected to be formatted as in
    the output of "make_cuboid_mesh".
    """
    num_levels = len(level_data)
    level_origins = []
    sparsity_pattern = []
    for lvl in range(num_levels):
        # Get the level mask from the level data
        level_mask = level_data[lvl][0]

        # Ensure level_0 is contiguous int32
        level_mask = np.ascontiguousarray(level_mask, dtype=np.int32)

        # Append the padded level mask to the sparsity pattern
        sparsity_pattern.append(level_mask)

        # Get the origin for this level
        level_origins.append(level_data[lvl][2])

    return sparsity_pattern, level_origins


def make_cuboid_mesh(voxel_size, cuboids, stl_filename):
    """
    Create a strongly-balanced multi-level cuboid mesh with a sequence of bounding boxes.
    Outputs mask arrays that are set to True only in regions not covered by finer levels.

    Args:
        voxel_size (float): Voxel size of the finest grid .
        cuboids (list): List of multipliers defining each level's domain.
        stl_name (str): Path to the STL file.

    Returns:
        list: Level data with mask arrays, voxel sizes, origins, and levels.
    """
    # Load the mesh and get its bounding box
    mesh = trimesh.load_mesh(stl_filename, process=False)
    assert not mesh.is_empty, ValueError("Loaded mesh is empty or invalid.")

    mesh_vertices = mesh.vertices
    min_bound = mesh_vertices.min(axis=0)
    max_bound = mesh_vertices.max(axis=0)
    partSize = max_bound - min_bound

    level_data = []
    adjusted_bboxes = []
    max_voxel_size = voxel_size * pow(2, (len(cuboids) - 1))
    # Step 1: Generate all levels and store their data
    for level in range(len(cuboids)):
        # Compute desired bounding box for this level
        cuboid_min = np.array(
            [
                min_bound[0] - cuboids[level][0] * partSize[0],
                min_bound[1] - cuboids[level][2] * partSize[1],
                min_bound[2] - cuboids[level][4] * partSize[2],
            ],
            dtype=float,
        )

        cuboid_max = np.array(
            [
                max_bound[0] + cuboids[level][1] * partSize[0],
                max_bound[1] + cuboids[level][3] * partSize[1],
                max_bound[2] + cuboids[level][5] * partSize[2],
            ],
            dtype=float,
        )

        # Set voxel size for this level
        voxel_size_level = max_voxel_size / pow(2, level)

        # Adjust bounding box to align with one level up (finer grid)
        if level > 0:
            voxel_level_up = max_voxel_size / pow(2, level - 1)
        else:
            voxel_level_up = voxel_size_level
        adjusted_min, adjusted_max = adjust_bbox(cuboid_max, cuboid_min, voxel_level_up)

        xmin, ymin, zmin = adjusted_min
        xmax, ymax, zmax = adjusted_max

        # Compute number of voxels based on level-specific voxel size
        nx = int(np.round((xmax - xmin) / voxel_size_level))
        ny = int(np.round((ymax - ymin) / voxel_size_level))
        nz = int(np.round((zmax - zmin) / voxel_size_level))
        print(f"Domain {nx}, {ny}, {nz}  Origin {adjusted_min}  Voxel Size {voxel_size_level} Voxel Level Up {voxel_level_up}")

        voxel_matrix = np.ones((nx, ny, nz), dtype=bool)

        origin = adjusted_min
        level_data.append((voxel_matrix, voxel_size_level, origin, level))
        adjusted_bboxes.append((adjusted_min, adjusted_max))

    # Step 2: Adjust coarser levels to exclude regions covered by finer levels
    for k in range(len(level_data) - 1):  # Exclude the finest level
        # Current level's data
        voxel_matrix_k = level_data[k][0]
        origin_k = level_data[k][2]
        voxel_size_k = level_data[k][1]
        nx, ny, nz = voxel_matrix_k.shape

        # Next finer level's bounding box
        adjusted_min_k1, adjusted_max_k1 = adjusted_bboxes[k + 1]

        # Compute index ranges in level k that overlap with level k+1's bounding box
        # Use epsilon (1e-10) to handle floating-point precision
        i_start = max(0, int(np.ceil((adjusted_min_k1[0] - origin_k[0] - 1e-10) / voxel_size_k)))
        i_end = min(nx, int(np.floor((adjusted_max_k1[0] - origin_k[0] + 1e-10) / voxel_size_k)))
        j_start = max(0, int(np.ceil((adjusted_min_k1[1] - origin_k[1] - 1e-10) / voxel_size_k)))
        j_end = min(ny, int(np.floor((adjusted_max_k1[1] - origin_k[1] + 1e-10) / voxel_size_k)))
        k_start = max(0, int(np.ceil((adjusted_min_k1[2] - origin_k[2] - 1e-10) / voxel_size_k)))
        k_end = min(nz, int(np.floor((adjusted_max_k1[2] - origin_k[2] + 1e-10) / voxel_size_k)))

        # Set overlapping region to zero
        voxel_matrix_k[i_start:i_end, j_start:j_end, k_start:k_end] = 0

    # Step 3 Convert to Indices from STL units
    num_levels = len(level_data)
    level_data = [(dr, int(v / voxel_size), np.round(dOrigin / v).astype(int), num_levels - 1 - l) for dr, v, dOrigin, l in level_data]

    return list(reversed(level_data))


class MultiresIO(object):
    def __init__(
        self,
        field_name_cardinality_dict,
        levels_data,
        unit_convertor: UnitConvertor = None,
        offset: Optional[tuple] = (0.0, 0.0, 0.0),
        store_precision=None,
    ):
        """
        Initialize the MultiresIO object.

        Aggressive performance path:
        - process_geometry now builds merged geometry directly.
        - _merge_duplicates is no longer called.
        - centroids are built lazily.
        - KDTree is built lazily.
        - WARP/NEON export fields are allocated lazily in get_fields_data().
        """
        import time

        start_time = time.time()

        self.unit_convertor = unit_convertor
        self.field_name_cardinality_dict = field_name_cardinality_dict
        self.levels_data = levels_data

        # Set precision policy early, before any lazy WARP allocation.
        from xlb import DefaultConfig

        if store_precision is None:
            self.store_precision = DefaultConfig.default_precision_policy.store_precision
        else:
            self.store_precision = store_precision

        self.store_dtype = self.store_precision.wp_dtype

        # Builds already-deduplicated geometry.
        coordinates, connectivity, level_id_field, total_cells = self.process_geometry(levels_data)

        assert coordinates.size != 0, "Error: No valid data to process. Check the input levels_data."

        # Transform coordinates to physical units and apply offset.
        coordinates = self._transform_coordinates(coordinates, offset)

        self.coordinates = coordinates
        self.connectivity = connectivity
        self.level_id_field = level_id_field
        self.total_cells = total_cells

        # Lazy expensive geometry-derived state.
        self._centroids = None
        self._kd_tree = None

        # Lazy WARP/NEON state.
        self.field_warp_dict = {}
        self.origin_list = None
        self.container = None

        # ---- Time-averaging state ----
        self._avg_sum: Dict[str, np.ndarray] = {}
        self._avg_weight: float = 0.0
        self._avg_active: bool = False

        # Cached finalized averages.
        self._avg_final_cache: Optional[Dict[str, np.ndarray]] = None
        self._avg_cache_weight: float = 0.0
        self._avg_cache: bool = False

        # Optional cache for derived quantities.
        self._avg_derived_cache: Dict[str, np.ndarray] = {}

        print(f"MutliResIO initialized in {time.time() - start_time}sec ")

    @property
    def centroids(self):
        """
        Lazily compute cell centroids.

        Avoids the huge temporary from:

            coordinates[connectivity]

        which has shape:

            (num_cells, 8, 3)

        For axis-aligned hex cells, corner 0 and corner 6 are opposite corners,
        so their midpoint is the cell centroid.
        """
        if self._centroids is None:
            self._centroids = 0.5 * (
                self.coordinates[self.connectivity[:, 0]] +
                self.coordinates[self.connectivity[:, 6]]
            )

        return self._centroids


    @property
    def kd_tree(self):
        """
        Lazily build KDTree only when slice/line/surface operations need it.
        Plain HDF5 export does not need this during initialization.
        """
        if self._kd_tree is None:
            from scipy.spatial import cKDTree

            self._kd_tree = cKDTree(self.centroids)

        return self._kd_tree

    def _ensure_container_inputs(self, needed_field_names=None):
        """
        Lazily allocate WARP dense fields and construct the NEON container.

        This avoids paying initialization cost for:
        - fields that are never exported
        - WARP allocations before to_hdf5 / slice / line / surface export
        """
        from xlb.compute_backend import ComputeBackend
        from xlb.grid import grid_factory

        try:
            wp_mod = wp
        except NameError:
            import warp as wp_mod

        if needed_field_names is None:
            needed_field_names = set(self.field_name_cardinality_dict.keys())
        else:
            needed_field_names = set(needed_field_names)

        if self.origin_list is None:
            self.origin_list = [
                wp_mod.vec3i(*([int(x) for x in self.levels_data[level][2]]))
                for level in range(len(self.levels_data))
            ]

        for field_name in needed_field_names:
            if field_name in self.field_warp_dict:
                continue

            cardinality = self.field_name_cardinality_dict[field_name]
            self.field_warp_dict[field_name] = []

            for level in range(len(self.levels_data)):
                box_shape = self.levels_data[level][0].shape
                grid_dense = grid_factory(box_shape, compute_backend=ComputeBackend.WARP)

                self.field_warp_dict[field_name].append(
                    grid_dense.create_field(
                        cardinality=cardinality,
                        dtype=self.store_precision,
                    )
                )

        if self.container is None:
            self.container = self._construct_neon_container()
    
    def _make_touched_vertex_mask(self, mask):
        """
        Given a cell mask of shape (nx, ny, nz), return a bool vertex mask
        of shape (nx + 1, ny + 1, nz + 1) marking all vertices touched by
        active cells.
        """
        import numpy as np

        mask = np.asarray(mask, dtype=np.bool_)
        nx, ny, nz = mask.shape

        vmask = np.zeros((nx + 1, ny + 1, nz + 1), dtype=np.bool_)

        vmask[:-1, :-1, :-1] |= mask
        vmask[1:,  :-1, :-1] |= mask
        vmask[1:,  1:,  :-1] |= mask
        vmask[:-1, 1:,  :-1] |= mask

        vmask[:-1, :-1, 1:] |= mask
        vmask[1:,  :-1, 1:] |= mask
        vmask[1:,  1:,  1:] |= mask
        vmask[:-1, 1:,  1:] |= mask

        return vmask

    def _count_touched_vertices(self, mask):
        """
        Count touched vertices for one level.
        """
        import numpy as np

        vmask = self._make_touched_vertex_mask(mask)
        n = int(np.count_nonzero(vmask))
        del vmask
        return n

    def _vertex_keys_from_vflat_chunk(
        self,
        vflat,
        vshape,
        meta,
        qmin_global,
        sx,
        sy,
        key_dtype,
        inv_finest,
    ):
        """
        Convert a chunk of flat per-level vertex-grid indices into global
        collision-free lattice keys.

        This is chunked to avoid materializing vx/vy/vz/x/y/z for the entire level.
        """
        import numpy as np

        nxp1, nyp1, nzp1 = vshape

        vz = vflat % nzp1
        tmp = vflat // nzp1
        vy = tmp % nyp1
        vx = tmp // nyp1
        del tmp

        if meta["integer_stride"]:
            stride_i = int(meta["stride_i"])
            origin_q = meta["origin_q"]

            x = origin_q[0] + vx.astype(np.int64, copy=False) * stride_i - qmin_global[0]
            y = origin_q[1] + vy.astype(np.int64, copy=False) * stride_i - qmin_global[1]
            z = origin_q[2] + vz.astype(np.int64, copy=False) * stride_i - qmin_global[2]
        else:
            voxel_size_f = float(meta["voxel_size"])
            origin_physical = meta["origin_physical"]

            x = np.rint(
                (origin_physical[0] + vx.astype(np.float64) * voxel_size_f) * inv_finest
            ).astype(np.int64)
            y = np.rint(
                (origin_physical[1] + vy.astype(np.float64) * voxel_size_f) * inv_finest
            ).astype(np.int64)
            z = np.rint(
                (origin_physical[2] + vz.astype(np.float64) * voxel_size_f) * inv_finest
            ).astype(np.int64)

            x -= qmin_global[0]
            y -= qmin_global[1]
            z -= qmin_global[2]

        del vx, vy, vz

        x = x.astype(key_dtype, copy=False)
        y = y.astype(key_dtype, copy=False)
        z = z.astype(key_dtype, copy=False)

        keys = x + sx * (y + sy * z)

        del x, y, z

        return keys

    def _fill_vertex_ids_from_rows(
        self,
        vertex_ids,
        vmask,
        meta,
        unique_keys,
        qmin_global,
        sx,
        sy,
        key_dtype,
        debug_validate=False,
    ):
        """
        Fill the temporary per-level vertex_ids grid by mapping touched vertices
        to global unique point ids row-by-row.

        This avoids doing np.searchsorted(unique_keys, keys) over the full global
        unique_keys array for every touched vertex.

        Requires integer_stride=True, which is the normal path for voxel sizes
        that are integer multiples of the finest voxel.
        """
        import numpy as np

        if not meta["integer_stride"]:
            return False

        vmask = np.asarray(vmask, dtype=np.bool_)
        nxp1, nyp1, nzp1 = vmask.shape

        origin_q = meta["origin_q"]
        stride_i = int(meta["stride_i"])

        sx_int = int(sx)
        sy_int = int(sy)

        # Local x coordinates can be reused for every row.
        local_x_all = np.arange(nxp1, dtype=np.int64)

        for vz in range(nzp1):
            qz = int(origin_q[2] + vz * stride_i - qmin_global[2])

            for vy in range(nyp1):
                row = vmask[:, vy, vz]

                if not row.any():
                    continue

                xs = np.flatnonzero(row)
                qx = origin_q[0] + local_x_all[xs] * stride_i - qmin_global[0]

                qy = int(origin_q[1] + vy * stride_i - qmin_global[1])

                line_id = qy + sy_int * qz
                line_base = sx_int * line_id

                row_start_key = key_dtype(line_base)
                row_end_key = key_dtype(line_base + sx_int)

                lo = int(np.searchsorted(unique_keys, row_start_key, side="left"))
                hi = int(np.searchsorted(unique_keys, row_end_key, side="left"))

                if hi <= lo:
                    raise RuntimeError(
                        "Internal error while filling vertex ids: empty global key row."
                    )

                row_unique_x = unique_keys[lo:hi] - key_dtype(line_base)

                qx_key = qx.astype(key_dtype, copy=False)
                pos = np.searchsorted(row_unique_x, qx_key, side="left")

                ids = lo + pos

                if debug_validate:
                    if np.any(pos >= row_unique_x.size):
                        raise RuntimeError("Vertex key lookup failed: position outside row.")
                    if np.any(row_unique_x[pos] != qx_key):
                        raise RuntimeError("Vertex key lookup failed: key mismatch.")

                vflat = (xs.astype(np.int64) * nyp1 + vy) * nzp1 + vz
                vertex_ids[vflat] = ids.astype(np.int32, copy=False)

        return True

    def _iter_touched_vertex_key_chunks(
        self,
        mask,
        meta,
        qmin_global,
        sx,
        sy,
        key_dtype,
        inv_finest,
        flat_chunk_vertices=8_000_000,
        vmask=None,
    ):
        """
        Yield chunks of touched vertex keys for one level.

        If vmask is provided, it is reused instead of rebuilt. This avoids
        repeatedly materializing the same (nx + 1, ny + 1, nz + 1) bool array.
        """
        import numpy as np

        owns_vmask = False

        if vmask is None:
            mask = np.asarray(mask, dtype=np.bool_)
            nx, ny, nz = mask.shape
            vshape = (nx + 1, ny + 1, nz + 1)
            vmask = self._make_touched_vertex_mask(mask)
            owns_vmask = True
        else:
            vmask = np.asarray(vmask, dtype=np.bool_)
            vshape = vmask.shape

        vmask_flat = vmask.ravel()
        total_vertices = int(vmask_flat.size)

        try:
            for base in range(0, total_vertices, flat_chunk_vertices):
                end = min(base + flat_chunk_vertices, total_vertices)

                local = np.flatnonzero(vmask_flat[base:end])
                if local.size == 0:
                    continue

                vflat = local.astype(np.int64, copy=False)
                vflat += np.int64(base)

                keys = self._vertex_keys_from_vflat_chunk(
                    vflat=vflat,
                    vshape=vshape,
                    meta=meta,
                    qmin_global=qmin_global,
                    sx=sx,
                    sy=sy,
                    key_dtype=key_dtype,
                    inv_finest=inv_finest,
                )

                yield keys, vflat, vshape

                del keys, vflat, local

        finally:
            del vmask_flat
            if owns_vmask:
                del vmask
    
    def _fill_connectivity_from_level_vertices(
        self,
        connectivity,
        dst0,
        mask,
        meta,
        unique_keys,
        qmin_global,
        sx,
        sy,
        key_dtype,
        inv_finest,
        flat_chunk_vertices=16_000_000,
        flat_chunk_cells=16_000_000,
        vmask=None,
    ):
        """
        Fill connectivity for one level.

        Fast path:
        - Reuse cached vmask.
        - Fill vertex_ids row-by-row using the structured key layout.
        - Avoid global np.searchsorted(unique_keys, keys) for every touched vertex.

        Fallback:
        - For non-integer voxel ratios, use the chunked key-generation path.
        """
        import time
        import numpy as np

        tic_total = time.perf_counter()

        mask = np.asarray(mask, dtype=np.bool_)
        nx, ny, nz = mask.shape

        if vmask is not None:
            vmask = np.asarray(vmask, dtype=np.bool_)
            vshape = vmask.shape
        else:
            vshape = (nx + 1, ny + 1, nz + 1)
            vmask = self._make_touched_vertex_mask(mask)

        vertex_ids = np.full(int(np.prod(vshape)), -1, dtype=np.int32)

        # ---------------------------------------------------------------------
        # Fast integer-stride path.
        # This is the important new optimization.
        # ---------------------------------------------------------------------
        tic = time.perf_counter()

        used_row_fast_path = self._fill_vertex_ids_from_rows(
            vertex_ids=vertex_ids,
            vmask=vmask,
            meta=meta,
            unique_keys=unique_keys,
            qmin_global=qmin_global,
            sx=sx,
            sy=sy,
            key_dtype=key_dtype,
            debug_validate=False,
        )

        # ---------------------------------------------------------------------
        # Fallback for non-integer voxel-size ratios.
        # ---------------------------------------------------------------------
        if not used_row_fast_path:
            for keys, vflat, _ in self._iter_touched_vertex_key_chunks(
                mask=mask,
                meta=meta,
                qmin_global=qmin_global,
                sx=sx,
                sy=sy,
                key_dtype=key_dtype,
                inv_finest=inv_finest,
                flat_chunk_vertices=flat_chunk_vertices,
                vmask=vmask,
            ):
                ids = np.searchsorted(unique_keys, keys).astype(np.int32)
                vertex_ids[vflat] = ids
                del ids

        toc = time.perf_counter()
        print(f"\t\tMapped level {meta['level']} vertex ids in {toc - tic:.2f} seconds")

        # ---------------------------------------------------------------------
        # Fill cell connectivity from the temporary vertex_ids grid.
        # ---------------------------------------------------------------------
        tic = time.perf_counter()

        mask_flat = mask.ravel()
        total_flat_cells = int(mask_flat.size)

        sy_v = nz + 1
        sx_v = (ny + 1) * (nz + 1)

        corner_offsets = np.array(
            [
                0,
                sx_v,
                sx_v + sy_v,
                sy_v,
                1,
                sx_v + 1,
                sx_v + sy_v + 1,
                sy_v + 1,
            ],
            dtype=np.int64,
        )

        written = 0

        for base_flat in range(0, total_flat_cells, flat_chunk_cells):
            end_flat = min(base_flat + flat_chunk_cells, total_flat_cells)

            local = np.flatnonzero(mask_flat[base_flat:end_flat])
            if local.size == 0:
                continue

            cf = local.astype(np.int64, copy=False)
            cf += np.int64(base_flat)

            cz = cf % nz
            tmp = cf // nz
            cy = tmp % ny
            cx = tmp // ny
            del tmp, cf

            base_vertex = cx * sx_v + cy * sy_v + cz

            out0 = dst0 + written
            out1 = out0 + int(local.size)

            for c, off in enumerate(corner_offsets):
                connectivity[out0:out1, c] = vertex_ids[base_vertex + off]

            written += int(local.size)

            del local, cx, cy, cz, base_vertex

        toc = time.perf_counter()
        print(f"\t\tFilled level {meta['level']} cell connectivity in {toc - tic:.2f} seconds")

        del vertex_ids

        toc_total = time.perf_counter()
        print(f"\t\tFinished level {meta['level']} connectivity in {toc_total - tic_total:.2f} seconds")
        
    def process_geometry(self, levels_data):
        """
        Low-peak-RAM touched-vertex geometry builder with cached per-level vmask.

        Improvements over the previous chunked version:
        - Builds each touched vertex mask once.
        - Reuses cached vmask during key generation.
        - Reuses cached vmask during connectivity construction.
        - Frees each cached vmask after that level's connectivity is built.
        """
        import time
        import numpy as np

        tic_total = time.perf_counter()

        # Tune these.
        # Larger chunks are usually faster but use more transient memory.
        flat_chunk_vertices = 8_000_000
        flat_chunk_cells = 8_000_000

        finest_voxel = min(float(voxel_size) for (_, voxel_size, _, _) in levels_data)
        inv_finest = 1.0 / finest_voxel

        num_levels = len(levels_data)

        cells_per_level = [0] * num_levels
        touched_vertices_per_level = [0] * num_levels
        level_meta = [None] * num_levels

        # Cached touched vertex masks.
        # This intentionally keeps one bool vertex mask per non-empty level so we
        # do not rebuild it in the second and third passes.
        level_vmasks = [None] * num_levels

        total_cells = 0

        qmin_global = np.full(3, np.iinfo(np.int64).max, dtype=np.int64)
        qmax_global = np.full(3, np.iinfo(np.int64).min, dtype=np.int64)

        tic = time.perf_counter()

        # -------------------------------------------------------------------------
        # First pass:
        #   - count active cells
        #   - compute active-cell bounding boxes
        #   - compute global quantized vertex bounding box
        #   - build and cache touched vertex mask once per level
        # -------------------------------------------------------------------------
        for level_idx, (data, voxel_size, origin, level) in enumerate(levels_data):
            mask = np.asarray(data, dtype=np.bool_)
            shape = mask.shape

            n_cells = int(np.count_nonzero(mask))
            cells_per_level[level_idx] = n_cells
            total_cells += n_cells

            voxel_size_f = float(voxel_size)
            origin_arr = np.asarray(origin, dtype=np.float64)
            origin_physical = origin_arr * voxel_size_f

            stride_f = voxel_size_f / finest_voxel
            stride_i = int(round(stride_f))
            integer_stride = np.isclose(stride_f, stride_i, rtol=0.0, atol=1e-9)

            origin_q = np.rint(origin_physical * inv_finest).astype(np.int64)

            meta = {
                "shape": shape,
                "voxel_size": voxel_size_f,
                "origin_physical": origin_physical,
                "origin_q": origin_q,
                "stride_i": stride_i,
                "integer_stride": bool(integer_stride),
                "level": level,
            }
            level_meta[level_idx] = meta

            if n_cells == 0:
                print(f"\tSkipping level {level} (no unique data)")
                continue

            # Bounding box via axis projections.
            xs = np.flatnonzero(mask.any(axis=(1, 2)))
            ys = np.flatnonzero(mask.any(axis=(0, 2)))
            zs = np.flatnonzero(mask.any(axis=(0, 1)))

            idx_min = np.array([xs[0], ys[0], zs[0]], dtype=np.int64)
            idx_max = np.array([xs[-1], ys[-1], zs[-1]], dtype=np.int64)

            del xs, ys, zs

            if integer_stride:
                level_qmin = origin_q + idx_min * stride_i
                level_qmax = origin_q + (idx_max + 1) * stride_i
            else:
                level_min_physical = origin_physical + idx_min.astype(np.float64) * voxel_size_f
                level_max_physical = origin_physical + (idx_max.astype(np.float64) + 1.0) * voxel_size_f

                level_qmin = np.rint(level_min_physical * inv_finest).astype(np.int64)
                level_qmax = np.rint(level_max_physical * inv_finest).astype(np.int64)

            qmin_global = np.minimum(qmin_global, level_qmin)
            qmax_global = np.maximum(qmax_global, level_qmax)

            # Build the touched vertex mask once and keep it.
            vmask = self._make_touched_vertex_mask(mask)
            n_touched_vertices = int(np.count_nonzero(vmask))

            level_vmasks[level_idx] = vmask
            touched_vertices_per_level[level_idx] = n_touched_vertices

            print(
                f"\tProcessing level {level}: "
                f"Voxel size {voxel_size}, Origin {origin_physical}, "
                f"Shape {shape}, Cells {n_cells:,}, "
                f"Touched vertices {n_touched_vertices:,}"
            )

        toc = time.perf_counter()
        print(f"\tGeometry first pass in {toc - tic:.2f} seconds")

        if total_cells == 0:
            self._cells_per_level = np.zeros(num_levels, dtype=np.int64)
            self._level_cell_offsets = np.zeros(num_levels + 1, dtype=np.int64)
            self._finest_voxel = finest_voxel

            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 8), dtype=np.int32),
                np.empty((0,), dtype=np.uint8),
                0,
            )

        cells_per_level_np = np.asarray(cells_per_level, dtype=np.int64)
        level_cell_offsets = np.empty(num_levels + 1, dtype=np.int64)
        level_cell_offsets[0] = 0
        np.cumsum(cells_per_level_np, out=level_cell_offsets[1:])

        self._cells_per_level = cells_per_level_np
        self._level_cell_offsets = level_cell_offsets
        self._finest_voxel = finest_voxel

        spans = qmax_global - qmin_global + 1

        if np.any(spans <= 0):
            raise ValueError("Invalid quantized coordinate span while building merged geometry.")

        total_span = int(spans[0]) * int(spans[1]) * int(spans[2])

        if total_span <= np.iinfo(np.uint32).max:
            key_dtype = np.uint32
        elif total_span < 2**64:
            key_dtype = np.uint64
        else:
            raise OverflowError(
                "Quantized coordinate bounding box is too large for uint64 hashing. "
                "Use a lexsort or structured-array fallback for this dataset."
            )

        sx = key_dtype(spans[0])
        sy = key_dtype(spans[1])
        plane = key_dtype(spans[0]) * key_dtype(spans[1])

        total_touched_vertices = int(np.sum(np.asarray(touched_vertices_per_level, dtype=np.int64)))

        print(
            f"\tQuantized span {tuple(int(s) for s in spans)}, "
            f"total span {total_span:,}, key dtype {np.dtype(key_dtype).name}"
        )
        print(
            f"\tSorting touched vertex keys: {total_touched_vertices:,} candidate vertices "
            f"instead of {total_cells * 8:,} raw cell corners"
        )

        # -------------------------------------------------------------------------
        # Second pass:
        #   - generate touched vertex keys only
        #   - reuse cached vmask
        # -------------------------------------------------------------------------
        tic = time.perf_counter()

        all_vertex_keys = np.empty(total_touched_vertices, dtype=key_dtype)

        write0 = 0

        for level_idx, (data, voxel_size, origin, level) in enumerate(levels_data):
            n_expected = int(touched_vertices_per_level[level_idx])
            if n_expected == 0:
                continue

            wrote_level = 0

            for keys, _, _ in self._iter_touched_vertex_key_chunks(
                mask=data,
                meta=level_meta[level_idx],
                qmin_global=qmin_global,
                sx=sx,
                sy=sy,
                key_dtype=key_dtype,
                inv_finest=inv_finest,
                flat_chunk_vertices=flat_chunk_vertices,
                vmask=level_vmasks[level_idx],
            ):
                n = int(keys.size)
                all_vertex_keys[write0:write0 + n] = keys
                write0 += n
                wrote_level += n

            assert wrote_level == n_expected, (
                f"Touched vertex count mismatch at level {level}: "
                f"expected {n_expected}, wrote {wrote_level}"
            )

        assert write0 == total_touched_vertices

        toc = time.perf_counter()
        print(f"\tGenerated touched vertex keys in {toc - tic:.2f} seconds")

        # -------------------------------------------------------------------------
        # Sort and deduplicate global vertex keys.
        # -------------------------------------------------------------------------
        tic = time.perf_counter()

        all_vertex_keys.sort(kind="quicksort")

        unique_mask = np.empty(total_touched_vertices, dtype=np.bool_)
        unique_mask[0] = True
        np.not_equal(all_vertex_keys[1:], all_vertex_keys[:-1], out=unique_mask[1:])

        # Boolean indexing already returns a copy.
        unique_keys = all_vertex_keys[unique_mask]

        del all_vertex_keys, unique_mask

        num_unique = int(unique_keys.size)

        if num_unique > np.iinfo(np.int32).max:
            raise OverflowError("Too many unique points for int32 connectivity.")

        toc = time.perf_counter()
        print(f"\tSorted and deduplicated to {num_unique:,} unique keys in {toc - tic:.2f} seconds")

        # -------------------------------------------------------------------------
        # Third pass:
        #   - build final connectivity from per-level temporary vertex-id grids
        #   - reuse cached vmask
        #   - free each cached vmask once its level is done
        # -------------------------------------------------------------------------
        tic = time.perf_counter()

        connectivity = np.empty((total_cells, 8), dtype=np.int32)
        level_id_field = np.empty(total_cells, dtype=np.uint8)

        for level_idx, (data, voxel_size, origin, level) in enumerate(levels_data):
            n_cells = int(cells_per_level[level_idx])
            if n_cells == 0:
                continue

            cell_start = int(level_cell_offsets[level_idx])
            cell_end = int(level_cell_offsets[level_idx + 1])

            level_id_field[cell_start:cell_end] = np.uint8(level)

            self._fill_connectivity_from_level_vertices(
                connectivity=connectivity,
                dst0=cell_start,
                mask=data,
                meta=level_meta[level_idx],
                unique_keys=unique_keys,
                qmin_global=qmin_global,
                sx=sx,
                sy=sy,
                key_dtype=key_dtype,
                inv_finest=inv_finest,
                flat_chunk_vertices=flat_chunk_vertices,
                flat_chunk_cells=flat_chunk_cells,
                vmask=level_vmasks[level_idx],
            )

            # Free this level's cached vertex mask as soon as it is no longer needed.
            level_vmasks[level_idx] = None

        toc = time.perf_counter()
        print(f"\tBuilt connectivity in {toc - tic:.2f} seconds")

        # Optional debug check. Keep disabled for timing.
        debug_validate_connectivity = False
        if debug_validate_connectivity:
            if np.any(connectivity < 0):
                raise RuntimeError("Connectivity contains negative vertex ids.")
            if int(connectivity.max()) >= num_unique:
                raise RuntimeError("Connectivity contains vertex ids outside unique point range.")

        # -------------------------------------------------------------------------
        # Reconstruct unique coordinates from sorted unique integer lattice keys.
        # -------------------------------------------------------------------------
        tic = time.perf_counter()

        z = unique_keys // plane
        rem = unique_keys - z * plane
        y = rem // sx
        x = rem - y * sx

        del unique_keys, rem

        coordinates = np.empty((num_unique, 3), dtype=np.float32)

        coordinates[:, 0] = x.astype(np.float32, copy=False)
        coordinates[:, 0] += np.float32(qmin_global[0])
        coordinates[:, 0] *= np.float32(finest_voxel)
        del x

        coordinates[:, 1] = y.astype(np.float32, copy=False)
        coordinates[:, 1] += np.float32(qmin_global[1])
        coordinates[:, 1] *= np.float32(finest_voxel)
        del y

        coordinates[:, 2] = z.astype(np.float32, copy=False)
        coordinates[:, 2] += np.float32(qmin_global[2])
        coordinates[:, 2] *= np.float32(finest_voxel)
        del z

        toc = time.perf_counter()
        print(f"\tReconstructed coordinates in {toc - tic:.2f} seconds")

        toc_total = time.perf_counter()

        raw_points = total_cells * 8
        reduction = 100.0 * (1.0 - float(num_unique) / float(raw_points))

        print(
            f"\tBuilt merged geometry: {total_cells:,} cells, "
            f"{num_unique:,} unique points "
            f"({reduction:.1f}% point reduction) in {toc_total - tic_total:.2f} seconds"
        )

        return coordinates, connectivity, level_id_field, total_cells
    
    def _transform_coordinates(self, coordinates, offset):
        """
        Transform coordinates to physical units and apply offset.

        Offset is applied in-place when possible to avoid an extra large
        coordinates-sized allocation.
        """
        import numpy as np

        if self.unit_convertor is not None:
            coordinates = self.unit_convertor.length_to_physical(coordinates)

        if offset is not None:
            offset_arr = np.asarray(offset, dtype=coordinates.dtype)
            if np.any(offset_arr):
                coordinates += offset_arr

        return coordinates
    
    def _process_level(self, data, voxel_size, origin, point_id_offset):
        """
        Given a voxel grid, returns all corners and connectivity in NumPy for this resolution level.
        """
        true_indices = np.argwhere(data)
        if true_indices.size == 0:
            return [], []

        max_voxels_per_chunk = 268_435_450
        chunks = np.array_split(true_indices, max(1, (len(true_indices) + max_voxels_per_chunk - 1) // max_voxels_per_chunk))

        all_corners = []
        all_connectivity = []
        pid_offset = point_id_offset

        for chunk in chunks:
            if chunk.size == 0:
                continue
            corners, connectivity = self._process_voxel_chunk(chunk, np.asarray(origin, dtype=np.float32), voxel_size, pid_offset)
            all_corners.append(corners)
            all_connectivity.append(connectivity)
            pid_offset += len(chunk) * 8

        return all_corners, all_connectivity

    def _process_voxel_chunk(self, true_indices, origin, voxel_size, point_id_offset):
        """
        Given a set of voxel indices, returns 8 corners and connectivity for each cube using NumPy.
        """
        true_indices = np.asarray(true_indices, dtype=np.float32)
        mins = origin + true_indices * voxel_size
        offsets = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ],
            dtype=np.float32,
        )

        corners = (mins[:, None, :] + offsets[None, :, :] * voxel_size).reshape(-1, 3).astype(np.float32)
        base_ids = point_id_offset + np.arange(len(true_indices), dtype=np.int32) * 8
        connectivity = (base_ids[:, None] + np.arange(8, dtype=np.int32)).astype(np.int32)

        return corners, connectivity

    def save_xdmf(self, h5_filename, xmf_filename, total_cells, num_points, fields={}):
        # Generate an XDMF file to accompany the HDF5 file
        print(f"\tGenerating XDMF file: {xmf_filename}")
        hdf5_rel_path = h5_filename.split("/")[-1]
        with open(xmf_filename, "w") as xmf:
            xmf.write(f'''<?xml version="1.0" ?>
    <!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
    <Xdmf Version="3.0">
        <Domain>
            <Grid Name="VoxelMesh" GridType="Uniform">
                <Topology TopologyType="Hexahedron" NumberOfElements="{total_cells}">
                    <DataItem Dimensions="{total_cells} 8" NumberType="Int" Format="HDF">
                        {hdf5_rel_path}:/Mesh/Connectivity
                    </DataItem>
                </Topology>
                <Geometry GeometryType="XYZ">
                    <DataItem Dimensions="{num_points} 3" NumberType="Float" Precision="4" Format="HDF">
                        {hdf5_rel_path}:/Mesh/Points
                    </DataItem>
                </Geometry>
                <Attribute Name="Level" AttributeType="Scalar" Center="Cell">
                    <DataItem Dimensions="{total_cells}" NumberType="UInt8" Format="HDF">
                        {hdf5_rel_path}:/Mesh/Level
                    </DataItem>
                </Attribute>
        ''')
            for field_name in fields.keys():
                xmf.write(f'''
            <Attribute Name="{field_name}" AttributeType="Scalar" Center="Cell">
                <DataItem Dimensions="{total_cells}" NumberType="Float" Precision="4" Format="HDF">
                {hdf5_rel_path}:/Fields/{field_name}
                </DataItem>
            </Attribute>
            ''')
            xmf.write("""
                </Grid>
            </Domain>
        </Xdmf>
        """)
        print("\tXDMF file written successfully")
        return

    def save_hdf5_file(self, filename, coordinates, connectivity, level_id_field, fields_data, compression="gzip", compression_opts=0):
        """Write the processed mesh data to an HDF5 file.
        Parameters
        ----------
        filename : str
            The name of the output HDF5 file.
        coordinates : numpy.ndarray
            An array of all coordinates.
        connectivity : numpy.ndarray
            An array of all connectivity data.
        level_id_field : numpy.ndarray
            An array of all level data.
        fields_data : dict
            A dictionary of all field data.
        compression : str, optional
            The compression method to use for the HDF5 file.
        compression_opts : int, optional
            The compression options to use for the HDF5 file.
        """
        import h5py

        with h5py.File(filename + ".h5", "w") as f:
            f.create_dataset("/Mesh/Points", data=coordinates, compression=compression, compression_opts=compression_opts, chunks=True)
            f.create_dataset(
                "/Mesh/Connectivity",
                data=connectivity,
                compression=compression,
                compression_opts=compression_opts,
                chunks=True,
            )
            f.create_dataset("/Mesh/Level", data=level_id_field, compression=compression, compression_opts=compression_opts)
            fg = f.create_group("/Fields")
            for fname, fdata in fields_data.items():
                fg.create_dataset(fname, data=fdata.astype(np.float32), compression=compression, compression_opts=compression_opts, chunks=True)

    def _merge_duplicates(self, coordinates, connectivity, levels_data):
        """
        Merge duplicate points using parallel sorting.
        ~3-4x faster than np.unique for large arrays.
        """
        import time
        tic = time.perf_counter()
        
        num_points = coordinates.shape[0]
        finest_voxel = min(voxel_size for (_, voxel_size, _, _) in levels_data)
        print(f"\tMerging duplicates from {num_points:,} points (finest voxel={finest_voxel}m)...")
        
        # Quantize coordinates to finest voxel grid
        inv = 1.0 / finest_voxel
        qi = np.rint(coordinates * inv).astype(np.int64)  # Removed unnecessary float64 cast
        
        # Compute tight bounding box
        mins = qi.min(axis=0)
        spans = qi.max(axis=0) - mins + 1
        shifts = qi - mins
        
        # Create hash keys
        hash_keys = shifts[:, 0] + spans[0] * (shifts[:, 1] + spans[1] * shifts[:, 2])        
        
        # Use stable sort (preserves first occurrence ordering)
        sort_idx = np.argsort(hash_keys, kind='stable')
        sorted_keys = hash_keys[sort_idx]
        
        # Find unique boundaries (vectorized)
        unique_mask = np.concatenate([[True], sorted_keys[1:] != sorted_keys[:-1]])
        unique_indices = sort_idx[unique_mask]
        
        # Build inverse mapping efficiently
        unique_ranks = np.cumsum(unique_mask) - 1
        inverse_indices = np.empty(num_points, dtype=np.int32)
        inverse_indices[sort_idx] = unique_ranks
        
        # Preserve original coordinates
        unique_coordinates = coordinates[unique_indices]
        new_connectivity = inverse_indices[connectivity]
        
        toc = time.perf_counter()
        reduction = 100 * (1 - len(unique_coordinates) / num_points)
        print(f"\tMerged to {len(unique_coordinates):,} unique points ({reduction:.1f}% reduction) in {toc - tic:.2f} seconds")
        
        return unique_coordinates, new_connectivity

    def _prepare_container_inputs(self):
        # load necessary modules
        from xlb.compute_backend import ComputeBackend
        from xlb.grid import grid_factory

        # Get the number of levels from the levels_data
        num_levels = len(self.levels_data)

        # Prepare lists to hold warp fields and origins allocated for each level
        field_warp_dict = {}
        origin_list = []
        for field_name, cardinality in self.field_name_cardinality_dict.items():
            field_warp_dict[field_name] = []
            for level in range(num_levels):
                # get the shape of the grid at this level
                box_shape = self.levels_data[level][0].shape

                # Use the warp backend to create dense fields to be written in multi-res NEON fields
                grid_dense = grid_factory(box_shape, compute_backend=ComputeBackend.WARP)
                field_warp_dict[field_name].append(grid_dense.create_field(cardinality=cardinality, dtype=self.store_precision))
                origin_list.append(wp.vec3i(*([int(x) for x in self.levels_data[level][2]])))

        return field_warp_dict, origin_list

    def _construct_neon_container(self):
        """
        Constructs a NEON container for exporting multi-resolution data to HDF5.
        This container will be used to transfer multi-resolution NEON fields into stacked warp fields.
        """

        @neon.Container.factory(name="HDF5MultiresExporter")
        def container(
            field_neon: Any,
            field_warp: Any,
            origin: Any,
            level: Any,
        ):
            def launcher(loader: neon.Loader):
                loader.set_mres_grid(field_neon.get_grid(), level)
                field_neon_hdl = loader.get_mres_read_handle(field_neon)
                refinement = 2**level

                @wp.func
                def kernel(index: Any):
                    cIdx = wp.neon_global_idx(field_neon_hdl, index)
                    # Get local indices by dividing the global indices (associated with the finest level) by 2^level
                    # Subtract the origin to get the local indices in the warp field
                    lx = wp.neon_get_x(cIdx) // refinement - origin[0]
                    ly = wp.neon_get_y(cIdx) // refinement - origin[1]
                    lz = wp.neon_get_z(cIdx) // refinement - origin[2]

                    # write the values to the warp field
                    cardinality = field_warp.shape[0]
                    for card in range(cardinality):
                        field_warp[card, lx, ly, lz] = self.store_dtype(wp.neon_read(field_neon_hdl, index, card))

                loader.declare_kernel(kernel)

            return launcher

        return container

    def get_fields_data(self, field_neon_dict):
        """
        Extracts and prepares fields data from NEON fields for export.

        Performance changes:
        - WARP fields are allocated lazily only for requested fields.
        - output arrays are preallocated to self.total_cells.
        - avoids per-field list accumulation and final np.concatenate().
        """
        import numpy as np

        try:
            wp_mod = wp
        except NameError:
            import warp as wp_mod

        if not field_neon_dict:
            return {}

        # Ensure that this operator is called on multires grids.
        grid_mres = next(iter(field_neon_dict.values())).get_grid()
        assert grid_mres.name == "mGrid", (
            f"Operation {self.__class__.__name__} is only applicable to multi-resolution cases!"
        )

        for field_name in field_neon_dict.keys():
            assert field_name in self.field_name_cardinality_dict.keys(), (
                f"Field {field_name} is not provided in the instantiation of the MultiresIO class!"
            )

        num_levels = grid_mres.num_levels
        assert num_levels == len(self.levels_data), "Error: Inconsistent number of levels!"

        needed_field_names = [
            field_name
            for field_name in field_neon_dict.keys()
            if field_name in self.field_name_cardinality_dict
        ]

        self._ensure_container_inputs(needed_field_names)

        fields_data = {}

        for field_name, cardinality in self.field_name_cardinality_dict.items():
            if field_name not in field_neon_dict:
                continue

            allocated = False

            for level in range(num_levels):
                dst0 = int(self._level_cell_offsets[level])
                dst1 = int(self._level_cell_offsets[level + 1])

                if dst1 == dst0:
                    continue

                c = self.container(
                    field_neon_dict[field_name],
                    self.field_warp_dict[field_name][level],
                    self.origin_list[level],
                    level,
                )
                c.run(0, container_runtime=neon.Container.ContainerRuntime.neon)

                wp_mod.synchronize()

                mask = self.levels_data[level][0]
                field_np = self.field_warp_dict[field_name][level].numpy()

                if not allocated:
                    for card in range(cardinality):
                        fields_data[f"{field_name}_{card}"] = np.empty(
                            self.total_cells,
                            dtype=field_np.dtype,
                        )
                    allocated = True

                for card in range(cardinality):
                    key = f"{field_name}_{card}"
                    fields_data[key][dst0:dst1] = field_np[card][mask]

            for card in range(cardinality):
                key = f"{field_name}_{card}"
                if key in fields_data:
                    assert fields_data[key].size == self.total_cells, (
                        f"Error: Field {key} size mismatch!"
                    )

        # Unit conversion if applicable.
        if self.unit_convertor is not None:
            for field_name in list(fields_data.keys()):
                lower = field_name.lower()

                if "velocity" in lower:
                    fields_data[field_name] = self.unit_convertor.velocity_to_physical(fields_data[field_name])
                elif "density" in lower:
                    fields_data[field_name] = self.unit_convertor.density_to_physical(fields_data[field_name])
                elif "pressure" in lower:
                    fields_data[field_name] = self.unit_convertor.pressure_to_physical(fields_data[field_name])

        return fields_data
    
    def to_hdf5(self, output_filename, field_neon_dict, compression="gzip", compression_opts=0):
        """
        Export the multi-resolution mesh data to an HDF5 file.
        Parameters
        ----------
        output_filename : str
            The name of the output HDF5 file (without extension).
        field_neon_dict : a dictionary of neon mGrid Fields
            Eg. The NEON fields containing velocity and density data as { "velocity": velocity_neon, "density": density_neon}
        compression : str, optional
            The compression method to use for the HDF5 file.
        compression_opts : int, optional
            The compression options to use for the HDF5 file.
        """
        import time

        # Get the fields data from the NEON fields
        fields_data = self.get_fields_data(field_neon_dict)

        # Save XDMF file
        self.save_xdmf(output_filename + ".h5", output_filename + ".xmf", self.total_cells, len(self.coordinates), fields_data)

        # Writing HDF5 file
        print("\tWriting HDF5 file")
        tic_write = time.perf_counter()
        self.save_hdf5_file(output_filename, self.coordinates, self.connectivity, self.level_id_field, fields_data, compression, compression_opts)
        toc_write = time.perf_counter()
        print(f"\tHDF5 file written in {toc_write - tic_write:0.1f} seconds")

    def to_slice_image(
        self,
        output_filename,
        field_neon_dict,
        plane_point,
        plane_normal,
        slice_thickness=1.0,
        bounds=[0, 1, 0, 1],
        grid_res=512,
        cmap=None,
        component=None,
        show_axes=False,
        show_colorbar=False,        
        normalize=1.0,
        output=None,
        width=None,
        height=None,
        width_vec=None,
        height_vec=None,
        **kwargs,
    ):
        """
        Export an arbitrary-plane slice from unstructured point data to PNG.

        Parameters
        ----------
        output_filename : str
            Output PNG filename (without extension).
        field_neon_dict : dict
            A dictionary of NEON fields containing the data to be plotted.
            Example: {"velocity": velocity_neon, "density": density_neon}
        plane_point : array_like
            A point [x, y, z] on the plane.
        plane_normal : array_like
            Plane normal vector [nx, ny, nz].
        slice_thickness : float
            How thick (in units of the coordinate system) the slice should be.
        grid_resolution : tuple
            Resolution of output image (pixels in plane u, v directions).
        grid_size : tuple
            Physical size of slice grid (width, height).
        cmap : str
            Matplotlib colormap.
        normalize : float
            Factor to scale and normalize data to ensure consistent images
        """
        # Get the fields data from the NEON fields
        assert len(field_neon_dict.keys()) == 1, "Error: This function is designed to plot a single field at a time."
        fields_data = self.get_fields_data(field_neon_dict)

        # Check if the component is within the valid range
        if component is None:
            print("\tCreating slice image of the field magnitude!")
            cell_data = list(fields_data.values())
            squared = [comp**2 for comp in cell_data]
            cell_data = np.sqrt(sum(squared))
            field_name = list(fields_data.keys())[0].split("_")[0] + "_magnitude"
        else:
            assert component < max(self.field_name_cardinality_dict.values()), (
                f"Error: Component {component} is out of range for the provided fields."
            )
            print(f"\tCreating slice image for component {component} of the input field!")
            field_name = list(fields_data.keys())[component]
            cell_data = fields_data[field_name]

        if normalize != 1.0:  
            cell_data = np.clip((cell_data / normalize),0,1)
        else:   
            cell_data = cell_data      

        # Plot each field in the dictionary
        self._to_slice_image_single_field(
            f"{output_filename}_{field_name}",
            cell_data,
            plane_point,
            plane_normal,
            slice_thickness=slice_thickness,
            bounds=bounds,
            grid_res=grid_res,
            cmap=cmap,
            show_axes=show_axes,
            show_colorbar=show_colorbar,
            normalize=normalize,
            width=width,
            height=height,
            width_vec=width_vec,
            height_vec=height_vec,
            **kwargs,
        )
        print(f"\tSlice image for field {field_name} saved as {output_filename}.png")

    def _to_slice_image_single_field(
        self,
        output_filename,
        field_data,
        plane_point,
        plane_normal,
        slice_thickness,
        bounds,
        grid_res,
        cmap,
        show_axes,
        show_colorbar,
        normalize,
        width,
        height,
        width_vec,
        height_vec,
        **kwargs,
    ):
        """
        Helper function to create a slice image for a single field.

        If width, height, width_vec, and height_vec are all provided, the image
        extent is pinned to the rectangle [plane_point, plane_point + width*width_vec + height*height_vec]
        so consecutive slices in a sweep share an identical pixel grid (no shift).
        """
        from matplotlib import cm
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.spatial import cKDTree

        # field data are associated with the cells centers
        cell_values = field_data

        fixed_extent = (
            width is not None
            and height is not None
            and width_vec is not None
            and height_vec is not None
        )

        # get the normalized plane normal (preserve sign)
        plane_normal = np.asarray(plane_normal, dtype=np.float64)
        n = plane_normal / np.linalg.norm(plane_normal)

        # Compute signed distances of each cell center to the plane.
        # Copy plane_point so concurrent slice renders cannot corrupt a shared array.
        plane_point = np.asarray(plane_point, dtype=np.float64).copy()
        sdf = np.dot(self.centroids - plane_point, n)

        # Filter: cells with centroid near plane.
        # On multires grids, scale the slab thickness per cell by its refinement level
        # so each level contributes ~one intersecting layer instead of many fine layers.
        if getattr(self, "level_id_field", None) is not None:
            local_thickness = float(slice_thickness) * (
                2.0 ** self.level_id_field.astype(np.float32)
            )
            mask = np.abs(sdf) <= 0.55 * local_thickness
        else:
            mask = np.abs(sdf) <= slice_thickness / 2
        if not np.any(mask):
            raise ValueError("No cells intersect the plane within thickness.")

        # Project centroids to plane
        centroids_slice = self.centroids[mask]
        sdf_slice = sdf[mask]
        proj = centroids_slice - np.outer(sdf_slice, n)

        values = cell_values[mask]

        # Build in-plane basis
        if fixed_extent:
            u1 = np.asarray(width_vec, dtype=np.float64)
            u1 = u1 / np.linalg.norm(u1)
            u2 = np.asarray(height_vec, dtype=np.float64)
            u2 = u2 / np.linalg.norm(u2)
        else:
            if np.allclose(n, [1, 0, 0]):
                u1 = np.array([0, 1, 0])
            else:
                u1 = np.array([1, 0, 0])
            u2 = np.abs(np.cross(n, u1))

        local_x = np.dot(proj - plane_point, u1)
        local_y = np.dot(proj - plane_point, u2)

        if cmap is None:
            cmap = cm.nipy_spectral

        if fixed_extent:
            # Pin grid to the published slice rectangle so every slice in a
            # sweep uses an identical pixel grid (no inter-slice shift).
            bounded_x_min = 0.0
            bounded_x_max = float(width)
            bounded_y_min = 0.0
            bounded_y_max = float(height)
            mask_bounds = (
                (bounded_x_min <= local_x) & (local_x <= bounded_x_max)
                & (bounded_y_min <= local_y) & (local_y <= bounded_y_max)
            )
            aspect_ratio = (bounded_y_max - bounded_y_min) / max(bounded_x_max - bounded_x_min, 1e-20)
            grid_resY = max(1, int(np.round(grid_res * aspect_ratio)))
        else:
            # Legacy: derive the extent from whichever cells fell into the slab.
            xmin, xmax, ymin, ymax = local_x.min(), local_x.max(), local_y.min(), local_y.max()
            Lx = xmax - xmin
            Ly = ymax - ymin
            extent = np.array([xmin + bounds[0] * Lx, xmin + bounds[1] * Lx, ymin + bounds[2] * Ly, ymin + bounds[3] * Ly])
            mask_bounds = (extent[0] <= local_x) & (local_x <= extent[1]) & (extent[2] <= local_y) & (local_y <= extent[3])

            bounded_x_min = local_x[mask_bounds].min()
            bounded_x_max = local_x[mask_bounds].max()
            bounded_y_min = local_y[mask_bounds].min()
            bounded_y_max = local_y[mask_bounds].max()
            aspect_ratio = (bounded_y_max - bounded_y_min) / max(bounded_x_max - bounded_x_min, 1e-20)
            grid_resY = max(1, int(np.round(grid_res * aspect_ratio)))

        # Create grid
        grid_x = np.linspace(bounded_x_min, bounded_x_max, grid_res)
        grid_y = np.linspace(bounded_y_min, bounded_y_max, grid_resY)
        xv, yv = np.meshgrid(grid_x, grid_y, indexing="xy")

        # Fast KDTree-based interpolation
        points = np.column_stack((local_x[mask_bounds], local_y[mask_bounds]))
        tree = cKDTree(points)

        # Query points
        query_points = np.column_stack((xv.ravel(), yv.ravel()))

        # Find k nearest neighbors for smoother interpolation
        k = min(4, len(points))
        distances, indices = tree.query(query_points, k=k, workers=-1)

        if k == 1:
            distances = distances[:, None]
            indices = indices[:, None]

        # Gaussian-weighted interpolation with an adaptive per-query bandwidth.
        # h scales with the actual neighbor spacing at each pixel, so coarse-grid
        # regions widen the kernel automatically and weights don't underflow to
        # zero between cell centers (which paints a dot grid).
        pixel_dx = (bounded_x_max - bounded_x_min) / max(grid_res - 1, 1)
        pixel_dy = (bounded_y_max - bounded_y_min) / max(grid_resY - 1, 1)
        floor_h = max(2.0 * float(slice_thickness), 2.0 * pixel_dx, 2.0 * pixel_dy, 1e-12)

        # Use the farthest of the k neighbors as the local scale: the closest
        # neighbor lands near exp(-0.5) instead of underflowing in coarse zones.
        if k > 1:
            local_h = distances[:, -1:]
        else:
            local_h = np.full((distances.shape[0], 1), floor_h)
        h = np.maximum(local_h, floor_h)

        weights = np.exp(-0.5 * (distances / h) ** 2)
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-20)

        # Interpolate values
        neighbor_values = values[mask_bounds][indices]
        grid_field = (neighbor_values * weights).sum(axis=1).reshape(grid_resY, grid_res)

        # Plot
        if show_colorbar or show_axes:
            dpi = 300
            plt.imshow(
                grid_field,
                extent=[bounded_x_min, bounded_x_max, bounded_y_min, bounded_y_max],
                cmap=cmap,
                origin="lower",
                aspect="equal",
                **kwargs,
            )
            if show_colorbar:
                plt.colorbar()
            if not show_axes:
                plt.axis("off")
            plt.savefig(output_filename + ".png", dpi=dpi, bbox_inches="tight", pad_inches=0)
            plt.close()
        else:
            if normalize != 1.0:
                plt.imsave(output_filename + ".png", grid_field, cmap=cmap, origin="lower", vmin=0, vmax=1)
            else:
                plt.imsave(output_filename + ".png", grid_field, cmap=cmap, origin="lower")

    def to_line(
        self,
        output_filename,
        field_neon_dict,
        start_point,
        end_point,
        resolution,
        component=None,
        radius=1.0,
        **kwargs,
    ):
        """
        Extract field data along a line between start_point and end_point and save to a CSV file.

        This function performs two main steps:
        1. Extracts field data from field_neon_dict, handling components or computing magnitude.
        2. Interpolates the field values along a line defined by start_point and end_point,
        then saves the results (coordinates and field values) to a CSV file.

        Parameters
        ----------
        output_filename : str
            The name of the output CSV file (without extension). Example: "velocity_profile".
        field_neon_dict : dict
            A dictionary containing the field data to extract, with a single key-value pair.
            The key is the field name (e.g., "velocity"), and the value is the NEON data object
            containing the field values. Example: {"velocity": velocity_neon}.
        start_point : array_like
            The starting point of the line in 3D space (e.g., [x0, y0, z0]).
            Units must match the coordinate system used in the class (voxel units if untransformed,
            or model units if scale/offset are applied).
        end_point : array_like
            The ending point of the line in 3D space (e.g., [x1, y1, z1]).
            Units must match the coordinate system used in the class.
        resolution : int
            The number of points along the line where the field will be interpolated.
            Example: 100 for 100 evenly spaced points.
        component : int, optional
            The specific component of the field to extract (e.g., 0 for x-component, 1 for y-component).
            If None, the magnitude of the field is computed. Default is None.
        radius : int
            The specified distance (in units of the coordinate system) to prefilter and query for line plot

        Returns
        -------
        None
            The function writes the output to a CSV file and prints a confirmation message.

        Notes
        -----
        - The output CSV file will contain columns: 'x', 'y', 'z', and the value of the field name (e.g., 'velocity_x' or 'velocity_magnitude').
        """

        # Get the fields data from the NEON fields
        assert len(field_neon_dict.keys()) == 1, "Error: This function is designed to plot a single field at a time."
        fields_data = self.get_fields_data(field_neon_dict)

        # Check if the component is within the valid range
        if component is None:
            print("\tCreating csv plot of the field magnitude!")
            cell_data = list(fields_data.values())
            squared = [comp**2 for comp in cell_data]
            cell_data = np.sqrt(sum(squared))
            field_name = list(fields_data.keys())[0].split("_")[0] + "_magnitude"

        else:
            assert component < max(self.field_name_cardinality_dict.values()), (
                f"Error: Component {component} is out of range for the provided fields."
            )
            print(f"\tCreating csv plot for component {component} of the input field!")
            field_name = list(fields_data.keys())[component]
            cell_data = fields_data[field_name]

        # Plot each field in the dictionary
        self._to_line_field(
            f"{output_filename}_{field_name}",
            cell_data,
            start_point,
            end_point,
            resolution,
            radius=radius,
            **kwargs,
        )
        print(f"\tLine Plot for field {field_name} saved as {output_filename}.csv")

    def _to_line_field(
        self,
        output_filename,
        cell_data,
        start_point,
        end_point,
        resolution,
        radius,
        **kwargs,
    ):
        """
        Helper function to create a line plot for a single field.
        """
        import numpy as np

        # cell_points = self.coordinates[self.connectivity]  # Shape: (M, K, 3), where M is num cells, K is nodes per cell
        # centroids = np.mean(cell_points, axis=1)  # Shape: (M, 3)
        centroids = self.centroids
        p0 = np.array(start_point, dtype=np.float32)
        p1 = np.array(end_point, dtype=np.float32)

        # direction and parameter t for each centroid
        d = p1 - p0
        L = np.linalg.norm(d)
        d_unit = d / L
        v = centroids - p0
        t = v.dot(d_unit)
        closest = p0 + np.outer(t, d_unit)
        perp_dist = np.linalg.norm(centroids - closest, axis=1)

        # optionally mask to [0,L] or a small perp-radius
        mask = (t >= 0) & (t <= L) & (perp_dist <= radius)
        t, data = t[mask], cell_data[mask]

        # sort by t
        idx = np.argsort(t)
        t_sorted = t[idx]
        data_sorted = data[idx]

        # target samples
        t_line = np.linspace(0, L, resolution)

        # 1D linear interpolation
        vals_line = np.interp(t_line, t_sorted, data_sorted, left=np.nan, right=np.nan)

        # reconstruct (x,y,z)
        line_xyz = p0[None, :] + t_line[:, None] * d_unit[None, :]

        # vectorized CSV dump
        out = np.hstack([line_xyz, vals_line[:, None]])
        np.savetxt(output_filename + ".csv", out, delimiter=",", header="x,y,z,value", comments="")
    
    def start_time_average(self):
        """
        Reset time-averaging accumulators.
        Call this once before your time loop.
        """
        self._avg_sum = {}
        self._avg_weight = 0.0
        self._avg_active = True
        self._avg_final_cache = None
        self._avg_cache_weight = 0.0
        self._avg_derived_cache = {}

    def accumulate_time_average(self, field_neon_dict: Dict[str, Any], weight: float = 1.0):
        """
        Accumulate a timestep into the running time-average using extracted cell-centered data.

        Parameters
        ----------
        field_neon_dict : dict
            Same as to_hdf5 / to_slice_image input (e.g. {"velocity": sim.u}).
            You can call this multiple times with different field keys over time;
            the accumulator stores by extracted component keys (e.g. "velocity_0").
        weight : float
            Typically dt for dt-weighted averaging, or 1.0 for simple arithmetic mean.
        """
        start_time = time.time()
        assert self._avg_active, "Call start_time_average() before accumulate_time_average()."
        fields_data = self.get_fields_data(field_neon_dict)
        self.accumulate_time_average_from_fields_data(fields_data, weight=weight)  
        print(f"Time Avg Accumulation in {time.time()-start_time}sec ")

    def accumulate_time_average_from_fields_data(self, fields_data: Dict[str, np.ndarray], weight: float = 1.0):
        """
        Same as accumulate_time_average(), but lets you pass already-extracted numpy arrays.
        Useful if you want to do custom pre-processing before accumulation.
        """
        assert self._avg_active, "Call start_time_average() before accumulating."
        w = float(weight)
        for k, v in fields_data.items():
            v64 = np.asarray(v, dtype=np.float64)
            if k not in self._avg_sum:
                self._avg_sum[k] = np.zeros_like(v64, dtype=np.float64)
            self._avg_sum[k] += w * v64
        self._avg_weight += w

    def finalize_time_average(self, keep_state: bool = True) -> Dict[str, np.ndarray]:
        assert self._avg_weight > 0.0, "No samples accumulated. Call accumulate_time_average() first."
        start_time = time.time()
        # Cache hit: only valid if no new accumulation since caching
        if self._avg_final_cache is None or self._avg_cache_weight != self._avg_weight:
            self._avg_final_cache = {
                k: (s / self._avg_weight).astype(np.float32)
                for k, s in self._avg_sum.items()
            }
            self._avg_cache_weight = self._avg_weight

        avg_fields = self._avg_final_cache

        if not keep_state:
            self._avg_active = False
            self._avg_sum = {}
            self._avg_weight = 0.0
            # Clear cache too (important)
            self._avg_final_cache = None
            self._avg_cache_weight = 0.0
        print(f"Finalize Time Avg in {time.time()-start_time}sec ")

        return avg_fields
    
    def to_hdf5_time_average(
        self,
        output_filename: str,
        compression: str = "gzip",
        compression_opts: int = 0,
        keep_state: bool = True,
    ):
        """
        Write an averaged HDF5/XDMF from the current accumulated average.
        """
        avg_fields = self.finalize_time_average(keep_state=keep_state)

        self.save_xdmf(
            output_filename + ".h5",
            output_filename + ".xmf",
            self.total_cells,
            len(self.coordinates),
            avg_fields,
        )

        print("\tWriting time-averaged HDF5 file")
        tic_write = time.perf_counter()
        self.save_hdf5_file(
            output_filename,
            self.coordinates,
            self.connectivity,
            self.level_id_field,
            avg_fields,
            compression,
            compression_opts,
        )
        toc_write = time.perf_counter()
        print(f"\tTime-averaged HDF5 file written in {toc_write - tic_write:0.1f} seconds")
        
    def to_slice_image_time_average(
        self,
        output_filename: str,
        field_base_name: str,
        plane_point,
        plane_normal,
        slice_thickness=1.0,
        bounds=[0, 1, 0, 1],
        grid_res=512,
        cmap=None,
        component=None,
        show_axes=False,
        show_colorbar=False,
        normalize=1.0,
        keep_state: bool = True,       
        width=None,
        height=None,
        width_vec=None,
        height_vec=None,
        **kwargs,
    ):
        """
        Export a slice image from the accumulated time-averaged data.

        Parameters match to_slice_image(), except:
        - field_base_name: e.g. "velocity" (used to find keys "velocity_0", "velocity_1", ...)
        """
        avg_fields = self.finalize_time_average(keep_state=keep_state)

        # Find all components for this base field
        comp_keys = [k for k in avg_fields.keys() if k.startswith(field_base_name + "_")]
        assert len(comp_keys) > 0, f"No averaged components found for base field '{field_base_name}'."

        # Sort by component index (expects suffix _0, _1, _2, ...)
        def _comp_index(k: str) -> int:
            try:
                return int(k.split("_")[-1])
            except Exception:
                return 0

        comp_keys = sorted(comp_keys, key=_comp_index)

        if component is None and len(comp_keys) > 1:
            print("\tCreating time-averaged slice image of the field magnitude!")
            comps = [avg_fields[k].astype(np.float64) for k in comp_keys]
            cell_data = np.sqrt(np.sum([c**2 for c in comps], axis=0)).astype(np.float32)
            field_name = field_base_name + "_magnitude"
        elif component is None and len(comp_keys) == 1:
            print("\tCreating time-averaged slice image of the scalar field!")
            field_name = comp_keys[0]
            cell_data = avg_fields[field_name]
        else:
            assert 0 <= int(component) < len(comp_keys), "Requested component out of range for averaged field."
            field_name = comp_keys[int(component)]
            print(f"\tCreating time-averaged slice image for component {component} of the field!")
            cell_data = avg_fields[field_name]

        if normalize != 1.0:
            cell_data = np.clip(cell_data / normalize, 0, 1)

        self._to_slice_image_single_field(
            f"{output_filename}_{field_name}",
            cell_data,
            plane_point,
            plane_normal,
            slice_thickness=slice_thickness,
            bounds=bounds,
            grid_res=grid_res,
            cmap=cmap,
            show_axes=show_axes,
            show_colorbar=show_colorbar,
            normalize=normalize,
            width=width,
            height=height,
            width_vec=width_vec,
            height_vec=height_vec,
            **kwargs,
        )
        print(f"\tTime-averaged slice image for field {field_name} saved as {output_filename}.png")

    def to_line_time_average(
        self,
        output_filename: str,
        field_base_name: str,
        start_point,
        end_point,
        resolution: int,
        component: Optional[int] = None,
        radius: float = 1.0,
        keep_state: bool = True,
        **kwargs,
    ):
        """
        Extract time-averaged field data along a line and save to CSV.

        Parameters match to_line(), except:
        - field_base_name: e.g. "velocity" (used to find keys "velocity_0", "velocity_1", ...)
        - keep_state: if True, does not clear the time-average accumulators/cache
        """
        avg_fields = self.finalize_time_average(keep_state=keep_state)

        # Find all components for this base field
        comp_keys = [k for k in avg_fields.keys() if k.startswith(field_base_name + "_")]
        assert len(comp_keys) > 0, f"No averaged components found for base field '{field_base_name}'."

        # Sort by component index (expects suffix _0, _1, _2, ...)
        def _comp_index(k: str) -> int:
            try:
                return int(k.split("_")[-1])
            except Exception:
                return 0

        comp_keys = sorted(comp_keys, key=_comp_index)

        # Build the cell_data to sample along the line
        if component is None and len(comp_keys) > 1:
            print("\tCreating time-averaged line CSV of the field magnitude!")
            comps = [avg_fields[k].astype(np.float64, copy=False) for k in comp_keys]
            cell_data = np.sqrt(np.sum([c * c for c in comps], axis=0)).astype(np.float32)
            field_name = field_base_name + "_magnitude"

        elif component is None and len(comp_keys) == 1:
            print("\tCreating time-averaged line CSV of the scalar field!")
            field_name = comp_keys[0]  # e.g. density_0
            cell_data = avg_fields[field_name]

        else:
            ci = int(component)
            assert 0 <= ci < len(comp_keys), (
                f"Requested component {component} out of range for averaged field '{field_base_name}' "
                f"(has {len(comp_keys)} components)."
            )
            field_name = comp_keys[ci]
            print(f"\tCreating time-averaged line CSV for component {component} of the field!")
            cell_data = avg_fields[field_name]

        # Reuse your existing instantaneous line sampler
        self._to_line_field(
            f"{output_filename}_{field_name}",
            cell_data,
            start_point,
            end_point,
            resolution,
            radius=radius,
            **kwargs,
        )
        print(f"\tTime-averaged line CSV for field {field_name} saved as {output_filename}.csv")

    def _select_surface_field(self, fields_data, field_base_name, component=None):
        comp_keys = [k for k in fields_data.keys() if k.startswith(field_base_name + "_")]
        if len(comp_keys) == 0:
            raise KeyError(f"No field components found for base field '{field_base_name}'")

        comp_keys = sorted(comp_keys, key=lambda k: int(k.rsplit("_", 1)[1]))

        if component is None:
            if len(comp_keys) == 1:
                return comp_keys[0], np.asarray(fields_data[comp_keys[0]], dtype=np.float32)
            comps = np.stack([np.asarray(fields_data[k], dtype=np.float64) for k in comp_keys], axis=1)
            mag = np.linalg.norm(comps, axis=1).astype(np.float32)
            return f"{field_base_name}_magnitude", mag

        ci = int(component)
        if ci < 0 or ci >= len(comp_keys):
            raise ValueError(
                f"Component {component} out of range for '{field_base_name}' "
                f"(has {len(comp_keys)} components)"
            )

        key = comp_keys[ci]
        return key, np.asarray(fields_data[key], dtype=np.float32)

    def _load_surface_mesh(self, surface_mesh_filename):
        import trimesh

        mesh = trimesh.load(surface_mesh_filename, force="mesh", process=False)

        if isinstance(mesh, trimesh.Scene):
            meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if len(meshes) == 0:
                raise ValueError(f"No triangle mesh found in '{surface_mesh_filename}'")
            mesh = trimesh.util.concatenate(meshes)

        if mesh.is_empty:
            raise ValueError(f"Loaded mesh '{surface_mesh_filename}' is empty")

        try:
            mesh.remove_unreferenced_vertices()
        except Exception:
            pass
        try:
            mesh.remove_duplicate_faces()
        except Exception:
            pass
        try:
            mesh.remove_degenerate_faces()
        except Exception:
            pass
        try:
            trimesh.repair.fix_normals(mesh, multibody=True)
        except Exception:
            pass

        _ = mesh.vertex_normals
        _ = mesh.face_normals
        return mesh

    def _build_vertex_neighbors(self, faces, n_vertices):
        neighbors = [set() for _ in range(n_vertices)]
        for tri in np.asarray(faces, dtype=np.int32):
            i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
            neighbors[i].update((j, k))
            neighbors[j].update((i, k))
            neighbors[k].update((i, j))
        return [
            np.fromiter(nbrs, dtype=np.int32) if len(nbrs) else np.empty(0, dtype=np.int32)
            for nbrs in neighbors
        ]

    def _repair_and_smooth_vertex_normals(self, mesh):
        """
        
        """
        normals = np.asarray(mesh.vertex_normals, dtype=np.float32).copy()
        verts = np.asarray(mesh.vertices, dtype=np.float32)

        # Check for bad normals
        nrm = np.linalg.norm(normals, axis=1)
        bad = nrm < 1e-12
        
        if np.any(bad):
            # Fallback: use vertex position relative to center
            center = verts.mean(axis=0)
            fallback = verts - center
            fallback_norm = np.linalg.norm(fallback, axis=1, keepdims=True)
            fallback /= np.maximum(fallback_norm, 1e-12)
            normals[bad] = fallback[bad]
        
        return normals.astype(np.float32)

    def _build_face_normals_from_vertex_normals(self, faces, vertex_normals):
        f = np.asarray(faces, dtype=np.int32)
        vn = np.asarray(vertex_normals, dtype=np.float32)
        face_normals = vn[f].mean(axis=1)
        face_normals /= np.maximum(np.linalg.norm(face_normals, axis=1, keepdims=True), 1e-20)
        return face_normals.astype(np.float32)

    def _sample_surface_scalar_one_side(
        self,
        surface_points,
        surface_normals,
        cell_values,
        sample_dx,
        shell_factors=(0.75, 1.25, 1.75),
        k=24,
        power=2.0,
        max_distance=None,
        half_space_tolerance=0.25,
        aggregate="median",
    ):
        
        sample_dx = float(sample_dx)
        surface_points = np.asarray(surface_points, dtype=np.float32)
        surface_normals = np.asarray(surface_normals, dtype=np.float32)
        cell_values = np.asarray(cell_values, dtype=np.float32)

        surface_normals /= np.maximum(np.linalg.norm(surface_normals, axis=1, keepdims=True), 1e-20)

        shell_factors = np.asarray(shell_factors, dtype=np.float32)
        shell_offsets = shell_factors * sample_dx

        queries = surface_points[:, None, :] + surface_normals[:, None, :] * shell_offsets[None, :, None]
        n_points = queries.shape[0]
        n_shells = queries.shape[1]

        queries_flat = queries.reshape(-1, 3)
        base_points_flat = np.repeat(surface_points, n_shells, axis=0)
        normals_flat = np.repeat(surface_normals, n_shells, axis=0)

        tree = self.kd_tree 
        kk = min(int(k), len(self.centroids))
        distances, indices = tree.query(queries_flat, k=kk)

        if kk == 1:
            distances = distances[:, None]
            indices = indices[:, None]

        neighbor_points = self.centroids[indices]
        neighbor_values = cell_values[indices]

        signed = np.einsum("qki,qi->qk", neighbor_points - base_points_flat[:, None, :], normals_flat)
        valid = signed >= (-half_space_tolerance * sample_dx)

        if max_distance is not None:
            valid &= distances <= float(max_distance)

        weights = 1.0 / np.maximum(distances, 1e-12) ** float(power)
        weights *= valid

        dead = np.sum(weights, axis=1) <= 0.0
        if np.any(dead):
            fallback = 1.0 / np.maximum(distances[dead], 1e-12) ** float(power)
            weights[dead] = fallback

        wsum = np.sum(weights, axis=1)
        sampled = np.sum(weights * neighbor_values, axis=1) / np.maximum(wsum, 1e-20)

        local_mean = sampled[:, None]
        local_spread = np.sum(weights * np.abs(neighbor_values - local_mean), axis=1) / np.maximum(wsum, 1e-20)

        sampled = sampled.reshape(n_points, n_shells)
        support = wsum.reshape(n_points, n_shells)
        spread = local_spread.reshape(n_points, n_shells)

        if aggregate == "mean":
            mapped = np.mean(sampled, axis=1)
        else:
            mapped = np.median(sampled, axis=1)

        score = np.median(support / np.maximum(spread, 1e-6), axis=1)

        return mapped.astype(np.float32), score.astype(np.float32)

    def _sample_surface_scalar_bidirectional(
        self,
        surface_points,
        surface_normals,
        cell_values,
        sample_dx,
        shell_factors=(0.75, 1.25, 1.75),
        k=24,
        power=2.0,
        max_distance=None,
        half_space_tolerance=0.25,
        aggregate="median",
    ):
        plus_vals, plus_score = self._sample_surface_scalar_one_side(
            surface_points=surface_points,
            surface_normals=surface_normals,
            cell_values=cell_values,
            sample_dx=sample_dx,
            shell_factors=shell_factors,
            k=k,
            power=power,
            max_distance=max_distance,
            half_space_tolerance=half_space_tolerance,
            aggregate=aggregate,
        )

        minus_vals, minus_score = self._sample_surface_scalar_one_side(
            surface_points=surface_points,
            surface_normals=-surface_normals,
            cell_values=cell_values,
            sample_dx=sample_dx,
            shell_factors=shell_factors,
            k=k,
            power=power,
            max_distance=max_distance,
            half_space_tolerance=half_space_tolerance,
            aggregate=aggregate,
        )

        use_minus = ~(minus_score > plus_score)

        mapped = plus_vals.copy()
        mapped[use_minus] = minus_vals[use_minus]

        chosen_normals = surface_normals.copy()
        chosen_normals[use_minus] *= -1.0

        return mapped.astype(np.float32), chosen_normals.astype(np.float32), use_minus

    def _smooth_surface_scalar(self, values, faces, iterations=1, relaxation=0.25):
        """
        Vectorized Laplacian smoothing via sparse matrix multiply.
        ~100x faster than per-vertex loops.
        """
        from scipy.sparse import csr_matrix       
        
        if iterations <= 0:
            return values
        
        n_verts = len(values)
        
        # Build adjacency matrix (sparse)
        rows, cols, data = [], [], []
        for tri in faces:
            i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
            # Each edge contributes to the Laplacian
            for (a, b) in [(i, j), (j, k), (k, i)]:
                rows.extend([a, b])
                cols.extend([b, a])
                data.extend([1.0, 1.0])
        
        # Normalize: each vertex's neighbors
        L = csr_matrix((data, (rows, cols)), shape=(n_verts, n_verts))
        L.data /= np.asarray(L.sum(axis=1)).flatten()[L.nonzero()[0]]
        
        # Apply smoothing
        v = values.copy()
        for _ in range(iterations):
            v = (1.0 - relaxation) * v + relaxation * (L @ v)
        
        return v.astype(np.float32)

    def _write_polydata_vtk(self, vtk_filename, vertices, faces, point_data=None, cell_data=None):
        """
        Fast VTK ASCII write using batch numpy formatting.
        """
        import io
        
        vertices = np.asarray(vertices, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.int32)
        
        buf = io.StringIO()
        
        # Header
        buf.write("# vtk DataFile Version 3.0\n")
        buf.write("surface_field_map\n")
        buf.write("ASCII\n")
        buf.write("DATASET POLYDATA\n")
        
        # Points — write entire array at once via numpy
        buf.write(f"POINTS {len(vertices)} float\n")
        np.savetxt(buf, vertices, fmt='%.6f', delimiter=' ')
        
        # Polygons — prepend "3" column, write at once
        buf.write(f"POLYGONS {len(faces)} {len(faces) * 4}\n")
        poly_block = np.column_stack([np.full(len(faces), 3, dtype=np.int32), faces])
        np.savetxt(buf, poly_block, fmt='%d', delimiter=' ')
        
        # Point data
        if point_data:
            buf.write(f"POINT_DATA {len(vertices)}\n")
            for name, arr in point_data.items():
                arr = np.asarray(arr, dtype=np.float32)
                if arr.ndim == 1:
                    buf.write(f"SCALARS {name} float 1\n")
                    buf.write("LOOKUP_TABLE default\n")
                    np.savetxt(buf, arr, fmt='%.6f')
                elif arr.ndim == 2 and arr.shape[1] == 3:
                    buf.write(f"VECTORS {name} float\n")
                    np.savetxt(buf, arr, fmt='%.6f', delimiter=' ')
                else:
                    raise ValueError(f"Unsupported point_data shape for '{name}': {arr.shape}")
        
        # Cell data
        if cell_data:
            buf.write(f"CELL_DATA {len(faces)}\n")
            for name, arr in cell_data.items():
                arr = np.asarray(arr, dtype=np.float32)
                if arr.ndim == 1:
                    buf.write(f"SCALARS {name} float 1\n")
                    buf.write("LOOKUP_TABLE default\n")
                    np.savetxt(buf, arr, fmt='%.6f')
                elif arr.ndim == 2 and arr.shape[1] == 3:
                    buf.write(f"VECTORS {name} float\n")
                    np.savetxt(buf, arr, fmt='%.6f', delimiter=' ')
                else:
                    raise ValueError(f"Unsupported cell_data shape for '{name}': {arr.shape}")
        
        # Single disk write
        with open(vtk_filename, "w") as f:
            f.write(buf.getvalue())
            
    def _fields_data_to_surface_vtk(
        self,
        output_filename,
        surface_mesh_filename,
        fields_data,
        field_base_name,
        component=None,
        sample_dx=None,
        shell_factors=(0.75, 1.25, 1.75),
        k=24,
        power=2.0,
        max_distance=None,
        half_space_tolerance=0.25,
        aggregate="median",
        smooth_iterations=2,
        smooth_relaxation=0.25,
        export_debug_arrays=True,
    ):
        tic_write = time.perf_counter()
        field_name, cell_values = self._select_surface_field(fields_data, field_base_name, component=component)

        mesh = surface_mesh_filename
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        toc_write = time.perf_counter()
        print(f"\tSurface field surface loaded in {toc_write - tic_write:0.1f} seconds")
        if sample_dx is None:
            sample_dx = min(float(vs) for (_, vs, _, _) in self.levels_data)

        if max_distance is None:
            max_distance = 3.0 * float(sample_dx)

        vertex_normals = self._repair_and_smooth_vertex_normals(mesh)
        toc_write = time.perf_counter()
        print(f"\tSurface field smoothed in {toc_write - tic_write:0.1f} seconds")
        vtk_filename = output_filename if output_filename.endswith(".vtk") else output_filename + ".vtk"

        
        mapped, chosen_normals, flipped = self._sample_surface_scalar_bidirectional(
            surface_points=vertices,
            surface_normals=vertex_normals,
            cell_values=cell_values,
            sample_dx=sample_dx,
            shell_factors=shell_factors,
            k=k,
            power=power,
            max_distance=max_distance,
            half_space_tolerance=half_space_tolerance,
            aggregate=aggregate,
        )
        toc_write = time.perf_counter()
        print(f"\tSurface field mapped in {toc_write - tic_write:0.1f} seconds")
        if smooth_iterations > 0:
            mapped = self._smooth_surface_scalar(
                mapped,
                faces,
                iterations=smooth_iterations,
                relaxation=smooth_relaxation,
            )
        toc_write = time.perf_counter()
        print(f"\tSurface field surface results smoothed in {toc_write - tic_write:0.1f} seconds")
        point_data = {field_name: mapped}
        if export_debug_arrays:
            point_data["chosen_normal"] = chosen_normals
            point_data["normal_flipped"] = flipped.astype(np.float32)

        self._write_polydata_vtk(
            vtk_filename,
            vertices,
            faces,
            point_data=point_data,
            cell_data=None,
        )
        
        toc_write = time.perf_counter()
        print(f"\tSurface field written to {vtk_filename} in {toc_write - tic_write:0.1f} seconds")
        return mapped

    def to_surface_vtk(
        self,
        output_filename,
        surface_mesh_filename,
        field_neon_dict,
        field_base_name,
        component=None,
        sample_dx=None,
        shell_factors=(1.25,),
        k=10,
        power=2.0,
        max_distance=None,
        half_space_tolerance=0.25,
        aggregate="median",
        smooth_iterations=2,
        smooth_relaxation=0.2,
        export_debug_arrays=True,
    ):
        fields_data = self.get_fields_data(field_neon_dict)
        return self._fields_data_to_surface_vtk(
            output_filename=output_filename,
            surface_mesh_filename=surface_mesh_filename,
            fields_data=fields_data,
            field_base_name=field_base_name,
            component=component,
            sample_dx=sample_dx,
            shell_factors=shell_factors,
            k=k,
            power=power,
            max_distance=max_distance,
            half_space_tolerance=half_space_tolerance,
            aggregate=aggregate,
            smooth_iterations=smooth_iterations,
            smooth_relaxation=smooth_relaxation,
            export_debug_arrays=export_debug_arrays,
        )

    def to_surface_vtk_time_average(
        self,
        output_filename,
        surface_mesh_filename,
        field_base_name,
        component=None,
        keep_state=True,
        sample_dx=None,
        shell_factors=(1.25, ),
        k=10,
        power=2.0,
        max_distance=None,
        half_space_tolerance=0.25,
        aggregate="median",
        smooth_iterations=2,
        smooth_relaxation=0.2,
        export_debug_arrays=True,
    ):
        avg_fields = self.finalize_time_average(keep_state=keep_state)
        return self._fields_data_to_surface_vtk(
            output_filename=output_filename,
            surface_mesh_filename=surface_mesh_filename,
            fields_data=avg_fields,
            field_base_name=field_base_name,
            component=component,
            sample_dx=sample_dx,
            shell_factors=shell_factors,
            k=k,
            power=power,
            max_distance=max_distance,
            half_space_tolerance=half_space_tolerance,
            aggregate=aggregate,
            smooth_iterations=smooth_iterations,
            smooth_relaxation=smooth_relaxation,
            export_debug_arrays=export_debug_arrays,
        )
    
