"""
Multi-resolution AABB-Close boundary masker with morphological closing.

Extends the AABB-Close masker for Neon multi-resolution grids, applying
dilate-then-erode operations to fill narrow channels with solid voxels.

It also (optionally) detects fluid pockets that are fully enclosed by solid
voxels and retags them as solid.  See ``_construct_neon`` for the
reachability flood fill and ``_seal_enclosed_fluid`` for the driver.
"""

import time
import numpy as np
import warp as wp
from typing import Any
from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.boundary_masker import MeshMaskerAABBClose
from xlb.operator.operator import Operator
from xlb.cell_type import BC_SOLID


# Slots of the per-level counter array used by the enclosed-fluid detection.  Counting
# happens with atomics into a warp array captured by the Neon kernels, which is the only
# reduction available: Neon's python layer exposes none.
SEAL_COUNT_UNREFINED = 0  # voxels actually simulated at this level (no children)
SEAL_COUNT_FLUID = 1  # of those, the non-solid ones before sealing
SEAL_COUNT_SEEDED = 2  # of those, the ones seeded as "outside"
SEAL_COUNT_REACHED = 3  # cumulative number of voxels the flood fill has flipped
SEAL_COUNT_SEALED = 4  # voxels retagged as solid
SEAL_COUNT_SLOTS = 5


class _SealTimer:
    """Wall-clock stopwatch for the enclosed-fluid phases.

    ``lap`` returns the time since the previous lap and ``total`` the time since the
    start.  Neon launches are asynchronous, so every measurement point waits for the
    device first; without that the laps would only time the launch calls.  The wait is
    Neon's own backend sync, not ``wp.synchronize()`` -- see ``_make_device_sync``.
    """

    def __init__(self, sync):
        self._sync = sync
        self._sync()
        self._start = time.perf_counter()
        self._lap = self._start

    def lap(self):
        self._sync()
        now = time.perf_counter()
        seconds = now - self._lap
        self._lap = now
        return seconds

    def elapsed(self):
        return time.perf_counter() - self._lap

    def total(self):
        self._sync()
        return time.perf_counter() - self._start


class MultiresMeshMaskerAABBClose(MeshMaskerAABBClose):
    """
    Operator for creating boundary missing_mask from mesh using Axis-Aligned Bounding Box (AABB) voxelization
    in multiresolution simulations (NEON backend). It takes in a number of close_voxels to perform morphological
    operations (dilate followed by erode) to ensure small channels are filled with solid voxels.

    This version provides NEON-specific functionals working on multires partitions (mPartition) and bIndex.

    Enclosed-fluid sealing
    ----------------------
    After voxelization, fluid voxels that cannot be reached from the outside of the
    domain through a 6-connected path of non-solid voxels are retagged as ``BC_SOLID``.
    Reachability is evaluated on the voxelized ``bc_mask`` (post morphological close),
    so watertight geometry is not required: whatever the voxelizer produced as a solid
    shell is what closes a pocket.  Multiple disjoint pockets (cabin, tyre interiors,
    ...) are handled at once, and regions that are open to the outside (engine bay,
    wheel arches, ...) are left untouched.

    The flood fill follows the same voxel-to-voxel connectivity the multires solver
    uses when streaming, so inter-level links are respected:

    * same level    -> ``neon_read_ngh`` on a neighbour that is a real (unrefined) cell,
    * coarse -> fine -> ``neon_lbm_read_coarser_ngh`` (the "uncle" a fine cell explodes
      from when it has no same-level neighbour in that direction),
    * fine -> coarse -> ``neon_read_child`` on the halo cell that sits on top of the
      finer cells (the cell a coarse voxel coalesces from), restricted to the four
      children touching the shared face.

    Cells with children (halo/ghost cells covering a refined region) are never updated
    directly; they only act as the fine/coarse hand-off, exactly as in the solver.

    Every ``seal_*`` parameter below can also be given per boundary condition through
    the voxelization method, without touching the stepper, e.g.::

        MeshVoxelizationMethod("AABB_CLOSE", close_voxels=3, seal_enclosed_fluid=False)

    Values found in ``bc.voxelization_method.options`` win over the constructor
    arguments, which act as the defaults.

    Parameters
    ----------
    seal_enclosed_fluid : bool
        Enable the enclosed-pocket detection and retagging.
    seal_seed_coarsest_level : bool
        Seed the flood fill with every non-solid unrefined voxel of the coarsest level
        (the far field) in addition to the domain-boundary voxels.  This is what makes
        the fill cheap, since the free stream does not have to be traversed voxel by
        voxel.  Disable it if an enclosed pocket may be large enough to contain
        unrefined voxels of the coarsest level.
    seal_max_iterations : int, optional
        Safety cap on the number of flood-fill sweeps.  ``None`` derives a bound from
        the per-level sparsity patterns.  The fill normally stops on its own as soon as
        a block of sweeps flips no voxel, which is the exact fixed point; the cap only
        bites if something pathological happens.  Hitting it means the fill is still
        walking, so nothing is sealed and a warning is printed rather than risking
        walling off fluid that is in fact open.
    seal_refresh_masks : bool
        Re-run the AABB pass after sealing so that voxels which end up next to a newly
        created solid voxel (possible for diagonal links, since connectivity is
        6-connected) get consistent ``bc_mask``/``missing_mask``/distances.
    seal_debug_vti : str, optional
        Path of a ``.vti`` file to dump the reachability field to for inspection.
    """

    def __init__(
        self,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
        close_voxels: int = None,
        seal_enclosed_fluid: bool = False,
        seal_seed_coarsest_level: bool = True,
        seal_max_iterations: int = None,
        seal_refresh_masks: bool = True,
        seal_debug_vti: str = None,
    ):
        self.seal_enclosed_fluid = seal_enclosed_fluid
        self.seal_seed_coarsest_level = seal_seed_coarsest_level
        self.seal_max_iterations = seal_max_iterations
        self.seal_refresh_masks = seal_refresh_masks
        self.seal_debug_vti = seal_debug_vti

        super().__init__(velocity_set, precision_policy, compute_backend, close_voxels)
        if self.compute_backend in [ComputeBackend.JAX, ComputeBackend.WARP]:
            raise NotImplementedError(f"Operator {self.__class__.__name__} not supported in {self.compute_backend} backend.")

        # Build and store NEON dicts
        self.neon_functional_dict, self.neon_container_dict = self._construct_neon()

    def _construct_neon(self):
        import neon

        # Use the warp functionals from the base (for reference), but implement NEON variants here
        functional_dict_warp, _ = self._construct_warp()
        functional_erode_warp = functional_dict_warp.get("functional_erode")
        functional_dilate_warp = functional_dict_warp.get("functional_dilate")
        functional_solid = functional_dict_warp.get("functional_solid")
        # We will not directly reuse functional_solid / functional_aabb from warp; we write NEON-specific ones.

        # We also need lattice info for neighbor iteration
        _c = self.velocity_set.c
        _q = self.velocity_set.q
        _opp_indices = self.velocity_set.opp_indices

        # Set local constants
        lattice_central_index = self.velocity_set.center_index

        # 6-connectivity axis directions used by the enclosed-fluid flood fill.
        _ax6_np = np.array(
            [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
            dtype=np.int32,
        )

        # For each axis direction, the local indices of the children of the neighbouring
        # halo cell that touch the shared face.  With a refinement factor of 2 a face is
        # covered by 4 children: the two off-axis coordinates run over {0, 1} while the
        # on-axis coordinate is pinned to the side facing us.
        _refinement_factor = 2
        _children_per_face = _refinement_factor**2
        _child6_np = np.zeros((_ax6_np.shape[0] * _children_per_face, 3), dtype=np.int32)
        for _i in range(_ax6_np.shape[0]):
            _axis = int(np.nonzero(_ax6_np[_i])[0][0])
            _face = 0 if _ax6_np[_i, _axis] > 0 else _refinement_factor - 1
            _others = [_a for _a in range(3) if _a != _axis]
            _ci = 0
            for _u in range(_refinement_factor):
                for _v in range(_refinement_factor):
                    _off = [0, 0, 0]
                    _off[_axis] = _face
                    _off[_others[0]] = _u
                    _off[_others[1]] = _v
                    _child6_np[_i * _children_per_face + _ci] = _off
                    _ci += 1

        # Warp cannot capture a raw numpy array, so these have to become warp constants
        # the same way the velocity set exposes its own lattice tables.
        _ax6 = wp.constant(wp.mat(_ax6_np.shape, dtype=wp.int32)(_ax6_np))
        _child6 = wp.constant(wp.mat(_child6_np.shape, dtype=wp.int32)(_child6_np))

        @wp.func
        def mres_hit_normal_weight(
            norm_dir: wp.vec3f,
            normal: wp.vec3f,
            ray_dist: wp.float32,
            max_length: wp.float32,
        ):
            # normal is oriented against the ray direction before this is called.
            # alignment = 1.0 for face-on hits, 0.0 for grazing hits.
            alignment = -wp.dot(norm_dir, normal)
            alignment = wp.clamp(alignment, wp.float32(0.0), wp.float32(1.0))

            # Prefer closer hits, but prevent tiny distances from dominating.
            rel_dist = ray_dist / (max_length if max_length > 0.0 else wp.float32(1.0))
            distance_weight = wp.float32(1.0) / wp.max(rel_dist, wp.float32(0.25))

            # D3Q27 has more diagonal links than axis links.
            # This reduces over-weighting of edge/corner directions.
            link_weight = wp.float32(1.0) / wp.max(max_length * max_length, wp.float32(1.0))

            return alignment * alignment * distance_weight * link_weight


        
        # Main AABB close: sets bc_mask, missing_mask, distances based on solid_mask
        # bc_mask: wp.uint8, missing_mask: wp.uint8, distances: dtype from precision policy (float)
        @wp.func
        def mres_functional_aabb(
            index: Any,
            mesh_id: wp.uint64,
            id_number: wp.int32,
            distances_pn: Any,
            bc_mask_pn: Any,
            missing_mask_pn: Any,
            solid_mask_pn: Any,
            needs_mesh_distance: bool,
            normal_vector: Any,
            normal_distance: Any,
        ):
            reset_done = wp.bool(False)

            cell_center = self.helper_masker.index_to_position(bc_mask_pn, index)

            solid_val = wp.neon_read(solid_mask_pn, index, 0)
            bc_val = wp.neon_read(bc_mask_pn, index, 0)

            # Keep existing solid or already-owned boundary cells untouched.
            if solid_val == wp.uint8(BC_SOLID) or bc_val == wp.uint8(BC_SOLID):
                wp.neon_write(bc_mask_pn, index, 0, wp.uint8(BC_SOLID))
                return

            # Four clusters allow one dominant wall plus a few local feature normals.
            cluster_n0 = wp.vec3f(0.0, 0.0, 0.0)
            cluster_n1 = wp.vec3f(0.0, 0.0, 0.0)
            cluster_n2 = wp.vec3f(0.0, 0.0, 0.0)
            cluster_n3 = wp.vec3f(0.0, 0.0, 0.0)

            cluster_w0 = wp.float32(0.0)
            cluster_w1 = wp.float32(0.0)
            cluster_w2 = wp.float32(0.0)
            cluster_w3 = wp.float32(0.0)

            # These store weighted projected centroid-to-wall distance, not ray length.
            cluster_d0 = wp.float32(0.0)
            cluster_d1 = wp.float32(0.0)
            cluster_d2 = wp.float32(0.0)
            cluster_d3 = wp.float32(0.0)

            total_normal = wp.vec3f(0.0, 0.0, 0.0)
            total_wall_dist = wp.float32(0.0)
            total_weight = wp.float32(0.0)

            # Normals within about 35 degrees are treated as the same surface cluster.
            cluster_cos = wp.float32(0.819152044)

            for direction_idx in range(_q):
                if direction_idx == lattice_central_index:
                    continue

                ngh = wp.neon_ngh_idx(wp.int8(_c[0, direction_idx]), wp.int8(_c[1, direction_idx]), wp.int8(_c[2, direction_idx]))

                is_valid = wp.bool(False)
                nval = wp.neon_read_ngh( solid_mask_pn,  index,  ngh,  0,  wp.uint8(0), is_valid )

                if not is_valid:
                    continue

                if nval != wp.uint8(BC_SOLID):
                    continue

                # If no mesh distance is requested, keep the base ownership behavior:
                # mark the cell as boundary and flag the missing opposite link.
                if not needs_mesh_distance:
                    if not reset_done:
                        for l in range(_q):
                            self.write_field( missing_mask_pn,  index, _opp_indices[l],   wp.uint8(False),  )
                        reset_done = wp.bool(True)

                    self.write_field(bc_mask_pn, index, 0, wp.uint8(id_number))
                    self.write_field( missing_mask_pn, index, _opp_indices[direction_idx],   wp.uint8(True), )
                    continue

                dir_vec = wp.vec3f( wp.float32(_c[0, direction_idx]), wp.float32(_c[1, direction_idx]), wp.float32(_c[2, direction_idx]), )

                max_length = wp.length(dir_vec)
                safe_length = max_length if max_length > 0.0 else wp.float32(1.0)
                norm_dir = dir_vec / safe_length

                # 2.5-link search distance so the second-layer case can still hit.
                query = wp.mesh_query_ray(mesh_id,  cell_center,norm_dir, wp.float32(2.5) * safe_length,)

                # Ownership, missing-mask reset, distances, and normal accumulation only
                # happen after a valid mesh hit. This avoids claiming ownership from
                # solid-mask adjacency when the mesh ray did not actually resolve.                

                if not reset_done:
                    for l in range(_q):
                        self.write_field( missing_mask_pn, index, _opp_indices[l], wp.uint8(False),)
                    reset_done = wp.bool(True)             

                self.write_field(bc_mask_pn, index, 0, wp.uint8(id_number))
                self.write_field(missing_mask_pn,  index,   _opp_indices[direction_idx],  wp.uint8(True),)

                if query.result:
                    ray_dist = query.t
                    normal = query.normal
                else:
                    ray_dist = wp.float32(1.5)
                    normal = -norm_dir
                    
                # Orient the triangle normal against the lattice ray.
                # This makes alignment positive and keeps normal_vector consistently
                # pointing from the wall back toward the fluid cell.
                if wp.dot(norm_dir, normal) > 0.0:
                    normal = -normal

                alignment = -wp.dot(norm_dir, normal)
                alignment = wp.clamp(alignment, wp.float32(0.0), wp.float32(1.0))

                # Goal 1:
                # Store per-link distance normalized by lattice-link length.
                # This remains the ray distance normalized by max_length.
                # Subtract 0.5 to account for aabb solid
                link_distance = ray_dist / safe_length                  
                link_distance = wp.clamp(link_distance-0.5, wp.float32(0.0), wp.float32(1.0))

                if wp.isnan(link_distance) or wp.isinf(link_distance):
                    link_distance = wp.float32(1.0)

                self.write_field( distances_pn,  index, direction_idx,  self.store_dtype(link_distance),)

                # Goal 3:
                # Convert ray length to true wall-normal distance.
                #
                # For a flat plane parallel to voxel faces:
                #   axis ray:     ray_dist = d,           alignment = 1
                #   edge ray:     ray_dist = d / cos,     alignment = cos
                #   corner ray:   ray_dist = d / cos,     alignment = cos
                #
                # Therefore:
                #   wall_dist = ray_dist * alignment = d
                #
                # This is the key fix for avoiding 0.65 / 1.9 in the flat-roof case.
                wall_dist = ray_dist * alignment
                wall_dist = wp.max(wall_dist, wp.float32(0.0))

                if wp.isnan(wall_dist) or wp.isinf(wall_dist):
                    continue

                hit_w = mres_hit_normal_weight(norm_dir, normal, ray_dist, safe_length)

                if hit_w <= 0.0:
                    continue

                weighted_normal = hit_w * normal

                total_normal += weighted_normal
                total_wall_dist += hit_w * wall_dist
                total_weight += hit_w

                added = wp.bool(False)

                # Try to add this hit to an existing normal cluster.
                if cluster_w0 > 0.0 and not added:
                    c_len = wp.length(cluster_n0)
                    c_dir = cluster_n0 / (c_len if c_len > 0.0 else wp.float32(1.0))

                    if wp.dot(normal, c_dir) >= cluster_cos:
                        cluster_n0 += weighted_normal
                        cluster_w0 += hit_w
                        cluster_d0 += hit_w * wall_dist
                        added = wp.bool(True)

                if cluster_w1 > 0.0 and not added:
                    c_len = wp.length(cluster_n1)
                    c_dir = cluster_n1 / (c_len if c_len > 0.0 else wp.float32(1.0))

                    if wp.dot(normal, c_dir) >= cluster_cos:
                        cluster_n1 += weighted_normal
                        cluster_w1 += hit_w
                        cluster_d1 += hit_w * wall_dist
                        added = wp.bool(True)

                if cluster_w2 > 0.0 and not added:
                    c_len = wp.length(cluster_n2)
                    c_dir = cluster_n2 / (c_len if c_len > 0.0 else wp.float32(1.0))

                    if wp.dot(normal, c_dir) >= cluster_cos:
                        cluster_n2 += weighted_normal
                        cluster_w2 += hit_w
                        cluster_d2 += hit_w * wall_dist
                        added = wp.bool(True)

                if cluster_w3 > 0.0 and not added:
                    c_len = wp.length(cluster_n3)
                    c_dir = cluster_n3 / (c_len if c_len > 0.0 else wp.float32(1.0))

                    if wp.dot(normal, c_dir) >= cluster_cos:
                        cluster_n3 += weighted_normal
                        cluster_w3 += hit_w
                        cluster_d3 += hit_w * wall_dist
                        added = wp.bool(True)

                # If no existing cluster matched, start a new cluster if possible.
                if not added:
                    if cluster_w0 <= 0.0:
                        cluster_n0 = weighted_normal
                        cluster_w0 = hit_w
                        cluster_d0 = hit_w * wall_dist
                        added = wp.bool(True)

                if not added:
                    if cluster_w1 <= 0.0:
                        cluster_n1 = weighted_normal
                        cluster_w1 = hit_w
                        cluster_d1 = hit_w * wall_dist
                        added = wp.bool(True)

                if not added:
                    if cluster_w2 <= 0.0:
                        cluster_n2 = weighted_normal
                        cluster_w2 = hit_w
                        cluster_d2 = hit_w * wall_dist
                        added = wp.bool(True)

                if not added:
                    if cluster_w3 <= 0.0:
                        cluster_n3 = weighted_normal
                        cluster_w3 = hit_w
                        cluster_d3 = hit_w * wall_dist
                        added = wp.bool(True)

                # If all clusters are occupied and this hit matches none, add it to
                # the closest normal cluster instead of dropping it.
                if not added:
                    best_i = wp.int32(0)
                    best_dot = wp.float32(-2.0)

                    c_len0 = wp.length(cluster_n0)
                    c_dir0 = cluster_n0 / (c_len0 if c_len0 > 0.0 else wp.float32(1.0))
                    d0 = wp.dot(normal, c_dir0)

                    if d0 > best_dot:
                        best_dot = d0
                        best_i = wp.int32(0)

                    c_len1 = wp.length(cluster_n1)
                    c_dir1 = cluster_n1 / (c_len1 if c_len1 > 0.0 else wp.float32(1.0))
                    d1 = wp.dot(normal, c_dir1)

                    if d1 > best_dot:
                        best_dot = d1
                        best_i = wp.int32(1)

                    c_len2 = wp.length(cluster_n2)
                    c_dir2 = cluster_n2 / (c_len2 if c_len2 > 0.0 else wp.float32(1.0))
                    d2 = wp.dot(normal, c_dir2)

                    if d2 > best_dot:
                        best_dot = d2
                        best_i = wp.int32(2)

                    c_len3 = wp.length(cluster_n3)
                    c_dir3 = cluster_n3 / (c_len3 if c_len3 > 0.0 else wp.float32(1.0))
                    d3 = wp.dot(normal, c_dir3)

                    if d3 > best_dot:
                        best_dot = d3
                        best_i = wp.int32(3)

                    if best_i == wp.int32(0):
                        cluster_n0 += weighted_normal
                        cluster_w0 += hit_w
                        cluster_d0 += hit_w * wall_dist
                    elif best_i == wp.int32(1):
                        cluster_n1 += weighted_normal
                        cluster_w1 += hit_w
                        cluster_d1 += hit_w * wall_dist
                    elif best_i == wp.int32(2):
                        cluster_n2 += weighted_normal
                        cluster_w2 += hit_w
                        cluster_d2 += hit_w * wall_dist
                    else:
                        cluster_n3 += weighted_normal
                        cluster_w3 += hit_w
                        cluster_d3 += hit_w * wall_dist

            if (total_weight <= 0.0) or (not needs_mesh_distance):
                return

            # Goal 2:
            # Use the dominant cluster, not the global average, so random outlier
            # triangle hits do not skew the stored normal.
            best_n = cluster_n0
            best_w = cluster_w0
            best_d = cluster_d0

            if cluster_w1 > best_w:
                best_n = cluster_n1
                best_w = cluster_w1
                best_d = cluster_d1

            if cluster_w2 > best_w:
                best_n = cluster_n2
                best_w = cluster_w2
                best_d = cluster_d2

            if cluster_w3 > best_w:
                best_n = cluster_n3
                best_w = cluster_w3
                best_d = cluster_d3

            # Fallback for pathological cases.
            if best_w <= 0.0:
                best_n = total_normal
                best_w = total_weight
                best_d = total_wall_dist

            avg_wall_dist = best_d / wp.max(best_w, wp.float32(1.0e-8))
            avg_wall_dist = wp.max(avg_wall_dist, wp.float32(0.0))

            avg_normal_len = wp.length(best_n)

            if avg_normal_len > 1.0e-8:
                avg_normal = best_n / avg_normal_len
            else:
                fallback_len = wp.length(total_normal)
                if fallback_len > 1.0e-8:
                    avg_normal = total_normal / fallback_len
                else:
                    avg_normal = wp.vec3f(0.0, 0.0, 0.0)

            self.write_field(normal_distance, index, 0, self.store_dtype(avg_wall_dist))
            self.write_field(normal_vector, index, 0, self.store_dtype(avg_normal[0]))
            self.write_field(normal_vector, index, 1, self.store_dtype(avg_normal[1]))
            self.write_field(normal_vector, index, 2, self.store_dtype(avg_normal[2]))
                        

        # Enclosed-fluid detection
        #
        # ``reach`` holds 1 for every non-solid voxel that is 6-connected to the outside
        # of the domain and 0 otherwise.  Only unrefined ("real") voxels are updated;
        # refined voxels (those with children) are pure hand-off cells.

        @wp.func
        def mres_functional_seed(
            index: Any,
            bc_mask_pn: Any,
            reach_pn: Any,
            dim_x: wp.int32,
            dim_y: wp.int32,
            dim_z: wp.int32,
            spacing: wp.int32,
            seed_whole_level: bool,
            counters: wp.array(dtype=wp.int32),
            counter_base: wp.int32,
        ):
            # Refined voxels are covered by finer voxels and are not simulated here.
            if wp.neon_has_child(reach_pn, index):
                return

            wp.atomic_add(counters, counter_base + SEAL_COUNT_UNREFINED, 1)

            if wp.neon_read(bc_mask_pn, index, 0) == wp.uint8(BC_SOLID):
                return

            wp.atomic_add(counters, counter_base + SEAL_COUNT_FLUID, 1)

            if seed_whole_level:
                wp.neon_write(reach_pn, index, 0, wp.uint8(1))
                wp.atomic_add(counters, counter_base + SEAL_COUNT_SEEDED, 1)
                return

            # Global index is expressed in finest-level voxels and points at the lower
            # corner of the voxel, so a voxel of this level spans [g, g + spacing).
            global_idx = wp.neon_global_idx(reach_pn, index)
            gx = wp.neon_get_x(global_idx)
            gy = wp.neon_get_y(global_idx)
            gz = wp.neon_get_z(global_idx)

            on_domain_boundary = (
                gx <= 0
                or gy <= 0
                or gz <= 0
                or gx + spacing >= dim_x
                or gy + spacing >= dim_y
                or gz + spacing >= dim_z
            )

            if on_domain_boundary:
                wp.neon_write(reach_pn, index, 0, wp.uint8(1))
                wp.atomic_add(counters, counter_base + SEAL_COUNT_SEEDED, 1)

        @wp.func
        def mres_functional_fill(
            index: Any,
            bc_mask_pn: Any,
            reach_pn: Any,
            counters: wp.array(dtype=wp.int32),
            counter_base: wp.int32,
        ):
            if wp.neon_has_child(reach_pn, index):
                return

            if wp.neon_read(bc_mask_pn, index, 0) == wp.uint8(BC_SOLID):
                return

            # Monotone update: once reached, always reached. Reading and writing the
            # same field in place is therefore race free and converges faster.
            if wp.neon_read(reach_pn, index, 0) != wp.uint8(0):
                return

            reached = wp.bool(False)

            for direction_idx in range(6):
                if reached:
                    continue

                ngh = wp.neon_ngh_idx(
                    wp.int8(_ax6[direction_idx, 0]),
                    wp.int8(_ax6[direction_idx, 1]),
                    wp.int8(_ax6[direction_idx, 2]),
                )

                is_valid = wp.bool(False)
                nval = wp.neon_read_ngh(reach_pn, index, ngh, 0, wp.uint8(0), is_valid)

                if is_valid:
                    if wp.neon_has_finer_ngh(reach_pn, index, ngh):
                        # The same-level neighbour is the halo cell sitting on top of a
                        # refined region, so the actual neighbours are its children at
                        # the finer level. This is the coalescence side of the solver.
                        halo_index = wp.neon_ngh_idx(reach_pn, index, ngh)

                        for child_idx in range(_children_per_face):
                            child_row = direction_idx * _children_per_face + child_idx
                            child = wp.neon_ngh_idx(
                                wp.int8(_child6[child_row, 0]),
                                wp.int8(_child6[child_row, 1]),
                                wp.int8(_child6[child_row, 2]),
                            )

                            child_valid = wp.bool(False)
                            cval = wp.neon_read_child(reach_pn, halo_index, child, 0, wp.uint8(0), child_valid)

                            if child_valid and cval != wp.uint8(0):
                                reached = wp.bool(True)
                    else:
                        # Plain same-level neighbour.
                        if nval != wp.uint8(0):
                            reached = wp.bool(True)
                else:
                    # No same-level neighbour: the neighbour may live on the coarser
                    # level. This is the explosion side of the solver.
                    if wp.neon_has_parent(reach_pn, index):
                        uncle_valid = wp.bool(False)
                        uval = wp.neon_lbm_read_coarser_ngh(reach_pn, index, ngh, 0, wp.uint8(0), uncle_valid)

                        if uncle_valid and uval != wp.uint8(0):
                            reached = wp.bool(True)

            if reached:
                wp.neon_write(reach_pn, index, 0, wp.uint8(1))

                # Every voxel flips at most once, so this counter both reports progress
                # and detects convergence: a sweep block that adds nothing is a fixed
                # point. The atomic traffic over the whole fill is one add per voxel.
                wp.atomic_add(counters, counter_base + SEAL_COUNT_REACHED, 1)

        @wp.func
        def mres_functional_seal(
            index: Any,
            bc_mask_pn: Any,
            solid_mask_pn: Any,
            reach_pn: Any,
            counters: wp.array(dtype=wp.int32),
            counter_base: wp.int32,
        ):
            if wp.neon_has_child(reach_pn, index):
                return

            if wp.neon_read(bc_mask_pn, index, 0) == wp.uint8(BC_SOLID):
                return

            if wp.neon_read(reach_pn, index, 0) != wp.uint8(0):
                return

            # Unreachable fluid: the whole pocket (its interior plus the boundary voxels
            # lining it) becomes solid.
            wp.neon_write(bc_mask_pn, index, 0, wp.uint8(BC_SOLID))
            wp.neon_write(solid_mask_pn, index, 0, wp.uint8(BC_SOLID))
            wp.atomic_add(counters, counter_base + SEAL_COUNT_SEALED, 1)

        # Containers

        # Erode: f_field -> f_field_out
        @neon.Container.factory(name="Erode")
        def container_erode(f_field: wp.array3d(dtype=Any), f_field_out: wp.array3d(dtype=Any), level: int):
            def erode_launcher(loader: neon.Loader):
                loader.set_mres_grid(f_field.get_grid(), level)
                f_field_pn = loader.get_mres_read_handle(f_field)
                f_field_out_pn = loader.get_mres_write_handle(f_field_out)

                @wp.func
                def erode_kernel(index: Any):
                    functional_erode_warp(index, f_field_pn, f_field_out_pn)

                loader.declare_kernel(erode_kernel)

            return erode_launcher

        # Dilate: f_field -> f_field_out
        @neon.Container.factory(name="Dilate")
        def container_dilate(f_field: wp.array3d(dtype=Any), f_field_out: wp.array3d(dtype=Any), level: int):
            def dilate_launcher(loader: neon.Loader):
                loader.set_mres_grid(f_field.get_grid(), level)
                f_field_pn = loader.get_mres_read_handle(f_field)
                f_field_out_pn = loader.get_mres_write_handle(f_field_out)

                @wp.func
                def dilate_kernel(index: Any):
                    functional_dilate_warp(index, f_field_pn, f_field_out_pn)

                loader.declare_kernel(dilate_kernel)

            return dilate_launcher

        # Solid mask: voxelize mesh into solid_mask
        @neon.Container.factory(name="Solid")
        def container_solid(mesh_id: wp.uint64, solid_mask: wp.array3d(dtype=wp.uint8), level: int):
            def solid_launcher(loader: neon.Loader):
                loader.set_mres_grid(solid_mask.get_grid(), level)
                solid_mask_pn = loader.get_mres_write_handle(solid_mask)

                @wp.func
                def solid_kernel(index: Any):
                    # apply the functional
                    functional_solid(index, mesh_id, solid_mask_pn, wp.vec3f(0.0, 0.0, 0.0))

                loader.declare_kernel(solid_kernel)

            return solid_launcher

        # Main AABB container
        @neon.Container.factory(name="MeshMaskerAABBClose")
        def container(
            mesh_id: Any,
            id_number: Any,
            distances: Any,
            bc_mask: Any,
            missing_mask: Any,
            solid_mask: Any,
            needs_mesh_distance: Any,            
            normal_vector: Any,
            normal_distance: Any,
            level: Any,
        ):
            def aabb_launcher(loader: neon.Loader):
                loader.set_mres_grid(bc_mask.get_grid(), level)
                distances_pn = loader.get_mres_write_handle(distances)
                bc_mask_pn = loader.get_mres_write_handle(bc_mask)
                missing_mask_pn = loader.get_mres_write_handle(missing_mask)
                solid_mask_pn = loader.get_mres_write_handle(solid_mask)
                norm_vec_pn = loader.get_mres_write_handle(normal_vector)
                norm_dist_pn = loader.get_mres_write_handle(normal_distance)

                @wp.func
                def aabb_kernel(index: Any):
                    mres_functional_aabb(
                        index,
                        mesh_id,
                        id_number,
                        distances_pn,
                        bc_mask_pn,
                        missing_mask_pn,
                        solid_mask_pn,
                        needs_mesh_distance,
                        norm_vec_pn,
                        norm_dist_pn
                    )

                loader.declare_kernel(aabb_kernel)

            return aabb_launcher

        # Seed the reachability field
        @neon.Container.factory(name="MeshMaskerSeedReach")
        def container_seed(
            bc_mask: Any,
            reach: Any,
            dim_x: Any,
            dim_y: Any,
            dim_z: Any,
            spacing: Any,
            seed_whole_level: Any,
            counters: Any,
            level: Any,
        ):
            def seed_launcher(loader: neon.Loader):
                loader.set_mres_grid(bc_mask.get_grid(), level)
                bc_mask_pn = loader.get_mres_read_handle(bc_mask)
                reach_pn = loader.get_mres_write_handle(reach)
                counter_base = level * SEAL_COUNT_SLOTS

                @wp.func
                def seed_kernel(index: Any):
                    mres_functional_seed(
                        index,
                        bc_mask_pn,
                        reach_pn,
                        dim_x,
                        dim_y,
                        dim_z,
                        spacing,
                        seed_whole_level,
                        counters,
                        counter_base,
                    )

                loader.declare_kernel(seed_kernel)

            return seed_launcher

        # One flood-fill sweep of the reachability field (in place)
        @neon.Container.factory(name="MeshMaskerFillReach")
        def container_fill(bc_mask: Any, reach: Any, counters: Any, level: Any):
            def fill_launcher(loader: neon.Loader):
                loader.set_mres_grid(bc_mask.get_grid(), level)
                bc_mask_pn = loader.get_mres_read_handle(bc_mask)
                reach_pn = loader.get_mres_write_handle(reach)
                counter_base = level * SEAL_COUNT_SLOTS

                @wp.func
                def fill_kernel(index: Any):
                    mres_functional_fill(index, bc_mask_pn, reach_pn, counters, counter_base)

                loader.declare_kernel(fill_kernel)

            return fill_launcher

        # Retag unreachable fluid as solid
        @neon.Container.factory(name="MeshMaskerSealEnclosed")
        def container_seal(bc_mask: Any, solid_mask: Any, reach: Any, counters: Any, level: Any):
            def seal_launcher(loader: neon.Loader):
                loader.set_mres_grid(bc_mask.get_grid(), level)
                bc_mask_pn = loader.get_mres_write_handle(bc_mask)
                solid_mask_pn = loader.get_mres_write_handle(solid_mask)
                reach_pn = loader.get_mres_read_handle(reach)
                counter_base = level * SEAL_COUNT_SLOTS

                @wp.func
                def seal_kernel(index: Any):
                    mres_functional_seal(index, bc_mask_pn, solid_mask_pn, reach_pn, counters, counter_base)

                loader.declare_kernel(seal_kernel)

            return seal_launcher

        container_dict = {
            "container_erode": container_erode,
            "container_dilate": container_dilate,
            "container_solid": container_solid,
            "container_aabb": container,
            "container_seed": container_seed,
            "container_fill": container_fill,
            "container_seal": container_seal,
        }

        # Expose NEON functionals too (in case callers want to reuse)
        functional_dict = {
            "mres_functional_aabb": mres_functional_aabb,
            "mres_functional_seed": mres_functional_seed,
            "mres_functional_fill": mres_functional_fill,
            "mres_functional_seal": mres_functional_seal,
        }

        return functional_dict, container_dict

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_implementation(
        self,
        bc,
        distances,
        bc_mask,
        missing_mask,        
        normal_vector, 
        normal_distance,
        stream=0,
    ):
        import neon

        # Prepare inputs
        mesh_id, bc_id = self._prepare_kernel_inputs(bc, bc_mask)

        grid = bc_mask.get_grid()
        # Create fields using new_field
        solid_mask = grid.new_field(cardinality=1, dtype=wp.uint8, memory_type=neon.MemoryType.device())
        solid_mask_out = grid.new_field(
            cardinality=1,
            dtype=wp.uint8,
            memory_type=neon.MemoryType.device(),
            # memory_type=neon.MemoryType.host_device()
        )

        for level in range(grid.num_levels):
            # Initialize to 0
            solid_mask.fill_run(level=level, value=wp.uint8(0), stream_idx=stream)
            solid_mask_out.fill_run(level=level, value=wp.uint8(0), stream_idx=stream)

            # Launch the neon containers
            container_solid = self.neon_container_dict["container_solid"](mesh_id, solid_mask, level)
            container_solid.run(0, container_runtime=neon.Container.ContainerRuntime.neon)

            for _ in range(self.close_voxels):
                container_dilate = self.neon_container_dict["container_dilate"](solid_mask, solid_mask_out, level)
                container_dilate.run(0, container_runtime=neon.Container.ContainerRuntime.neon)
                solid_mask, solid_mask_out = solid_mask_out, solid_mask

            if self.close_voxels % 2 > 0:
                solid_mask, solid_mask_out = solid_mask_out, solid_mask

            for _ in range(self.close_voxels):
                container_erode = self.neon_container_dict["container_erode"](solid_mask_out, solid_mask, level)
                container_erode.run(0, container_runtime=neon.Container.ContainerRuntime.neon)
                solid_mask, solid_mask_out = solid_mask_out, solid_mask

            if self.close_voxels % 2 > 0:
                solid_mask, solid_mask_out = solid_mask_out, solid_mask

            container_aabb = self.neon_container_dict["container_aabb"](
                mesh_id, bc_id, distances, bc_mask, missing_mask, solid_mask, wp.static(bc.needs_mesh_distance), normal_vector, normal_distance, level
            )
            container_aabb.run(0, container_runtime=neon.Container.ContainerRuntime.neon)

        seal_options = self._resolve_seal_options(bc)

        if seal_options["seal_enclosed_fluid"]:
            self._seal_enclosed_fluid(
                seal_options=seal_options,
                grid=grid,
                bc=bc,
                mesh_id=mesh_id,
                bc_id=bc_id,
                distances=distances,
                bc_mask=bc_mask,
                missing_mask=missing_mask,
                solid_mask=solid_mask,
                normal_vector=normal_vector,
                normal_distance=normal_distance,
                stream=stream,
            )

        return distances, bc_mask, missing_mask, normal_vector, normal_distance

    def _resolve_seal_options(self, bc):
        """Merge the enclosed-fluid options of ``bc`` over the constructor defaults.

        Options set on the voxelization method (``MeshVoxelizationMethod("AABB_CLOSE",
        ..., seal_enclosed_fluid=False)``) take precedence, so a case can steer the
        sealing per boundary condition without the stepper having to forward anything.
        """
        defaults = {
            "seal_enclosed_fluid": self.seal_enclosed_fluid,
            "seal_seed_coarsest_level": self.seal_seed_coarsest_level,
            "seal_max_iterations": self.seal_max_iterations,
            "seal_refresh_masks": self.seal_refresh_masks,
            "seal_debug_vti": self.seal_debug_vti,
        }

        options = getattr(getattr(bc, "voxelization_method", None), "options", None) or {}

        unknown = [key for key in options if key.startswith("seal_") and key not in defaults]
        if unknown:
            raise ValueError(f"Unknown enclosed-fluid sealing options {unknown}. Supported: {sorted(defaults.keys())}")

        return {key: options.get(key, value) for key, value in defaults.items()}

    def _estimate_seal_iterations(self, grid, seed_coarsest_level):
        """Number of flood-fill sweeps needed for the reachability fill to converge.

        One sweep advances the reachability front by one voxel per level, so the bound
        is the length of the longest voxel path the front has to walk.  A path can
        descend through the refinement hierarchy, hence the per-level traversal lengths
        are summed.  Levels that are seeded wholesale need no traversal at all.

        The per-level sparsity patterns give the extent actually occupied by each level,
        which is far tighter than the full domain.  If they are unavailable the full
        domain extent is used instead.
        """
        num_levels = grid.num_levels
        dim = grid.dim

        # Levels that are seeded in full do not have to be traversed.
        seed_whole_levels = 1 if (seed_coarsest_level and num_levels > 1) else 0
        levels_to_traverse = range(num_levels - seed_whole_levels)

        patterns = getattr(grid, "sparsity_pattern_list", None)

        if patterns is None or len(patterns) != num_levels:
            # Fall back to the full domain measured in the finest voxels.
            return max(128, 2 * int(dim.x + dim.y + dim.z))

        # A sparsity pattern entry may cover more than one voxel of its level. Recover
        # the ratio from the coarsest pattern, which spans the whole domain.
        coarsest_span = int(patterns[num_levels - 1].shape[0]) * 2 ** (num_levels - 1)
        voxels_per_entry = max(1, int(round(int(dim.x) / max(1, coarsest_span))))

        total = 0
        for level in levels_to_traverse:
            occupied = np.nonzero(np.asarray(patterns[level]))

            if len(occupied) != 3 or occupied[0].size == 0:
                extent = sum(int(a) for a in np.asarray(patterns[level]).shape)
            else:
                extent = sum(int(axis.max() - axis.min() + 1) for axis in occupied)

            # Factor 2 leaves room for a path that is not a straight line.
            total += 2 * extent * voxels_per_entry

        return max(128, total)

    def _seal_enclosed_fluid(
        self,
        seal_options,
        grid,
        bc,
        mesh_id,
        bc_id,
        distances,
        bc_mask,
        missing_mask,
        solid_mask,
        normal_vector,
        normal_distance,
        stream,
    ):
        """Retag fluid voxels that are not reachable from outside the domain as solid."""
        import neon

        num_levels = grid.num_levels
        debug_vti = seal_options["seal_debug_vti"]
        memory_type = neon.MemoryType.host_device() if debug_vti else neon.MemoryType.device()

        device_sync = self._make_device_sync(grid)
        wall_clock = _SealTimer(device_sync)

        reach = grid.new_field(cardinality=1, dtype=wp.uint8, memory_type=memory_type)

        for level in range(num_levels):
            reach.fill_run(level=level, value=wp.uint8(0), stream_idx=stream)

        # Counting and convergence detection both go through this array. Neon kernels can
        # capture a warp array, which is the only reduction path available.
        counters = wp.zeros(num_levels * SEAL_COUNT_SLOTS, dtype=wp.int32)

        # Seed: the domain boundary is by definition outside, and the coarsest level is
        # the free stream around the body.
        for level in range(num_levels):
            seed_whole_level = seal_options["seal_seed_coarsest_level"] and num_levels > 1 and level == num_levels - 1
            container_seed = self.neon_container_dict["container_seed"](
                bc_mask,
                reach,
                int(grid.dim.x),
                int(grid.dim.y),
                int(grid.dim.z),
                int(2**level),
                bool(seed_whole_level),
                counters,
                level,
            )
            container_seed.run(stream, container_runtime=neon.Container.ContainerRuntime.neon)

        seed_seconds = wall_clock.lap()
        counts = self._read_seal_counters(counters, num_levels, device_sync)

        print("")
        print("Enclosed-fluid detection")
        print(f"  simulated voxels : {sum(counts[SEAL_COUNT_UNREFINED]):,}")
        print(f"  of which fluid   : {sum(counts[SEAL_COUNT_FLUID]):,}")
        print(f"  seeded as outside: {sum(counts[SEAL_COUNT_SEEDED]):,}  [{seed_seconds:.2f} s]")

        # Containers are built once and re-launched: building one compiles a kernel.
        fill_containers = [self.neon_container_dict["container_fill"](bc_mask, reach, counters, level) for level in range(num_levels)]

        print(f"  kernel build     : [{wall_clock.lap():.2f} s]")

        max_iterations = seal_options["seal_max_iterations"]
        if max_iterations is None:
            max_iterations = self._estimate_seal_iterations(grid, seal_options["seal_seed_coarsest_level"])

        # The fill is monotone, so a block of sweeps that flips nothing is the fixed
        # point. Checking in blocks keeps the sync/read overhead off the sweep loop.
        check_interval = max(1, min(64, max_iterations))
        sweeps = 0
        converged = False
        previous_reached = -1

        while sweeps < max_iterations and not converged:
            for _ in range(min(check_interval, max_iterations - sweeps)):
                for container_fill in fill_containers:
                    container_fill.run(stream, container_runtime=neon.Container.ContainerRuntime.neon)
                sweeps += 1

            counts = self._read_seal_counters(counters, num_levels, device_sync)
            reached = sum(counts[SEAL_COUNT_REACHED])
            converged = reached == previous_reached
            previous_reached = reached

            print(f"    sweep {sweeps:>6} of {max_iterations}: {reached:,} voxels reached  [{wall_clock.elapsed():.2f} s]")

        fill_seconds = wall_clock.lap()
        fill_state = "converged" if converged else "NOT CONVERGED"
        print(f"  flood fill       : {sweeps} sweeps, {fill_state}  [{fill_seconds:.2f} s, {fill_seconds / max(1, sweeps) * 1e3:.1f} ms/sweep]")

        if debug_vti:
            reach.update_host(stream)
            reach.export_vti(debug_vti, "reach")
            print(f"  reachability     : written to {debug_vti}")

        if not converged:
            # Sealing now would tag fluid that the fill simply has not walked to yet,
            # which would wall off open regions. Leaving the masks untouched is the
            # safe outcome.
            print(f"  WARNING: reachability did not converge within {max_iterations} sweeps, nothing sealed.")
            print("  WARNING: raise seal_max_iterations (or set seal_seed_coarsest_level=True) and re-run.")
            print("")
            self._reach_field = reach
            return

        for level in range(num_levels):
            container_seal = self.neon_container_dict["container_seal"](bc_mask, solid_mask, reach, counters, level)
            container_seal.run(stream, container_runtime=neon.Container.ContainerRuntime.neon)

        seal_seconds = wall_clock.lap()
        counts = self._read_seal_counters(counters, num_levels, device_sync)
        self._report_sealed_voxels(counts, num_levels, seal_seconds)

        # Sealing creates new solid voxels. Voxels that are only diagonally connected to
        # a sealed pocket are still fluid but now sit next to a solid voxel, so their
        # ownership, missing links and distances have to be recomputed.
        if seal_options["seal_refresh_masks"]:
            for level in range(num_levels):
                container_aabb = self.neon_container_dict["container_aabb"](
                    mesh_id,
                    bc_id,
                    distances,
                    bc_mask,
                    missing_mask,
                    solid_mask,
                    wp.static(bc.needs_mesh_distance),
                    normal_vector,
                    normal_distance,
                    level,
                )
                container_aabb.run(stream, container_runtime=neon.Container.ContainerRuntime.neon)

            print(f"  mask refresh     : [{wall_clock.lap():.2f} s]")

        print(f"  complete         : [{wall_clock.total():.2f} s total]")
        print("")

        # Keep the field alive until the queued work has been issued.
        self._reach_field = reach

    @staticmethod
    def _make_device_sync(grid):
        """Return a callable that waits for the queued Neon work to finish.

        This deliberately prefers Neon's backend sync over ``wp.synchronize()``.  Warp
        queues its device frees and module unloads and only flushes them inside a CUDA
        context synchronize, so calling ``wp.synchronize()`` here would tear down memory
        that the rest of the masker still uses -- including the mesh BVH behind
        ``mesh_id``, which nothing holds a reference to.  Neon's sync waits on the same
        device without running warp's deferred cleanup.
        """
        backend = getattr(grid, "backend", None)

        if backend is not None and hasattr(backend, "sync"):
            return backend.sync

        return wp.synchronize

    @staticmethod
    def _read_seal_counters(counters, num_levels, sync):
        """Per-slot lists of the enclosed-fluid counters, indexed ``[slot][level]``.

        Neon runs on its own streams, so the queued work has to complete before the
        counters are copied back.  The copy itself goes through warp's null stream, which
        is a plain async memcpy and does not trigger warp's deferred cleanup.
        """
        sync()
        values = counters.numpy()
        return [[int(values[level * SEAL_COUNT_SLOTS + slot]) for level in range(num_levels)] for slot in range(SEAL_COUNT_SLOTS)]

    def _report_sealed_voxels(self, counts, num_levels, seal_seconds):
        """Print what was retagged and what it buys.

        Sealed voxels become solid, which every LBM operator skips.  Sealing does not
        change how many steps the solver takes, it removes work from inside a global
        step, and only at the levels that lost voxels.  A level-``l`` voxel is updated
        ``2**(num_levels - 1 - l)`` times per global step under acoustic time refinement,
        so a finest-level voxel is worth far more than a coarse one.

        This is the same weighting the cases use for "Total lattice updates per global
        step", the quantity MLUPS is derived from, so the saving reported here is
        directly comparable to that figure.
        """
        sealed = counts[SEAL_COUNT_SEALED]
        fluid = counts[SEAL_COUNT_FLUID]

        total_sealed = sum(sealed)
        total_fluid = sum(fluid)

        def level_weight(level):
            return 2 ** (num_levels - 1 - level)

        def lattice_updates(per_level):
            return sum(per_level[level] * level_weight(level) for level in range(num_levels))

        fluid_updates = lattice_updates(fluid)
        sealed_updates = lattice_updates(sealed)

        voxel_share = 100.0 * total_sealed / total_fluid if total_fluid else 0.0
        update_share = 100.0 * sealed_updates / fluid_updates if fluid_updates else 0.0

        print(f"  enclosed voxels  : {total_sealed:,} retagged solid, {voxel_share:.2f}% of fluid voxels  [{seal_seconds:.2f} s]")

        if total_sealed:
            per_level = ", ".join(f"L{level}: {sealed[level]:,} (x{level_weight(level)})" for level in range(num_levels))
            print(f"  by level         : {per_level}")
            print(f"  lattice updates  : {sealed_updates:,} fewer per global step, {update_share:.2f}% of {fluid_updates:,}")
