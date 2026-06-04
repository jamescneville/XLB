import warp as wp
from typing import Any
from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.boundary_masker import MeshMaskerAABBClose
from xlb.operator.operator import Operator
import neon


class MultiresMeshMaskerAABBCloseSolid(MeshMaskerAABBClose):
    """
    Operator for creating boundary missing_mask from mesh using Axis-Aligned Bounding Box (AABB) voxelization
    in multiresolution simulations (NEON backend). It takes in a number of close_voxels to perform morphological
    operations (dilate followed by erode) to ensure small channels are filled with solid voxels.

    This version provides NEON-specific functionals working on multires partitions (mPartition) and bIndex.
    """

    def __init__(
        self,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
        close_voxels: int = None,
    ):
        super().__init__(velocity_set, precision_policy, compute_backend, close_voxels)
        if self.compute_backend in [ComputeBackend.JAX, ComputeBackend.WARP]:
            raise NotImplementedError(f"Operator {self.__class__.__name__} not supported in {self.compute_backend} backend.")

        # Build and store NEON dicts
        self.neon_functional_dict, self.neon_container_dict = self._construct_neon()

    def _construct_neon(self):
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

        @wp.func
        def is_reachability_axis_direction(direction_idx: wp.int32):
            cx = wp.int32(_c[0, direction_idx])
            cy = wp.int32(_c[1, direction_idx])
            cz = wp.int32(_c[2, direction_idx])

            # Reachability is a geometric trapped-volume classifier, not the full
            # D3Q27 solver communication graph.  Only face-connected neighbors
            # should make a region reachable from the exterior; edge/corner contact
            # is too permissive and can leak through voxelized cavities.
            return (cx * cx + cy * cy + cz * cz) == wp.int32(1)

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
            if solid_val == wp.uint8(255) or bc_val == wp.uint8(255):
                wp.neon_write(bc_mask_pn, index, 0, wp.uint8(255))
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

                ngh = wp.neon_ngh_idx(
                    wp.int8(_c[0, direction_idx]),
                    wp.int8(_c[1, direction_idx]),
                    wp.int8(_c[2, direction_idx]),
                )

                is_valid = wp.bool(False)
                nval = wp.neon_read_ngh(
                    solid_mask_pn,
                    index,
                    ngh,
                    0,
                    wp.uint8(0),
                    is_valid,
                )

                if not is_valid:
                    continue

                if nval != wp.uint8(255):
                    continue

                # If no mesh distance is requested, keep the base ownership behavior:
                # mark the cell as boundary and flag the missing opposite link.
                if not needs_mesh_distance:
                    if not reset_done:
                        for l in range(_q):
                            self.write_field(
                                missing_mask_pn,
                                index,
                                _opp_indices[l],
                                wp.uint8(False),
                            )
                        reset_done = wp.bool(True)

                    self.write_field(bc_mask_pn, index, 0, wp.uint8(id_number))
                    self.write_field(
                        missing_mask_pn,
                        index,
                        _opp_indices[direction_idx],
                        wp.uint8(True),
                    )
                    continue

                dir_vec = wp.vec3f(
                    wp.float32(_c[0, direction_idx]),
                    wp.float32(_c[1, direction_idx]),
                    wp.float32(_c[2, direction_idx]),
                )

                max_length = wp.length(dir_vec)
                safe_length = max_length if max_length > 0.0 else wp.float32(1.0)
                norm_dir = dir_vec / safe_length

                # Keep the 1.5-link search distance so the second-layer case can still hit.
                query = wp.mesh_query_ray(
                    mesh_id,
                    cell_center,
                    norm_dir,
                    wp.float32(2.5) * safe_length,
                )

                # Ownership, missing-mask reset, distances, and normal accumulation only
                # happen after a valid mesh hit. This avoids claiming ownership from
                # solid-mask adjacency when the mesh ray did not actually resolve.                

                if not reset_done:
                    for l in range(_q):
                        self.write_field(
                            missing_mask_pn,
                            index,
                            _opp_indices[l],
                            wp.uint8(False),
                        )
                    reset_done = wp.bool(True)
                
                

                self.write_field(bc_mask_pn, index, 0, wp.uint8(id_number))
                self.write_field(
                    missing_mask_pn,
                    index,
                    _opp_indices[direction_idx],
                    wp.uint8(True),
                )

                
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
                link_distance = ray_dist / safe_length  
                link_distance -= 0.5              
                link_distance = wp.clamp(link_distance, wp.float32(0.0), wp.float32(1.0))

                if wp.isnan(link_distance) or wp.isinf(link_distance):
                    link_distance = wp.float32(1.0)

                self.write_field(
                    distances_pn,
                    index,
                    direction_idx,
                    self.store_dtype(link_distance),
                )

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

            if total_weight < 0.0:
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

            self.write_field(
                normal_distance,
                index,
                0,
                self.store_dtype(avg_wall_dist),
            )

            self.write_field(normal_vector, index, 0, avg_normal[0])
            self.write_field(normal_vector, index, 1, avg_normal[1])
            self.write_field(normal_vector, index, 2, avg_normal[2])
                        

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



        # Reachability initialization: mark cells reachable if they can see the exterior
        # of the multires domain without crossing a solid/intersection voxel.  This pass
        # intentionally ignores bc ids; only solid_mask == 255 is a blocker.
        @neon.Container.factory(name="ReachSeedExterior")
        def container_reach_seed_exterior(solid_mask: Any, reachable: Any, level: int):
            def reach_seed_launcher(loader: neon.Loader):
                loader.set_mres_grid(solid_mask.get_grid(), level)
                solid_mask_pn = loader.get_mres_read_handle(solid_mask)
                reachable_pn = loader.get_mres_write_handle(reachable)

                @wp.func
                def reach_seed_kernel(index: Any):
                    if wp.neon_has_child(solid_mask_pn, index):
                        return

                    if wp.neon_read(solid_mask_pn, index, 0) == wp.uint8(255):
                        wp.neon_write(reachable_pn, index, 0, wp.uint8(0))
                        return

                    touches_exterior = wp.bool(False)

                    for direction_idx in range(_q):
                        if direction_idx == lattice_central_index:
                            continue

                        if not is_reachability_axis_direction(wp.int32(direction_idx)):
                            continue

                        ngh = wp.neon_ngh_idx(
                            wp.int8(_c[0, direction_idx]),
                            wp.int8(_c[1, direction_idx]),
                            wp.int8(_c[2, direction_idx]),
                        )

                        is_valid = wp.bool(False)
                        _ = wp.neon_read_ngh(
                            solid_mask_pn,
                            index,
                            ngh,
                            0,
                            wp.uint8(0),
                            is_valid,
                        )

                        # Invalid same-level neighbors can mean either a true domain
                        # exterior, or a multires transition.  The transition cases are
                        # not exterior seeds: fine cells generally have a parent; coarse
                        # cells can have a finer neighbor in this direction.
                        if not is_valid:
                            has_finer = wp.neon_has_finer_ngh(solid_mask_pn, index, ngh)
                            has_parent = wp.neon_has_parent(solid_mask_pn, index)
                            if (not has_finer) and (not has_parent):
                                touches_exterior = wp.bool(True)

                    if touches_exterior:
                        wp.neon_write(reachable_pn, index, 0, wp.uint8(1))
                    else:
                        wp.neon_write(reachable_pn, index, 0, wp.uint8(0))

                loader.declare_kernel(reach_seed_kernel)

            return reach_seed_launcher

        # Scatter scalar reachability into a q-cardinality multires communication
        # field.  This mirrors the stepper's fine-to-coarse store path: reachable
        # fine leaf cells push a value toward coarser neighbors so coarse leaves can
        # see adjacent reachable finer cells during the next pull-style propagation
        # pass.  This field is transient and is rebuilt every reachability iteration.
        @neon.Container.factory(name="ReachScatterToCoarser")
        def container_reach_scatter_to_coarser(reachable: Any, reachable_mres_links: Any, level: int):
            def reach_scatter_launcher(loader: neon.Loader):
                loader.set_mres_grid(reachable.get_grid(), level)
                reachable_pn = loader.get_mres_read_handle(reachable)
                num_levels = reachable.get_grid().num_levels

                # This mirrors the stepper: only levels that have a coarser level above
                # them request a stencil_up write handle and use neon_mres_lbm_store_op.
                # Requesting stencil_up on the coarsest level is avoided.
                if level + 1 < num_levels:
                    reachable_links_pn = loader.get_mres_write_handle(
                        reachable_mres_links,
                        neon.Loader.Operation.stencil_up,
                    )

                    @wp.func
                    def reach_scatter_kernel(index: Any):
                        if wp.neon_has_child(reachable_pn, index):
                            return

                        if wp.neon_read(reachable_pn, index, 0) == wp.uint8(0):
                            return

                        # Push to coarser-level halo/neighbor storage.  This is intentionally
                        # a q-cardinality float32 field, because neon_mres_lbm_store_op uses
                        # atomicAdd internally and float32 is the path exercised by the LBM
                        # stepper.  The field is only a boolean accumulator: any value > 0
                        # means at least one touching finer leaf is reachable.
                        for direction_idx in range(_q):
                            if direction_idx == lattice_central_index:
                                continue

                            if not is_reachability_axis_direction(wp.int32(direction_idx)):
                                continue

                            push_direction = wp.neon_ngh_idx(
                                wp.int8(_c[0, direction_idx]),
                                wp.int8(_c[1, direction_idx]),
                                wp.int8(_c[2, direction_idx]),
                            )
                            wp.neon_mres_lbm_store_op(
                                reachable_links_pn,
                                index,
                                direction_idx,
                                push_direction,
                                wp.float32(1.0),
                            )

                    loader.declare_kernel(reach_scatter_kernel)
                else:
                    @wp.func
                    def reach_scatter_kernel(index: Any):
                        return

                    loader.declare_kernel(reach_scatter_kernel)

            return reach_scatter_launcher

        # Propagate reachability by one fixed-point iteration.  This uses the same
        # D3Q stencil as the solver and treats every non-solid cell as passable,
        # including cells that will later become bc_id boundary cells.
        @neon.Container.factory(name="ReachPropagate")
        def container_reach_propagate(
            solid_mask: Any,
            reachable: Any,
            reachable_next: Any,
            reachable_mres_links: Any,
            changed: Any,
            level: int,
        ):
            def reach_propagate_launcher(loader: neon.Loader):
                loader.set_mres_grid(solid_mask.get_grid(), level)
                solid_mask_pn = loader.get_mres_read_handle(solid_mask)
                reachable_pn = loader.get_mres_read_handle(reachable)
                reachable_next_pn = loader.get_mres_write_handle(reachable_next)
                reachable_links_pn = loader.get_mres_read_handle(reachable_mres_links)

                @wp.func
                def reach_propagate_kernel(index: Any):
                    if wp.neon_has_child(solid_mask_pn, index):
                        return

                    if wp.neon_read(solid_mask_pn, index, 0) == wp.uint8(255):
                        wp.neon_write(reachable_next_pn, index, 0, wp.uint8(0))
                        return

                    already_reachable = wp.neon_read(reachable_pn, index, 0)
                    if already_reachable != wp.uint8(0):
                        wp.neon_write(reachable_next_pn, index, 0, wp.uint8(1))
                        return

                    found_reachable = wp.bool(False)

                    for direction_idx in range(_q):
                        if direction_idx == lattice_central_index:
                            continue

                        if not is_reachability_axis_direction(wp.int32(direction_idx)):
                            continue

                        ngh = wp.neon_ngh_idx(
                            wp.int8(_c[0, direction_idx]),
                            wp.int8(_c[1, direction_idx]),
                            wp.int8(_c[2, direction_idx]),
                        )

                        solid_valid = wp.bool(False)
                        ngh_solid = wp.neon_read_ngh(
                            solid_mask_pn,
                            index,
                            ngh,
                            0,
                            wp.uint8(255),
                            solid_valid,
                        )

                        reach_valid = wp.bool(False)
                        ngh_reachable = wp.neon_read_ngh(
                            reachable_pn,
                            index,
                            ngh,
                            0,
                            wp.uint8(0),
                            reach_valid,
                        )

                        if solid_valid and reach_valid:
                            if ngh_solid != wp.uint8(255) and ngh_reachable != wp.uint8(0):
                                found_reachable = wp.bool(True)

                        # Fine-cell candidate -> coarser reachable neighbor.  This mirrors
                        # the stepper's explosion/read-from-coarser path.
                        if (not found_reachable) and (not reach_valid):
                            if wp.neon_has_parent(reachable_pn, index):
                                has_coarser = wp.bool(False)
                                coarse_reachable = wp.neon_lbm_read_coarser_ngh(
                                    reachable_pn,
                                    index,
                                    ngh,
                                    0,
                                    wp.uint8(0),
                                    has_coarser,
                                )
                                if has_coarser and coarse_reachable != wp.uint8(0):
                                    found_reachable = wp.bool(True)

                        # Coarse-cell candidate -> finer reachable neighbor.  This is the
                        # case that prevents a coarse leaf surrounded by reachable fine
                        # cells from being incorrectly classified as trapped.  Reachable
                        # fine leaves push into reachable_mres_links with the same
                        # stencil_up mechanism used by the stepper, and the coarse cell
                        # pulls the opposite population from the neighbor direction.
                        if (not found_reachable) and wp.neon_has_finer_ngh(reachable_links_pn, index, ngh):
                            fine_valid = wp.bool(False)
                            fine_reachable = wp.neon_read_ngh(
                                reachable_links_pn,
                                index,
                                ngh,
                                _opp_indices[direction_idx],
                                wp.float32(0.0),
                                fine_valid,
                            )
                            if fine_valid and fine_reachable > wp.float32(0.0):
                                found_reachable = wp.bool(True)

                    if found_reachable:
                        wp.neon_write(reachable_next_pn, index, 0, wp.uint8(1))
                        changed[0] = wp.int32(1)
                    else:
                        wp.neon_write(reachable_next_pn, index, 0, wp.uint8(0))

                loader.declare_kernel(reach_propagate_kernel)

            return reach_propagate_launcher

        # Convert every non-solid leaf cell that was not reached from the outside into
        # solid/intersection.  AABB boundary assignment runs after this, so the newly
        # trapped solid region creates normal bc_id cells around itself.
        @neon.Container.factory(name="TagUnreachableAsSolid")
        def container_tag_unreachable_as_solid(solid_mask: Any, reachable: Any, level: int):
            def tag_launcher(loader: neon.Loader):
                loader.set_mres_grid(solid_mask.get_grid(), level)
                solid_mask_pn = loader.get_mres_write_handle(solid_mask)
                reachable_pn = loader.get_mres_read_handle(reachable)

                @wp.func
                def tag_kernel(index: Any):
                    if wp.neon_has_child(solid_mask_pn, index):
                        return

                    if wp.neon_read(solid_mask_pn, index, 0) == wp.uint8(255):
                        return

                    if wp.neon_read(reachable_pn, index, 0) == wp.uint8(0):
                        wp.neon_write(solid_mask_pn, index, 0, wp.uint8(255))

                loader.declare_kernel(tag_kernel)

            return tag_launcher



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

        container_dict = {
            "container_erode": container_erode,
            "container_dilate": container_dilate,
            "container_solid": container_solid,
            "container_reach_seed_exterior": container_reach_seed_exterior,
            "container_reach_scatter_to_coarser": container_reach_scatter_to_coarser,
            "container_reach_propagate": container_reach_propagate,
            "container_tag_unreachable_as_solid": container_tag_unreachable_as_solid,
            "container_aabb": container,
        }

        # Expose NEON functionals too (in case callers want to reuse)
        functional_dict = {
            "mres_functional_aabb": mres_functional_aabb,
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

        # Phase 1: build the closed mesh-intersection/solid mask on every level.
        # AABB boundary ownership is intentionally delayed until after trapped-region
        # detection, so newly trapped cells become part of the solid mask before bc_id
        # cells and missing links are generated.
        for level in range(grid.num_levels):
            solid_mask.fill_run(level=level, value=wp.uint8(0), stream_idx=stream)
            solid_mask_out.fill_run(level=level, value=wp.uint8(0), stream_idx=stream)

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

        # Phase 2: outside reachability on the multires graph.  Only solid_mask == 255
        # blocks traversal; bc_id cells are fluid-side cells and are deliberately allowed.
        reachable = grid.new_field(cardinality=1, dtype=wp.uint8, memory_type=neon.MemoryType.device())
        reachable_next = grid.new_field(cardinality=1, dtype=wp.uint8, memory_type=neon.MemoryType.device())
        reachable_mres_links = grid.new_field(cardinality=self.velocity_set.q, dtype=wp.float32, memory_type=neon.MemoryType.device())

        for level in range(grid.num_levels):
            reachable.fill_run(level=level, value=wp.uint8(0), stream_idx=stream)
            reachable_next.fill_run(level=level, value=wp.uint8(0), stream_idx=stream)
            reachable_mres_links.fill_run(level=level, value=wp.float32(0.0), stream_idx=stream)
            container_seed = self.neon_container_dict["container_reach_seed_exterior"](solid_mask, reachable, level)
            container_seed.run(0, container_runtime=neon.Container.ContainerRuntime.neon)

        # Fixed-point flood fill.  The upper bound prevents an infinite loop if a device
        # changed flag cannot be observed because of a backend/runtime issue.  In normal
        # operation, the loop exits as soon as no new cells are marked reachable.
        max_reachability_iterations = 4096
        changed = wp.zeros(1, dtype=wp.int32)

        for _ in range(max_reachability_iterations):
            changed.zero_()

            for level in range(grid.num_levels):
                reachable_next.fill_run(level=level, value=wp.uint8(0), stream_idx=stream)
                reachable_mres_links.fill_run(level=level, value=wp.float32(0.0), stream_idx=stream)

            # Build transient fine-to-coarse reachability links from the current
            # reachable set before the pull-style propagation pass.
            for level in range(grid.num_levels):
                container_scatter = self.neon_container_dict["container_reach_scatter_to_coarser"](
                    reachable,
                    reachable_mres_links,
                    level,
                )
                container_scatter.run(0, container_runtime=neon.Container.ContainerRuntime.neon)

            for level in range(grid.num_levels):
                container_propagate = self.neon_container_dict["container_reach_propagate"](
                    solid_mask,
                    reachable,
                    reachable_next,
                    reachable_mres_links,
                    changed,
                    level,
                )
                container_propagate.run(0, container_runtime=neon.Container.ContainerRuntime.neon)

            reachable, reachable_next = reachable_next, reachable

            if int(changed.numpy()[0]) == 0:
                break

        for level in range(grid.num_levels):
            container_tag = self.neon_container_dict["container_tag_unreachable_as_solid"](solid_mask, reachable, level)
            container_tag.run(0, container_runtime=neon.Container.ContainerRuntime.neon)

        # Phase 3: run the original AABB boundary generation using the augmented solid
        # mask.  This preserves previous behavior and creates bc_id cells around both
        # mesh intersections and newly trapped solid regions.
        for level in range(grid.num_levels):
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
            container_aabb.run(0, container_runtime=neon.Container.ContainerRuntime.neon)

        return distances, bc_mask, missing_mask, normal_vector, normal_distance
