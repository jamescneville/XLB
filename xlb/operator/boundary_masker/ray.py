import warp as wp
from typing import Any
from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.boundary_masker.mesh_boundary_masker import MeshBoundaryMasker
from xlb.operator.operator import Operator
import neon


class MeshMaskerRay(MeshBoundaryMasker):
    """
    Operator for creating a boundary missing_mask from an STL file
    """

    def __init__(
        self,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
    ):
        # Call super
        super().__init__(velocity_set, precision_policy, compute_backend)

    def _construct_warp(self):
        # Make constants for warp
        _c = self.velocity_set.c
        _q = self.velocity_set.q
        _opp_indices = self.velocity_set.opp_indices

        # Set local constants
        lattice_central_index = self.velocity_set.center_index

        @wp.func
        def functional(
            index: Any,
            mesh_id: Any,
            id_number: Any,
            distances: Any,
            bc_mask: Any,
            missing_mask: Any,
            needs_mesh_distance: Any,
            normal_vector: Any, 
            normal_distance: Any,
        ):
            # position of the point
            cell_center_pos = self.helper_masker.index_to_position(bc_mask, index)

            epsilon = wp.float32(1.0e-6)

            # Tolerances for ambiguity / center-cut detection
            near_zero_tol = epsilon
            small_weight_tol = wp.float32(0.001)

            # Local candidate storage
            cand_hit = wp.vec(length=_q, dtype=wp.uint8)
            cand_weight = wp.vec(length=_q, dtype=wp.float32)
            cand_dist = wp.vec(length=_q, dtype=wp.float32)
            cand_nx = wp.vec(length=_q, dtype=wp.float32)
            cand_ny = wp.vec(length=_q, dtype=wp.float32)
            cand_nz = wp.vec(length=_q, dtype=wp.float32)
            cand_mode = wp.vec(length=_q, dtype=wp.float32)

            for l in range(_q):
                cand_hit[l] = wp.uint8(0)
                cand_weight[l] = wp.float32(0.0)
                cand_dist[l] = wp.float32(0.0)
                cand_nx[l] = wp.float32(0.0)
                cand_ny[l] = wp.float32(0.0)
                cand_nz[l] = wp.float32(0.0)
                cand_mode[l] = wp.float32(5.0)

            # ------------------------------------------------------------
            # Pass 1: gather ray evidence in every direction
            # ------------------------------------------------------------
            for direction_idx in range(_q):
                if direction_idx == lattice_central_index:
                    continue

                direction_vec = wp.vec3f( wp.float32(_c[0, direction_idx]),  wp.float32(_c[1, direction_idx]),wp.float32(_c[2, direction_idx]),)
                max_length = wp.length(direction_vec)
                dir_norm = direction_vec / max_length
                
                # Origin based query
                query0 = wp.mesh_query_ray(mesh_id, cell_center_pos, dir_norm, max_length)
                dist = wp.float32(0.0)
                count = wp.float32(0.0)
                min_dist = wp.float32(1.0)
                mode = wp.float32(5.0)
                normal = wp.vec3f(0.0, 0.0, 0.0)

                if query0.result:
                    dist = wp.float32(query0.t)
                    count += 1.0
                    normal = query0.normal
                    sign = wp.dot(dir_norm, normal)
                    if sign > 0.0:
                        normal = -normal
                    min_dist = wp.min(min_dist, dist)
                    mode = wp.float32(0.0)

                else:
                    # forward-shifted origin
                    query1 = wp.mesh_query_ray( mesh_id, cell_center_pos + epsilon * dir_norm, dir_norm, max_length - epsilon,)
                    if query1.result:
                        dist = wp.float32(query1.t + epsilon)
                        count += 1.0
                        normal = query1.normal
                        sign = wp.dot(dir_norm, normal)
                        if sign > 0.0:
                            normal = -normal
                        min_dist = wp.min(min_dist, dist)
                        mode = wp.float32(1.0)

                    else:
                        # backward-shifted origin
                        query2 = wp.mesh_query_ray( mesh_id, cell_center_pos - epsilon * dir_norm,  dir_norm,  max_length + epsilon, )
                        if query2.result:
                            dist2 = wp.float32(query2.t - epsilon)
                            if dist2 < 0.0:
                                continue
                            else:
                                dist = dist2
                                normal = query2.normal
                                sign = wp.dot(dir_norm, normal)
                                if sign > 0.0:
                                    normal = -normal
                                min_dist = wp.min(min_dist, dist)
                                mode = wp.float32(2.0)

                            count += 1.0

                if count > 0.0:
                    weight = dist / max_length

                    cand_hit[direction_idx] = wp.uint8(1)
                    cand_weight[direction_idx] = weight
                    cand_dist[direction_idx] = dist
                    cand_nx[direction_idx] = normal[0]
                    cand_ny[direction_idx] = normal[1]
                    cand_nz[direction_idx] = normal[2]
                    cand_mode[direction_idx] = mode

            # ------------------------------------------------------------
            # Pass 2: classify the voxel
            #
            # Rules:
            # - no reliable hits -> leave fluid
            # - one-sided evidence -> BC voxel
            # - dual-sided / center-cut ambiguity -> solid 255
            # ------------------------------------------------------------
            # ------------------------------------------------------------
            # Pass 2: classify the voxel
            # ------------------------------------------------------------
            any_hit = wp.bool(False)

            accepted_hit = wp.vec(length=_q, dtype=wp.uint8)
            for l in range(_q):
                accepted_hit[l] = wp.uint8(0)

            center_cut_pairs = wp.int32(0)
            usable_pairs = wp.int32(0)

            solid_center_pairs_min = wp.int32(2)

            for direction_idx in range(_q):
                if direction_idx == lattice_central_index:
                    continue

                opp_idx = _opp_indices[direction_idx]

                # process each opposite pair only once
                if direction_idx > opp_idx:
                    continue

                hit_a = wp.bool(cand_hit[direction_idx] != 0)
                hit_b = wp.bool(cand_hit[opp_idx] != 0)

                if (not hit_a) and (not hit_b):
                    continue

                any_hit = wp.bool(True)

                # clear one-sided evidence
                if hit_a and (not hit_b):
                    accepted_hit[direction_idx] = wp.uint8(1)
                    usable_pairs += 1
                    continue

                if hit_b and (not hit_a):
                    accepted_hit[opp_idx] = wp.uint8(1)
                    usable_pairs += 1
                    continue

                # both sides hit
                w_a = cand_weight[direction_idx]
                w_b = cand_weight[opp_idx]

                a_small = wp.bool(w_a <= small_weight_tol)
                b_small = wp.bool(w_b <= small_weight_tol)

                # true center-cut / voxel-centered pair
                if a_small and b_small:
                    center_cut_pairs += 1
                    continue

                # appreciable dual hit:
                # keep BOTH directions so the BC helper sees a sandwich pair
                accepted_hit[direction_idx] = wp.uint8(1)
                accepted_hit[opp_idx] = wp.uint8(1)
                usable_pairs += 1

            # ------------------------------------------------------------
            # Pass 3: final classification
            # ------------------------------------------------------------

            # Case 1: true center-cut voxel -> mark solid
            if center_cut_pairs >= solid_center_pairs_min:
                self.write_field(bc_mask, index, 0, wp.uint8(255))

                for l in range(_q):
                    self.write_field(missing_mask, index, l, wp.uint8(False))
                    if needs_mesh_distance:
                        self.write_field(distances, index, l, self.store_dtype(0.0))

                self.write_field(normal_distance, index, 0, self.store_dtype(0.0))
                self.write_field(normal_vector, index, 0, wp.float32(0.0))
                self.write_field(normal_vector, index, 1, wp.float32(0.0))
                self.write_field(normal_vector, index, 2, wp.float32(0.0))
                return

            # Case 2: no usable evidence -> leave fluid
            if (not any_hit) or (usable_pairs == 0):
                return

            # Case 3: BC voxel (one-sided and/or sandwich)
            self.write_field(bc_mask, index, 0, wp.uint8(id_number))

            for l in range(_q):
                self.write_field(missing_mask, index, l, wp.uint8(False))
                if needs_mesh_distance:
                    self.write_field(distances, index, l, self.store_dtype(0.0))

            total_wall_dist = wp.float32(0.0)
            total_normal = wp.vec3f(0.0, 0.0, 0.0)
            hit_count = wp.float32(0.0)

            for direction_idx in range(_q):
                if direction_idx == lattice_central_index:
                    continue

                if accepted_hit[direction_idx] != 0:
                    opp_idx = _opp_indices[direction_idx]

                    # hit in direction_idx => missing incoming population is opp_idx
                    self.write_field(missing_mask, index, opp_idx, wp.uint8(True))

                    if needs_mesh_distance:
                        self.write_field(
                            distances,
                            index,
                            direction_idx,
                            self.store_dtype(cand_weight[direction_idx]),
                        )

                    total_wall_dist += cand_dist[direction_idx]
                    total_normal += wp.vec3f(
                        cand_nx[direction_idx],
                        cand_ny[direction_idx],
                        cand_nz[direction_idx],
                    )
                    hit_count += 1.0

            # Safety guard
            if hit_count <= 0.0:
                # Better to leave as fluid than keep an invalid BC voxel
                self.write_field(bc_mask, index, 0, wp.uint8(0))
                return

            avg_wall_dist = total_wall_dist / hit_count
            avg_wall_dist = wp.max(avg_wall_dist, wp.float32(0.01))
            self.write_field(normal_distance, index, 0, self.store_dtype(avg_wall_dist))

            avg_normal_len = wp.length(total_normal)
            if avg_normal_len > 0.0:
                avg_normal = total_normal / avg_normal_len
            else:
                avg_normal = total_normal

            self.write_field(normal_vector, index, 0, avg_normal[0])
            self.write_field(normal_vector, index, 1, avg_normal[1])
            self.write_field(normal_vector, index, 2, avg_normal[2])
                

        @wp.kernel
        def kernel(
            mesh_id: wp.uint64,
            id_number: wp.int32,
            distances: wp.array4d(dtype=Any),
            bc_mask: wp.array4d(dtype=wp.uint8),
            missing_mask: wp.array4d(dtype=wp.uint8),
            needs_mesh_distance: bool,
            normal_vector: wp.array4d(dtype=Any),
            normal_distance: wp.array4d(dtype=Any),
        ):
            # get index
            i, j, k = wp.tid()

            # Get local indices
            index = wp.vec3i(i, j, k)

            # apply the functional
            functional(
                index,
                mesh_id,
                id_number,
                distances,
                bc_mask,
                missing_mask,
                needs_mesh_distance,                
                normal_vector,
                normal_distance,
            )

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(
        self,
        bc,
        distances,
        bc_mask,
        missing_mask,
        normal_vector,
        normal_distance,
    ):
        return self.warp_implementation_base(
            bc,
            distances,
            bc_mask,
            missing_mask,
            normal_vector,
            normal_distance,
        )

    def _construct_neon(self):
        # Use the warp functional for the NEON backend
        functional, _ = self._construct_warp()

        @neon.Container.factory(name="MeshMaskerRay")
        def container(
            mesh_id: Any,
            id_number: Any,
            distances: Any,
            bc_mask: Any,
            missing_mask: Any,
            needs_mesh_distance: Any,
            normal_vector: Any,
            normal_distance: Any,
        ):
            def ray_launcher(loader: neon.Loader):
                loader.set_grid(bc_mask.get_grid())
                bc_mask_pn = loader.get_write_handle(bc_mask)
                missing_mask_pn = loader.get_write_handle(missing_mask)
                distances_pn = loader.get_write_handle(distances)
                norm_vec_pn = loader.get_write_handle(normal_vector)
                norm_dist_pn = loader.get_write_handle(normal_distance)

                @wp.func
                def ray_kernel(index: Any):
                    # apply the functional
                    functional(
                        index,
                        mesh_id,
                        id_number,
                        distances_pn,
                        bc_mask_pn,
                        missing_mask_pn,
                        needs_mesh_distance,
                        norm_vec_pn,
                        norm_dist_pn
                    )

                loader.declare_kernel(ray_kernel)

            return ray_launcher

        return functional, container

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_implementation(
        self,
        bc,
        distances,
        bc_mask,
        missing_mask,
        normal_vector, 
        normal_distance,
    ):
        # Prepare inputs
        mesh_id, bc_id = self._prepare_kernel_inputs(bc, bc_mask)

        # Launch the appropriate neon container
        c = self.neon_container(mesh_id, bc_id, distances, bc_mask, missing_mask, wp.static(bc.needs_mesh_distance), normal_vector, normal_distance)
        c.run(0, container_runtime=neon.Container.ContainerRuntime.neon)
        return distances, bc_mask, missing_mask, normal_vector, normal_distance
