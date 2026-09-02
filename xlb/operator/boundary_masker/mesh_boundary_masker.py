"""
Abstract base class for mesh-based boundary maskers.

Provides shared input preparation logic (mesh construction, kernel arrays)
used by AABB, Ray, Winding, and AABB-Close masker subclasses.
"""

import numpy as np
import warp as wp
from typing import Any
from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb.operator.boundary_masker.helper_functions_masker import HelperFunctionsMasker


class MeshBoundaryMasker(Operator):
    """
    Operator for creating a boundary missing_mask from a mesh file
    """

    def __init__(
        self,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
    ):
        # Call super
        super().__init__(velocity_set, precision_policy, compute_backend)

        assert self.compute_backend in [ComputeBackend.WARP, ComputeBackend.NEON], (
            f"MeshBoundaryMasker is only implemented for {ComputeBackend.WARP} and {ComputeBackend.NEON} backends!"
        )

        assert self.velocity_set.d == 3, "MeshBoundaryMasker is only implemented for 3D velocity sets!"
        # Raise error if used for 2d examples:
        if self.velocity_set.d == 2:
            raise NotImplementedError("This Operator is not implemented in 2D!")

        # Make constants for warp
        _c = self.velocity_set.c
        _q = self.velocity_set.q

        if self.compute_backend in [ComputeBackend.WARP, ComputeBackend.NEON]:
            # Define masker helper functions
            self.helper_masker = HelperFunctionsMasker(
                velocity_set=self.velocity_set,
                precision_policy=self.precision_policy,
                compute_backend=self.compute_backend,
            )

        @wp.func
        def out_of_bound_pull_index(
            lattice_dir: wp.int32,
            index: wp.vec3i,
            field: wp.array4d(dtype=wp.uint8),
            grid_shape: wp.vec3i,
        ):
            # Get the index of the streaming direction
            pull_index = wp.vec3i()
            for d in range(self.velocity_set.d):
                pull_index[d] = index[d] - _c[d, lattice_dir]

            # check if pull index is out of bound
            # These directions will have missing information after streaming
            missing = not self.helper_masker.is_in_bounds(pull_index, grid_shape)
            return missing

        @wp.func
        def _min3(a: wp.float32, b: wp.float32, c: wp.float32):
            return wp.min(a, wp.min(b, c))

        @wp.func
        def _max3(a: wp.float32, b: wp.float32, c: wp.float32):
            return wp.max(a, wp.max(b, c))

        @wp.func
        def _sat_axis_overlap(
            axis: wp.vec3f,
            v0: wp.vec3f,
            v1: wp.vec3f,
            v2: wp.vec3f,
            half: wp.vec3f,
            eps: wp.float32,
        ):
            """
            Separating-axis test for one candidate axis.

            This does not care about normal orientation. The axis sign is irrelevant
            because we test both min/max projection against +/- projected box radius.
            """
            axis_len2 = wp.dot(axis, axis)

            # Degenerate cross-product axis. It cannot separate.
            if axis_len2 <= wp.float32(1.0e-20):
                return True

            p0 = wp.dot(v0, axis)
            p1 = wp.dot(v1, axis)
            p2 = wp.dot(v2, axis)

            tri_min = _min3(p0, p1, p2)
            tri_max = _max3(p0, p1, p2)

            box_radius = (
                half[0] * wp.abs(axis[0])
                + half[1] * wp.abs(axis[1])
                + half[2] * wp.abs(axis[2])
            )

            if tri_min > box_radius + eps:
                return False

            if tri_max < -box_radius - eps:
                return False

            return True

        @wp.func
        def _triangle_overlaps_expanded_unit_cube(
            v0_local: wp.vec3f,
            v1_local: wp.vec3f,
            v2_local: wp.vec3f,
            surface_band_voxels: wp.float32,
        ):
            """
            Conservative triangle / expanded-voxel overlap test.

            The voxel is the unit cube [0, 1]^3 in local coordinates.
            We expand that cube by `surface_band_voxels` in all directions, so
            the tested box is:

                [-band, 1 + band]^3

            A triangle exactly on a shared voxel face will overlap both adjacent
            expanded voxel boxes.

            This is a SAT triangle-AABB test. It uses a triangle plane normal
            computed from geometry, but the sign/orientation of that normal is
            irrelevant.
            """
            eps = wp.float32(1.0e-6)

            center = wp.vec3f(0.5, 0.5, 0.5)
            half = wp.vec3f(
                0.5 + surface_band_voxels,
                0.5 + surface_band_voxels,
                0.5 + surface_band_voxels,
            )

            # Shift triangle into box-centered coordinates.
            v0 = v0_local - center
            v1 = v1_local - center
            v2 = v2_local - center

            # 1. Test overlap along box x/y/z axes.
            min_x = _min3(v0[0], v1[0], v2[0])
            max_x = _max3(v0[0], v1[0], v2[0])
            if min_x > half[0] + eps or max_x < -half[0] - eps:
                return False

            min_y = _min3(v0[1], v1[1], v2[1])
            max_y = _max3(v0[1], v1[1], v2[1])
            if min_y > half[1] + eps or max_y < -half[1] - eps:
                return False

            min_z = _min3(v0[2], v1[2], v2[2])
            max_z = _max3(v0[2], v1[2], v2[2])
            if min_z > half[2] + eps or max_z < -half[2] - eps:
                return False

            # Triangle edges.
            e0 = v1 - v0
            e1 = v2 - v1
            e2 = v0 - v2

            # 2. Test triangle plane axis.
            # Orientation does not matter because the plane test is symmetric.
            normal = wp.cross(e0, v2 - v0)
            normal_len2 = wp.dot(normal, normal)

            if normal_len2 <= wp.float32(1.0e-20):
                # Degenerate triangle. Conservatively ignore it rather than
                # letting zero-area geometry thicken the solid mask.
                return False

            plane_dist = wp.dot(normal, v0)
            plane_radius = (
                half[0] * wp.abs(normal[0])
                + half[1] * wp.abs(normal[1])
                + half[2] * wp.abs(normal[2])
            )

            if plane_dist > plane_radius + eps:
                return False

            if plane_dist < -plane_radius - eps:
                return False

            # 3. Test the 9 cross-product axes: edge x box-axis.
            # For edge e = (ex, ey, ez):
            #   e x X = (0, ez, -ey)
            #   e x Y = (-ez, 0, ex)
            #   e x Z = (ey, -ex, 0)

            axis = wp.vec3f(0.0, e0[2], -e0[1])
            if not _sat_axis_overlap(axis, v0, v1, v2, half, eps):
                return False

            axis = wp.vec3f(-e0[2], 0.0, e0[0])
            if not _sat_axis_overlap(axis, v0, v1, v2, half, eps):
                return False

            axis = wp.vec3f(e0[1], -e0[0], 0.0)
            if not _sat_axis_overlap(axis, v0, v1, v2, half, eps):
                return False

            axis = wp.vec3f(0.0, e1[2], -e1[1])
            if not _sat_axis_overlap(axis, v0, v1, v2, half, eps):
                return False

            axis = wp.vec3f(-e1[2], 0.0, e1[0])
            if not _sat_axis_overlap(axis, v0, v1, v2, half, eps):
                return False

            axis = wp.vec3f(e1[1], -e1[0], 0.0)
            if not _sat_axis_overlap(axis, v0, v1, v2, half, eps):
                return False

            axis = wp.vec3f(0.0, e2[2], -e2[1])
            if not _sat_axis_overlap(axis, v0, v1, v2, half, eps):
                return False

            axis = wp.vec3f(-e2[2], 0.0, e2[0])
            if not _sat_axis_overlap(axis, v0, v1, v2, half, eps):
                return False

            axis = wp.vec3f(e2[1], -e2[0], 0.0)
            if not _sat_axis_overlap(axis, v0, v1, v2, half, eps):
                return False

            return True

        # Check whether the unit voxel at position low intersects the warp mesh, assumes mesh has valid normals
        #  inputs: mesh_id: mesh id, low: position of the voxel
        #  outputs: True if intersection, False otherwise
        @wp.func
        def mesh_voxel_intersect(mesh_id: wp.uint64, low: wp.vec3):
            """
            Conservative surface-band voxelization.

            A voxel is solid if the triangle overlaps the voxel cube expanded by
            SURFACE_BAND_VOXELS.

            This intentionally does not ask which side of the surface is solid.
            Therefore, if a triangle lies on or very near a voxel face, both
            adjacent voxels will be tagged solid.
            """
            # In voxel units.
            #
            # Start with 1e-3. This means the surface gets a 0.001-voxel
            # conservative band. If your STL/grid alignment noise is larger,
            # increase to 1e-2. If you only want exact face ties, reduce to 1e-4.
            SURFACE_BAND_VOXELS = wp.float32(5.0e-3)

            band_vec = wp.vec3f(
                SURFACE_BAND_VOXELS,
                SURFACE_BAND_VOXELS,
                SURFACE_BAND_VOXELS,
            )

            high = low + wp.vec3f(1.0, 1.0, 1.0)

            # Broad phase: query the same expanded box used by the narrow phase.
            # This prevents triangles on the voxel face from being dropped before
            # the actual overlap test.
            query = wp.mesh_query_aabb(
                mesh_id,
                low - band_vec,
                high + band_vec,
            )

            for f in query:
                v0_global = wp.mesh_eval_position(mesh_id, f, 1.0, 0.0)
                v1_global = wp.mesh_eval_position(mesh_id, f, 0.0, 1.0)
                v2_global = wp.mesh_eval_position(mesh_id, f, 0.0, 0.0)

                # Shift triangle into local voxel coordinates.
                # The voxel cube is now [0, 1]^3.
                v0_local = v0_global - low
                v1_local = v1_global - low
                v2_local = v2_global - low

                if _triangle_overlaps_expanded_unit_cube(
                    v0_local,
                    v1_local,
                    v2_local,
                    SURFACE_BAND_VOXELS,
                ):
                    return True

            return False
        
        @wp.kernel
        def resolve_out_of_bound_kernel(
            id_number: wp.int32,
            bc_mask: wp.array4d(dtype=wp.uint8),
            missing_mask: wp.array4d(dtype=wp.uint8),
        ):
            # get index
            i, j, k = wp.tid()

            # Get local indices
            index = wp.vec3i(i, j, k)

            # domain shape to check for out of bounds
            grid_shape = wp.vec3i(bc_mask.shape[1], bc_mask.shape[2], bc_mask.shape[3])

            # Find the fractional distance to the mesh in each direction
            if bc_mask[0, index[0], index[1], index[2]] == wp.uint8(id_number):
                for l in range(1, _q):
                    # Ensuring out of bound pull indices are properly considered in the missing_mask
                    if out_of_bound_pull_index(l, index, missing_mask, grid_shape):
                        missing_mask[l, index[0], index[1], index[2]] = wp.uint8(True)

        # Construct some helper warp functions
        self.mesh_voxel_intersect = mesh_voxel_intersect
        self.resolve_out_of_bound_kernel = resolve_out_of_bound_kernel

    def _prepare_kernel_inputs(
        self,
        bc,
        bc_mask,
    ):
        assert bc.mesh_vertices is not None, f'Please provide the mesh vertices for {bc.__class__.__name__} BC using keyword "mesh_vertices"!'
        assert bc.indices is None, f"Please use IndicesBoundaryMasker operator if {bc.__class__.__name__} is imposed on known indices of the grid!"
        assert bc.mesh_vertices.shape[1] == self.velocity_set.d, (
            "Mesh points must be reshaped into an array (N, 3) where N indicates number of points!"
        )

        grid_shape = self.helper_masker.get_grid_shape(bc_mask)  # (nx, ny, nz)
        mesh_vertices = bc.mesh_vertices
        mesh_min = np.min(mesh_vertices, axis=0)
        mesh_max = np.max(mesh_vertices, axis=0)

        if any(mesh_min < 0) or any(mesh_max >= grid_shape):
            raise ValueError(
                f"Mesh extents ({mesh_min}, {mesh_max}) exceed domain dimensions {grid_shape}. The mesh must be fully contained within the domain."
            )

        # We are done with bc.mesh_vertices. Remove them from BC objects
        bc.__dict__.pop("mesh_vertices", None)

        mesh_indices = np.arange(mesh_vertices.shape[0])

        # Only mesh.id is handed to the kernels, so the wp.Mesh and the arrays it points
        # at have to outlive this call. Letting them go out of scope leaves mesh_id
        # dangling: it keeps working only because warp defers its frees until the next
        # context synchronize, so anything that synchronizes before the mesh queries run
        # (a timing probe, a counter read-back, an exporter) frees the BVH underneath it.
        self._mesh_points = wp.array(mesh_vertices, dtype=wp.vec3)
        self._mesh_indices = wp.array(mesh_indices, dtype=wp.int32)
        self._mesh = wp.Mesh(points=self._mesh_points, indices=self._mesh_indices)
        mesh_id = wp.uint64(self._mesh.id)
        bc_id = bc.id
        return mesh_id, bc_id

    @Operator.register_backend(ComputeBackend.JAX)
    def jax_implementation(
        self,
        bc,
        bc_mask,
        missing_mask,
    ):
        raise NotImplementedError(f"Operation {self.__class__.__name__} not implemented in JAX!")

    def warp_implementation_base(
        self,
        bc,
        distances,
        bc_mask,
        missing_mask,
    ):
        # Prepare inputs
        mesh_id, bc_id = self._prepare_kernel_inputs(bc, bc_mask)

        # Launch the appropriate warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[mesh_id, bc_id, distances, bc_mask, missing_mask, wp.static(bc.needs_mesh_distance)],
            dim=bc_mask.shape[1:],
        )
        wp.launch(
            self.resolve_out_of_bound_kernel,
            inputs=[bc_id, bc_mask, missing_mask],
            dim=bc_mask.shape[1:],
        )
        return distances, bc_mask, missing_mask
