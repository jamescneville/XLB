import warp as wp
from typing import Any
from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.boundary_masker.mesh_boundary_masker import MeshBoundaryMasker
from xlb.operator.operator import Operator
import neon


class MeshMaskerSolid(Operator):
    """
    Operator looping to find trapped BC voxels
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
        ):
            bc_val = wp.neon_read(bc_mask, index, 0)
            if bc_val == wp.uint8(255) or bc_val == wp.uint8(254) or bc_val == wp.uint8(0):
                return
            
           
            # Find BC adjacent voxels
            for direction_idx in range(_q):
                if direction_idx == lattice_central_index:
                    # Skip the central index as it is not relevant for boundary masking
                    continue

               
                # If neighbor index is valid at this resolution level
                ngh = wp.neon_ngh_idx(wp.int8(_c[0, direction_idx]), wp.int8(_c[1, direction_idx]), wp.int8(_c[2, direction_idx]))
                is_valid = wp.bool(False)
                nval = wp.neon_read_ngh(bc_mask, index, ngh, 0, wp.uint8(0), is_valid)
                if is_valid:
                    if wp.neon_read(missing_mask, index, _opp_indices[direction_idx]) == wp.uint8(True):
                        continue

                    else: 
                        # If not Fluid or the same BC in this direction
                        if (nval != wp.uint8(0)) and (nval != wp.uint8(254)) and (nval != bc_val):
                            # Another BC exists in this direction mark missing to be sandwiched case
                            self.write_field(missing_mask, index, _opp_indices[direction_idx], wp.uint8(True))
                        
                

                

        @wp.kernel
        def kernel(
            mesh_id: wp.uint64,
            id_number: wp.int32,
            distances: wp.array4d(dtype=Any),
            bc_mask: wp.array4d(dtype=wp.uint8),
            missing_mask: wp.array4d(dtype=wp.uint8),
            needs_mesh_distance: bool,
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
            )

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(
        self,
        bc,
        distances,
        bc_mask,
        missing_mask,
    ):
        return self.warp_implementation_base(
            bc,
            distances,
            bc_mask,
            missing_mask,
        )

    def _construct_neon(self):
        # Use the warp functional for the NEON backend
        functional, _ = self._construct_warp()

        @neon.Container.factory(name="MeshMaskerSolid")
        def container(
            mesh_id: Any,
            id_number: Any,
            distances: Any,
            bc_mask: Any,
            missing_mask: Any,
            needs_mesh_distance: Any,
        ):
            def solid_launcher(loader: neon.Loader):
                loader.set_grid(bc_mask.get_grid())
                bc_mask_pn = loader.get_write_handle(bc_mask)
                missing_mask_pn = loader.get_write_handle(missing_mask)
                distances_pn = loader.get_write_handle(distances)

                @wp.func
                def solid_kernel(index: Any):
                    # apply the functional
                    functional(
                        index,
                        mesh_id,
                        id_number,
                        distances_pn,
                        bc_mask_pn,
                        missing_mask_pn,
                        needs_mesh_distance,
                    )

                loader.declare_kernel(solid_kernel)

            return solid_launcher

        return functional, container

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_implementation(
        self,
        bc,
        distances,
        bc_mask,
        missing_mask,
    ):
        mesh_id = wp.uint64(0)
        bc_id = wp.uint8(255)
        # Launch the appropriate neon container
        c = self.neon_container(mesh_id, bc_id, distances, bc_mask, missing_mask, wp.static(bc.needs_mesh_distance))
        c.run(0, container_runtime=neon.Container.ContainerRuntime.neon)
        return distances, bc_mask, missing_mask
