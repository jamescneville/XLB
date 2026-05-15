import warp as wp
from typing import Any
from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.boundary_masker import MeshMaskerAABBFill
from xlb.operator.operator import Operator
import neon


class MultiresMeshMaskerAABBFill(MeshMaskerAABBFill):
    """
    Multiresolution AABB 'fill' boundary masker.

    Per level:
      1) Build padded solid mask (Warp):
         - voxelize via mesh AABB test (kernel_solid)
         - dilate then erode (tile ops)
         - crop back to the level's domain shape
      2) Run NEON kernel to set bc_mask/missing_mask (+ optional distances)
      3) After all levels: resolve out-of-bounds (Warp)
    """

    def __init__(
        self,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
        fill_in_voxels: int = 3,
    ):
        super().__init__(velocity_set, precision_policy, compute_backend, fill_in_voxels)
        if self.compute_backend in [ComputeBackend.JAX, ComputeBackend.WARP]:
            raise NotImplementedError(f"Operator {self.__class__.__name__} not supported in {self.compute_backend} backend.")

    def _construct_neon(self):
        _c = self.velocity_set.c
        _q = self.velocity_set.q
        _opp_indices = self.velocity_set.opp_indices

        @neon.Container.factory(name="MultiresMeshMaskerAABBFill")
        def container(
            mesh_id: Any,
            id_number: Any,
            distances: Any,
            bc_mask: Any,
            missing_mask: Any,
            solid_mask: Any,              # plain level-shaped wp.array; DO NOT call loader.get_read_handle on it
            needs_mesh_distance: Any,
            level: Any,
        ):
            def launcher(loader: neon.Loader):
                # Configure the NEON runtime for the requested level
                loader.set_mres_grid(bc_mask.get_grid(), level)

                # Multires write handles for grid-backed fields
                distances_pn     = loader.get_mres_write_handle(distances)
                bc_mask_pn       = loader.get_mres_write_handle(bc_mask)
                missing_mask_pn  = loader.get_mres_write_handle(missing_mask)

                # NOTE: solid_mask is a plain wp.array created in neon_implementation per level.
                # Do NOT call loader.get_read_handle(solid_mask) on it. We'll access it directly from closure.
                c0 = _c[0]; c1 = _c[1]; c2 = _c[2]

                @wp.func
                def main_kernel(index: Any):
                    # Get thread indices
                    i, j, k = wp.tid()

                    # Compute physical position using NEON-aware helper
                    cell_center_pos = self.helper_masker.index_to_position(bc_mask_pn, index)

                    # Use plain device array 'solid_mask' captured from closure (indexed by i,j,k)
                    if solid_mask[i, j, k] == wp.uint8(255) or self.read_field(bc_mask_pn, index, 0) == wp.uint8(255):
                        self.write_field(bc_mask_pn, index, 0, wp.uint8(255))
                    else:
                        for direction_idx in range(1, _q):
                            dir_vec = wp.vec3f(
                                wp.float32(c0[direction_idx]),
                                wp.float32(c1[direction_idx]),
                                wp.float32(c2[direction_idx]),
                            )
                            ni = i + c0[direction_idx]
                            nj = j + c1[direction_idx]
                            nk = k + c2[direction_idx]

                            # Neighbor check on the plain solid_mask array with bounds check
                            if ni >= 0 and ni < solid_mask.shape[0] and nj >= 0 and nj < solid_mask.shape[1] and nk >= 0 and nk < solid_mask.shape[2]:
                                if solid_mask[ni, nj, nk] == wp.uint8(255):
                                    self.write_field(bc_mask_pn, index, 0, wp.uint8(id_number))
                                    self.write_field(missing_mask_pn, index, _opp_indices[direction_idx], wp.uint8(True))

                                    if not needs_mesh_distance:
                                        continue

                                    max_length = wp.length(dir_vec)
                                    q = wp.mesh_query_ray(mesh_id, cell_center_pos, dir_vec / max_length, 1.5 * max_length)
                                    if q.result:
                                        pos_mesh = wp.mesh_eval_position(mesh_id, q.face, q.u, q.v)
                                        dist = wp.length(pos_mesh - cell_center_pos) - 0.5 * max_length
                                        self.write_field(distances_pn, index, direction_idx, self.store_dtype(dist / max_length))
                                    else:
                                        self.write_field(distances_pn, index, direction_idx, self.store_dtype(1.0))

                loader.declare_kernel(main_kernel)

            return launcher

        return None, container

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_implementation(
        self,
        bc,
        distances,
        bc_mask,
        missing_mask,
        stream=0,
    ):
        # Build mesh + bc id
        mesh_id, bc_id = self._prepare_kernel_inputs(bc, bc_mask)

        # Fetch Warp preprocessing kernels on-demand
        _, kernel_dict = super()._construct_warp()  # {"kernel_solid","dilate_tile","erode_tile"}

        grid = bc_mask.get_grid()
        num_levels = grid.num_levels

        # Finest grid shape (as used in MultiresIndicesBoundaryMasker)
        grid_shape_finest = self.helper_masker.get_grid_shape(bc_mask)

        # Helper to decide whether to forward a Warp stream
        def _launch_with_optional_stream(kernel, **kwargs):
            kw = kwargs.copy()
            stream_obj = kw.pop("stream", None)
            if stream_obj is None or isinstance(stream_obj, int):
                # call wp.launch without a stream kw (int is treated as "no stream")
                return wp.launch(kernel=kernel, **kw)
            else:
                return wp.launch(kernel=kernel, stream=stream_obj, **kw)

        # Define a crop kernel to stay on device
        @wp.kernel
        def crop_kernel(
            input_mask: wp.array3d(dtype=wp.int32),
            output_mask: wp.array3d(dtype=wp.uint8),
            tile_length: int,
        ):
            i, j, k = wp.tid()
            output_mask[i, j, k] = wp.uint8(input_mask[i + tile_length, j + tile_length, k + tile_length])

        for level in range(num_levels):
            # Compute shape at this level by downsampling the finest
            nx, ny, nz = [s // (2**level) for s in grid_shape_finest]

            # --- Build padded solid mask for this level (Warp preprocessing) ---
            tile_length = 2 * self.tile_half
            offset = wp.vec3f(-tile_length, -tile_length, -tile_length)
            pad = 2 * tile_length

            solid_mask = wp.zeros((nx + pad, ny + pad, nz + pad), dtype=wp.int32)
            solid_mask_out = wp.zeros((nx + pad, ny + pad, nz + pad), dtype=wp.int32)

            # Use helper to optionally pass stream
            _launch_with_optional_stream(
                kernel=kernel_dict["kernel_solid"],
                inputs=[mesh_id, solid_mask, offset],
                dim=solid_mask.shape,
                stream=stream,
            )
            _launch_with_optional_stream(
                kernel=kernel_dict["dilate_tile"],
                dim=solid_mask.shape,
                block_dim=32,
                inputs=[solid_mask, solid_mask_out],
                stream=stream,
            )
            _launch_with_optional_stream(
                kernel=kernel_dict["erode_tile"],
                dim=solid_mask.shape,
                block_dim=32,
                inputs=[solid_mask_out, solid_mask],
                stream=stream,
            )

            # Crop back to level domain on device; convert to uint8 for the NEON kernel
            solid_mask_cropped = wp.zeros((nx, ny, nz), dtype=wp.uint8)
            _launch_with_optional_stream(
                kernel=crop_kernel,
                dim=(nx, ny, nz),
                inputs=[solid_mask, solid_mask_cropped, tile_length],
                stream=stream,
            )

            # --- Run NEON kernel for this level ---
            # Pass the plain device array solid_mask_cropped directly into the container (do NOT request get_read_handle)
            c = self.neon_container(
                mesh_id,
                bc_id,
                distances,
                bc_mask,
                missing_mask,
                solid_mask_cropped,                # plain wp.array: used from closure inside container
                wp.static(bc.needs_mesh_distance),
                level,
            )
            c.run(stream, container_runtime=neon.Container.ContainerRuntime.neon)

        # After all levels: resolve OOB on finest
        _launch_with_optional_stream(
            kernel=self.resolve_out_of_bound_kernel,
            inputs=[bc_id, bc_mask, missing_mask],
            dim=bc_mask.shape[1:],   # finest level shape
            stream=stream,
        )

        return distances, bc_mask, missing_mask