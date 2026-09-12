"""
Single-resolution momentum transfer with wall-model support.

The base :class:`MomentumTransfer`'s ``@wp.func functional`` (and
:class:`FetchPopulations`'s functional it calls) already accept the
wall-model fields (``_rho``, ``_u``, ``_relax``, ``_norm_vec_pn``,
``_norm_dist_pn``) that ``HybridBC``'s wall-model variant needs -- but the
base's own ``_construct_neon``/``neon_implementation``/container chain never
threads those 5 fields through; its container hardcodes a 6-arg
``functional(index, f_0, f_1, bc_mask, missing_mask, force)`` call against a
function that actually takes 11. :class:`WallModelMomentumTransfer` only
needs to complete that wiring -- the physics/functional code is already
correct.
"""

from typing import Any

import warp as wp

from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb.operator.force.momentum_transfer import MomentumTransfer


class WallModelMomentumTransfer(MomentumTransfer):
    """MomentumTransfer that threads relax/normal_vector/normal_distance
    (and previous-timestep rho/u) through to the wall-model BC functional."""

    def _construct_neon(self):
        import neon

        functional, _ = self._construct_warp()

        @neon.Container.factory(name="WallModelMomentumTransfer")
        def container(
            f_0: Any,
            f_1: Any,
            bc_mask: Any,
            missing_mask: Any,
            force: Any,
            rho0: Any,
            u0: Any,
            relax: Any,
            normal_vector: Any,
            normal_distance: Any,
        ):
            def container_launcher(loader: neon.Loader):
                loader.set_grid(bc_mask.get_grid())
                bc_mask_pn = loader.get_write_handle(bc_mask)
                missing_mask_pn = loader.get_write_handle(missing_mask)
                f_0_pn = loader.get_write_handle(f_0)
                f_1_pn = loader.get_write_handle(f_1)
                rho0_pn = loader.get_write_handle(rho0)
                u0_pn = loader.get_write_handle(u0)
                relax_pn = loader.get_write_handle(relax)
                norm_vec_pn = loader.get_write_handle(normal_vector)
                norm_dist_pn = loader.get_write_handle(normal_distance)

                @wp.func
                def container_kernel(index: Any):
                    functional(
                        index,
                        f_0_pn,
                        f_1_pn,
                        bc_mask_pn,
                        missing_mask_pn,
                        force,
                        rho0_pn,
                        u0_pn,
                        relax_pn,
                        norm_vec_pn,
                        norm_dist_pn,
                    )

                loader.declare_kernel(container_kernel)

            return container_launcher

        return functional, container

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_implementation(self, f_0, f_1, bc_mask, missing_mask, rho0, u0, relax, normal_vector, normal_distance, stream=0):
        import neon

        self.force *= self.compute_dtype(0.0)
        self.fetcher_functional = self.fetcher.neon_functional

        c = self.neon_container(f_0, f_1, bc_mask, missing_mask, self.force, rho0, u0, relax, normal_vector, normal_distance)
        c.run(stream, container_runtime=neon.Container.ContainerRuntime.neon)
        return self.force.numpy()[0]
