from functools import partial
import jax.numpy as jnp
from jax import jit
import warp as wp
import os

from typing import Any

from xlb.compute_backend import ComputeBackend
from xlb.operator.equilibrium import Equilibrium
from xlb.operator import Operator


class QuadraticEquilibrium(Equilibrium):
    """
    Quadratic equilibrium of Boltzmann equation using hermite polynomials.
    Standard equilibrium model for LBM.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0))
    def jax_implementation(self, rho, u):
        cu = 3.0 * jnp.tensordot(self.velocity_set.c, u, axes=(0, 0))
        usqr = 1.5 * jnp.sum(jnp.square(u), axis=0, keepdims=True)
        w = self.velocity_set.w.reshape((-1,) + (1,) * (len(u.shape) - 1))
        feq = rho * w * (1.0 + cu * (1.0 + 0.5 * cu) - usqr)
        return feq

    def _construct_warp(self):
        # Set local constants TODO: This is a hack and should be fixed with warp update
        _c = self.velocity_set.c
        _c_float = self.velocity_set.c_float
        _w = self.velocity_set.w
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)
        _u_vec = wp.vec(self.velocity_set.d, dtype=self.compute_dtype)

        # Construct the equilibrium functional
        @wp.func
        def functional_quadratic(
            rho: Any,
            u: Any,
        ):
            # Allocate the equilibrium
            feq = _f_vec()

            # Compute the equilibrium
            for l in range(self.velocity_set.q):
                # Compute cu
                cu = self.compute_dtype(0.0)
                for d in range(self.velocity_set.d):
                    if _c[d, l] == 1:
                        cu += u[d]
                    elif _c[d, l] == -1:
                        cu -= u[d]
                cu *= self.compute_dtype(3.0)

                # Compute usqr
                usqr = self.compute_dtype(1.5) * wp.dot(u, u)

                # Compute feq
                feq[l] = rho * _w[l] * (self.compute_dtype(1.0) + cu * (self.compute_dtype(1.0) + self.compute_dtype(0.5) * cu) - usqr)

            return feq
        
        # q = 27 exceeds Warp's default max_unroll (16), so a single loop over
        # the velocity set stays rolled.  That makes _w[l] and _c[d, l] runtime
        # reads of a wp.constant matrix and turns the six sign tests below into
        # real branches -- 162 of them per voxel for D3Q27 -- as well as forcing
        # feq into local memory via the dynamic write.
        #
        # Splitting the sweep into groups of four puts every loop under the
        # unroll limit, so l becomes a compile-time constant: the coefficient
        # reads fold, all six branches resolve at compile time, and each
        # direction collapses to between one and four multiplies.
        _n_groups = self.velocity_set.q // 4
        _tail_start = 4 * _n_groups

        @wp.func
        def product_equilibrium_entry(
            l: int,
            base: Any,
            Bx: Any,
            inv_Bx: Any,
            By: Any,
            inv_By: Any,
            Bz: Any,
            inv_Bz: Any,
        ):
            # One lattice direction of the product-form equilibrium.  Called
            # with a literal l so _c/_w fold and the sign tests disappear.
            val = base * self.compute_dtype(_w[l])

            cx = _c[0, l]
            if cx == 1:
                val *= Bx
            elif cx == -1:
                val *= inv_Bx

            cy = _c[1, l]
            if cy == 1:
                val *= By
            elif cy == -1:
                val *= inv_By

            if wp.static(self.velocity_set.d == 3):
                cz = _c[2, l]
                if cz == 1:
                    val *= Bz
                elif cz == -1:
                    val *= inv_Bz

            return val

        # Construct the full product equilibrium functional
        @wp.func
        def functional(
            rho: Any,
            u: Any,
        ):
            # Allocate the equilibrium
            feq = _f_vec()

            # # Compute the equilibrium
            # Product-form entropic equilibrium
            one   = self.compute_dtype(1.0)
            two   = self.compute_dtype(2.0)
            three = self.compute_dtype(3.0)
            max_u = self.compute_dtype(0.7)

            ux = wp.clamp(u[0], -max_u, max_u)
            uy = wp.clamp(u[1], -max_u, max_u)

            sx = wp.sqrt(one + three * ux * ux)
            sy = wp.sqrt(one + three * uy * uy)

            Ax = two - sx
            Ay = two - sy

            num_x = (two * ux + sx)
            den_x = (one - ux)
            Bx = num_x / den_x
            inv_Bx = den_x / num_x
            
            num_y = (two * uy + sy)
            den_y = (one - uy)
            By = num_y / den_y
            inv_By = den_y / num_y

            Psi = Ax * Ay

            # defaults so variables exist even in 2D builds
            Bz = one
            inv_Bz = one            

            if wp.static(self.velocity_set.d == 3):
                uz = wp.clamp(u[2], -max_u, max_u)
                sz = wp.sqrt(one + three * uz * uz)
                Az = two - sz
                num_z = (two * uz + sz)
                den_z = (one - uz)
                Bz = num_z / den_z
                inv_Bz = den_z / num_z
                Psi = Psi * Az

            base = rho * Psi

            # Grouped by four so each loop stays under Warp's unroll limit; see
            # the note on product_equilibrium_entry above.
            for g in range(_n_groups):
                feq[4 * g + 0] = product_equilibrium_entry(4 * g + 0, base, Bx, inv_Bx, By, inv_By, Bz, inv_Bz)
                feq[4 * g + 1] = product_equilibrium_entry(4 * g + 1, base, Bx, inv_Bx, By, inv_By, Bz, inv_Bz)
                feq[4 * g + 2] = product_equilibrium_entry(4 * g + 2, base, Bx, inv_Bx, By, inv_By, Bz, inv_Bz)
                feq[4 * g + 3] = product_equilibrium_entry(4 * g + 3, base, Bx, inv_Bx, By, inv_By, Bz, inv_Bz)

            for l in range(_tail_start, self.velocity_set.q):
                feq[l] = product_equilibrium_entry(l, base, Bx, inv_Bx, By, inv_By, Bz, inv_Bz)

            return feq


        # Construct the warp kernel
        @wp.kernel
        def kernel(
            rho: wp.array4d(dtype=Any),
            u: wp.array4d(dtype=Any),
            f: wp.array4d(dtype=Any),
        ):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)

            # Get the equilibrium
            _u = _u_vec()
            for d in range(self.velocity_set.d):
                _u[d] = u[d, index[0], index[1], index[2]]
            _rho = rho[0, index[0], index[1], index[2]]
            feq = functional(_rho, _u)

            # Set the output
            for l in range(self.velocity_set.q):
                f[l, index[0], index[1], index[2]] = self.store_dtype(feq[l])

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, rho, u, f):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[
                rho,
                u,
                f,
            ],
            dim=rho.shape[1:],
        )
        return f

    def _construct_neon(self):
        import neon

        # Use the warp functional for the NEON backend
        functional, _ = self._construct_warp()

        # Set local constants TODO: This is a hack and should be fixed with warp update
        _u_vec = wp.vec(self.velocity_set.d, dtype=self.compute_dtype)

        @neon.Container.factory(name="QuadraticEquilibrium")
        def container(
            rho: Any,
            u: Any,
            f: Any,
        ):
            def quadratic_equilibrium_ll(loader: neon.Loader):
                loader.set_grid(rho.get_grid())
                rho_pn = loader.get_read_handle(rho)
                u_pn = loader.get_read_handle(u)
                f_pn = loader.get_write_handle(f)

                @wp.func
                def quadratic_equilibrium_cl(index: Any):
                    _u = _u_vec()
                    for d in range(self.velocity_set.d):
                        _u[d] = self.compute_dtype(wp.neon_read(u_pn, index, d))
                    _rho = self.compute_dtype(wp.neon_read(rho_pn, index, 0))
                    feq = functional(_rho, _u)

                    # Set the output
                    for l in range(self.velocity_set.q):
                        wp.neon_write(f_pn, index, l, self.store_dtype(feq[l]))

                loader.declare_kernel(quadratic_equilibrium_cl)

            return quadratic_equilibrium_ll

        return functional, container

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_implementation(self, rho, u, f):
        import neon

        c = self.neon_container(rho, u, f)
        c.run(0, container_runtime=neon.Container.ContainerRuntime.neon)
        return f
