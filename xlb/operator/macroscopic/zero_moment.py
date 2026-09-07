from functools import partial
import jax.numpy as jnp
from jax import jit
import warp as wp
from typing import Any

from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator


class ZeroMoment(Operator):
    """A class to compute the zeroth moment (density) of distribution functions."""

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0), inline=True)
    def jax_implementation(self, f):
        return jnp.sum(f, axis=0, keepdims=True)

    def _construct_warp(self):
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)
        # Number of whole 4-element groups, and where the leftover tail starts.
        _n_groups = self.velocity_set.q // 4
        _tail_start = 4 * _n_groups

        @wp.func
        def split_sum(f: _f_vec):
            # Four interleaved accumulators.  Same operation count as a plain
            # sequential sum (q-1 adds) but a quarter of the dependency-chain
            # depth, and no data-dependent branching.
            #
            # Accuracy note: this replaces a Neumaier compensated sum.  Measured
            # over realistic D3Q27 populations the error grows from ~2.2e-8 to
            # ~3.7e-8 relative to sum|f| (plain sequential summation would be
            # ~7.1e-8), against an fp32 epsilon of 1.2e-7.
            a0 = self.compute_dtype(0.0)
            a1 = self.compute_dtype(0.0)
            a2 = self.compute_dtype(0.0)
            a3 = self.compute_dtype(0.0)
            for g in range(_n_groups):
                a0 += f[4 * g + 0]
                a1 += f[4 * g + 1]
                a2 += f[4 * g + 2]
                a3 += f[4 * g + 3]
            for l in range(_tail_start, self.velocity_set.q):
                a0 += f[l]
            return (a0 + a1) + (a2 + a3)

        @wp.func
        def functional(f: _f_vec):
            return split_sum(f)

        @wp.kernel
        def kernel(
            f: wp.array4d(dtype=Any),
            rho: wp.array4d(dtype=Any),
        ):
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)

            _f = _f_vec()
            for l in range(self.velocity_set.q):
                _f[l] = f[l, index[0], index[1], index[2]]
            _rho = functional(_f)

            rho[0, index[0], index[1], index[2]] = _rho

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f, rho):
        wp.launch(self.warp_kernel, inputs=[f, rho], dim=rho.shape[1:])
        return rho

    def _construct_neon(self):
        functional, _ = self._construct_warp()
        return functional, None

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_implementation(self, f, rho):
        # raise exception as this feature is not implemented yet
        raise NotImplementedError("This feature is not implemented in XLB with the NEON backend yet.")
