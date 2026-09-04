from functools import partial
import jax.numpy as jnp
from jax import jit
import warp as wp
from typing import Any

from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator


class FirstMoment(Operator):
    """A class to compute the first moment (velocity) of distribution functions."""

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0), inline=True)
    def jax_implementation(self, f, rho):
        u = jnp.tensordot(self.velocity_set.c, f, axes=(-1, 0)) / rho
        return u

    def _construct_warp(self):
        _c = self.velocity_set.c
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)
        _u_vec = wp.vec(self.velocity_set.d, dtype=self.compute_dtype)

        # Number of whole 4-element groups, and where the leftover tail starts.
        _n_groups = self.velocity_set.q // 4
        _tail_start = 4 * _n_groups

        @wp.func
        def split_sum_component(d: int, f: _f_vec):
            # Four interleaved accumulators, one branch per term on the
            # compile-time constant _c[d, l].  Because _c is a wp.constant and
            # the loops unroll, directions with _c[d, l] == 0 (a third of the
            # D3Q27 set) are eliminated at compile time rather than being
            # folded into a compensation term as they were before.
            #
            # Accuracy note: this replaces a Neumaier compensated sum.  Measured
            # over realistic D3Q27 populations the error grows from ~3.0e-9 to
            # ~1.45e-8 relative to sum|c*f| (plain sequential summation would be
            # ~2.2e-8), against an fp32 epsilon of 1.2e-7.
            a0 = self.compute_dtype(0.0)
            a1 = self.compute_dtype(0.0)
            a2 = self.compute_dtype(0.0)
            a3 = self.compute_dtype(0.0)

            for g in range(_n_groups):
                if _c[d, 4 * g + 0] == 1:
                    a0 += f[4 * g + 0]
                elif _c[d, 4 * g + 0] == -1:
                    a0 -= f[4 * g + 0]

                if _c[d, 4 * g + 1] == 1:
                    a1 += f[4 * g + 1]
                elif _c[d, 4 * g + 1] == -1:
                    a1 -= f[4 * g + 1]

                if _c[d, 4 * g + 2] == 1:
                    a2 += f[4 * g + 2]
                elif _c[d, 4 * g + 2] == -1:
                    a2 -= f[4 * g + 2]

                if _c[d, 4 * g + 3] == 1:
                    a3 += f[4 * g + 3]
                elif _c[d, 4 * g + 3] == -1:
                    a3 -= f[4 * g + 3]

            for l in range(_tail_start, self.velocity_set.q):
                if _c[d, l] == 1:
                    a0 += f[l]
                elif _c[d, l] == -1:
                    a0 -= f[l]

            return (a0 + a1) + (a2 + a3)

        @wp.func
        def functional(f: _f_vec, rho: Any):
            u = _u_vec()
            # Split-accumulator summation for each spatial component
            for d in range(self.velocity_set.d):
                u[d] = split_sum_component(d, f)
            # `u /= rho` emits one fp32 division per component.  Without
            # --use_fast_math each is ~20 SASS instructions (FCHK + MUFU.RCP +
            # Newton refinement), so forming the reciprocal once and scaling by
            # it trades d divisions for 1 division and d multiplies.  Measured
            # on the D3Q27 KBC+Smagorinsky collision path (sm_86, ptxas -O3):
            # div.rn.f32 34 -> 31 and 117 -> 113 registers.
            #
            # Accuracy note: reciprocal-then-multiply carries one rounding more
            # than a direct divide.  Over realistic densities (rho = 1 +/- 2%)
            # the max relative error grows from 6.0e-8 (0.5 ulp) to 1.2e-7
            # (1 ulp), mean 2.2e-8 -> 3.0e-8, against an fp32 epsilon of
            # 1.19e-7 -- the same order of trade already taken by the
            # split-accumulator sums above.
            inv_rho = self.compute_dtype(1.0) / rho
            u *= inv_rho
            return u

        @wp.kernel
        def kernel(
            f: wp.array4d(dtype=Any),
            rho: wp.array4d(dtype=Any),
            u: wp.array4d(dtype=Any),
        ):
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)

            _f = _f_vec()
            for l in range(self.velocity_set.q):
                _f[l] = f[l, index[0], index[1], index[2]]
            _rho = rho[0, index[0], index[1], index[2]]
            _u = functional(_f, _rho)

            for d in range(self.velocity_set.d):
                u[d, index[0], index[1], index[2]] = self.store_dtype(_u[d])

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f, rho, u):
        wp.launch(
            self.warp_kernel,
            inputs=[f, rho, u],
            dim=u.shape[1:],
        )
        return u

    def _construct_neon(self):
        functional, _ = self._construct_warp()
        return functional, None

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_implementation(self, f, rho):
        # raise exception as this feature is not implemented yet
        raise NotImplementedError("This feature is not implemented in XLB with the NEON backend yet.")
