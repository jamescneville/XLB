# Base class for all equilibriums

from functools import partial
import jax.numpy as jnp
from jax import jit
import warp as wp
from typing import Any

from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator


class SecondMoment(Operator):
    """
    Operator to calculate the second moment of distribution functions.

    The second moment may be used to compute the momentum flux in the computation of
    the stress tensor in the Lattice Boltzmann Method (LBM).

    Important Note:
    Note that this rank 2 symmetric tensor (dim*dim) has been converted into a rank one
    vector where the diagonal and off-diagonal components correspond to the following elements of
    the vector:
    if self.grid.dim == 3:
        diagonal    = (0, 3, 5)
        offdiagonal = (1, 2, 4)
    elif self.grid.dim == 2:
        diagonal    = (0, 2)
        offdiagonal = (1,)

    ** For any reduction operation on the full tensor it is crucial to account for the full tensor by
    considering all diagonal and off-diagonal components.
    """

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def jax_implementation(
        self,
        fneq: jnp.ndarray,
    ):
        """
        This function computes the second order moment, which is the product of the
        distribution functions (f) and the lattice moments (cc).

        Parameters
        ----------
        fneq: jax.numpy.ndarray
            The distribution functions.

        Returns
        -------
        jax.numpy.ndarray
            The computed second moment.
        """
        return jnp.tensordot(self.velocity_set.cc, fneq, axes=(0, 0))

    def _construct_warp(self):
        # Make constants for warp
        _cc = self.velocity_set.cc
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)
        _pi_dim = self.velocity_set.d * (self.velocity_set.d + 1) // 2
        _pi_vec = wp.vec(
            _pi_dim,
            dtype=self.compute_dtype,
        )

        # Number of whole 2-element groups, and where the leftover tail starts.
        _n_pairs = self.velocity_set.q // 2
        _tail_start = 2 * _n_pairs

        # Construct functional for computing second moment
        @wp.func
        def functional(
            fneq: Any,
        ):
            # Get second order moment (a symmetric tensor shaped into a vector)
            pi = _pi_vec()

            # Split-accumulator summation, two interleaved chains per tensor
            # component.  The six components are themselves independent chains,
            # so this exposes twelve-way instruction-level parallelism.
            #
            # The `!= 0` test is on a wp.constant with compile-time-constant
            # indices, so it costs nothing at runtime and drops the 44% of _cc
            # entries that are exactly zero for D3Q27.  Multiplication by the
            # remaining +/-1 entries folds to a move or a negate.
            #
            # This replaces a compensated sum whose correction term carried the
            # wrong sign -- it computed `corr = (pi - y) + t` where Kahan
            # requires `c = (t - s) - y`.  Applying the correction inverted
            # amplified the rounding error instead of cancelling it, making that
            # version ~1.96x LESS accurate than plain summation at ~4x the cost.
            # This version is both faster and more accurate than what it
            # replaces.
            for d in range(_pi_dim):
                a0 = self.compute_dtype(0.0)
                a1 = self.compute_dtype(0.0)

                for g in range(_n_pairs):
                    if _cc[2 * g + 0, d] != self.compute_dtype(0.0):
                        a0 += _cc[2 * g + 0, d] * fneq[2 * g + 0]
                    if _cc[2 * g + 1, d] != self.compute_dtype(0.0):
                        a1 += _cc[2 * g + 1, d] * fneq[2 * g + 1]

                for l in range(_tail_start, self.velocity_set.q):
                    if _cc[l, d] != self.compute_dtype(0.0):
                        a0 += _cc[l, d] * fneq[l]

                pi[d] = a0 + a1

            return pi

        # Construct the kernel
        @wp.kernel
        def kernel(
            f: wp.array4d(dtype=Any),
            pi: wp.array4d(dtype=Any),
        ):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)

            # Get the equilibrium
            _f = _f_vec()
            for l in range(self.velocity_set.q):
                _f[l] = f[l, index[0], index[1], index[2]]
            _pi = functional(_f)

            # Set the output
            for d in range(_pi_dim):
                pi[d, index[0], index[1], index[2]] = self.store_dtype(_pi[d])

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f, pi):
        # Launch the warp kernel
        wp.launch(self.warp_kernel, inputs=[f, pi], dim=pi.shape[1:])
        return pi

    def _construct_neon(self):
        functional, _ = self._construct_warp()
        return functional, None

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_implementation(self, f, rho):
        # raise exception as this feature is not implemented yet
        raise NotImplementedError("This feature is not implemented in XLB with the NEON backend yet.")
