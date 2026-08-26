"""
KBC collision operator for LBM.
"""

import jax.numpy as jnp
from jax import jit
import warp as wp
from typing import Any
from functools import partial

from xlb.velocity_set import VelocitySet, D2Q9, D3Q27
from xlb.compute_backend import ComputeBackend
from xlb.operator.collision.collision import Collision
from xlb.operator import Operator
from xlb.operator.macroscopic import SecondMoment as MomentumFlux


class SmagorinskyLESKBC(Collision):
    """KBC collision with Smagorinsky LES turbulence modelling.

    Adjusts the effective relaxation time based on the local strain rate
    estimated from the non-equilibrium stress tensor, using the
    Smagorinsky model constant *C_s*.

    Parameters
    ----------
    velocity_set : VelocitySet, optional
    precision_policy : PrecisionPolicy, optional
    compute_backend : ComputeBackend, optional
    smagorinsky_coef : float
        Smagorinsky model constant (default 0.1).
    """

    def __init__(
        self,
        velocity_set: VelocitySet = None,
        precision_policy=None,
        compute_backend=None,
        smagorinsky_constant = 0.1
    ):
        self.momentum_flux = MomentumFlux()
        self.epsilon = 1e-7
        # Smagorinsky constant - tuned for automotive aero applications
        # Typical values: 0.1 (light), 0.15 (moderate), 0.2 (strong damping)
        self.smagorinsky_constant = smagorinsky_constant
        print(f" Smagorinsky Constant: {smagorinsky_constant}")
        # Precompute (C_s * delta)^2, assuming delta = 1 in lattice units
        self.cs2_delta2 = self.smagorinsky_constant ** 2

        super().__init__(
            velocity_set=velocity_set,
            precision_policy=precision_policy,
            compute_backend=compute_backend,
        )

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0,), donate_argnums=(1, 2, 3))
    def jax_implementation(
        self,
        f: jnp.ndarray,
        feq: jnp.ndarray,
        rho: jnp.ndarray,
        u: jnp.ndarray,
        omega,
    ):
        """
        KBC collision step for lattice.

        Parameters
        ----------
        f : jax.numpy.array
            Distribution function.
        feq : jax.numpy.array
            Equilibrium distribution function.
        rho : jax.numpy.array
            Density.
        u : jax.numpy.array
            Velocity.
        """
        fneq = f - feq
        if isinstance(self.velocity_set, D2Q9):
            shear = self.decompose_shear_d2q9_jax(fneq)
            delta_s = shear * rho / 4.0
        elif isinstance(self.velocity_set, D3Q27):
            shear = self.decompose_shear_d3q27_jax(fneq)
            delta_s = shear * rho
        else:
            raise NotImplementedError("Velocity set not supported: {}".format(type(self.velocity_set)))

        # Compute required constants based on the input omega (omega is the inverse relaxation time)
        beta = self.compute_dtype(0.5) * self.compute_dtype(omega)
        inv_beta = 1.0 / beta

        # Perform collision
        delta_h = fneq - delta_s
        sp1, sp2 = self.compute_entropic_scalar_products(delta_s, delta_h, feq)
        gamma = inv_beta - (2.0 - inv_beta) * sp1 / (self.epsilon + sp2)

        fout = f - beta * (2.0 * delta_s + gamma[None, ...] * delta_h)

        return fout

    @partial(jit, static_argnums=(0,), inline=True)
    def compute_entropic_scalar_products(self, delta_s: jnp.ndarray, delta_h: jnp.ndarray, feq: jnp.ndarray):
        """
        Compute the entropic scalar products to approximate gamma in KBC.

        Returns
        -------
        jax.numpy.array
            sp1 and sp2: Entropic scalar products of delta_s, delta_h, and feq.
        """
        temp = delta_h / feq
        sp1 = jnp.sum(temp * delta_s, axis=0)
        sp2 = jnp.sum(temp * delta_h, axis=0)
        return sp1, sp2

    @partial(jit, static_argnums=(0,), inline=True)
    def decompose_shear_d3q27_jax(self, fneq):
        """
        Decompose fneq into shear components for D3Q27 lattice.

        Parameters
        ----------
        fneq : jax.numpy.ndarray
            Non-equilibrium distribution function.

        Returns
        -------
        jax.numpy.ndarray
            Shear components of fneq.
        """

        # Calculate the momentum flux
        Pi = self.momentum_flux(fneq)
        # Calculating Nxz and Nyz with indices moved to the first dimension
        Nxz = Pi[0, ...] - Pi[5, ...]
        Nyz = Pi[3, ...] - Pi[5, ...]

        # For c = (i, 0, 0), c = (0, j, 0) and c = (0, 0, k)
        s = jnp.zeros_like(fneq)
        s = s.at[9, ...].set((2.0 * Nxz - Nyz) / 6.0)
        s = s.at[18, ...].set((2.0 * Nxz - Nyz) / 6.0)
        s = s.at[3, ...].set((-Nxz + 2.0 * Nyz) / 6.0)
        s = s.at[6, ...].set((-Nxz + 2.0 * Nyz) / 6.0)
        s = s.at[1, ...].set((-Nxz - Nyz) / 6.0)
        s = s.at[2, ...].set((-Nxz - Nyz) / 6.0)

        # For c = (i, j, 0)
        s = s.at[12, ...].set(Pi[1, ...] / 4.0)
        s = s.at[24, ...].set(Pi[1, ...] / 4.0)
        s = s.at[21, ...].set(-Pi[1, ...] / 4.0)
        s = s.at[15, ...].set(-Pi[1, ...] / 4.0)

        # For c = (i, 0, k)
        s = s.at[10, ...].set(Pi[2, ...] / 4.0)
        s = s.at[20, ...].set(Pi[2, ...] / 4.0)
        s = s.at[19, ...].set(-Pi[2, ...] / 4.0)
        s = s.at[11, ...].set(-Pi[2, ...] / 4.0)

        # For c = (0, j, k)
        s = s.at[8, ...].set(Pi[4, ...] / 4.0)
        s = s.at[4, ...].set(Pi[4, ...] / 4.0)
        s = s.at[7, ...].set(-Pi[4, ...] / 4.0)
        s = s.at[5, ...].set(-Pi[4, ...] / 4.0)

        return s

    @partial(jit, static_argnums=(0,), inline=True)
    def decompose_shear_d2q9_jax(self, fneq):
        """
        Decompose fneq into shear components for D2Q9 lattice.

        Parameters
        ----------
        fneq : jax.numpy.array
            Non-equilibrium distribution function.

        Returns
        -------
        jax.numpy.array
            Shear components of fneq.
        """
        Pi = self.momentum_flux(fneq)
        N = Pi[0, ...] - Pi[2, ...]
        s = jnp.zeros_like(fneq)
        s = s.at[3, ...].set(N)
        s = s.at[6, ...].set(N)
        s = s.at[2, ...].set(-N)
        s = s.at[1, ...].set(-N)
        s = s.at[8, ...].set(Pi[1, ...])
        s = s.at[4, ...].set(-Pi[1, ...])
        s = s.at[5, ...].set(-Pi[1, ...])
        s = s.at[7, ...].set(Pi[1, ...])

        return s

    def _construct_warp(self):
        # Raise error if velocity set is not supported
        if not (isinstance(self.velocity_set, D3Q27) or isinstance(self.velocity_set, D2Q9)):
            raise NotImplementedError("Velocity set not supported for warp backend: {}".format(type(self.velocity_set)))

        # Set local constants TODO: This is a hack and should be fixed with warp update
        _u_vec = wp.vec(self.velocity_set.d, dtype=self.compute_dtype)
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)
        _epsilon = wp.constant(self.compute_dtype(self.epsilon))
        _cs2_delta2 = wp.constant(self.compute_dtype(self.cs2_delta2))
        _inv_cs2 = wp.constant(self.compute_dtype(self.velocity_set.inv_cs2))

        @wp.func
        def decompose_shear_d2q9(pineq: Any):
            #pi = self.momentum_flux.warp_functional(fneq)
            N = pineq[0] - pineq[2]
            s = _f_vec()
            s[3] = N
            s[6] = N
            s[2] = -N
            s[1] = -N
            s[8] = pineq[1]
            s[4] = -pineq[1]
            s[5] = -pineq[1]
            s[7] = pineq[1]
            return s

        # Construct functional for decomposing shear
        @wp.func
        def decompose_shear_d3q27(
            pineq: Any,
        ):
            # Get momentum flux
            #pineq = self.momentum_flux.warp_functional(fneq)
            nxz = pineq[0] - pineq[5]
            nyz = pineq[3] - pineq[5]

            # set shear components
            s = _f_vec()
            # for i in range(self.velocity_set.q):
            #     s[i] = self.compute_dtype(0.0)

            # For c = (i, 0, 0), c = (0, j, 0) and c = (0, 0, k)
            two = self.compute_dtype(2.0)
            four = self.compute_dtype(1.0/4.0)
            six = self.compute_dtype(1.0/6.0)

            s[9] = (two * nxz - nyz) * six
            s[18] = (two * nxz - nyz) * six
            s[3] = (-nxz + two * nyz) * six
            s[6] = (-nxz + two * nyz) * six
            s[1] = (-nxz - nyz) * six
            s[2] = (-nxz - nyz) * six

            # For c = (i, j, 0)
            s[12] = pineq[1] * four
            s[24] = pineq[1] * four
            s[21] = -pineq[1] * four
            s[15] = -pineq[1] * four

            # For c = (i, 0, k)
            s[10] = pineq[2] * four
            s[20] = pineq[2] * four
            s[19] = -pineq[2] * four
            s[11] = -pineq[2] * four

            # For c = (0, j, k)
            s[8] = pineq[4] * four
            s[4] = pineq[4] * four
            s[7] = -pineq[4] * four
            s[5] = -pineq[4] * four

            return s

        @wp.func
        def fused_entropic_products_and_gamma_bounds(
            f: Any,
            feq: Any,
            delta_s: Any,
            delta_h: Any,
            beta: Any,
            f_floor: Any,   # e.g. _epsilon
        ):
            """
            Single-pass over q:
            - compute entropic scalar products sp1, sp2
            - compute gamma feasibility bounds [gamma_min, gamma_max] from fout >= f_floor

            Returns: (sp1, sp2, gamma_min, gamma_max)
            """
            # Plain accumulation.  sp1 and sp2 are already two independent
            # dependency chains, and gamma is subsequently clamped to [0, 20] by
            # apply_gamma_bounds, so the extra precision of a compensated sum
            # was not reaching the result.  Dropping the compensation removes
            # six of the ten arithmetic operations per direction here.
            sp1 = self.compute_dtype(0.0)
            sp2 = self.compute_dtype(0.0)

            # Wide initial bounds
            gamma_min = self.compute_dtype(-1.0e6)
            gamma_max = self.compute_dtype( 1.0e6)

            # constants
            zero = self.compute_dtype(0.0)
            two  = self.compute_dtype(2.0)

            _feq_floor = _epsilon
            _ratio_max = self.compute_dtype(1e4)

            for i in range(wp.static(self.velocity_set.q)):
                # -------- entropic scalar products (sp1/sp2) --------
                # Floor feq to prevent catastrophic division
                feq_safe = wp.max(feq[i], _feq_floor)

                temp_i = delta_h[i] / feq_safe
                temp_i = wp.clamp(temp_i, -_ratio_max, _ratio_max)

                sp1 += temp_i * delta_s[i]
                sp2 += temp_i * delta_h[i]

                # -------- gamma feasibility bounds from positivity --------
                # pre_i = f[i] - 2*beta*delta_s[i]
                pre_i = f[i] - two * beta * delta_s[i]
                headroom = pre_i - f_floor   # must keep >= 0 after gamma term

                # fout_i = pre_i - beta*gamma*delta_h[i] >= f_floor
                # => headroom - gamma*(beta*delta_h[i]) >= 0
                coeff = beta * delta_h[i]

                if coeff > _epsilon:
                    # gamma <= headroom/coeff
                    gamma_max = wp.min(gamma_max, headroom / coeff)
                elif coeff < -_epsilon:
                    # gamma >= headroom/coeff
                    gamma_min = wp.max(gamma_min, headroom / coeff)
                # else: no constraint
            
            return sp1, sp2, gamma_min, gamma_max

        @wp.func
        def apply_gamma_bounds(
            gamma: Any,
            gamma_min: Any,
            gamma_max: Any,
        ):
            """
            Clamp gamma to [gamma_min, gamma_max] if feasible;
            otherwise fall back to gamma=2 (BGK-like).
            Also applies your physics bounds [0, 20].
            """
            zero = self.compute_dtype(0.0)
            two  = self.compute_dtype(2.0)

            if gamma_min <= gamma_max:
                gamma = wp.clamp(gamma, gamma_min, gamma_max)
                gamma = wp.clamp(gamma, zero, self.compute_dtype(20.0))
            else:
                gamma = two
            return gamma

        @wp.func
        def compute_smagorinsky_omega_d3q27(
            omega: Any,
            rho: Any,
            pineq: Any,
        ):
            """Compute effective omega with Smagorinsky SGS for D3Q27."""
            # Get momentum flux (Pi_neq)
            #pi_neq = self.momentum_flux.warp_functional(fneq)
            
            # Extract stress components: [Pi_xx, Pi_xy, Pi_xz, Pi_yy, Pi_yz, Pi_zz]
            Pi_xx = pineq[0]
            Pi_xy = pineq[1]
            Pi_xz = pineq[2]
            Pi_yy = pineq[3]
            Pi_yz = pineq[4]
            Pi_zz = pineq[5]
            
            # Compute |Pi_neq|
            two = self.compute_dtype(2.0)
            Pi_neq_magnitude_sq = (
                Pi_xx * Pi_xx + Pi_yy * Pi_yy + Pi_zz * Pi_zz +
                two * (Pi_xy * Pi_xy + Pi_xz * Pi_xz + Pi_yz * Pi_yz)
            )
            Pi_neq_magnitude = wp.sqrt(Pi_neq_magnitude_sq + _epsilon)
            
            # Base relaxation time
            one = self.compute_dtype(1.0)
            tau_0 = one / omega
            
            # Compute discriminant
            eighteen = self.compute_dtype(18.0)
            discriminant = tau_0 * tau_0 + eighteen * _cs2_delta2 * Pi_neq_magnitude / (rho + _epsilon)
            
            # Effective relaxation time
            half = self.compute_dtype(0.5)
            tau_eff = half * (tau_0 + wp.sqrt(discriminant + _epsilon))
            
            # Clamp for stability
            tau_min = self.compute_dtype(0.5001)
            tau_eff = wp.max(tau_eff, tau_min)
            
            # Convert to omega
            omega_eff = one / tau_eff
            
            return omega_eff
        
        @wp.func
        def compute_smagorinsky_omega_d2q9(
            omega: Any,
            rho: Any,
            pi_neq: Any,
        ):
            """Compute effective omega with Smagorinsky SGS for D2Q9."""
                       
            # Extract stress components: [Pi_xx, Pi_xy, Pi_yy]
            Pi_xx = pi_neq[0]
            Pi_xy = pi_neq[1]
            Pi_yy = pi_neq[2]
            
            # Compute |Pi_neq|
            two = self.compute_dtype(2.0)
            Pi_neq_magnitude_sq = Pi_xx * Pi_xx + two * Pi_xy * Pi_xy + Pi_yy * Pi_yy
            Pi_neq_magnitude = wp.sqrt(Pi_neq_magnitude_sq + _epsilon)
            
            # Base relaxation time
            one = self.compute_dtype(1.0)
            tau_0 = one / omega
            
            # Compute discriminant
            eighteen = self.compute_dtype(18.0)
            discriminant = tau_0 * tau_0 + eighteen * _cs2_delta2 * Pi_neq_magnitude / (rho + _epsilon)
            
            # Effective relaxation time
            half = self.compute_dtype(0.5)
            tau_eff = half * (tau_0 + wp.sqrt(discriminant + _epsilon))
            
            # Clamp for stability
            tau_min = self.compute_dtype(0.5001)
            tau_eff = wp.max(tau_eff, tau_min)
            
            # Convert to omega
            omega_eff = one / tau_eff
            
            return omega_eff

        # Construct the functional
        @wp.func
        def functional(
            f: Any,
            feq: Any,
            rho: Any,
            u: Any,
            omega: Any,
        ):
            # Compute shear and delta_s
            fneq = f - feq
            pineq = self.momentum_flux.warp_functional(fneq)
            if wp.static(self.velocity_set.d == 3):
                shear = decompose_shear_d3q27(pineq)
                delta_s = shear 
                omega_eff = compute_smagorinsky_omega_d3q27(omega, rho, pineq)
            else:
                shear = decompose_shear_d2q9(pineq)
                delta_s = shear  / self.compute_dtype(4.0)
                omega_eff = compute_smagorinsky_omega_d2q9(omega, rho, pineq)

            # Compute required constants based on the input omega (omega is the inverse relaxation time)
            _beta = self.compute_dtype(0.5) * self.compute_dtype(omega_eff)
            _inv_beta = self.compute_dtype(1.0) / _beta

            # Perform collision
            delta_h = fneq - delta_s
            two = self.compute_dtype(2.0)
            sp1, sp2, gmin, gmax = fused_entropic_products_and_gamma_bounds(f, feq, delta_s, delta_h, _beta, _epsilon)
             
            gamma = _inv_beta - (two - _inv_beta) * sp1 / wp.max(sp2, _epsilon)
            # Update Gamma based on Positivity Range enforcement
            gamma = apply_gamma_bounds(gamma, gmin, gmax)
            
            fout = f - _beta * (two * delta_s + gamma * delta_h)
            
            return fout

            

        # Construct the warp kernel
        @wp.kernel
        def kernel(
            f: wp.array4d(dtype=Any),
            feq: wp.array4d(dtype=Any),
            fout: wp.array4d(dtype=Any),
            rho: wp.array4d(dtype=Any),
            u: wp.array4d(dtype=Any),
            omega: Any,
        ):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)  # TODO: Warp needs to fix this

            # Load needed values
            _f = _f_vec()
            _feq = _f_vec()
            _d = self.velocity_set.d
            for l in range(self.velocity_set.q):
                _f[l] = f[l, index[0], index[1], index[2]]
                _feq[l] = feq[l, index[0], index[1], index[2]]
            _u = _u_vec()
            for l in range(_d):
                _u[l] = u[l, index[0], index[1], index[2]]
            _rho = rho[0, index[0], index[1], index[2]]

            # Compute the collision
            _fout = functional(_f, _feq, _rho, _u, omega)

            # Write the result
            for l in range(self.velocity_set.q):
                fout[l, index[0], index[1], index[2]] = self.store_dtype(_fout[l])

        return functional, kernel

    def _construct_neon(self):
        # Redefine the momentum flux operator for the neon backend
        # This is because the neon backend relies on the warp functionals for its operations.
        self.momentum_flux = MomentumFlux(compute_backend=ComputeBackend.WARP)
        functional, _ = self._construct_warp()
        return functional, None

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f, feq, fout, rho, u, omega):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[
                f,
                feq,
                fout,
                rho,
                u,
                omega,
            ],
            dim=f.shape[1:],
        )
        return fout

