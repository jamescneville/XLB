"""
Warp/Neon helper functions shared by multiple boundary conditions.

:class:`HelperFunctionsBC` exposes ``@wp.func`` helpers for bounce-back,
regularization, Grad's approximation, moving-wall corrections,
interpolated BCs, and BC thread-data loading.  These are used as building
blocks by the concrete BC classes.

Also contains :class:`EncodeAuxiliaryData` and
:class:`MultiresEncodeAuxiliaryData` operators for writing user-prescribed
BC profiles into the ``f_1`` buffer during initialization.
"""

import inspect
from typing import Any, Callable

import warp as wp

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb import DefaultConfig, ComputeBackend
from xlb.operator.operator import Operator
from xlb.operator.macroscopic import SecondMoment as MomentumFlux
from xlb.operator.macroscopic import Macroscopic
from xlb.operator.equilibrium import QuadraticEquilibrium


class HelperFunctionsBC(object):
    """Collection of Warp/Neon ``@wp.func`` helpers for boundary conditions.

    Parameters
    ----------
    velocity_set : VelocitySet, optional
    precision_policy : PrecisionPolicy, optional
    compute_backend : ComputeBackend, optional
        Must be ``WARP`` or ``NEON`` (JAX not supported).
    distance_decoder_function : callable, optional
        Function to decode wall-distance data for interpolated BCs.
    """

    def __init__(self, velocity_set=None, precision_policy=None, compute_backend=None, distance_decoder_function=None):
        if compute_backend == ComputeBackend.JAX:
            raise ValueError("This helper class contains helper functions only for the WARP implementation of some BCs not JAX!")

        # Set the default values from the global config
        self.velocity_set = velocity_set or DefaultConfig.velocity_set
        self.precision_policy = precision_policy or DefaultConfig.default_precision_policy
        self.compute_backend = compute_backend or DefaultConfig.default_backend
        self.distance_decoder_function = distance_decoder_function

        # Set the compute and Store dtypes
        compute_dtype = self.precision_policy.compute_precision.wp_dtype
        store_dtype = self.precision_policy.store_precision.wp_dtype

        # Set local constants
        _d = self.velocity_set.d
        _q = self.velocity_set.q
        _opp_indices = self.velocity_set.opp_indices
        _w = self.velocity_set.w
        _c = self.velocity_set.c
        _c_float = self.velocity_set.c_float
        _cs2 = compute_dtype(self.velocity_set.cs2)
        _qi = self.velocity_set.qi
        _u_vec = wp.vec(_d, dtype=compute_dtype)
        _f_vec = wp.vec(_q, dtype=compute_dtype)
        _missing_mask_vec = wp.vec(_q, dtype=wp.uint8)  # TODO fix vec bool
        _nt = _d * (_d + 1) // 2
        _epsilon = compute_dtype(1e-6)
        # Wall model constants
        _kappa = compute_dtype(0.41)  # von Karman constant

        # Define the operator needed for computing equilibrium
        equilibrium = QuadraticEquilibrium(velocity_set, precision_policy, compute_backend)

        # Define the operator needed for computing macroscopic variables
        macroscopic = Macroscopic(velocity_set, precision_policy, compute_backend)

        # Define the operator needed for computing equilibrium
        equilibrium = QuadraticEquilibrium(velocity_set, precision_policy, compute_backend)

        # Define the operator needed for computing macroscopic variables
        macroscopic = Macroscopic(velocity_set, precision_policy, compute_backend)

        # Define the operator needed for computing the momentum flux
        momentum_flux = MomentumFlux(velocity_set, precision_policy, compute_backend)

        @wp.func
        def get_bc_thread_data(
            f_pre: wp.array4d(dtype=Any),
            f_post: wp.array4d(dtype=Any),
            bc_mask: wp.array4d(dtype=wp.uint8),
            missing_mask: wp.array4d(dtype=wp.uint8),
            index: wp.vec3i,
        ):
            # Get the boundary id and missing mask
            _f_pre = _f_vec()
            _f_post = _f_vec()
            _boundary_id = bc_mask[0, index[0], index[1], index[2]]
            _missing_mask = _missing_mask_vec()
            for l in range(_q):
                # q-sized vector of populations
                _f_pre[l] = compute_dtype(f_pre[l, index[0], index[1], index[2]])
                _f_post[l] = compute_dtype(f_post[l, index[0], index[1], index[2]])

                # TODO fix vec bool
                if missing_mask[l, index[0], index[1], index[2]]:
                    _missing_mask[l] = wp.uint8(1)
                else:
                    _missing_mask[l] = wp.uint8(0)
            return _f_pre, _f_post, _boundary_id, _missing_mask

        @wp.func
        def neon_get_bc_thread_data(
            f_pre_pn: Any,
            f_post_pn: Any,
            bc_mask_pn: Any,
            missing_mask_pn: Any,
            index: Any,
        ):
            # Get the boundary id and missing mask
            _f_pre = _f_vec()
            _f_post = _f_vec()
            _boundary_id = wp.neon_read(bc_mask_pn, index, 0)
            _missing_mask = _missing_mask_vec()
            for l in range(_q):
                # q-sized vector of populations
                _f_pre[l] = compute_dtype(wp.neon_read(f_pre_pn, index, l))
                _f_post[l] = compute_dtype(wp.neon_read(f_post_pn, index, l))
                _missing_mask[l] = wp.neon_read(missing_mask_pn, index, l)

            return _f_pre, _f_post, _boundary_id, _missing_mask

        @wp.func
        def get_bc_fsum(
            fpop: Any,
            _missing_mask: Any,
        ):
            fsum_known = compute_dtype(0.0)
            fsum_middle = compute_dtype(0.0)
            for l in range(_q):
                if _missing_mask[_opp_indices[l]] == wp.uint8(1):
                    fsum_known += compute_dtype(2.0) * fpop[l]
                elif _missing_mask[l] != wp.uint8(1):
                    fsum_middle += fpop[l]
            return fsum_known + fsum_middle

        @wp.func
        def get_normal_vectors(
            _missing_mask: Any,
        ):
            if wp.static(_d == 3):
                for l in range(_q):
                    if _missing_mask[l] == wp.uint8(1) and wp.abs(_c[0, l]) + wp.abs(_c[1, l]) + wp.abs(_c[2, l]) == 1:
                        return -_u_vec(_c_float[0, l], _c_float[1, l], _c_float[2, l])
            else:
                for l in range(_q):
                    if _missing_mask[l] == wp.uint8(1) and wp.abs(_c[0, l]) + wp.abs(_c[1, l]) == 1:
                        return -_u_vec(_c_float[0, l], _c_float[1, l])

        @wp.func
        def bounceback_nonequilibrium(
            fpop: Any,
            feq: Any,
            _missing_mask: Any,
        ):
            for l in range(_q):
                if _missing_mask[l] == wp.uint8(1):
                    fpop[l] = fpop[_opp_indices[l]] + feq[l] - feq[_opp_indices[l]]
            return fpop

        @wp.func
        def regularize_fpop(
            fpop: Any,
            feq: Any,
        ):
            """
            Regularizes the distribution functions by adding non-equilibrium contributions based on second moments of fpop.
            """
            # Compute momentum flux of off-equilibrium populations for regularization: Pi^1 = Pi^{neq}
            for l in range(_q):
                fpop[l] = fpop[l] - feq[l]

            PiNeq = momentum_flux.warp_functional(fpop)
            three = compute_dtype(3.0)
            trace = (PiNeq[0] + PiNeq[3] + PiNeq[5]) / three

            # Compute double dot product Qi:Pi1 (where Pi1 = PiNeq)            
            for l in range(_q):
                QiPi = compute_dtype(0.0)
                for t in range(_nt):
                    if t == 0 or t == 3 or t == 5:
                        QiPi += _qi[l, t] * (PiNeq[t] - trace)
                    else:
                        QiPi += _qi[l, t] * PiNeq[t]

                # assign all populations based on eq 45 of Latt et al (2008)
                # fneq ~ f^1
                fpop1 = compute_dtype(4.5) * _w[l] * QiPi
                fpop[l] = feq[l] + fpop1                
                fpop[l] = wp.max(fpop[l], _epsilon)

            return fpop

        @wp.func
        def grads_approximate_fpop(
            _missing_mask: Any,
            rho: Any,
            u: Any,
            f_post: Any,
        ):
            # Purpose: Using Grad's approximation to represent fpop based on macroscopic inputs used for outflow [1] and
            # Dirichlet BCs [2]
            # [1] S. Chikatax`marla, S. Ansumali, and I. Karlin, "Grad's approximation for missing data in lattice Boltzmann
            #   simulations", Europhys. Lett. 74, 215 (2006).
            # [2] Dorschner, B., Chikatamarla, S. S., Bösch, F., & Karlin, I. V. (2015). Grad's approximation for moving and
            #    stationary walls in entropic lattice Boltzmann simulations. Journal of Computational Physics, 295, 340-354.

            # Note: See also self.regularize_fpop function which is somewhat similar.

            # Compute pressure tensor Pi using all f_post-streaming values
            Pi = momentum_flux.warp_functional(f_post)

            # Compute double dot product Qi:Pi1 (where Pi1 = PiNeq)
            nt = _d * (_d + 1) // 2
            for l in range(_q):
                if _missing_mask[l] == wp.uint8(1):
                    # compute dot product of qi and Pi
                    QiPi = compute_dtype(0.0)
                    for t in range(nt):
                        if t == 0 or t == 3 or t == 5:
                            QiPi += _qi[l, t] * (Pi[t] - rho / compute_dtype(3.0))
                        else:
                            QiPi += _qi[l, t] * Pi[t]

                    # Compute c.u
                    cu = compute_dtype(0.0)
                    for d in range(_d):
                        if _c[d, l] == 1:
                            cu += u[d]
                        elif _c[d, l] == -1:
                            cu -= u[d]
                    cu *= compute_dtype(3.0)

                    # change f_post using the Grad's approximation
                    f_post[l] = rho * _w[l] * (compute_dtype(1.0) + cu) + _w[l] * compute_dtype(4.5) * QiPi

            return f_post

        @wp.func
        def moving_wall_fpop_correction(
            u_wall: Any,
            lattice_direction: Any,
            rho: Any,
        ):
            # Add forcing term necessary to account for the local density changes caused by the mass displacement
            # as the object moves with velocity u_wall.
            # [1] L.-S. Luo, Unified theory of lattice Boltzmann models for nonideal gases, Phys. Rev. Lett. 81 (1998) 1618-1621.
            # [2] L.-S. Luo, Theory of the lattice Boltzmann method: Lattice Boltzmann models for nonideal gases, Phys. Rev. E 62 (2000) 4982-4996.
            #
            # Note: this function must be called within a for-loop over all lattice directions and the populations to be modified must
            # be only those in the missing direction (the check for missing direction must be outside of this function).
            cu = compute_dtype(0.0)
            l = lattice_direction
            for d in range(_d):                
                cu += u_wall[d] * _c_float[d, l]                
                
            cu *= compute_dtype(6.0) * _w[l] * rho
            return cu

        @wp.func
        def interpolated_bounceback(
            index: Any,
            _missing_mask: Any,
            f_0: Any,
            f_1: Any,
            f_pre: Any,
            f_post: Any,
            u_wall: Any,
            needs_moving_wall_treatment: bool,
            needs_mesh_distance: bool,
            _rho: Any,
        ):
            # A local single-node version of the interpolated bounce-back boundary condition due to Bouzidi for a lattice
            # Boltzmann method simulation.
            # Ref:
            # [1] Yu, D., Mei, R., Shyy, W., 2003. A unified boundary treatment in lattice boltzmann method,
            # in: 41st aerospace sciences meeting and exhibit, p. 953.

            rho  = wp.neon_read(_rho, index, 0) 
            zero = compute_dtype(0.0)
            half = compute_dtype(0.5)
            one = compute_dtype(1.0)
            two = compute_dtype(2.0)

            for l in range(_q):
                # If the mask is missing then take the opposite index
                if _missing_mask[l] == wp.uint8(1):
                    # Handle sandwiched boundaries
                    if _missing_mask[_opp_indices[l]] == wp.uint8(1):
                        f_post[l] = f_pre[_opp_indices[l]]   
                    else:
                        # The normalized distance to the mesh or "weights" have been stored in known directions of f_1
                        if needs_mesh_distance:
                            # use weights associated with curved boundaries that are properly stored in f_1.
                            weight = compute_dtype(self.distance_decoder_function(f_1, index, l))
                            weight = wp.clamp(weight, compute_dtype(0.01), compute_dtype(0.99))                          
                            # Use differentiable interpolated BB to find f_missing:
                            f_post[l] = ((one - weight) * f_post[_opp_indices[l]] + weight * (f_pre[l] + f_pre[_opp_indices[l]])) / (one + weight)
                            
                            # Add contribution due to moving_wall to f_missing as is usual in regular Bouzidi BC
                            if needs_moving_wall_treatment:
                                correction = moving_wall_fpop_correction(u_wall, l, rho)
                                f_post[l] += correction / (one + weight)  
                        else:
                            # Use regular halfway bounceback
                            f_post[l] = f_pre[_opp_indices[l]]
                     
                            # Add contribution due to moving_wall to f_missing as is usual in regular Bouzidi BC
                            if needs_moving_wall_treatment:
                                f_post[l] += moving_wall_fpop_correction(u_wall, l, rho)
            return f_post

        @wp.func
        def interpolated_nonequilibrium_bounceback(
            index: Any,
            _missing_mask: Any,
            f_0: Any,
            f_1: Any,
            f_pre: Any,
            f_post: Any,
            u_wall: Any,
            needs_moving_wall_treatment: bool,
            needs_mesh_distance: bool,
        ):
            
            one = compute_dtype(1.0)
            
            # Compute density, velocity using all f_post-collision values
            rho, u = macroscopic.warp_functional(f_pre)
            feq = equilibrium.warp_functional(rho, u)

            # Compute equilibrium distribution at the wall
            if needs_moving_wall_treatment:
                feq_wall = equilibrium.warp_functional(rho, u_wall)
            else:
                feq_wall = _f_vec()

            # Apply method in Tao et al (2018) [1] to find missing populations at the boundary
            
            for l in range(_q):
                # If the mask is missing then take the opposite index
                if _missing_mask[l] == wp.uint8(1):
                    # Handle sandwiched boundaries
                    if _missing_mask[_opp_indices[l]] == wp.uint8(1):
                        #Sandwich set to FEQ and move on
                        f_post[l] = feq[l]                    
                    else:
                        # The normalized distance to the mesh or "weights" have been stored in known directions of f_1
                        if needs_mesh_distance:
                            # use weights associated with curved boundaries that are properly stored in f_1.
                            weight = compute_dtype(self.distance_decoder_function(f_1, index, l))
                            weight = wp.clamp(weight, compute_dtype(0.01), compute_dtype(0.99))
                        else:
                            weight = compute_dtype(0.5)

                        # Use non-equilibrium bounceback to find f_missing:
                        fneq = f_pre[_opp_indices[l]] - feq[_opp_indices[l]]

                        # Compute equilibrium distribution at the wall
                        # Same quadratic equilibrium but accounting for zero velocity (no-slip)
                        if not needs_moving_wall_treatment:
                            feq_wall[l] = _w[l] * rho

                        # Assemble wall population for doing interpolation at the boundary
                        f_wall = feq_wall[l] + fneq
                        f_post[l] = (f_wall + weight * f_pre[l]) / (one + weight)

            return f_post

        @wp.func
        def neon_index_to_warp(neon_field_hdl: Any, index: Any):
            # Unpack the global index in Neon at the finest level and convert it to a warp vector
            cIdx = wp.neon_global_idx(neon_field_hdl, index)
            gx = wp.neon_get_x(cIdx)
            gy = wp.neon_get_y(cIdx)
            gz = wp.neon_get_z(cIdx)

            # XLB is flattening the z dimension in 3D, while neon uses the y dimension
            if _d == 2:
                gy, gz = gz, gy

            # Get warp indices
            index_wp = wp.vec3i(gx, gy, gz)
            return index_wp

        # ============================================================================
        # Wall Model Functions
        # ============================================================================    

        @wp.func
        def smoothstep(edge0: Any, edge1: Any, x: Any):
            zero = compute_dtype(0.0)
            one = compute_dtype(1.0)
            two = compute_dtype(2.0)
            three = compute_dtype(3.0)

            denom = wp.max(edge1 - edge0, _epsilon)
            t = wp.clamp((x - edge0) / denom, zero, one)

            return t * t * (three - two * t)
        
        @wp.func
        def reichardt_profile(y_plus: Any) -> Any:
            """
            Reichardt (1951) velocity profile - explicit u+(y+).
            Correctly captures: u+ = y+ as y+ -> 0, log-law as y+ -> inf.
            """            
            C = compute_dtype(11.0)

            log_term = (compute_dtype(1.0) / _kappa) * wp.log(
                compute_dtype(1.0) + _kappa * y_plus
            )

            damping = compute_dtype(8.5) * (
                compute_dtype(1.0)
                - wp.exp(-y_plus / C)
                - (y_plus / C) * wp.exp(-y_plus / compute_dtype(3.0))
            )

            u_plus = log_term + damping
            return wp.max(u_plus, compute_dtype(0.0))

        @wp.func
        def reichardt_derivative(y_plus: Any) -> Any:
            """
            Analytical derivative du+/dy+ of the Reichardt profile.
            """
           
            C = compute_dtype(11.0)
            C3 = compute_dtype(3.0)

            d_log = compute_dtype(1.0) / (compute_dtype(1.0) + _kappa * y_plus)

            exp_C = wp.exp(-y_plus / C)
            exp_C3 = wp.exp(-y_plus / C3)

            d_damping = compute_dtype(8.5) * (
                exp_C / C
                - (compute_dtype(1.0) / C) * exp_C3
                + (y_plus / (C * C3)) * exp_C3
            )

            return wp.max(d_log + d_damping, _epsilon)

        @wp.func
        def reichardt_profile_and_derivative(y_plus: Any):
            """
            Fused Reichardt profile and derivative for Newton iteration.

            Keeps the same arithmetic structure as reichardt_profile() and
            reichardt_derivative(), but shares exp(-y/C) and exp(-y/3).
            """
            C = compute_dtype(11.0)
            C3 = compute_dtype(3.0)

            inv_C = compute_dtype(1.0) / C
            inv_C3 = compute_dtype(1.0) / C3

            exp_C = wp.exp(-y_plus / C)
            exp_C3 = wp.exp(-y_plus / C3)

            log_term = (compute_dtype(1.0) / _kappa) * wp.log(
                compute_dtype(1.0) + _kappa * y_plus
            )

            damping = compute_dtype(8.5) * (
                compute_dtype(1.0)
                - exp_C
                - (y_plus / C) * exp_C3
            )

            u_plus = log_term + damping
            u_plus = wp.max(u_plus, compute_dtype(0.0))

            d_log = compute_dtype(1.0) / (compute_dtype(1.0) + _kappa * y_plus)

            d_damping = compute_dtype(8.5) * (
                exp_C * inv_C
                - inv_C * exp_C3
                + (y_plus * inv_C * inv_C3) * exp_C3
            )

            du_dy = wp.max(d_log + d_damping, _epsilon)

            return u_plus, du_dy
        
        @wp.func
        def solve_wall_function(K: Any) -> Any:
            """
            Solve for u_plus given K = |u_parallel| * y / nu = y+ * u+.
            Uses Newton iteration with the Reichardt profile.
            Returns u_plus.
            """
            if K < _epsilon:
                return wp.sqrt(wp.max(K, compute_dtype(0.0)))

            if K < compute_dtype(25.0):
                u_plus = wp.sqrt(K)
            else:
                u_plus = wp.sqrt(K / wp.max(wp.log(K), compute_dtype(1.0)))

            for _ in range(15):
                y_plus = K / wp.max(u_plus, _epsilon)

                u_profile, du_dy = reichardt_profile_and_derivative(y_plus)

                residual = u_plus - u_profile

                if wp.abs(residual) < compute_dtype(1e-6) * wp.max(u_plus, compute_dtype(1.0)):
                    break

                g_prime = compute_dtype(1.0) + du_dy * K / (u_plus * u_plus + _epsilon)

                delta = residual / wp.max(g_prime, _epsilon)
                u_plus = u_plus - delta
                u_plus = wp.max(u_plus, _epsilon)

            return wp.max(u_plus, _epsilon)

        @wp.func
        def solve_wall_function0(K: Any) -> Any:
            """
            Solve for u_plus given K = |u_parallel| * y / nu = y+ * u+.
            Uses Newton iteration with the Reichardt profile.
            Returns u_plus (dimensionless velocity).
            """
            if K < _epsilon:
                return wp.sqrt(wp.max(K, compute_dtype(0.0)))

            # Initial guess
            if K < compute_dtype(25.0):
                u_plus = wp.sqrt(K)
            else:
                u_plus = wp.sqrt(K / wp.max(wp.log(K), compute_dtype(1.0)))

            # Newton iteration: solve g(u+) = u+ - profile(K/u+) = 0
            for _ in range(15):
                y_plus = K / wp.max(u_plus, _epsilon)
                u_profile = reichardt_profile(y_plus)
                residual = u_plus - u_profile

                if wp.abs(residual) < compute_dtype(1e-6) * wp.max(u_plus, compute_dtype(1.0)):
                    break

                du_dy = reichardt_derivative(y_plus)
                g_prime = compute_dtype(1.0) + du_dy * K / (u_plus * u_plus + _epsilon)

                delta = residual / wp.max(g_prime, _epsilon)
                u_plus = u_plus - delta
                u_plus = wp.max(u_plus, _epsilon)

            return wp.max(u_plus, _epsilon)

        @wp.func
        def sample_neighbor(
            index: Any,
            normal: Any,
            _rho: Any, 
            _u: Any,
            streamwise: Any,
        ):
            """
            1. Snap to the lattice link best aligned with `normal` → step 1 voxel to the "neighbor".
            2. Snap to the lattice link best aligned with `streamwise` → from the neighbor,
            step 1 voxel in downdstream and upstream to get data.

            
            """
            
            # ============================================================
            # Find best lattice link aligned with the NORMAL 
            # ============================================================
            best_n_l   = wp.int32(0)
            best_n_dot = compute_dtype(-1.0e9)
            for l in range(1, _q):
                # Start with 1 as zero is rest position
                cx = _c_float[0, l]
                cy = _c_float[1, l]
                cz = _c_float[2, l]
                if (cx == compute_dtype(0.0)) and (cy == compute_dtype(0.0)) and (cz == compute_dtype(0.0)):
                    continue
                c    = wp.vec3(cx, cy, cz)
                cmag = wp.length(c)
                if cmag <= _epsilon:
                    continue
                c_unit = c / cmag
                a = wp.dot(c_unit, normal)
                if a > best_n_dot:
                    best_n_dot = a
                    best_n_l   = wp.int32(l)

            # Flip if best dot was negative.
            step_dir = wp.vec3i(_c[0, best_n_l], _c[1, best_n_l], _c[2, best_n_l])
            if best_n_dot < compute_dtype(0.0):
                step_dir = -step_dir
            
            # ============================================================
            # Find neighbor properties and streamwise vector
            # ============================================================
            ngh_n = wp.neon_ngh_idx(wp.int8(step_dir[0]), wp.int8(step_dir[1]), wp.int8(step_dir[2]))

            # Neighbor Read
            rho_center = compute_dtype(wp.neon_read(_rho, index, 0))
            u_neighbor = _u_vec()
            for d in range(_d):
                has_neihbor = wp.bool(False)
                f_aux = compute_dtype(wp.neon_read_ngh(_u, index, ngh_n, d, store_dtype(0.0), has_neihbor))
                if has_neihbor:                    
                    u_neighbor[d] = f_aux

            neighbor_dist = wp.length(wp.vec3(compute_dtype(step_dir[0]), compute_dtype(step_dir[1]), compute_dtype(step_dir[2]), ))
    
            # ============================================================
            # Find best lattice link aligned with the STREAMWISE
            # ============================================================
            best_s_l   = wp.int32(0)
            best_s_dot = compute_dtype(-1.0e9)
            for l in range(1, _q):
                # Start with 1 as zero is rest position
                cx = _c_float[0, l]
                cy = _c_float[1, l]
                cz = _c_float[2, l]
                if (cx == compute_dtype(0.0)) and (cy == compute_dtype(0.0)) and (cz == compute_dtype(0.0)):
                    continue
                c    = wp.vec3(cx, cy, cz)
                cmag = wp.length(c)
                if cmag <= _epsilon:
                    continue
                c_unit = c / cmag
                a = wp.dot(c_unit, streamwise)                
                if a > best_s_dot:
                    best_s_dot = a
                    best_s_l   = wp.int32(l)

            # Flip if best dot was negative.            
            stream_step = wp.vec3i(_c[0, best_s_l], _c[1, best_s_l], _c[2, best_s_l])
            if best_s_dot < compute_dtype(0.0):
                stream_step = -stream_step
            #Test reaching 2 voxels away rather than just 1 out from center
            stream_step *= 2
            upstream_dist = wp.length(wp.vec3(
                compute_dtype(stream_step[0]),
                compute_dtype(stream_step[1]),
                compute_dtype(stream_step[2]),
            )) 
            # Upstream dist is for neighbor to upstream 
            # To include Downstream we x2 the length
            streamwise_dist = wp.max(compute_dtype(2.0) * upstream_dist, _epsilon)

            # ============================================================
            # Sample UPSTREAM and DOWNSTREAM
            # ============================================================
                
            # Upstream Reads
            f_upstream_off = wp.vec3i( step_dir[0] - stream_step[0], step_dir[1] - stream_step[1], step_dir[2] - stream_step[2], )
            ngh_uf = wp.neon_ngh_idx(wp.int8(f_upstream_off[0]), wp.int8(f_upstream_off[1]), wp.int8(f_upstream_off[2]))
            
            f_rho_upstream = rho_center
            has_upstrem = wp.bool(False)
            f_aux = compute_dtype(
            wp.neon_read_ngh(_rho, index, ngh_uf, 0, store_dtype(0.0), has_upstrem))
            if has_upstrem:                    
                f_rho_upstream = f_aux

            f_u_upstream = u_neighbor
            for d in range(_d):
                has_neihbor = wp.bool(False)
                f_aux = compute_dtype(wp.neon_read_ngh(_u, index, ngh_uf, d, store_dtype(0.0), has_neihbor))
                if has_neihbor:                    
                    f_u_upstream[d] = f_aux

            # Downstream Reads    
            f_downstream_off = wp.vec3i( step_dir[0] + stream_step[0], step_dir[1] + stream_step[1], step_dir[2] + stream_step[2], )
            ngh_df = wp.neon_ngh_idx(wp.int8(f_downstream_off[0]), wp.int8(f_downstream_off[1]), wp.int8(f_downstream_off[2]))        
            
            f_rho_downstream = rho_center
            has_downstream = wp.bool(False)
            f_aux = compute_dtype(
                wp.neon_read_ngh(_rho, index, ngh_df, 0, store_dtype(0.0), has_downstream))
            if has_downstream:                    
                f_rho_downstream = f_aux  
                
            f_u_downstream = u_neighbor
            for d in range(_d):
                has_neihbor = wp.bool(False)
                f_aux = compute_dtype(wp.neon_read_ngh(_u, index, ngh_df, d, store_dtype(0.0), has_neihbor))
                if has_neihbor:                    
                    f_u_downstream[d] = f_aux    

            # Coherence check
            f_u_upstream_normal = normal * wp.dot(f_u_upstream, normal)
            f_u_upstream_streamwise = f_u_upstream - f_u_upstream_normal
            f_u_upstream_mag = wp.length(f_u_upstream_streamwise)    
            f_u_upstream_streamwise = f_u_upstream_streamwise / f_u_upstream_mag

            f_u_downstream_normal = normal * wp.dot(f_u_downstream, normal)
            f_u_downstream_streamwise = f_u_downstream - f_u_downstream_normal
            f_u_downstream_mag = wp.length(f_u_downstream_streamwise)    
            f_u_downstream_streamwise = f_u_downstream_streamwise / f_u_downstream_mag

            dot_up = compute_dtype(0.0)
            dot_dn = compute_dtype(0.0)

            if f_u_upstream_mag > _epsilon:
                dot_up = wp.dot(streamwise, f_u_upstream_streamwise)

            if f_u_downstream_mag > _epsilon:
                dot_dn = wp.dot(streamwise,f_u_downstream_streamwise)

            dot_tol = compute_dtype(0.25) #~80deg    0.5 ~60deg higher = tighter 0.45was working well
            
            if (dot_up > dot_tol) and (dot_dn > dot_tol):
                rho_up = f_rho_upstream
                rho_down = f_rho_downstream
                coherent = compute_dtype(1.0)
            else:
                rho_up = rho_center
                rho_down = rho_center
                coherent = compute_dtype(0.0)          
         
                                     
            return u_neighbor, rho_up, rho_down, neighbor_dist, streamwise, streamwise_dist, coherent
            
        @wp.func
        def compute_wall_modeled_velocity(
            index: Any,
            _missing_mask: Any,
            u_wall: Any,
            nu: Any,
            _rho: Any,
            _u: Any,
            _relax: Any,
            _norm_vec:Any,
            _norm_dist:Any,
        ):
            
            zero = compute_dtype(0.0)
            one = compute_dtype(1.0)

            # -----------------------------------------------------------------
            # SANDWICH DETECTION
            # -----------------------------------------------------------------
            for l in range(_q):
                if _missing_mask[l] == wp.uint8(1) and _missing_mask[_opp_indices[l]] == wp.uint8(1):
                    return u_wall, _relax

            # -----------------------------------------------------------------
            # GEOMETRY: SURFACE NORMAL AND WALL DISTANCE
            # -----------------------------------------------------------------            
            normal = _u_vec()
            for d in range(_d):
                normal[d] = compute_dtype(wp.neon_read(_norm_vec, index, d))
            
            y_b = compute_dtype(wp.neon_read(_norm_dist, index, 0))

            # -----------------------------------------------------------------
            # VELOCITY AT BOUNDARY CELL B
            # -----------------------------------------------------------------
            u_B = _u_vec()
            for d in range(_d):
                u_B[d] = compute_dtype(wp.neon_read(_u, index, d))

            nu = wp.max(compute_dtype(nu), _epsilon)

            # -----------------------------------------------------------------
            # TANGENTIAL VELOCITY AND STREAMWISE DIRECTION AT B
            # -----------------------------------------------------------------
            u_B = u_B - u_wall
            u_b_norm = wp.dot(u_B, normal)
            u_b_tangent = u_B - normal * u_b_norm
            u_b_tangent_len = wp.length(u_b_tangent)

            if u_b_tangent_len < _epsilon:
                return u_wall, _relax

            streamwiseb = u_b_tangent / u_b_tangent_len

            # -----------------------------------------------------------------
            # NEIGHBOR SAMPLING
            # -----------------------------------------------------------------
            u_f, rho_up, rho_down, neighbor_dist, streamwise, streamwise_dist, coherent = sample_neighbor(
                index, normal, _rho, _u, streamwiseb
            )

            # =================================================================
            # SECTION 1: ZPG WALL FUNCTION — BASELINE u_tau AND U_wm
            # =================================================================
            y_f = y_b + neighbor_dist

            u_f = u_f - u_wall
            u_f_mag = wp.length(u_f)

            # Tangential direction at F (safe fallback)
            # u_f_norm = wp.dot(u_f, normal)
            # u_f_tangent = u_f - normal * u_f_norm
            # u_f_tangent_len = wp.length(u_f_tangent)
            # if u_f_tangent_len > _epsilon:
            #     streamwisef = u_f_tangent / u_f_tangent_len
            # else:
            #     streamwisef = streamwise          

            # Use B-streamwise for signed streamwise speed
            u_f_signed = wp.dot(u_f, streamwise)
            u_f_par_mag = wp.abs(u_f_signed)

            if u_f_par_mag < _epsilon:
                return u_wall, _relax

            K = u_f_par_mag * y_f / nu
            K = wp.max(K, _epsilon)

            u_plus_F = solve_wall_function(K)
            u_tau = u_f_par_mag / wp.max(u_plus_F, _epsilon)
            u_tau_zpg = wp.max(u_tau, _epsilon)

            y_plus_B = y_b * u_tau_zpg / nu
            u_plus_B = reichardt_profile(y_plus_B)
            U_wm_B_zpg = u_tau_zpg * u_plus_B

            # =================================================================
            # SECTION 2: STREAMWISE PRESSURE GRADIENT
            # =================================================================
            dp_ds = _cs2 * (rho_down - rho_up) / wp.max(streamwise_dist, _epsilon)
            rho_b = wp.max(compute_dtype(wp.neon_read(_rho, index, 0)), compute_dtype(1.0e-6))

            a_pg = dp_ds / rho_b

            sign = one
            if a_pg < compute_dtype(0.0):
                sign = -one

            u_p = sign * wp.pow(nu * wp.abs(a_pg), _cs2)
            pg_instant = u_p / wp.max(u_tau_zpg, _epsilon)

            # =================================================================
            # SECTION 3: EMA OF R_pg
            # =================================================================
            ema     = compute_dtype(0.001)

            if coherent == zero:
                pg_avg = _relax * compute_dtype(0.98)
            else:       
                pg_avg = ema * pg_instant + (one - ema) * _relax

            pg_avg = wp.clamp(pg_avg, compute_dtype(-1.2), compute_dtype(1.2))  
   

            # =================================================================
            # SECTION 4: PRESSURE GRADIENT CORRECTION FACTOR 
            # =================================================================
            
            # Deadband Pressure between +/- p0
            # p0 = zero   
            # pressure_gate = one
            
            # if pg_avg > p0:
            #     # APG Correction: Nonlinear reduction
            #     p_eff = pg_avg - p0
            #     k0 = compute_dtype(9.0)
            #     n0 = compute_dtype(1.5) #1.75
            #     pressure_gate = wp.exp(-k0 * wp.pow(p_eff, n0))
          
            #     pressure_gate = wp.max(pressure_gate, compute_dtype(0.025))
            
            # if pg_avg < -p0:
            #     p_eff = -pg_avg - p0
            #     k1 = compute_dtype(2.0)
            #     n1 = compute_dtype(1.5)
            #     fpg_max = compute_dtype(1.0)
            #     pressure_gate = one + (fpg_max - one) * (one - wp.exp(-k1 * wp.pow(p_eff, n1)))
      

          
            # =================================================================
            # SECTION 5: LIFTOFF CORRECTION
            # =================================================================    
            # Signed wall-normal angle indicator
            # f_angle > 0 : lifting away from wall
            # f_angle < 0 : pointing into wall
            # f_angle ~ 0 : tangent to wall
                   
            separation_gate = one
            if pg_avg > zero:
                f_angle = wp.dot(u_f / wp.max(u_f_mag, _epsilon), normal) 
                deg_to_rad = compute_dtype(0.017453292519943295)

                # Base liftoff rolloff in separation-angle space
                separation_rolloff_start_deg = compute_dtype(2.5)
                separation_rolloff_full_deg  = compute_dtype(17.0)

                # FPG shield tuning in degree space
                # fpg_shield_start_pg   = compute_dtype(-0.1)  # shield begins once pg_avg goes below this
                # fpg_shield_deg_per_pg = compute_dtype(60.0)   # protection gained per 1.0 of extra negative pg_avg
                # fpg_shield_max_deg    = compute_dtype(12.0)   # cap on total shield protection

               
                # if pg_avg < fpg_shield_start_pg:
                #     shield_shift_deg = wp.min(
                #         (fpg_shield_start_pg - pg_avg) * fpg_shield_deg_per_pg,
                #         fpg_shield_max_deg,
                #     )

                rolloff_start_deg = separation_rolloff_start_deg 
                rolloff_full_deg  = separation_rolloff_full_deg  

                rolloff_start = wp.sin(rolloff_start_deg * deg_to_rad)
                rolloff_full  = wp.sin(rolloff_full_deg  * deg_to_rad)
                rolloff_full  = wp.max(rolloff_full, rolloff_start + compute_dtype(1.0e-6))

                if f_angle > rolloff_start:
                    penalty = wp.clamp(
                        (f_angle - rolloff_start) / (rolloff_full - rolloff_start),
                        zero,
                        one,
                    )
                    penalty_smooth = penalty * penalty * (compute_dtype(3.0) - compute_dtype(2.0) * penalty)
                    separation_gate = one - penalty_smooth
           
            # =================================================================
            # SECTION 6: COHERENCE CHECK
            # =================================================================
            coherence_gate = compute_dtype(1.0) 
            if coherent == zero:
                coherence_gate = compute_dtype(0.01) 

            # =================================================================
            # SECTION 7: FINAL VELOCITY ASSEMBLY
            # =================================================================
            U_wm_B_final = U_wm_B_zpg * separation_gate * coherence_gate
            U_wm_B_final = wp.clamp(
                U_wm_B_final,
                zero,
                compute_dtype(2.0) * u_f_par_mag,  
            )

            u_wall_eff = u_wall + U_wm_B_final * streamwise           

            return u_wall_eff, pg_avg
        
        self.get_bc_thread_data = get_bc_thread_data
        self.get_bc_fsum = get_bc_fsum
        self.get_normal_vectors = get_normal_vectors
        self.bounceback_nonequilibrium = bounceback_nonequilibrium
        self.regularize_fpop = regularize_fpop
        self.grads_approximate_fpop = grads_approximate_fpop
        self.moving_wall_fpop_correction = moving_wall_fpop_correction
        self.interpolated_bounceback = interpolated_bounceback
        self.interpolated_nonequilibrium_bounceback = interpolated_nonequilibrium_bounceback
        self.neon_get_bc_thread_data = neon_get_bc_thread_data
        self.neon_index_to_warp = neon_index_to_warp
        self.compute_wall_modeled_velocity = compute_wall_modeled_velocity


class EncodeAuxiliaryData(Operator):
    """
    Operator for encoding boundary auxiliary data during initialization.
    """

    def __init__(
        self,
        boundary_id: int,
        num_of_aux_data: int,
        user_defined_functional: Callable,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
    ):
        self.user_defined_functional = user_defined_functional
        self.boundary_id = wp.uint8(boundary_id)
        self.num_of_aux_data = num_of_aux_data

        super().__init__(velocity_set, precision_policy, compute_backend)

        # Inspect the signature of the user-defined functional.
        # We assume the profile function takes only the index as input and is hence time-independent.
        sig = inspect.signature(user_defined_functional)
        assert self.compute_backend != ComputeBackend.JAX, "Encoding/decoding of auxiliary data are not required for boundary conditions in JAX"
        assert len(sig.parameters) == 1, f"User-defined functional must take exactly one argument (the index), it received {len(sig.parameters)}."

        # Define a HelperFunctionsBC instance
        self.bc_helper = HelperFunctionsBC(
            velocity_set=self.velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=self.compute_backend,
        )

        # TODO: Somehow raise an error if the number of prescribed values does not match the number of missing directions

    def _construct_warp(self):
        """
        Constructs the warp kernel for the auxiliary data recovery.
        """
        # Find velocity index for (0, 0, 0)
        lattice_central_index = self.velocity_set.center_index
        _opp_indices = self.velocity_set.opp_indices
        _id = self.boundary_id
        _num_of_aux_data = self.num_of_aux_data
        _aux_vec = wp.vec(_num_of_aux_data, dtype=self.compute_dtype)

        @wp.func
        def encoder_functional(
            index: Any,
            _missing_mask: Any,
            field_storage: Any,
            prescribed_values: Any,
        ):
            if len(prescribed_values) != _num_of_aux_data:
                wp.printf("Error: User-defined profile must return a vector of size %d\n", _num_of_aux_data)
                return

            # Write the result for all q directions, but only store up to _num_of_aux_data
            counter = wp.int32(0)
            for l in range(self.velocity_set.q):
                # Only store up to _num_of_aux_data
                if counter == _num_of_aux_data:
                    return

                if l == lattice_central_index:
                    # The first BC auxiliary data is stored in the zero'th index of f_1 associated with its center.
                    self.write_field(field_storage, index, l, self.store_dtype(prescribed_values[l]))
                    counter += 1
                elif _missing_mask[l] == wp.uint8(1):
                    # The other remaining BC auxiliary data are stored in missing directions of f_1.
                    self.write_field(field_storage, index, _opp_indices[l], self.store_dtype(prescribed_values[l]))
                    counter += 1

        @wp.func
        def decoder_functional(
            field_storage: Any,
            index: Any,
            _missing_mask: Any,
        ):
            """
            Decode the encoded values needed for the boundary condition treatment from the center location in field_storage.
            """

            # Define a vector to hold prescribed_values
            prescribed_values = _aux_vec()

            # Read all q directions, but only retrieve up to _num_of_aux_data
            counter = wp.int32(0)
            for l in range(self.velocity_set.q):
                # Only retrieve up to _num_of_aux_data
                if counter == _num_of_aux_data:
                    return prescribed_values

                if l == lattice_central_index:
                    # The first BC auxiliary data is stored in the zero'th index of f_1 associated with its center.
                    value = self.read_field(field_storage, index, l)
                    prescribed_values[counter] = self.compute_dtype(value)
                    counter += 1
                elif _missing_mask[l] == wp.uint8(1):
                    # The other remaining BC auxiliary data are stored in missing directions of f_1.
                    value = self.read_field(field_storage, index, _opp_indices[l])
                    prescribed_values[counter] = self.compute_dtype(value)
                    counter += 1

        # Construct the warp kernel
        @wp.kernel
        def kernel(
            f_1: wp.array4d(dtype=Any),
            bc_mask: wp.array4d(dtype=wp.uint8),
            missing_mask: wp.array4d(dtype=wp.uint8),
        ):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)

            # read tid data
            _, _, _boundary_id, _missing_mask = self.bc_helper.get_bc_thread_data(f_1, f_1, bc_mask, missing_mask, index)

            # Apply the functional
            # change this to use central location
            if _boundary_id == _id:
                # prescribed_values is a q-sized vector of type wp.vec
                prescribed_values = self.user_defined_functional(index)

                # call the functional
                encoder_functional(index, _missing_mask, f_1, prescribed_values)

        functional_dict = {"encoder": encoder_functional, "decoder": decoder_functional}
        return functional_dict, kernel

    def _construct_neon(self):
        import neon

        """
        Constructs the Neon container for encoding auxiliary data recovery.
        """
        # Use the warp functional for the Neon backend
        functional_dict, _ = self._construct_warp()
        encoder_functional = functional_dict["encoder"]
        _id = self.boundary_id

        # Construct the Neon container
        @neon.Container.factory(name="EncodingAuxData_" + str(_id))
        def aux_data_init_container(
            f_1: Any,
            bc_mask: Any,
            missing_mask: Any,
        ):
            def aux_data_init_ll(loader: neon.Loader):
                loader.set_grid(f_1.get_grid())

                f_1_pn = loader.get_write_handle(f_1)
                bc_mask_pn = loader.get_read_handle(bc_mask)
                missing_mask_pn = loader.get_read_handle(missing_mask)

                @wp.func
                def aux_data_init_cl(index: Any):
                    # read tid data
                    _, _, _boundary_id, _missing_mask = self.bc_helper.neon_get_bc_thread_data(f_1_pn, f_1_pn, bc_mask_pn, missing_mask_pn, index)

                    # Apply the functional
                    if _boundary_id == _id:
                        warp_index = self.bc_helper.neon_index_to_warp(f_1_pn, index)
                        prescribed_values = self.user_defined_functional(warp_index)

                        # Call the functional
                        encoder_functional(index, _missing_mask, f_1_pn, prescribed_values)

                # Declare the kernel in the Neon loader
                loader.declare_kernel(aux_data_init_cl)

            return aux_data_init_ll

        return functional_dict, aux_data_init_container

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f_1, bc_mask, missing_mask):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[f_1, bc_mask, missing_mask],
            dim=f_1.shape[1:],
        )
        return f_1

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_implementation(self, f_1, bc_mask, missing_mask):
        import neon

        c = self.neon_container(f_1, bc_mask, missing_mask)
        c.run(0, container_runtime=neon.Container.ContainerRuntime.neon)
        return f_1


class MultiresEncodeAuxiliaryData(EncodeAuxiliaryData):
    """
    Operator for encoding boundary auxiliary data during initialization.
    """

    def __init__(
        self,
        boundary_id: int,
        num_of_aux_data: int,
        user_defined_functional: Callable,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
    ):
        super().__init__(
            boundary_id=boundary_id,
            num_of_aux_data=num_of_aux_data,
            user_defined_functional=user_defined_functional,
            velocity_set=velocity_set,
            precision_policy=precision_policy,
            compute_backend=compute_backend,
        )

        assert self.compute_backend == ComputeBackend.NEON, f"Operator {self.__class__.__name__} not supported in {self.compute_backend} backend."

    def _construct_neon(self):
        """
        Constructs the Neon container for encoding auxiliary data recovery.
        """
        import neon

        # Borrow the functional from the warp implementation
        functional_dict, _ = self._construct_warp()
        encoder_functional = functional_dict["encoder"]
        _id = self.boundary_id

        # Construct the Neon container
        @neon.Container.factory(name="MultiresEncodingAuxData_" + str(_id))
        def aux_data_init_container(
            f_1: Any,
            bc_mask: Any,
            missing_mask: Any,
            level: Any,
        ):
            def aux_data_init_ll(loader: neon.Loader):
                loader.set_mres_grid(f_1.get_grid(), level)

                f_1_pn = loader.get_mres_write_handle(f_1)
                bc_mask_pn = loader.get_mres_read_handle(bc_mask)
                missing_mask_pn = loader.get_mres_read_handle(missing_mask)

                @wp.func
                def aux_data_init_cl(index: Any):
                    # read tid data
                    _, _, _boundary_id, _missing_mask = self.bc_helper.neon_get_bc_thread_data(f_1_pn, f_1_pn, bc_mask_pn, missing_mask_pn, index)

                    # Apply the functional
                    if _boundary_id == _id:
                        # IMPORTANT: XLB assumes the user_defined_functional in multi-res
                        # simulations uses finest-level indices, enabling BCs that span
                        # multiple levels.
                        warp_index = self.bc_helper.neon_index_to_warp(f_1_pn, index)
                        prescribed_values = self.user_defined_functional(warp_index)

                        # Call the functional
                        encoder_functional(index, _missing_mask, f_1_pn, prescribed_values)

                # Declare the kernel in the Neon loader
                loader.declare_kernel(aux_data_init_cl)

            return aux_data_init_ll

        return functional_dict, aux_data_init_container

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_implementation(self, f_1, bc_mask, missing_mask, stream):
        import neon

        grid = bc_mask.get_grid()
        for level in range(grid.num_levels):
            c = self.neon_container(f_1, bc_mask, missing_mask, level)
            c.run(stream, container_runtime=neon.Container.ContainerRuntime.neon)
        return f_1
