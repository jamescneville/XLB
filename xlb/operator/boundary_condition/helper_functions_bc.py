import inspect
from typing import Any, Callable

import warp as wp
import neon

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb import DefaultConfig, ComputeBackend
from xlb.operator.operator import Operator
from xlb.operator.macroscopic import SecondMoment as MomentumFlux
from xlb.operator.macroscopic import Macroscopic
from xlb.operator.equilibrium import QuadraticEquilibrium


class HelperFunctionsBC(object):
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
        _qi = self.velocity_set.qi
        _u_vec = wp.vec(_d, dtype=compute_dtype)
        _f_vec = wp.vec(_q, dtype=compute_dtype)
        _missing_mask_vec = wp.vec(_q, dtype=wp.uint8)  # TODO fix vec bool
        _nt = _d * (_d + 1) // 2
        _nt_vec = wp.vec(_nt, dtype=compute_dtype)
        _epsilon = compute_dtype(1e-8)
        _c_center_index = self.velocity_set.center_index

        # Define the operator needed for computing equilibrium
        equilibrium = QuadraticEquilibrium(velocity_set, precision_policy, compute_backend)

        # Define the operator needed for computing macroscopic variables
        macroscopic = Macroscopic(velocity_set, precision_policy, compute_backend)

        # Define the operator needed for computing the momentum flux
        momentum_flux = MomentumFlux(velocity_set, precision_policy, compute_backend)

        # Wall model constants
        _kappa = compute_dtype(0.41)  # von Karman constant
        _B = compute_dtype(5.2)       # Log-law constant
        _A_plus = compute_dtype(26.0)  # van Driest damping constant
        _cs2 = compute_dtype(self.velocity_set.cs2)
        _pi = compute_dtype(3.14159265358979323846)

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
        def get_normal_and_distance(
            _missing_mask: Any,
            f_1: Any,
            index: Any,
        ):
            """
            Compute both the surface normal vector and normal distance based on
            actual distances from voxel center to mesh surface.
            """
            normal = _u_vec(0.0, 0.0, 0.0)
            normal_distance = compute_dtype(0.0)
            weight_sum = compute_dtype(0.0)
            
            # First pass: Compute weighted normal based on actual distances
            for l in range(_q):
                if _missing_mask[l] == wp.uint8(1):
                    # Get distance fraction (0-1) from distance decoder
                    dist_fraction = compute_dtype(self.distance_decoder_function(f_1, index, l))
                    if dist_fraction == compute_dtype(5.0):
                        #dist_fraction = compute_dtype(1.0)
                        continue  # Skip miss-hits
                    dist_fraction = wp.max(dist_fraction, _epsilon)
                    
                    # Get lattice vector
                    c_l = wp.vec3(_c_float[0, l], _c_float[1, l], _c_float[2, l])
                    c_mag = wp.length(c_l)
                    
                    # Actual distance from voxel center to mesh along this direction
                    actual_distance = dist_fraction * c_mag# + 0.5 * c_mag
                    
                    # Unit vector in this direction
                    c_unit = c_l / c_mag  # Streamlined: use vector division
                    
                    # Weight by inverse of actual distance (closer surfaces have more influence)
                    # Add small epsilon to avoid division by zero
                    weight =  compute_dtype(1.0) / (actual_distance * actual_distance + _epsilon)
                    
                    # Accumulate weighted normal vector
                    normal += c_unit * weight  # Streamlined: vector addition and scalar multiplication
                    weight_sum += weight
            
            # Normalize the weighted normal
            if weight_sum > compute_dtype(0.0):
                normal /= weight_sum  # Streamlined: vector division
            
            # Unit normalize the normal vector
            mag = wp.length(normal)  # Optimized: use wp.length instead of manual sqrt
            if mag > compute_dtype(0.0):
                normal /= mag  # Streamlined: vector division
            
            # Invert normal to point into fluid
            normal = -normal

            # Fallback if no real hits: Use all missing directions with assumed dist_fraction=0.5
            if weight_sum <= _epsilon:
                weight_sum = compute_dtype(0.0)  # Reset for fallback
                normal = _u_vec(0.0, 0.0, 0.0)
                for l in range(_q):
                    if _missing_mask[l] == wp.uint8(1):
                        dist_fraction = compute_dtype(0.5)
                        c_l = wp.vec3(_c_float[0, l], _c_float[1, l], _c_float[2, l])
                        c_mag = wp.length(c_l)
                        actual_distance = dist_fraction * c_mag# + 0.5 * c_mag
                        c_unit = c_l / c_mag  # Streamlined: vector division
                        weight = compute_dtype(1.0) / (actual_distance * actual_distance + _epsilon)
                        normal += c_unit * weight  # Streamlined: vector addition and scalar multiplication
                        weight_sum += weight
                # Normalize the weighted normal
                if weight_sum > compute_dtype(0.0):
                    normal /= weight_sum  # Streamlined: vector division
                # Unit normalize the normal vector
                mag = wp.length(normal)  # Optimized: use wp.length instead of manual sqrt
                if mag > compute_dtype(0.0):
                    normal /= mag  # Streamlined: vector division
                # Invert normal to point into fluid
                normal = -normal
            
            # Second pass: Project actual distances onto normal and average
            distance_weight_sum = compute_dtype(0.0)
            
            for l in range(_q):
                if _missing_mask[l] == wp.uint8(1):
                    # Get distance fraction and actual distance again
                    dist_fraction = compute_dtype(self.distance_decoder_function(f_1, index, l))
                    if dist_fraction == compute_dtype(5.0):
                        #dist_fraction = compute_dtype(1.0)
                        continue  # Skip miss-hits
                    dist_fraction = wp.max(dist_fraction, _epsilon)
                    c_l = wp.vec3(_c_float[0, l], _c_float[1, l], _c_float[2, l])
                    c_mag = wp.length(c_l)
                    actual_distance = dist_fraction * c_mag# + 0.5 * c_mag
                    
                    # Unit vector in this direction
                    c_unit = c_l / c_mag  # Streamlined: vector division
                    
                    # Project this distance vector onto the normal
                    # (dot product gives projection length, negative because normal points into fluid)
                    projection = -wp.dot(c_unit, normal) * actual_distance
                    projection = wp.max(projection, compute_dtype(0.0)) # Zero-out negative projections
                    
                    # Weight by inverse distance for averaging
                    # weight = compute_dtype(1.0) / (actual_distance * actual_distance + _epsilon) # Distance weighted
                    # weight = compute_dtype(1.0) / (projection * projection + _epsilon) # Projection weighted
                    weight = (projection * projection) / (actual_distance * actual_distance * actual_distance *actual_distance + _epsilon)
                    normal_distance += projection * weight
                    distance_weight_sum += weight
            
            # Normalize the distance
            if distance_weight_sum > compute_dtype(0.0):
                normal_distance /= distance_weight_sum

            # Fallback for distance if no real hits
            if distance_weight_sum <= _epsilon:
                distance_weight_sum = compute_dtype(0.0)  # Reset for fallback
                normal_distance = compute_dtype(0.0)
                for l in range(_q):
                    if _missing_mask[l] == wp.uint8(1):
                        dist_fraction = compute_dtype(0.5)
                        c_l = wp.vec3(_c_float[0, l], _c_float[1, l], _c_float[2, l])
                        c_mag = wp.length(c_l)
                        actual_distance = dist_fraction * c_mag# + 0.5 * c_mag
                        c_unit = c_l / c_mag  # Streamlined: vector division
                        projection = -wp.dot(c_unit, normal) * actual_distance
                        projection = wp.max(projection, compute_dtype(0.0)) # Zero-out negative projections
                        # weight = compute_dtype(1.0) / (actual_distance + _epsilon)
                        weight = (projection * projection) / (actual_distance * actual_distance * actual_distance *actual_distance + _epsilon)
                        normal_distance += projection * weight
                        distance_weight_sum += weight
                # Normalize the distance
                if distance_weight_sum > compute_dtype(0.0):
                    normal_distance /= distance_weight_sum
            
            return normal, normal_distance

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
            f_neq = fpop - feq
            PiNeq = momentum_flux.warp_functional(f_neq)
            zero = compute_dtype(0.0)   
            three = compute_dtype(3.0)

            # Compute double dot product Qi:Pi1 (where Pi1 = PiNeq)
            #trace = (PiNeq[0] + PiNeq[3] + PiNeq[5]) / three
            trace = zero

            for l in range(_q):
                QiPi = zero
                for t in range(_nt):
                        if t == 0 or t == 3 or t == 5:
                            QiPi += _qi[l, t] * (PiNeq[t] - trace)
                        else:
                            QiPi += _qi[l, t] * PiNeq[t]

                # assign all populations based on eq 45 of Latt et al (2008)
                # fneq ~ f^1
                fpop1 = compute_dtype(4.5) * _w[l] * QiPi                
                fpop[l] = feq[l] + fpop1
                # if fpop[l] < _epsilon:
                #     fpop[l] = feq[l]
                fpop[l] = wp.max(fpop[l], _epsilon)
            return fpop

        @wp.func
        def regularize_wallModel(
            fpop: Any,
            feq: Any,
            y_plus: Any,
        ):
            """
            Regularizes the distribution functions by adding non-equilibrium contributions based on second moments of fpop.
            """
            # Compute momentum flux of off-equilibrium populations for regularization: Pi^1 = Pi^{neq}
            f_neq = fpop - feq
            PiNeq = momentum_flux.warp_functional(f_neq)
            zero = compute_dtype(0.0)  
            three = compute_dtype(3.0)
            scale = wp.clamp(y_plus / compute_dtype(30.0), compute_dtype(0.0), compute_dtype(1.0))

            # Compute double dot product Qi:Pi1 (where Pi1 = PiNeq)
            #trace = (PiNeq[0] + PiNeq[3] + PiNeq[5]) / three
            trace = zero

            for l in range(_q):
                QiPi = zero
                for t in range(_nt):
                        if t == 0 or t == 3 or t == 5:
                            QiPi += _qi[l, t] * (PiNeq[t] - trace)
                        else:
                            QiPi += _qi[l, t] * PiNeq[t]

                # assign all populations based on eq 45 of Latt et al (2008)
                # fneq ~ f^1
                fpop1 = compute_dtype(4.5) * _w[l] * QiPi                
                fpop[l] = feq[l] + fpop1 * scale
                # if fpop[l] < _epsilon:
                #     fpop[l] = feq[l]
                fpop[l] = wp.max(fpop[l], _epsilon)

            return fpop

        @wp.func
        def regularize_bounceback(
            _missing_mask: Any,
            rho: Any,
            u: Any,
            f_post: Any,
        ):
            """
            Regularizes the distribution functions by adding non-equilibrium contributions based on second moments of fpop.
            """

            # Compute pressure tensor Pi using all f_post-streaming values
            Pi = momentum_flux.warp_functional(f_post)
            epsilon = compute_dtype(1e-7)            
            zero = compute_dtype(0.0)
            scale = compute_dtype(1.0)
            one = compute_dtype(1.0)            
            one_pt_five = compute_dtype(1.5) 
            three = compute_dtype(3.0)
            four_pt_five = compute_dtype(4.5)

            missing_count = zero
            for l in range(_q):
                if _missing_mask[l] == wp.uint8(1):
                    missing_count += one
            scale = one - scale * (missing_count / compute_dtype(_q))             
            
          
            # Remove convective portion of Pi
            Pi[0] -= rho * u[0] * u[0]
            Pi[1] -= rho * u[0] * u[1]
            Pi[2] -= rho * u[0] * u[2]
            Pi[3] -= rho * u[1] * u[1]
            Pi[4] -= rho * u[1] * u[2]
            Pi[5] -= rho * u[2] * u[2]
            

            u_sqr = zero
            for d in range(_d):
                u_sqr += u[d] * u[d]

            # Compute double dot product Qi:Pi1 (where Pi1 = PiNeq)
            for l in range(_q):
                QiPi = zero
                for t in range(_nt):                        
                    if t == 0 or t == 3 or t == 5:
                        QiPi += _qi[l, t] * (Pi[t] - rho / three)
                    else:
                        QiPi += _qi[l, t] * Pi[t]
                
                # Compute c.u
                cu = zero
                for d in range(_d):
                    cu += _c_float[d, l] * u[d]
                
                cu_sq = cu * cu                               
                f_post[l] = _w[l] * rho * (one + three * cu + four_pt_five * cu_sq - one_pt_five * u_sqr) + _w[l] * four_pt_five * QiPi * scale
                f_post[l] = wp.max(epsilon, f_post[l])
                

            return f_post

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
            zero = compute_dtype(0.0)
            one = compute_dtype(1.0)
            three = compute_dtype(3.0)
            four_pt_five = compute_dtype(4.5)
            missing_count = zero
            # Scale on QiPi helps stability at Re1e6+ not required for Re1e5 and below
            for l in range(_q):
                if _missing_mask[l] == wp.uint8(1):
                    missing_count += one
            scale = one - ((one + missing_count) / compute_dtype(_q)) 

            # Compute double dot product Qi:Pi1 (where Pi1 = PiNeq)
            nt = _d * (_d + 1) // 2
            for l in range(_q):
                if _missing_mask[l] == wp.uint8(1):
                    # compute dot product of qi and Pi
                    QiPi = zero
                    for t in range(nt):
                        if t == 0 or t == 3 or t == 5:
                            QiPi += _qi[l, t] * (Pi[t] - rho / three)
                        else:
                            QiPi += _qi[l, t] * Pi[t]

                    # Compute c.u
                    cu = zero
                    for d in range(_d):
                        cu += _c_float[d, l] * u[d]
                    cu *= three

                    # change f_post using the Grad's approximation
                    f_post[l] = rho * _w[l] * (one + cu) + _w[l] * four_pt_five * QiPi * scale

                    f_post[l] = wp.max(_epsilon, f_post[l])
                else:
                    f_post[l] = wp.max(_epsilon, f_post[l])

            return f_post
        
        @wp.func
        def moving_wall_fpop_correction(
            u_wall: Any,
            lattice_direction: Any,
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
                if _c[d, l] == 1:
                    cu += u_wall[d]
                elif _c[d, l] == -1:
                    cu -= u_wall[d]
            cu *= compute_dtype(6.0) * _w[l]
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
        ):
            # A local single-node version of the interpolated bounce-back boundary condition due to Bouzidi for a lattice
            # Boltzmann method simulation.
            # Ref:
            # [1] Yu, D., Mei, R., Shyy, W., 2003. A unified boundary treatment in lattice boltzmann method,
            # in: 41st aerospace sciences meeting and exhibit, p. 953.

            one = compute_dtype(1.0)
            two = compute_dtype(2.0)
            three = compute_dtype(3.0)
            for l in range(_q):
                # If the mask is missing then take the opposite index
                if _missing_mask[l] == wp.uint8(1):
                    # The normalized distance to the mesh or "weights" have been stored in known directions of f_1
                    if needs_mesh_distance:
                        # use weights associated with curved boundaries that are properly stored in f_1.
                        weight = compute_dtype(self.distance_decoder_function(f_1, index, l))                   
                        weight = wp.clamp(weight, compute_dtype(0.001), compute_dtype(1.0))

                        # Use differentiable interpolated BB to find f_missing:
                        f_post[l] = ((one - weight) * f_post[_opp_indices[l]] + weight * (f_pre[l] + f_pre[_opp_indices[l]])) / (one + weight)
                        #f_near = two * weight * f_pre[_opp_indices[l]] + (one - two * weight) * f_post[_opp_indices[l]]
                        #f_far = (one/ (two * weight)) * f_pre[_opp_indices[l]] + ((two * weight - one) / (two * weight)) * f_pre[l]
                        #blend = three * wp.pow(weight, two) - two * wp.pow(weight, three)

                        #f_post[l] = (one - blend)*f_near + blend*f_far
                        
                    else:
                        # Use regular halfway bounceback
                        f_post[l] = f_pre[_opp_indices[l]]

                    if _missing_mask[_opp_indices[l]] == wp.uint8(1):
                        # These are cases where the boundary is sandwiched between 2 solid cells and so both opposite directions are missing.
                        f_post[l] = f_pre[_opp_indices[l]]

                    # Add contribution due to moving_wall to f_missing as is usual in regular Bouzidi BC
                    if needs_moving_wall_treatment:
                        f_post[l] += moving_wall_fpop_correction(u_wall, l)
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
            # Compute density, velocity using all f_post-collision values
            rho, u = macroscopic.warp_functional(f_pre)
            feq = equilibrium.warp_functional(rho, u)

            # Compute equilibrium distribution at the wall
            if needs_moving_wall_treatment:
                feq_wall = equilibrium.warp_functional(rho, u_wall)
            else:
                feq_wall = _f_vec()

            
            # Apply method in Tao et al (2018) [1] to find missing populations at the boundary
            one = compute_dtype(1.0)
            half = compute_dtype(0.5)
            for l in range(_q):
                # If the mask is missing then take the opposite index
                if _missing_mask[l] == wp.uint8(1):
                    # The normalized distance to the mesh or "weights" have been stored in known directions of f_1
                    if needs_mesh_distance:
                        # use weights associated with curved boundaries that are properly stored in f_1.
                        weight = compute_dtype(self.distance_decoder_function(f_1, index, l))                   
                        weight = wp.clamp(weight, compute_dtype(0.001), compute_dtype(1.0))
                    else:
                        weight = half

                    # Use non-equilibrium bounceback to find f_missing:
                    fneq = f_pre[_opp_indices[l]] - feq[_opp_indices[l]]

                    # Compute equilibrium distribution at the wall
                    # Same quadratic equilibrium but accounting for zero velocity (no-slip)
                    if not needs_moving_wall_treatment:
                        feq_wall[l] = _w[l] * rho

                    # Assemble wall population for doing interpolation at the boundary
                    f_wall = feq_wall[l] + fneq
                    f_post[l] = (f_wall + weight * f_pre[l]) / (one + weight)

                    f_post[l] = wp.max(_epsilon, f_post[l])
                else:
                    f_post[l] = wp.max(_epsilon, f_post[l])

            return f_post

        # ============================================================================
        # Wall Model Functions
        # ============================================================================       
        @wp.func
        def solve_musker1(K: Any):
            """
            Solve for u⁺ given K = u_parallel * y / ν
            Refined initial guess for u⁺ using Newton on log-law approx"""
            if K < compute_dtype(1.0):
                u_plus = wp.sqrt(K)
            else:
                one_k = compute_dtype(1.0) / _kappa
                u_plus = one_k * wp.log(K) + _B
                
                for i in range(5):
                    if u_plus <= compute_dtype(0.0):
                        u_plus = compute_dtype(0.01)
                    g = u_plus - _B - one_k * wp.log(K / u_plus)
                    g_prime = compute_dtype(1.0) + one_k / u_plus
                    delta = g / g_prime
                    u_plus = u_plus - delta    
            # Iteratively refine initial guess (still explicit, just improving starting point)
            for _ in range(3):
                y_plus, dy_du = musker_y_from_u(u_plus)  # ← Now using correct function!
                if y_plus > _epsilon:
                    u_plus = K / y_plus  # Direct solution: u⁺ = K / y⁺
                   # wp.printf(" y+: %f  u+: %f", y_plus, u_plus_init)           
            
            return wp.clamp(u_plus, compute_dtype(0.01), compute_dtype(50.0)), dy_du

        
        @wp.func
        def musker_y_from_u(u_plus: Any) -> Any:
            """Numerical inverse Musker: Solve y+ from u+ via Newton with robust init/clamps"""
            
            
            # Better initial guess: Always use log-law, adjusted for low u+
            if u_plus < compute_dtype(5.0):
                y_guess = u_plus * compute_dtype(0.8)  # Slight under for viscous to aid convergence
            else:
                y_guess = wp.exp(_kappa * (u_plus - _B)) * compute_dtype(0.5)  # Under-guess log to avoid overshoot
            
            tol = compute_dtype(1e-6)
            max_iter = 50  # Increased for high u+
            converged = wp.bool(False)
            
            for i in range(max_iter):
                # Exact Musker u+ calc
                term1 = compute_dtype(5.424) * wp.atan((compute_dtype(2.0) * y_guess - compute_dtype(8.15)) / compute_dtype(16.7))
                denom = y_guess * y_guess - compute_dtype(8.15) * y_guess + compute_dtype(86.0)
                denom = wp.max(denom, compute_dtype(1e-6))  # Safer clamp >0
                term2 = wp.log((y_guess + compute_dtype(10.6))**compute_dtype(9.6) / denom) / wp.log(compute_dtype(10.0))  # Use natural log equiv for stability
                u_calc = term1 + term2 - compute_dtype(3.52)
                
                # Derivative du+/dy+ (improved approx to avoid tiny/neg)
                dterm1 = compute_dtype(5.424) * compute_dtype(2.0) / (compute_dtype(1.0) + ((compute_dtype(2.0) * y_guess - compute_dtype(8.15)) / compute_dtype(16.7))**compute_dtype(2.0))
                dterm2 = (compute_dtype(9.6) / (y_guess + compute_dtype(10.6))) - (compute_dtype(2.0) * y_guess - compute_dtype(8.15)) / denom
                du_dy = dterm1 + dterm2 / wp.log(compute_dtype(10.0))
                du_dy = wp.max(du_dy, compute_dtype(1e-6))  # Clamp to prevent div0 or neg
                
                # Update with damping to prevent explosion
                dy = (u_plus - u_calc) / du_dy
                dy = wp.clamp(dy, -y_guess * compute_dtype(0.5), y_guess * compute_dtype(2.0))  # Limit step size
                y_guess += dy
                y_guess = wp.max(y_guess, compute_dtype(1e-3))  # Clamp low
                
                if wp.abs(dy) < tol * wp.max(compute_dtype(1.0), y_guess):
                    converged = wp.bool(True)
                    break
            
            # Fallback to log-law if not converged (rare now)
            if not converged:
                y_guess = wp.exp(_kappa * (u_plus - _B))
            
            # Recompute du+/dy+ at final y_guess for accurate derivative
            term1 = compute_dtype(5.424) * wp.atan((compute_dtype(2.0) * y_guess - compute_dtype(8.15)) / compute_dtype(16.7))
            denom = y_guess * y_guess - compute_dtype(8.15) * y_guess + compute_dtype(86.0)
            denom = wp.max(denom, compute_dtype(1e-6))
            term2 = wp.log((y_guess + compute_dtype(10.6))**compute_dtype(9.6) / denom) / wp.log(compute_dtype(10.0))
            # u_calc not needed
            
            dterm1 = compute_dtype(5.424) * compute_dtype(2.0) / (compute_dtype(1.0) + ((compute_dtype(2.0) * y_guess - compute_dtype(8.15)) / compute_dtype(16.7))**compute_dtype(2.0))
            dterm2 = (compute_dtype(9.6) / (y_guess + compute_dtype(10.6))) - (compute_dtype(2.0) * y_guess - compute_dtype(8.15)) / denom
            du_dy = dterm1 + dterm2 / wp.log(compute_dtype(10.0))
            du_dy = wp.max(du_dy, compute_dtype(1e-6))
            
            dy_du = compute_dtype(1.0) / du_dy  # dy+/du+
            
            return y_guess, dy_du
            
        @wp.func
        def apg_correction_factor(
            y_distance: Any,
            rho: Any,
            u_parallel_mag: Any,
            u_tau: Any,
            nu: Any,
            u_est: Any,
            normal: Any,
        ):
            """
            Compute pressure gradient correction factor for wall model.
            
            Uses the Clauser parameter β = (δ*/τ_w) × (dp/dx) as basis,
            but approximated from local quantities.
            
            Returns factor in range [0.3, 1.3]:
            - 1.0 = zero pressure gradient (equilibrium log-law valid)
            - <1.0 = adverse pressure gradient (reduce wall model correction)
            - >1.0 = favorable pressure gradient (increase wall model correction)
            """
            
            # Compute y+ for regime detection
            y_plus = y_distance * u_tau / nu
            
            # =========================================
            # Method: Use velocity profile shape factor
            # =========================================
            # In equilibrium, u/u_τ = f(y⁺) with specific shape
            # In APG, velocity profile is "fuller" near wall (higher u⁺ at same y⁺)
            # In FPG, velocity profile is "thinner"
            #
            # Compare measured u⁺ to expected equilibrium u⁺
            
            u_plus_measured = u_parallel_mag / wp.max(u_tau, _epsilon)
            u_plus_equilibrium = musker_profile(y_plus)
            
            # Shape deviation: positive if measured > equilibrium (FPG-like)
            #                  negative if measured < equilibrium (APG-like)
            shape_deviation = (u_plus_measured - u_plus_equilibrium) / wp.max(u_plus_equilibrium, compute_dtype(1.0))
            
            # =========================================
            # Additional indicator: normal velocity
            # =========================================
            # Flow toward wall (v·n > 0, where n points into fluid) suggests FPG
            # Flow away from wall (v·n < 0) suggests APG/separation
            
            v_normal = wp.dot(u_est, normal)
            v_normal_normalized = v_normal / wp.max(u_parallel_mag, _epsilon)
            
            # Combine indicators (shape deviation is primary)
            # Scale down normal velocity contribution to avoid noise issues
            indicator = shape_deviation + compute_dtype(0.1) * v_normal_normalized
            
            # =========================================
            # Convert to correction factor
            # =========================================
            # Use smooth mapping to bounded range
            
            # Sensitivity parameter (how much the factor changes per unit indicator)
            sensitivity = compute_dtype(0.5)
            
            # Compute factor using tanh for smooth saturation
            factor = compute_dtype(1.0) + sensitivity * wp.tanh(indicator)
            
            # Clamp to reasonable range
            factor = wp.clamp(factor, compute_dtype(0.3), compute_dtype(1.3))
            
            # =========================================
            # Reduce correction at low y+ (viscous layer)
            # =========================================
            # APG/FPG effects are less pronounced in viscous sublayer
            
            y_plus_threshold = compute_dtype(30.0)
            viscous_damping = wp.clamp(y_plus / y_plus_threshold, compute_dtype(0.0), compute_dtype(1.0))
            
            # Blend factor toward 1.0 as we approach viscous layer
            factor = compute_dtype(1.0) + (factor - compute_dtype(1.0)) * viscous_damping
            
            return factor

        @wp.func
        def musker_profile(y_plus: Any) -> Any:
            """
            Modernized Musker (1979) / Nagib-Chauhan-Monkewitz (2007) form for u+ = f(y+)
            Valid from y+ = 0 to y+ >1e6, with correct asymptotic to log-law.
            """
            kappa = _kappa
            B = _B
            a = compute_dtype(1.0) / kappa + compute_dtype(2.0)  # =4.44 for kappa=0.41
            alpha = (a - compute_dtype(1.0)/kappa) * compute_dtype(0.5)  # =1.0 for correct slope
            beta = wp.sqrt(compute_dtype(2.0) * a * alpha - alpha * alpha)

            term1 = y_plus / a
            
            arg_log = wp.max(a * y_plus / alpha, compute_dtype(1.0) + _epsilon)
            term2 = (alpha / kappa) * wp.log(arg_log)
            
            arg_arctan = (a * y_plus - alpha) / beta
            term3 = (beta / (compute_dtype(2.0) * kappa)) * wp.atan(arg_arctan)
            
            # Constant to match B: cancels extra terms in asymptotic (1/kappa) ln(y+) + B
            const = B - (alpha / kappa) * wp.log(a / alpha) - (beta / (compute_dtype(2.0) * kappa)) * (_pi / compute_dtype(2.0))
            
            u = term1 + term2 + term3 + const
            return wp.max(u, compute_dtype(0.0))  # Clamp for robustness at low y+

        @wp.func
        def solve_musker(K: Any) -> Any:
            """
            Solve u+ such that u+ = musker_profile(K / u+)
            Bisection with fixed bracket and condition for stable convergence.
            """
            if K < _epsilon:
                return wp.sqrt(K)  # Viscous fallback
            
            u_lo = _epsilon
            u_hi = K * compute_dtype(2.0)  # Wider for small K robustness

            kappa = _kappa
            B = _B
            if K > compute_dtype(30.0):
                u_log = (wp.log(K) / kappa) + B
                u_hi = wp.max(u_hi, u_log * compute_dtype(2.5))  # Optional tighten, consistent kappa
            
            tol_rel = compute_dtype(1e-4) 
            for i in range(20):
                if (u_hi - u_lo) < (tol_rel * u_hi + _epsilon):
                    break
                u_mid = (u_lo + u_hi) * compute_dtype(0.5)
                if u_mid < _epsilon:
                    u_mid = _epsilon
                y_mid = K / u_mid
                u_computed = musker_profile(y_mid)
                if u_computed > u_mid:
                    u_lo = u_mid  # Reversed for correct convergence
                else:
                    u_hi = u_mid

            u_final = (u_lo + u_hi) * compute_dtype(0.5)
            return wp.max(u_final, _epsilon)

        
        # Reichardt WM
        @wp.func
        def reichardt_profile(y_plus: Any) -> Any:
            """
            Reichardt (1951) velocity profile - explicit u+(y+)
            Correctly captures: u+ = y+ as y+ → 0, log-law as y+ → ∞
            """
            kappa = _kappa  # 0.41
            C = compute_dtype(11.0)  # Transition parameter
            # Logarithmic part (transitions correctly)
            log_term = (compute_dtype(1.0) / kappa) * wp.log(
                compute_dtype(1.0) + kappa * y_plus
            )
            # Damping function that enforces u+ = y+ near wall
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
            Analytical derivative du+/dy+ of Reichardt profile
            """
            kappa = _kappa
            C = compute_dtype(11.0)
            C3 = compute_dtype(3.0)
            # d/dy+ of log(1 + κy+)/κ = 1/(1 + κy+)
            d_log = compute_dtype(1.0) / (compute_dtype(1.0) + kappa * y_plus)
            # d/dy+ of damping terms
            # d/dy+ of (1 - exp(-y+/C)) = (1/C) exp(-y+/C)
            # d/dy+ of (y+/C) exp(-y+/C3) = (1/C) exp(-y+/C3) - (y+/C/C3) exp(-y+/C3)
            exp_C = wp.exp(-y_plus / C)
            exp_C3 = wp.exp(-y_plus / C3)
            d_damping = compute_dtype(8.5) * (
                exp_C / C
                - (compute_dtype(1.0) / C) * exp_C3
                + (y_plus / (C * C3)) * exp_C3
            )
            return wp.max(d_log + d_damping, _epsilon)
        
        @wp.func
        def solve_wall_function(u_parallel_mag: Any, y_distance: Any, nu: Any) -> Any:
            """
            Solve for u_tau given u_parallel_mag, y_distance, and nu using the Reichardt profile with refinement.
            """
            # Compute K = y+ * u+ ≈ u_parallel_mag * y_distance / nu
            K = u_parallel_mag * y_distance / nu
            K = wp.max(K, _epsilon)

            # Initial solve for u+ using the profile iteration
            if K < _epsilon:
                u_plus = wp.sqrt(K)  # Viscous limit: u+ = y+ → K = u+²
            else:
                # Initial guess
                if K < compute_dtype(25.0):
                    # Viscous regime: K ≈ u+² → u+ ≈ √K
                    u_plus = wp.sqrt(K)
                else:
                    # Log regime: use approximate inverse
                    # K = y+ * u+ where u+ ≈ (1/κ)ln(y+) + B
                    # Approximate: u+ ≈ √(K/ln(K)) as starting point
                    u_plus = wp.sqrt(K / wp.max(wp.log(K), compute_dtype(1.0)))

                # Fixed point iteration: solve u+ = profile(K/u+)
                for _ in range(20):
                    y_plus = K / wp.max(u_plus, _epsilon)
                    # Use Reichardt profile
                    u_computed = reichardt_profile(y_plus)
                    # Secant-like update (stable)
                    u_plus_new = (u_plus + u_computed) * compute_dtype(0.5)
                    if wp.abs(u_plus_new - u_plus) < compute_dtype(1e-6) * u_plus:
                        break
                    u_plus = u_plus_new

            u_plus = wp.max(u_plus, _epsilon)
            # Initial u_tau
            u_tau = u_parallel_mag / u_plus

            # Refine u_tau to ensure consistency
            tolerance = compute_dtype(1e-4)
            max_iter = 5  # Usually converges in 3-4 iterations
            for iter in range(max_iter):
                # Compute y+ with current u_tau
                y_plus = y_distance * u_tau / nu
                # Get u+ from Reichardt profile
                u_plus = reichardt_profile(y_plus)
                # Velocity implied by this u_tau and u+
                u_implied = u_tau * u_plus
                # Update u_tau to match observed velocity
                u_tau_old = u_tau
                if u_implied > _epsilon:
                    u_tau = u_tau * (u_parallel_mag / u_implied)
                # Check convergence
                if wp.abs(u_tau - u_tau_old) < tolerance * wp.max(u_tau, _epsilon):
                    break

            return u_tau

        @wp.func
        def p_grad_proxy_density(
            f_pop: Any,   # distributions at the node (typically f_pre)
            rho: Any,     # local density
            u: Any,       # local velocity vector (same as used for macroscopic)
            normal: Any,  # wall-normal unit vector (points into fluid)
            s_hat: Any,   # wall-tangential / streamwise unit vector
            nu:Any,
        ):
            """
            Compute τ(h)/ρ along the wall-tangent direction s_hat at the first
            off-wall cell using the LB momentum flux (second moment).

            τ(h)/ρ = nᵀ P s_hat, with P the (deviatoric + isotropic) stress tensor
            after removing the convective part ρ u uᵀ. The isotropic part drops
            out because n · s_hat = 0.
            """

           
            feq = equilibrium.warp_functional(rho, u)
            f_neq = f_pop - feq
            # Second moment from LB populations
            Pi = momentum_flux.warp_functional(f_neq)

            # Remove convective part ρ u uᵀ to get stress-like tensor
            if wp.static(_d == 3):
                # Pi layout (3D): [Pxx, Pxy, Pxz, Pyy, Pyz, Pzz]
                Pxx = Pi[0]# - rho * u[0] * u[0]
                Pxy = Pi[1]# - rho * u[0] * u[1]
                Pxz = Pi[2]# - rho * u[0] * u[2]
                Pyy = Pi[3]# - rho * u[1] * u[1]
                Pyz = Pi[4]# - rho * u[1] * u[2]
                Pzz = Pi[5]# - rho * u[2] * u[2]

                # P · s_hat
                Psx = Pxx * s_hat[0] + Pxy * s_hat[1] + Pxz * s_hat[2]
                Psy = Pxy * s_hat[0] + Pyy * s_hat[1] + Pyz * s_hat[2]
                Psz = Pxz * s_hat[0] + Pyz * s_hat[1] + Pzz * s_hat[2]

                # traction τ_s = nᵀ (P · s_hat)
                traction = (
                    normal[0] * Psx +
                    normal[1] * Psy +
                    normal[2] * Psz
                )

            else:
                # Pi layout (2D): [Pxx, Pxy, Pyy]  (if ever used in 2D)
                Pxx = Pi[0] - rho * u[0] * u[0]
                Pxy = Pi[1] - rho * u[0] * u[1]
                Pyy = Pi[2] - rho * u[1] * u[1]

                Psx = Pxx * s_hat[0] + Pxy * s_hat[1]
                Psy = Pxy * s_hat[0] + Pyy * s_hat[1]

                traction = normal[0] * Psx + normal[1] * Psy

            # τ/ρ (can be signed; use directly in TBLE)
            #tau_over_rho = traction / wp.max(rho, _epsilon)
            tau_relax = compute_dtype(0.5) + nu / wp.max(_cs2, _epsilon)
            tau_relax = wp.max(tau_relax, compute_dtype(0.500001))  # avoid omega=2 singular

            omega = compute_dtype(1.0) / tau_relax

            # conversion factor (BGK-style): (1 - omega/2)
            conv = compute_dtype(1.0) - compute_dtype(0.5) * omega

            # keep it sane
            conv = wp.clamp(conv, compute_dtype(0.0), compute_dtype(1.0))

            # tau/ρ in the same "units family" as u_tau^2
            tau_over_rho = (traction * conv) / wp.max(rho, _epsilon)
            return tau_over_rho


        @wp.func
        def compute_wall_modeled_velocity(
            index: Any,
            _missing_mask: Any,
            f_0: Any,
            f_1: Any,
            f_pre: Any,
            f_post: Any,
            u_wall: Any,
            nu: Any, 
        ):
            
            # Get wall geometry
            normal, y_distance = get_normal_and_distance(_missing_mask, f_1, index)
            
            # Macroscopic state
            rho, u_est = macroscopic.warp_functional(f_pre)
            
            # # Decompose velocity
            u_normal = normal * wp.dot(u_est, normal)
            u_parallel = u_est - u_normal
            u_parallel_mag = wp.length(u_parallel)
            if u_parallel_mag > _epsilon: 
                u_parallel_unit = u_parallel / u_parallel_mag                
            else: 
                u_parallel_mag = compute_dtype(0.0)               
                u_parallel_unit = _u_vec(0.0, 0.0, 0.0)  
                   
            # Solve u_tau from Reichardt
            u_tau = solve_wall_function(u_parallel_mag, y_distance, nu)
            # Get approximate y+ (based on f_pre)
            y_plus = y_distance * u_tau / nu
            y_plusR =y_plus
            u_tauR = u_tau
                    
            # APG/FPG assessment                        
            tau_h_over_rho  = p_grad_proxy_density(f_pre, rho, u_est, normal, u_parallel_unit, nu) 
            sign_tau = wp.sign(tau_h_over_rho)
            if sign_tau == compute_dtype(0.0):
                sign_tau = compute_dtype(1.0)
            tau_w_over_rho = sign_tau * u_tauR * u_tauR  # Wall shear from Reichardt (baseline)  
            # TBLE-style local pressure-gradient proxy
            p_grad_proxy = (tau_h_over_rho - tau_w_over_rho) / wp.max(y_distance, _epsilon)
           
            p_grad_raw = p_grad_proxy
            betaR = (p_grad_proxy *y_distance) / (u_tauR*u_tauR +_epsilon)
            beta_clamped = wp.clamp(betaR, compute_dtype(-10.0), compute_dtype(10.0))
            p_grad_tble = beta_clamped * (u_tauR*u_tauR) / max(y_distance, _epsilon)
            #p_grad_proxy = wp.clamp(p_grad_proxy, compute_dtype(-5e-3), compute_dtype(5e-3)) #was 5e-4
            if u_parallel_mag < compute_dtype(5e-3):
                p_grad_tble = compute_dtype(0.0)
            # Solve for dudy based on TBLE
            
            u_tau, dudy = solve_tble(y_distance, nu, rho, u_parallel_mag, p_grad_tble)
            y_plus = u_tau * y_distance / nu 
            du_plus_dy_plus = reichardt_derivative(y_plus)
            velocity_gradient = (u_tau * u_tau / nu) * du_plus_dy_plus  

            # Solve for Slip Mag based on TBLE        
            u_slip_mag = u_parallel_mag - dudy * y_distance
            u_slip_mag = wp.clamp(u_slip_mag, compute_dtype(-0.01) * u_parallel_mag, compute_dtype(1.5) * u_parallel_mag)
            u_slip = u_parallel_unit * u_slip_mag
            

            # Effective wall velocity
            u_wall_effective_t = u_wall + u_slip

            # y+ gating – don’t use TBLE in very low y+
            y0 = compute_dtype(15.0)   
            y1 = compute_dtype(20.0)  
            w_y = (y_plusR - y0) / (y1 - y0)
            w_y = wp.clamp(w_y, compute_dtype(0.0), compute_dtype(1.0))
            # # Blend between No-slip and TBLE based on 
            u_wall_effective = ((compute_dtype(1.0) - w_y) * u_wall) + (w_y * u_wall_effective_t)                
            
            #wp.printf("uParallel, %f, testVel, %f, \n",
            #            u_parallel_mag, test_vel)
            # # Debug print
            #if (normal[2] > compute_dtype(0.5)):
            #    wp.printf("NormalX, %f, NormalY, %f, NormalZ, %f, uParallel, %f, y_dist, %f, utauR, %f, y_plusR, %f, tau_h, %f, tau_w, %f, P_grad_raw, %f, P_grad_tble, %f, u_tauT, %f, betaR, %f, yplusT, %f, dudy, %f, dupdyp, %f, vel_grad, %f, blend, %f, uslip, %f\n",
              #        normal[0], normal[1], normal[2],  u_parallel_mag, y_distance, u_tauR, y_plusR, tau_h_over_rho, tau_w_over_rho, p_grad_raw, p_grad_tble, u_tau, betaR, y_plus, dudy, du_plus_dy_plus,velocity_gradient, w_y, u_slip_mag)
    
            # # Enforce zero normal component        
            u_wall_effective = u_wall_effective - normal * wp.dot(u_wall_effective, normal)

            return u_wall_effective
        
        @wp.func
        def solve_tble(
            y_distance: Any,  # Wall-normal distance h (e.g., boundary layer thickness or outer scale)
            nu: Any,          # Kinematic viscosity
            rho: Any,         # Density (unused, kept for interface)
            u_target: Any,    # Target parallel velocity at y_distance (u_parallel_mag)
            p_grad: Any,      # Local pressure gradient / rho
        ):
            """
            Wall-model gradient solver returning dudy_at_ydist.

            Features:
            - Laminar / stagnation cutoff based on local Re_y to avoid PG-driven
                blow-ups near stagnation.
            - Separation check in high-Re attached regions.
            - Equilibrium turbulent BL integration with PG-dependent damping and
                outer mixing-length cap.
            - Quadratic local dudy with a cap against a laminar estimate.
            """

            zero = compute_dtype(0.0)
            ydist = y_distance

            # Precompute laminar gradient used in several places
            dudy_lam = u_target / wp.max(ydist, _epsilon)

         

            u_tau_lo = _epsilon
            u_tau_hi = wp.max(u_target * compute_dtype(2.0), _epsilon)

            # 1.1 quick separation check using u_tau_hi
            tau_wall_hi  = u_tau_hi * u_tau_hi          # τ/ρ at the wall
            tau_outer_hi = tau_wall_hi + p_grad * ydist # τ/ρ at y=ydist

      

            # ------------------------------------------------------------------
            # 2. ATTACHED HIGH-Re equilibrium BL solve
            # ------------------------------------------------------------------
            N_steps = wp.int32(50)
            alpha   = compute_dtype(2.0)
            tol     = compute_dtype(1e-4)

            for iter in range(30):
                u_tau_mid = (u_tau_lo + u_tau_hi) * compute_dtype(0.5)
                if u_tau_mid < _epsilon:
                    u_tau_mid = _epsilon

                # Pressure-gradient parameter
                p_plus = (nu / (u_tau_mid * u_tau_mid * u_tau_mid)) * p_grad

                # PG-dependent van Driest; clamp A+ to avoid zeros/negatives
                A_plus_factor = wp.max(compute_dtype(1.0) - compute_dtype(11.8) * p_plus, compute_dtype(0.005)) if p_plus > compute_dtype(0.0) else wp.max(compute_dtype(1.0) + compute_dtype(11.8) * wp.abs(p_plus), compute_dtype(0.005))  # Symmetric for FPG/APG
                A_plus = _A_plus * wp.sqrt(compute_dtype(1.0) / A_plus_factor)  # Now reduces for APG, increases for FPG
                #A_plus_factor = wp.max(compute_dtype(1.0) - compute_dtype(11.8) * p_plus, compute_dtype(0.1))
                #A_plus = _A_plus * wp.pow(A_plus_factor, compute_dtype(-0.5))

                u_computed = zero
                y_prev     = zero

                for step in range(1, N_steps + 1):
                    frac = compute_dtype(step) / compute_dtype(N_steps)
                    y    = ydist * wp.pow(frac, alpha)
                    dy   = y - y_prev

                    y_plus = y * u_tau_mid / nu
                    damp   = compute_dtype(1.0) - wp.exp(-y_plus / A_plus)
                    l_mix_uncapped = _kappa * y * damp
                    l_mix          = wp.min(l_mix_uncapped, compute_dtype(0.085) * ydist)

                    dudy_est = u_computed / wp.max(y, _epsilon)
                    nu_t     = l_mix * l_mix * wp.max(wp.abs(dudy_est), _epsilon)

                    tau_over_rho = (u_tau_mid * u_tau_mid) + p_grad * y
                    sign         = wp.sign(tau_over_rho)
                    tau_mag      = wp.max(
                        wp.abs(tau_over_rho),
                        (u_tau_mid * u_tau_mid) * compute_dtype(1e-3)
                    )
                    tau_over_rho = sign * tau_mag

                    dudy       = tau_over_rho / (nu + nu_t + _epsilon)
                    u_computed += dudy * dy

                    y_prev = y

                if u_computed < u_target:
                    u_tau_lo = u_tau_mid
                else:
                    u_tau_hi = u_tau_mid

                if (u_tau_hi - u_tau_lo) < (tol * u_tau_hi + _epsilon):
                    break

            u_tau = (u_tau_lo + u_tau_hi) * compute_dtype(0.5)
            u_tau = wp.max(u_tau, _epsilon)

            # 2.1 final PG-dependent A+
            p_plus = (nu / (u_tau * u_tau * u_tau)) * p_grad
            A_plus_factor = wp.max(compute_dtype(1.0) - compute_dtype(11.8) * p_plus, compute_dtype(0.005)) if p_plus > compute_dtype(0.0) else wp.max(compute_dtype(1.0) + compute_dtype(11.8) * wp.abs(p_plus), compute_dtype(0.005))  # Symmetric for FPG/APG
            A_plus = _A_plus * wp.sqrt(compute_dtype(1.0) / A_plus_factor)  # Now reduces for APG, increases for FPG
            #A_plus_factor = wp.max(compute_dtype(1.0) - compute_dtype(11.8) * p_plus, compute_dtype(0.1))
            #A_plus = _A_plus * wp.pow(A_plus_factor, compute_dtype(-0.5))

            # 2.2 local dudy at ydist from quadratic stress balance
            y_plus = ydist * u_tau / nu
            damp   = compute_dtype(1.0) - wp.exp(-y_plus / A_plus)

            l_mix_uncapped = _kappa * ydist * damp
            l_mix          = wp.min(l_mix_uncapped, compute_dtype(0.085) * ydist)

            # total shear at y = ydist
            tau_over_rho = (u_tau * u_tau) + p_grad * ydist

            # If total shear goes negative, you're effectively in separation / non-eq:
            # return a safe baseline gradient (prevents sign-flip ringing).
            if tau_over_rho <= zero:
                y_plus_reich = ydist * u_tau / nu
                dplus_reich  = reichardt_derivative(y_plus_reich)
                dudy_reich   = (u_tau * u_tau / nu) * dplus_reich
                return u_tau, dudy_reich

            tau_mag = wp.max(wp.abs(tau_over_rho), (u_tau * u_tau) * compute_dtype(1e-3) )

            # Baselines (smooth)
            dudy_lam   = u_target / wp.max(ydist, _epsilon)
            y_plus_r   = ydist * u_tau / nu
            dplus_r    = reichardt_derivative(y_plus_r)
            dudy_reich = (u_tau * u_tau / nu) * dplus_r

            # Seed for fixed-point iteration: smooth & positive
            S = wp.max(wp.abs(dudy_reich), wp.abs(dudy_lam))
            S = wp.max(S, _epsilon)

            # Under-relaxed fixed-point iterations (2 is usually enough)
            omega = compute_dtype(0.55)  # 0.2–0.5 recommended
            for _ in range(3):
                nu_t = l_mix * l_mix * wp.max(S, compute_dtype(1e-8))
                S_new = tau_mag / (nu + nu_t + _epsilon)
                S = (compute_dtype(1.0) - omega) * S + omega * S_new

            # Optional: ratio limiter relative to Reichardt to prevent "too free-slip pockets"
            base = wp.max(wp.abs(dudy_reich), compute_dtype(1e-8))
            r = S / base

            r_min = compute_dtype(0.0001)  # allow up to 20x slip vs Reichardt
            r_max = compute_dtype(1.0)  # don't exceed Reichardt too much
            r = wp.clamp(r, r_min, r_max)

            dudy_smooth = r * base

            # Enforce non-negative gradient since u_target is a magnitude and slip is along u_parallel_unit
            dudy_smooth = wp.max(dudy_smooth, zero)

            return u_tau, dudy_smooth


        @wp.func
        def compute_profile_u(
            y: Any,
            u_tau: Any,
            nu: Any,
            p_grad: Any,
        ):
            zero = compute_dtype(0.0)
            N_steps = wp.int32(30)
            alpha = compute_dtype(2.0)
            u_computed = zero
            y_prev = zero
            for step in range(1, N_steps + 1):
                frac = compute_dtype(step) / compute_dtype(N_steps)
                y_current = y * wp.pow(frac, alpha)
                dy = y_current - y_prev
                y_plus = y_current * u_tau / nu
                damp = compute_dtype(1.0) - wp.exp(-y_plus / _A_plus)
                l_mix = _kappa * y_current * damp
                dudy_est = u_computed / wp.max(y_current + _epsilon, _epsilon)
                nu_t = l_mix * l_mix * wp.max(wp.abs(dudy_est), _epsilon)
                tau_over_rho = (u_tau * u_tau) + p_grad * y_current
                sign    = wp.sign(tau_over_rho)
                tau_mag = wp.max(wp.abs(tau_over_rho), (u_tau*u_tau)*compute_dtype(1e-3))
                tau_over_rho = sign * tau_mag
                dudy = tau_over_rho / (nu + nu_t + _epsilon)
                u_computed += dudy * dy
                y_prev = y_current
            return u_computed
        
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

        # Store all functions as class attributes
        self.get_bc_thread_data = get_bc_thread_data
        self.get_bc_fsum = get_bc_fsum
        self.get_normal_vectors = get_normal_vectors
        self.bounceback_nonequilibrium = bounceback_nonequilibrium
        self.regularize_fpop = regularize_fpop
        self.regularize_bounceback = regularize_bounceback
        self.grads_approximate_fpop = grads_approximate_fpop
        self.moving_wall_fpop_correction = moving_wall_fpop_correction
        self.interpolated_bounceback = interpolated_bounceback
        self.interpolated_nonequilibrium_bounceback = interpolated_nonequilibrium_bounceback
        self.neon_get_bc_thread_data = neon_get_bc_thread_data
        self.neon_index_to_warp = neon_index_to_warp
        
        # Wall model functions 
        self.p_grad_proxy_density = p_grad_proxy_density
        self.reichardt_profile = reichardt_profile
        self.reichardt_derivative = reichardt_derivative
        self.solve_wall_function = solve_wall_function
        self.apg_correction_factor = apg_correction_factor
        self.compute_profile_u = compute_profile_u
        self.solve_tble = solve_tble
        self.regularize_wallModel = regularize_wallModel
        self.solve_musker = solve_musker
        self.musker_profile = musker_profile
        self.musker_y_from_u = musker_y_from_u
        self.compute_wall_modeled_velocity = compute_wall_modeled_velocity        
        self.get_normal_and_distance = get_normal_and_distance
        


# Keep all the encoding classes unchanged
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
        """
        Constructs the Neon container for encoding auxilary data recovery.
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
        Constructs the Neon container for encoding auxilary data recovery.
        """

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
                        # IMPORTANT NOTE:
                        # It is assumed in XLB that the user_defined_functional in multi-res simulations is defined in terms of the indices at the finest level.
                        # This assumption enables handling of BCs whose indices span multiple levels
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
        grid = bc_mask.get_grid()
        for level in range(grid.num_levels):
            c = self.neon_container(f_1, bc_mask, missing_mask, level)
            c.run(stream, container_runtime=neon.Container.ContainerRuntime.neon)
        return f_1