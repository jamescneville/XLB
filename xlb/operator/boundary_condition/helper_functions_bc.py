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

        # Define the operator needed for computing equilibrium
        equilibrium = QuadraticEquilibrium(velocity_set, precision_policy, compute_backend)

        # Define the operator needed for computing macroscopic variables
        macroscopic = Macroscopic(velocity_set, precision_policy, compute_backend)

        # Define the operator needed for computing the momentum flux
        momentum_flux = MomentumFlux(velocity_set, precision_policy, compute_backend)

        # Wall model constants
        _kappa = compute_dtype(0.41)  # von Karman constant
        _B = compute_dtype(8.2)       # Log-law constant
        _A_plus = compute_dtype(50.0)  # van Driest damping constant
        _cs2 = compute_dtype(self.velocity_set.cs2)

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
        def get_known_fsum(fpop: Any, _missing_mask: Any):
            fsum = compute_dtype(0.0)
            for l in range(_q):
                if _missing_mask[l] == wp.uint8(0):
                    fsum += fpop[l]
            return fsum

        @wp.func
        def estimate_from_known(
            fpop: Any, 
            _missing_mask: Any,
            normal: Any,
        ):
            zero = compute_dtype(0.0)
            
            # Density estimation with direction-weighted mirroring
            rho_known = zero
            rho_mirrored = zero
            
            # Initialize momentum vector (only from known populations)
            momentum = _u_vec(zero, zero, zero)
            
            for l in range(_q):
                if _missing_mask[l] == wp.uint8(0):  # Known
                    rho_known += fpop[l]
                    # Add to momentum from known directions only
                    for d in range(_d):
                        momentum[d] += _c_float[d, l] * fpop[l]
                else:  # Missing
                    opp = _opp_indices[l]
                    if _missing_mask[opp] == wp.uint8(0):  # Opposite is known
                        # Weight based on how aligned this direction is with normal
                        c_dot_n = zero
                        for d in range(_d):
                            c_dot_n += _c_float[d, l] * normal[d]
                        weight = wp.abs(c_dot_n)
                        
                        # Mirror for density only
                        rho_mirrored += weight * fpop[opp]
                        
                        # DON'T add momentum from missing directions - this causes mass leakage
                        # Instead, we could optionally add a reflected momentum contribution
                        # but for stability, it's better to only use known populations
            
            _rho = wp.max(rho_known + rho_mirrored, compute_dtype(1e-6))
            
            # Compute u from momentum / rho (only from known directions)
            _u = _u_vec(zero, zero, zero)
            if _rho > zero:
                for d in range(_d):
                    _u[d] = momentum[d] / _rho
                    
                # Optional: Apply normal velocity correction to ensure no penetration
                # This helps with stability near walls
                u_dot_n = _u[0]*normal[0] + _u[1]*normal[1] + _u[2]*normal[2]
                if u_dot_n < zero:  # Flow into wall
                    # Remove normal component pointing into wall
                    for d in range(_d):
                        _u[d] -= u_dot_n * normal[d]
            
            return _rho, _u
        
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
        def get_avg_normal_vectors(
            _missing_mask: Any,
        ):
            normal = _u_vec(0.0, 0.0, 0.0)
            weight_sum = compute_dtype(0.0)            
         
            for l in range(_q):
                if _missing_mask[l] == wp.uint8(1):
                    c_l = wp.vec3(_c_float[0, l], _c_float[1, l], _c_float[2, l])
                    c_mag = wp.length(c_l)                    
               
                    c_unit = wp.vec3(c_l[0] / c_mag, c_l[1] / c_mag, c_l[2] / c_mag)
                    
                
                    # Weighted by lattice weight (prioritizes orthogonal directions)
                    weight = _w[l] / c_mag
                    
                    normal[0] += c_unit[0] * weight
                    normal[1] += c_unit[1] * weight
                    normal[2] += c_unit[2] * weight
                    weight_sum += weight
            
            if weight_sum > compute_dtype(0.0):
                normal[0] /= weight_sum
                normal[1] /= weight_sum
                normal[2] /= weight_sum
            
            # Unit normalize
            mag = wp.sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2])
            if mag > compute_dtype(0.0):
                normal[0] /= mag
                normal[1] /= mag
                normal[2] /= mag
            
            return -normal  # Points into fluid

        @wp.func
        def get_wall_distance(
            _missing_mask: Any,
            f_1: Any,
            index: Any,
        ):
            """
            Compute wall distance using weighted average of all missing directions.
            This provides more accurate distance estimation for curved boundaries.
            """
            zero = compute_dtype(0.0)
            y_distance = zero
            weight_sum = zero            
            
            # Set to True for even (equal) weighting; False for distance-based (inverse magnitude)
            use_even_weighting = wp.static(False)  # Toggle this to test; recompile after change
            test_dist = compute_dtype(1.0)
            for l in range(_q):
                if _missing_mask[l] == wp.uint8(1):
                    dist = compute_dtype(self.distance_decoder_function(f_1, index, l))  # q_l (0-1 fraction)
                    
                    c_l = wp.vec3(_c_float[0, l], _c_float[1, l], _c_float[2, l])
                    c_mag = wp.length(c_l)                    
                    along_dist = dist * c_mag  # Physical along-link distance
                    proj = wp.max(zero, wp.dot())
                   
                     # Distance-based weighting: Lattice weight / link magnitude
                    weight = _w[l] / c_mag
                    
                    y_distance += along_dist * weight
                    weight_sum += weight
                    test_dist = wp.min(along_dist, test_dist)
            
            # Normalize by total weight
            if weight_sum > compute_dtype(0.0):
                y_distance /= weight_sum
            
            return test_dist
     
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
            epsilon = compute_dtype(1e-8)
            
            # First pass: Compute weighted normal based on actual distances
            for l in range(_q):
                if _missing_mask[l] == wp.uint8(1):
                    # Get distance fraction (0-1) from distance decoder
                    dist_fraction = compute_dtype(self.distance_decoder_function(f_1, index, l))
                    
                    # Get lattice vector
                    c_l = wp.vec3(_c_float[0, l], _c_float[1, l], _c_float[2, l])
                    c_mag = wp.length(c_l)
                    
                    # Actual distance from voxel center to mesh along this direction
                    actual_distance = dist_fraction * c_mag
                    
                    # Unit vector in this direction
                    c_unit = wp.vec3(c_l[0] / c_mag, c_l[1] / c_mag, c_l[2] / c_mag)
                    
                    # Weight by inverse of actual distance (closer surfaces have more influence)
                    # Add small epsilon to avoid division by zero
                    
                    weight = _w[l] * compute_dtype(1.0) / (actual_distance + epsilon)
                    
                    # Accumulate weighted normal vector
                    normal += c_unit * weight
                    weight_sum += weight
            
            # Normalize the weighted normal
            if weight_sum > compute_dtype(0.0):
                normal /= weight_sum
                
            
            # Unit normalize the normal vector
            mag = wp.length(normal)
            if mag > compute_dtype(0.0):
                normal /= mag
            
            # Invert normal to point into fluid
            normal = -normal
            
            # Second pass: Project actual distances onto normal and average
            distance_weight_sum = compute_dtype(0.0)
            
            for l in range(_q):
                if _missing_mask[l] == wp.uint8(1):
                    # Get distance fraction and actual distance again
                    dist_fraction = compute_dtype(self.distance_decoder_function(f_1, index, l))
                    c_l = wp.vec3(_c_float[0, l], _c_float[1, l], _c_float[2, l])
                    c_mag = wp.length(c_l)
                    actual_distance = dist_fraction * c_mag
                    
                    # Unit vector in this direction
                    c_unit = wp.vec3(c_l[0] / c_mag, c_l[1] / c_mag, c_l[2] / c_mag)
                    
                    # Project this distance vector onto the normal
                    # (dot product gives projection length, negative because normal points into fluid)
                    projection = -wp.dot(c_unit, normal) * actual_distance
                    
                    # Weight by inverse distance for averaging                    
                    weight = compute_dtype(1.0) / (actual_distance + epsilon)
                    
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
            epsilon = compute_dtype(1e-7)
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
                if fpop[l] < epsilon:
                    fpop[l] = feq[l]

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
            epsilon = compute_dtype(1e-7)
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
                if fpop[l] < epsilon:
                    fpop[l] = feq[l]

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
            epsilon = compute_dtype(1e-7)            
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

                    f_post[l] = wp.max(epsilon, f_post[l])
                else:
                    f_post[l] = wp.max(epsilon, f_post[l])

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
                        weight = wp.clamp(weight, compute_dtype(0.05), compute_dtype(1.0))

                        # Use differentiable interpolated BB to find f_missing:
                        #f_post[l] = ((one - weight) * f_post[_opp_indices[l]] + weight * (f_pre[l] + f_pre[_opp_indices[l]])) / (one + weight)
                        f_near = two * weight * f_pre[_opp_indices[l]] + (one - two * weight) * f_post[_opp_indices[l]]
                        f_far = (one/ (two * weight)) * f_pre[_opp_indices[l]] + ((two * weight - one) / (two * weight)) * f_pre[l]
                        blend = three * wp.pow(weight, two) - two * wp.pow(weight, three)

                        f_post[l] = (one - blend)*f_near + blend*f_far
                        
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
            epsilon = compute_dtype(1e-7)
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
                       # weight = wp.clamp(weight, compute_dtype(0.001), compute_dtype(1.0))
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

                    f_post[l] = wp.max(epsilon, f_post[l])
                else:
                    f_post[l] = wp.max(epsilon, f_post[l])

            return f_post

        # ============================================================================
        # Wall Model Functions (Spalding's Law)
        # ============================================================================
        
        @wp.func
        def spalding_law(u_plus: Any) -> Any:
            """
            Spalding's law: y+ = u+ + e^(-κB) * [e^(κu+) - 1 - κu+ - (κu+)^2/2 - (κu+)^3/6]
            
            This provides a continuous formulation valid across viscous sublayer, buffer layer, and log layer.
            
            References:
            [1] Spalding, D. B. (1961). A single formula for the law of the wall. 
                Journal of Applied Mechanics, 28(3), 455-458.
            """
            exp_kB = wp.exp(-_kappa * _B)
            ku = _kappa * u_plus
            exp_ku = wp.exp(ku)
            
            y_plus = u_plus + exp_kB * (exp_ku - compute_dtype(1.0) - ku - 
                                        ku * ku / compute_dtype(2.0) - 
                                        ku * ku * ku / compute_dtype(6.0))
            return y_plus
        
        @wp.func
        def spalding_derivative(u_plus: Any) -> Any:
            """
            Derivative of Spalding's law with respect to u+: dy+/du+
            
            Used in Newton-Raphson iteration for solving u+ given y+.
            """
            exp_kB = wp.exp(-_kappa * _B)
            ku = _kappa * u_plus
            exp_ku = wp.exp(ku)
            
            dy_du = compute_dtype(1.0) + exp_kB * _kappa * (exp_ku - compute_dtype(1.0) - 
                                                            ku - ku * ku / compute_dtype(2.0))
            return dy_du
        
        @wp.func
        def solve_spalding(K: Any) -> Any:  
            """
            Solve for u+ given K = u_parallel * y / nu using Newton-Raphson on spalding(u+) - K / u+ = 0.
            """
            # NEW: Better initial guess: Use log-law for K >= 11*u+ (approx y+>11), viscous otherwise
            if K < compute_dtype(11.0):
                u_plus = wp.sqrt(K)  # Viscous: u+ ≈ sqrt(K), since y+ ≈ u+^2 in pure linear (but Spalding blends)
            else:
                # Log-law approx: u+ ≈ (1/κ) ln(K / u+) + B, but iterative init: Start with (1/κ) ln(K) + B
                u_plus = (compute_dtype(1.0) / _kappa) * wp.log(K) + _B
                # Quick fixed-point refinement for better start (helps convergence at high K)
                for i in range(15):
                    u_plus = (compute_dtype(1.0) / _kappa) * wp.log(K / u_plus) + _B
            
            # Newton-Raphson iteration
            max_iter = wp.int32(10)  # Increased for high Re/K
            tolerance = compute_dtype(1e-6)
            epsilon = compute_dtype(1e-6)  # For div safety
            
            for iter in range(max_iter):
                # NEW: Correct residual: spalding(u+) - K / u+
                residual = spalding_law(u_plus) - K / wp.max(u_plus, epsilon)
                
                if wp.abs(residual) < tolerance:
                    break
                
                # NEW: Correct derivative: d/du [spalding(u+) - K / u+] = dy_du + K / u+^2
                dy_du = spalding_derivative(u_plus)
                df_du = dy_du + K / (u_plus * u_plus)
                u_plus -= residual / wp.max(df_du, epsilon)  # Avoid div0
                
                # Clamp for stability (prevents negatives/overshoot at transients)
                u_plus = wp.max(compute_dtype(0.0001), u_plus)
            
            return u_plus

        @wp.func
        def musker_u_from_y(y_plus: Any) -> Any:
            """Forward Musker: Given y⁺, return u⁺ (for applying wall BCs)"""
            a1 = compute_dtype( 5.43304476)
            a2 = compute_dtype(-2.69469647e-1)
            a3 = compute_dtype( 9.06392720e-2)
            a4 = compute_dtype(-1.37518637e-2)
            a5 = compute_dtype( 1.05789855e-3)
            a6 = compute_dtype(-4.49275492e-5)
            a7 = compute_dtype( 9.35634096e-7)
            a8 = compute_dtype(-7.53851505e-9)

            log_term = wp.log(y_plus + compute_dtype(11.0))
            u_plus = (
                a1 * log_term +
                a2 +
                a3 * y_plus +
                a4 * y_plus * y_plus +
                a5 * y_plus * y_plus * y_plus +
                a6 * y_plus * y_plus * y_plus * y_plus +
                a7 * y_plus * y_plus * y_plus * y_plus * y_plus +
                a8 * y_plus * y_plus * y_plus * y_plus * y_plus * y_plus
            )
            return u_plus

        @wp.func
        def musker_y_from_u(u_plus: Any) -> Any:
            """Inverse Musker: Given u⁺, return y⁺ (for solving wall model)"""
            b1 = compute_dtype(0.18394189)
            b2 = compute_dtype(-2.01702635e-2)
            b3 = compute_dtype(1.96943917e-2)
            b4 = compute_dtype(-8.98701033e-3)
            b5 = compute_dtype(2.11896938e-3)
            b6 = compute_dtype(-2.73136397e-4)
            b7 = compute_dtype(1.83406310e-5)
            b8 = compute_dtype(-5.08426253e-7)

            y_plus = (
                wp.exp(b1 * (u_plus - compute_dtype(5.0))) +
                b2 * u_plus +
                b3 +
                b4 * u_plus * u_plus +
                b5 * u_plus * u_plus * u_plus +
                b6 * u_plus * u_plus * u_plus * u_plus +
                b7 * u_plus * u_plus * u_plus * u_plus * u_plus +
                b8 * u_plus * u_plus * u_plus * u_plus * u_plus * u_plus
            )
            return y_plus

        @wp.func
        def solve_musker(K: Any) -> Any:
            """
            Solve for u⁺ given K = u_parallel * y / ν
            Pure explicit - no iteration needed!
            """
            # Handle viscous sublayer
            if K < compute_dtype(1e-8):
                return wp.sqrt(K)
            
            # Initial guess from log-law
            u_plus_init = (compute_dtype(1.0) / _kappa) * wp.log(K) + _B
            
            # Iteratively refine initial guess (still explicit, just improving starting point)
            for _ in range(3):
                y_plus = musker_y_from_u(u_plus_init)  # ← Now using correct function!
                if y_plus > compute_dtype(1e-8):
                    u_plus_init = K / y_plus  # Direct solution: u⁺ = K / y⁺
            
            return wp.max(u_plus_init, compute_dtype(0.01))

        
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
            zero = compute_dtype(0.0)
            epsilon = compute_dtype(1e-8)
            scale = compute_dtype(0.01)
            y_limit = compute_dtype(30.0)
            u_wall_physical = u_wall
            # Get wall geometry info               
            normal, y_distance = get_normal_and_distance(_missing_mask, f_1, index) 
            
            # Estimate flow state from known populations          
            rho, u_est = macroscopic.warp_functional(f_pre)
                
            
            # Decompose estimated velocity into normal and parallel components
            dot = wp.dot(u_est, normal)
            u_normal = normal * dot
            u_parallel = u_est - u_normal

            u_parallel_mag = compute_dtype(wp.length(u_parallel))           
            
            # Compute K = u_parallel_mag * y_distance / nu (unchanged)
            K = u_parallel_mag * y_distance / nu
            
            # Clamp K for safety (avoids div0 or exp overflow at transients/low nu)
            K = wp.max(epsilon, K)
            
            # Solve for u+
            #u_plus = solve_spalding(K)   
            u_plus = solve_musker(K)             
            u_tau = u_parallel_mag / wp.max(u_plus, epsilon)
            
            # Compute wall shear stress magnitude
            tau_wall = u_tau * u_tau * rho # In lattice units (assuming ρ=1)
            
            # Nonlinear adjustment: effective viscosity including turbulent part from log-law
            y_plus = y_distance * u_tau / nu  # Final y+
            y_plus += epsilon
            
            
            # Compute damping factor
            vd_damp = compute_dtype(1.0) - wp.exp(-y_plus / _A_plus)
            nu_t_B = _kappa * u_tau * y_distance #* vd_damp * vd_damp # van Driest damping
            nu_eff = nu + nu_t_B
            
            # Effective velocity gradient incorporating nonlinearity
            velocity_gradient = tau_wall / nu_eff
            
            # Slip magnitude: subtract to increase effective shear
            u_slip_mag = u_parallel_mag - velocity_gradient * y_distance

            #wp.printf(" yplus: %f, nu: %f, nu_eff: %f, u_slip_mag: %f, uParallel: %f vel_Grad: %f", y_plus, nu, nu_eff, u_slip_mag, u_parallel_mag, velocity_gradient)   
            # Clamp for stability: prevent excessively negative
            u_slip_mag = wp.max(u_slip_mag, -u_parallel_mag * compute_dtype(1.0))
            
            # Reconstruct slip velocity vector
            if u_parallel_mag > epsilon:
                u_parallel_unit = u_parallel / u_parallel_mag
                
                u_slip = u_parallel_unit * u_slip_mag
    
            else:
                u_slip = _u_vec(0.0, 0.0, 0.0)
                
            # Effective wall velocity: add slip to physical
            u_wall_effective = u_wall_physical + u_slip
            
            # Enforce zero normal component (no penetration)
            dot = wp.dot(u_wall_effective, normal)
            u_wall_effective = u_wall_effective - dot*normal
            
            return u_wall_effective, y_plus

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
        self.get_wall_distance = get_wall_distance
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
        self.regularize_wallModel = regularize_wallModel
        self.solve_musker = solve_musker
        self.musker_y_from_u = musker_y_from_u
        self.musker_u_from_y = musker_u_from_y
        self.get_known_fsum = get_known_fsum
        self.estimate_from_known = estimate_from_known
        self.get_avg_normal_vectors = get_avg_normal_vectors
        self.spalding_law = spalding_law
        self.spalding_derivative = spalding_derivative
        self.solve_spalding = solve_spalding
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