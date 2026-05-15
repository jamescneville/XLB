import numpy as np
import warp as wp
import jax.numpy as jnp
from jax import jit
from functools import partial
from typing import Any

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb.operator.macroscopic import Macroscopic
from xlb.operator.equilibrium import QuadraticEquilibrium
from xlb.operator.boundary_condition.boundary_condition import (
    ImplementationStep,
    BoundaryCondition,
    HelperFunctionsBC,
)
from xlb.operator.boundary_masker.mesh_voxelization_method import MeshVoxelizationMethod


class SlipBC(BoundaryCondition):
    """
    Slip boundary condition. 
    """
    def __init__(
        self,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
        indices=None,
        mesh_vertices=None,
        voxelization_method: MeshVoxelizationMethod = None,
        normal_axis: int=2,
    ):
        """Initialize the Slip boundary condition.

        Args:
            normal_axis (int): The axis normal to the slip surface (0 for x, 1 for y, 2 for z).
            indices (array-like, optional): Indices where the BC applies.
            
        """
        self.normal_axis = normal_axis
        # Precompute mapping once on CPU
        
        super().__init__(
            ImplementationStep.STREAMING,
            velocity_set,
            precision_policy,
            compute_backend,
            indices,
            mesh_vertices,
            voxelization_method,
        )
   
        
        # Define BC helper functions. Explicitly using the WARP backend for helper functions as it may also be called by the Neon backend.
        self.bc_helper = HelperFunctionsBC(
            velocity_set=self.velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=ComputeBackend.WARP,
            distance_decoder_function=self._construct_distance_decoder_function(),
        )
        self.macroscopic = Macroscopic(compute_backend=ComputeBackend.WARP)
        self.equilibrium = QuadraticEquilibrium(compute_backend=ComputeBackend.WARP)

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0))
    def jax_implementation(self, f_pre, f_post, bc_mask, missing_mask):
        """JAX implementation of the Slip boundary condition.

        Args:
            f_pre: Pre-collision distribution functions.
            f_post: Post-collision distribution functions.
            bc_mask: Mask identifying boundary nodes.
            missing_mask: Mask for missing populations.

        Returns:
            Updated post-collision distribution functions.
        """
        mapping = self._compute_mapping()
        boundary = bc_mask == self.id  # Shape: (1, nx, ny, nz)
        boundary = jnp.broadcast_to(boundary, (self.velocity_set.q,) + boundary.shape[1:])
        condition = boundary & missing_mask  # Shape: (q, nx, ny, nz)
        f_post_new = jnp.where(condition, f_post[mapping], f_post)
        return f_post_new

    def _construct_distance_decoder_function(self):
        """
        Constructs the distance decoder function for this BC.
        """
        # Get the opposite indices for the velocity set
        _opp_indices = self.velocity_set.opp_indices

        # Define the distance decoder function for this BC
        if self.compute_backend == ComputeBackend.WARP:

            @wp.func
            def distance_decoder_function(f_1: Any, index: Any, direction: Any):
                return f_1[_opp_indices[direction], index[0], index[1], index[2]]

        elif self.compute_backend == ComputeBackend.NEON:

            @wp.func
            def distance_decoder_function(f_1_pn: Any, index: Any, direction: Any):
                return wp.neon_read(f_1_pn, index, _opp_indices[direction])

        return distance_decoder_function
        
    def _construct_warp(self):
        """Construct the Warp kernel for the Slip boundary condition.

        Returns:
            tuple: Warp functional and kernel.
        """
        _normal_axis = wp.constant(self.normal_axis)

        @wp.func
        def functional(
            index: Any,
            timestep: Any,
            missing_mask: Any,
            f_0: Any,
            f_1: Any,
            f_pre: Any,
            f_post: Any,
        ):
            f_post = self.bc_helper.slip_mapping(
                index, 
                missing_mask, 
                f_0,
                f_1,
                f_pre,
                f_post,
                _normal_axis
            )
            # Compute density, velocity using all f_post-streaming values
            rho, u = self.macroscopic.warp_functional(f_post)

            # Regularize the resulting populations
            feq = self.equilibrium.warp_functional(rho, u)
            f_post = self.bc_helper.regularize_fpop(f_post, feq)
         
            return f_post

        kernel = self._construct_kernel(functional)
        return functional, kernel
        
    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f_pre, f_post, bc_mask, missing_mask):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[f_pre, f_post, bc_mask, missing_mask],
            dim=f_pre.shape[1:],
        )
        return f_post
    
    def _construct_neon(self):
        functional, _ = self._construct_warp()
        return functional, None

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_implementation(self, f_pre, f_post, bc_mask, missing_mask):
        # rise exception as this feature is not implemented yet
        raise NotImplementedError("This feature is not implemented in XLB with the NEON backend yet.")
