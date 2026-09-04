"""
Multi-Resolution Navier-Stokes Stepper for the NEON Backend

This module implements the multi-resolution LBM stepper using Warp kernels on the
Neon multi-GPU runtime. It uses several programming patterns specific to Warp's
compile-time code generation model.

Compile-Time Specialization Pattern
-----------------------------------
Warp's @wp.func decorator traces Python code at kernel compilation time, not runtime.
This means runtime boolean parameters cause Warp to emit branching code for both paths,
increasing register pressure even when only one path is ever taken.

To generate optimized, branch-free kernels, we use a **factory pattern** that captures
boolean configuration at function-definition time:

    def make_specialized_func(do_feature: bool):
        @wp.func
        def impl(...):
            if wp.static(do_feature):  # Evaluated at compile time
                # This code is only emitted when do_feature=True
                ...
            else:
                # This code is only emitted when do_feature=False
                ...
        return impl

    # Generate specialized variants
    func_with_feature = make_specialized_func(do_feature=True)
    func_without_feature = make_specialized_func(do_feature=False)

The `wp.static()` call evaluates its argument during Warp's tracing phase. Since
`do_feature` is a Python bool captured in the closure, Warp sees a constant and
eliminates the dead branch entirely.

This pattern is used for:
- `apply_bc_post_streaming` / `apply_bc_post_collision`: Specialized BC application
  for streaming vs collision implementation steps
- `collide_bc_accum` / `collide_simple`: Collision pipeline variants with/without
  BC application and multi-resolution accumulation

Closure Capture for Self Attributes
-----------------------------------
Warp cannot resolve `self.X` in plain assignments inside @wp.func bodies (e.g.,
`_c = self.velocity_set.c` fails with "Invalid external reference type"). However,
it can resolve `self.X` in:
- Function call contexts: `self.stream.neon_functional(...)`
- Range arguments: `range(self.velocity_set.q)`
- Type casts: `self.compute_dtype(0)`

For other uses, we pre-capture attributes at the Python level before defining the
@wp.func, making them available as simple closure variables:

    _c = self.velocity_set.c  # Captured in Python scope

    @wp.func
    def my_kernel(...):
        # Use _c directly — Warp sees it as a closure variable
        direction = wp.neon_ngh_idx(wp.int8(_c[0, l]), ...)

Cell Type Constants
-------------------
Cell types are defined in `xlb.cell_type`:
- BC_SFV (254): Simple Fluid Voxel — no BC, no explosion/coalescence
- BC_SOLID (255): Solid obstacle voxel
- BC_NONE (0): Regular fluid voxel with potential BCs or multi-res interactions
"""

import os

import nvtx
import warp as wp
from typing import Any

from xlb import DefaultConfig
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import Precision
from xlb.operator import Operator
from xlb.operator.stream import Stream
from xlb.operator.collision import BGK, KBC, SmagorinskyLESBGK, SmagorinskyLESKBC
from xlb.operator.equilibrium import MultiresQuadraticEquilibrium
from xlb.operator.macroscopic import MultiresMacroscopic
from xlb.operator.stepper import Stepper
from xlb.operator.boundary_condition.boundary_condition import ImplementationStep
from xlb.operator.boundary_condition.boundary_condition_registry import boundary_condition_registry
from xlb.operator.collision import ForcedCollision
from xlb.helper import check_bc_overlaps
from xlb.operator.boundary_masker import (
    MeshVoxelizationMethod,
    MultiresMeshMaskerAABB,
    MultiresMeshMaskerAABBClose,
    MultiresIndicesBoundaryMasker,
    MultiresMeshMaskerRay,
    MultiresMeshMaskerTrapped,
)
from xlb.operator.boundary_condition.helper_functions_bc import MultiresEncodeAuxiliaryData
from xlb.cell_type import BC_NONE, BC_SFV, BC_SOLID

"""
SFV = Simple Fluid Voxel: a fluid voxel that is not a BC nor is involved in explosion or coalescence
CFV = Complex Fluid Voxel: a fluid voxel that is not a SFV
"""


class MultiresIncompressibleNavierStokesStepper(Stepper):
    """Multi-resolution incompressible Navier-Stokes stepper for the Neon backend.

    Implements the full LBM step (stream, collide, boundary conditions) across
    a hierarchy of grid levels using Neon containers.  Each container is a
    compile-time specialized Warp kernel wrapped in a Neon execution-graph
    node.

    The stepper supports several performance optimization strategies (see
    :class:`MresPerfOptimizationType`):

    * **NAIVE_COLLIDE_STREAM** — separate collide and stream containers at
      every level.
    * **FUSION_AT_FINEST** — fused stream+collide at the finest level.
    * **FUSION_AT_FINEST_SFV** — additionally splits SFV / CFV voxels at
      the finest level for reduced branching.
    * **FUSION_AT_FINEST_SFV_ALL** — SFV / CFV splitting at all levels.

    Parameters
    ----------
    grid : NeonMultiresGrid
        The multi-resolution grid.
    boundary_conditions : list of BoundaryCondition
        Boundary conditions to apply.
    collision_type : str
        Collision operator type: ``"BGK"`` or ``"KBC"`` or ``"SmagorinskyLESBGK"`` or ``"SmagorinskyLESKBC"``.
    forcing_scheme : str
        Forcing scheme name (only used when *force_vector* is given).
    force_vector : array-like, optional
        External body force vector.
    """

    def __init__(
        self,
        grid,
        boundary_conditions=[],
        collision_type="BGK",
        forcing_scheme="exact_difference",
        force_vector=None,
        smagorinsky_constant = 0.0
    ):
        super().__init__(grid, boundary_conditions)

        # Construct the collision operator
        self._collision_requires_macro = wp.bool(False)
        if collision_type == "BGK":
            self.collision = BGK(self.velocity_set, self.precision_policy, self.compute_backend)
        elif collision_type == "KBC":
            self.collision = KBC(self.velocity_set, self.precision_policy, self.compute_backend)
        elif collision_type == "SmagorinskyLESBGK":
            self.collision = SmagorinskyLESBGK(self.velocity_set, self.precision_policy, self.compute_backend)
        elif collision_type == "SmagorinskyLESKBC":
            self.collision = SmagorinskyLESKBC(self.velocity_set, self.precision_policy, self.compute_backend, smagorinsky_constant)
            self._collision_requires_macro = wp.bool(True)

        if force_vector is not None:
            self.collision = ForcedCollision(collision_operator=self.collision, forcing_scheme=forcing_scheme, force_vector=force_vector)

        # Construct the operators
        self.stream = Stream(self.velocity_set, self.precision_policy, self.compute_backend)
        self.equilibrium = MultiresQuadraticEquilibrium(self.velocity_set, self.precision_policy, self.compute_backend)
        self.macroscopic = MultiresMacroscopic(self.velocity_set, self.precision_policy, self.compute_backend)

    def prepare_fields(self, rho, u, initializer=None):
        import neon

        """Prepare the fields required for the stepper.

        Args:
            initializer: Optional operator to initialize the distribution functions.
                        If provided, it should be a callable that takes (grid, velocity_set,
                        precision_policy, compute_backend) as arguments and returns initialized f_0.
                        If None, default equilibrium initialization is used with rho=1 and u=0.

        Returns:
            Tuple of (f_0, f_1, bc_mask, missing_mask):
                - f_0: Initial distribution functions
                - f_1: Copy of f_0 for double-buffering
                - bc_mask: Boundary condition mask indicating which BC applies to each node
                - missing_mask: Mask indicating which populations are missing at boundary nodes
        """

        f_0 = self.grid.create_field(
            cardinality=self.velocity_set.q, dtype=self.precision_policy.store_precision, neon_memory_type=neon.MemoryType.device()
        )

        f_1 = self.grid.create_field(
            cardinality=self.velocity_set.q, dtype=self.precision_policy.store_precision, neon_memory_type=neon.MemoryType.device()
        )

        missing_mask = self.grid.create_field(cardinality=self.velocity_set.q, dtype=Precision.UINT8)
        bc_mask = self.grid.create_field(cardinality=1, dtype=Precision.UINT8)
        normal_vector = self.grid.create_field(cardinality=self.velocity_set.d, dtype=self.precision_policy.store_precision)
        normal_distance = self.grid.create_field(cardinality=1, dtype=self.precision_policy.store_precision)

        for level in range(self.grid.count_levels):
            f_1.copy_from_run(level, f_0, 0)

        # Process boundary conditions and update masks
        f_1, bc_mask, missing_mask,  normal_vector, normal_distance= self._process_boundary_conditions(self.boundary_conditions, f_1, bc_mask, missing_mask, normal_vector, normal_distance)
        # Initialize auxiliary data if needed
        f_1 = self._initialize_auxiliary_data(self.boundary_conditions, f_1, bc_mask, missing_mask)

        # Initialize distribution functions if initializer is provided
        if initializer is not None:
            # Refer to xlb.helper.initializers for available initializers
            f_0 = initializer(bc_mask, f_0)
        else:
            from xlb.helper.initializers import initialize_multires_eq

            f_0 = initialize_multires_eq(f_0, self.grid, self.velocity_set, self.precision_policy, self.compute_backend, rho=rho, u=u)

        return f_0, f_1, bc_mask, missing_mask, normal_vector, normal_distance

    def prepare_coalescence_count(self, coalescence_factor, bc_mask):
        """Precompute coalescence weighting factors for multi-resolution streaming.

        For each non-halo voxel at every level, this method accumulates
        the number of finer neighbours that contribute populations via
        coalescence (child-to-parent transfer), then inverts the count
        so that the streaming kernel can apply the correct averaging weight.

        Parameters
        ----------
        coalescence_factor : field
            Multi-resolution field to store the per-direction coalescence
            weights (modified in-place).
        bc_mask : field
            Boundary-condition mask used to skip solid voxels.
        """
        import neon

        lattice_central_index = self.velocity_set.center_index
        num_levels = coalescence_factor.get_grid().num_levels

        @neon.Container.factory(name="sum_kernel_by_level")
        def sum_kernel_by_level(level):
            def ll_coalescence_count(loader: neon.Loader):
                loader.set_mres_grid(coalescence_factor.get_grid(), level)

                coalescence_factor_pn = loader.get_mres_read_handle(coalescence_factor)
                bc_mask_pn = loader.get_mres_read_handle(bc_mask)

                _c = self.velocity_set.c
                _w = self.velocity_set.w

                @wp.func
                def cl_collide_coarse(index: Any):
                    _boundary_id = wp.neon_read(bc_mask_pn, index, 0)
                    if _boundary_id == wp.uint8(BC_SOLID):
                        return
                    if not wp.neon_has_child(coalescence_factor_pn, index):
                        for l in range(self.velocity_set.q):
                            if level < num_levels - 1:
                                push_direction = wp.neon_ngh_idx(wp.int8(_c[0, l]), wp.int8(_c[1, l]), wp.int8(_c[2, l]))
                                val = self.store_dtype(1)
                                wp.neon_mres_lbm_store_op(coalescence_factor_pn, index, l, push_direction, val)

                loader.declare_kernel(cl_collide_coarse)

            return ll_coalescence_count

        for level in range(num_levels):
            sum_kernel = sum_kernel_by_level(level)
            sum_kernel.run(0)

        @neon.Container.factory(name="sum_kernel_by_level")
        def invert_count(level):
            def loading(loader: neon.Loader):
                loader.set_mres_grid(coalescence_factor.get_grid(), level)

                coalescence_factor_pn = loader.get_mres_read_handle(coalescence_factor)
                bc_mask_pn = loader.get_mres_read_handle(bc_mask)

                _c = self.velocity_set.c
                _w = self.velocity_set.w

                @wp.func
                def compute(index: Any):
                    _boundary_id = wp.neon_read(bc_mask_pn, index, 0)
                    if _boundary_id == wp.uint8(BC_SOLID):
                        return

                    if wp.neon_has_child(coalescence_factor_pn, index):
                        # we are a halo cell so we just exit
                        return

                    for l in range(self.velocity_set.q):
                        if l == lattice_central_index:
                            continue

                        pull_direction = wp.neon_ngh_idx(wp.int8(-_c[0, l]), wp.int8(-_c[1, l]), wp.int8(-_c[2, l]))

                        has_ngh_at_same_level = wp.bool(False)
                        coalescence_factor = self.compute_dtype(
                            wp.neon_read_ngh(coalescence_factor_pn, index, pull_direction, l, self.store_dtype(0), has_ngh_at_same_level)
                        )

                        if not wp.neon_has_finer_ngh(coalescence_factor_pn, index, pull_direction):
                            pass
                        else:
                            # Finer neighbour exists in the pull direction (opposite of l).
                            # Read from the halo sitting on top of that finer neighbour.
                            if has_ngh_at_same_level:
                                # Finer ngh in pull direction: YES
                                # Same-level ngh:              YES
                                # Compute coalescence factor
                                if coalescence_factor > self.compute_dtype(0):
                                    coalescence_factor = self.compute_dtype(1) / (self.compute_dtype(2) * coalescence_factor)
                                    wp.neon_write(coalescence_factor_pn, index, l, self.store_dtype(coalescence_factor))

                loader.declare_kernel(compute)

            return loading

        for level in range(num_levels):
            sum_kernel = invert_count(level)
            sum_kernel.run(0)
        return

    @classmethod
    def _process_boundary_conditions(cls, boundary_conditions, f_1, bc_mask, missing_mask, normal_vector, normal_distance):
        """Process boundary conditions and update boundary masks."""

        # Check for boundary condition overlaps
        # TODO! check_bc_overlaps(boundary_conditions, DefaultConfig.velocity_set.d, DefaultConfig.default_backend)

        # Create boundary maskers
        indices_masker = MultiresIndicesBoundaryMasker(
            velocity_set=DefaultConfig.velocity_set,
            precision_policy=DefaultConfig.default_precision_policy,
            compute_backend=DefaultConfig.default_backend,
        )

        # Split boundary conditions by type
        bc_with_vertices = [bc for bc in boundary_conditions if bc.mesh_vertices is not None]
        bc_with_indices = [bc for bc in boundary_conditions if bc.indices is not None]

        # Process indices-based boundary conditions
        if bc_with_indices:
            bc_mask, missing_mask = indices_masker(bc_with_indices, bc_mask, missing_mask)

        # Process mesh-based boundary conditions for 3D
        if DefaultConfig.velocity_set.d == 3 and bc_with_vertices:
            for bc in bc_with_vertices:
                if bc.voxelization_method.id is MeshVoxelizationMethod("AABB").id:
                    mesh_masker = MultiresMeshMaskerAABB(
                        velocity_set=DefaultConfig.velocity_set,
                        precision_policy=DefaultConfig.default_precision_policy,
                        compute_backend=DefaultConfig.default_backend,
                    )
                elif bc.voxelization_method.id is MeshVoxelizationMethod("RAY").id:
                    mesh_masker = MultiresMeshMaskerRay(
                        velocity_set=DefaultConfig.velocity_set,
                        precision_policy=DefaultConfig.default_precision_policy,
                        compute_backend=DefaultConfig.default_backend,
                    )
                elif bc.voxelization_method.id is MeshVoxelizationMethod("AABB_CLOSE").id:
                    mesh_masker = MultiresMeshMaskerAABBClose(
                        velocity_set=DefaultConfig.velocity_set,
                        precision_policy=DefaultConfig.default_precision_policy,
                        compute_backend=DefaultConfig.default_backend,
                        close_voxels=bc.voxelization_method.options.get("close_voxels"),
                    )
                else:
                    raise ValueError(f"Unsupported voxelization method for multi-res: {bc.voxelization_method}")
                # Apply the mesh masker to the boundary condition
                f_1, bc_mask, missing_mask, normal_vector, normal_distance = mesh_masker(bc, f_1, bc_mask, missing_mask, normal_vector, normal_distance)

            # Run Trapped Masker looking for sandwich voxels
            trapped_masker = MultiresMeshMaskerTrapped(
                    velocity_set=DefaultConfig.velocity_set,
                    precision_policy=DefaultConfig.default_precision_policy,
                    compute_backend=DefaultConfig.default_backend,
            )
            
            f_1, bc_mask, missing_mask = trapped_masker(bc_with_vertices[0], f_1, bc_mask, missing_mask)



        return f_1, bc_mask, missing_mask, normal_vector, normal_distance

    @staticmethod
    def _initialize_auxiliary_data(boundary_conditions, f_1, bc_mask, missing_mask):
        """Initialize auxiliary data for boundary conditions that require it."""
        for bc in boundary_conditions:
            if bc.needs_aux_init and not bc.is_initialized_with_aux_data:
                # Create the encoder operator for storing the auxiliary data
                encode_auxiliary_data = MultiresEncodeAuxiliaryData(
                    bc.id,
                    bc.num_of_aux_data,
                    bc.profile,
                    velocity_set=bc.velocity_set,
                    precision_policy=bc.precision_policy,
                    compute_backend=bc.compute_backend,
                )

                # Encode the auxiliary data in f_1
                f_1 = encode_auxiliary_data(f_1, bc_mask, missing_mask, stream=0)
                bc.is_initialized_with_aux_data = True
        return f_1

    def _construct_neon(self):
        import neon

        # Pre-capture self attributes that Warp cannot resolve inside @wp.func bodies.
        # Warp rejects `self` as an "Invalid external reference type" when it appears
        # in a plain assignment (e.g. `_c = self.velocity_set.c`).  Capturing here
        # makes these values available as simple closure variables.
        # ---------------------------------------------------------------
        # Diagnostic ablations for CFV_finest_fused_pull, the container that
        # dominates the profile.  Each token removes one stage so its cost
        # share can be measured directly.  Every ablation makes the physics
        # WRONG -- this exists only to attribute time.
        #   XLB_ABLATE=accum       skip the 27 neon_mres_lbm_store_op coalescence writes
        #   XLB_ABLATE=explosion   plain streaming instead of stream-with-explosion
        #   XLB_ABLATE=bc          skip BC application and aux recovery
        #   XLB_ABLATE=thread_data skip the local f_0 / missing_mask reads
        # Comma-separate to combine, e.g. XLB_ABLATE=accum,explosion
        # ---------------------------------------------------------------
        _ablate = {t.strip() for t in os.environ.get("XLB_ABLATE", "").split(",") if t.strip()}
        _unknown = _ablate - {"accum", "explosion", "bc", "thread_data"}
        if _unknown:
            raise ValueError(f"XLB_ABLATE: unknown token(s) {sorted(_unknown)}")
        if _ablate:
            print(f"*** XLB_ABLATE={sorted(_ablate)} -- CFV finest kernel ABLATED, physics INVALID ***")
        _abl_accum = "accum" in _ablate
        _abl_explosion = "explosion" in _ablate
        _abl_bc = "bc" in _ablate
        _abl_thread_data = "thread_data" in _ablate

        # Per-voxel gating of the multiresolution stages (explosion reads and
        # coalescence accumulation) on a precomputed `needs_mres` flag.  Set
        # XLB_NO_MRES_GATE=1 to fall back to the unconditional behaviour for A/B.
        _mres_gate_disabled = os.environ.get("XLB_NO_MRES_GATE", "0").strip().lower() in ("1", "true", "yes", "on")
        self._mres_gate_enabled = not _mres_gate_disabled

        lattice_central_index = self.velocity_set.center_index
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)
        _missing_mask_vec = wp.vec(self.velocity_set.q, dtype=wp.uint8)
        _opp_indices = self.velocity_set.opp_indices
        _c = self.velocity_set.c

        # Read the list of bc_to_id created upon instantiation
        bc_to_id = boundary_condition_registry.bc_to_id

        # Gather IDs of ExtrapolationOutflowBC boundary conditions
        extrapolation_outflow_bc_ids = []
        for bc_name, bc_id in bc_to_id.items():
            if bc_name.startswith("ExtrapolationOutflowBC"):
                extrapolation_outflow_bc_ids.append(bc_id)
        # Gather IDs of HybridBC boundary conditions
        hybrid_bc_ids = []
        for bc_name, bc_id in bc_to_id.items():
            if bc_name.startswith("HybridBC"):
                hybrid_bc_ids.append(bc_id)

        
        # Factory for apply_bc: generates compile-time specialized variants
        def make_apply_bc(is_post_streaming: bool):
            @wp.func
            def apply_bc_impl(
                index: Any,
                timestep: Any,
                _boundary_id: Any,
                _missing_mask: Any,
                f_0: Any,
                f_1: Any,
                f_pre: Any,
                f_post: Any,
                _rho: Any,
                _u: Any,
                _relax:Any, 
                _norm_vec_pn: Any,
                _norm_dist_pn: Any,
            ):
                f_result = f_post

                for i in range(wp.static(len(self.boundary_conditions))):
                    if wp.static(is_post_streaming):
                        if wp.static(self.boundary_conditions[i].implementation_step == ImplementationStep.STREAMING):
                            if wp.static(self.boundary_conditions[i].id in hybrid_bc_ids):
                                if _boundary_id == wp.static(self.boundary_conditions[i].id):
                                    f_result = wp.static(self.boundary_conditions[i].neon_functional)(
                                        index, timestep, _missing_mask, f_0, f_1, f_pre, f_post, _rho, _u, _relax, _norm_vec_pn, _norm_dist_pn)
                            else:
                                if _boundary_id == wp.static(self.boundary_conditions[i].id):
                                    f_result = wp.static(self.boundary_conditions[i].neon_functional)(
                                        index, timestep, _missing_mask, f_0, f_1, f_pre, f_post
                                    )
                    else:
                        if wp.static(self.boundary_conditions[i].implementation_step == ImplementationStep.COLLISION):
                            if _boundary_id == wp.static(self.boundary_conditions[i].id):
                                f_result = wp.static(self.boundary_conditions[i].neon_functional)(
                                    index, timestep, _missing_mask, f_0, f_1, f_pre, f_post
                                )
                        if wp.static(self.boundary_conditions[i].id in extrapolation_outflow_bc_ids):
                            if _boundary_id == wp.static(self.boundary_conditions[i].id):
                                f_result = wp.static(self.boundary_conditions[i].assemble_auxiliary_data)(
                                    index, timestep, _missing_mask, f_0, f_1, f_pre, f_post
                                )
                return f_result

            return apply_bc_impl

        # Compile-time specialized BC application variants
        apply_bc_post_streaming = make_apply_bc(is_post_streaming=True)
        apply_bc_post_collision = make_apply_bc(is_post_streaming=False)

        @wp.func
        def neon_get_thread_data(
            f0_pn: Any,
            missing_mask_pn: Any,
            index: Any,
        ):
            # Read thread data for populations
            _f0_thread = _f_vec()
            _missing_mask = _missing_mask_vec()
            for l in range(self.velocity_set.q):
                # q-sized vector of pre-streaming populations
                _f0_thread[l] = self.compute_dtype(wp.neon_read(f0_pn, index, l))
                _missing_mask[l] = wp.neon_read(missing_mask_pn, index, l)

            return _f0_thread, _missing_mask

        @wp.func
        def neon_get_f0(
            f0_pn: Any,
            index: Any,
        ):
            # Populations-only variant of neon_get_thread_data, for the SFV fast
            # paths that never consult the missing mask.  Reading missing_mask
            # there costs q bytes of traffic per voxel plus q registers for a
            # vector that is immediately discarded.
            _f0_thread = _f_vec()
            for l in range(self.velocity_set.q):
                _f0_thread[l] = self.compute_dtype(wp.neon_read(f0_pn, index, l))
            return _f0_thread

        @wp.func
        def neon_apply_aux_recovery_bc(
            index: Any,
            _boundary_id: Any,
            _missing_mask: Any,
            f_0_pn: Any,
            f_1_pn: Any,
        ):
            # Note:
            # In XLB, the BC auxiliary data (e.g. prescribed values of pressure or normal velocity) are stored in (i) central index of f_1 and/or
            # (ii) missing directions of f_1. Some BCs may or may not need all these available storage space. This function checks whether
            # the BC needs recovery of auxiliary data and then recovers the information for the next iteration (due to buffer swapping) by
            # writting the values of f_1 into f_0.

            # Unroll the loop over boundary conditions
            for i in range(wp.static(len(self.boundary_conditions))):
                if wp.static(self.boundary_conditions[i].needs_aux_recovery):
                    if _boundary_id == wp.static(self.boundary_conditions[i].id):
                        for l in range(self.velocity_set.q):
                            # Perform the swapping of data
                            if l == lattice_central_index:
                                # (i) Recover the values stored in the central index of f_1
                                _f1_thread = wp.neon_read(f_1_pn, index, l)
                                wp.neon_write(f_0_pn, index, l, self.store_dtype(_f1_thread))
                            elif _missing_mask[l] == wp.uint8(1):
                                # (ii) Recover the values stored in the missing directions of f_1
                                _f1_thread = wp.neon_read(f_1_pn, index, _opp_indices[l])
                                wp.neon_write(f_0_pn, index, _opp_indices[l], self.store_dtype(_f1_thread))

        # Factory for neon_collide_pipeline: generates compile-time specialized variants
        def make_collide_pipeline(do_bc: bool, do_accumulation: bool):
            @wp.func
            def collide_pipeline_impl(
                index: Any,
                timestep: Any,
                _boundary_id: Any,
                _missing_mask: Any,
                f_0_pn: Any,
                f_1_pn: Any,
                _f_post_stream: Any,
                omega: Any,
                num_levels: int,
                level: int,
                accumulation_pn: Any,
                _rho0_pn: Any,
                _u0_pn: Any,
                _rho1_pn: Any,
                _u1_pn: Any,
                _relax_pn: Any,
                _norm_vec_pn: Any,
                _norm_dist_pn: Any,
            ):
                _rho, _u = self.macroscopic.neon_functional(_f_post_stream)
                _feq = self.equilibrium.neon_functional(_rho, _u)
                if wp.static(self._collision_requires_macro):
                    _f_post_collision = self.collision.neon_functional(
                        _f_post_stream, _feq, _rho, _u, omega
                    )
                else:
                    _f_post_collision = self.collision.neon_functional(
                        _f_post_stream, _feq, omega
                    )

                if wp.static(do_bc):
                    _f_post_collision = apply_bc_post_collision(
                        index, timestep, _boundary_id, _missing_mask, f_0_pn, f_1_pn, _f_post_stream, _f_post_collision,
                        _rho0_pn, _u0_pn, _relax_pn, _norm_vec_pn, _norm_dist_pn
                    )
                    neon_apply_aux_recovery_bc(index, _boundary_id, _missing_mask, f_0_pn, f_1_pn)

                if wp.static(do_accumulation):
                    for l in range(self.velocity_set.q):
                        push_direction = wp.neon_ngh_idx(wp.int8(_c[0, l]), wp.int8(_c[1, l]), wp.int8(_c[2, l]))
                        if level < num_levels - 1:
                            wp.neon_mres_lbm_store_op(accumulation_pn, index, l, push_direction, self.store_dtype(_f_post_collision[l]))
                        wp.neon_write(f_1_pn, index, l, self.store_dtype(_f_post_collision[l]))
                else:
                    for l in range(self.velocity_set.q):
                        wp.neon_write(f_1_pn, index, l, self.store_dtype(_f_post_collision[l]))
                
                # Update rho / u fields for wall model                
                wp.neon_write(_rho1_pn, index, 0, self.store_dtype(_rho))
                for d in range(self.velocity_set.d):
                    wp.neon_write(_u1_pn, index, d, self.store_dtype(_u[d]))
             

                return _f_post_collision

            return collide_pipeline_impl

        # Compile-time specialized collision pipeline variants
        collide_bc_accum = make_collide_pipeline(do_bc=True, do_accumulation=True)
        collide_bc_only = make_collide_pipeline(do_bc=True, do_accumulation=False)
        collide_simple = make_collide_pipeline(do_bc=False, do_accumulation=False)
        collide_accum_only = make_collide_pipeline(do_bc=False, do_accumulation=True)
        # Used by CFV_finest_fused_pull; identical to collide_bc_accum unless an
        # XLB_ABLATE token disables a stage.
        collide_cfv_finest = make_collide_pipeline(do_bc=not _abl_bc, do_accumulation=not _abl_accum)

        @wp.func
        def neon_stream_explode_coalesce(
            index: Any,
            f_0_pn: Any,
            coalescence_factor_pn: Any,
        ):
            _f_post_stream = self.stream.neon_functional(f_0_pn, index)

            for l in range(self.velocity_set.q):
                if l == lattice_central_index:
                    continue

                pull_direction = wp.neon_ngh_idx(wp.int8(-_c[0, l]), wp.int8(-_c[1, l]), wp.int8(-_c[2, l]))

                has_ngh_at_same_level = wp.bool(False)
                accumulated = wp.neon_read_ngh(f_0_pn, index, pull_direction, l, self.store_dtype(0), has_ngh_at_same_level)
                accumulated = wp.max(accumulated, self.compute_dtype(0.0))

                if not wp.neon_has_finer_ngh(f_0_pn, index, pull_direction):
                    # No finer ngh in the pull direction (opposite of l)
                    if not has_ngh_at_same_level:
                        # No same-level ngh — could we have a coarser-level ngh?
                        if wp.neon_has_parent(f_0_pn, index):
                            # Halo cell on top of us (parent exists)
                            has_a_coarser_ngh = wp.bool(False)
                            exploded_pop = wp.neon_lbm_read_coarser_ngh(f_0_pn, index, pull_direction, l, self.store_dtype(0), has_a_coarser_ngh)
                            if has_a_coarser_ngh:
                                # No finer ngh in pull direction, no same-level ngh,
                                # but a parent (ghost cell) exists with a coarser ngh
                                # -> Explosion: read the exploded population from the
                                #    coarser level's halo.
                                _f_post_stream[l] = self.compute_dtype(exploded_pop)
                else:
                    # Finer ngh exists in the pull direction (opposite of l).
                    # Read from the halo on top of that finer ngh.
                    if has_ngh_at_same_level:
                        # Finer ngh in pull direction: YES
                        # Same-level ngh:              YES
                        # -> Coalescence
                        coalescence_factor = wp.neon_read(coalescence_factor_pn, index, l)
                        accumulated = accumulated * coalescence_factor
                        _f_post_stream[l] = self.compute_dtype(accumulated)

            return _f_post_stream

        @neon.Container.factory(name="collide_coarse")
        def collide_coarse(level: int, f_0_fd: Any, f_1_fd: Any, bc_mask_fd: Any, missing_mask_fd: Any, omega: Any, timestep: int, _rho0: Any, _u0: Any,  _rho1: Any,  _u1: Any,  _relax: Any, normal_vector: Any,normal_distance: Any):
            num_levels = f_0_fd.get_grid().num_levels

            def ll(loader: neon.Loader):
                loader.set_mres_grid(bc_mask_fd.get_grid(), level)
                if level + 1 < f_0_fd.get_grid().num_levels:
                    f_0_pn = loader.get_mres_write_handle(f_0_fd, neon.Loader.Operation.stencil_up)
                    f_1_pn = loader.get_mres_write_handle(f_1_fd, neon.Loader.Operation.stencil_up)
                else:
                    f_0_pn = loader.get_mres_read_handle(f_0_fd)
                    f_1_pn = loader.get_mres_write_handle(f_1_fd)
                bc_mask_pn = loader.get_mres_read_handle(bc_mask_fd)
                missing_mask_pn = loader.get_mres_read_handle(missing_mask_fd)
                _rho0_pn = loader.get_mres_read_handle(_rho0)
                _u0_pn = loader.get_mres_read_handle(_u0)
                _rho1_pn = loader.get_mres_write_handle(_rho1)
                _u1_pn = loader.get_mres_write_handle(_u1)
                _relax_pn = loader.get_mres_write_handle(_relax)
                _norm_vec_pn = loader.get_mres_write_handle(normal_vector)
                _norm_dist_pn = loader.get_mres_write_handle(normal_distance)


                @wp.func
                def device(index: Any):
                    _boundary_id = wp.neon_read(bc_mask_pn, index, 0)
                    if _boundary_id == wp.uint8(BC_SOLID):
                        return
                    if not wp.neon_has_child(f_0_pn, index):
                        _f0_thread, _missing_mask = neon_get_thread_data(f_0_pn, missing_mask_pn, index)
                        collide_bc_accum(
                            index,
                            timestep,
                            _boundary_id,
                            _missing_mask,
                            f_0_pn,
                            f_1_pn,
                            _f0_thread,
                            omega,
                            num_levels,
                            level,
                            f_1_pn,
                            _rho0_pn,
                            _u0_pn,
                            _rho1_pn,
                            _u1_pn,
                            _relax_pn,
                            _norm_vec_pn,
                            _norm_dist_pn,
                        )
                    else:
                        for l in range(self.velocity_set.q):
                            wp.neon_write(f_1_pn, index, l, self.store_dtype(0))

                loader.declare_kernel(device)

            return ll

        @neon.Container.factory(name="SFV_collide_coarse")
        def SFV_collide_coarse(level: int, f_0_fd: Any, f_1_fd: Any, bc_mask_fd: Any, missing_mask_fd: Any, omega: Any, timestep: int, _rho0: Any, _u0: Any,  _rho1: Any,  _u1: Any,  _relax: Any, normal_vector: Any,normal_distance: Any):
            """Collision on SFV voxels only — no BCs, no multi-resolution accumulation."""

            def ll(loader: neon.Loader):
                loader.set_mres_grid(bc_mask_fd.get_grid(), level)
                f_0_pn = loader.get_mres_read_handle(f_0_fd)
                f_1_pn = loader.get_mres_write_handle(f_1_fd)
                bc_mask_pn = loader.get_mres_read_handle(bc_mask_fd)
                missing_mask_pn = loader.get_mres_read_handle(missing_mask_fd)
                _rho0_pn = loader.get_mres_read_handle(_rho0)
                _u0_pn = loader.get_mres_read_handle(_u0)
                _rho1_pn = loader.get_mres_write_handle(_rho1)
                _u1_pn = loader.get_mres_write_handle(_u1)
                _relax_pn = loader.get_mres_write_handle(_relax)
                _norm_vec_pn = loader.get_mres_write_handle(normal_vector)
                _norm_dist_pn = loader.get_mres_write_handle(normal_distance)

                @wp.func
                def device(index: Any):
                    _boundary_id = wp.neon_read(bc_mask_pn, index, 0)
                    if _boundary_id != wp.uint8(BC_SFV):
                        return
                    # Collide-only kernel: the local populations are the collision
                    # input.  collide_simple is do_bc=False, so the missing mask is
                    # never consulted -- do not pay to read it.
                    _f0_thread = neon_get_f0(f_0_pn, index)
                    _no_missing_mask = _missing_mask_vec()
                    collide_simple(
                        index,
                        0,
                        _boundary_id,
                        _no_missing_mask,
                        f_0_pn,
                        f_1_pn,
                        _f0_thread,
                        omega,
                        0,
                        level,
                        f_1_pn,
                        _rho0_pn,
                        _u0_pn,
                        _rho1_pn,
                        _u1_pn,
                        _relax_pn,
                        _norm_vec_pn,
                        _norm_dist_pn,
                    )

                loader.declare_kernel(device)

            return ll

        @neon.Container.factory(name="CFV_collide_coarse")
        def CFV_collide_coarse(level: int, f_0_fd: Any, f_1_fd: Any, bc_mask_fd: Any, missing_mask_fd: Any, omega: Any, timestep: int, _rho0: Any, _u0: Any,  _rho1: Any,  _u1: Any,  _relax: Any, normal_vector: Any,normal_distance: Any):
            """Collision on CFV voxels only — skips both solid and SFV."""
            num_levels = f_0_fd.get_grid().num_levels

            def ll(loader: neon.Loader):
                loader.set_mres_grid(bc_mask_fd.get_grid(), level)
                if level + 1 < f_0_fd.get_grid().num_levels:
                    f_0_pn = loader.get_mres_write_handle(f_0_fd, neon.Loader.Operation.stencil_up)
                    f_1_pn = loader.get_mres_write_handle(f_1_fd, neon.Loader.Operation.stencil_up)
                else:
                    f_0_pn = loader.get_mres_read_handle(f_0_fd)
                    f_1_pn = loader.get_mres_write_handle(f_1_fd)
                bc_mask_pn = loader.get_mres_read_handle(bc_mask_fd)
                missing_mask_pn = loader.get_mres_read_handle(missing_mask_fd)
                _rho0_pn = loader.get_mres_read_handle(_rho0)
                _u0_pn = loader.get_mres_read_handle(_u0)
                _rho1_pn = loader.get_mres_write_handle(_rho1)
                _u1_pn = loader.get_mres_write_handle(_u1)
                _relax_pn = loader.get_mres_write_handle(_relax)
                _norm_vec_pn = loader.get_mres_write_handle(normal_vector)
                _norm_dist_pn = loader.get_mres_write_handle(normal_distance)

                @wp.func
                def device(index: Any):
                    _boundary_id = wp.neon_read(bc_mask_pn, index, 0)
                    if _boundary_id == wp.uint8(BC_SOLID):
                        return
                    if _boundary_id == wp.uint8(BC_SFV):
                        return
                    if not wp.neon_has_child(f_0_pn, index):
                        _f0_thread, _missing_mask = neon_get_thread_data(f_0_pn, missing_mask_pn, index)
                        collide_bc_accum(
                            index,
                            timestep,
                            _boundary_id,
                            _missing_mask,
                            f_0_pn,
                            f_1_pn,
                            _f0_thread,
                            omega,
                            num_levels,
                            level,
                            f_1_pn,
                            _rho0_pn,
                            _u0_pn,
                            _rho1_pn,
                            _u1_pn,
                            _relax_pn,
                            _norm_vec_pn,
                            _norm_dist_pn,
                        )
                    else:
                        for l in range(self.velocity_set.q):
                            wp.neon_write(f_1_pn, index, l, self.store_dtype(0))

                loader.declare_kernel(device)

            return ll

        @neon.Container.factory(name="stream_coarse_step_ABC")
        def stream_coarse_step_ABC(
            level: int,
            f_0_fd: Any,
            f_1_fd: Any,
            bc_mask_fd: Any,
            missing_mask_fd: Any,
            coalescence_factor: Any,
            timestep: int,
            _rho0: Any,
            _u0: Any,
            _rho1: Any,
            _u1: Any,
            _relax: Any,
            normal_vector: Any,
            normal_distance: Any,
        ):
            def ll(loader: neon.Loader):
                loader.set_mres_grid(bc_mask_fd.get_grid(), level)
                f_0_pn = loader.get_mres_read_handle(f_0_fd)
                f_1_pn = loader.get_mres_write_handle(f_1_fd)
                bc_mask_pn = loader.get_mres_read_handle(bc_mask_fd)
                missing_mask_pn = loader.get_mres_read_handle(missing_mask_fd)
                _rho0_pn = loader.get_mres_read_handle(_rho0)
                _u0_pn = loader.get_mres_read_handle(_u0)
                _rho1_pn = loader.get_mres_write_handle(_rho1)
                _u1_pn = loader.get_mres_write_handle(_u1)
                _relax_pn = loader.get_mres_write_handle(_relax)
                _norm_vec_pn = loader.get_mres_write_handle(normal_vector)
                _norm_dist_pn = loader.get_mres_write_handle(normal_distance)
                coalescence_factor_pn = loader.get_mres_read_handle(coalescence_factor)

                @wp.func
                def device(index: Any):
                    _boundary_id = wp.neon_read(bc_mask_pn, index, 0)
                    if _boundary_id == wp.uint8(BC_SOLID):
                        return
                    if wp.neon_has_child(f_0_pn, index):
                        return

                    _f0_thread, _missing_mask = neon_get_thread_data(f_0_pn, missing_mask_pn, index)
                    _f_post_collision = _f0_thread
                    _f_post_stream = neon_stream_explode_coalesce(index, f_0_pn, coalescence_factor_pn)

                    _f_post_stream = apply_bc_post_streaming(
                        index, timestep, _boundary_id, _missing_mask, f_0_pn, f_1_pn, _f_post_collision, _f_post_stream,
                        _rho0_pn, _u0_pn, _relax_pn,   _norm_vec_pn, _norm_dist_pn,
                    )
                    neon_apply_aux_recovery_bc(index, _boundary_id, _missing_mask, f_0_pn, f_1_pn)

                    for l in range(self.velocity_set.q):
                        wp.neon_write(f_1_pn, index, l, self.store_dtype(_f_post_stream[l]))
                    # Update rho / u fields for wall model
                    _rho, _u = self.macroscopic.neon_functional(_f_post_stream)
                    wp.neon_write(_rho1_pn, index, 0, self.store_dtype(_rho))
                    for d in range(self.velocity_set.d):
                        wp.neon_write(_u1_pn, index, d, self.store_dtype(_u[d]))

                loader.declare_kernel(device)

            return ll

        @neon.Container.factory(name="SFV_stream_coarse_step_ABC")
        def SFV_stream_coarse_step_ABC(
            level: int,
            f_0_fd: Any,
            f_1_fd: Any,
            bc_mask_fd: Any,
            missing_mask_fd: Any,
            coalescence_factor: Any,
            timestep: int,
            _rho0: Any,
            _u0: Any,
            _rho1: Any,
            _u1: Any,
            _relax: Any,
            normal_vector: Any,
            normal_distance: Any,
        ):
            """Stream on CFV voxels only — skips SFV and solid."""

            def ll(loader: neon.Loader):
                loader.set_mres_grid(bc_mask_fd.get_grid(), level)
                f_0_pn = loader.get_mres_read_handle(f_0_fd)
                f_1_pn = loader.get_mres_write_handle(f_1_fd)
                bc_mask_pn = loader.get_mres_read_handle(bc_mask_fd)
                missing_mask_pn = loader.get_mres_read_handle(missing_mask_fd)
                _rho0_pn = loader.get_mres_read_handle(_rho0)
                _u0_pn = loader.get_mres_read_handle(_u0)
                _rho1_pn = loader.get_mres_write_handle(_rho1)
                _u1_pn = loader.get_mres_write_handle(_u1)
                _relax_pn = loader.get_mres_write_handle(_relax)
                _norm_vec_pn = loader.get_mres_write_handle(normal_vector)
                _norm_dist_pn = loader.get_mres_write_handle(normal_distance)
                coalescence_factor_pn = loader.get_mres_read_handle(coalescence_factor)

                @wp.func
                def device(index: Any):
                    _boundary_id = wp.neon_read(bc_mask_pn, index, 0)
                    if _boundary_id == wp.uint8(BC_SFV):
                        return
                    if _boundary_id == wp.uint8(BC_SOLID):
                        return
                    if wp.neon_has_child(f_0_pn, index):
                        return

                    _f0_thread, _missing_mask = neon_get_thread_data(f_0_pn, missing_mask_pn, index)
                    _f_post_collision = _f0_thread
                    _f_post_stream = neon_stream_explode_coalesce(index, f_0_pn, coalescence_factor_pn)

                    _f_post_stream = apply_bc_post_streaming(
                        index, timestep, _boundary_id, _missing_mask, f_0_pn, f_1_pn, _f_post_collision, _f_post_stream,
                        _rho0_pn, _u0_pn, _relax_pn,   _norm_vec_pn, _norm_dist_pn,
                    )
                    neon_apply_aux_recovery_bc(index, _boundary_id, _missing_mask, f_0_pn, f_1_pn)

                    for l in range(self.velocity_set.q):
                        wp.neon_write(f_1_pn, index, l, self.store_dtype(_f_post_stream[l]))

                    # Update rho / u fields for wall model
                    _rho, _u = self.macroscopic.neon_functional(_f_post_stream)
                    wp.neon_write(_rho1_pn, index, 0, self.store_dtype(_rho))
                    for d in range(self.velocity_set.d):
                        wp.neon_write(_u1_pn, index, d, self.store_dtype(_u[d]))

                loader.declare_kernel(device)

            return ll

        @neon.Container.factory(name="SFV_reset_bc_mask")
        def SFV_reset_bc_mask(
            level: int,
            f_0_fd: Any,
            f_1_fd: Any,
            bc_mask_fd: Any,
            missing_mask_fd: Any,
            _rho0: Any,
            _u0: Any,
            _rho1: Any,
            _u1: Any,
            _relax: Any,
            normal_vector: Any,
            normal_distance: Any,
        ):
            """
            Setting the BC type to BC_SFV
            """

            def ll_stream_coarse(loader: neon.Loader):
                loader.set_mres_grid(bc_mask_fd.get_grid(), level)

                f_0_pn = loader.get_mres_read_handle(f_0_fd)

                bc_mask_pn = loader.get_mres_read_handle(bc_mask_fd)
                missing_mask_pn = loader.get_mres_read_handle(missing_mask_fd)
                _rho0_pn = loader.get_mres_read_handle(_rho0)
                _u0_pn = loader.get_mres_read_handle(_u0)
                _rho1_pn = loader.get_mres_write_handle(_rho1)
                _u1_pn = loader.get_mres_write_handle(_u1)
                _relax_pn = loader.get_mres_write_handle(_relax)
                _norm_vec_pn = loader.get_mres_write_handle(normal_vector)
                _norm_dist_pn = loader.get_mres_write_handle(normal_distance)

                _c = self.velocity_set.c

                @wp.func
                def cl_stream_coarse(index: Any):
                    _boundary_id = wp.neon_read(bc_mask_pn, index, 0)
                    if _boundary_id == wp.uint8(BC_SOLID):
                        return
                    if _boundary_id != 0:
                        return

                    if wp.neon_has_child(f_0_pn, index):
                        # we are a halo cell so we just exit
                        return

                    # do stream normally
                    _missing_mask = _missing_mask_vec()
                    _f0_thread, _missing_mask = neon_get_thread_data(f_0_pn, missing_mask_pn, index)
                    _f_post_collision = _f0_thread
                    _f_post_stream = self.stream.neon_functional(f_0_pn, index)

                    for l in range(self.velocity_set.q):
                        if l == lattice_central_index:
                            continue

                        pull_direction = wp.neon_ngh_idx(wp.int8(-_c[0, l]), wp.int8(-_c[1, l]), wp.int8(-_c[2, l]))

                        has_ngh_at_same_level = wp.bool(False)
                        wp.neon_read_ngh(f_0_pn, index, pull_direction, l, self.store_dtype(0), has_ngh_at_same_level)

                        if not wp.neon_has_finer_ngh(f_0_pn, index, pull_direction):
                            if not has_ngh_at_same_level:
                                if wp.neon_has_parent(f_0_pn, index):
                                    has_a_coarser_ngh = wp.bool(False)
                                    wp.neon_lbm_read_coarser_ngh(f_0_pn, index, pull_direction, l, self.store_dtype(0), has_a_coarser_ngh)
                                    if has_a_coarser_ngh:
                                        # Explosion: not an SFV
                                        return
                        else:
                            if has_ngh_at_same_level:
                                # Coalescence: not an SFV
                                return

                    # Voxel is a pure fluid cell with no multi-resolution interactions — mark as SFV
                    wp.neon_write(bc_mask_pn, index, 0, wp.uint8(BC_SFV))
                    # Update rho / u fields for wall model
                    _rho, _u = self.macroscopic.neon_functional(_f_post_stream)
                    wp.neon_write(_rho1_pn, index, 0, self.store_dtype(_rho))
                    for d in range(self.velocity_set.d):
                        wp.neon_write(_u1_pn, index, d, self.store_dtype(_u[d]))
                    
                    

                loader.declare_kernel(cl_stream_coarse)

            return ll_stream_coarse

        @neon.Container.factory(name="SFV_stream_coarse_step")
        def SFV_stream_coarse_step(level: int, f_0_fd: Any, f_1_fd: Any, bc_mask_fd: Any, missing_mask_fd: Any, _rho0: Any, _u0: Any,  _rho1: Any,  _u1: Any,  _relax: Any, normal_vector: Any,normal_distance: Any):
            def ll_stream_coarse(loader: neon.Loader):
                loader.set_mres_grid(bc_mask_fd.get_grid(), level)

                f_0_pn = loader.get_mres_read_handle(f_0_fd)
                f_1_pn = loader.get_mres_write_handle(f_1_fd)

                bc_mask_pn = loader.get_mres_read_handle(bc_mask_fd)
                missing_mask_pn = loader.get_mres_read_handle(missing_mask_fd)
                _rho0_pn = loader.get_mres_read_handle(_rho0)
                _u0_pn = loader.get_mres_read_handle(_u0)
                _rho1_pn = loader.get_mres_write_handle(_rho1)
                _u1_pn = loader.get_mres_write_handle(_u1)
                _relax_pn = loader.get_mres_write_handle(_relax)
                _norm_vec_pn = loader.get_mres_write_handle(normal_vector)
                _norm_dist_pn = loader.get_mres_write_handle(normal_distance)

                _c = self.velocity_set.c

                @wp.func
                def cl_stream_coarse(index: Any):
                    _boundary_id = wp.neon_read(bc_mask_pn, index, 0)
                    if _boundary_id != wp.uint8(BC_SFV):
                        return
                    # BC_SFV voxel type:
                    #   - They are not BC voxels
                    #   - They are not on a resolution jump -> they do not do coalescence or explosion
                    #   - They are not mr halo cells

                    # Stream-only kernel: nothing downstream reads the local
                    # pre-streaming populations or the missing mask.
                    _f_post_stream = self.stream.neon_functional(f_0_pn, index)

                    for l in range(self.velocity_set.q):
                        wp.neon_write(f_1_pn, index, l, self.store_dtype(_f_post_stream[l]))
                    
                    # Update rho / u fields for wall model
                    _rho, _u = self.macroscopic.neon_functional(_f_post_stream)
                    wp.neon_write(_rho1_pn, index, 0, self.store_dtype(_rho))
                    for d in range(self.velocity_set.d):
                        wp.neon_write(_u1_pn, index, d, self.store_dtype(_u[d]))

                loader.declare_kernel(cl_stream_coarse)

            return ll_stream_coarse

        @wp.func
        def neon_stream_finest_with_explosion(
            index: Any,
            f_0_pn: Any,
            explosion_src_pn: Any,
        ):
            _f_post_stream = self.stream.neon_functional(f_0_pn, index)

            for l in range(self.velocity_set.q):
                if l == lattice_central_index:
                    continue

                pull_direction = wp.neon_ngh_idx(wp.int8(-_c[0, l]), wp.int8(-_c[1, l]), wp.int8(-_c[2, l]))

                has_ngh_at_same_level = wp.bool(False)
                wp.neon_read_ngh(f_0_pn, index, pull_direction, l, self.store_dtype(0), has_ngh_at_same_level)

                if not has_ngh_at_same_level:
                    # No same-level ngh — could we have a coarser-level ngh?
                    if wp.neon_has_parent(f_0_pn, index):
                        # Parent exists — try to read the exploded population from the coarser level
                        has_a_coarser_ngh = wp.bool(False)
                        exploded_pop = wp.neon_lbm_read_coarser_ngh(
                            explosion_src_pn, index, pull_direction, l, self.store_dtype(0), has_a_coarser_ngh
                        )
                        if has_a_coarser_ngh:
                            # No finer ngh in pull direction, no same-level ngh,
                            # but a parent (ghost cell) exists with a coarser ngh
                            # -> Explosion: read the exploded population from the
                            #    coarser level's halo.
                            _f_post_stream[l] = self.compute_dtype(exploded_pop)

            return _f_post_stream

        @neon.Container.factory(name="finest_fused_pull")
        def finest_fused_pull(
            level: int,
            f_0_fd: Any,
            f_1_fd: Any,
            bc_mask_fd: Any,
            missing_mask_fd: Any,
            omega: Any,
            timestep: Any,
            is_f1_the_explosion_src_field: bool,
            _rho0: Any,
            _u0: Any,
            _rho1: Any,
            _u1: Any,
            _relax: Any,
            normal_vector: Any,
            normal_distance: Any,
        ):
            if level != 0:
                raise Exception("Only the finest level is supported for now")
            num_levels = f_0_fd.get_grid().num_levels

            def ll(loader: neon.Loader):
                loader.set_mres_grid(bc_mask_fd.get_grid(), level)
                if level + 1 < f_0_fd.get_grid().num_levels:
                    f_0_pn = loader.get_mres_write_handle(f_0_fd, neon.Loader.Operation.stencil_up)
                    f_1_pn = loader.get_mres_write_handle(f_1_fd, neon.Loader.Operation.stencil_up)
                else:
                    f_0_pn = loader.get_mres_read_handle(f_0_fd)
                    f_1_pn = loader.get_mres_write_handle(f_1_fd)
                bc_mask_pn = loader.get_mres_read_handle(bc_mask_fd)
                missing_mask_pn = loader.get_mres_read_handle(missing_mask_fd)
                _rho0_pn = loader.get_mres_read_handle(_rho0)
                _u0_pn = loader.get_mres_read_handle(_u0)
                _rho1_pn = loader.get_mres_write_handle(_rho1)
                _u1_pn = loader.get_mres_write_handle(_u1)
                _relax_pn = loader.get_mres_write_handle(_relax)
                _norm_vec_pn = loader.get_mres_write_handle(normal_vector)
                _norm_dist_pn = loader.get_mres_write_handle(normal_distance)
                explosion_src_pn = f_1_pn if is_f1_the_explosion_src_field else f_0_pn
                accumulation_pn = f_1_pn if is_f1_the_explosion_src_field else f_0_pn

                @wp.func
                def device(index: Any):
                    _boundary_id = wp.neon_read(bc_mask_pn, index, 0)
                    if _boundary_id == wp.uint8(BC_SOLID):
                        return
                    if wp.neon_has_child(f_0_pn, index):
                        return

                    _f0_thread, _missing_mask = neon_get_thread_data(f_0_pn, missing_mask_pn, index)
                    _f_post_collision = _f0_thread
                    _f_post_stream = neon_stream_finest_with_explosion(index, f_0_pn, explosion_src_pn)

                    _f_post_stream = apply_bc_post_streaming(
                        index, timestep, _boundary_id, _missing_mask, f_0_pn, f_1_pn, _f_post_collision, _f_post_stream,
                        _rho0_pn, _u0_pn, _relax_pn,  _norm_vec_pn, _norm_dist_pn,
                    )

                    collide_bc_accum(
                        index,
                        timestep,
                        _boundary_id,
                        _missing_mask,
                        f_0_pn,
                        f_1_pn,
                        _f_post_stream,
                        omega,
                        num_levels,
                        level,
                        accumulation_pn,
                        _rho0_pn,
                        _u0_pn,
                        _rho1_pn,
                        _u1_pn,
                        _relax_pn,
                        _norm_vec_pn,
                        _norm_dist_pn,
                    )

                loader.declare_kernel(device)

            return ll

        @neon.Container.factory(name="mark_needs_mres")
        def mark_needs_mres(level: int, f_0_fd: Any, bc_mask_fd: Any, needs_mres_fd: Any):
            """Precompute, once, which voxels can participate in a level jump.

            Both multiresolution stages in the finest fused kernel only fire
            where the voxel actually borders a coarser region:

            * explosion -- ``neon_stream_finest_with_explosion`` reads the
              coarser level only where there is no same-level neighbour *and*
              ``neon_lbm_read_coarser_ngh`` finds a coarser one;
            * coalescence -- ``neon_mres_lbm_store_op`` reaches its ``atomicAdd``
              only inside ``if (!pout.isActive(cn))`` with an active uncle.

            This reuses the exact predicate ``SFV_reset_bc_mask`` applies to
            plain fluid voxels, extended to BC-carrying voxels -- that routine
            returns early on ``_boundary_id != 0``, so wall voxels are never
            evaluated and end up in the expensive class by default.

            A missing same-level neighbour alone is NOT a valid test: it is also
            true all over the body surface, where the interior is not meshed.
            Using it flags nearly every wall voxel and gains nothing (measured).

            D3Q27's direction set is {-1,0,1}^3, so one sweep covers every
            offset, and push directions are the negation of pull directions so
            the union covers both.  Topology is static, so this is computed once
            at setup rather than rediscovered 27 times per voxel per timestep.
            """
            def ll(loader: neon.Loader):
                loader.set_mres_grid(bc_mask_fd.get_grid(), level)
                f_0_pn = loader.get_mres_read_handle(f_0_fd)
                bc_mask_pn = loader.get_mres_read_handle(bc_mask_fd)
                needs_pn = loader.get_mres_write_handle(needs_mres_fd)

                @wp.func
                def cl(index: Any):
                    _boundary_id = wp.neon_read(bc_mask_pn, index, 0)
                    if _boundary_id == wp.uint8(BC_SOLID):
                        wp.neon_write(needs_pn, index, 0, wp.uint8(0))
                        return

                    # Same test SFV_reset_bc_mask uses to classify a voxel as
                    # SFV, but applied to BC-carrying voxels too -- that routine
                    # returns early on `_boundary_id != 0`, so wall voxels are
                    # never evaluated for whether they touch a level jump.
                    #
                    # A missing same-level neighbour is NOT sufficient: it also
                    # happens at the body surface, where the interior is not
                    # meshed.  The coarser cell must actually exist, which is
                    # what neon_lbm_read_coarser_ngh reports.
                    _flag = wp.uint8(0)
                    for l in range(self.velocity_set.q):
                        if l == lattice_central_index:
                            continue

                        pull_direction = wp.neon_ngh_idx(wp.int8(-_c[0, l]), wp.int8(-_c[1, l]), wp.int8(-_c[2, l]))

                        has_ngh_at_same_level = wp.bool(False)
                        wp.neon_read_ngh(f_0_pn, index, pull_direction, l, self.store_dtype(0), has_ngh_at_same_level)

                        if not wp.neon_has_finer_ngh(f_0_pn, index, pull_direction):
                            if not has_ngh_at_same_level:
                                if wp.neon_has_parent(f_0_pn, index):
                                    has_a_coarser_ngh = wp.bool(False)
                                    wp.neon_lbm_read_coarser_ngh(
                                        f_0_pn, index, pull_direction, l, self.store_dtype(0), has_a_coarser_ngh
                                    )
                                    if has_a_coarser_ngh:
                                        _flag = wp.uint8(1)  # explosion
                        else:
                            if has_ngh_at_same_level:
                                _flag = wp.uint8(1)  # coalescence

                    # A voxel with children is a halo cell of a finer level and
                    # is skipped by the fused kernel anyway, but flag it so the
                    # predicate stays conservative if that ever changes.
                    if wp.neon_has_child(f_0_pn, index):
                        _flag = wp.uint8(1)

                    wp.neon_write(needs_pn, index, 0, _flag)

                loader.declare_kernel(cl)

            return ll

        @neon.Container.factory(name="CFV_finest_fused_pull")
        def CFV_finest_fused_pull(
            level: int,
            f_0_fd: Any,
            f_1_fd: Any,
            bc_mask_fd: Any,
            missing_mask_fd: Any,
            omega: Any,
            timestep: Any,
            is_f1_the_explosion_src_field: bool,
            _rho0: Any,
            _u0: Any,
            _rho1: Any,
            _u1: Any,
            _relax: Any,
            normal_vector: Any,
            normal_distance: Any,
        ):
            """Fused stream+collide on CFV voxels at the finest level — skips SFV and solid."""
            if level != 0:
                raise Exception("Only the finest level is supported for now")
            num_levels = f_0_fd.get_grid().num_levels

            def ll(loader: neon.Loader):
                loader.set_mres_grid(bc_mask_fd.get_grid(), level)
                if level + 1 < f_0_fd.get_grid().num_levels:
                    f_0_pn = loader.get_mres_write_handle(f_0_fd, neon.Loader.Operation.stencil_up)
                    f_1_pn = loader.get_mres_write_handle(f_1_fd, neon.Loader.Operation.stencil_up)
                else:
                    f_0_pn = loader.get_mres_read_handle(f_0_fd)
                    f_1_pn = loader.get_mres_write_handle(f_1_fd)
                bc_mask_pn = loader.get_mres_read_handle(bc_mask_fd)
                missing_mask_pn = loader.get_mres_read_handle(missing_mask_fd)
                _rho0_pn = loader.get_mres_read_handle(_rho0)
                _u0_pn = loader.get_mres_read_handle(_u0)
                _rho1_pn = loader.get_mres_write_handle(_rho1)
                _u1_pn = loader.get_mres_write_handle(_u1)
                _relax_pn = loader.get_mres_write_handle(_relax)
                _norm_vec_pn = loader.get_mres_write_handle(normal_vector)
                _norm_dist_pn = loader.get_mres_write_handle(normal_distance)
                explosion_src_pn = f_1_pn if is_f1_the_explosion_src_field else f_0_pn
                accumulation_pn = f_1_pn if is_f1_the_explosion_src_field else f_0_pn


                @wp.func
                def device(index: Any):
                    _boundary_id = wp.neon_read(bc_mask_pn, index, 0)
                    if _boundary_id == wp.uint8(BC_SOLID):
                        return
                    if _boundary_id == wp.uint8(BC_SFV):
                        return
                    if wp.neon_has_child(f_0_pn, index):
                        return

                    if wp.static(_abl_thread_data):
                        _f0_thread = _f_vec()
                        _missing_mask = _missing_mask_vec()
                    else:
                        _f0_thread, _missing_mask = neon_get_thread_data(f_0_pn, missing_mask_pn, index)
                    _f_post_collision = _f0_thread

                    if wp.static(_abl_explosion):
                        _f_post_stream = self.stream.neon_functional(f_0_pn, index)
                    else:
                        _f_post_stream = neon_stream_finest_with_explosion(index, f_0_pn, explosion_src_pn)

                    if wp.static(not _abl_bc):
                        _f_post_stream = apply_bc_post_streaming(
                            index, timestep, _boundary_id, _missing_mask, f_0_pn, f_1_pn, _f_post_collision, _f_post_stream,
                            _rho0_pn, _u0_pn, _relax_pn,  _norm_vec_pn, _norm_dist_pn,
                        )

                    collide_cfv_finest(
                        index,
                        timestep,
                        _boundary_id,
                        _missing_mask,
                        f_0_pn,
                        f_1_pn,
                        _f_post_stream,
                        omega,
                        num_levels,
                        level,
                        accumulation_pn,
                        _rho0_pn,
                        _u0_pn,
                        _rho1_pn,
                        _u1_pn,
                        _relax_pn,
                        _norm_vec_pn,
                        _norm_dist_pn,
                    )

                loader.declare_kernel(device)

            return ll

        def make_cfv_finest_split(does_mres: bool, does_bc: bool, name: str):
            """Build one of the three specialised finest-level CFV containers.

            Splitting into separate containers rather than branching inside one
            kernel is deliberate, and measured: a runtime branch inlines every
            variant into a single kernel, so all CFV voxels pay the register and
            occupancy cost while only fully-uniform warps see the saving.  At
            ~8% CFV scattered over 64-thread blocks that is a net loss
            (CFV/SFV 2.035 branched vs 1.853 unbranched vs 1.548 split).

            The three classes partition CFV exactly:

            ==========================  ==============================  ====  ====
            container                   selector                        BC    mres
            ==========================  ==============================  ====  ====
            CFV_BC_finest_fused_pull    needs_mres == 0                 yes   no
            CFV_MRES_NOBC_...           needs_mres != 0, bc_mask == 0    no   yes
            CFV_MRES_BC_...             needs_mres != 0, bc_mask != 0   yes   yes
            ==========================  ==============================  ====  ====

            The middle class is ~99.8% of the multiresolution voxels and carries
            no boundary condition, so it is compiled with ``do_bc=False``: none
            of the registered BC functionals are inlined into it at all.  The
            handful of voxels that are both a wall and a level jump (~2,300 at
            the finest level) fall into the third container and keep the full
            path -- the split is on measured state, never on an assumption that
            those two populations are disjoint.

            With ``do_bc=False`` the local pre-streaming populations and the
            missing mask are dead (they exist only as ``f_pre`` for the BC
            chain), so that kernel skips ``neon_get_thread_data`` entirely.
            """

            @neon.Container.factory(name=name)
            def cfv_finest_split(
                level: int,
                f_0_fd: Any,
                f_1_fd: Any,
                bc_mask_fd: Any,
                missing_mask_fd: Any,
                omega: Any,
                timestep: Any,
                is_f1_the_explosion_src_field: bool,
                _rho0: Any,
                _u0: Any,
                _rho1: Any,
                _u1: Any,
                _relax: Any,
                normal_vector: Any,
                normal_distance: Any,
            ):
                if level != 0:
                    raise Exception("Only the finest level is supported for now")
                num_levels = f_0_fd.get_grid().num_levels

                def ll(loader: neon.Loader):
                    loader.set_mres_grid(bc_mask_fd.get_grid(), level)
                    if level + 1 < f_0_fd.get_grid().num_levels:
                        f_0_pn = loader.get_mres_write_handle(f_0_fd, neon.Loader.Operation.stencil_up)
                        f_1_pn = loader.get_mres_write_handle(f_1_fd, neon.Loader.Operation.stencil_up)
                    else:
                        f_0_pn = loader.get_mres_read_handle(f_0_fd)
                        f_1_pn = loader.get_mres_write_handle(f_1_fd)
                    bc_mask_pn = loader.get_mres_read_handle(bc_mask_fd)
                    missing_mask_pn = loader.get_mres_read_handle(missing_mask_fd)
                    needs_mres_pn = loader.get_mres_read_handle(self.needs_mres)
                    _rho0_pn = loader.get_mres_read_handle(_rho0)
                    _u0_pn = loader.get_mres_read_handle(_u0)
                    _rho1_pn = loader.get_mres_write_handle(_rho1)
                    _u1_pn = loader.get_mres_write_handle(_u1)
                    _relax_pn = loader.get_mres_write_handle(_relax)
                    _norm_vec_pn = loader.get_mres_write_handle(normal_vector)
                    _norm_dist_pn = loader.get_mres_write_handle(normal_distance)
                    explosion_src_pn = f_1_pn if is_f1_the_explosion_src_field else f_0_pn
                    accumulation_pn = f_1_pn if is_f1_the_explosion_src_field else f_0_pn

                    @wp.func
                    def device(index: Any):
                        _boundary_id = wp.neon_read(bc_mask_pn, index, 0)
                        if _boundary_id == wp.uint8(BC_SOLID):
                            return
                        if _boundary_id == wp.uint8(BC_SFV):
                            return
                        if wp.neon_has_child(f_0_pn, index):
                            return

                        _needs_mres = wp.neon_read(needs_mres_pn, index, 0)
                        if wp.static(does_mres):
                            if _needs_mres == wp.uint8(0):
                                return
                            if wp.static(does_bc):
                                if _boundary_id == wp.uint8(BC_NONE):
                                    return
                            else:
                                if _boundary_id != wp.uint8(BC_NONE):
                                    return
                        else:
                            if _needs_mres != wp.uint8(0):
                                return

                        if wp.static(does_bc):
                            _f0_thread, _missing_mask = neon_get_thread_data(f_0_pn, missing_mask_pn, index)
                        else:
                            # Dead without the BC chain: they exist only as f_pre.
                            _f0_thread = _f_vec()
                            _missing_mask = _missing_mask_vec()
                        _f_post_collision = _f0_thread

                        if wp.static(does_mres):
                            _f_post_stream = neon_stream_finest_with_explosion(index, f_0_pn, explosion_src_pn)
                        else:
                            _f_post_stream = self.stream.neon_functional(f_0_pn, index)

                        if wp.static(does_bc):
                            _f_post_stream = apply_bc_post_streaming(
                                index, timestep, _boundary_id, _missing_mask, f_0_pn, f_1_pn, _f_post_collision, _f_post_stream,
                                _rho0_pn, _u0_pn, _relax_pn, _norm_vec_pn, _norm_dist_pn,
                            )

                        if wp.static(does_mres and does_bc):
                            collide_bc_accum(
                                index, timestep, _boundary_id, _missing_mask, f_0_pn, f_1_pn, _f_post_stream,
                                omega, num_levels, level, accumulation_pn,
                                _rho0_pn, _u0_pn, _rho1_pn, _u1_pn, _relax_pn, _norm_vec_pn, _norm_dist_pn,
                            )
                        if wp.static(does_mres and not does_bc):
                            collide_accum_only(
                                index, timestep, _boundary_id, _missing_mask, f_0_pn, f_1_pn, _f_post_stream,
                                omega, num_levels, level, accumulation_pn,
                                _rho0_pn, _u0_pn, _rho1_pn, _u1_pn, _relax_pn, _norm_vec_pn, _norm_dist_pn,
                            )
                        if wp.static(not does_mres):
                            collide_bc_only(
                                index, timestep, _boundary_id, _missing_mask, f_0_pn, f_1_pn, _f_post_stream,
                                omega, num_levels, level, accumulation_pn,
                                _rho0_pn, _u0_pn, _rho1_pn, _u1_pn, _relax_pn, _norm_vec_pn, _norm_dist_pn,
                            )

                    loader.declare_kernel(device)

                return ll

            return cfv_finest_split

        CFV_BC_finest_fused_pull = make_cfv_finest_split(
            does_mres=False, does_bc=True, name="CFV_BC_finest_fused_pull"
        )
        CFV_MRES_NOBC_finest_fused_pull = make_cfv_finest_split(
            does_mres=True, does_bc=False, name="CFV_MRES_NOBC_finest_fused_pull"
        )
        CFV_MRES_BC_finest_fused_pull = make_cfv_finest_split(
            does_mres=True, does_bc=True, name="CFV_MRES_BC_finest_fused_pull"
        )

        @neon.Container.factory(name="SFV_finest_fused_pull")
        def SFV_finest_fused_pull(level: int, f_0_fd: Any, f_1_fd: Any, bc_mask_fd: Any, missing_mask_fd: Any, omega: Any, _rho0: Any, _u0: Any,  _rho1: Any,  _u1: Any,  _relax: Any, normal_vector: Any,normal_distance: Any):
            """Fused stream+collide on SFV voxels at the finest level — no BCs, no explosion."""
            if level != 0:
                raise Exception("Only the finest level is supported for now")

            def ll(loader: neon.Loader):
                loader.set_mres_grid(bc_mask_fd.get_grid(), level)
                f_0_pn = loader.get_mres_read_handle(f_0_fd)
                f_1_pn = loader.get_mres_write_handle(f_1_fd)
                bc_mask_pn = loader.get_mres_read_handle(bc_mask_fd)
                missing_mask_pn = loader.get_mres_read_handle(missing_mask_fd)
                _rho0_pn = loader.get_mres_read_handle(_rho0)
                _u0_pn = loader.get_mres_read_handle(_u0)
                _rho1_pn = loader.get_mres_write_handle(_rho1)
                _u1_pn = loader.get_mres_write_handle(_u1)
                _relax_pn = loader.get_mres_write_handle(_relax)
                _norm_vec_pn = loader.get_mres_write_handle(normal_vector)
                _norm_dist_pn = loader.get_mres_write_handle(normal_distance)

                @wp.func
                def device(index: Any):
                    _boundary_id = wp.neon_read(bc_mask_pn, index, 0)
                    if _boundary_id != wp.uint8(BC_SFV):
                        return
                    # SFV voxels carry no BC and no explosion, so the streamed
                    # populations are the only collision input: neither the local
                    # pre-streaming populations nor the missing mask are read.
                    _f_post_stream = self.stream.neon_functional(f_0_pn, index)
                    _no_missing_mask = _missing_mask_vec()
                    collide_simple(
                        index,
                        0,
                        _boundary_id,
                        _no_missing_mask,
                        f_0_pn,
                        f_1_pn,
                        _f_post_stream,
                        omega,
                        0,
                        0,
                        f_1_pn,
                        _rho0_pn,
                        _u0_pn,
                        _rho1_pn,
                        _u1_pn,
                        _relax_pn,
                        _norm_vec_pn,
                        _norm_dist_pn,
                    )

                loader.declare_kernel(device)

            return ll

        return None, {
            "collide_coarse": collide_coarse,
            "stream_coarse_step_ABC": stream_coarse_step_ABC,
            "finest_fused_pull": finest_fused_pull,
            "CFV_finest_fused_pull": CFV_finest_fused_pull,
            "SFV_finest_fused_pull": SFV_finest_fused_pull,
            "SFV_reset_bc_mask": SFV_reset_bc_mask,
            "mark_needs_mres": mark_needs_mres,
            "CFV_BC_finest_fused_pull": CFV_BC_finest_fused_pull,
            "CFV_MRES_NOBC_finest_fused_pull": CFV_MRES_NOBC_finest_fused_pull,
            "CFV_MRES_BC_finest_fused_pull": CFV_MRES_BC_finest_fused_pull,
            "CFV_collide_coarse": CFV_collide_coarse,
            "SFV_collide_coarse": SFV_collide_coarse,
            "SFV_stream_coarse_step_ABC": SFV_stream_coarse_step_ABC,
            "SFV_stream_coarse_step": SFV_stream_coarse_step,
        }

    def add_to_app(self, **kwargs):
        """Append a container invocation to the Neon skeleton application list.

        Required keyword arguments are ``op_name`` (str) and ``app`` (list).
        All remaining keyword arguments are forwarded to the container
        factory for the given ``op_name``.  Argument validation is performed
        before the call, and a ``ValueError`` is raised on mismatch.
        """
        import inspect

        def validate_kwargs_forward(func, kwargs):
            """
            Check whether `func(**kwargs)` would be valid,
            and return *all* the issues instead of raising on the first one.

            Returns a dict; empty dict means "everything is OK".
            """
            sig = inspect.signature(func)
            params = sig.parameters

            errors = {}

            # --- 1. Positional-only required params (cannot be given via kwargs) ---
            pos_only_required = [name for name, p in params.items() if p.kind == inspect.Parameter.POSITIONAL_ONLY and p.default is inspect._empty]
            if pos_only_required:
                errors["positional_only_required"] = pos_only_required

            # --- 2. Unexpected kwargs (if no **kwargs in target) ---
            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
            if not has_var_kw:
                allowed_kw = {
                    name
                    for name, p in params.items()
                    if p.kind
                    in (
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.KEYWORD_ONLY,
                    )
                }
                unexpected = sorted(set(kwargs) - allowed_kw)
                if unexpected:
                    errors["unexpected_kwargs"] = unexpected

            # --- 3. Missing required keyword-passable params ---
            missing_required = [
                name
                for name, p in params.items()
                if p.kind
                in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
                and p.default is inspect._empty  # no default
                and name not in kwargs  # not provided
            ]
            if missing_required:
                errors["missing_required"] = missing_required

            return errors

        container_generator = None
        try:
            op_name = kwargs.pop("op_name")
            app = kwargs.pop("app")
        except KeyError:
            raise ValueError("op_name and app must be provided as keyword arguments")

        try:
            container_generator = self.neon_container[op_name]
        except KeyError:
            raise ValueError(f"Operator {op_name} not found in neon container. Available operators: {list(self.neon_container.keys())}")

        errors = validate_kwargs_forward(container_generator, kwargs)
        if errors:
            raise ValueError(f"Cannot forward kwargs to target: {errors}")

        nvtx.push_range(f"New Container {op_name}", color="yellow")
        app.append(container_generator(**kwargs))
        nvtx.pop_range()

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_launch(self, f_0, f_1, bc_mask, missing_mask, omega, timestep):
        raise NotImplementedError("Use MultiresSimulationManager.step() instead of launching this stepper directly.")
