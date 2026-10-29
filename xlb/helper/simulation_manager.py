"""
High-level simulation manager for multi-resolution LBM on the Neon backend.

:class:`MultiresSimulationManager` orchestrates the complete simulation
lifecycle: field allocation, boundary-condition setup, coalescence-factor
precomputation, and the recursive time-stepping skeleton that correctly
interleaves coarse and fine grid updates.
"""

import warp as wp
from xlb.operator.stepper import MultiresIncompressibleNavierStokesStepper
from xlb.operator.macroscopic import MultiresMacroscopic
from xlb.mres_perf_optimization_type import MresPerfOptimizationType
from xlb.precision_policy import Precision


class MultiresSimulationManager(MultiresIncompressibleNavierStokesStepper):
    """Orchestrates multi-resolution LBM simulations on the Neon backend.

    Inherits from :class:`MultiresIncompressibleNavierStokesStepper` and
    adds field management, omega computation across levels, and the
    recursive skeleton builder that encodes the multi-resolution
    time-stepping order.

    Parameters
    ----------
    omega_finest : float
        Relaxation parameter at the finest grid level.
    grid : NeonMultiresGrid
        Multi-resolution grid.
    boundary_conditions : list of BoundaryCondition
        Boundary conditions to apply.
    collision_type : str
        ``"BGK"`` or ``"KBC"``.
    forcing_scheme : str
        Forcing scheme (used only when *force_vector* is given).
    force_vector : array-like, optional
        External body force.
    initializer : Operator, optional
        Custom initializer for distribution functions.  If ``None``
        the default equilibrium initialization is used.
    mres_perf_opt : MresPerfOptimizationType
        Performance optimization strategy.
    """

    def __init__(
        self,
        omega_finest,
        grid,
        boundary_conditions=[],
        collision_type="BGK",
        forcing_scheme="exact_difference",
        force_vector=None,
        initializer=None,
        mres_perf_opt: MresPerfOptimizationType = MresPerfOptimizationType.NAIVE_COLLIDE_STREAM,
        smagorinsky_constant = 0.0
    ):
        super().__init__(grid, boundary_conditions, collision_type, forcing_scheme, force_vector, smagorinsky_constant)

        self.initializer = initializer
        self.count_levels = grid.count_levels
        self.omega_list = [self.compute_omega(omega_finest, level) for level in range(self.count_levels)]
        self.mres_perf_opt = mres_perf_opt
        # Create fields
        self.rho = grid.create_field(cardinality=1, dtype=self.precision_policy.store_precision)
        self.u = grid.create_field(cardinality=3, dtype=self.precision_policy.store_precision)
        self.rho0 = grid.create_field(cardinality=1, dtype=self.precision_policy.store_precision)
        self.u0 = grid.create_field(cardinality=3, dtype=self.precision_policy.store_precision)
        self.rho1 = grid.create_field(cardinality=1, dtype=self.precision_policy.store_precision)
        self.u1 = grid.create_field(cardinality=3, dtype=self.precision_policy.store_precision)
        self.relax = grid.create_field(cardinality=1, dtype=self.precision_policy.store_precision)
        self.coalescence_factor = grid.create_field(cardinality=self.velocity_set.q, dtype=self.precision_policy.store_precision)

        # Dedicated integer restriction accumulator.  Fine-to-coarse population
        # deposits are quantized and atomically accumulated here, never inside
        # f_0/f_1.  This keeps the physical population fields valid FP32/FP64
        # values throughout streaming, explosion, boundary processing, and I/O.
        import neon

        if self.precision_policy.store_precision in (Precision.FP32, Precision.FP64):
            # Always use a 64-bit integer accumulator.  For FP32 populations this
            # permits a 2^48 scale, which preserves every meaningful FP32 LBM
            # population exactly instead of discarding low mantissa bits as the
            # prior int32/2^27 path did.
            restriction_accumulator_dtype = wp.int64
        else:
            raise ValueError(
                "deterministic fine-to-coarse restriction requires FP32 or FP64 "
                "population storage; Neon multires FP16 storage is not supported"
            )

        # NeonMultiresGrid.create_field currently accepts only XLB floating/byte
        # Precision values.  Allocate the integer mField through the wrapped
        # Neon mGrid using the same lifecycle as create_field().
        self.restriction_accumulator = grid.grid.new_field(
            cardinality=self.velocity_set.q,
            dtype=restriction_accumulator_dtype,
            memory_type=neon.MemoryType.device(),
        )
        for level in range(self.count_levels):
            self.restriction_accumulator.zero_run(level, stream_idx=0)

        # Dedicated ExtrapolationOutflowBC storage.  The BC no longer hides
        # auxiliary values inside physical distribution populations because
        # those populations participate in multires coalescence/explosion.
        #
        # Allocate only when ExtrapolationOutflowBC is present and only the
        # number of crossing populations required by a planar face
        # (D3Q27 -> 9 rather than 27).  Neon mFields inherit the parent mGrid
        # sparsity, so this first correctness pass still allocates those nine
        # components on active voxels; only actual Extrap BC cells are accessed.
        extrap_bcs = [
            bc for bc in self.boundary_conditions
            if getattr(bc, "uses_dedicated_aux_field", False)
        ]
        self.extrap_aux_levels = []
        if extrap_bcs:
            import neon

            aux_cardinality = max(
                int(bc.extrap_aux_cardinality) for bc in extrap_bcs
            )
            self.extrap_aux = grid.create_field(
                cardinality=aux_cardinality,
                dtype=self.precision_policy.store_precision,
                neon_memory_type=neon.MemoryType.device(),
            )

            # We already know the explicit outlet indices.  Use them to avoid
            # launching the one-time initialization kernel on levels where this
            # BC does not exist.  This reduces initialization work (allocation
            # remains tied to the parent mGrid; see comment above).
            levels = set()
            for bc in extrap_bcs:
                if bc.indices is None:
                    levels.update(range(self.count_levels))
                    continue
                for level, level_indices in enumerate(bc.indices):
                    if level_indices is None or len(level_indices) == 0:
                        continue
                    try:
                        has_points = len(level_indices[0]) > 0
                    except (TypeError, IndexError):
                        has_points = False
                    if has_points:
                        levels.add(level)
            self.extrap_aux_levels = sorted(levels)
        else:
            self.extrap_aux = None

        # One byte/voxel topology classification used only by the specialised
        # finest SFV/CFV modes. It is computed once after masks are finalised
        # and reused for every timestep because the multires grid is static.
        if self.mres_perf_opt in (
            MresPerfOptimizationType.FUSION_AT_FINEST_SFV,
            MresPerfOptimizationType.FUSION_AT_FINEST_SFV_ALL,
        ):
            self.needs_mres = grid.create_field(cardinality=1, dtype=Precision.UINT8)
        else:
            self.needs_mres = None

        for level in range(self.count_levels):
            self.u.fill_run(level, 0.0, 0)
            self.rho.fill_run(level, 1.0, 0)
            self.u0.fill_run(level, 0.0, 0)
            self.rho0.fill_run(level, 1.0, 0)
            self.u1.fill_run(level, 0.0, 0)
            self.rho1.fill_run(level, 1.0, 0)
            self.relax.fill_run(level, 0.0, 0)
            self.coalescence_factor.fill_run(level, 0.0, 0)
            if self.needs_mres is not None:
                self.needs_mres.fill_run(level, 0, 0)

        # Prepare fields
        self.f_0, self.f_1, self.bc_mask, self.missing_mask, self.normal_vector, self.normal_distance = self.prepare_fields(self.rho, self.u, self.initializer)

        if self.extrap_aux is not None:
            for level in self.extrap_aux_levels:
                self.neon_container["initialize_extrap_aux"](
                    level,
                    self.f_0,
                    self.bc_mask,
                    self.missing_mask,
                    self.extrap_aux,
                ).run(0)
            wp.synchronize()

        self.prepare_coalescence_count(coalescence_factor=self.coalescence_factor, bc_mask=self.bc_mask)

        self.iteration_idx = -1
        self.macro = MultiresMacroscopic(
            compute_backend=self.compute_backend,
            precision_policy=self.precision_policy,
            velocity_set=self.velocity_set,
        )

        # Construct the stepper skeleton
        self._construct_stepper_skeleton()

    def compute_omega(self, omega_finest, level):
        """
        Compute the relaxation parameter omega at a given grid level based on the finest level omega.
        We select a refinement ratio of 2 where a coarse cell at level L is uniformly divided into 2^d cells
        where d is the dimension. to arrive at level L - 1, or in other words ∆x_{L-1} = ∆x_L/2.
        For neighboring cells that interface two grid levels, a maximum jump in grid level of ∆L = 1 is
        allowed. Due to acoustic scaling which requires the speed of sound cs to remain constant across various grid levels,
        ∆tL ∝ ∆xL and hence ∆t_{L-1} = ∆t_{L}/2. In addition, the fluid viscosity \nu must also remain constant on each
        grid level which leads to the following relationship for the relaxation parameter omega at grid level L base
        on the finest grid level omega_finest.

        Args:
            omega_finest: Relaxation parameter at the finest grid level.
            level: Current grid level (0-indexed, with 0 being the finest level).

        Returns:
            Relaxation parameter omega at the specified grid level.
        """
        omega0 = omega_finest
        return 2 ** (level + 1) * omega0 / ((2**level - 1.0) * omega0 + 2.0)

    def export_macroscopic(self, fname_prefix):
        """Compute macroscopic fields and export velocity to a VTI file.

        Parameters
        ----------
        fname_prefix : str
            Output filename prefix.  The iteration index is appended
            automatically (e.g. ``"u_"`` → ``"u_42.vti"``).
        """
        print(f"exporting macroscopic: #levels {self.count_levels}")
        self.macro(self.f_0, self.bc_mask, self.rho, self.u, streamId=0)

        wp.synchronize()
        self.u.update_host(0)
        wp.synchronize()
        self.u.export_vti(f"{fname_prefix}{self.iteration_idx}.vti", "u")
        print("DONE exporting macroscopic")

        return

    def step(self):
        """Advance the simulation by one coarsest-level timestep.

        Internally this executes the pre-compiled Neon skeleton which
        performs the correct number of sub-steps at each finer level
        according to the acoustic-scaling time refinement ratio.
        """
        self.iteration_idx = self.iteration_idx + 1
        self.sk.run()

    def _build_recursion(self, level, app, config, accumulate_to_coarser=False):
        """Unified multi-resolution recursion builder.

        config keys:
            finest_ops:         list of (op_name, swap_fields, extra_kwargs) for level 0,
                                or None to treat level 0 like any coarse level.
            coarse_collide_ops: list of op_names for coarse collision.
            coarse_stream_ops:  list of (op_name, extra_kwargs) for coarse streaming.
            fuse_finest:        if True, recurse once (not twice) when child is at level 0.

        ``accumulate_to_coarser`` is true only for the final fine substep of
        each 2:1 adjacent-level interval.  This implements raw-last restriction
        throughout the hierarchy.

        The parent collision clears the dedicated integer accumulator on its
        local child-halo cells before either fine substep executes.  This fuses
        reset work into an existing container while preserving raw-last timing.
        """
        if level < 0:
            return

        omega = self.omega_list[level]
        fields = dict(f_0_fd=self.f_0, f_1_fd=self.f_1, bc_mask_fd=self.bc_mask, missing_mask_fd=self.missing_mask, _rho0=self.rho0, _u0=self.u0, _rho1=self.rho1, _u1=self.u1, _relax=self.relax,normal_vector = self.normal_vector, normal_distance = self.normal_distance)
        fields_swapped = dict(f_0_fd=self.f_1, f_1_fd=self.f_0, bc_mask_fd=self.bc_mask, missing_mask_fd=self.missing_mask, _rho0=self.rho1, _u0=self.u1, _rho1=self.rho0, _u1=self.u0, _relax=self.relax,normal_vector = self.normal_vector, normal_distance = self.normal_distance)

        if level == 0 and config["finest_ops"] is not None:
            for op_name, swap, extra_template in config["finest_ops"]:
                base = fields_swapped if swap else fields
                extra = dict(extra_template)

                if op_name != "SFV_finest_fused_pull":
                    extra["restriction_accumulator_fd"] = self.restriction_accumulator

                self.add_to_app(
                    app=app,
                    op_name=op_name,
                    level=level,
                    **base,
                    omega=omega,
                    **extra,
                )
            return

        for op_name in config["coarse_collide_ops"]:
            collide_extra = {}
            if op_name in ("collide_coarse", "CFV_collide_coarse"):
                collide_extra["extrap_aux_fd"] = self.extrap_aux
                collide_extra["accumulate_to_coarser"] = bool(accumulate_to_coarser)
                collide_extra["restriction_accumulator_fd"] = self.restriction_accumulator
            self.add_to_app(
                app=app,
                op_name=op_name,
                level=level,
                **fields,
                omega=omega,
                timestep=0,
                **collide_extra,
            )

        if config["fuse_finest"] and level - 1 == 0:
            # The fused finest operation list already contains both L0 phases.
            self._build_recursion(level - 1, app, config, accumulate_to_coarser=False)
        else:
            # First fine substep advances normally but does not restrict.
            self._build_recursion(level - 1, app, config, accumulate_to_coarser=False)
            # Final fine substep advances and supplies the endpoint state to L+1.
            self._build_recursion(level - 1, app, config, accumulate_to_coarser=True)

        for op_name, extra in config["coarse_stream_ops"]:
            self.add_to_app(app=app, op_name=op_name, level=level, **fields_swapped, **extra)

    def _construct_stepper_skeleton(self):
        import neon

        """Build the Neon skeleton that encodes the recursive time-stepping order.

        The skeleton is a list of Neon container invocations that, when
        executed in sequence, perform one coarsest-level timestep with the
        correct sub-cycling at finer levels.  The structure depends on
        ``self.mres_perf_opt``.
        """
        self.app = []

        stream_abc = {
            "coalescence_factor": self.coalescence_factor,
            "restriction_accumulator_fd": self.restriction_accumulator,
            "timestep": 0,
            "extrap_aux_fd": self.extrap_aux,
        }

        # Finest-level op descriptors: (op_name, swap_f0_f1, extra_kwargs)
        fused_pull_finest = [
            ("finest_fused_pull", False, {
                "timestep": 0,
                "is_f1_the_explosion_src_field": True,
                "accumulate_to_coarser": False,
                "extrap_aux_fd": self.extrap_aux,
            }),
            ("finest_fused_pull", True, {
                "timestep": 0,
                "is_f1_the_explosion_src_field": False,
                "accumulate_to_coarser": True,
                "extrap_aux_fd": self.extrap_aux,
            }),
        ]
        # Finest-level structural specialization. Each CFV class is a separate
        # compiled kernel so the common MRES/no-BC path contains no BC/wall code.
        sfv_fused_pull_finest = [
            ("CFV_BC_finest_fused_pull", False, {
                "timestep": 0,
                "is_f1_the_explosion_src_field": True,
                "accumulate_to_coarser": False,
                "extrap_aux_fd": self.extrap_aux,
            }),
            ("CFV_MRES_NOBC_finest_fused_pull", False, {
                "timestep": 0,
                "is_f1_the_explosion_src_field": True,
                "accumulate_to_coarser": False,
                "extrap_aux_fd": self.extrap_aux,
            }),
            ("CFV_MRES_BC_finest_fused_pull", False, {
                "timestep": 0,
                "is_f1_the_explosion_src_field": True,
                "accumulate_to_coarser": False,
                "extrap_aux_fd": self.extrap_aux,
            }),
            ("SFV_finest_fused_pull", False, {}),
            ("CFV_BC_finest_fused_pull", True, {
                "timestep": 0,
                "is_f1_the_explosion_src_field": False,
                "accumulate_to_coarser": True,
                "extrap_aux_fd": self.extrap_aux,
            }),
            ("CFV_MRES_NOBC_finest_fused_pull", True, {
                "timestep": 0,
                "is_f1_the_explosion_src_field": False,
                "accumulate_to_coarser": True,
                "extrap_aux_fd": self.extrap_aux,
            }),
            ("CFV_MRES_BC_finest_fused_pull", True, {
                "timestep": 0,
                "is_f1_the_explosion_src_field": False,
                "accumulate_to_coarser": True,
                "extrap_aux_fd": self.extrap_aux,
            }),
            ("SFV_finest_fused_pull", True, {}),
        ]

        configs = {
            MresPerfOptimizationType.NAIVE_COLLIDE_STREAM: {
                "finest_ops": None,
                "coarse_collide_ops": ["collide_coarse"],
                "coarse_stream_ops": [("stream_coarse_step_ABC", stream_abc)],
                "fuse_finest": False,
            },
            MresPerfOptimizationType.FUSION_AT_FINEST: {
                "finest_ops": fused_pull_finest,
                "coarse_collide_ops": ["collide_coarse"],
                "coarse_stream_ops": [("stream_coarse_step_ABC", stream_abc)],
                "fuse_finest": True,
            },
            MresPerfOptimizationType.FUSION_AT_FINEST_SFV: {
                "finest_ops": sfv_fused_pull_finest,
                "coarse_collide_ops": ["collide_coarse"],
                "coarse_stream_ops": [("stream_coarse_step_ABC", stream_abc)],
                "fuse_finest": True,
            },
            MresPerfOptimizationType.FUSION_AT_FINEST_SFV_ALL: {
                "finest_ops": sfv_fused_pull_finest,
                "coarse_collide_ops": ["CFV_collide_coarse", "SFV_collide_coarse"],
                "coarse_stream_ops": [("SFV_stream_coarse_step_ABC", stream_abc), ("SFV_stream_coarse_step", {})],
                "fuse_finest": True,
            },
        }

        config = configs.get(self.mres_perf_opt)
        if config is None:
            raise ValueError(f"Unknown optimization level: {self.mres_perf_opt}")

        # Pre-recursion SFV mask setup
        if self.mres_perf_opt == MresPerfOptimizationType.FUSION_AT_FINEST_SFV:
            wp.synchronize()
            self.neon_container["SFV_reset_bc_mask"](0, self.f_0, self.f_1, self.bc_mask, self.bc_mask,
                    self.rho0, self.u0, self.rho1, self.u1,self.relax,self.normal_vector,self.normal_distance).run(0)
            wp.synchronize()
        elif self.mres_perf_opt == MresPerfOptimizationType.FUSION_AT_FINEST_SFV_ALL:
            wp.synchronize()
            for l in range(self.f_0.get_grid().num_levels):
                self.neon_container["SFV_reset_bc_mask"](l, self.f_0, self.f_1, self.bc_mask, self.bc_mask,
                    self.rho0, self.u0, self.rho1, self.u1,self.relax,self.normal_vector,self.normal_distance).run(0)
            wp.synchronize()

        if self.needs_mres is not None:
            # Topology and BC masks are now final. Classify the finest level once;
            # both ping and pong fused passes read the same immutable byte field.
            self.neon_container["mark_needs_mres"](
                0, self.f_0, self.bc_mask, self.needs_mres
            ).run(0)
            wp.synchronize()

        self._build_recursion(
            self.count_levels - 1,
            self.app,
            config,
            accumulate_to_coarser=False,
        )

        bk = self.grid.get_neon_backend()
        self.sk = neon.Skeleton(backend=bk)
        self.sk.sequence("mres_nse_stepper", self.app)
