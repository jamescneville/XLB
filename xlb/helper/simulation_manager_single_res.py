"""
Single-resolution simulation manager for the Neon backend, with wall-model support.

:class:`SimulationManagerSingleRes` is the non-multires analogue of
:class:`MultiresSimulationManager`. It drives a
:class:`WallModelNavierStokesStepper` on a plain ``NeonGrid`` (single
``neon.dense.dGrid``, no adaptive refinement) -- reusing the same field
management / step / export_macroscopic pattern, minus everything that only
makes sense with multiple grid levels (per-level omega, coalescence,
needs_mres classification, the recursive skeleton builder).
"""

import warp as wp
from xlb.operator.macroscopic import Macroscopic
from xlb.operator.stepper.nse_stepper_wall_model import WallModelNavierStokesStepper


class SimulationManagerSingleRes:
    """Orchestrates single-resolution LBM simulations on the Neon backend.

    Parameters
    ----------
    omega : float
        Relaxation parameter (no per-level scaling -- there is only one level).
    grid : NeonGrid
        Single-resolution Neon grid.
    boundary_conditions : list of BoundaryCondition
        Boundary conditions to apply.
    collision_type : str
        ``"BGK"``, ``"KBC"``, or ``"SmagorinskyLESBGK"``.
    forcing_scheme : str
        Forcing scheme (used only when *force_vector* is given).
    force_vector : array-like, optional
        External body force.
    initializer : Operator, optional
        Custom initializer for distribution functions. If ``None`` the
        default equilibrium initialization is used.
    backend_config : dict, optional
        Neon backend configuration (OCC, etc.), forwarded to the stepper.
    """

    def __init__(
        self,
        omega,
        grid,
        boundary_conditions=[],
        collision_type="BGK",
        forcing_scheme="exact_difference",
        force_vector=None,
        initializer=None,
        backend_config={},
        smagorinsky_constant=0.17,
    ):
        self.grid = grid
        self.omega = omega
        self.initializer = initializer

        self.stepper = WallModelNavierStokesStepper(
            grid=grid,
            boundary_conditions=boundary_conditions,
            collision_type=collision_type,
            forcing_scheme=forcing_scheme,
            force_vector=force_vector,
            backend_config=backend_config,
            smagorinsky_constant=smagorinsky_constant,
        )
        self.velocity_set = self.stepper.velocity_set
        self.precision_policy = self.stepper.precision_policy
        self.compute_backend = self.stepper.compute_backend

        # Output macroscopic fields (recomputed fresh from f_0 for export).
        self.rho = grid.create_field(cardinality=1, dtype=self.precision_policy.store_precision, fill_value=1.0)
        self.u = grid.create_field(cardinality=3, dtype=self.precision_policy.store_precision, fill_value=0.0)

        # Wall-model state: double-buffered macroscopic fields the HybridBC
        # wall model reads (rho0/u0, previous timestep) and the collision
        # step writes (rho1/u1, this timestep) -- see
        # WallModelNavierStokesStepper._construct_neon for the read/write
        # split. Alternation between steps is handled entirely by the
        # stepper's pre-built odd/even skeleton (see prepare_skeleton); these
        # attributes are not swapped here.
        self.rho0 = grid.create_field(cardinality=1, dtype=self.precision_policy.store_precision, fill_value=1.0)
        self.u0 = grid.create_field(cardinality=3, dtype=self.precision_policy.store_precision, fill_value=0.0)
        self.rho1 = grid.create_field(cardinality=1, dtype=self.precision_policy.store_precision, fill_value=1.0)
        self.u1 = grid.create_field(cardinality=3, dtype=self.precision_policy.store_precision, fill_value=0.0)
        self.relax = grid.create_field(cardinality=1, dtype=self.precision_policy.store_precision, fill_value=0.0)
        self.normal_vector = grid.create_field(cardinality=self.velocity_set.d, dtype=self.precision_policy.store_precision, fill_value=0.0)
        self.normal_distance = grid.create_field(cardinality=1, dtype=self.precision_policy.store_precision, fill_value=0.0)

        # Prepare fields (also computes normal_vector/normal_distance from
        # the mesh once, via WallModelMeshMaskerAABBClose).
        self.f_0, self.f_1, self.bc_mask, self.missing_mask, self.normal_vector, self.normal_distance = self.stepper.prepare_fields(
            self.rho0, self.u0, self.rho1, self.u1, self.relax, self.normal_vector, self.normal_distance, self.initializer
        )

        self.iteration_idx = -1
        self.macro = Macroscopic(
            compute_backend=self.compute_backend,
            precision_policy=self.precision_policy,
            velocity_set=self.velocity_set,
        )

    def export_macroscopic(self, fname_prefix):
        """Compute macroscopic fields and export velocity to a VTI file.

        Parameters
        ----------
        fname_prefix : str
            Output filename prefix. The iteration index is appended
            automatically (e.g. ``"u_"`` -> ``"u_42.vti"``).
        """
        self.macro(self.f_0, self.rho, self.u)

        wp.synchronize()
        self.u.update_host(0)
        wp.synchronize()
        self.u.export_vti(f"{fname_prefix}{self.iteration_idx}.vti", "u")

    def step(self):
        """Advance the simulation by one timestep."""
        self.iteration_idx = self.iteration_idx + 1
        self.f_0, self.f_1 = self.stepper(
            self.f_0,
            self.f_1,
            self.bc_mask,
            self.missing_mask,
            self.omega,
            self.iteration_idx,
            self.rho0,
            self.u0,
            self.rho1,
            self.u1,
            self.relax,
            self.normal_vector,
            self.normal_distance,
        )
        self.f_0, self.f_1 = self.f_1, self.f_0
