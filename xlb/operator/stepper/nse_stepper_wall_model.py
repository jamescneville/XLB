"""
Single-resolution incompressible Navier-Stokes stepper with wall-model support.

:class:`WallModelNavierStokesStepper` extends
:class:`IncompressibleNavierStokesStepper` (the plain, non-multires NEON
stepper) to also create and thread the ``relax``/``normal_vector``/
``normal_distance`` per-voxel fields that the ``HybridBC`` wall-model variants
(``use_wall_model=True``) need. The base stepper's ``apply_bc`` only forwards
7 arguments to ``boundary_conditions[i].neon_functional``; the wall-model
``HybridBC`` variants require 12 (``..., _rho, _u, _relax, _norm_vec,
_norm_dist``). This mirrors the field-threading pattern used by
``nse_multires_stepper.py``'s ``apply_bc_post_streaming``/
``apply_bc_post_collision``, without any multi-level machinery.

Ported without touching ``nse_stepper.py`` -- everything here is a subclass
override plus one new masker (``WallModelMeshMaskerAABBClose``).
"""

from typing import Any

import warp as wp

from xlb import DefaultConfig
from xlb.compute_backend import ComputeBackend
from xlb.operator import Operator
from xlb.operator.stepper.nse_stepper import IncompressibleNavierStokesStepper
from xlb.operator.boundary_condition.boundary_condition import ImplementationStep
from xlb.operator.boundary_condition.boundary_condition_registry import boundary_condition_registry
from xlb.operator.boundary_masker import (
    IndicesBoundaryMasker,
    MeshVoxelizationMethod,
    MeshMaskerRay,
    MeshMaskerWinding,
    MeshMaskerTrapped,
)
from xlb.operator.boundary_masker.wall_model_aabb_close import WallModelMeshMaskerAABBClose
from xlb.helper.nse_fields import create_nse_fields
from xlb.helper import check_bc_overlaps
from xlb.operator.boundary_condition.helper_functions_bc import EncodeAuxiliaryData
from xlb.operator.collision import SmagorinskyLESBGK
from xlb.cell_type import BC_SOLID


class WallModelNavierStokesStepper(IncompressibleNavierStokesStepper):
    """Single-resolution NEON stepper supporting HybridBC's wall model.

    Same constructor/collision/streaming setup as
    :class:`IncompressibleNavierStokesStepper`. Adds:

    * ``prepare_fields`` also creates/returns ``normal_vector``/
      ``normal_distance`` (mesh-based wall geometry, computed once by
      :class:`WallModelMeshMaskerAABBClose`).
    * A NEON container (``_construct_neon``) that reads ``_rho0``/``_u0``
      (previous-timestep macroscopic fields), writes ``_rho1``/``_u1``
      (this-timestep's, for the next step to read), and threads
      ``_relax``/``normal_vector``/``normal_distance`` through
      ``apply_bc`` for ``HybridBC`` boundary conditions specifically.

    Only ``MeshVoxelizationMethod("AABB_CLOSE")`` is supported for
    mesh-based boundary conditions here (matching the only voxelization
    method this wall-model path has been exercised with); other methods
    fall back to the base (non-wall-model) mesh maskers with no normal/
    distance output, so a BC using them cannot also request
    ``use_wall_model=True``.
    """

    def __init__(
        self,
        grid,
        boundary_conditions=[],
        collision_type="BGK",
        streaming_scheme="pull",
        forcing_scheme="exact_difference",
        force_vector=None,
        backend_config={},
        smagorinsky_constant=0.17,
    ):
        super().__init__(grid, boundary_conditions, collision_type, streaming_scheme, forcing_scheme, force_vector, backend_config)
        # The base stepper's SmagorinskyLESKBC/SmagorinskyLESBGK dispatch
        # always uses SmagorinskyLESBGK's default coefficient (0.17) since
        # it has no way to take a custom value; rebuild it here with the
        # requested constant (mirrors nse_multires_stepper.py's handling).
        if collision_type in ("SmagorinskyLESBGK", "SmagorinskyLESKBC"):
            fixed_collision = SmagorinskyLESBGK(self.velocity_set, self.precision_policy, self.compute_backend, smagorinsky_constant)
            if force_vector is not None:
                from xlb.operator.collision import ForcedCollision

                fixed_collision = ForcedCollision(collision_operator=fixed_collision, forcing_scheme=forcing_scheme, force_vector=force_vector)
            self.collision = fixed_collision

    def prepare_fields(self, rho0, u0, rho1, u1, relax, normal_vector, normal_distance, initializer=None):
        """Prepare fields, including the wall-model geometry fields.

        Parameters
        ----------
        rho0, u0, rho1, u1 : Field
            Persistent double-buffered macroscopic fields (created by the
            caller/SimulationManager) that the wall model reads (``_rho0``/
            ``_u0``, previous timestep) and the collision step writes
            (``_rho1``/``_u1``, this timestep, for next step to read).
        relax : Field
            Persistent per-voxel relaxation-parameter cache the wall model
            updates every step (EMA of the local pressure gradient).
        normal_vector, normal_distance : Field
            Persistent per-voxel wall geometry, computed once here from the
            mesh (not recomputed per timestep).
        initializer : Operator, optional
            Same as the base stepper.

        Returns
        -------
        Tuple of (f_0, f_1, bc_mask, missing_mask, normal_vector, normal_distance)
        """
        _, f_0, f_1, missing_mask, bc_mask = create_nse_fields(
            grid=self.grid, velocity_set=self.velocity_set, compute_backend=self.compute_backend, precision_policy=self.precision_policy
        )

        if self.compute_backend == ComputeBackend.NEON:
            f_1.copy_from_run(f_0, 0)
        else:
            raise NotImplementedError("WallModelNavierStokesStepper only supports the NEON backend.")

        # Important note: XLB uses f_1's central index / missing directions to
        # store auxiliary BC data (and, here, per-direction mesh distances).
        f_1, bc_mask, missing_mask, normal_vector, normal_distance = self._process_boundary_conditions(
            self.boundary_conditions, f_1, bc_mask, missing_mask, normal_vector, normal_distance
        )

        f_1 = self._initialize_auxiliary_data(self.boundary_conditions, f_1, bc_mask, missing_mask)
        wp.synchronize()

        if initializer is not None:
            f_0 = initializer(bc_mask, f_0)
        else:
            from xlb.helper.initializers import initialize_eq

            f_0 = initialize_eq(f_0, self.grid, self.velocity_set, self.precision_policy, self.compute_backend)

        return f_0, f_1, bc_mask, missing_mask, normal_vector, normal_distance

    def _process_boundary_conditions(self, boundary_conditions, f_1, bc_mask, missing_mask, normal_vector, normal_distance):
        """Process boundary conditions, threading normal/distance through mesh-based ones."""
        check_bc_overlaps(boundary_conditions, DefaultConfig.velocity_set.d, DefaultConfig.default_backend)

        indices_masker = IndicesBoundaryMasker(
            velocity_set=DefaultConfig.velocity_set,
            precision_policy=DefaultConfig.default_precision_policy,
            compute_backend=DefaultConfig.default_backend,
            grid=self.grid,
        )

        bc_with_vertices = [bc for bc in boundary_conditions if bc.mesh_vertices is not None]
        bc_with_indices = [bc for bc in boundary_conditions if bc.indices is not None]

        if bc_with_indices:
            bc_mask, missing_mask = indices_masker(bc_with_indices, bc_mask, missing_mask)

        if DefaultConfig.velocity_set.d == 3 and bc_with_vertices:
            for bc in bc_with_vertices:
                if bc.voxelization_method.id is MeshVoxelizationMethod("AABB_CLOSE").id:
                    mesh_masker = WallModelMeshMaskerAABBClose(
                        velocity_set=DefaultConfig.velocity_set,
                        precision_policy=DefaultConfig.default_precision_policy,
                        compute_backend=DefaultConfig.default_backend,
                        close_voxels=bc.voxelization_method.options.get("close_voxels"),
                    )
                    f_1, bc_mask, missing_mask, normal_vector, normal_distance = mesh_masker(
                        bc, f_1, bc_mask, missing_mask, normal_vector, normal_distance
                    )
                elif bc.voxelization_method.id is MeshVoxelizationMethod("RAY").id:
                    mesh_masker = MeshMaskerRay(
                        velocity_set=DefaultConfig.velocity_set,
                        precision_policy=DefaultConfig.default_precision_policy,
                        compute_backend=DefaultConfig.default_backend,
                    )
                    if bc.use_wall_model:
                        raise ValueError("use_wall_model requires voxelization_method='AABB_CLOSE' in WallModelNavierStokesStepper.")
                    f_1, bc_mask, missing_mask = mesh_masker(bc, f_1, bc_mask, missing_mask)
                elif bc.voxelization_method.id is MeshVoxelizationMethod("WINDING").id:
                    mesh_masker = MeshMaskerWinding(
                        velocity_set=DefaultConfig.velocity_set,
                        precision_policy=DefaultConfig.default_precision_policy,
                        compute_backend=DefaultConfig.default_backend,
                    )
                    if bc.use_wall_model:
                        raise ValueError("use_wall_model requires voxelization_method='AABB_CLOSE' in WallModelNavierStokesStepper.")
                    f_1, bc_mask, missing_mask = mesh_masker(bc, f_1, bc_mask, missing_mask)
                else:
                    raise ValueError(f"Unsupported voxelization method: {bc.voxelization_method}")

            trapped_masker = MeshMaskerTrapped(
                velocity_set=DefaultConfig.velocity_set,
                precision_policy=DefaultConfig.default_precision_policy,
                compute_backend=DefaultConfig.default_backend,
            )
            f_1, bc_mask, missing_mask = trapped_masker(bc_with_vertices[0], f_1, bc_mask, missing_mask)

        return f_1, bc_mask, missing_mask, normal_vector, normal_distance

    @staticmethod
    def _initialize_auxiliary_data(boundary_conditions, f_1, bc_mask, missing_mask):
        for bc in boundary_conditions:
            if bc.needs_aux_init and not bc.is_initialized_with_aux_data:
                encode_auxiliary_data = EncodeAuxiliaryData(
                    bc.id,
                    bc.num_of_aux_data,
                    bc.profile,
                    velocity_set=bc.velocity_set,
                    precision_policy=bc.precision_policy,
                    compute_backend=bc.compute_backend,
                )
                f_1 = encode_auxiliary_data(f_1, bc_mask, missing_mask)
                bc.is_initialized_with_aux_data = True
        return f_1

    def _construct_neon(self):
        import neon

        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)
        _missing_mask_vec = wp.vec(self.velocity_set.q, dtype=wp.uint8)
        _u_vec = wp.vec(self.velocity_set.d, dtype=self.compute_dtype)
        _opp_indices = self.velocity_set.opp_indices
        lattice_central_index = self.velocity_set.center_index

        bc_to_id = boundary_condition_registry.bc_to_id
        extrapolation_outflow_bc_ids = []
        for bc_name, bc_id in bc_to_id.items():
            if bc_name.startswith("ExtrapolationOutflowBC"):
                extrapolation_outflow_bc_ids.append(bc_id)
        hybrid_bc_ids = []
        for bc_name, bc_id in bc_to_id.items():
            if bc_name.startswith("HybridBC"):
                hybrid_bc_ids.append(bc_id)

        @wp.func
        def apply_bc(
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
            _relax: Any,
            _norm_vec: Any,
            _norm_dist: Any,
            is_post_streaming: bool,
        ):
            f_result = f_post

            for i in range(wp.static(len(self.boundary_conditions))):
                if is_post_streaming:
                    if wp.static(self.boundary_conditions[i].implementation_step == ImplementationStep.STREAMING):
                        if wp.static(self.boundary_conditions[i].id in hybrid_bc_ids):
                            if _boundary_id == wp.static(self.boundary_conditions[i].id):
                                f_result = wp.static(self.boundary_conditions[i].neon_functional)(
                                    index, timestep, _missing_mask, f_0, f_1, f_pre, f_post, _rho, _u, _relax, _norm_vec, _norm_dist
                                )
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

        @wp.func
        def neon_get_thread_data(
            f0_pn: Any,
            missing_mask_pn: Any,
            index: Any,
        ):
            _f0_thread = _f_vec()
            _missing_mask = _missing_mask_vec()
            for l in range(self.velocity_set.q):
                _f0_thread[l] = self.compute_dtype(wp.neon_read(f0_pn, index, l))
                _missing_mask[l] = wp.neon_read(missing_mask_pn, index, l)
            return _f0_thread, _missing_mask

        @wp.func
        def neon_apply_aux_recovery_bc(
            index: Any,
            _boundary_id: Any,
            _missing_mask: Any,
            f_0_pn: Any,
            f_1_pn: Any,
        ):
            for i in range(wp.static(len(self.boundary_conditions))):
                if wp.static(self.boundary_conditions[i].needs_aux_recovery):
                    if _boundary_id == wp.static(self.boundary_conditions[i].id):
                        for l in range(self.velocity_set.q):
                            if l == lattice_central_index:
                                _f1_thread = wp.neon_read(f_1_pn, index, l)
                                wp.neon_write(f_0_pn, index, l, self.store_dtype(_f1_thread))
                            elif _missing_mask[l] == wp.uint8(1):
                                _f1_thread = wp.neon_read(f_1_pn, index, _opp_indices[l])
                                wp.neon_write(f_0_pn, index, _opp_indices[l], self.store_dtype(_f1_thread))

        @neon.Container.factory(name="nse_stepper_wall_model")
        def container(
            f_0_fd: Any,
            f_1_fd: Any,
            bc_mask_fd: Any,
            missing_mask_fd: Any,
            omega: Any,
            timestep: int,
            _rho0_fd: Any,
            _u0_fd: Any,
            _rho1_fd: Any,
            _u1_fd: Any,
            _relax_fd: Any,
            normal_vector_fd: Any,
            normal_distance_fd: Any,
        ):
            def nse_stepper_ll(loader: neon.Loader):
                loader.set_grid(bc_mask_fd.get_grid())

                f_0_pn = loader.get_read_handle(
                    f_0_fd,
                    operation=neon.Loader.Operation.stencil,
                    discretization=neon.Loader.Discretization.lattice,
                )
                bc_mask_pn = loader.get_read_handle(bc_mask_fd)
                missing_mask_pn = loader.get_read_handle(missing_mask_fd)
                f_1_pn = loader.get_write_handle(f_1_fd)

                _rho0_pn = loader.get_read_handle(_rho0_fd)
                _u0_pn = loader.get_read_handle(_u0_fd)
                _rho1_pn = loader.get_write_handle(_rho1_fd)
                _u1_pn = loader.get_write_handle(_u1_fd)
                _relax_pn = loader.get_write_handle(_relax_fd)
                _norm_vec_pn = loader.get_write_handle(normal_vector_fd)
                _norm_dist_pn = loader.get_write_handle(normal_distance_fd)

                @wp.func
                def nse_stepper_cl(index: Any):
                    _boundary_id = wp.neon_read(bc_mask_pn, index, 0)
                    if _boundary_id == wp.uint8(BC_SOLID):
                        return

                    _f_post_stream = self.stream.neon_functional(f_0_pn, index)

                    _f0_thread, _missing_mask = neon_get_thread_data(f_0_pn, missing_mask_pn, index)
                    _f_post_collision = _f0_thread

                    # Post-streaming BCs (incl. HybridBC wall model). Wall model
                    # reads _rho0/_u0 -- the PREVIOUS timestep's macroscopic
                    # fields -- not values derived from this step's _f_post_stream.
                    _f_post_stream = apply_bc(
                        index, timestep, _boundary_id, _missing_mask, f_0_pn, f_1_pn, _f_post_collision, _f_post_stream,
                        _rho0_pn, _u0_pn, _relax_pn, _norm_vec_pn, _norm_dist_pn, True,
                    )

                    _rho, _u = self.macroscopic.neon_functional(_f_post_stream)
                    _feq = self.equilibrium.neon_functional(_rho, _u)
                    _f_post_collision = self.collision.neon_functional(_f_post_stream, _feq, omega)

                    _f_post_collision = apply_bc(
                        index, timestep, _boundary_id, _missing_mask, f_0_pn, f_1_pn, _f_post_stream, _f_post_collision,
                        _rho0_pn, _u0_pn, _relax_pn, _norm_vec_pn, _norm_dist_pn, False,
                    )

                    neon_apply_aux_recovery_bc(index, _boundary_id, _missing_mask, f_0_pn, f_1_pn)

                    for l in range(self.velocity_set.q):
                        wp.neon_write(f_1_pn, index, l, self.store_dtype(_f_post_collision[l]))

                    # Cache this step's macroscopic values for next step's wall
                    # model to read (via _rho0/_u0 once buffers swap).
                    wp.neon_write(_rho1_pn, index, 0, self.store_dtype(_rho))
                    for d in range(self.velocity_set.d):
                        wp.neon_write(_u1_pn, index, d, self.store_dtype(_u[d]))

                loader.declare_kernel(nse_stepper_cl)

            return nse_stepper_ll

        return None, container

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_launch(self, f_0, f_1, bc_mask, missing_mask, omega, timestep, rho0, u0, rho1, u1, relax, normal_vector, normal_distance):
        if timestep == 0:
            self.prepare_skeleton(f_0, f_1, bc_mask, missing_mask, omega, rho0, u0, rho1, u1, relax, normal_vector, normal_distance)
        self.sk[self.sk_iter].run()
        self.sk_iter = (self.sk_iter + 1) % 2
        return f_0, f_1

    def prepare_skeleton(self, f_0, f_1, bc_mask, missing_mask, omega, rho0, u0, rho1, u1, relax, normal_vector, normal_distance):
        """Build the Neon odd/even skeletons for double-buffered time stepping.

        rho0/u0/rho1/u1 alternate in lockstep with f_0/f_1: the "odd" call
        reads (rho0, u0) and writes (rho1, u1); the "even" call reads
        (rho1, u1) [renamed rho0/u0 for that call] and writes back into the
        other buffer -- exactly mirroring how f_0/f_1 swap identity between
        calls. relax/normal_vector/normal_distance do not swap; the wall
        model owns their persistent state directly.
        """
        import neon

        grid = f_0.get_grid()
        bk = grid.backend
        self.neon_skeleton = {"odd": {}, "even": {}}
        self.neon_skeleton["odd"]["container"] = self.neon_container(
            f_0, f_1, bc_mask, missing_mask, omega, 0, rho0, u0, rho1, u1, relax, normal_vector, normal_distance
        )
        self.neon_skeleton["even"]["container"] = self.neon_container(
            f_1, f_0, bc_mask, missing_mask, omega, 1, rho1, u1, rho0, u0, relax, normal_vector, normal_distance
        )

        if "occ" not in self.backend_config:
            occ = neon.SkeletonConfig.OCC.none()
        else:
            occ = self.backend_config["occ"]
            if not isinstance(occ, neon.SkeletonConfig.OCC):
                raise ValueError("occ must be of type neon.SkeletonConfig.OCC")

        for key in self.neon_skeleton:
            self.neon_skeleton[key]["app"] = [self.neon_skeleton[key]["container"]]
            self.neon_skeleton[key]["skeleton"] = neon.Skeleton(backend=bk)
            self.neon_skeleton[key]["skeleton"].sequence(name="nse_stepper_wall_model", containers=self.neon_skeleton[key]["app"], occ=occ)

        self.sk = [self.neon_skeleton["odd"]["skeleton"], self.neon_skeleton["even"]["skeleton"]]
        self.sk_iter = 0
