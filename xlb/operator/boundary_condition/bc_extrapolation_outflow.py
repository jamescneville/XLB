"""
Extrapolation outflow boundary condition.

Uses first-order extrapolation from the interior to set the unknown
populations at outflow boundaries, avoiding strong wave reflections.

Reference
---------
Geier, M. et al. (2015). "The cumulant lattice Boltzmann equation in
three dimensions: Theory and validation." *Computers & Mathematics
with Applications*, 70(4), 507-547.
"""

import jax.numpy as jnp
from jax import jit
import jax.lax as lax
from functools import partial
import warp as wp
from typing import Any
from collections import Counter
import numpy as np

from xlb import DefaultConfig
from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb.operator.equilibrium import QuadraticEquilibrium
from xlb.operator.boundary_condition.boundary_condition import (
    ImplementationStep,
    BoundaryCondition,
)
from xlb.operator.boundary_masker.mesh_voxelization_method import MeshVoxelizationMethod


class ExtrapolationOutflowBC(BoundaryCondition):
    """
    Extrapolation outflow boundary condition for a lattice Boltzmann method simulation.

    This class implements the extrapolation outflow boundary condition, which is a type of outflow boundary condition
    that uses extrapolation to avoid strong wave reflections.

    Parameters
    ----------
    outlet_normal : tuple, optional
        Prescribed outward unit normal for a planar outlet, e.g. ``(1, 0, 0)``
        for the +X/right boundary.  When supplied, the BC uses this normal
        directly instead of inferring a normal from ``missing_mask``.  The
        prescribed normal must be axis-aligned with exactly one non-zero
        component equal to +1 or -1.
    rho_reference : float, optional
        Local reference density used by the weak multires Neon pressure anchor.
        Defaults to 1.0.
    rho_relaxation : float, optional
        Under-relaxation factor for the equilibrium-only density correction.
        ``0`` disables the correction and ``1`` imposes the local reference in
        one application.  A small value is recommended for outflow use; the
        default is 0.001.

    References
    ----------
    Geier, M., Schönherr, M., Pasquali, A., & Krafczyk, M. (2015). The cumulant lattice Boltzmann equation in three
    dimensions: Theory and validation. Computers & Mathematics with Applications, 70(4), 507-547.
    doi:10.1016/j.camwa.2015.05.001.
    """

    def __init__(
        self,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
        indices=None,
        mesh_vertices=None,
        voxelization_method: MeshVoxelizationMethod = None,
        outlet_normal=None,
        rho_reference: float = 1.0,
        rho_relaxation: float = 0.0,
    ):
        # IMPORTANT: Operator.__init__ constructs the backend functions during
        # super().__init__().  _construct_warp() and _construct_neon() both
        # inspect self.outlet_normal, so the attribute must exist before the
        # parent constructor is called.
        resolved_velocity_set = velocity_set or DefaultConfig.velocity_set

        self.outlet_normal = None
        if outlet_normal is not None:
            if len(outlet_normal) != resolved_velocity_set.d:
                raise ValueError(
                    f"outlet_normal must have {resolved_velocity_set.d} components"
                )
            if any(float(v) not in (-1.0, 0.0, 1.0) for v in outlet_normal):
                raise ValueError(
                    "outlet_normal components must each be -1, 0, or 1"
                )
            normal = tuple(int(v) for v in outlet_normal)
            if sum(abs(v) for v in normal) != 1:
                raise ValueError(
                    "outlet_normal must be an axis-aligned unit vector, e.g. (1, 0, 0)"
                )
            self.outlet_normal = normal

        # Call the parent constructor only after attributes needed by backend
        # construction have been initialized.
        super().__init__(
            ImplementationStep.STREAMING,
            velocity_set,
            precision_policy,
            compute_backend,
            indices,
            mesh_vertices,
            voxelization_method,
        )

        # Optional weak local density/pressure reference for Neon multires.
        # rho_relaxation = 0 disables the correction; 1 would impose the
        # reference density in a single application and is intentionally not
        # recommended for an outflow because it becomes strongly reflective.
        if rho_relaxation < 0.0 or rho_relaxation > 1.0:
            raise ValueError("rho_relaxation must be in the range [0, 1]")
        self.rho_reference = float(rho_reference)
        self.rho_relaxation = float(rho_relaxation)

        # Multires/Neon uses a dedicated auxiliary field instead of hiding
        # next-step outflow data inside physical population slots.
        #
        # Legacy/default behavior (no prescribed normal) assumes a planar
        # outlet with edges removed and therefore keeps the compact planar
        # cardinality (D3Q27 -> 9).
        #
        # With a prescribed normal we allow ExtrapBC to own the complete
        # outlet plane, including geometric edges/corners.  A D3Q27 corner can
        # have up to 19 missing populations, so reserve 19 auxiliary values to
        # extrapolate every missing population without a bounce-back fallback.
        c_np = np.asarray(self.velocity_set._c)
        crossing_counts = []
        for d in range(self.velocity_set.d):
            crossing_counts.append(int(np.count_nonzero(c_np[d, :] < 0)))
            crossing_counts.append(int(np.count_nonzero(c_np[d, :] > 0)))
        planar_cardinality = max(crossing_counts) if crossing_counts else self.velocity_set.q

        if self.outlet_normal is not None and self.velocity_set.d == 3 and self.velocity_set.q == 27:
            self.extrap_aux_cardinality = 19
        else:
            self.extrap_aux_cardinality = planar_cardinality

        self.uses_dedicated_aux_field = True

        # JAX stores the normal on the Python object.  Prefer the prescribed
        # outlet normal when available; otherwise preserve the legacy inference
        # from the boundary indices.
        if self.compute_backend == ComputeBackend.JAX:
            if self.outlet_normal is not None:
                self.normal = np.asarray(self.outlet_normal, dtype=np.int32)
            else:
                self._get_normal_vectors(indices)

        # Unpack the two warp functionals needed for this BC!
        if self.compute_backend == ComputeBackend.WARP:
            self.warp_functional, self.assemble_auxiliary_data = self.warp_functional
        elif self.compute_backend == ComputeBackend.NEON:
            # Keep the legacy Neon API for the single-resolution stepper.
            self.neon_functional, self.assemble_auxiliary_data = self.neon_functional
            # Multires uses dedicated auxiliary storage so physical populations
            # remain untouched for coalescence/explosion.
            (
                self.neon_multires_functional,
                self.assemble_auxiliary_data_multires,
            ) = self._construct_neon_multires()

    def _get_normal_vectors(self, indices):
        # Get the frequency count and most common element directly
        freq_counts = [Counter(coord).most_common(1)[0] for coord in indices]

        # Extract counts and elements
        counts = np.array([count for _, count in freq_counts])
        elements = np.array([element for element, _ in freq_counts])

        # Normalize the counts
        self.normal = counts // counts.max()

        # Reverse the normal vector if the most frequent element is 0
        if elements[np.argmax(counts)] == 0:
            self.normal *= -1
        return

    @partial(jit, static_argnums=(0,), inline=True)
    def _roll(self, fld, vec):
        """
        Perform rolling operation of a field with dimentions [q, nx, ny, nz] in a direction
        given by vec. All q-directions are rolled at the same time.
        # TODO: how to improve this for multi-gpu runs?
        """
        if self.velocity_set.d == 2:
            return jnp.roll(fld, (vec[0], vec[1]), axis=(1, 2))
        elif self.velocity_set.d == 3:
            return jnp.roll(fld, (vec[0], vec[1], vec[2]), axis=(1, 2, 3))

    @partial(jit, static_argnums=(0,), inline=True)
    def assemble_auxiliary_data(self, f_pre, f_post, bc_mask, missing_mask):
        """
        Prepare time-dependent dynamic data for imposing the boundary condition in the next iteration after streaming.
        We use directions that leave the domain for storing this prepared data.
        Since this function is called post-collisiotn: f_pre = f_post_stream and f_post = f_post_collision
        """
        sound_speed = 1.0 / jnp.sqrt(3.0)
        boundary = bc_mask == self.id
        new_shape = (self.velocity_set.q,) + boundary.shape[1:]
        boundary = lax.broadcast_in_dim(boundary, new_shape, tuple(range(self.velocity_set.d + 1)))

        # Roll boundary mask in the opposite of the normal vector to mask its next immediate neighbour
        neighbour = self._roll(boundary, -self.normal)

        # gather post-streaming values associated with previous time-step to construct the required data for BC
        fpop = jnp.where(boundary, f_pre, f_post)
        fpop_neighbour = jnp.where(neighbour, f_pre, f_post)

        # With fpop_neighbour isolated, now roll it back to be positioned at the boundary for subsequent operations
        fpop_neighbour = self._roll(fpop_neighbour, self.normal)
        fpop_extrapolated = sound_speed * fpop_neighbour + (1.0 - sound_speed) * fpop

        # Use the iknown directions of f_postcollision that leave the domain during streaming to store the BC data
        opp = self.velocity_set.opp_indices
        known_mask = missing_mask[opp]
        f_post = jnp.where(jnp.logical_and(boundary, known_mask), fpop_extrapolated[opp], f_post)
        return f_post

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0))
    def jax_implementation(self, f_pre, f_post, bc_mask, missing_mask):
        boundary = bc_mask == self.id
        new_shape = (self.velocity_set.q,) + boundary.shape[1:]
        boundary = lax.broadcast_in_dim(boundary, new_shape, tuple(range(self.velocity_set.d + 1)))
        return jnp.where(
            jnp.logical_and(missing_mask, boundary),
            f_pre[self.velocity_set.opp_indices],
            f_post,
        )

    def _construct_warp(self):
        # Set local constants
        sound_speed = self.compute_dtype(1.0 / wp.sqrt(3.0))
        _c = self.velocity_set.c
        _q = self.velocity_set.q
        _opp_indices = self.velocity_set.opp_indices
        _has_prescribed_normal = self.outlet_normal is not None
        _normal_x = self.outlet_normal[0] if self.outlet_normal is not None else 0
        _normal_y = self.outlet_normal[1] if self.outlet_normal is not None else 0
        _normal_z = (
            self.outlet_normal[2]
            if self.outlet_normal is not None and self.velocity_set.d == 3
            else 0
        )

        @wp.func
        def get_normal_vectors(
            missing_mask: Any,
        ):
            if wp.static(_has_prescribed_normal):
                if wp.static(self.velocity_set.d == 3):
                    return wp.vec3i(
                        wp.int32(_normal_x),
                        wp.int32(_normal_y),
                        wp.int32(_normal_z),
                    )
                else:
                    return wp.vec2i(
                        wp.int32(_normal_x),
                        wp.int32(_normal_y),
                    )

            if wp.static(self.velocity_set.d == 3):
                for l in range(_q):
                    if missing_mask[l] == wp.uint8(1) and wp.abs(_c[0, l]) + wp.abs(_c[1, l]) + wp.abs(_c[2, l]) == 1:
                        return -wp.vec3i(_c[0, l], _c[1, l], _c[2, l])
            else:
                for l in range(_q):
                    if missing_mask[l] == wp.uint8(1) and wp.abs(_c[0, l]) + wp.abs(_c[1, l]) == 1:
                        return -wp.vec2i(_c[0, l], _c[1, l])

        # Construct the functionals for this BC
        @wp.func
        def functional(
            index: Any,
            timestep: Any,
            missing_mask: Any,
            f_0: Any,
            f_1: Any,
            _f_pre: Any,
            _f_post: Any,
        ):
            # Post-streaming values are only modified at missing direction
            _f = _f_post
            for l in range(self.velocity_set.q):
                # If the mask is missing then take the opposite index
                if missing_mask[l] == wp.uint8(1):
                    _f[l] = _f_pre[_opp_indices[l]]
            return _f

        @wp.func
        def assemble_auxiliary_data_warp(
            index: Any,
            timestep: Any,
            missing_mask: Any,
            f_0: Any,
            f_1: Any,
            _f_pre: Any,
            _f_post: Any,
        ):
            # Prepare time-dependent dynamic data for imposing the boundary condition in the next iteration after streaming.
            # We use directions that leave the domain for storing this prepared data.
            # Since this function is called post-collisiotn: f_pre = f_post_stream and f_post = f_post_collision
            _f = _f_post
            nv = get_normal_vectors(missing_mask)
            for l in range(self.velocity_set.q):
                if missing_mask[l] == wp.uint8(1):
                    # f_0 is the post-collision values of the current time-step
                    # Get pull index associated with the "neighbours" pull_index
                    pull_index = type(index)()
                    for d in range(self.velocity_set.d):
                        pull_index[d] = index[d] - (_c[d, l] + nv[d])
                    # The following is the post-streaming values of the neighbor cell
                    f_aux = self.compute_dtype(f_0[l, pull_index[0], pull_index[1], pull_index[2]])
                    _f[_opp_indices[l]] = (self.compute_dtype(1.0) - sound_speed) * _f_pre[l] + sound_speed * f_aux
            return _f

        kernel = self._construct_kernel(functional)
        return (functional, assemble_auxiliary_data_warp), kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, _f_pre, _f_post, bc_mask, missing_mask):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[_f_pre, _f_post, bc_mask, missing_mask],
            dim=_f_pre.shape[1:],
        )
        return _f_post

    def _construct_neon_multires(self):
        """Construct the multires Neon implementation using dedicated auxiliary storage.

        If ``outlet_normal`` is not supplied, the outlet normal is inferred from
        ``missing_mask`` exactly as in the prior implementation.

        If ``outlet_normal`` is supplied, that fixed normal is used for every
        ExtrapBC voxel.  For D3Q27 the dedicated auxiliary field is then sized
        for up to 19 missing populations so a complete face, including edges
        and corners, can be owned by ExtrapBC without using opposite-population
        bounce-back as a closure.
        """
        sound_speed = self.compute_dtype(1.0 / np.sqrt(3.0))
        _c = self.velocity_set.c
        _q = self.velocity_set.q
        _opp_indices = self.velocity_set.opp_indices
        _u_vec = wp.vec(self.velocity_set.d, dtype=self.compute_dtype)

        # Capture the optional prescribed normal as compile-time constants for
        # Warp.  When absent, get_normal_vectors_neon() retains the original
        # missing-mask-based normal lookup.
        _has_prescribed_normal = self.outlet_normal is not None
        _normal_x = self.outlet_normal[0] if self.outlet_normal is not None else 0
        _normal_y = self.outlet_normal[1] if self.outlet_normal is not None else 0
        _normal_z = (
            self.outlet_normal[2]
            if self.outlet_normal is not None and self.velocity_set.d == 3
            else 0
        )

        # IMPORTANT: use the same cardinality exposed to simulation_manager.
        # This is 9 for the legacy/default planar D3Q27 outlet and 19 when a
        # prescribed normal is used for a full D3Q27 outlet face.
        _aux_cardinality = int(self.extrap_aux_cardinality)

        # Use the same equilibrium formulation used by the multires stepper.
        # QuadraticEquilibrium is linear in rho for fixed u, therefore
        # feq(delta_rho, u) == feq(rho + delta_rho, u) - feq(rho, u).
        equilibrium = QuadraticEquilibrium(
            self.velocity_set,
            self.precision_policy,
            ComputeBackend.WARP,
        )
        _rho_reference = self.compute_dtype(self.rho_reference)
        _rho_relaxation = self.compute_dtype(self.rho_relaxation)
        _enable_rho_reference = self.rho_relaxation > 0.0
        _rho_epsilon = self.compute_dtype(1.0e-12)

        @wp.func
        def get_normal_vectors_neon(missing_mask: Any):
            # Fixed/prescribed normal takes precedence when provided.
            if wp.static(_has_prescribed_normal):
                if wp.static(self.velocity_set.d == 3):
                    return wp.vec3i(
                        wp.int32(_normal_x),
                        wp.int32(_normal_y),
                        wp.int32(_normal_z),
                    )
                else:
                    return wp.vec2i(
                        wp.int32(_normal_x),
                        wp.int32(_normal_y),
                    )

            # No prescribed normal: preserve the prior behavior and infer the
            # local axis-aligned normal from the missing population mask.
            if wp.static(self.velocity_set.d == 3):
                for l in range(_q):
                    if (
                        missing_mask[l] == wp.uint8(1)
                        and wp.abs(_c[0, l]) + wp.abs(_c[1, l]) + wp.abs(_c[2, l]) == 1
                    ):
                        return -wp.vec3i(_c[0, l], _c[1, l], _c[2, l])
            else:
                for l in range(_q):
                    if (
                        missing_mask[l] == wp.uint8(1)
                        and wp.abs(_c[0, l]) + wp.abs(_c[1, l]) == 1
                    ):
                        return -wp.vec2i(_c[0, l], _c[1, l])

        @wp.func
        def functional(
            index: Any,
            timestep: Any,
            missing_mask: Any,
            f_0: Any,
            f_1: Any,
            _f_pre: Any,
            _f_post: Any,
            extrap_aux_pn: Any,
        ):
            """Restore prepared Extrap populations after streaming.

            Values are packed in ascending lattice-direction order for every
            missing population.  With a prescribed D3Q27 outlet normal this
            naturally consumes 9 slots on a face, 15 on an edge, and 19 at a
            three-plane corner.
            """
            _f = _f_post
            aux_slot = wp.int32(0)

            for l in range(_q):
                if missing_mask[l] == wp.uint8(1):
                    if aux_slot < _aux_cardinality:
                        _f[l] = self.compute_dtype(
                            wp.neon_read(
                                extrap_aux_pn,
                                index,
                                aux_slot,
                            )
                        )
                    else:
                        # Defensive compatibility path for the legacy compact
                        # planar allocation if a non-planar voxel is supplied
                        # without an explicit outlet_normal.
                        _f[l] = _f_pre[_opp_indices[l]]

                    aux_slot += wp.int32(1)

            # Optional equilibrium-only density reference.  rho_relaxation=0
            # compiles this block away.
            if wp.static(_enable_rho_reference):
                rho_local = self.compute_dtype(0.0)
                momentum = _u_vec()
                for d in range(self.velocity_set.d):
                    momentum[d] = self.compute_dtype(0.0)

                for l in range(_q):
                    fi = _f[l]
                    rho_local += fi
                    for d in range(self.velocity_set.d):
                        if _c[d, l] == 1:
                            momentum[d] += fi
                        elif _c[d, l] == -1:
                            momentum[d] -= fi

                if rho_local > _rho_epsilon:
                    u_local = _u_vec()
                    inv_rho = self.compute_dtype(1.0) / rho_local
                    for d in range(self.velocity_set.d):
                        u_local[d] = momentum[d] * inv_rho

                    delta_rho = _rho_relaxation * (_rho_reference - rho_local)
                    feq_delta = equilibrium.warp_functional(delta_rho, u_local)

                    for l in range(_q):
                        _f[l] += feq_delta[l]

            return _f

        @wp.func
        def assemble_auxiliary_data_neon(
            index: Any,
            timestep: Any,
            missing_mask: Any,
            f_0: Any,
            f_1: Any,
            _f_pre: Any,
            _f_post: Any,
            extrap_aux_pn: Any,
            f0_is_pre_stream_source: Any = 1,
        ):
            """Prepare next-stream Extrap data without altering physical f.

            Split multires levels already hold post-stream populations in f_0,
            so the interior sample is x-n.  A fused finest-level path instead
            sees the pre-stream source field and uses x-n-c_l.

            With a prescribed normal every missing population is extrapolated
            along that normal and packed into the dedicated auxiliary field.
            Without a prescribed normal the same code uses the legacy inferred
            normal from ``missing_mask``.
            """
            nv = get_normal_vectors_neon(missing_mask)
            aux_slot = wp.int32(0)

            for lattice_dir in range(_q):
                if missing_mask[lattice_dir] == wp.uint8(1):
                    # Split path: sample the already-streamed interior voxel at
                    # x - n.  The prescribed normal, when present, is used here
                    # directly for every face/edge/corner population.
                    offset = wp.vec3i(-nv[0], -nv[1], -nv[2])

                    # Fused finest path sees pre-stream source data.  Recover
                    # what the post-stream value at x-n would have been by also
                    # applying the population pull direction.
                    if f0_is_pre_stream_source != wp.int32(0):
                        offset[0] = offset[0] - _c[0, lattice_dir]
                        offset[1] = offset[1] - _c[1, lattice_dir]
                        offset[2] = offset[2] - _c[2, lattice_dir]

                    offset_pull_index = wp.neon_ngh_idx(
                        wp.int8(offset[0]),
                        wp.int8(offset[1]),
                        wp.int8(offset[2]),
                    )

                    is_valid = wp.bool(False)
                    f_aux = self.compute_dtype(
                        wp.neon_read_ngh(
                            f_0,
                            index,
                            offset_pull_index,
                            lattice_dir,
                            self.store_dtype(0.0),
                            is_valid,
                        )
                    )

                    if not is_valid:
                        # Same-population zero-gradient fallback for an invalid
                        # interior lookup.  This is not opposite-population
                        # bounce-back.
                        f_aux = _f_pre[lattice_dir]

                    if aux_slot < _aux_cardinality:
                        prepared = (
                            (self.compute_dtype(1.0) - sound_speed) * _f_pre[lattice_dir]
                            + sound_speed * f_aux
                        )
                        wp.neon_write(
                            extrap_aux_pn,
                            index,
                            aux_slot,
                            self.store_dtype(prepared),
                        )

                    aux_slot += wp.int32(1)

            # Do not replace outgoing physical populations with BC scratch data.
            return _f_post

        return functional, assemble_auxiliary_data_neon

    def _construct_neon(self):
        """Construct the legacy single-resolution Neon implementation.

        Single-resolution Neon continues to use the historical population-slot
        storage so the existing ``nse_stepper.py`` call signature is unchanged.
        Multires never calls these functions; it uses
        ``neon_multires_functional`` / ``assemble_auxiliary_data_multires``.
        """
        sound_speed = self.compute_dtype(1.0 / np.sqrt(3.0))
        _c = self.velocity_set.c
        _q = self.velocity_set.q
        _opp_indices = self.velocity_set.opp_indices
        _has_prescribed_normal = self.outlet_normal is not None
        _normal_x = self.outlet_normal[0] if self.outlet_normal is not None else 0
        _normal_y = self.outlet_normal[1] if self.outlet_normal is not None else 0
        _normal_z = (
            self.outlet_normal[2]
            if self.outlet_normal is not None and self.velocity_set.d == 3
            else 0
        )

        @wp.func
        def get_normal_vectors_neon_legacy(missing_mask: Any):
            if wp.static(_has_prescribed_normal):
                if wp.static(self.velocity_set.d == 3):
                    return wp.vec3i(
                        wp.int32(_normal_x),
                        wp.int32(_normal_y),
                        wp.int32(_normal_z),
                    )
                else:
                    return wp.vec2i(
                        wp.int32(_normal_x),
                        wp.int32(_normal_y),
                    )
            if wp.static(self.velocity_set.d == 3):
                for l in range(_q):
                    if (
                        missing_mask[l] == wp.uint8(1)
                        and wp.abs(_c[0, l]) + wp.abs(_c[1, l]) + wp.abs(_c[2, l]) == 1
                    ):
                        return -wp.vec3i(_c[0, l], _c[1, l], _c[2, l])
            else:
                for l in range(_q):
                    if missing_mask[l] == wp.uint8(1) and wp.abs(_c[0, l]) + wp.abs(_c[1, l]) == 1:
                        return -wp.vec2i(_c[0, l], _c[1, l])

        @wp.func
        def functional(
            index: Any,
            timestep: Any,
            missing_mask: Any,
            f_0: Any,
            f_1: Any,
            _f_pre: Any,
            _f_post: Any,
        ):
            _f = _f_post
            for l in range(_q):
                if missing_mask[l] == wp.uint8(1):
                    _f[l] = _f_pre[_opp_indices[l]]
            return _f

        @wp.func
        def assemble_auxiliary_data_neon_legacy(
            index: Any,
            timestep: Any,
            missing_mask: Any,
            f_0: Any,
            f_1: Any,
            _f_pre: Any,
            _f_post: Any,
        ):
            _f = _f_post
            nv = get_normal_vectors_neon_legacy(missing_mask)

            for lattice_dir in range(_q):
                if missing_mask[lattice_dir] == wp.uint8(1):
                    offset = wp.vec3i(
                        -nv[0] - _c[0, lattice_dir],
                        -nv[1] - _c[1, lattice_dir],
                        -nv[2] - _c[2, lattice_dir],
                    )
                    offset_pull_index = wp.neon_ngh_idx(
                        wp.int8(offset[0]), wp.int8(offset[1]), wp.int8(offset[2])
                    )
                    is_valid = wp.bool(False)
                    f_aux = self.compute_dtype(
                        wp.neon_read_ngh(
                            f_0,
                            index,
                            offset_pull_index,
                            lattice_dir,
                            self.store_dtype(0.0),
                            is_valid,
                        )
                    )
                    if not is_valid:
                        f_aux = _f_pre[lattice_dir]

                    _f[_opp_indices[lattice_dir]] = (
                        (self.compute_dtype(1.0) - sound_speed) * _f_pre[lattice_dir]
                        + sound_speed * f_aux
                    )

            return _f

        return (functional, assemble_auxiliary_data_neon_legacy), None

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_implementation(self, f_pre, f_post, bc_mask, missing_mask):
        # raise exception as this feature is not implemented yet
        raise NotImplementedError("This feature is not implemented in XLB with the NEON backend yet.")
