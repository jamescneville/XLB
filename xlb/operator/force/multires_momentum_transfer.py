"""Deterministic L0 momentum exchange for Neon mGrid, Option B prototype.

This implementation still launches the existing Neon mGrid traversal over every
active L0 cell.  It replaces highly contended float atomics with one unique
per-boundary-cell output slot and a deterministic CUB reduction.  Unlike
Option D, it does NOT execute a compact boundary-only force kernel.

All BC reconstruction is delegated to the existing FetchPopulations operator;
its caller must have the read-only ``update_wall_state=False`` modification.
"""

from typing import Any

import warp as wp

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb.operator.force import MomentumTransfer
from xlb.mres_perf_optimization_type import MresPerfOptimizationType


# Run once after sorting, not on every force output.  Using an atomic int32
# here does not affect the reproducibility of the floating-point reduction.
@wp.kernel
def _check_duplicate_l0_keys(
    keys: wp.array(dtype=wp.int64),
    duplicate_count: wp.array(dtype=wp.int32),
    n: int,
):
    i = wp.tid()
    if i + 1 < n:
        if keys[i] == keys[i + 1]:
            wp.atomic_add(duplicate_count, 0, wp.int32(1))


class MultiresMomentumTransfer(MomentumTransfer):
    """Neon momentum-transfer force with deterministic, L0-only reduction.

    One instance still represents ONE boundary-condition object. Body and
    individual wheels remain separate and retain their original BC physics.

    The L0 boundary-mask and central missing-mask predicates must remain
    unchanged during the life of the cached index. Call reset_l0_index() if
    either mask changes in-place (and rebuild on the next evaluation).

    ``validate_runtime=True`` enables a per-call failed-lookup check. It
    introduces an extra device-wide barrier and is meant for initial tests.
    """

    def __init__(
        self,
        no_slip_bc_instance,
        mres_perf_opt=MresPerfOptimizationType.NAIVE_COLLIDE_STREAM,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
        moment_reference_point=None,
        validate_runtime=True,
        gauge_reference_density=None,
        gauge_force_diagnostic=False,
    ):
        from xlb.operator.force.momentum_transfer import LBMOperationSequence

        if compute_backend != ComputeBackend.NEON:
            raise NotImplementedError("The L0 indexed implementation requires the Neon backend")

        if mres_perf_opt == MresPerfOptimizationType.NAIVE_COLLIDE_STREAM:
            operation_sequence = LBMOperationSequence.COLLIDE_THEN_STREAM
        elif mres_perf_opt in (
            MresPerfOptimizationType.FUSION_AT_FINEST,
            MresPerfOptimizationType.FUSION_AT_FINEST_SFV,
            MresPerfOptimizationType.FUSION_AT_FINEST_SFV_ALL,
        ):
            operation_sequence = LBMOperationSequence.STREAM_THEN_COLLIDE
        else:
            raise ValueError(f"Unknown performance optimization type: {mres_perf_opt}")

        if operation_sequence != LBMOperationSequence.STREAM_THEN_COLLIDE:
            assert not no_slip_bc_instance.needs_mesh_distance, (
                "Mesh distance is supported only with STREAM_THEN_COLLIDE")

        self.validate_runtime = bool(validate_runtime)
        # Experimental: remove the uniform reference-pressure contribution from
        # every counted MEM link.  This changes *force reporting only*; it does
        # not change the streaming BC or the computed flow field.
        self.gauge_reference_density = (
            None if gauge_reference_density is None else float(gauge_reference_density)
        )
        # The diagnostic observes the pressure offset in the SAME force kernel,
        # avoiding additional Neon operator instances (which returned zero in v1).
        self.capture_reference_pressure_bias = bool(gauge_force_diagnostic)
        self._l0_count = None
        self._indexed_fields = None
        self._force_container_cached = None
        self._l0_keys = None
        self._force_buf = None
        self._bias_force_buf = None
        self._moment_buf = None
        self._lookup_errors = None

        if velocity_set is not None and velocity_set.d != 3:
            raise NotImplementedError("This L0 prototype requires a 3D velocity set")

        super().__init__(
            no_slip_bc_instance,
            operation_sequence,
            velocity_set,
            precision_policy,
            compute_backend,
            moment_reference_point=moment_reference_point,
        )

        # CUB requires the output dtype to equal the contribution dtype.
        # Keep the per-voxel momentum calculation in its original compute
        # precision; convert only its final 3-vectors to double for summation.
        device = self.force.device
        self.force = wp.zeros((1,), dtype=wp.vec3d, device=device)
        self.moment = wp.zeros((1,), dtype=wp.vec3d, device=device)
        self.pressure_bias_force = (
            wp.zeros((1,), dtype=wp.vec3d, device=device)
            if self.capture_reference_pressure_bias else None
        )

    @property
    def boundary_voxel_count(self):
        """Number of indexed L0 cells; None before first force evaluation."""
        return self._l0_count

    def reset_l0_index(self):
        """Invalidate cached L0 mapping; next call rebuilds it from masks."""
        self._l0_count = None
        self._indexed_fields = None
        self._force_container_cached = None
        self._l0_keys = None
        self._force_buf = None
        self._bias_force_buf = None
        self._moment_buf = None
        self._lookup_errors = None

    def get_gauge_diagnostic_force_pair(self):
        """(raw, gauge, pressure_bias), from this operator's last evaluation.

        ``pressure_bias`` is the MEM contribution of the uniform reference
        density on the *same missing links* used for the primary force.
        Body primary: raw; wheel primary: gauge when wheelForceGaugeCorrect.
        This does NOT alter flow state, wall-link ownership, or BC physics.
        """
        if not self.capture_reference_pressure_bias:
            raise RuntimeError('Enable gauge_force_diagnostic on this operator first')
        if self._l0_count is None:
            raise RuntimeError('Evaluate the momentum-transfer operator before reading its diagnostic')
        primary = self.force.numpy()[0].astype('float64').copy()
        bias = self.pressure_bias_force.numpy()[0].astype('float64').copy()
        if self.gauge_reference_density is None:
            raw = primary
            gauge = primary - bias
        else:
            gauge = primary
            raw = primary + bias
        return raw, gauge, bias

    def _construct_neon(self):
        import neon

        bc_id = wp.uint8(self.no_slip_bc_instance.id)
        center = self.velocity_set.center_index
        q = self.velocity_set.q
        d = self.velocity_set.d
        compute_dtype = self.compute_dtype
        _moment_enabled = self.moment_reference_point is not None
        _apply_gauge_correction = self.gauge_reference_density is not None
        _gauge_rho = compute_dtype(
            self.gauge_reference_density if _apply_gauge_correction else 0.0
        )
        _w = self.velocity_set.w
        _capture_bias = self.capture_reference_pressure_bias
        _bias_rho = compute_dtype(
            self.gauge_reference_density if self.gauge_reference_density is not None else 1.0
        )
        _c = self.velocity_set.c
        _opp = self.velocity_set.opp_indices
        _fmask = wp.types.vector(q, dtype=wp.uint8)
        _uvec = wp.types.vector(d, dtype=compute_dtype)
        _ref = (
            _uvec(
                compute_dtype(self.moment_reference_point[0]),
                compute_dtype(self.moment_reference_point[1]),
                compute_dtype(self.moment_reference_point[2]),
            )
            if _moment_enabled else _uvec()
        )
        _debug_lookup = self.validate_runtime
        _maxcoord = 1 << 21

        @wp.func
        def _is_edge(index: Any, bc_pn: Any, missing_pn: Any):
            # Test cheap BC ID before any distribution or 27-direction reads.
            if wp.neon_read(bc_pn, index, 0) != bc_id:
                return wp.bool(False)
            return wp.neon_read(missing_pn, index, center) == wp.uint8(0)

        @wp.func
        def _xyz(index: Any, bc_pn: Any):
            p = wp.neon_global_idx(bc_pn, index)
            return wp.vec3i(
                wp.neon_get_x(p), wp.neon_get_y(p), wp.neon_get_z(p)
            )

        @wp.func
        def _valid_xyz(p: wp.vec3i):
            return (
                p[0] >= 0 and p[0] < _maxcoord and
                p[1] >= 0 and p[1] < _maxcoord and
                p[2] >= 0 and p[2] < _maxcoord
            )

        @wp.func
        def _pack_xyz(p: wp.vec3i):
            # 21 bits per coordinate; bits 0..62 only => positive int64 key.
            return (
                (wp.int64(p[0]) << wp.int64(42)) |
                (wp.int64(p[1]) << wp.int64(21)) |
                wp.int64(p[2])
            )

        @wp.func
        def _find_slot(keys: wp.array(dtype=wp.int64), target: wp.int64, n: wp.int32):
            lo = wp.int32(0)
            hi = n - wp.int32(1)
            while lo <= hi:
                mid = lo + (hi - lo) // wp.int32(2)
                val = keys[mid]
                if val == target:
                    return mid
                if val < target:
                    lo = mid + wp.int32(1)
                else:
                    hi = mid - wp.int32(1)
            return wp.int32(-1)

        @neon.Container.factory(name="MomentumTransfer_L0_count")
        def count_container(bc_mask: Any, missing_mask: Any, count: Any):
            def launcher(loader: neon.Loader):
                loader.set_mres_grid(bc_mask.get_grid(), 0)
                bc_pn = loader.get_mres_read_handle(bc_mask)
                missing_pn = loader.get_mres_read_handle(missing_mask)

                @wp.func
                def kernel(index: Any):
                    if _is_edge(index, bc_pn, missing_pn):
                        wp.atomic_add(count, 0, wp.int32(1))
                loader.declare_kernel(kernel)
            return launcher

        @neon.Container.factory(name="MomentumTransfer_L0_fill")
        def fill_container(
            bc_mask: Any,
            missing_mask: Any,
            keys: Any,
            count: Any,
            invalid: Any,
        ):
            def launcher(loader: neon.Loader):
                loader.set_mres_grid(bc_mask.get_grid(), 0)
                bc_pn = loader.get_mres_read_handle(bc_mask)
                missing_pn = loader.get_mres_read_handle(missing_mask)

                @wp.func
                def kernel(index: Any):
                    if _is_edge(index, bc_pn, missing_pn):
                        p = _xyz(index, bc_pn)
                        if not _valid_xyz(p):
                            wp.atomic_add(invalid, 0, wp.int32(1))
                        else:
                            i = wp.atomic_add(count, 0, wp.int32(1))
                            keys[i] = _pack_xyz(p)
                loader.declare_kernel(kernel)
            return launcher

        @wp.func
        def functional(
            index: Any,
            f0_pn: Any,
            f1_pn: Any,
            bc_pn: Any,
            missing_pn: Any,
            keys: wp.array(dtype=wp.int64),
            n: wp.int32,
            force_buf: wp.array(dtype=wp.vec3d),
            bias_buf: wp.array(dtype=wp.vec3d),
            moment_buf: wp.array(dtype=wp.vec3d),
            lookup_errors: wp.array(dtype=wp.int32),
            rho_pn: Any,
            u_pn: Any,
            relax_pn: Any,
            normvec_pn: Any,
            normdist_pn: Any,
        ):
            if not _is_edge(index, bc_pn, missing_pn):
                return
            slot = _find_slot(keys, _pack_xyz(_xyz(index, bc_pn)), n)
            if slot < 0:
                if wp.static(_debug_lookup):
                    wp.atomic_add(lookup_errors, 0, wp.int32(1))
                return

            # Match original functional's operations and accumulation order.
            # Only after determining the slot do we read all 27 mask entries.
            mask = _fmask()
            for l in range(q):
                mask[l] = wp.neon_read(missing_pn, index, l)
            pre, post = self.fetcher_functional(
                index, f0_pn, f1_pn, mask,
                rho_pn, u_pn, relax_pn, normvec_pn, normdist_pn
            )
            m = _uvec()
            bias = _uvec()
            for axis in range(d):
                m[axis] = compute_dtype(0.0)
                if wp.static(_capture_bias):
                    bias[axis] = compute_dtype(0.0)
                for l in range(q):
                    if mask[l] == wp.uint8(1):
                        phi = pre[_opp[l]] + post[l]
                        if wp.static(_apply_gauge_correction):
                            # Retain the primary force exactly as before.
                            phi -= compute_dtype(2.0) * _w[l] * _gauge_rho
                        if wp.static(_capture_bias):
                            # This link's uniform-pressure contribution, using
                            # the very same link selection and direction.
                            p_ref_link = compute_dtype(2.0) * _w[l] * _bias_rho
                        if _c[axis, _opp[l]] == 1:
                            m[axis] += phi
                            if wp.static(_capture_bias):
                                bias[axis] += p_ref_link
                        elif _c[axis, _opp[l]] == -1:
                            m[axis] -= phi
                            if wp.static(_capture_bias):
                                bias[axis] -= p_ref_link
            force_buf[slot] = wp.vec3d(
                wp.float64(m[0]), wp.float64(m[1]), wp.float64(m[2])
            )
            if wp.static(_capture_bias):
                bias_buf[slot] = wp.vec3d(
                    wp.float64(bias[0]), wp.float64(bias[1]), wp.float64(bias[2])
                )
            if wp.static(_moment_enabled):
                p = _xyz(index, bc_pn)
                r = _uvec()
                r[0] = compute_dtype(p[0]) - _ref[0]
                r[1] = compute_dtype(p[1]) - _ref[1]
                r[2] = compute_dtype(p[2]) - _ref[2]
                torque = wp.cross(r, m)
                moment_buf[slot] = wp.vec3d(
                    wp.float64(torque[0]),
                    wp.float64(torque[1]),
                    wp.float64(torque[2]),
                )

        @neon.Container.factory(name="MomentumTransfer_L0_buffer")
        def force_container(
            f_0: Any, f_1: Any, bc_mask: Any, missing_mask: Any,
            keys: Any, n: Any, force_buf: Any, bias_buf: Any, moment_buf: Any,
            lookup_errors: Any, rho: Any, u: Any, relax: Any,
            normvec: Any, normdist: Any,
        ):
            def launcher(loader: neon.Loader):
                loader.set_mres_grid(bc_mask.get_grid(), 0)
                bc_pn = loader.get_mres_read_handle(bc_mask)
                missing_pn = loader.get_mres_read_handle(missing_mask)
                f0_pn = loader.get_mres_read_handle(f_0)
                f1_pn = loader.get_mres_read_handle(f_1)
                rho_pn = loader.get_mres_read_handle(rho)
                u_pn = loader.get_mres_read_handle(u)
                relax_pn = loader.get_mres_read_handle(relax)
                normvec_pn = loader.get_mres_read_handle(normvec)
                normdist_pn = loader.get_mres_read_handle(normdist)

                @wp.func
                def kernel(index: Any):
                    functional(
                        index, f0_pn, f1_pn, bc_pn, missing_pn,
                        keys, wp.int32(n), force_buf, bias_buf, moment_buf, lookup_errors,
                        rho_pn, u_pn, relax_pn, normvec_pn, normdist_pn,
                    )
                loader.declare_kernel(kernel)
            return launcher

        self._count_factory = count_container
        self._fill_factory = fill_container
        return functional, force_container

    def _ensure_l0_index(self, bc_mask, missing_mask, stream):
        import neon

        if self._l0_count is not None:
            if (self._indexed_fields[0] is not bc_mask or
                    self._indexed_fields[1] is not missing_mask):
                raise RuntimeError(
                    "Boundary-mask fields changed: call reset_l0_index() to rebuild"
                )
            return

        device = self.force.device
        count = wp.zeros((1,), dtype=wp.int32, device=device)
        c = self._count_factory(bc_mask, missing_mask, count)
        c.run(stream, container_runtime=neon.Container.ContainerRuntime.neon)

        # Different Neon/Warp streams may be in use; synchronize to transfer
        # the scalar count correctly. This is ONLY once during initialization.
        wp.synchronize_device(device)
        n = int(count.numpy()[0])
        self._l0_count = n
        self._indexed_fields = (bc_mask, missing_mask)
        if n == 0:
            self._l0_keys = wp.empty((0,), dtype=wp.int64, device=device)
            return

        # radix_sort_pairs requires 2*n capacity, even though only n keys
        # are valid. Keep only the first n afterward to reduce persistent VRAM.
        working_keys = wp.empty((2*n,), dtype=wp.int64, device=device)
        working_values = wp.zeros((2*n,), dtype=wp.int32, device=device)
        filled = wp.zeros((1,), dtype=wp.int32, device=device)
        invalid = wp.zeros((1,), dtype=wp.int32, device=device)
        c = self._fill_factory(bc_mask, missing_mask, working_keys, filled, invalid)
        c.run(stream, container_runtime=neon.Container.ContainerRuntime.neon)
        wp.synchronize_device(device)
        n_filled = int(filled.numpy()[0])
        n_invalid = int(invalid.numpy()[0])
        if n_invalid or n_filled != n:
            self.reset_l0_index()
            raise RuntimeError(
                f"Invalid L0 keys: filled={n_filled}, counted={n}, "
                f"out_of_range={n_invalid}. Each xyz must be in [0, 2**21)."
            )

        wp.utils.radix_sort_pairs(working_keys, working_values, n)
        duplicates = wp.zeros((1,), dtype=wp.int32, device=device)
        if n > 1:
            wp.launch(
                _check_duplicate_l0_keys,
                dim=n-1,
                inputs=[working_keys, duplicates, n],
                device=device,
            )
        wp.synchronize_device(device)
        n_duplicate = int(duplicates.numpy()[0])
        if n_duplicate:
            self.reset_l0_index()
            raise RuntimeError(f"L0 index contains {n_duplicate} duplicate coordinate keys")

        self._l0_keys = wp.empty((n,), dtype=wp.int64, device=device)
        wp.copy(self._l0_keys, working_keys, count=n)
        # Barrier before Neon accesses sorted keys on its own CUDA stream.
        wp.synchronize_device(device)
        self._force_buf = wp.empty((n,), dtype=wp.vec3d, device=device)
        self._bias_force_buf = (
            wp.empty((n,), dtype=wp.vec3d, device=device)
            if self.capture_reference_pressure_bias else self._force_buf
        )
        self._moment_buf = wp.empty((n,), dtype=wp.vec3d, device=device) if self.moment_reference_point is not None else wp.empty((1,), dtype=wp.vec3d, device=device)
        self._lookup_errors = wp.zeros((1,), dtype=wp.int32, device=device)
        print(f"[MomentumTransfer Option B] BC {self.no_slip_bc_instance.id}: "
              f"indexed {n:,} eligible cells on L0")

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_implementation(
        self, f_0, f_1, bc_mask, missing_mask,
        _rho=None, _u=None, _relax=None,
        _norm_vec_pn=None, _norm_dist_pn=None, stream=0,
    ):
        import neon

        if any(x is None for x in (_rho, _u, _relax, _norm_vec_pn, _norm_dist_pn)):
            raise TypeError("Neon momentum transfer requires rho, u, relax, normals, distances")

        self.fetcher_functional = self.fetcher.neon_functional
        self._ensure_l0_index(bc_mask, missing_mask, stream)
        if self._l0_count == 0:
            self.force.zero_()
            self.moment.zero_()
            if self.capture_reference_pressure_bias:
                self.pressure_bias_force.zero_()
            return self.force.numpy()[0]

        if self._force_container_cached is None:
            self._force_container_cached = self.neon_container(
                f_0, f_1, bc_mask, missing_mask,
                self._l0_keys, self._l0_count,
                self._force_buf, self._bias_force_buf, self._moment_buf, self._lookup_errors,
                _rho, _u, _relax, _norm_vec_pn, _norm_dist_pn,
            )

        if self.validate_runtime:
            self._lookup_errors.zero_()
            # Debug-only barrier: counter must be zero before Neon kernel.
            wp.synchronize_device(self.force.device)

        self._force_container_cached.run(
            stream, container_runtime=neon.Container.ContainerRuntime.neon
        )
        # Conservative Neon->Warp ordering. Replace with shared-stream/event
        # sequencing after verifying stream interoperability in your build.
        wp.synchronize_device(self.force.device)
        wp.utils.array_sum(self._force_buf, out=self.force)
        if self.capture_reference_pressure_bias:
            wp.utils.array_sum(self._bias_force_buf, out=self.pressure_bias_force)
        if self.moment_reference_point is not None:
            wp.utils.array_sum(self._moment_buf, out=self.moment)
        result = self.force.numpy()[0]
        if self.validate_runtime:
            n_fail = int(self._lookup_errors.numpy()[0])
            if n_fail:
                raise RuntimeError(
                    f"{n_fail} L0 edge cells could not find their sorted slots. "
                    "Masks may have changed; call reset_l0_index()."
                )
        return result
