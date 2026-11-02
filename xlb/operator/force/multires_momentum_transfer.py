"""Multi-resolution momentum-transfer force operator for the Neon backend."""

from typing import Any

import numpy as np
import warp as wp

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb.operator.force import MomentumTransfer
from xlb.mres_perf_optimization_type import MresPerfOptimizationType


@wp.kernel
def _accumulate_cell_force(src: wp.array(dtype=wp.vec3d), dst: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    dst[i] = dst[i] + src[i]


class MultiresMomentumTransfer(MomentumTransfer):
    """Compute boundary forces on L0 without floating-point atomic accumulation.

    Boundary-cell indices are built once from bc_mask and missing_mask. Each
    eligible cell writes to one fixed slot, followed by a GPU reduction.
    The masks must remain unchanged after the first force evaluation.
    """

    def __init__(
        self,
        no_slip_bc_instance,
        mres_perf_opt=MresPerfOptimizationType.NAIVE_COLLIDE_STREAM,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
        moment_reference_point=None,
        reference_density=None,
    ):
        from xlb.operator.force.momentum_transfer import LBMOperationSequence

        if compute_backend in (ComputeBackend.JAX, ComputeBackend.WARP):
            raise NotImplementedError(f"Operator {self.__class__.__name__} not supported in {compute_backend} backend.")

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
                "Mesh distance is only supported in the MultiresMomentumTransfer operator "
                "when the LBM operation sequence is STREAM_THEN_COLLIDE."
            )

        # Optional reference density in lattice units (freestream is 1.0). The equivalent
        # reference pressure is p_ref = rho_ref / 3 (cs^2 = 1/3), but the input stays a
        # density because the populations are density-based: a uniform state has f_i = w_i * rho_ref.
        # None leaves the force untouched. A value subtracts the momentum exchange of
        # that uniform state (2 * w_i * rho_ref per link) from every counted missing link, which removes the net
        # pressure offset on surfaces that are not closed (trimmed wheels, contact patches).
        # Affects the reported force/moment only, not the flow field.
        self.reference_density = None if reference_density is None else float(reference_density)

        self._l0_keys = None
        self._l0_masks = None
        self._force_buf = None
        self._moment_buf = None
        self._force_container = None

        # Optional per-cell force accumulation (set accumulate_cells before a force
        # evaluation to include it in the running mean returned by get_cell_forces).
        self.accumulate_cells = False
        self._cell_sum = None
        self._cell_samples = 0

        super().__init__(
            no_slip_bc_instance,
            operation_sequence,
            velocity_set,
            precision_policy,
            compute_backend,
            moment_reference_point=moment_reference_point,
        )

        if self.velocity_set.d != 3:
            raise NotImplementedError("L0 force-buffer reduction requires a 3D velocity set.")

        # CUB requires matching buffer and output types.
        self.force = wp.zeros((1,), dtype=wp.vec3d)
        self.moment = wp.zeros((1,), dtype=wp.vec3d)

    def _construct_neon(self):
        import neon

        self.fetcher_functional = self.fetcher.neon_functional

        _c = self.velocity_set.c
        _opp_indices = self.velocity_set.opp_indices
        _q = self.velocity_set.q
        _d = self.velocity_set.d
        _center = self.velocity_set.center_index
        _bc_id = self.no_slip_bc_instance.id
        _u_vec = wp.types.vector(_d, dtype=self.compute_dtype)
        _missing_mask_vec = wp.types.vector(_q, dtype=wp.uint8)
        _compute_moment = self.moment_reference_point is not None
        _w = self.velocity_set.w
        _use_ref_density = self.reference_density is not None
        _two_ref_rho = self.compute_dtype(2.0 * (self.reference_density if _use_ref_density else 0.0))
        _moment_ref = _u_vec()
        if _compute_moment:
            _moment_ref = _u_vec(*(self.compute_dtype(x) for x in self.moment_reference_point))

        @wp.func
        def is_edge(index: Any, bc_mask: Any, missing_mask: Any):
            return (
                wp.neon_read(bc_mask, index, 0) == wp.uint8(_bc_id)
                and wp.neon_read(missing_mask, index, _center) == wp.uint8(0)
            )

        @wp.func
        def position(index: Any, bc_mask: Any):
            p = wp.neon_global_idx(bc_mask, index)
            return wp.vec3i(wp.neon_get_x(p), wp.neon_get_y(p), wp.neon_get_z(p))

        @wp.func
        def key(p: wp.vec3i):
            # Three unsigned 21-bit coordinates packed into a positive int64.
            return (wp.int64(p[0]) << wp.int64(42)) | (wp.int64(p[1]) << wp.int64(21)) | wp.int64(p[2])

        @wp.func
        def find_slot(keys: wp.array(dtype=wp.int64), target: wp.int64, count: wp.int32):
            lo = wp.int32(0)
            hi = count - wp.int32(1)
            while lo <= hi:
                mid = lo + (hi - lo) // wp.int32(2)
                value = keys[mid]
                if value == target:
                    return mid
                if value < target:
                    lo = mid + wp.int32(1)
                else:
                    hi = mid - wp.int32(1)
            return wp.int32(-1)

        @neon.Container.factory(name="MomentumTransfer_count")
        def count_container(bc_mask: Any, missing_mask: Any, count: Any):
            def container_launcher(loader: neon.Loader):
                loader.set_mres_grid(bc_mask.get_grid(), 0)
                bc_mask_pn = loader.get_mres_read_handle(bc_mask)
                missing_mask_pn = loader.get_mres_read_handle(missing_mask)

                @wp.func
                def kernel(index: Any):
                    if is_edge(index, bc_mask_pn, missing_mask_pn):
                        wp.atomic_add(count, 0, wp.int32(1))

                loader.declare_kernel(kernel)

            return container_launcher

        @neon.Container.factory(name="MomentumTransfer_index")
        def index_container(bc_mask: Any, missing_mask: Any, keys: Any, count: Any, invalid: Any):
            def container_launcher(loader: neon.Loader):
                loader.set_mres_grid(bc_mask.get_grid(), 0)
                bc_mask_pn = loader.get_mres_read_handle(bc_mask)
                missing_mask_pn = loader.get_mres_read_handle(missing_mask)

                @wp.func
                def kernel(index: Any):
                    if is_edge(index, bc_mask_pn, missing_mask_pn):
                        p = position(index, bc_mask_pn)
                        if (p[0] < 0 or p[0] >= 2097152 or
                                p[1] < 0 or p[1] >= 2097152 or
                                p[2] < 0 or p[2] >= 2097152):
                            wp.atomic_add(invalid, 0, wp.int32(1))
                        else:
                            i = wp.atomic_add(count, 0, wp.int32(1))
                            keys[i] = key(p)

                loader.declare_kernel(kernel)

            return container_launcher

        @wp.func
        def functional(
            index: Any,
            f_0: Any,
            f_1: Any,
            bc_mask: Any,
            missing_mask: Any,
            keys: wp.array(dtype=wp.int64),
            count: wp.int32,
            force_buf: wp.array(dtype=wp.vec3d),
            moment_buf: wp.array(dtype=wp.vec3d),
            _rho: Any,
            _u: Any,
            _relax: Any,
            _norm_vec: Any,
            _norm_dist: Any,
        ):
            if is_edge(index, bc_mask, missing_mask):
                p = position(index, bc_mask)
                slot = find_slot(keys, key(p), count)
                if slot >= 0:
                    # Use the same population reconstruction and momentum
                    # calculation as MomentumTransfer, without its atomics.
                    mask = _missing_mask_vec()
                    for l in range(_q):
                        mask[l] = wp.neon_read(missing_mask, index, l)

                    f_post_collision, f_post_stream = self.fetcher_functional(
                        index, f_0, f_1, mask, _rho, _u, _relax,
                        _norm_vec, _norm_dist, wp.bool(False)
                    )

                    m = _u_vec()
                    for d in range(_d):
                        m[d] = self.compute_dtype(0.0)
                        for l in range(_q):
                            if mask[l] == wp.uint8(1):
                                phi = f_post_collision[_opp_indices[l]] + f_post_stream[l]
                                if wp.static(_use_ref_density):
                                    phi -= _two_ref_rho * _w[l]
                                if _c[d, _opp_indices[l]] == 1:
                                    m[d] += phi
                                elif _c[d, _opp_indices[l]] == -1:
                                    m[d] -= phi

                    force_buf[slot] = wp.vec3d(wp.float64(m[0]), wp.float64(m[1]), wp.float64(m[2]))
                    if wp.static(_compute_moment):
                        r = _u_vec()
                        r[0] = self.compute_dtype(p[0]) - _moment_ref[0]
                        r[1] = self.compute_dtype(p[1]) - _moment_ref[1]
                        r[2] = self.compute_dtype(p[2]) - _moment_ref[2]
                        moment = wp.cross(r, m)
                        moment_buf[slot] = wp.vec3d(
                            wp.float64(moment[0]), wp.float64(moment[1]), wp.float64(moment[2])
                        )

        @neon.Container.factory(name="MomentumTransfer")
        def container(
            f_0: Any, f_1: Any, bc_mask: Any, missing_mask: Any,
            keys: Any, count: Any, force_buf: Any, moment_buf: Any,
            _rho: Any, _u: Any, _relax: Any, _norm_vec: Any, _norm_dist: Any,
        ):
            def container_launcher(loader: neon.Loader):
                loader.set_mres_grid(bc_mask.get_grid(), 0)
                bc_mask_pn = loader.get_mres_read_handle(bc_mask)
                missing_mask_pn = loader.get_mres_read_handle(missing_mask)
                f_0_pn = loader.get_mres_read_handle(f_0)
                f_1_pn = loader.get_mres_read_handle(f_1)
                _rho_pn = loader.get_mres_read_handle(_rho)
                _u_pn = loader.get_mres_read_handle(_u)
                _relax_pn = loader.get_mres_read_handle(_relax)
                _norm_vec_pn = loader.get_mres_read_handle(_norm_vec)
                _norm_dist_pn = loader.get_mres_read_handle(_norm_dist)

                @wp.func
                def container_kernel(index: Any):
                    functional(
                        index, f_0_pn, f_1_pn, bc_mask_pn, missing_mask_pn,
                        keys, wp.int32(count), force_buf, moment_buf,
                        _rho_pn, _u_pn, _relax_pn, _norm_vec_pn, _norm_dist_pn,
                    )

                loader.declare_kernel(container_kernel)

            return container_launcher

        self._count_container = count_container
        self._index_container = index_container
        return functional, container

    def _build_l0_index(self, bc_mask, missing_mask, stream):
        import neon

        if self._l0_keys is not None:
            if bc_mask is not self._l0_masks[0] or missing_mask is not self._l0_masks[1]:
                raise ValueError("Boundary masks changed; construct a new momentum-transfer operator.")
            return

        count = wp.zeros((1,), dtype=wp.int32)
        self._count_container(bc_mask, missing_mask, count).run(stream)
        wp.synchronize_device()
        n = int(count.numpy()[0])

        if n == 0:
            self._l0_keys = wp.empty((0,), dtype=wp.int64)
        else:
            keys = wp.empty((n,), dtype=wp.int64)
            filled = wp.zeros((1,), dtype=wp.int32)
            invalid = wp.zeros((1,), dtype=wp.int32)
            self._index_container(bc_mask, missing_mask, keys, filled, invalid).run(stream)
            wp.synchronize_device()

            if int(invalid.numpy()[0]) != 0 or int(filled.numpy()[0]) != n:
                raise RuntimeError("Failed to build L0 boundary index: invalid coordinates or changed masks.")

            # Sorting once on the host keeps the index deterministic and avoids
            # an additional GPU sort kernel; force evaluations stay on the GPU.
            sorted_keys = np.sort(keys.numpy())
            if np.any(sorted_keys[1:] == sorted_keys[:-1]):
                raise RuntimeError("L0 boundary index has duplicate coordinates.")
            self._l0_keys = wp.array(sorted_keys, dtype=wp.int64)

        self._l0_masks = (bc_mask, missing_mask)
        self._force_buf = wp.empty((n,), dtype=wp.vec3d)
        self._moment_buf = wp.empty((n if self.moment_reference_point is not None else 1,), dtype=wp.vec3d)
        self._cell_sum = wp.zeros((n,), dtype=wp.vec3d)
        #print(f"MomentumTransfer BC {self.no_slip_bc_instance.id}: {n} L0 boundary cells")

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_implementation(
        self,
        f_0,
        f_1,
        bc_mask,
        missing_mask,
        _rho=None,
        _u=None,
        _relax=None,
        _norm_vec_pn=None,
        _norm_dist_pn=None,
        stream=0,
    ):
        if _rho is None or _u is None or _relax is None or _norm_vec_pn is None or _norm_dist_pn is None:
            raise TypeError("rho, u, relax, normal_vector and normal_distance are required")

        self._build_l0_index(bc_mask, missing_mask, stream)
        n = self._l0_keys.size
        if n == 0:
            self.force.zero_()
            self.moment.zero_()
            return self.force.numpy()[0]

        if self._force_container is None:
            self._force_container = self.neon_container(
                f_0, f_1, bc_mask, missing_mask,
                self._l0_keys, n, self._force_buf, self._moment_buf,
                _rho, _u, _relax, _norm_vec_pn, _norm_dist_pn,
            )
        self._force_container.run(stream)

        # Neon stream 0 is not guaranteed to be Warp's current stream.
        wp.synchronize_device()
        wp.utils.array_sum(self._force_buf, out=self.force)
        if self.moment_reference_point is not None:
            wp.utils.array_sum(self._moment_buf, out=self.moment)
        if self.accumulate_cells:
            wp.launch(_accumulate_cell_force, dim=n, inputs=[self._force_buf, self._cell_sum])
            self._cell_samples += 1
        return self.force.numpy()[0]

    def get_cell_forces(self):
        """Return finest-level boundary-cell indices (n, 3) and per-cell force (n, 3).

        The force is the mean over the accumulated evaluations, or the most recent
        evaluation if none were accumulated. Per-cell values are the same ones that
        are summed into the total, so they include the reference_density correction.
        Returns None before the first force evaluation.
        """
        if self._l0_keys is None or self._l0_keys.size == 0:
            return None
        keys = self._l0_keys.numpy()
        xyz = np.stack([keys >> 42, (keys >> 21) & 0x1FFFFF, keys & 0x1FFFFF], axis=1)
        if self._cell_samples > 0:
            force = self._cell_sum.numpy() / self._cell_samples
        else:
            force = self._force_buf.numpy()
        return xyz, force
