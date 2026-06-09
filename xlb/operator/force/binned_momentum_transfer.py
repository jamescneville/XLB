"""
Streamwise-binned multi-resolution momentum-transfer force operator.

This extends :class:`MultiresMomentumTransfer` so that, instead of accumulating
the boundary force into a single global vector, the per-boundary-node momentum
contribution is accumulated into ``num_bins`` bins indexed by the node's
streamwise (x) position.  Summing all bins reproduces the total force exactly,
so this operator can be used as a drop-in replacement for the scalar Cd/Cl
computation while additionally yielding the force distribution along the
vehicle (used to plot cumulative Cd(x) / Cl(x)).

The car/no-slip boundary lives only on the finest level, and Neon's
``wp.neon_get_x`` always returns the global coordinate in finest-level units
(see ``xlb/utils/mesher.py`` and ``helper_functions_bc.py``), so the global x
index is exactly the finest-level streamwise cell index used for binning.
"""

from typing import Any

import warp as wp

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb.operator.force.multires_momentum_transfer import MultiresMomentumTransfer
from xlb.mres_perf_optimization_type import MresPerfOptimizationType


class MultiresBinnedMomentumTransfer(MultiresMomentumTransfer):
    """Momentum-transfer force binned along the streamwise (x) direction.

    Parameters
    ----------
    no_slip_bc_instance : BoundaryCondition
        The no-slip BC whose tagged voxels define the force integration surface.
    num_bins : int
        Number of streamwise bins.
    bin_origin_x : int
        Global finest-level x index corresponding to the start (left edge) of
        the first bin (e.g. the streamwise index of the vehicle nose).
    bin_width_cells : float
        Width of each bin in finest-level cells:
        ``(vehicle length in finest cells) / num_bins``.
    mres_perf_opt, velocity_set, precision_policy, compute_backend
        Same meaning as in :class:`MultiresMomentumTransfer`.
    """

    def __init__(
        self,
        no_slip_bc_instance,
        num_bins: int,
        bin_origin_x: int,
        bin_width_cells: float,
        mres_perf_opt=MresPerfOptimizationType.NAIVE_COLLIDE_STREAM,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
    ):
        self.num_bins = int(num_bins)
        self.bin_origin_x = float(bin_origin_x)
        self.bin_width_cells = float(bin_width_cells)

        # Builds the fetcher and (via the base) calls _construct_neon, which in
        # turn calls our overridden _construct_warp.
        super().__init__(
            no_slip_bc_instance,
            mres_perf_opt,
            velocity_set,
            precision_policy,
            compute_backend,
        )

        # Re-allocate the force accumulator as one d-vector per bin
        # (the base allocated a length-1 array).
        _u_vec = wp.vec(self.velocity_set.d, dtype=self.compute_dtype)
        self.force = wp.zeros((self.num_bins), dtype=_u_vec)

    def _construct_warp(self):
        # Local constants (mirrors MomentumTransfer._construct_warp)
        _c = self.velocity_set.c
        _opp_indices = self.velocity_set.opp_indices
        _u_vec = wp.vec(self.velocity_set.d, dtype=self.compute_dtype)
        _missing_mask_vec = wp.vec(self.velocity_set.q, dtype=wp.uint8)
        _no_slip_id = self.no_slip_bc_instance.id
        lattice_central_index = self.velocity_set.center_index

        # Binning constants
        _bin_x0 = self.compute_dtype(self.bin_origin_x)
        _inv_bin_width = self.compute_dtype(1.0 / self.bin_width_cells)
        _num_bins = wp.int32(self.num_bins)

        @wp.func
        def functional(
            index: Any,
            f_0: Any,
            f_1: Any,
            bc_mask: Any,
            missing_mask: Any,
            force: Any,
            _rho: Any,
            _u: Any,
            _relax: Any,
            _norm_vec_pn: Any,
            _norm_dist_pn: Any,
        ):
            # Get the boundary id
            _boundary_id = self.read_field(bc_mask, index, 0)
            _missing_mask = _missing_mask_vec()
            for l in range(self.velocity_set.q):
                _missing_mask[l] = self.read_field(missing_mask, index, l)

            # Determine if boundary is an edge by checking if center is missing
            is_edge = wp.bool(False)
            if _boundary_id == wp.uint8(_no_slip_id):
                if _missing_mask[lattice_central_index] == wp.uint8(0):
                    is_edge = wp.bool(True)

            # If the boundary is an edge then add the momentum transfer
            m = _u_vec()
            if is_edge:
                # fetch the post-collision and post-streaming populations
                f_post_collision, f_post_stream = self.fetcher_functional(
                    index, f_0, f_1, _missing_mask, _rho, _u, _relax, _norm_vec_pn, _norm_dist_pn
                )

                # Compute the momentum transfer
                for d in range(self.velocity_set.d):
                    m[d] = self.compute_dtype(0.0)
                    for l in range(self.velocity_set.q):
                        if _missing_mask[l] == wp.uint8(1):
                            phi = f_post_collision[_opp_indices[l]] + f_post_stream[l]
                            if _c[d, _opp_indices[l]] == 1:
                                m[d] += phi
                            elif _c[d, _opp_indices[l]] == -1:
                                m[d] -= phi

                # Streamwise bin from the global x index (finest-level units)
                cIdx = wp.neon_global_idx(bc_mask, index)
                gx = wp.neon_get_x(cIdx)
                b = wp.int32((self.compute_dtype(gx) - _bin_x0) * _inv_bin_width)
                if b < wp.int32(0):
                    b = wp.int32(0)
                if b >= _num_bins:
                    b = _num_bins - wp.int32(1)

                # Atomic sum into the streamwise bin
                wp.atomic_add(force, b, m)

        return functional, None

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
        import neon

        if _rho is None or _u is None:
            raise TypeError("rho and u must be provided: momentum_transfer(f_0, f_1, bc_mask, missing_mask, rho, u)")

        # Zero the per-bin force accumulator
        self.force *= self.compute_dtype(0.0)

        self.fetcher_functional = self.fetcher.neon_functional

        grid = bc_mask.get_grid()
        for level in range(grid.num_levels):
            c = self.neon_container(
                f_0, f_1, bc_mask, missing_mask, self.force, _rho, _u, _relax, _norm_vec_pn, _norm_dist_pn, level
            )
            c.run(stream, container_runtime=neon.Container.ContainerRuntime.neon)

        # Returns the full (num_bins, d) distribution. Sum over axis 0 to get
        # the total force vector (identical to MultiresMomentumTransfer).
        return self.force.numpy()
