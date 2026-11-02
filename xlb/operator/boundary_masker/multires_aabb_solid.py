"""
Multi-resolution AABB-Solid boundary masker.

AABB-Close, plus: fluid that cannot reach the domain exterior through face-connected fluid, on any combination of
levels, is retagged as solid (255). Hollow bodies and wheel cavities that the AABB shell leaves sealed become solid,
so the flow solver does not run inside them.

The search runs once, after the boundaries of the last AABB-Solid mesh are in ``bc_mask``. Solid is simply
``bc_mask == 255``, which by then holds every mesh, so a cavity that the wheel and the body only seal together is
found. Enclosed cells are tagged in ``bc_mask`` directly and never go through a boundary pass, so no mesh BC (a
rotating wheel, say) can end up owning boundary cells it did not create.

The search runs entirely on the Neon grid. A single uint8 ``reach`` field is relaxed in place until nothing changes:

* a leaf cell is reached when a face neighbour is reached. The neighbour is read on the same level, or on the next
  coarser level where the cell sits on a level transition (the explosion read of the stepper);
* a refined cell (one with children, i.e. the coarse halo over a finer region) is reached when any child is, so
  reachability climbs from fine to coarse through ``neon_read_child``;
* a face with no cell on any level is the domain exterior and seeds the search.

Fields only change from 0 to 1, so racing writes are harmless and one container per level is enough. The containers
are built once and re-launched in a fine-to-coarse then coarse-to-fine sweep, which carries reachability across
several level transitions in a single pass.
"""

import time
import warp as wp
import neon
from typing import Any
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb.operator.boundary_masker.multires_aabb_close import MultiresMeshMaskerAABBClose
from xlb.cell_type import BC_SOLID

# Device counters shared by every masker. The kernels capture the array pointer, so the arrays are allocated once and
# never replaced: [changed] is set by any cell that becomes reached, [tagged] counts cells made solid, per level.
_MAX_LEVELS = 16
_changed = None
_tagged = None


def _device_counters():
    global _changed, _tagged
    if _changed is None:
        _changed = wp.zeros(1, dtype=wp.int32)
        _tagged = wp.zeros(_MAX_LEVELS, dtype=wp.int32)
    return _changed, _tagged


class MultiresMeshMaskerAABBSolid(MultiresMeshMaskerAABBClose):
    """
    AABB-Close masker that also solidifies fluid regions fully enclosed by solid cells (NEON backend).

    Parameters
    ----------
    fill_enclosed : bool
        Search for enclosed regions after this mesh. The search covers every solid cell already in ``bc_mask``, so it
        only needs to run once, for the last mesh. Otherwise this masker behaves like AABB-Close.
    seed_floor : bool
        If False (default) the lower z face of the domain is the ground, not an exterior opening, so a cavity that only
        touches the ground is still sealed. If True it counts as exterior like the other faces.
    max_passes : int
        Upper bound on relaxation passes. The search normally stops much earlier, when a batch changes nothing.
    batch : int
        Passes launched between convergence checks (each check synchronizes the device).
    """

    def __init__(
        self,
        velocity_set=None,
        precision_policy=None,
        compute_backend=None,
        close_voxels: int = None,
        fill_enclosed: bool = True,
        seed_floor: bool = False,
        max_passes: int = 4096,
        batch: int = 4,
    ):
        super().__init__(velocity_set, precision_policy, compute_backend, close_voxels)
        self.fill_enclosed = fill_enclosed
        self.seed_floor = seed_floor
        self.max_passes = max_passes
        self.batch = batch
        self.relax_container, self.tag_container = self._construct_fill_containers()

    def _construct_fill_containers(self):
        floor_is_exterior = 1 if self.seed_floor else 0

        # Is the cell reached through the face (cx, cy, cz)? Written once so the kernel stays small.
        @wp.func
        def reached_through_face(index: Any, reach_pn: Any, cx: wp.int32, cy: wp.int32, cz: wp.int32):
            ngh = wp.neon_ngh_idx(wp.int8(cx), wp.int8(cy), wp.int8(cz))
            reached = wp.bool(False)

            same_level = wp.bool(False)
            nval = wp.neon_read_ngh(reach_pn, index, ngh, 0, wp.uint8(0), same_level)
            if same_level:
                # A refined neighbour is a valid read too: its value is "any child reached"
                if nval != wp.uint8(0):
                    reached = wp.bool(True)
            else:
                coarser = wp.bool(False)
                cval = wp.uint8(0)
                if wp.neon_has_parent(reach_pn, index):
                    # On a level transition: the neighbour across this face is a coarser cell
                    cval = wp.neon_lbm_read_coarser_ngh(reach_pn, index, ngh, 0, wp.uint8(0), coarser)
                if coarser:
                    if cval != wp.uint8(0):
                        reached = wp.bool(True)
                else:
                    if not wp.neon_has_finer_ngh(reach_pn, index, ngh):
                        # No cell on any level across this face: the domain exterior (the ground, unless told otherwise)
                        if cz != wp.int32(-1) or floor_is_exterior == 1:
                            reached = wp.bool(True)
            return reached

        # One relaxation step on one level
        @neon.Container.factory(name="EnclosedFluidRelax")
        def relax(bc_mask: Any, reach: Any, changed: Any, level: Any):
            def launcher(loader: neon.Loader):
                loader.set_mres_grid(bc_mask.get_grid(), level)
                bc_pn = loader.get_mres_read_handle(bc_mask)
                reach_pn = loader.get_mres_write_handle(reach)

                @wp.func
                def kernel(index: Any):
                    if wp.neon_read(reach_pn, index, 0) != wp.uint8(0):
                        return

                    found = wp.bool(False)
                    if wp.neon_has_child(reach_pn, index):
                        # Refined cell: reached once any of its children is
                        for cx in range(2):
                            for cy in range(2):
                                for cz in range(2):
                                    child = wp.neon_ngh_idx(wp.int8(cx), wp.int8(cy), wp.int8(cz))
                                    child_valid = wp.bool(False)
                                    cval = wp.neon_read_child(reach_pn, index, child, 0, wp.uint8(0), child_valid)
                                    if child_valid:
                                        if cval != wp.uint8(0):
                                            found = wp.bool(True)
                    else:
                        if wp.neon_read(bc_pn, index, 0) == wp.uint8(BC_SOLID):
                            return
                        if reached_through_face(index, reach_pn, 1, 0, 0):
                            found = wp.bool(True)
                        if reached_through_face(index, reach_pn, -1, 0, 0):
                            found = wp.bool(True)
                        if reached_through_face(index, reach_pn, 0, 1, 0):
                            found = wp.bool(True)
                        if reached_through_face(index, reach_pn, 0, -1, 0):
                            found = wp.bool(True)
                        if reached_through_face(index, reach_pn, 0, 0, 1):
                            found = wp.bool(True)
                        if reached_through_face(index, reach_pn, 0, 0, -1):
                            found = wp.bool(True)

                    if found:
                        wp.neon_write(reach_pn, index, 0, wp.uint8(1))
                        changed[0] = wp.int32(1)

                loader.declare_kernel(kernel)

            return launcher

        # Fluid leaf cells that were never reached become solid
        @neon.Container.factory(name="EnclosedFluidTag")
        def tag(bc_mask: Any, reach: Any, tagged: Any, level: Any):
            def launcher(loader: neon.Loader):
                loader.set_mres_grid(bc_mask.get_grid(), level)
                bc_pn = loader.get_mres_write_handle(bc_mask)
                reach_pn = loader.get_mres_read_handle(reach)

                @wp.func
                def kernel(index: Any):
                    if wp.neon_has_child(bc_pn, index):
                        return
                    if wp.neon_read(bc_pn, index, 0) == wp.uint8(BC_SOLID):
                        return
                    if wp.neon_read(reach_pn, index, 0) == wp.uint8(0):
                        wp.neon_write(bc_pn, index, 0, wp.uint8(BC_SOLID))
                        wp.atomic_add(tagged, level, 1)

                loader.declare_kernel(kernel)

            return launcher

        return relax, tag

    # Backends are registered per defining class, so the inherited implementation needs its own entry
    @Operator.register_backend(ComputeBackend.NEON)
    def neon_implementation(self, bc, distances, bc_mask, missing_mask, normal_vector, normal_distance, stream=0):
        return super().neon_implementation(bc, distances, bc_mask, missing_mask, normal_vector, normal_distance, stream)

    def _finalize_bc_mask(self, grid, bc_mask, stream):
        if not self.fill_enclosed:
            return
        t0 = time.time()
        run = neon.Container.ContainerRuntime.neon
        levels = list(range(grid.num_levels))
        changed, tagged = _device_counters()

        reach = grid.new_field(cardinality=1, dtype=wp.uint8, memory_type=neon.MemoryType.device())
        for level in levels:
            reach.fill_run(level=level, value=wp.uint8(0), stream_idx=stream)

        # Built once, launched many times
        relax = {level: self.relax_container(bc_mask, reach, changed, level) for level in levels}
        sweep = levels + levels[-2::-1]  # fine to coarse, then back to fine

        passes = 0
        while passes < self.max_passes:
            changed.zero_()
            wp.synchronize()
            for _ in range(self.batch):
                for level in sweep:
                    relax[level].run(0, container_runtime=run)
            passes += self.batch
            grid.backend.sync()
            if int(changed.numpy()[0]) == 0:
                break
        t_search = time.time() - t0

        tagged.zero_()
        wp.synchronize()
        for level in levels:
            self.tag_container(bc_mask, reach, tagged, level).run(0, container_runtime=run)
        grid.backend.sync()
        counts = [int(n) for n in tagged.numpy()[: len(levels)]]
        print(f"\tAABB_SOLID: {sum(counts):,} enclosed fluid cells made solid ({counts} per level), {passes} passes, search {t_search:.1f} s, total {time.time() - t0:.1f} s", flush=True)
