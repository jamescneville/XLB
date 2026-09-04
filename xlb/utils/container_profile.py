"""Per-container timing for the multi-resolution Neon skeleton.

A profiler-free alternative to ``nsys`` for answering "which container eats the
wall clock".  Neon dispatches the whole timestep as one skeleton run, so the
individual container costs are not visible from Python during a normal step.
This module runs each container in isolation with explicit synchronisation and
reports the breakdown.

It also reports the gap between the sum of the isolated container times and the
cost of a real ``sim.step()``.  A large positive gap means launch overhead or
skeleton serialisation rather than kernel work.

Enable with ``XLB_PROFILE_CONTAINERS=1``.
"""

import os
import time
from collections import OrderedDict

import numpy as np
import warp as wp

from xlb.cell_type import BC_NONE, BC_SFV, BC_SOLID


def enabled():
    """True when ``XLB_PROFILE_CONTAINERS`` requests a profiling run."""
    return os.environ.get("XLB_PROFILE_CONTAINERS", "0").strip().lower() in ("1", "true", "yes", "on")


def report_voxel_classes(sim, h5exporter):
    """Print the per-level bc_mask histogram (SOLID / SFV / CFV breakdown).

    The finest-level SFV:CFV ratio is what determines whether
    ``CFV_finest_fused_pull`` is expensive because of how many voxels it touches
    or because of what it does to each one.  Both finest-level containers launch
    over every voxel at the level and early-return on a ``bc_mask`` mismatch, so
    their cost ratio divided by their voxel-count ratio is the per-voxel cost
    ratio.
    """
    counts = [int(np.count_nonzero(p)) for p in sim.grid.sparsity_pattern_list]
    offsets = np.concatenate(([0], np.cumsum(counts)))
    data = np.asarray(h5exporter.get_fields_data({"bc_mask": sim.bc_mask})["bc_mask_0"]).reshape(-1)
    if data.size != offsets[-1]:
        print("  (bc_mask size %d != expected %d; skipping histogram)" % (data.size, offsets[-1]))
        return None

    print("\nVoxel classes by level (0 = finest):")
    print("  %-6s %12s %12s %12s %12s" % ("level", "total", "solid", "SFV", "CFV(BC+none)"))
    per_level = []
    for level in range(len(counts)):
        chunk = data[offsets[level]:offsets[level + 1]]
        n_solid = int(np.count_nonzero(chunk == BC_SOLID))
        n_sfv = int(np.count_nonzero(chunk == BC_SFV))
        n_cfv = int(chunk.size - n_solid - n_sfv)
        per_level.append((chunk.size, n_solid, n_sfv, n_cfv))
        print("  %-6d %12d %12d %12d %12d" % (level, chunk.size, n_solid, n_sfv, n_cfv))

    # needs_mres cross-tab: how many CFV voxels actually reach a coarser level
    nm_field = getattr(sim, "needs_mres", None)
    if nm_field is not None:
        try:
            # MultiresIO only accepts field names registered at construction;
            # register this one for the diagnostic read.
            cardinality_map = getattr(h5exporter, "field_name_cardinality_dict", None)
            if cardinality_map is not None and "needs_mres" not in cardinality_map:
                cardinality_map["needs_mres"] = 1
            nm = np.asarray(h5exporter.get_fields_data({"needs_mres": nm_field})["needs_mres_0"]).reshape(-1)
            if nm.size == data.size:
                print("\nneeds_mres split (0 = can skip explosion + coalescence):")
                print("  %-6s %14s %14s %14s" % ("level", "CFV total", "CFV needs=1", "CFV needs=0"))
                for level in range(len(counts)):
                    lo, hi = offsets[level], offsets[level + 1]
                    chunk, nmc = data[lo:hi], nm[lo:hi]
                    is_cfv = (chunk != BC_SOLID) & (chunk != BC_SFV)
                    n_cfv = int(is_cfv.sum())
                    n_need = int((is_cfv & (nmc != 0)).sum())
                    pct = (100.0 * (n_cfv - n_need) / n_cfv) if n_cfv else 0.0
                    print("  %-6d %14d %14d %14d  (%.1f%% skippable)" % (level, n_cfv, n_need, n_cfv - n_need, pct))
        except Exception as exc:
            print("  (needs_mres cross-tab unavailable: %r)" % (exc,))

    finest = data[offsets[0]:offsets[1]]
    ids, id_counts = np.unique(finest, return_counts=True)
    print("\nFinest-level bc_mask id histogram (id 1-253 are registered BCs):")
    for i, c in sorted(zip(ids.tolist(), id_counts.tolist()), key=lambda kv: -kv[1]):
        label = {BC_SOLID: "BC_SOLID", BC_SFV: "BC_SFV", BC_NONE: "BC_NONE (fluid, non-SFV)"}.get(i, "BC id %d" % i)
        print("  %-28s %12d  %6.2f%%" % (label, c, 100.0 * c / finest.size))
    return per_level


def profile_containers(sim, repeats=20, warmup=5, settle_steps=20, h5exporter=None):
    """Time every container in ``sim.app`` individually and print a breakdown.

    Parameters
    ----------
    sim : MultiresSimulationManager
        A fully constructed simulation.  ``sim.app`` holds the ordered list of
        Neon containers that make up one coarsest-level timestep.
    repeats : int
        Timed iterations per container.
    warmup : int
        Untimed iterations per container, to cover first-touch and any lazy
        module load.
    settle_steps : int
        Real ``sim.step()`` calls before measuring, so the field contents are
        representative rather than the initial condition.

    Notes
    -----
    Re-running a single container repeatedly does not advance the simulation
    correctly, so this is a *timing* harness only -- the fields are left in a
    meaningless state and the process should be terminated afterwards.  Branch
    behaviour is unaffected because every early-return in these kernels keys off
    ``bc_mask``, which is static after setup.
    """
    print("\n" + "=" * 78)
    print("XLB container profile (XLB_PROFILE_CONTAINERS=1)")
    print("=" * 78)

    per_level = None
    if h5exporter is not None:
        try:
            per_level = report_voxel_classes(sim, h5exporter)
        except Exception as exc:  # diagnostics must never break the timing run
            print("  (voxel-class histogram unavailable: %r)" % (exc,))
    else:
        print("  (no h5exporter passed; skipping voxel-class histogram)")

    for _ in range(settle_steps):
        sim.step()
    wp.synchronize()

    # Reference: cost of one real timestep, skeleton-dispatched.
    step_repeats = max(5, repeats // 2)
    for _ in range(3):
        sim.step()
    wp.synchronize()
    t0 = time.perf_counter()
    for _ in range(step_repeats):
        sim.step()
    wp.synchronize()
    full_step_ms = (time.perf_counter() - t0) * 1e3 / step_repeats

    # Per-container isolated timings.
    results = []
    for idx, container in enumerate(sim.app):
        name = getattr(container, "name", f"<container {idx}>")
        for _ in range(warmup):
            container.run(0)
        wp.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            container.run(0)
        wp.synchronize()
        results.append((idx, name, (time.perf_counter() - t0) * 1e3 / repeats))

    total = sum(ms for _, _, ms in results)

    # Aggregate by container name, since the same container appears many times
    # per timestep at different levels and buffer parities.
    by_name = OrderedDict()
    for _, name, ms in results:
        count, acc = by_name.get(name, (0, 0.0))
        by_name[name] = (count + 1, acc + ms)

    print("\nBy container type (summed over every invocation in one timestep):")
    print("  %-34s %6s %10s %10s %7s" % ("container", "count", "total ms", "each ms", "share"))
    for name, (count, acc) in sorted(by_name.items(), key=lambda kv: -kv[1][1]):
        print("  %-34s %6d %10.3f %10.4f %6.1f%%" % (name, count, acc, acc / count, 100.0 * acc / total))
    print("  %-34s %6d %10.3f %10s %6.1f%%" % ("TOTAL", len(results), total, "", 100.0))

    print("\nLaunch order (one full timestep):")
    for idx, name, ms in results:
        print("  %3d  %-34s %8.4f ms  %5.1f%%" % (idx, name, ms, 100.0 * ms / total))

    gap = full_step_ms - total
    print("\nSum of isolated containers : %8.3f ms" % total)
    print("Real sim.step()            : %8.3f ms" % full_step_ms)
    print("Gap (overhead / stalls)    : %8.3f ms  (%.1f%% of the real step)" % (gap, 100.0 * gap / full_step_ms))
    if gap > 0.15 * full_step_ms:
        print("  -> Large gap: time is going to launch overhead or skeleton")
        print("     serialisation, not kernel work.")
    else:
        print("  -> Small gap: the step cost is dominated by kernel work, so")
        print("     optimise the containers at the top of the table above.")
    # Normalise the two finest-level containers by the number of voxels each
    # actually processes.  They launch identical thread counts and differ only
    # in which threads early-return, so this isolates per-voxel cost.
    if per_level:
        _, _, n_sfv, n_cfv = per_level[0]
        # CFV may be one container or the needs_mres split pair; sum whichever
        # are present, per buffer-parity invocation.
        cfv_names = [k for k in by_name if k.startswith("CFV_") and k.endswith("_finest_fused_pull")]
        sfv = by_name.get("SFV_finest_fused_pull")
        if cfv_names and sfv and n_sfv and n_cfv:
            parities = max(by_name[k][0] for k in cfv_names)
            cfv_total = sum(by_name[k][1] for k in cfv_names)
            cfv = (parities, cfv_total)
            if len(cfv_names) > 1:
                print("\n  (CFV split across %s)" % ", ".join(sorted(cfv_names)))
            cfv_ns = cfv[1] / cfv[0] * 1e6 / n_cfv
            sfv_ns = sfv[1] / sfv[0] * 1e6 / n_sfv
            print("\nFinest level, cost per processed voxel:")
            print("  SFV_finest_fused_pull : %10.2f ns/voxel  (%d voxels)" % (sfv_ns, n_sfv))
            print("  CFV_finest_fused_pull : %10.2f ns/voxel  (%d voxels)" % (cfv_ns, n_cfv))
            print("  CFV is %.1fx the per-voxel cost of SFV" % (cfv_ns / sfv_ns))

    print("=" * 78 + "\n")

    return {"by_name": by_name, "ordered": results, "full_step_ms": full_step_ms, "gap_ms": gap}
