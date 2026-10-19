"""
Bit-for-bit run-to-run determinism check for the multi-resolution Neon stepper.

Builds the same small multi-resolution flow-past-sphere case several times in one process, advances it a fixed
number of steps, and compares the resulting velocity/density fields bit for bit.

    python multires_determinism_check.py                  # float atomics (default)  -> expected to DIFFER run to run
    python multires_determinism_check.py --deterministic  # int32 fixed-point atomics -> expected to be IDENTICAL

Exit code 0 when all runs are bit-identical, 1 otherwise, so it can be used as a CI / bisect check.
With --compare-modes the final fields of the deterministic and float-atomic modes are also compared, which gives
the size of the fixed-point quantisation error relative to the float path.
"""

try:
    import neon
except ModuleNotFoundError:
    raise ModuleNotFoundError("The 'neon' module is required (Neon backend; Linux, Python 3.11/3.12).") from None

import argparse
import os
import sys

import numpy as np
import warp as wp

import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import multires_grid_factory
from xlb.operator.boundary_condition import FullwayBounceBackBC, HalfwayBounceBackBC, RegularizedBC, DoNothingBC
from xlb.operator.boundary_masker import MeshVoxelizationMethod
from xlb.utils import MultiresIO
from xlb.utils.mesher import make_cuboid_mesh, prepare_sparsity_pattern

HERE = os.path.dirname(os.path.abspath(__file__))


def build_level_data(stl_filename, num_finest_voxels_across_part):
    """Small 3-level cuboid refinement around the part (same recipe as multires_flow_past_sphere_3d.py)."""
    import trimesh

    domain_multiplier = [
        [4, 10, 4, 4, 4, 4],
        [2, 6, 3, 3, 3, 3],
        [1, 3, 2, 2, 2, 2],
    ]
    mesh = trimesh.load_mesh(stl_filename, process=False)
    assert not mesh.is_empty, "Loaded mesh is empty or invalid."
    min_bound = mesh.vertices.min(axis=0)
    part_size = mesh.vertices.max(axis=0) - min_bound
    voxel_size = min(part_size) / num_finest_voxels_across_part
    shift = np.array(
        [
            domain_multiplier[0][0] * part_size[0] - min_bound[0],
            domain_multiplier[0][2] * part_size[1] - min_bound[1],
            domain_multiplier[0][4] * part_size[2] - min_bound[2],
        ],
        dtype=float,
    )
    mesh.apply_translation(shift)
    mesh_vertices = np.asarray(mesh.vertices) / voxel_size
    tmp = os.path.join(HERE, "temp_determinism.stl")
    mesh.export(tmp)
    level_data = make_cuboid_mesh(voxel_size, domain_multiplier, tmp)
    os.remove(tmp)
    grid_shape_finest = tuple(int(i * 2 ** (len(level_data) - 1)) for i in level_data[-1][0].shape)
    return level_data, mesh_vertices, grid_shape_finest


def run_once(deterministic, num_steps, sphere_radius):
    """Build everything from scratch, advance num_steps, return {name: ndarray} of the final fields."""
    compute_backend = ComputeBackend.NEON
    precision_policy = PrecisionPolicy.FP32FP32
    velocity_set = xlb.velocity_set.D3Q27(precision_policy=precision_policy, compute_backend=compute_backend)
    xlb.init(velocity_set=velocity_set, default_backend=compute_backend, default_precision_policy=precision_policy)

    u_max = 0.04
    Re = 1000.0
    num_finest_voxels_across_part = 2 * sphere_radius
    level_data, sphere, grid_shape_finest = build_level_data(
        os.path.join(HERE, "stl-files", "sphere.stl"), num_finest_voxels_across_part
    )
    num_levels = len(level_data)
    sparsity_pattern, level_origins = prepare_sparsity_pattern(level_data)
    exporter = MultiresIO({"velocity": 3, "density": 1}, level_data)

    grid = multires_grid_factory(
        grid_shape_finest,
        velocity_set=velocity_set,
        sparsity_pattern_list=sparsity_pattern,
        sparsity_pattern_origins=[neon.Index_3d(*o) for o in level_origins],
    )

    coarsest = grid.count_levels - 1
    box = grid.bounding_box_indices(shape=grid.level_to_shape(coarsest))
    box_no_edge = grid.bounding_box_indices(shape=grid.level_to_shape(coarsest), remove_edges=True)
    inlet = [[] for _ in range(num_levels - 1)] + [box_no_edge["left"]]
    outlet = [[] for _ in range(num_levels - 1)] + [box_no_edge["right"]]
    walls = [box["bottom"][i] + box["top"][i] + box["front"][i] + box["back"][i] for i in range(velocity_set.d)]
    walls = [[] for _ in range(num_levels - 1)] + [np.unique(np.array(walls), axis=-1).tolist()]

    bc_left = RegularizedBC("velocity", prescribed_value=(u_max, 0.0, 0.0), indices=inlet)
    bc_walls = FullwayBounceBackBC(indices=walls)
    bc_outlet = DoNothingBC(indices=outlet)
    bc_sphere = HalfwayBounceBackBC(mesh_vertices=sphere, voxelization_method=MeshVoxelizationMethod("AABB"))

    visc = u_max * num_finest_voxels_across_part / Re
    omega_finest = 1.0 / (3.0 * visc + 0.5)

    from xlb.helper.initializers import CustomMultiresInitializer

    initializer = CustomMultiresInitializer(
        bc_id=bc_outlet.id,
        constant_velocity_vector=(u_max, 0.0, 0.0),
        velocity_set=velocity_set,
        precision_policy=precision_policy,
        compute_backend=compute_backend,
    )

    sim = xlb.helper.MultiresSimulationManager(
        omega_finest=omega_finest,
        grid=grid,
        boundary_conditions=[bc_walls, bc_left, bc_outlet, bc_sphere],
        collision_type="BGK",
        initializer=initializer,
        mres_perf_opt=xlb.mres_perf_optimization_type.MresPerfOptimizationType.FUSION_AT_FINEST,
        deterministic_accumulation=deterministic,
    )

    for _ in range(num_steps):
        sim.step()
    wp.synchronize()
    sim.macro(sim.f_0, sim.bc_mask, sim.rho, sim.u, streamId=0)
    wp.synchronize()

    data = exporter.get_fields_data({"velocity": sim.u, "density": sim.rho})
    return {k: np.array(v, copy=True) for k, v in data.items() if isinstance(v, np.ndarray)}


def compare(a, b):
    """Bitwise comparison of two field dicts. Returns (identical, summary string)."""
    identical = True
    lines = []
    for name in sorted(a):
        if name not in b or a[name].shape != b[name].shape:
            identical = False
            lines.append(f"  {name}: shape/name mismatch")
            continue
        x, y = a[name], b[name]
        bits_x = np.ascontiguousarray(x).view(np.uint8)
        bits_y = np.ascontiguousarray(y).view(np.uint8)
        same = np.array_equal(bits_x, bits_y)
        identical &= same
        diff = np.abs(x.astype(np.float64) - y.astype(np.float64))
        n_diff = int(np.count_nonzero(x != y))
        lines.append(f"  {name}: {'IDENTICAL' if same else 'DIFFERENT'}  differing values={n_diff}/{x.size}  max|diff|={diff.max():.3e}")
    return identical, "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deterministic", action="store_true", help="use int32 fixed-point accumulation")
    parser.add_argument("--compare-modes", action="store_true", help="also compare deterministic vs float-atomic output")
    parser.add_argument("--steps", type=int, default=200, help="coarsest-level steps per run")
    parser.add_argument("--runs", type=int, default=2, help="number of identical runs to compare")
    parser.add_argument("--sphere-radius", type=int, default=5)
    args = parser.parse_args()

    runs = []
    for r in range(args.runs):
        print(f"[run {r + 1}/{args.runs}] deterministic={args.deterministic} steps={args.steps}", flush=True)
        runs.append(run_once(args.deterministic, args.steps, args.sphere_radius))

    all_identical = True
    for r in range(1, len(runs)):
        ok, summary = compare(runs[0], runs[r])
        all_identical &= ok
        print(f"run 1 vs run {r + 1}: {'BIT-IDENTICAL' if ok else 'DIFFERENT'}\n{summary}")

    if args.compare_modes:
        print(f"[extra run] deterministic={not args.deterministic}", flush=True)
        other = run_once(not args.deterministic, args.steps, args.sphere_radius)
        _, summary = compare(runs[0], other)
        print(f"deterministic vs float-atomic (quantisation / ordering effect):\n{summary}")

    print("RESULT:", "PASS (bit-identical)" if all_identical else "FAIL (runs differ)")
    return 0 if all_identical else 1


if __name__ == "__main__":
    sys.exit(main())
