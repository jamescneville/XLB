import neon
import warp as wp
import numpy as np
import time
import os
import re
import matplotlib.pyplot as plt
import trimesh
import shutil
from tabulate import tabulate
from typing import Any

import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import multires_grid_factory
from xlb.operator.boundary_condition import (
    FullwayBounceBackBC,
    HalfwayBounceBackBC,
    RegularizedBC,
    ExtrapolationOutflowBC,
    DoNothingBC,
    ZouHeBC,
    HybridBC,
)
from xlb.operator.boundary_masker import MeshVoxelizationMethod
from xlb.utils.mesher import prepare_sparsity_pattern, make_cuboid_mesh, MultiresIO
from xlb.utils.makemesh import generate_mesh
from xlb.utils import UnitConvertor
from xlb.operator.force import MultiresMomentumTransfer
from xlb.helper.initializers import CustomMultiresInitializer
from xlb import MresPerfOptimizationType

wp.clear_kernel_cache()
wp.config.quiet = True

# User Configuration
# =================
# Physical and simulation parameters
voxel_size = 0.002  # Finest voxel size in meters
ulb = 0.05         # Lattice velocity
u_physical = 40  # Physical inlet velocity in m/s (user input)
flow_passes = 5    # Domain flow passes
kinematic_viscosity = 1.508e-5  # Kinematic viscosity of air in m^2/s

# STL filename
stl_filename = "examples/stl/Ahmed_25_NoLegs.stl"
script_name = "Ahmed 25 2mm BBReg_noIC1a_ULB05"

# I/O settings
print_interval_percentage = 1   # Print every 1% of iterations
file_output_crossover_percentage = 80  # Crossover at 50% of iterations
num_file_outputs_pre_crossover =1    # Outputs before crossover
num_file_outputs_post_crossover = 1   # Outputs after crossover

# Other setup parameters
compute_backend = ComputeBackend.NEON
precision_policy = PrecisionPolicy.FP32FP32
velocity_set = xlb.velocity_set.D3Q27(precision_policy=precision_policy, compute_backend=compute_backend)

# Choose mesher type
mesher_type = "makemesh"  # Options: "makemesh" or "cuboid"

# Mesh Generation Functions
# =========================
def generate_makemesh_mesh(stl_filename, voxel_size, ground_refinement_level=3, ground_voxel_height=6):
    """
    Generate a makemesh mesh based on the provided voxel size in meters, domain multipliers, and padding values.
    """
    # Number of requested refinement levels
    num_levels = 5

    # Domain multipliers for the full domain
    domain_multiplier = {
        "-x": 2,
        "x": 3,
        "-y": 6,
        "y": 6,
        "-z": 0.17361, # gap size in meters 50mm = 0.17361 multiplier
        "z": 6,
    }


    padding_values = [
        [20, 20, 20, 20, 20, 20],
		[20, 80, 20, 20, 20, 20],
		[15, 30, 15, 15, 15, 15],
		[10, 20, 15, 15, 15, 15],
        [15, 15, 15, 15, 15, 15],
		[20, 20, 20, 20, 20, 20],
		[20, 30, 20, 20, 20, 20],
		[20, 20, 20, 20, 20, 20]
		]

    # padding_values = [
        # [20, 20, 20, 20, 40, 20],
        # [30, 80, 30, 30, 30, 30],
        # [10, 80, 10, 10, 10, 10],
        # [10, 80, 10, 10, 10, 10],
        # [8, 20, 8, 8, 8, 8],
        # [8, 20, 8, 8, 8, 8],
        # [4, 4, 4, 4, 4, 4],
        # [4, 4, 4, 4, 4, 4],
        # [4, 4, 4, 4, 4, 4],
        # [4, 4, 4, 4, 4, 4],
    # ]

    # Load the mesh
    mesh = trimesh.load_mesh(stl_filename, process=False)
    if mesh.is_empty:
        raise ValueError("Loaded mesh is empty or invalid.")

    # Compute original bounds
    min_bound = mesh.vertices.min(axis=0)
    max_bound = mesh.vertices.max(axis=0)
    partSize = max_bound - min_bound

    # Compute translation to put mesh into first octant of the domain
    shift = np.array(
        [
            domain_multiplier["-x"] * partSize[0] - min_bound[0],
            domain_multiplier["-y"] * partSize[1] - min_bound[1],
            domain_multiplier["-z"] * partSize[2] - min_bound[2],
        ],
        dtype=float,
    )

    # Apply translation and save out temp STL
    mesh.apply_translation(shift)
    _ = mesh.vertex_normals
    mesh_vertices = np.asarray(mesh.vertices) / voxel_size
    mesh.export("temp.stl")

    # Generate mesh using generate_mesh with ground refinement
    level_data = generate_mesh(
        num_levels,
        "temp.stl",
        voxel_size,
        padding_values,
        domain_multiplier,
        ground_refinement_level=ground_refinement_level,
        ground_voxel_height=ground_voxel_height,
    )
    sparsity_pattern, level_origins = prepare_sparsity_pattern(level_data)
    x0 = max_bound[0]
    actual_num_levels = len(level_data)
    grid_shape_finest = tuple([int(i * 2 ** (actual_num_levels - 1)) for i in level_data[-1][0].shape])
    print(f"Requested levels: {num_levels}, Actual levels: {actual_num_levels}")
    print(f"Full shape based on finest voxel size is {grid_shape_finest}")
    os.remove("temp.stl")

    return level_data, mesh_vertices, tuple([int(a) for a in grid_shape_finest]), partSize, actual_num_levels, shift, sparsity_pattern, level_origins, x0

def generate_cuboid_mesh(stl_filename, voxel_size):
    """
    Alternative cuboid mesh generation based on Apolo's method with domain multipliers per level.
    """
    # Domain multipliers for each refinement level
    domain_multiplier = [
        [4, 10, 6, 6, 6, 6],  # -x, x, -y, y, -z, z
        [2, 8, 2, 2, 2, 2],  # -x, x, -y, y, -z, z
        [1, 6, 1, 1, 1, 1],
        [0.5, 3, 0.5, 0.5, 0.5, 0.5],
        # [1, 2, 1, 1, 1, 1],
        # [0.4, 1, 0.4, 0.4, 0.4, 0.4],
        # [0.2, 0.4, 0.2, 0.2, 0.2, 0.2],
    ]

    # domain_multiplier = [
        # [5, 8, 5, 5, 5, 5],
        # [2, 4, 2, 2, 2, 2],
        # [.6, 2, 0.6, 0.6, 0.6, 0.6],
        # [0.4, 0.9, 0.4, 0.4, 0.4, 0.4],
    # ]

    # Load the mesh
    mesh = trimesh.load_mesh(stl_filename, process=False)
    if mesh.is_empty:
        raise ValueError("Loaded mesh is empty or invalid.")

    # Compute original bounds
    min_bound = mesh.vertices.min(axis=0)
    max_bound = mesh.vertices.max(axis=0)
    partSize = max_bound - min_bound

    # Compute translation to put mesh into first octant of the domain
    shift = np.array(
        [
            domain_multiplier[0][0] * partSize[0] - min_bound[0],
            domain_multiplier[0][2] * partSize[1] - min_bound[1],
            domain_multiplier[0][4] * partSize[2] - min_bound[2],
        ],
        dtype=float,
    )

    # Apply translation and save out temp STL
    mesh.apply_translation(shift)
    _ = mesh.vertex_normals
    mesh_vertices = np.asarray(mesh.vertices) / voxel_size
    mesh.export("temp.stl")

    # Generate mesh using make_cuboid_mesh
    level_data = make_cuboid_mesh(
        voxel_size,
        domain_multiplier,
        "temp.stl",
    )
    sparsity_pattern, level_origins = prepare_sparsity_pattern(level_data)
    actual_num_levels = len(level_data)
    grid_shape_finest = tuple([int(i * 2 ** (actual_num_levels - 1)) for i in level_data[-1][0].shape])
    print(f"Requested levels: {len(domain_multiplier)}, Actual levels: {actual_num_levels}")
    print(f"Full shape based on finest voxel size is {grid_shape_finest}")
    os.remove("temp.stl")

    return level_data, mesh_vertices, tuple([int(a) for a in grid_shape_finest]), partSize, actual_num_levels, shift, sparsity_pattern, level_origins

# Boundary Conditions Setup
# =========================
def setup_boundary_conditions(grid, level_data, body_vertices, ulb, nu_lattice, compute_backend=ComputeBackend.NEON):
    """
    Set up boundary conditions for the simulation.
    """
    num_levels = len(level_data)
    coarsest_level = num_levels - 1
    box = grid.bounding_box_indices(shape=grid.level_to_shape(coarsest_level))
    left_indices = grid.boundary_indices_across_levels(level_data, box_side="left", remove_edges=True)
    right_indices = grid.boundary_indices_across_levels(level_data, box_side="right", remove_edges=True)
    top_indices = grid.boundary_indices_across_levels(level_data, box_side="top", remove_edges=False)
    bottom_indices = grid.boundary_indices_across_levels(level_data, box_side="bottom", remove_edges=False)
    front_indices = grid.boundary_indices_across_levels(level_data, box_side="front", remove_edges=False)
    back_indices = grid.boundary_indices_across_levels(level_data, box_side="back", remove_edges=False)

    # Filter front and back indices to remove overlaps with top and bottom at each level
    filtered_front_indices = []
    filtered_back_indices = []
    filtered_top_indices = []
    filtered_bottom_indices = []
    for level in range(num_levels):
        left_set = set(zip(*left_indices[level])) if left_indices[level] else set()
        right_set = set(zip(*right_indices[level])) if right_indices[level] else set()
        top_set = set(zip(*top_indices[level])) if top_indices[level] else set()
        bottom_set = set(zip(*bottom_indices[level])) if bottom_indices[level] else set()
        front_set = set(zip(*front_indices[level])) if front_indices[level] else set()
        back_set = set(zip(*back_indices[level])) if back_indices[level] else set()
        filtered_front_set = front_set - (top_set | bottom_set)
        filtered_back_set = back_set - (top_set | bottom_set)
        filtered_top_set = top_set - (left_set | right_set)
        filtered_bottom_set = bottom_set - (left_set | right_set)
        filtered_front_indices.append(
            [list(coords) for coords in zip(*filtered_front_set)] if filtered_front_set else []
        )
        filtered_back_indices.append(
            [list(coords) for coords in zip(*filtered_back_set)] if filtered_back_set else []
        )
        filtered_top_indices.append(
            [list(coords) for coords in zip(*filtered_top_set)] if filtered_top_set else []
        )
        filtered_bottom_indices.append(
            [list(coords) for coords in zip(*filtered_bottom_set)] if filtered_bottom_set else []
        )

    # Turbulent Flow Profile
    def bc_profile(zero_height_voxels=8):  # Single param for bottom zero-velocity height
        assert compute_backend == ComputeBackend.NEON
        _, _, nz = grid_shape_zip
        dtype = precision_policy.compute_precision.wp_dtype
        H_z = dtype(nz)
        zero = dtype(0.0)
        ulb_wp = dtype(ulb)
        bottom_zero_height = dtype(zero_height_voxels)  # Voxels from bottom to zero out
        _vec3d = wp.vec(velocity_set.d, dtype=dtype)

        @wp.func
        def bc_profile_warp(index: wp.vec3i, timestep: Any):
            z = dtype(index[2])
            # Hard zero for bottom z <= zero_height_voxels, uniform ulb above
            velocity = zero if z <= bottom_zero_height else ulb_wp
            return _vec3d(velocity, zero, zero)

        return bc_profile_warp

    # Initialize boundary conditions
    bc_inlet = HybridBC(
        bc_method="nonequilibrium_regularized",
        prescribed_value=(ulb, 0.0, 0.0),
        #profile = bc_profile(),
        indices=left_indices,
    )

    bc_outlet = DoNothingBC(indices=right_indices)
    # bc_outlet = ExtrapolationOutflowBC(indices=right_indices)

    # bc_top = FullwayBounceBackBC(indices=top_indices)
    # bc_bottom = FullwayBounceBackBC(indices=bottom_indices)
    # bc_front = FullwayBounceBackBC(indices=filtered_front_indices)
    # bc_back = FullwayBounceBackBC(indices=filtered_back_indices)
    
    bc_top = HybridBC(bc_method="nonequilibrium_regularized", prescribed_value=(ulb, 0.0, 0.0), indices=top_indices)
    bc_bottom = HybridBC(bc_method="nonequilibrium_regularized", indices=bottom_indices)
    # bc_bottom = HybridBC(
        # bc_method="bounceback_grads",
        # indices=bottom_indices,
    # )

    bc_front = HybridBC(bc_method="nonequilibrium_regularized", prescribed_value=(ulb, 0.0, 0.0), indices=filtered_front_indices)
    bc_back = HybridBC(bc_method="nonequilibrium_regularized", prescribed_value=(ulb, 0.0, 0.0), indices=filtered_back_indices)

    # bc_body = FullwayBounceBackBC(mesh_vertices=body_vertices, voxelization_method=MeshVoxelizationMethod.AABB_FILL)
    bc_body = HybridBC(
        bc_method="bounceback_regularized",
        mesh_vertices=body_vertices,
        voxelization_method=MeshVoxelizationMethod("AABB_CLOSE", close_voxels=1),        
    )

    return [bc_top, bc_bottom, bc_front, bc_back, bc_inlet, bc_outlet, bc_body] # Body must be last. Outlet must be second to last

# Simulation Initialization
# =========================
def initialize_simulation(grid, boundary_conditions, omega_finest, initializer, collision_type="KBC", mres_perf_opt=xlb.MresPerfOptimizationType.FUSION_AT_FINEST):
    """
    Initialize the multiresolution simulation manager.
    """
    sim = xlb.helper.MultiresSimulationManager(
        omega_finest=omega_finest,
        grid=grid,
        boundary_conditions=boundary_conditions,
        collision_type=collision_type,
        initializer=initializer,
        mres_perf_opt=mres_perf_opt,
    )
    return sim

# Utility Functions
# =================
def print_lift_drag(sim, step, momentum_transfer, ulb, reference_area, voxel_size):
    """
    Calculate and print lift and drag coefficients.
    """
    boundary_force = momentum_transfer(sim.f_0, sim.f_1, sim.bc_mask, sim.missing_mask)
    drag = boundary_force[0]
    lift = boundary_force[2]
    cd = 2.0 * drag / (ulb**2 * reference_area)
    cl = 2.0 * lift / (ulb**2 * reference_area)
    if np.isnan(cd) or np.isnan(cl):
        raise ValueError(f"NaN detected in coefficients at step {step}: Cd={cd}, Cl={cl}")
    drag_values.append([cd, cl])
    # print(f"CD={cd:.3f}, CL={cl:.3f}, Drag Force (lattice units)={drag:.6f}")
    return cd, cl, drag

def plot_drag_lift(drag_values, output_dir, print_interval, script_name, percentile_range=(15, 85), use_log_scale=False):
    """
    Plot CD and CL over time and save the plot to the output directory.
    """
    drag_values_array = np.array(drag_values)
    steps = np.arange(0, len(drag_values) * print_interval, print_interval)
    cd_values = drag_values_array[:, 0]
    cl_values = drag_values_array[:, 1]
    y_min = min(np.percentile(cd_values, percentile_range[0]), np.percentile(cl_values, percentile_range[0]))
    y_max = max(np.percentile(cd_values, percentile_range[1]), np.percentile(cl_values, percentile_range[1]))
    padding = (y_max - y_min) * 0.1
    y_min, y_max = y_min - padding, y_max + padding
    if use_log_scale:
        y_min = max(y_min, 1e-6)
    plt.figure(figsize=(10, 6))
    plt.plot(steps, cd_values, label='Drag Coefficient (Cd)', color='blue')
    plt.plot(steps, cl_values, label='Lift Coefficient (Cl)', color='red')
    plt.xlabel('Simulation Step')
    plt.ylabel('Coefficient')
    plt.title(f'{script_name}: Drag and Lift Coefficients Over Time')
    plt.legend()
    plt.grid(True)
    plt.ylim(y_min, y_max)
    if use_log_scale:
        plt.yscale('log')
    plt.savefig(os.path.join(output_dir, 'drag_lift_plot.png'))
    plt.close()

def compute_voxel_statistics_and_reference_area(sim, bc_mask_exporter, level_data, actual_num_levels, sparsity_pattern, boundary_conditions, voxel_size):
    """
    Compute active/solid voxels, totals, lattice updates, and reference area based on simulation data.
    """
    # Compute macro fields
    sim.macro(sim.f_0, sim.bc_mask, sim.rho, sim.u, streamId=0)
    fields_data = bc_mask_exporter.get_fields_data({"bc_mask": sim.bc_mask})
    bc_mask_data = fields_data["bc_mask_0"]
    level_id_field = bc_mask_exporter.level_id_field

    # Compute solid voxels per level (assuming 255 is the solid marker)
    solid_voxels = []
    for lvl in range(actual_num_levels):
        level_mask = level_id_field == lvl
        solid_voxels.append(np.sum(bc_mask_data[level_mask] == 255))

    # Compute active voxels (total non-zero in sparsity minus solids)
    active_voxels = [np.count_nonzero(mask) for mask in sparsity_pattern]
    active_voxels = [max(0, active_voxels[lvl] - solid_voxels[lvl]) for lvl in range(actual_num_levels)]

    # Totals
    total_voxels = sum(active_voxels)
    total_lattice_updates_per_step = sum(active_voxels[lvl] * (2 ** (actual_num_levels - 1 - lvl)) for lvl in range(actual_num_levels))

    # Compute reference area (projected on YZ plane at finest level)
    finest_level = 0
    mask_finest = level_id_field == finest_level
    bc_mask_finest = bc_mask_data[mask_finest]
    active_indices_finest = np.argwhere(level_data[0][0])
    bc_body_id = boundary_conditions[-1].id  # Assuming last BC is bc_body
    solid_voxels_indices = active_indices_finest[bc_mask_finest == bc_body_id]
    unique_jk = np.unique(solid_voxels_indices[:, 1:3], axis=0)
    reference_area = unique_jk.shape[0]
    reference_area_physical = reference_area * (voxel_size ** 2)

    return {
        "active_voxels": active_voxels,
        "solid_voxels": solid_voxels,
        "total_voxels": total_voxels,
        "total_lattice_updates_per_step": total_lattice_updates_per_step,
        "reference_area": reference_area,
        "reference_area_physical": reference_area_physical
    }


def plot_data(x0, output_dir, delta_x_coarse, sim, IOexporter, prefix='Ahmed25'):
    '''       
        Ahmed Car Model, slant - angle = 25 degree
        Profiles on symmetry plane (y=0) covering entire field
        Origin of coordinate system: 
             x=0: end of the car, y=0: symmetry plane, z=0: ground plane
        
        S.Becker/H. Lienhart/C.Stoots
        Insitute of Fluid Mechanics
        University Erlangen-Nuremberg
        Erlangen, Germany
        Coordaintes in meters need to convert to voxels
        Velocity data in m/s 
    '''
    
    def _load_sim_line(csv_path):
        """
        Read a CSV exported by IOexporter.to_line without pandas.
        Returns (z, Ux).
        """
        # Read with header as column names
        data = np.genfromtxt(
            csv_path,
            delimiter=',',
            names=True,         # use header
            autostrip=True,
            dtype=None,         # let numpy infer dtypes
            encoding='utf-8'    # handle any non-ascii names
        )
        if data.size == 0:
            raise ValueError(f"No data in {csv_path}")

        names = data.dtype.names
        lower = {n: n.lower() for n in names}

        # Find z-like column (fallback: first column)
        z_candidates = [
            n for n in names
            if lower[n] == 'z'
            or lower[n] in ('s', 'distance', 'arc_length', 'arclength')
            or 'z' == lower[n].split('_')[-1]
        ]
        z_name = z_candidates[0] if z_candidates else names[0]

        # Find velocity-x column (fallback: last column)
        vel_candidates = [n for n in names if any(k in lower[n] for k in ('vel', 'u', 'velocity'))]
        # Prefer an x-component if present (common patterns after numpy sanitizes names)
        vel_x_pref = [n for n in vel_candidates if any(k in lower[n] for k in ('x', '_0', '0'))]
        vel_name = vel_x_pref[0] if vel_x_pref else (vel_candidates[0] if vel_candidates else names[-1])

        z = np.asarray(data[z_name], dtype=float)
        ux = np.asarray(data[vel_name], dtype=float)
        return z, ux
    testData = {   
     '-1.162' : { 'x' : [26.995,29.825,29.182,28.488,27.703,26.988,26.456,26.163,26.190,26.523,27.083,28.033,29.131,30.429,31.747,33.036,34.268,35.354,36.312,37.083,37.770,38.484,39.033,39.447,39.839,40.086,40.268,40.380,40.451], 'y' : [0.028,0.048,0.068,0.088,0.108,0.128,0.148,0.168,0.188,0.208,0.228,0.248,0.268,0.288,0.308,0.328,0.348,0.368,0.388,0.408,0.428,0.458,0.488,0.518,0.558,0.598,0.638,0.688,0.7388]},
     '-1.062' : { 'x' : [30.307,28.962,25.812,21.232,15.848,10.812,7.459,6.080,5.845,6.196,7.428,10.456,15.718,22.129,28.090,32.707,35.888,37.891,39.071,39.840,40.261,40.604,40.767,40.820,40.870,40.890,40.907,40.871,40.853], 'y' : [0.028,0.048,0.068,0.088,0.108,0.128,0.148,0.168,0.188,0.208,0.228,0.248,0.268,0.288,0.308,0.328,0.348,0.368,0.388,0.408,0.428,0.458,0.488,0.518,0.558,0.598,0.638,0.688,0.738]},
     '-0.962' : { 'x' : [52.216,51.303,50.196,48.833,47.728,46.790,45.514,44.222,43.379,42.829,42.322,42.056,41.876,41.706,41.584], 'y' : [0.363,0.368,0.378,0.388,0.398,0.408,0.428,0.458,0.488,0.518,0.558,0.598,0.638,0.688,0.738]},
     '-0.862' : { 'x' : [46.589,46.538,46.228,46.033,45.810,45.554,45.056,44.369,43.789,43.275,42.789,42.344,42.148,41.913,41.720], 'y' : [0.363,0.368,0.378,0.388,0.398,0.408,0.428,0.458,0.488,0.518,0.558,0.598,0.638,0.688,0.738]},
     '-0.562' : { 'x' : [43.237,43.262,43.248,43.225,43.183,43.145,43.083,43.030,42.904,42.776,42.685,42.434,42.358,42.197,42.042], 'y' : [0.363,0.368,0.378,0.388,0.398,0.408,0.428,0.458,0.488,0.518,0.558,0.598,0.638,0.688,0.738]},
     '-0.362' : { 'x' : [44.493,44.491,44.443,44.379,44.297,44.215,44.067,43.867,43.577,43.306,43.061,42.689,42.527,42.293,42.105], 'y' : [0.363,0.368,0.378,0.388,0.398,0.408,0.428,0.458,0.488,0.518,0.558,0.598,0.638,0.688,0.738]},
     '-0.212' : { 'x' : [49.202,48.429,47.805,46.697,45.883,44.913,44.195,43.650,43.130,42.677,42.432,42.154,41.961], 'y' : [0.368,0.378,0.388,0.408,0.428,0.458,0.488,0.518,0.558,0.598,0.638,0.688,0.738]},
     '-0.162' : { 'x' : [50.511,49.784,48.894,48.103,47.468,46.322,45.563,44.581,43.933,43.383,42.905,42.505,42.293,42.042,41.863], 'y' : [0.348,0.358,0.368,0.378,0.388,0.408,0.428,0.458,0.488,0.518,0.558,0.598,0.638,0.688,0.738]},
     '-0.062' : { 'x' : [22.891,27.789,32.292,36.568,39.533,41.426,42.371,42.971,43.030,43.081,43.074,43.065,43.039,42.996,42.908,42.665,42.456,42.294,42.105,41.929,41.827,41.660,41.546], 'y' : [22.891,27.789,32.292,36.568,39.533,41.426,42.371,42.971,43.030,43.081,43.074,43.065,43.039,42.996,42.908,42.665,42.456,42.294,42.105,41.929,41.827,41.660,41.546]},
     '-0.112' : { 'x' : [27.615,35.449,41.526,46.068,46.277,46.038,45.774,45.505,45.237,44.701,44.326,43.765,43.284,42.890,42.529,42.247,42.082,41.880,41.732], 'y' : [0.318,0.323,0.328,0.338,0.348,0.358,0.368,0.378,0.388,0.408,0.428,0.458,0.488,0.518,0.558,0.598,0.638,0.688,0.738]},
     '-0.062' : { 'x' : [22.891,27.789,32.292,36.568,39.533,41.426,42.371,42.971,43.030,43.081,43.074,43.065,43.039,42.996,42.908,42.665,42.456,42.294,42.105,41.929,41.827,41.660,41.546], 'y' : [0.298,0.303,0.308,0.313,0.318,0.323,0.328,0.338,0.348,0.358,0.368,0.378,0.388,0.408,0.428,0.458,0.488,0.518,0.558,0.598,0.638,0.688,0.738]},
     '-0.012' : { 'x' : [23.304,26.317,29.429,32.341,34.923,37.106,38.673,39.841,40.447,40.780,40.973,41.085,41.193,41.282,41.359,41.442,41.522,41.699,41.737,41.749,41.724,41.714,41.642,41.574,41.518,41.431,41.366], 'y' : [0.278,0.283,0.288,0.293,0.298,0.303,0.308,0.313,0.318,0.323,0.328,0.338,0.348,0.358,0.368,0.378,0.388,0.408,0.428,0.458,0.488,0.518,0.558,0.598,0.638,0.688,0.738]},    
     '0.038' : { 'x' : [42.752,37.392,15.320,-4.501,-8.079,-8.892,-8.420,-7.027,-5.143,-2.903,-0.936,0.927,2.200,3.099,3.622,4.026,4.280,4.520,5.620,8.938,13.913,17.872,21.148,24.814,29.075,33.188,36.424,38.490,39.388,39.675,39.794,39.911,40.007,40.219,40.425,40.643,40.757,40.896,40.994,41.058,41.124,41.127,41.143,41.106,41.080], 'y' : [0.028,0.038,0.048,0.058,0.068,0.078,0.088,0.098,0.108,0.118,0.128,0.138,0.148,0.158,0.168,0.178,0.188,0.198,0.208,0.218,0.228,0.238,0.248,0.258,0.268,0.278,0.288,0.298,0.308,0.318,0.328,0.338,0.348,0.368,0.388,0.408,0.428,0.458,0.488,0.518,0.558,0.598,0.638,0.688,0.738]},
     '0.088' : { 'x' : [41.859,35.830,22.660,7.745,-5.808,-12.650,-14.748,-13.756,-10.659,-6.484,-2.121,1.303,3.672,5.441,7.066,9.157,11.613,14.620,17.662,20.639,23.565,26.437,29.484,32.441,35.024,36.938,37.938,38.377,38.595,38.728,38.856,38.976,39.133,39.438,39.749,39.975,40.129,40.344,40.499,40.649,40.783,40.853,40.927,40.945,40.960], 'y' : [0.028,0.038,0.048,0.058,0.068,0.078,0.088,0.098,0.108,0.118,0.128,0.138,0.148,0.158,0.168,0.178,0.188,0.198,0.208,0.218,0.228,0.238,0.248,0.258,0.268,0.278,0.288,0.298,0.308,0.318,0.328,0.338,0.348,0.368,0.388,0.408,0.428,0.458,0.488,0.518,0.558,0.598,0.638,0.688,0.738]},
     '0.138' : { 'x' : [36.223,32.501,24.752,14.281,2.799,-6.218,-10.908,-11.892,-9.708,-5.258,-0.140,4.331,7.882,10.995,13.961,16.699,19.477,22.063,24.651,27.081,29.524,31.950,34.043,35.594,36.506,37.053,37.386,37.614,37.832,38.032,38.214,38.397,38.575,38.940,39.298,39.533,39.749,40.028,40.206,40.404,40.580,40.691,40.803,40.858,40.921], 'y' : [0.028,0.038,0.048,0.058,0.068,0.078,0.088,0.098,0.108,0.118,0.128,0.138,0.148,0.158,0.168,0.178,0.188,0.198,0.208,0.218,0.228,0.238,0.248,0.258,0.268,0.278,0.288,0.298,0.308,0.318,0.328,0.338,0.348,0.368,0.388,0.408,0.428,0.458,0.488,0.518,0.558,0.598,0.638,0.688,0.738]},
     '0.188' : { 'x' : [29.417,27.755,23.967,18.261,11.662,5.405,0.676,-0.652,0.937,4.261,7.958,11.427,14.366,17.138,19.735,22.151,24.577,26.883,29.165,31.111,32.781,34.072,34.893,35.524,35.974,36.329,36.604,36.872,37.138,37.402,37.673,37.900,38.112,38.518,38.829,39.088,39.326,39.639,39.871,40.096,40.275,40.423,40.523,40.603,40.687], 'y' : [0.028,0.038,0.048,0.058,0.068,0.078,0.088,0.098,0.108,0.118,0.128,0.138,0.148,0.158,0.168,0.178,0.188,0.198,0.208,0.218,0.228,0.238,0.248,0.258,0.268,0.278,0.288,0.298,0.308,0.318,0.328,0.338,0.348,0.368,0.388,0.408,0.428,0.458,0.488,0.518,0.558,0.598,0.638,0.688,0.738]},
     '0.238' : { 'x' : [24.405,24.168,22.782,20.196,16.970,13.937,12.137,11.757,12.851,14.649,16.780,18.995,21.070,23.335,25.280,27.468,29.262,30.832,32.133,33.102,33.856,34.473,34.922,35.340,35.698,36.039,36.336,36.629,36.906,37.193,37.454,37.691,37.929,38.329,38.611,38.875,39.126,39.414,39.677,39.917,40.097,40.259,40.380,40.478,40.568], 'y' : [0.028,0.038,0.048,0.058,0.068,0.078,0.088,0.098,0.108,0.118,0.128,0.138,0.148,0.158,0.168,0.178,0.188,0.198,0.208,0.218,0.228,0.238,0.248,0.258,0.268,0.278,0.288,0.298,0.308,0.318,0.328,0.338,0.348,0.368,0.388,0.408,0.428,0.458,0.488,0.518,0.558,0.598,0.638,0.688,0.738]},
     '0.288' : { 'x' : [21.489,22.225,22.127,21.456,20.404,19.743,19.541,19.909,21.002,22.381,24.018,25.670,27.421,28.998,30.371,31.523,32.406,33.111,33.670,34.155,34.532,34.893,35.240,35.567,35.875,36.158,36.437,36.708,36.974,37.230,37.473,37.709,37.932,38.266,38.515,38.773,39.008,39.270,39.562,39.782,39.962,40.148,40.266,40.369,40.475], 'y' : [0.028,0.038,0.048,0.058,0.068,0.078,0.088,0.098,0.108,0.118,0.128,0.138,0.148,0.158,0.168,0.178,0.188,0.198,0.208,0.218,0.228,0.238,0.248,0.258,0.268,0.278,0.288,0.298,0.308,0.318,0.328,0.338,0.348,0.368,0.388,0.408,0.428,0.458,0.488,0.518,0.558,0.598,0.638,0.688,0.738]}
    
     }
                  
    xData =[-1.162,-1.062,-0.962,-0.862,-0.562,-0.362,-0.212,-0.162,-0.112,-0.062,-0.012,0.038,0.088,0.138,0.188,0.238,0.288]
    
    
    for i in xData:
        #Extract y dimension
        refY = np.array(testData[str(i)]['y'])
        #u is already converted to model units (m/s) no need to convert reference velocity
        refX = np.array(testData[str(i)]['x'])
    
        #From reference x0 (rear of body) find x1 for plot            
        x1 = x0 + i
        
        print(f' x1 is {x1}')
        sim.macro(sim.f_0, sim.bc_mask, sim.rho, sim.u, streamId=0)
        filename = os.path.join(output_dir, f"{prefix}_{str(i)}")
        wp.synchronize()                 
        IOexporter.to_line(
            filename,
            {"velocity": sim.u},
            start_point=(x1, 0, 0),
            end_point=(x1, 0, 0.8),            
            resolution=200,   
            component=0,
            radius=delta_x_coarse #needed with model units
        )
        # read the CSV written by the exporter
        csv_path = filename + "_velocity_0.csv"  # adjust if your exporter uses another extension
        print(f"CSV path is {csv_path}")
        
        try:
            sim_z, sim_ux = _load_sim_line(csv_path)
        except Exception as e:
            print(f"Failed to read {csv_path}: {e}")
            continue

        # plot reference vs simulation
        plt.figure(figsize=(4.5, 6))
        plt.plot(refX, refY, 'o', mfc='none', label='Ahmed (exp)')
        plt.plot(sim_ux, sim_z, '-', lw=2, label='Simulation')
        plt.xlim(np.min(refX)*.9, np.max(refX)*1.1)
        plt.ylim(np.min(refY), np.max(refY))
        plt.xlabel('Ux [m/s]')
        plt.ylabel('z [m]')
        plt.title(f'Velocity Plot at {i:+.3f}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename + ".png", dpi=150)
        plt.close()
        
        
  
# Main Script
# ===========
# Initialize XLB
wp.clear_kernel_cache()
xlb.init(
    velocity_set=velocity_set,
    default_backend=compute_backend,
    default_precision_policy=precision_policy,
)

# Generate mesh
if mesher_type == "makemesh":
    level_data, body_vertices, grid_shape_zip, partSize, actual_num_levels, shift, sparsity_pattern, level_origins, x0 = generate_makemesh_mesh(
        stl_filename, voxel_size
    )
elif mesher_type == "cuboid":
    level_data, body_vertices, grid_shape_zip, partSize, actual_num_levels, shift, sparsity_pattern, level_origins = generate_cuboid_mesh(
        stl_filename, voxel_size
    )
else:
    raise ValueError(f"Invalid mesher_type: {mesher_type}. Must be 'makemesh' or 'cuboid'.")

# Characteristic length
L = partSize[0]
L = float(L)  # Cast to built-in float to avoid NumPy type propagation issues with Warp

# Compute Re
Re = u_physical * L / kinematic_viscosity

# Define a unit convertor
unit_convertor = UnitConvertor(
velocity_lbm_unit=ulb,
velocity_physical_unit=u_physical,
voxel_size_physical_unit=voxel_size,
)

# Calculate lattice parameters
delta_x_coarse = voxel_size * 2 ** (actual_num_levels - 1)
delta_t = voxel_size * ulb / u_physical
nu_lattice = kinematic_viscosity * delta_t / (voxel_size ** 2)
omega_finest = 1.0 / (3.0 * nu_lattice + 0.5)

# Create output directory
current_dir = os.path.join(os.path.dirname(__file__))
output_dir = os.path.join(current_dir, script_name)
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

# Define exporter objects
field_name_cardinality_dict = {"velocity": 3, "density": 1}
h5exporter = MultiresIO(
    field_name_cardinality_dict,
    level_data,
    unit_convertor=unit_convertor,
    offset=-shift,
)
bc_mask_exporter = MultiresIO(
    {"bc_mask": 1},
    level_data,
    unit_convertor=unit_convertor,
    offset=-shift,
)

# Create grid
grid = multires_grid_factory(
    grid_shape_zip,
    velocity_set=velocity_set,
    sparsity_pattern_list=sparsity_pattern,
    sparsity_pattern_origins=[neon.Index_3d(*box_origin) for box_origin in level_origins],
)

# Calculate num_steps
coarsest_level = grid.count_levels - 1
grid_shape_x_coarsest = grid.level_to_shape(coarsest_level)[0]
num_steps = int(flow_passes * (grid_shape_x_coarsest / ulb))

# Calculate print and file output intervals
print_interval = max(1, int(num_steps * (print_interval_percentage / 100.0)))
crossover_step = int(num_steps * (file_output_crossover_percentage / 100.0))
file_output_interval_pre_crossover = max(1, int(crossover_step / num_file_outputs_pre_crossover)) if num_file_outputs_pre_crossover > 0 else num_steps + 1
file_output_interval_post_crossover = max(1, int((num_steps - crossover_step) / num_file_outputs_post_crossover)) if num_file_outputs_post_crossover > 0 else num_steps + 1

# Setup boundary conditions
boundary_conditions = setup_boundary_conditions(grid, level_data, body_vertices, ulb, nu_lattice, compute_backend)

# Create initializer
initializer = CustomMultiresInitializer(
    bc_id=boundary_conditions[-2].id,  # bc_outlet
    constant_velocity_vector=(ulb, 0.0, 0.0),
    velocity_set=velocity_set,
    precision_policy=precision_policy,
    compute_backend=compute_backend,
)

# Initialize simulation
sim = initialize_simulation(grid, boundary_conditions, omega_finest, initializer)

# Compute voxel statistics and reference area
stats = compute_voxel_statistics_and_reference_area(sim, bc_mask_exporter, level_data, actual_num_levels, sparsity_pattern, boundary_conditions, voxel_size)
active_voxels = stats["active_voxels"]
solid_voxels = stats["solid_voxels"]
total_voxels = stats["total_voxels"]
total_lattice_updates_per_step = stats["total_lattice_updates_per_step"]
reference_area = stats["reference_area"]
reference_area_physical = stats["reference_area_physical"]

# Save initial bc_mask
filename = os.path.join(output_dir, f"{script_name}_initial_bc_mask")
try:
    bc_mask_exporter.to_hdf5(filename, {"bc_mask": sim.bc_mask}, compression="gzip", compression_opts=0)
    xmf_filename = f"{filename}.xmf"
    hdf5_basename = f"{script_name}_initial_bc_mask.h5"
except Exception as e:
    print(f"Error during initial bc_mask output: {e}")
wp.synchronize()

# Setup momentum transfer
momentum_transfer = MultiresMomentumTransfer(
    boundary_conditions[-1],
    mres_perf_opt=xlb.MresPerfOptimizationType.FUSION_AT_FINEST,
    compute_backend=compute_backend,
)

# Print simulation info
print("\n" + "=" * 50 + "\n")
print(f"Simulation Configuration for Re = {Re}:")
# print(f"Grid shape at finest level: {grid_shape_zip}")
# print(f"Grid shape at coarsest level: {grid.level_to_shape(coarsest_level)}")
print(f"Number of flow passes: {flow_passes}")
print(f"Calculated iterations: {num_steps:,}")
# print(f"Output directory: {output_dir}")
# print(f"Print interval: {print_interval} steps (every {print_interval_percentage}% of iterations)")
# print(f"File output interval pre-crossover (0-{file_output_crossover_percentage}%): {file_output_interval_pre_crossover} steps")
# print(f"File output interval post-crossover ({file_output_crossover_percentage}-100%): {file_output_interval_post_crossover} steps")
print(f"Finest voxel size: {voxel_size} meters")
print(f"Coarsest voxel size: {delta_x_coarse} meters")
print(f"Total voxels: {sum(np.count_nonzero(mask) for mask in sparsity_pattern):,}")
print(f"Total active voxels: {total_voxels:,}")
print(f"Active voxels per level: {active_voxels}")
print(f"Solid voxels per level: {solid_voxels}")
print(f"Total lattice updates per global step: {total_lattice_updates_per_step:,}")
print(f"Actual number of refinement levels: {actual_num_levels}")
print(f"Physical inlet velocity: {u_physical:.4f} m/s")
print(f"Lattice velocity (ulb): {ulb}")
print(f"Characteristic length: {L: .4f} meters")
# print(f"Kinematic viscosity: {kinematic_viscosity} m^2/s")
print(f"Computed reference area (bc_mask): {reference_area} lattice units")
print(f"Physical reference area (bc_mask): {reference_area_physical:.6f} m^2")
print(f"Reynolds number: {Re:,.2f}")
# print(f"Lattice viscosity: {nu_lattice:.5f}")
print(f"Relaxation parameter (omega): {omega_finest}")
time_remaining =(total_lattice_updates_per_step * num_steps) / (100 * 1e6)
hours, rem = divmod(time_remaining, 3600)
minutes, seconds = divmod(rem, 60)
time_remaining_str = f"{int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s"
print(f"Approx Runtime (assuming 100mlups): {time_remaining_str} \n")
print("\n" + "=" * 50 + "\n")

# Active Voxels Distribution Table with Computation Percentage
print("Active Voxel Distribution per Level:")
headers = ["Level", "Voxel Size (m)", "Active Voxels (M)", "Percentage (%)", "Computation (%)"]
table_data = []
total_active_voxels = stats["total_voxels"]
# Calculate computational work
comp_work = []
for lvl in range(actual_num_levels):
    active_count = stats["active_voxels"][lvl]
    # Inner iterations: finest level (lvl=0) has 2^(num_levels-1), coarsest (lvl=num_levels-1) has 2^0=1
    inner_iterations = 2 ** (actual_num_levels - 1 - lvl)
    work = active_count * inner_iterations
    comp_work.append(work)
total_work = sum(comp_work) if comp_work else 0
for lvl, active_count in enumerate(stats["active_voxels"]):
    voxel_size_level = voxel_size * (2 ** lvl)  # Voxel size doubles each level
    active_voxels_millions = active_count / 1e6
    percentage = (active_count / total_active_voxels) * 100 if total_active_voxels > 0 else 0
    comp_percentage = (comp_work[lvl] / total_work) * 100 if total_work > 0 else 0
    table_data.append([lvl, f"{voxel_size_level:.6f}", f"{active_voxels_millions:.2f}", f"{percentage:.2f}", f"{comp_percentage:.2f}"])
print(tabulate(table_data, headers=headers, tablefmt="grid"))
print()

# -------------------------- Simulation Loop --------------------------
wp.synchronize()
start_time = time.time()
compute_time = 0.0
steps_since_last_print = 0
drag_values = []

for step in range(num_steps):
    step_start = time.time()
    sim.step()
    wp.synchronize()
    compute_time += time.time() - step_start
    steps_since_last_print += 1
    if step % print_interval == 0 or step == num_steps - 1:
        sim.macro(sim.f_0, sim.bc_mask, sim.rho, sim.u, streamId=0)
        wp.synchronize()
        cd, cl, drag = print_lift_drag(sim, step, momentum_transfer, ulb, reference_area, voxel_size)
        filename = os.path.join(output_dir, f"{script_name}_{step:04d}")
        h5exporter.to_slice_image(
            filename,
            {"velocity": sim.u},
            plane_point=(1, 0, 0),
            plane_normal=(0, 1, 0),
            grid_res=2500,
            bounds=(0, 1, 0, 1),
            show_axes=False,
            show_colorbar=False,
            slice_thickness=delta_x_coarse, #needed when using model units
            normalize = u_physical*1.5, #eventually we could have the 1.5 read from json as we did before
        )
        end_time = time.time()
        elapsed = end_time - start_time
        total_lattice_updates = total_lattice_updates_per_step * steps_since_last_print
        MLUPS = total_lattice_updates / compute_time / 1e6 if compute_time > 0 else 0.0
        current_flow_passes = step * ulb / grid_shape_x_coarsest
        remaining_steps = num_steps - step - 1
        time_remaining = 0.0 if MLUPS == 0 else (total_lattice_updates_per_step * remaining_steps) / (MLUPS * 1e6)
        hours, rem = divmod(time_remaining, 3600)
        minutes, seconds = divmod(rem, 60)
        time_remaining_str = f"{int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s"
        percent_complete = (step + 1) / num_steps * 100
        print(f"Completed step {step}/{num_steps} ({percent_complete:.2f}% complete)")
        print(f"  Flow Passes: {current_flow_passes:.2f}")
        print(f"  Time elapsed: {elapsed:.1f}s, Compute time: {compute_time:.1f}s, ETA: {time_remaining_str}")
        print(f"  MLUPS: {MLUPS:.1f}")
        print(f"  Cd={cd:.3f}, Cl={cl:.3f}, Drag Force (lattice units)={drag:.3f}")
        start_time = time.time()
        compute_time = 0.0
        steps_since_last_print = 0
    file_output_interval = file_output_interval_pre_crossover if step < crossover_step else file_output_interval_post_crossover
    if step % file_output_interval == 0 or step == num_steps - 1:
        sim.macro(sim.f_0, sim.bc_mask, sim.rho, sim.u, streamId=0)
        filename = os.path.join(output_dir, f"{script_name}_{step:04d}")
        try:
            h5exporter.to_hdf5(filename, {"velocity": sim.u, "density": sim.rho}, compression="gzip", compression_opts=0)
            xmf_filename = f"{filename}.xmf"
            hdf5_basename = f"{script_name}_{step:04d}.h5"
        except Exception as e:
            print(f"Error during file output at step {step}: {e}")
        wp.synchronize()
    if step == num_steps - 1:
        plot_data(x0, output_dir, delta_x_coarse, sim, h5exporter, prefix='Ahmed25')

# Save drag and lift data to CSV
if len(drag_values) > 0:
    with open(os.path.join(output_dir, "drag_lift.csv"), 'w') as fd:
        fd.write("Step,Cd,Cl\n")
        for i, (cd, cl) in enumerate(drag_values):
            fd.write(f"{i * print_interval},{cd},{cl}\n")
    plot_drag_lift(drag_values, output_dir, print_interval, script_name)

# Calculate and print average Cd and Cl for the last 50%
drag_values_array = np.array(drag_values)
if len(drag_values) > 0:
    start_index = len(drag_values) // 2
    last_half = drag_values_array[start_index:, :]
    avg_cd = np.mean(last_half[:, 0])
    avg_cl = np.mean(last_half[:, 1])
    print(f"Average Drag Coefficient (Cd) for last 50%: {avg_cd:.6f}")
    print(f"Average Lift Coefficient (Cl) for last 50%: {avg_cl:.6f}")
    print(f"Experimental Drag Coefficient (Cd): {0.285}")  
    print(f"Error Drag Coefficient (Cd): {((avg_cd-0.285)/0.285)*100:.2f}%")  
    
else:
    print("No drag or lift data collected.")