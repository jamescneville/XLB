import neon
import warp as wp
import numpy as np
import time
import os
import re
import matplotlib.pyplot as plt
import trimesh
import shutil

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
from xlb.utils import UnitConvertor
from xlb.utils.makemesh import generate_mesh
from xlb.operator.force import MultiresMomentumTransfer
from xlb.helper.initializers import CustomMultiresInitializer
from xlb import MresPerfOptimizationType



wp.clear_kernel_cache()
wp.config.quiet = True

# User Configuration
# =================
# Physical and simulation parameters
voxel_size = 0.003 # Finest voxel size in meters
ulb = 0.08         # Lattice velocity
u_physical = 38.0  # Physical inlet velocity in m/s (user input)
flow_passes = 4    # Domain flow passes
kinematic_viscosity = 1.508e-5  # Kinematic viscosity of air in m^2/s 1.508e-5

trim = True
trim_voxels = 3

# STL filename
stl_filename = "examples/stl/sae.stl"
script_name = "SAE_3mm_wm2"
# I/O settings
print_interval_percentage = 1   # Print every 1% of iterations
file_output_crossover_percentage = 90  # Crossover at 50% of iterations
num_file_outputs_pre_crossover = 1    # Outputs before crossover
num_file_outputs_post_crossover = 2  # Outputs after crossover

# Other setup parameters
compute_backend = ComputeBackend.NEON
precision_policy = PrecisionPolicy.FP32FP32
velocity_set = xlb.velocity_set.D3Q27(precision_policy=precision_policy, compute_backend=compute_backend)

# Choose mesher type
mesher_type = "makemesh"  # Options: "makemesh" or "cuboid"

# Mesh Generation Functions
# =========================
def generate_makemesh_mesh(stl_filename, voxel_size, trim, trim_voxels, ground_refinement_level=-1, ground_voxel_height=6):
    """
    Generate a makemesh mesh based on the provided voxel size in meters, domain multipliers, and padding values.
    """
    # Number of requested refinement levels
    num_levels = 5

    # Domain multipliers for the full domain
    domain_multiplier = {
        "-x": 2,
        "x": 3.25,
        "-y": 1.35,
        "y": 1.35,
        "-z": 0.0,
        "z": 3.8,
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
    

    # Load the mesh
    mesh = trimesh.load_mesh(stl_filename, process=False)
    if mesh.is_empty:
        raise ValueError("Loaded mesh is empty or invalid.")

    # Compute original bounds
    min_bound = mesh.vertices.min(axis=0)
    max_bound = mesh.vertices.max(axis=0)
    partSize = max_bound - min_bound
    x0 = [max_bound[0]-.486, min_bound[1]+(0.5*partSize[1]), min_bound[2]] #Center of wheelbase for Drivaer
    

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
    mesh.export("temp.stl")
     # Generate mesh using make_cuboid_mesh
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
    if trim == True:
        zShift = trim_voxels
        plane_origin = np.array([0, 0, mesh.bounds[0][2]+(zShift* voxel_size)])
        plane_normal = np.array([0, 0, 1])  # Upward pointing normal
        # Slice the mesh using the defined plane.
        # With cap=True, the open slice is automatically closed off.
        mesh_above = mesh.slice_plane(plane_origin=plane_origin,
                    plane_normal=plane_normal,
                    cap=True)
        mesh_above.export('temp.stl')
        body_stl =  'temp.stl'
        mesh = trimesh.load_mesh(body_stl, process=False)
        mesh_vertices = np.asarray(mesh.vertices) / voxel_size
    else:
        mesh_vertices = np.asarray(mesh.vertices) / voxel_size
    

    actual_num_levels = len(level_data)
    grid_shape_finest = tuple([int(i * 2 ** (actual_num_levels - 1)) for i in level_data[-1][0].shape])
    print(f"Requested levels: {num_levels}, Actual levels: {actual_num_levels}")
    print(f"Full shape based on finest voxel size is {grid_shape_finest}")
    #os.remove("temp.stl")

    return level_data, mesh_vertices, tuple([int(a) for a in grid_shape_finest]), partSize, actual_num_levels, shift, sparsity_pattern, level_origins, x0

def generate_cuboid_mesh(stl_filename, voxel_size, trim, trim_voxels):
    """
    Alternative cuboid mesh generation based on Apolo's method with domain multipliers per level.
    """
    # Domain multipliers for each refinement level
    domain_multiplier = [
        [3.0,  3.5,  1.5,  1.5,  0.0, 3.7],  # -x, x, -y, y, -z, z
        [1.8,  1.6, 1.2,  1.2 , 0.0, 2.0],  # -x, x, -y, y, -z, z
        [1.4,  1.25, 1.0, 1.0, 0.0, 1.6],  # -x, x, -y, y, -z, z
        [0.8,  1.0,  0.6, 0.6, 0.0, 1.2],
        [0.4, 0.4,  0.25, 0.25, 0.0, 0.25],  # -x, x, -y, y, -z, z
        #[0.5,  0.65, 0.6,  0.60, 0.0, 0.6],
        #[0.25, 0.25, 0.2, 0.2, 0.0, 0.2],
        
    ]


    # Load the mesh
    mesh = trimesh.load_mesh(stl_filename, process=False)
    if mesh.is_empty:
        raise ValueError("Loaded mesh is empty or invalid.")

    # Compute original bounds
    min_bound = mesh.vertices.min(axis=0)
    max_bound = mesh.vertices.max(axis=0)
    partSize = max_bound - min_bound
    x0 = [max_bound[0]/2, min_bound[1]+(0.5*partSize[1]), min_bound[2]] #Center of wheelbase for Drivaer
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
    mesh.export("temp.stl")
     # Generate mesh using make_cuboid_mesh
    level_data = make_cuboid_mesh(
        voxel_size,
        domain_multiplier,
        "temp.stl",
    )
    sparsity_pattern, level_origins = prepare_sparsity_pattern(level_data)
    if trim == True:
        zShift = trim_voxels
        plane_origin = np.array([0, 0, mesh.bounds[0][2]+(zShift* voxel_size)])
        plane_normal = np.array([0, 0, 1])  # Upward pointing normal
        # Slice the mesh using the defined plane.
        # With cap=True, the open slice is automatically closed off.
        mesh_above = mesh.slice_plane(plane_origin=plane_origin,
                    plane_normal=plane_normal,
                    cap=True)
        mesh_above.export('temp.stl')
        body_stl =  'temp.stl'
        mesh = trimesh.load_mesh(body_stl, process=False)
        mesh_vertices = np.asarray(mesh.vertices) / voxel_size
    else:
        mesh_vertices = np.asarray(mesh.vertices) / voxel_size
        
    

   
    actual_num_levels = len(level_data)
    grid_shape_finest = tuple([int(i * 2 ** (actual_num_levels - 1)) for i in level_data[-1][0].shape])
    print(f"Requested levels: {len(domain_multiplier)}, Actual levels: {actual_num_levels}")
    print(f"Full shape based on finest voxel size is {grid_shape_finest}")
    #os.remove("temp.stl")

    return level_data, mesh_vertices, tuple([int(a) for a in grid_shape_finest]), partSize, actual_num_levels, shift, sparsity_pattern, level_origins, x0

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
        filtered_front_set = front_set - (top_set | bottom_set )
        filtered_back_set = back_set - (top_set | bottom_set )
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


    #bc_inlet = RegularizedBC("velocity", prescribed_value=(ulb, 0.0, 0.0), indices=left_indices, )
    bc_inlet = HybridBC(bc_method="nonequilibrium_regularized",prescribed_value=(ulb, 0.0, 0.0),indices=left_indices)

    bc_outlet = DoNothingBC(indices=right_indices)


    bc_top = HybridBC(bc_method="nonequilibrium_regularized",prescribed_value=(ulb, 0.0, 0.0),indices=top_indices)
    bc_bottom = HybridBC(bc_method="nonequilibrium_regularized",prescribed_value=(ulb, 0.0, 0.0),indices=bottom_indices)
    bc_front = HybridBC(bc_method="nonequilibrium_regularized",prescribed_value=(ulb, 0.0, 0.0),indices=filtered_front_indices)
    bc_back = HybridBC(bc_method="nonequilibrium_regularized",prescribed_value=(ulb, 0.0, 0.0),indices=filtered_back_indices)

    bc_body = HybridBC(
        bc_method="nonequilibrium_regularized",
        mesh_vertices=body_vertices,
        voxelization_method=MeshVoxelizationMethod("AABB_CLOSE", close_voxels=1),
        use_mesh_distance=True,
        use_wall_model=True,
        kinematic_viscosity=nu_lattice,
    )

    return [bc_top, bc_bottom, bc_front, bc_back, bc_inlet, bc_outlet, bc_body] # Body must be last. Outlet must be second to last



# Simulation Initialization
# =========================
def initialize_simulation(grid, boundary_conditions, omega, initializer, collision_type="KBC", mres_perf_opt=xlb.MresPerfOptimizationType.FUSION_AT_FINEST):
    """
    Initialize the multiresolution simulation manager.
    """
    sim = xlb.helper.MultiresSimulationManager(
        omega_finest=omega,
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


def plot_data(x0, output_dir, delta_x_coarse, sim, IOexporter, prefix='SAE'):
    '''       
         SAE Notchback Car Model
            https://repository.lboro.ac.uk/articles/dataset/SAE_reference_model_20_degree_notchback_validation_dataset_reference_SAE_paper_2014-01-0590_/9230000?file=24427865
             Profiles on symmetry plane (y=0) covering entire field
             Origin of coordinate system: 
                  x=0: center of the car, y=0: symmetry plane, z=0: ground plane
            
              Coordaintes in millimeters 
              Velocity data in m/s 
            
            
        In Test X+ goes to front X- Rear
        We will invert as we have X+ to rear
        Z is inverted -40mm bottom of vehicle, -280mm is top
        We will invert values 
        
        Key is Xlocation
        Value X is vx
        Value Y is z
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
        '423.82464' : { 'x' : [0,-43.88937,-40.61368,-32.92829,-38.54242,-29.32868,-33.79841,-39.12689,-36.48399,-37.07242,-38.76211,-36.60545,-36.89559,-36.29334,-37.35702,-36.20492,-37.97495,-38.97984,-39.8124,-39.31747,-40.33504,-38.89908,-39.71669,-40.76311,-41.19835,-40.91254,-39.88253,-38.8862,-38.91076,-38.1293,-37.93947,-39.77282,-40.20782,-39.12199,-38.7486,-37.57369,-38.86437,-38.06003,-38.06692,-38.97876,-39.91949,-39.97973,-39.37485,-37.94375,-38.60412,-38.83829,-39.05637,-38.79934,-38.49119,-38.12508,-38.24782,-39.36546,-38.72419,-38.51912,-38.68303,-37.99958,-36.33649,-36.16541,-36.50895,-37.43194,-36.99399,-36.19157,-36.60335,-37.80757,-37.87192,-37.68175,-37.35025,-36.31354,-36.85625,-35.83224,-37.16736,-35.57886,-35.48411,-34.61574,-33.06036,-34.64411,-35.90629,-36.51319,-34.72717,-33.99935,-34.20146,-32.09901,-33.45765,-32.17021,-32.93775,-34.06752,-33.99866,-33.62223,-33.37662,-30.79467,-29.53656,-28.56509,-28.4351,-27.85609,-26.79307,-27.85012,-27.11479,-25.93097,-25.39359,-24.52418,-25.38498,-23.2986,-23.2055,-22.19554,-21.68066,-20.65243,-19.44393,-17.75917,-16.88123,-18.04087,-14.64139,-14.67718,-15.12614,-14.36887,-10.8442,-9.83381,-9.64478,-7.15688,-8.29617,-7.90275,-7.09349,-10.6122,-15.73408,-19.20149,-18.8485,-14.20016,-8.82543,-7.36782,0,0,0], 'y' : [-509.04055,-505.43365,-501.82675,-498.21986,-494.61296,-491.00607,-487.39917,-483.79228,-480.18538,-476.57848,-472.97159,-469.36469,-465.75779,-462.1509,-458.544,-454.93711,-451.33021,-447.72332,-444.11642,-440.50952,-436.90263,-433.29573,-429.68884,-426.08194,-422.47504,-418.86815,-415.26125,-411.65436,-408.04746,-404.44056,-400.83367,-397.22677,-393.61988,-390.01298,-386.40608,-382.79919,-379.19229,-375.5854,-371.9785,-368.3716,-364.76471,-361.15781,-357.55092,-353.94402,-350.33713,-346.73023,-343.12333,-339.51644,-335.90954,-332.30265,-328.69575,-325.08885,-321.48196,-317.87506,-314.26817,-310.66127,-307.05437,-303.44748,-299.84058,-296.23369,-292.62679,-289.0199,-285.413,-281.8061,-278.19921,-274.59231,-270.98541,-267.37852,-263.77162,-260.16473,-256.55783,-252.95094,-249.34404,-245.73714,-242.13025,-238.52335,-234.91646,-231.30956,-227.70266,-224.09577,-220.48887,-216.88198,-213.27508,-209.66818,-206.06129,-202.45439,-198.8475,-195.2406,-191.6337,-188.02681,-184.41991,-180.81302,-177.20612,-173.59922,-169.99233,-166.38543,-162.77854,-159.17164,-155.56475,-151.95785,-148.35095,-144.74406,-141.13716,-137.53027,-133.92337,-130.31647,-126.70958,-123.10268,-119.49579,-115.88889,-112.28199,-108.6751,-105.0682,-101.46131,-97.85441,-94.24752,-90.64062,-87.03372,-83.42683,-79.81993,-76.21304,-72.60614,-68.99924,-65.39235,-61.78545,-58.17856,-54.57166,-50.96476,-47.35787,-43.75097,-40.14408]},
        '351.68672' : { 'x' : [0,-35.82332,-36.71613,-36.93024,-39.14271,-39.66171,-39.34607,-39.45934,-39.75603,-39.66507,-39.83869,-39.8442,-40.03991,-39.70252,-40.2212,-40.16842,-40.03166,-40.38552,-40.57657,-40.4124,-39.91504,-39.6967,-39.69385,-39.66115,-39.88188,-39.67558,-39.79702,-39.27636,-39.38751,-39.48462,-39.42266,-39.29125,-39.41234,-39.37153,-39.11465,-39.24489,-39.12078,-39.26703,-39.23408,-38.9926,-38.86375,-38.59503,-38.73572,-38.50098,-38.16545,-38.1546,-38.43249,-38.53798,-38.57212,-38.35255,-38.32625,-38.35024,-38.09314,-38.02264,-37.74175,-37.60618,-37.33062,-37.51258,-37.06576,-37.09844,-37.19652,-37.38486,-37.23984,-37.2065,-37.21374,-37.03967,-36.75381,-36.69456,-36.23351,-35.90798,-35.58163,-35.5893,-35.46375,-35.3884,-35.25679,-35.3233,-35.21813,-34.99791,-34.45804,-33.97931,-33.57084,-32.86375,-32.16122,-31.48923,-30.95731,-30.8653,-30.77191,-30.67261,-30.60633,-30.76692,-30.50353,-30.14565,-30.09264,-28.96423,-25.56901,-13.00323,-8.17247,-4.98892,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 'y' : [-509.04055,-505.43365,-501.82675,-498.21986,-494.61296,-491.00607,-487.39917,-483.79228,-480.18538,-476.57848,-472.97159,-469.36469,-465.75779,-462.1509,-458.544,-454.93711,-451.33021,-447.72332,-444.11642,-440.50952,-436.90263,-433.29573,-429.68884,-426.08194,-422.47504,-418.86815,-415.26125,-411.65436,-408.04746,-404.44056,-400.83367,-397.22677,-393.61988,-390.01298,-386.40608,-382.79919,-379.19229,-375.5854,-371.9785,-368.3716,-364.76471,-361.15781,-357.55092,-353.94402,-350.33713,-346.73023,-343.12333,-339.51644,-335.90954,-332.30265,-328.69575,-325.08885,-321.48196,-317.87506,-314.26817,-310.66127,-307.05437,-303.44748,-299.84058,-296.23369,-292.62679,-289.0199,-285.413,-281.8061,-278.19921,-274.59231,-270.98541,-267.37852,-263.77162,-260.16473,-256.55783,-252.95094,-249.34404,-245.73714,-242.13025,-238.52335,-234.91646,-231.30956,-227.70266,-224.09577,-220.48887,-216.88198,-213.27508,-209.66818,-206.06129,-202.45439,-198.8475,-195.2406,-191.6337,-188.02681,-184.41991,-180.81302,-177.20612,-173.59922,-169.99233,-166.38543,-162.77854,-159.17164,-155.56475,-151.95785,-148.35095,-144.74406,-141.13716,-137.53027,-133.92337,-130.31647,-126.70958,-123.10268,-119.49579,-115.88889,-112.28199,-108.6751,-105.0682,-101.46131,-97.85441,-94.24752,-90.64062,-87.03372,-83.42683,-79.81993,-76.21304,-72.60614,-68.99924,-65.39235,-61.78545,-58.17856,-54.57166,-50.96476,-47.35787,-43.75097,-40.14408]},
        '250.69364' : { 'x' : [0,-41.12762,-41.10353,-41.18286,-40.63047,-41.11374,-41.01312,-40.94579,-40.98027,-41.30701,-41.26125,-41.35321,-41.31038,-41.35625,-41.11908,-41.10636,-41.17849,-40.9925,-41.05931,-40.95564,-40.82377,-41.0212,-41.09239,-41.11376,-41.33117,-41.34277,-41.1865,-41.07085,-40.95535,-40.80567,-40.63022,-40.49942,-40.45366,-40.43001,-40.37754,-40.40798,-40.48236,-40.58816,-40.8555,-40.94068,-40.90691,-40.7241,-40.4806,-40.20548,-40.01435,-39.79583,-39.60767,-39.37581,-39.17543,-39.11528,-39.00791,-39.03652,-39.0126,-38.90703,-38.79548,-38.59152,-38.41076,-38.2526,-38.15916,-38.06703,-37.906,-37.78979,-37.65957,-37.59591,-37.40029,-37.23094,-36.99284,-36.65838,-36.25972,-35.88881,-35.52005,-35.41332,-35.37445,-35.45368,-35.84497,-36.36585,-36.46692,-33.29004,-18.5936,-11.44076,-5.99633,-2.9655,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 'y' : [-509.04055,-505.43365,-501.82675,-498.21986,-494.61296,-491.00607,-487.39917,-483.79228,-480.18538,-476.57848,-472.97159,-469.36469,-465.75779,-462.1509,-458.544,-454.93711,-451.33021,-447.72332,-444.11642,-440.50952,-436.90263,-433.29573,-429.68884,-426.08194,-422.47504,-418.86815,-415.26125,-411.65436,-408.04746,-404.44056,-400.83367,-397.22677,-393.61988,-390.01298,-386.40608,-382.79919,-379.19229,-375.5854,-371.9785,-368.3716,-364.76471,-361.15781,-357.55092,-353.94402,-350.33713,-346.73023,-343.12333,-339.51644,-335.90954,-332.30265,-328.69575,-325.08885,-321.48196,-317.87506,-314.26817,-310.66127,-307.05437,-303.44748,-299.84058,-296.23369,-292.62679,-289.0199,-285.413,-281.8061,-278.19921,-274.59231,-270.98541,-267.37852,-263.77162,-260.16473,-256.55783,-252.95094,-249.34404,-245.73714,-242.13025,-238.52335,-234.91646,-231.30956,-227.70266,-224.09577,-220.48887,-216.88198,-213.27508,-209.66818,-206.06129,-202.45439,-198.8475,-195.2406,-191.6337,-188.02681,-184.41991,-180.81302,-177.20612,-173.59922,-169.99233,-166.38543,-162.77854,-159.17164,-155.56475,-151.95785,-148.35095,-144.74406,-141.13716,-137.53027,-133.92337,-130.31647,-126.70958,-123.10268,-119.49579,-115.88889,-112.28199,-108.6751,-105.0682,-101.46131,-97.85441,-94.24752,-90.64062,-87.03372,-83.42683,-79.81993,-76.21304,-72.60614,-68.99924,-65.39235,-61.78545,-58.17856,-54.57166,-50.96476,-47.35787,-43.75097,-40.14408]},
        '149.70055' : { 'x' : [-43.04073,-43.04301,-43.35744,-43.3755,-43.3978,-43.34026,-43.54477,-43.45795,-43.18733,-42.99693,-42.92295,-42.72651,-42.95482,-43.11361,-43.36419,-43.66395,-43.93164,-44.22411,-44.40821,-44.48496,-44.40862,-44.23736,-43.90535,-43.66964,-43.40343,-43.42273,-43.58646,-43.75674,-44.05275,-44.41964,-44.97883,-45.31151,-45.56562,-45.7463,-45.67908,-45.4679,-45.1576,-44.78646,-44.52164,-44.44081,-44.47343,-44.58876,-44.8364,-45.24479,-45.64526,-45.9915,-46.23465,-46.24294,-46.12262,-45.71673,-45.21209,-44.55192,-44.07991,-43.74076,-43.36269,-43.14587,-43.11758,-43.28086,-43.41209,-42.65349,-40.79814,-38.17664,-17.25621,-9.7385,-4.90888,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 'y' : [-509.04055,-505.43365,-501.82675,-498.21986,-494.61296,-491.00607,-487.39917,-483.79228,-480.18538,-476.57848,-472.97159,-469.36469,-465.75779,-462.1509,-458.544,-454.93711,-451.33021,-447.72332,-444.11642,-440.50952,-436.90263,-433.29573,-429.68884,-426.08194,-422.47504,-418.86815,-415.26125,-411.65436,-408.04746,-404.44056,-400.83367,-397.22677,-393.61988,-390.01298,-386.40608,-382.79919,-379.19229,-375.5854,-371.9785,-368.3716,-364.76471,-361.15781,-357.55092,-353.94402,-350.33713,-346.73023,-343.12333,-339.51644,-335.90954,-332.30265,-328.69575,-325.08885,-321.48196,-317.87506,-314.26817,-310.66127,-307.05437,-303.44748,-299.84058,-296.23369,-292.62679,-289.0199,-285.413,-281.8061,-278.19921,-274.59231,-270.98541,-267.37852,-263.77162,-260.16473,-256.55783,-252.95094,-249.34404,-245.73714,-242.13025,-238.52335,-234.91646,-231.30956,-227.70266,-224.09577,-220.48887,-216.88198,-213.27508,-209.66818,-206.06129,-202.45439,-198.8475,-195.2406,-191.6337,-188.02681,-184.41991,-180.81302,-177.20612,-173.59922,-169.99233,-166.38543,-162.77854,-159.17164,-155.56475,-151.95785,-148.35095,-144.74406,-141.13716,-137.53027,-133.92337,-130.31647,-126.70958,-123.10268,-119.49579,-115.88889,-112.28199,-108.6751,-105.0682,-101.46131,-97.85441,-94.24752,-90.64062,-87.03372,-83.42683,-79.81993,-76.21304,-72.60614,-68.99924,-65.39235,-61.78545,-58.17856,-54.57166,-50.96476,-47.35787,-43.75097,-40.14408]},
        '52.31346' : { 'x' : [-43.6195,-43.98336,-44.60835,-44.75996,-44.99353,-45.1682,-45.01848,-44.82324,-44.68268,-44.45565,-44.30043,-44.39098,-44.56374,-44.85364,-45.17233,-45.37177,-45.53787,-45.61482,-45.64027,-45.64867,-45.68178,-45.7448,-45.92438,-46.14451,-46.27243,-46.36672,-46.38896,-46.31539,-46.23009,-46.2018,-46.23675,-46.50073,-46.89004,-47.35808,-47.82906,-48.1278,-48.33542,-48.38527,-48.36044,-48.21544,-48.0046,-47.95386,-48.12313,-48.55862,-49.05601,-49.47647,-49.71499,-49.81926,-49.7689,-49.67,-49.5284,-49.60388,-49.89296,-50.18518,-49.04577,-44.61401,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 'y' : [-509.04055,-505.43365,-501.82675,-498.21986,-494.61296,-491.00607,-487.39917,-483.79228,-480.18538,-476.57848,-472.97159,-469.36469,-465.75779,-462.1509,-458.544,-454.93711,-451.33021,-447.72332,-444.11642,-440.50952,-436.90263,-433.29573,-429.68884,-426.08194,-422.47504,-418.86815,-415.26125,-411.65436,-408.04746,-404.44056,-400.83367,-397.22677,-393.61988,-390.01298,-386.40608,-382.79919,-379.19229,-375.5854,-371.9785,-368.3716,-364.76471,-361.15781,-357.55092,-353.94402,-350.33713,-346.73023,-343.12333,-339.51644,-335.90954,-332.30265,-328.69575,-325.08885,-321.48196,-317.87506,-314.26817,-310.66127,-307.05437,-303.44748,-299.84058,-296.23369,-292.62679,-289.0199,-285.413,-281.8061,-278.19921,-274.59231,-270.98541,-267.37852,-263.77162,-260.16473,-256.55783,-252.95094,-249.34404,-245.73714,-242.13025,-238.52335,-234.91646,-231.30956,-227.70266,-224.09577,-220.48887,-216.88198,-213.27508,-209.66818,-206.06129,-202.45439,-198.8475,-195.2406,-191.6337,-188.02681,-184.41991,-180.81302,-177.20612,-173.59922,-169.99233,-166.38543,-162.77854,-159.17164,-155.56475,-151.95785,-148.35095,-144.74406,-141.13716,-137.53027,-133.92337,-130.31647,-126.70958,-123.10268,-119.49579,-115.88889,-112.28199,-108.6751,-105.0682,-101.46131,-97.85441,-94.24752,-90.64062,-87.03372,-83.42683,-79.81993,-76.21304,-72.60614,-68.99924,-65.39235,-61.78545,-58.17856,-54.57166,-50.96476,-47.35787,-43.75097,-40.14408]},
        '-52.28562' : { 'x' : [0,0,-45.08439,-44.85216,-44.5715,-44.08494,-43.60524,-43.37512,-43.37848,-43.64629,-44.12841,-44.75451,-45.28163,-45.52933,-45.47296,-45.12441,-44.62423,-44.17583,-43.93015,-43.94692,-44.28012,-44.86466,-45.53847,-46.05763,-46.32445,-46.28696,-45.95347,-45.44557,-45.00565,-44.80158,-44.94874,-45.41338,-46.1347,-46.84951,-47.32182,-47.53704,-47.48403,-47.31303,-47.04646,-46.85718,-46.90006,-47.24277,-47.8826,-48.62152,-49.18649,-49.51833,-49.73021,-49.87907,-50.07334,-50.38764,-50.86037,-51.52793,-52.11065,-52.38463,-45.05057,-20.48912,-13.41677,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 'y' : [-509.04055,-505.43365,-501.82675,-498.21986,-494.61296,-491.00607,-487.39917,-483.79228,-480.18538,-476.57848,-472.97159,-469.36469,-465.75779,-462.1509,-458.544,-454.93711,-451.33021,-447.72332,-444.11642,-440.50952,-436.90263,-433.29573,-429.68884,-426.08194,-422.47504,-418.86815,-415.26125,-411.65436,-408.04746,-404.44056,-400.83367,-397.22677,-393.61988,-390.01298,-386.40608,-382.79919,-379.19229,-375.5854,-371.9785,-368.3716,-364.76471,-361.15781,-357.55092,-353.94402,-350.33713,-346.73023,-343.12333,-339.51644,-335.90954,-332.30265,-328.69575,-325.08885,-321.48196,-317.87506,-314.26817,-310.66127,-307.05437,-303.44748,-299.84058,-296.23369,-292.62679,-289.0199,-285.413,-281.8061,-278.19921,-274.59231,-270.98541,-267.37852,-263.77162,-260.16473,-256.55783,-252.95094,-249.34404,-245.73714,-242.13025,-238.52335,-234.91646,-231.30956,-227.70266,-224.09577,-220.48887,-216.88198,-213.27508,-209.66818,-206.06129,-202.45439,-198.8475,-195.2406,-191.6337,-188.02681,-184.41991,-180.81302,-177.20612,-173.59922,-169.99233,-166.38543,-162.77854,-159.17164,-155.56475,-151.95785,-148.35095,-144.74406,-141.13716,-137.53027,-133.92337,-130.31647,-126.70958,-123.10268,-119.49579,-115.88889,-112.28199,-108.6751,-105.0682,-101.46131,-97.85441,-94.24752,-90.64062,-87.03372,-83.42683,-79.81993,-76.21304,-72.60614,-68.99924,-65.39235,-61.78545,-58.17856,-54.57166,-50.96476,-47.35787,-43.75097,-40.14408]},
        '-149.67181' : { 'x' : [0,0,-42.00595,-42.43527,-42.4674,-42.98524,-43.49476,-43.85803,-43.88801,-43.55816,-43.06686,-42.56186,-42.28705,-42.25049,-42.48604,-42.95151,-43.53651,-43.92133,-44.00049,-43.70188,-43.16453,-42.64325,-42.31244,-42.25754,-42.47174,-42.94811,-43.49004,-43.90948,-43.98344,-43.66183,-43.10093,-42.54948,-42.22186,-42.15801,-42.35464,-42.76939,-43.22771,-43.59416,-43.66143,-43.40812,-42.90545,-42.32654,-41.91282,-41.80799,-41.96671,-42.28943,-42.68972,-42.96169,-42.97648,-42.67495,-42.12684,-41.59261,-41.22733,-41.03272,-41.02536,-41.23125,-41.48983,-41.65489,-41.54476,-41.23432,-40.76307,-39.87049,-37.18825,-30.85169,-21.47157,-13.23779,-8.10664,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 'y' : [-509.04055,-505.43365,-501.82675,-498.21986,-494.61296,-491.00607,-487.39917,-483.79228,-480.18538,-476.57848,-472.97159,-469.36469,-465.75779,-462.1509,-458.544,-454.93711,-451.33021,-447.72332,-444.11642,-440.50952,-436.90263,-433.29573,-429.68884,-426.08194,-422.47504,-418.86815,-415.26125,-411.65436,-408.04746,-404.44056,-400.83367,-397.22677,-393.61988,-390.01298,-386.40608,-382.79919,-379.19229,-375.5854,-371.9785,-368.3716,-364.76471,-361.15781,-357.55092,-353.94402,-350.33713,-346.73023,-343.12333,-339.51644,-335.90954,-332.30265,-328.69575,-325.08885,-321.48196,-317.87506,-314.26817,-310.66127,-307.05437,-303.44748,-299.84058,-296.23369,-292.62679,-289.0199,-285.413,-281.8061,-278.19921,-274.59231,-270.98541,-267.37852,-263.77162,-260.16473,-256.55783,-252.95094,-249.34404,-245.73714,-242.13025,-238.52335,-234.91646,-231.30956,-227.70266,-224.09577,-220.48887,-216.88198,-213.27508,-209.66818,-206.06129,-202.45439,-198.8475,-195.2406,-191.6337,-188.02681,-184.41991,-180.81302,-177.20612,-173.59922,-169.99233,-166.38543,-162.77854,-159.17164,-155.56475,-151.95785,-148.35095,-144.74406,-141.13716,-137.53027,-133.92337,-130.31647,-126.70958,-123.10268,-119.49579,-115.88889,-112.28199,-108.6751,-105.0682,-101.46131,-97.85441,-94.24752,-90.64062,-87.03372,-83.42683,-79.81993,-76.21304,-72.60614,-68.99924,-65.39235,-61.78545,-58.17856,-54.57166,-50.96476,-47.35787,-43.75097,-40.14408]},
        '-250.6649' : { 'x' : [0,0,0,0,-43.17766,-42.89289,-42.34594,-41.90469,-41.70026,-41.72293,-42.04218,-42.47118,-42.94416,-43.1443,-42.97304,-42.5302,-41.98482,-41.61671,-41.54921,-41.75225,-42.218,-42.70282,-42.9473,-42.84417,-42.46563,-42.00358,-41.57538,-41.40322,-41.52595,-41.87161,-42.29916,-42.6291,-42.66689,-42.35695,-41.84528,-41.37887,-41.1422,-41.12528,-41.35384,-41.79047,-42.14848,-42.28374,-42.06119,-41.63283,-41.13888,-40.81297,-40.71802,-40.73575,-40.90263,-41.1777,-41.39709,-41.38274,-41.1245,-40.66669,-40.27596,-40.09475,-39.97624,-39.93125,-40.10514,-40.36093,-40.47293,-40.28931,-39.97087,-39.71387,-39.59727,-39.53438,-39.1989,-37.56511,-33.80654,-28.27594,-22.31136,-16.52278,-11.10503,-6.76369,-3.95744,-2.36247,-1.41188,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 'y' : [-509.04055,-505.43365,-501.82675,-498.21986,-494.61296,-491.00607,-487.39917,-483.79228,-480.18538,-476.57848,-472.97159,-469.36469,-465.75779,-462.1509,-458.544,-454.93711,-451.33021,-447.72332,-444.11642,-440.50952,-436.90263,-433.29573,-429.68884,-426.08194,-422.47504,-418.86815,-415.26125,-411.65436,-408.04746,-404.44056,-400.83367,-397.22677,-393.61988,-390.01298,-386.40608,-382.79919,-379.19229,-375.5854,-371.9785,-368.3716,-364.76471,-361.15781,-357.55092,-353.94402,-350.33713,-346.73023,-343.12333,-339.51644,-335.90954,-332.30265,-328.69575,-325.08885,-321.48196,-317.87506,-314.26817,-310.66127,-307.05437,-303.44748,-299.84058,-296.23369,-292.62679,-289.0199,-285.413,-281.8061,-278.19921,-274.59231,-270.98541,-267.37852,-263.77162,-260.16473,-256.55783,-252.95094,-249.34404,-245.73714,-242.13025,-238.52335,-234.91646,-231.30956,-227.70266,-224.09577,-220.48887,-216.88198,-213.27508,-209.66818,-206.06129,-202.45439,-198.8475,-195.2406,-191.6337,-188.02681,-184.41991,-180.81302,-177.20612,-173.59922,-169.99233,-166.38543,-162.77854,-159.17164,-155.56475,-151.95785,-148.35095,-144.74406,-141.13716,-137.53027,-133.92337,-130.31647,-126.70958,-123.10268,-119.49579,-115.88889,-112.28199,-108.6751,-105.0682,-101.46131,-97.85441,-94.24752,-90.64062,-87.03372,-83.42683,-79.81993,-76.21304,-72.60614,-68.99924,-65.39235,-61.78545,-58.17856,-54.57166,-50.96476,-47.35787,-43.75097,-40.14408]},
        '-337.2304' : { 'x' : [0,0,0,0,-42.27765,-42.60048,-42.45622,-41.97988,-41.6654,-41.28435,-41.17909,-41.44032,-41.8107,-42.32526,-42.50633,-42.45419,-41.99228,-41.61097,-41.29943,-41.13369,-41.33702,-41.72906,-42.03708,-42.37822,-42.47163,-42.1874,-41.73112,-41.44131,-41.27233,-41.21888,-41.43282,-41.84888,-42.2442,-42.40089,-42.21707,-41.83088,-41.56843,-41.29417,-41.15288,-41.17486,-41.52779,-42.01587,-42.24551,-42.20157,-41.96715,-41.67031,-41.28934,-41.06696,-40.96365,-41.1198,-41.4102,-41.76774,-41.95019,-42.02945,-41.84493,-41.40466,-41.06462,-40.83219,-40.68342,-40.70931,-41.03493,-41.28902,-41.54973,-41.68461,-41.58116,-41.38721,-41.12413,-40.77567,-40.47257,-40.06689,-39.28149,-37.05093,-33.63775,-29.82283,-26.7019,-24.20079,-20.99117,-17.95106,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 'y' : [-509.04055,-505.43365,-501.82675,-498.21986,-494.61296,-491.00607,-487.39917,-483.79228,-480.18538,-476.57848,-472.97159,-469.36469,-465.75779,-462.1509,-458.544,-454.93711,-451.33021,-447.72332,-444.11642,-440.50952,-436.90263,-433.29573,-429.68884,-426.08194,-422.47504,-418.86815,-415.26125,-411.65436,-408.04746,-404.44056,-400.83367,-397.22677,-393.61988,-390.01298,-386.40608,-382.79919,-379.19229,-375.5854,-371.9785,-368.3716,-364.76471,-361.15781,-357.55092,-353.94402,-350.33713,-346.73023,-343.12333,-339.51644,-335.90954,-332.30265,-328.69575,-325.08885,-321.48196,-317.87506,-314.26817,-310.66127,-307.05437,-303.44748,-299.84058,-296.23369,-292.62679,-289.0199,-285.413,-281.8061,-278.19921,-274.59231,-270.98541,-267.37852,-263.77162,-260.16473,-256.55783,-252.95094,-249.34404,-245.73714,-242.13025,-238.52335,-234.91646,-231.30956,-227.70266,-224.09577,-220.48887,-216.88198,-213.27508,-209.66818,-206.06129,-202.45439,-198.8475,-195.2406,-191.6337,-188.02681,-184.41991,-180.81302,-177.20612,-173.59922,-169.99233,-166.38543,-162.77854,-159.17164,-155.56475,-151.95785,-148.35095,-144.74406,-141.13716,-137.53027,-133.92337,-130.31647,-126.70958,-123.10268,-119.49579,-115.88889,-112.28199,-108.6751,-105.0682,-101.46131,-97.85441,-94.24752,-90.64062,-87.03372,-83.42683,-79.81993,-76.21304,-72.60614,-68.99924,-65.39235,-61.78545,-58.17856,-54.57166,-50.96476,-47.35787,-43.75097,-40.14408]},
        '-380.51315' : { 'x' : [0,0,0,0,-42.07063,-42.08302,-42.13489,-42.89521,-42.88972,-42.30242,-42.13338,-42.22357,-42.26422,-41.66594,-42.03601,-42.08232,-42.60303,-42.63281,-42.69367,-42.40635,-41.97062,-41.94201,-41.79264,-41.96244,-42.351,-42.95755,-43.15381,-43.18877,-42.81261,-42.35627,-42.12322,-41.88153,-41.67507,-41.96904,-42.38514,-42.82665,-42.98139,-42.83695,-42.6315,-42.42045,-42.12315,-41.7611,-41.72852,-42.02555,-42.46332,-43.04164,-43.33906,-43.50203,-43.35519,-42.64236,-42.22851,-41.94441,-41.93073,-41.86641,-42.52629,-42.97324,-42.76897,-42.38886,-42.33268,-42.21077,-41.85395,-41.72571,-41.73141,-41.56452,-41.90872,-42.16292,-42.84671,-43.25271,-43.36883,-43.02794,-42.38402,-41.6992,-40.38714,-38.79851,-35.87535,-33.04859,-31.17659,-29.78452,-26.29896,-22.54885,-15.96828,-7.77095,-3.51757,-2.38281,-1.80708,-3.94414,-2.37112,-5.5183,-3.64206,-3.17117,-3.10209,-2.52393,-2.87579,-2.9633,-3.74749,-1.877,-3.29435,-1.86436,-0.96257,-0.80598,-0.4121,0.01553,-0.21318,0.95164,-0.21426,2.31256,0.22006,-0.70118,-0.36672,2.80327,0.00199,1.89094,-1.35753,-1.79618,-0.57995,-3.13614,-1.89008,-12.35293,-16.31416,-18.98481,-20.76801,-27.67342,-28.46497,-15.51234,-21.46011,3.05087,-6.73499,-8.15703,3.66124,-0.94633,6.01571], 'y' : [-509.04055,-505.43365,-501.82675,-498.21986,-494.61296,-491.00607,-487.39917,-483.79228,-480.18538,-476.57848,-472.97159,-469.36469,-465.75779,-462.1509,-458.544,-454.93711,-451.33021,-447.72332,-444.11642,-440.50952,-436.90263,-433.29573,-429.68884,-426.08194,-422.47504,-418.86815,-415.26125,-411.65436,-408.04746,-404.44056,-400.83367,-397.22677,-393.61988,-390.01298,-386.40608,-382.79919,-379.19229,-375.5854,-371.9785,-368.3716,-364.76471,-361.15781,-357.55092,-353.94402,-350.33713,-346.73023,-343.12333,-339.51644,-335.90954,-332.30265,-328.69575,-325.08885,-321.48196,-317.87506,-314.26817,-310.66127,-307.05437,-303.44748,-299.84058,-296.23369,-292.62679,-289.0199,-285.413,-281.8061,-278.19921,-274.59231,-270.98541,-267.37852,-263.77162,-260.16473,-256.55783,-252.95094,-249.34404,-245.73714,-242.13025,-238.52335,-234.91646,-231.30956,-227.70266,-224.09577,-220.48887,-216.88198,-213.27508,-209.66818,-206.06129,-202.45439,-198.8475,-195.2406,-191.6337,-188.02681,-184.41991,-180.81302,-177.20612,-173.59922,-169.99233,-166.38543,-162.77854,-159.17164,-155.56475,-151.95785,-148.35095,-144.74406,-141.13716,-137.53027,-133.92337,-130.31647,-126.70958,-123.10268,-119.49579,-115.88889,-112.28199,-108.6751,-105.0682,-101.46131,-97.85441,-94.24752,-90.64062,-87.03372,-83.42683,-79.81993,-76.21304,-72.60614,-68.99924,-65.39235,-61.78545,-58.17856,-54.57166,-50.96476,-47.35787,-43.75097,-40.14408]},
        
         }
                  
    xData =[-380.51315,-337.2304 ,-250.6649,-149.67181,-52.28562, 52.31346, 149.70055 ,  250.69364, 351.68672, 423.82464  ]
    
    
    for i in xData:
        #Extract y dimension and convert from mm to meter
        refY = np.array(testData[str(i)]['y'])/-1000
        #u is already converted to model units (m/s) no need to convert reference velocity
        #Ref uses neg X for flow direction we'll reverse for posX
        refX = np.array(testData[str(i)]['x'])*-1
    
        #From reference x0 (rear of body) find x1 for plot            
        x1 = x0[0] + (i/-1000)
        
        print(f' x1 is {x1}')
        sim.macro(sim.f_0, sim.bc_mask, sim.rho, sim.u, streamId=0)
        filename = os.path.join(output_dir, f"{prefix}_{str(i)}")
        wp.synchronize()                 
        IOexporter.to_line(
            filename,
            {"velocity": sim.u},
            start_point=(x1, x0[1], x0[2]),
            end_point=(x1,  x0[1], x0[2]+1.0),            
            resolution=50,   
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
        plt.plot(refX, refY, 'o', mfc='none', label='Experimental')
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
xlb.init(
    velocity_set=velocity_set,
    default_backend=compute_backend,
    default_precision_policy=precision_policy,
)

# Generate mesh
if mesher_type == "makemesh":
    level_data, body_vertices, grid_shape_zip, partSize, actual_num_levels, shift, sparsity_pattern, level_origins, x0 = generate_makemesh_mesh(
        stl_filename, voxel_size, trim, trim_voxels
    )
elif mesher_type == "cuboid":
    level_data, body_vertices, grid_shape_zip, partSize, actual_num_levels, shift, sparsity_pattern, level_origins, x0 = generate_cuboid_mesh(
        stl_filename, voxel_size, trim, trim_voxels
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
omega = 1.0 / (3.0 * nu_lattice + 0.5)

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
    offset=-shift,
    unit_convertor=unit_convertor,
    )
bc_mask_exporter = MultiresIO(
    {"bc_mask": 1},
    level_data,
    offset=-shift,
    unit_convertor=unit_convertor,
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
final_print_interval = max(1, int((num_steps-crossover_step) * (print_interval_percentage / 100.0)))

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
sim = initialize_simulation(grid, boundary_conditions, omega, initializer)

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
bc_mask_exporter.to_hdf5(filename, {"bc_mask": sim.bc_mask}, compression="gzip", compression_opts=0)

wp.synchronize()

# Setup momentum transfer
# momentum_transfer = MultiresMomentumTransfer(boundary_conditions[-1], compute_backend=compute_backend)  # bc_body

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
print(f"Relaxation parameter (omega): {omega:.5f}")
time_remaining =(total_lattice_updates_per_step * num_steps) / (100 * 1e6)
hours, rem = divmod(time_remaining, 3600)
minutes, seconds = divmod(rem, 60)
time_remaining_str = f"{int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s"
print(f"Approx Runtime (assuming 100mlups): {time_remaining_str} \n")
print("\n" + "=" * 50 + "\n")

# -------------------------- Simulation Loop --------------------------
wp.synchronize()
start_time = time.time()
compute_time = 0.0
steps_since_last_print = 0
drag_values = []

for step in range(num_steps):
    step_start = time.time()
    sim.step()   
    compute_time += time.time() - step_start
    steps_since_last_print += 1
    if step % print_interval == 0 or step == num_steps - 1:
        sim.macro(sim.f_0, sim.bc_mask, sim.rho, sim.u, streamId=0)
               
        filename = os.path.join(output_dir, f"{script_name}_{step:04d}")
        h5exporter.to_slice_image(
           filename,
           {"velocity": sim.u},
           plane_point=(1, 0, 0),
           plane_normal=(0, 1, 0),
           grid_res=1000,
           bounds=(0, 1, 0, 1),
           cmap="nipy_spectral",
           show_axes=False,
           show_colorbar=False,
           slice_thickness=delta_x_coarse, #needed when using model units
           normalize = u_physical*1.5, #eventually we could have the 1.5 read from json as we did before
        )
        wp.synchronize() 
        cd, cl, drag = print_lift_drag(sim, step, momentum_transfer, ulb, reference_area, voxel_size)
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
        print(f"  Cd= {cd:.3f}, Cl= {cl:.3f}, Drag Force (lattice units)={drag:.3f}")
        start_time = time.time()
        compute_time = 0.0
        steps_since_last_print = 0
    file_output_interval = file_output_interval_pre_crossover if step < crossover_step else file_output_interval_post_crossover
    if step % file_output_interval == 0 or step == num_steps - 1:        
        sim.macro(sim.f_0, sim.bc_mask, sim.rho, sim.u, streamId=0)
        filename = os.path.join(output_dir, f"{script_name}_{step:04d}")
        h5exporter.to_hdf5(filename, {"velocity": sim.u, "density": sim.rho}, compression="gzip", compression_opts=1)
        wp.synchronize()
           
        
    if step >= crossover_step and step % final_print_interval ==0 :
        sim.macro(sim.f_0, sim.bc_mask, sim.rho, sim.u, streamId=0)        
        filename = os.path.join(output_dir, f"{script_name}_{step:04d}")
        percent_complete = (step + 1) / num_steps * 100
        print(f"Completed step {step}/{num_steps} ({percent_complete:.2f}% complete)")
        h5exporter.to_slice_image(
            filename,
            {"velocity": sim.u},
            plane_point=(1, 0, 0),
            plane_normal=(0, 1, 0),
            grid_res=1000,
            bounds=(0, 1, 0, 1),
            show_axes=False,
            show_colorbar=False,
            cmap="nipy_spectral",
            slice_thickness=delta_x_coarse, #needed when using model units
            normalize = u_physical*1.5, #eventually we could have the 1.5 read from json as we did before
        )
        wp.synchronize()        
        cd, cl, drag = print_lift_drag(sim, step, momentum_transfer, ulb, reference_area, voxel_size)
        print(f"  Cd= {cd:.3f}, Cl= {cl:.3f}, Drag Force (lattice units)={drag:.3f}")
        
    if step == num_steps - 1:
        plot_data(x0, output_dir, delta_x_coarse, sim, h5exporter, prefix='SAE')

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
    start_index = int(len(drag_values) * (file_output_crossover_percentage / 100.0))
    last_half = drag_values_array[start_index:, :]
    avg_cd = np.mean(last_half[:, 0])
    avg_cl = np.mean(last_half[:, 1])
    print(f"Average Drag Coefficient (Cd) for last {(100-file_output_crossover_percentage)}%: {avg_cd:.6f}")
    print(f"Average Lift Coefficient (Cl) for last {(100-file_output_crossover_percentage)}%: {avg_cl:.6f}")
    print(f"Experimental Drag Coefficient (Cd): {0.207}")  
    print(f"Error Drag Coefficient (Cd): {((avg_cd-0.207)/0.207)*100:.2f}%")  
    
else:
    print("No drag or lift data collected.")
    
