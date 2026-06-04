from __future__ import annotations
import neon
import warp as wp
import numpy as np
import os, sys, time, trimesh
import matplotlib.pyplot as plt
import gc


from pathlib import Path
from array import array
import struct
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

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
import httpx, logging, getopt, json
from json.decoder import JSONDecodeError
from uuid import uuid4
from threading import Thread

# Use 8 CPU devices if running on ACP
acp_env = os.environ.get('ACP_ENVIRONMENT', '')
if acp_env not in ('', 'local'):
    os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

WORKER_PROTOCOL = os.environ.get('SCM_PROTOCOL', '')
WORKER_HOST = os.environ.get('SCM_HOST', '')
WORKER_PORT = os.environ.get('SCM_PORT', '')

HEARTBEAT_SLEEP = int(float(os.environ.get('SCM_SOLVERHEARTBEAT', 1000)) / 1000)
HEARTBEAT_THREAD = None
HEARTBEAT_CANCELLED = False

### SCM Functions ###
def running_via_scm():
    """
    Checks if the code is running via the SCM worker protocol.

    Returns:
        bool: True if WORKER_PROTOCOL is set (indicating execution via SCM), False otherwise.
    """

    if WORKER_PROTOCOL:
        return True

    return False

def scm_event(endpoint, data=0, event_id=''):
    """
    Sends an event to a specified SCM worker endpoint using HTTP POST and returns the response.

    Args:
        endpoint (str): The endpoint path to send the event to.
        data (int, optional): The data payload to send. Defaults to 0.
        event_id (str, optional): An identifier for the event. Defaults to ''.

    Returns:
        Any: The 'response' field from the JSON response if available, otherwise the provided event_id.

    Notes:
        - If any of WORKER_PROTOCOL, WORKER_HOST, WORKER_PORT, or endpoint are not set, returns the event_id.
        - If the response cannot be decoded as JSON, returns the event_id.
    """

    if not endpoint or not WORKER_PROTOCOL or not WORKER_HOST or not WORKER_PORT:
        return event_id

    url = f'{WORKER_PROTOCOL}://{WORKER_HOST}:{WORKER_PORT}{endpoint}'

    headers = {
        'Content-Type': 'application/json'
    }

    data = {
        'data': data,
        'id': event_id,
    }

    response = httpx.post(url, headers=headers, json=data)

    try:
        return response.json().get('response', event_id)
    except JSONDecodeError:
        return event_id

    return event_id

def heartbeat():
    """
    Continuously sends a heartbeat signal to the compute worker endpoint to indicate the process is alive.

    The function repeatedly calls the `scm_event` function with the '/ComputeWorker/v1/heartbeat' endpoint.
    If the response is 'canceled' or the global variable `HEARTBEAT_CANCELLED` is set to True, the loop breaks and the function returns.
    Otherwise, the function sleeps for a duration specified by the global variable `HEARTBEAT_SLEEP` before sending the next heartbeat.

    Returns:
        None
    """

    while True:
        response = scm_event('/ComputeWorker/v1/heartbeat')

        if response == 'canceled' or HEARTBEAT_CANCELLED:
            return

        time.sleep(HEARTBEAT_SLEEP)

def scm_init():
    """
    Performs SCM initialization by attaching to the compute worker and starting the heartbeat thread.

    This function performs the following actions:
    1. Sends an attach event to the compute worker endpoint.
    2. Creates and starts a global heartbeat thread to maintain regular communication and status checks.

    Globals:
        HEARTBEAT_THREAD: Thread object responsible for running the heartbeat function.

    Side Effects:
        Modifies the global HEARTBEAT_THREAD variable and starts a new thread.
    """

    global HEARTBEAT_THREAD

    scm_event('/ComputeWorker/v1/attach', 1)

    HEARTBEAT_THREAD = Thread(target=heartbeat)
    HEARTBEAT_THREAD.start()

    scm_progress(0)

def scm_progress(progress):
    """
    Sends a progress update to the SCM compute worker.

    Args:
        progress (int): The progress value to send, between 0 and 100.

    Returns:
        None
    """

    scm_event('/ComputeWorker/v1/progress', progress)

def scm_results_available(final_update=False):
    """
    Notifies that results are available by sending an event to the '/ComputeWorker/v1/results' endpoint.

    Args:
        final_update (bool, optional): Indicates whether this is the final update. Defaults to False.

    Returns:
        None
    """

    scm_event('/ComputeWorker/v1/results', int(final_update))

def scm_cancel_heartbeat():
    """
    Cancels the ongoing heartbeat process by setting the HEARTBEAT_CANCELLED flag to True.
    If a heartbeat thread is running, waits for it to finish and then resets the thread reference.
    """

    global HEARTBEAT_CANCELLED
    global HEARTBEAT_THREAD

    HEARTBEAT_CANCELLED = True
    if HEARTBEAT_THREAD:
        HEARTBEAT_THREAD.join()
        HEARTBEAT_THREAD = None

def scm_set_error(code, message):
    """
    Sets an error state by sending an error code and message to the ComputeWorker event handler.

    Args:
        code (int): The error code representing the type of error.
        message (str): A descriptive message explaining the error.

    Returns:
        None

    Side Effects:
        Triggers the '/ComputeWorker/v1/seterror' event with the provided code and message.
    """

    scm_event('/ComputeWorker/v1/seterror', code, message)

def scm_complete():
    """
    Notifies the SCM worker that the process is complete by sending a completion event.

    Returns:
        None
    """

    scm_progress(100)

    scm_event('/ComputeWorker/v1/complete', 1, str(uuid4()))

    scm_cancel_heartbeat()

def obj_to_binary_stl_stream(
    obj_path: str | Path,
    stl_path: str | Path,
    *,
    scale: float = 0.01,
    compute_normals: bool = False,
    assume_triangular_faces: bool = True,
    batch_triangles: int = 1_000_000,
    progress_every: int = 5_000_000,
    allow_leading_whitespace: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Stream a large OBJ directly to binary STL without trimesh/Open3D.

    Memory behavior:
        Stores only vertices as float32.
        Does not store faces.
        Streams STL triangles in batches.

    Parameters
    ----------
    obj_path:
        Input OBJ path.

    stl_path:
        Output binary STL path.

    scale:
        Scale factor applied while reading vertices.
        Example: 0.01 converts cm -> m, or scales by 1%.

    compute_normals:
        If False, writes zero normals for speed.
        Most STL consumers recompute normals anyway.
        If True, computes real per-triangle normals.

    assume_triangular_faces:
        If True, assumes every face is exactly a triangle:
            f a b c
        This is faster.
        If False, supports quads/ngons via fan triangulation.

    batch_triangles:
        Number of STL triangles buffered before writing.
        1,000,000 triangles = 50 MB buffer.

    progress_every:
        Print progress every N triangles.
        Use 0 to disable.

    allow_leading_whitespace:
        If True, handles lines like:
            "   v ..."
            "   f ..."
        Slightly slower. Most OBJ files do not need this.

    verbose:
        Print conversion summary/progress to stderr.

    Returns
    -------
    dict with:
        vertices
        triangles
        seconds
        output_path
    """

    obj_path = Path(obj_path)
    stl_path = Path(stl_path)

    if batch_triangles <= 0:
        raise ValueError("batch_triangles must be positive")

    if not obj_path.exists():
        raise FileNotFoundError(obj_path)

    tri_struct = struct.Struct("<12fH")
    pack_into = tri_struct.pack_into

    # Compact vertex storage:
    # flat float32 array:
    # [x0, y0, z0, x1, y1, z1, ...]
    vertices = array("f")
    v_append = vertices.append

    # STL triangle buffer.
    # Each binary STL triangle is 50 bytes.
    tri_buffer = bytearray(batch_triangles * 50)
    tri_view = memoryview(tri_buffer)
    buffer_offset = 0

    vertex_count = 0
    triangle_count = 0
    line_count = 0

    start = time.perf_counter()

    def log(msg: str) -> None:
        if verbose:
            print(msg, file=sys.stderr, flush=True)

    def parse_index(token: bytes, current_vertex_count: int) -> int:
        """
        Parse OBJ index token:
            b"123"
            b"123/45"
            b"123//67"
            b"123/45/67"
            b"-1"
        Returns zero-based vertex index.
        """
        slash = token.find(b"/")
        if slash != -1:
            token = token[:slash]

        raw = int(token)

        if raw > 0:
            idx = raw - 1
        elif raw < 0:
            idx = current_vertex_count + raw
        else:
            raise ValueError("OBJ index 0 is invalid")

        if idx < 0 or idx >= current_vertex_count:
            raise IndexError(
                f"OBJ index {raw} resolved to invalid index {idx}; "
                f"current vertex count is {current_vertex_count}"
            )

        return idx

    def flush(out_file) -> None:
        nonlocal buffer_offset

        if buffer_offset:
            out_file.write(tri_view[:buffer_offset])
            buffer_offset = 0

    def write_triangle(out_file, ia: int, ib: int, ic: int) -> None:
        nonlocal buffer_offset, triangle_count

        if buffer_offset + 50 > len(tri_buffer):
            flush(out_file)

        a = ia * 3
        b = ib * 3
        c = ic * 3

        ax = vertices[a]
        ay = vertices[a + 1]
        az = vertices[a + 2]

        bx = vertices[b]
        by = vertices[b + 1]
        bz = vertices[b + 2]

        cx = vertices[c]
        cy = vertices[c + 1]
        cz = vertices[c + 2]

        if compute_normals:
            ux = bx - ax
            uy = by - ay
            uz = bz - az

            vx = cx - ax
            vy = cy - ay
            vz = cz - az

            nx = uy * vz - uz * vy
            ny = uz * vx - ux * vz
            nz = ux * vy - uy * vx

            length_sq = nx * nx + ny * ny + nz * nz

            if length_sq > 0.0:
                inv_len = 1.0 / math.sqrt(length_sq)
                nx *= inv_len
                ny *= inv_len
                nz *= inv_len
            else:
                nx = ny = nz = 0.0
        else:
            nx = ny = nz = 0.0

        pack_into(
            tri_buffer,
            buffer_offset,

            # normal
            nx, ny, nz,

            # vertices
            ax, ay, az,
            bx, by, bz,
            cx, cy, cz,

            # STL attribute byte count
            0,
        )

        buffer_offset += 50
        triangle_count += 1

        if progress_every and triangle_count % progress_every == 0:
            elapsed = time.perf_counter() - start
            rate = triangle_count / elapsed if elapsed > 0 else 0.0
            log(
                f"triangles={triangle_count:,} "
                f"vertices={vertex_count:,} "
                f"lines={line_count:,} "
                f"rate={rate:,.0f} tri/s"
            )

    with obj_path.open("rb", buffering=16 * 1024 * 1024) as inp, \
         stl_path.open("wb", buffering=16 * 1024 * 1024) as out:

        header = (
            b"binary STL generated by obj_to_binary_stl_stream "
            b"scale=" + str(scale).encode("ascii")
        )
        out.write(header[:80].ljust(80, b"\0"))

        # Placeholder triangle count; overwritten at the end.
        out.write(struct.pack("<I", 0))

        for raw_line in inp:
            line_count += 1

            if allow_leading_whitespace:
                line = raw_line.lstrip()
            else:
                line = raw_line

            if not line or line.startswith(b"#"):
                continue

            # Vertex line: v x y z
            if len(line) >= 2 and line[0] == 118 and line[1] in (32, 9):  # b"v "
                parts = line.split(maxsplit=4)

                if len(parts) < 4:
                    raise ValueError(f"Invalid vertex near line {line_count}: {line[:120]!r}")

                v_append(float(parts[1]) * scale)
                v_append(float(parts[2]) * scale)
                v_append(float(parts[3]) * scale)

                vertex_count += 1

            # Face line: f a b c ...
            elif len(line) >= 2 and line[0] == 102 and line[1] in (32, 9):  # b"f "
                parts = line.split()

                if len(parts) < 4:
                    continue

                if assume_triangular_faces:
                    # Fast path for triangle-only OBJ files.
                    ia = parse_index(parts[1], vertex_count)
                    ib = parse_index(parts[2], vertex_count)
                    ic = parse_index(parts[3], vertex_count)

                    write_triangle(out, ia, ib, ic)

                else:
                    # Generic path for triangles, quads, and ngons.
                    indices = []

                    for tok in parts[1:]:
                        if tok.startswith(b"#"):
                            break
                        indices.append(parse_index(tok, vertex_count))

                    if len(indices) < 3:
                        continue

                    root = indices[0]

                    # Fan triangulation:
                    # f a b c d -> abc, acd
                    for i in range(1, len(indices) - 1):
                        write_triangle(out, root, indices[i], indices[i + 1])

                if triangle_count > 0xFFFFFFFF:
                    raise OverflowError(
                        "Binary STL stores triangle count as uint32; too many triangles."
                    )

        flush(out)

        # Write final STL triangle count.
        out.seek(80)
        out.write(struct.pack("<I", triangle_count))

    elapsed = time.perf_counter() - start

    log("Done.")
    log(f"Input:      {obj_path}")
    log(f"Output:     {stl_path}")
    log(f"Scale:      {scale}")
    log(f"Normals:    {'computed' if compute_normals else 'zero'}")
    log(f"Vertices:   {vertex_count:,}")
    log(f"Triangles:  {triangle_count:,}")
    log(f"Elapsed:    {elapsed:.1f}s")

    if elapsed > 0:
        log(f"Rate:       {triangle_count / elapsed:,.0f} triangles/s")

    # return {
    #     "vertices": vertex_count,
    #     "triangles": triangle_count,
    #     "seconds": elapsed,
    #     "output_path": str(stl_path),
    # }

####################

wp.clear_kernel_cache()
wp.config.quiet = True

def prep_inputs(input_file):
    version = '1.0'
    start_time = time.time()
    f = open(input_file)
    jsonfile = json.load(f)
    proj_path = os.path.dirname(os.path.abspath(input_file))
    jsonfile['projPath'] = proj_path
    settings = jsonfile['settings']
    voxel_size = settings['voxelSize']
    ulb = settings['ulb']
    # Extract the inlet velocity from the json dict
    prescribed_velocity_phys = jsonfile['InletBC']['x']
    if running_via_scm():
        output_dir = proj_path
    else:
        output_dir = os.path.join(proj_path, jsonfile['outputName'])    
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for fx in [os.path.join(output_dir,f) for f in os.listdir(output_dir)]:
            os.remove(fx)
    
        
    with open(os.path.join(output_dir, "project.log"),'w') as fd:
        fd.write("***  Studio Wind Tunnel Solver Log File ***\n\n\n")
        fd.write("Solver Version: "+version+" \n")
        fd.write("Date Created: "+time.asctime(time.localtime())+" \n\n")  
        fd.write("Processing input json ... \n\n") 
    logging.info("Processing input json ...")

    # Set accuracy and lattice type
    if settings['doublePrecision']==True:
        precision_policy = PrecisionPolicy.FP64FP64
    elif settings['doublePrecision']==-1:
        precision_policy = PrecisionPolicy.FP16FP16
    else:
        precision_policy = PrecisionPolicy.FP32FP32
    
    compute_backend = ComputeBackend.NEON
    velocity_set = xlb.velocity_set.D3Q27(precision_policy=precision_policy, compute_backend=compute_backend)
  
    ### Process Car for obj and scale
    body_stl = os.path.join(proj_path, str(jsonfile['vehicle']['body'][0]))
    filename, file_extension = os.path.splitext(body_stl)    
    print(f' STL path {body_stl}') 
    print(' Loading STL....')   
    #body_mesh = trimesh.load_mesh(body_stl, process=False)        
    if  file_extension =='.obj':
        print(' Loaded obj scaling to STL....')   
        obj_to_binary_stl_stream(
            body_stl,
            os.path.join(output_dir, filename+'.stl'),
            scale=0.01,
            assume_triangular_faces=True,
            compute_normals=False,
        )
        #body_mesh.apply_scale(0.01)
        #body_mesh.export(os.path.join(output_dir, filename+'.stl'))
        #del body_mesh
        #gc.collect()
        body_mesh = trimesh.load_mesh(os.path.join(output_dir, filename+'.stl'), process=False)        
    else:
        body_mesh = trimesh.load_mesh(body_stl, process=False)            
    print(' Body Loaded....')   
    #If any wheels listed
    if len(jsonfile['vehicle']['wheels']) > 0:
        print(' Loading Wheels...') 
        wheel_stls = []
        for wheel in jsonfile['vehicle']['wheels']:
            wheel = os.path.join(proj_path, wheel)
            wheel_stls.append(wheel)    
        wheel_meshes =[]
        w=1
        for wheel in wheel_stls:
            
            if file_extension =='.obj':
                obj_to_binary_stl_stream(
                    wheel,
                    os.path.join(output_dir, 'wheel'+str(w)+'.stl'),
                    scale=0.01,
                    assume_triangular_faces=True,
                    compute_normals=False,
                )
                #wheel_mesh.apply_scale(0.01)        
                #wheel_mesh.export(os.path.join(output_dir, 'wheel'+str(w)+'.stl'))
                #del wheel_mesh
                #gc.collect()
                wheel_mesh = trimesh.load_mesh(os.path.join(output_dir, 'wheel'+str(w)+'.stl'))
            else:
                wheel_mesh = trimesh.load_mesh(wheel, process=False)
            w+=1                
            wheel_meshes.append(wheel_mesh) 
            del wheel_mesh
            gc.collect()
        print(' Wheels Loaded....') 
        print(' Concatenate...')     
        car_mesh = trimesh.util.concatenate([body_mesh] + wheel_meshes)
        print(' Concatenate Done...') 
    else:
        car_mesh = body_mesh.copy()
        wheel_meshes=None
    #print(car_mesh)
    # ===========
    # Initialize XLB
    xlb.init(
        velocity_set=velocity_set,
        default_backend=compute_backend,
        default_precision_policy=precision_policy,
    )

    surface_mesh_for_vtk = car_mesh.copy()
    level_data, body_vertices, wheel_vertices, wheel_centers, grid_shape_zip, partSize, actual_num_levels, shift, sparsity_pattern, level_origins = mesh_prep(
            voxel_size, car_mesh, body_mesh, wheel_meshes, output_dir, jsonfile
        )
  
    # Define a unit convertor
    unit_convertor = UnitConvertor(
    velocity_lbm_unit=ulb,
    velocity_physical_unit=prescribed_velocity_phys,
    voxel_size_physical_unit=voxel_size,
    )

    # Characteristic length
    L = float(partSize[0])    
    #Material Setup
    material = jsonfile['fluid']
    density = material['density']
    dynamic_viscosity = material['viscosity']
    kinematic_viscosity = dynamic_viscosity / density

    # Compute Re   
    Re = abs(prescribed_velocity_phys) * L / kinematic_viscosity

    # Calculate lattice parameters
    delta_x_coarse = voxel_size * 2 ** (actual_num_levels - 1)
    delta_t = voxel_size * ulb / prescribed_velocity_phys
    lbm_visc = unit_convertor.viscosity_to_lbm(kinematic_viscosity)
    omega = 1.0 / (3.0 * lbm_visc + 0.5)

    # Define exporter objects
    field_name_cardinality_dict = {"velocity": 3, "density": 1, "bc_mask": 1}
    h5exporter = MultiresIO(
        field_name_cardinality_dict,
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
    if jsonfile['settings']['flowPasses'] > 0:
        num_steps = int(jsonfile['settings']['flowPasses'] * (grid_shape_x_coarsest / ulb))
    else:
        num_steps = int(jsonfile['settings']['iterations'])
    

    # Setup boundary conditions
    boundary_conditions = setup_boundary_conditions(grid, level_data, body_vertices, wheel_vertices, wheel_centers, ulb, lbm_visc, grid_shape_zip, precision_policy, jsonfile, compute_backend)

    # Create initializer
    initializer = CustomMultiresInitializer(
        bc_id=boundary_conditions[1].id,  # bc_outlet
        constant_velocity_vector=(ulb, 0.0, 0.0),
        velocity_set=velocity_set,
        precision_policy=precision_policy,
        compute_backend=compute_backend,
    )
    
    if jsonfile['settings']['optimization'] == "254":
        mres_perf_opt = xlb.MresPerfOptimizationType.FUSION_AT_FINEST_SFV_ALL
        #mres_perf_opt = xlb.MresPerfOptimizationType.FUSION_AT_FINEST_254_ALL
    else:
        mres_perf_opt = xlb.MresPerfOptimizationType.FUSION_AT_FINEST
    # Initialize simulation   
    sim = xlb.helper.MultiresSimulationManager(
        omega_finest=omega,
        grid=grid,
        boundary_conditions=boundary_conditions,
        collision_type="SmagorinskyLESKBC",
        #collision_type="KBC",
        initializer=initializer,
        mres_perf_opt=mres_perf_opt,
        smagorinsky_constant=jsonfile['settings']['sgs']
    )
    wheel_ids = []
    if wheel_vertices is not None:
        for i, _ in enumerate(wheel_vertices):
            wheel_ids.append(boundary_conditions[-1-i])

    # Compute voxel statistics and reference area
    stats = compute_voxel_statistics_and_reference_area(jsonfile, sim, h5exporter, level_data, actual_num_levels, sparsity_pattern, boundary_conditions, voxel_size, wheel_ids)
    active_voxels = stats["active_voxels"]
    solid_voxels = stats["solid_voxels"]
    total_voxels = stats["total_voxels"]
    total_lattice_updates_per_step = stats["total_lattice_updates_per_step"]
    reference_area = stats["reference_area"]
    reference_area_physical = stats["reference_area_physical"]
             
    wp.synchronize()

    

    # Setup momentum transfer    
    momentum_transfer = MultiresMomentumTransfer(
        boundary_conditions[0],
        mres_perf_opt=xlb.MresPerfOptimizationType.FUSION_AT_FINEST,
        compute_backend=compute_backend,
    )
    wheel_momentum = None
    if wheel_vertices is not None:
        wheel_momentum = []
        for i, _ in enumerate(wheel_vertices):
            mt = MultiresMomentumTransfer(
                boundary_conditions[-1-i],
                mres_perf_opt=xlb.MresPerfOptimizationType.FUSION_AT_FINEST,
                compute_backend=compute_backend,
            )
            wheel_momentum.append(mt)
    
    if settings['debug'] == True: 
        # bcMask = os.path.join(output_dir, f"{jsonfile['outputName']}_initial_bc_mask") 
        # bc_mask_exporter.to_hdf5(bcMask, {"bc_mask": sim.bc_mask}, compression="gzip", compression_opts=0)
 
        # Print simulation info
        print("\n" + "=" * 50 + "\n")
        print(f"Simulation Configuration for Re = {Re}:")
        print(f"Number of flow passes: {jsonfile['settings']['flowPasses']}")
        print(f"Calculated iterations: {num_steps:,}")
        print(f"Finest voxel size: {voxel_size} meters")
        print(f"Coarsest voxel size: {delta_x_coarse} meters")
        print(f"Total voxels: {sum(np.count_nonzero(mask) for mask in sparsity_pattern):,}")
        print(f"Total active voxels: {total_voxels:,}")
        print(f"Active voxels per level: {active_voxels}")
        print(f"Solid voxels per level: {solid_voxels}")
        print(f"Total lattice updates per global step: {total_lattice_updates_per_step:,}")
        print(f"Actual number of refinement levels: {actual_num_levels}")
        print(f"Physical inlet velocity: {prescribed_velocity_phys:.4f} m/s")
        print(f"Lattice velocity (ulb): {ulb}")
        print(f"Characteristic length: {L: .4f} meters")
        print(f"Computed reference area (bc_mask): {reference_area} lattice units")
        print(f"Physical reference area (bc_mask): {reference_area_physical:.6f} m^2")
        print(f"Reynolds number: {Re:,.2f}")
        print(f"Relaxation parameter (omega): {omega:.5f}")
        time_remaining =(total_lattice_updates_per_step * num_steps) / (100 * 1e6)
        hours, rem = divmod(time_remaining, 3600)
        minutes, seconds = divmod(rem, 60)
        time_remaining_str = f"{int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s"
        print(f"Approx Runtime (assuming 100mlups): {time_remaining_str} \n")
        print("\n" + "=" * 50 + "\n")

    with open(os.path.join(output_dir, "project.log"),'a') as fd:
        fd.write('Material Properties\n')
        fd.write('___________________\n')
        fd.write(f'Density:  {density:.4f} kg/m3\n')
        fd.write(f'Visc Dyn: {dynamic_viscosity:.4e} Pa-s\n')
        fd.write(f'Visc Kin: {kinematic_viscosity:.4e} m2/s\n')
        fd.write(f'Visc LBM: {lbm_visc:.4e} \n\n')
        fd.write('Solver Parameters\n')
        fd.write('___________________\n')
        fd.write(f"Number of flow passes: {jsonfile['settings']['flowPasses']}\n")
        fd.write(f"Calculated iterations: {num_steps:,}\n")
        fd.write(f"Finest voxel size: {voxel_size} meters\n")
        fd.write(f"Coarsest voxel size: {delta_x_coarse} meters\n")
        fd.write(f"Total voxels: {sum(np.count_nonzero(mask) for mask in sparsity_pattern):,}\n")
        fd.write(f"Total active voxels: {total_voxels:,}\n")
        fd.write(f"Active voxels per level: {active_voxels}\n")
        fd.write(f"Solid voxels per level: {solid_voxels}\n")
        fd.write(f"Total lattice updates per global step: {total_lattice_updates_per_step:,}\n")
        fd.write(f"Actual number of refinement levels: {actual_num_levels}\n")
        fd.write(f"Physical inlet velocity: {prescribed_velocity_phys:.4f} m/s\n")
        fd.write(f"Lattice velocity (ulb): {ulb}\n")
        fd.write(f"Characteristic length: {L: .4f} meters\n")
        fd.write(f"Computed reference area (bc_mask): {reference_area} lattice units\n")
        fd.write(f"Physical reference area (bc_mask): {reference_area_physical:.6f} m^2\n")
        fd.write(f"Reynolds number: {Re:,.2f}\n")
        fd.write(f'Inlet Velocity:    {prescribed_velocity_phys:.1f} m/s \n')
        fd.write(f'Timestep Size:     {delta_t:.4e} seconds\n')
        fd.write(f'Omega: {omega:.8f}\n')
        
        fd.write('\nResults\n')
        fd.write('___________________\n')
        fd.write(f'Time to initialize:   {(time.time()-start_time)/60:.2f} min\n')  
    
  
    solve(
        sim, 
        ulb,
        num_steps, 
        h5exporter, 
        output_dir, 
        grid_shape_zip,
        grid_shape_x_coarsest, 
        delta_x_coarse, 
        shift,
        momentum_transfer,
        wheel_momentum,
        reference_area,
        reference_area_physical,
        voxel_size,
        prescribed_velocity_phys,
        total_lattice_updates_per_step,
        jsonfile,
        partSize,
        surface_mesh_for_vtk
        )


# Mesh Generation Functions
# =========================
def mesh_prep(voxel_size, car_mesh, body_mesh, wheel_meshes, output_dir, jsonfile):
    
    # Compute bounds on full car
    min_bound = car_mesh.vertices.min(axis=0)
    max_bound = car_mesh.vertices.max(axis=0)
    partSize = max_bound - min_bound  
    
    
    mesher_type = jsonfile['mesher']['type'] 
    # Generate mesh
    if mesher_type == "mres": 
        shift = np.array(
            [               
                jsonfile['mesher']['mres']['domain']["-x"] * partSize[0] - min_bound[0],
                jsonfile['mesher']['mres']['domain']["-y"] * partSize[1] - min_bound[1],
                jsonfile['mesher']['mres']['domain']["-z"] * partSize[2] - min_bound[2],
            ],
            dtype=float,
        ) 
        print(' Shift and export for meshing...') 
        #Apply shift to car mesh for meshing purpose
        car_mesh.apply_translation(shift)
        _ = car_mesh.vertex_normals
        car_mesh.export("temp.stl")
        del car_mesh
        gc.collect()
        print(' Generate Mesh...') 
        # Generate mesh using generate_mesh with ground refinement
        level_data = generate_mesh(
            jsonfile['mesher']['mres']['levels'],
            "temp.stl",
            jsonfile['settings']['voxelSize'],
            jsonfile['mesher']['mres']['padding'],
            jsonfile['mesher']['mres']['domain'],
            ground_refinement_level=jsonfile['mesher']['mres']['ground_refinement_level'],
            ground_voxel_height=jsonfile['mesher']['mres']['ground_voxel_height'],
        )
    elif mesher_type == "cuboid":  
        # Compute translation to put mesh into first octant of the domain
        domain_multiplier = jsonfile['mesher']['cuboid']
        shift = np.array(
            [
                domain_multiplier[0][0] * partSize[0] - min_bound[0],
                domain_multiplier[0][2] * partSize[1] - min_bound[1],
                domain_multiplier[0][4] * partSize[2] - min_bound[2],
            ],
            dtype=float,
        )
        #Apply shift to car mesh for meshing purpose
        car_mesh.apply_translation(shift)
        _ = car_mesh.vertex_normals
        car_mesh.export("temp.stl")
        # Generate mesh using Cuboid Mesher on full car
        level_data = make_cuboid_mesh(
            jsonfile['settings']['voxelSize'],
            domain_multiplier,
            "temp.stl",
        )
    else:
        raise ValueError(f"Invalid mesher_type: {mesher_type}. Must be 'mres' or 'cuboid'.")
    sparsity_pattern, level_origins = prepare_sparsity_pattern(level_data)
    print(' Shift each part...') 
    # Apply translation to each part 
    body_mesh.apply_translation(shift)
    
    vertex_dtype = np.float64 if jsonfile['settings']["doublePrecision"] is True else np.float32
    if wheel_meshes is not None:
        wheel_vertices = []
        wheel_centers = []
        body_vertices = np.asarray(body_mesh.vertices, dtype=vertex_dtype) / vertex_dtype(voxel_size)
        for mesh in wheel_meshes:    
            mesh.apply_translation(shift)
            wheelSize = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0) 
            wheelCenter = ((0.5 * wheelSize) + mesh.vertices.min(axis=0)) / voxel_size
            wheel_centers.append((wheelCenter, wheelSize[0]/voxel_size))
            if jsonfile['mesher']['trim'] == True:
                zShift = jsonfile['mesher']['trim_voxels']
                plane_origin = np.array([0, 0, mesh.bounds[0][2]+(zShift* voxel_size)])
                plane_normal = np.array([0, 0, 1])  # Upward pointing normal
                # Slice the mesh using the defined plane.
                # With cap=True, the open slice is automatically closed off.
                mesh_above = mesh.slice_plane(plane_origin=plane_origin,
                                  plane_normal=plane_normal,
                                  cap=True)
                
                mesh_above.export(os.path.join(output_dir, 'temp.stl'))
                wheel_stl = os.path.join(output_dir, 'temp.stl')
                print(' Load Trimmed Wheel...') 
                wheel_mesh = trimesh.load_mesh(wheel_stl, process=False)
                wheel_vertices.append(np.asarray(wheel_mesh.vertices, dtype=vertex_dtype) / vertex_dtype(voxel_size))
                del wheel_mesh
                gc.collect()
                # print(f"DEBUG: Wheel {mesh}")
                # print(f"  - Bounding Box Size (Model Units): {wheelSize }")
                # print(f"  - Bounding Box Size (Lattice Units): {wheelSize / voxel_size}")
                # print(f"  - Calculated Diameter (Lattice Units): {(wheelSize[0])/voxel_size}")
                # print(f"  - Calculated Center (Lattice Units): {wheelCenter}")
                # print(f"  - Inlet Velocity (ulb): {jsonfile['settings']['ulb']}")
                # print(f"  - Rotation Rate (rads/step): {-2.0 * jsonfile['settings']['ulb'] / ((wheelSize[0])/voxel_size)}")
    
            else:
                wheel_vertices.append(np.asarray(mesh.vertices, dtype=vertex_dtype) / vertex_dtype(voxel_size))
                del mesh
                gc.collect()
    else:
        #No Wheels trim body as needed
        
        wheel_vertices=None
        wheel_centers = None
        if jsonfile['mesher']['trim'] == True:
            zShift = jsonfile['mesher']['trim_voxels']
            plane_origin = np.array([0, 0, body_mesh.bounds[0][2]+(zShift* voxel_size)])
            plane_normal = np.array([0, 0, 1])  # Upward pointing normal
            # Slice the mesh using the defined plane.
            # With cap=True, the open slice is automatically closed off.
            mesh_above = body_mesh.slice_plane(plane_origin=plane_origin,
                        plane_normal=plane_normal,
                        cap=True)
            mesh_above.export(os.path.join(output_dir, 'temp.stl'))
            body_stl = os.path.join(output_dir, 'temp.stl')
            print(' Load Trimmed Mesh...') 
            body_mesh = trimesh.load_mesh(body_stl, process=False)
            body_vertices = np.asarray(body_mesh.vertices, dtype=vertex_dtype) / vertex_dtype(voxel_size)
            
        else:
            body_vertices = np.asarray(body_mesh.vertices, dtype=vertex_dtype) / vertex_dtype(voxel_size)
        del body_mesh
        gc.collect()
        

    actual_num_levels = len(level_data)
    grid_shape_finest = tuple([int(i * 2 ** (actual_num_levels - 1)) for i in level_data[-1][0].shape])
    
    print(f"Full shape based on finest voxel size is {grid_shape_finest}")
    # Clean all temp stls in the folder
    for filename in os.listdir(output_dir):
        # Check if the file ends with '.stl' and is a file (not a directory)
        if filename.endswith('.stl') and os.path.isfile(os.path.join(output_dir, filename)):
            file_path = os.path.join(output_dir, filename)
            os.remove(file_path)
    
    return level_data, body_vertices, wheel_vertices, wheel_centers, grid_shape_finest, partSize, actual_num_levels, shift, sparsity_pattern, level_origins

# Boundary Conditions Setup
# =========================
def setup_boundary_conditions(grid, level_data, body_vertices, wheel_vertices, wheel_centers, ulb, lbm_visc, grid_shape_zip, precision_policy, jsonfile, compute_backend=ComputeBackend.NEON):
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

    if jsonfile['BCtypes']['inlet'] == "RegularizedBC":
        bc_inlet = RegularizedBC("velocity",
            #profile=bc_profile_new(),
            prescribed_value=(ulb, 0.0, 0.0),
            indices=left_indices,
            )
    else:
        bc_inlet = HybridBC(
            bc_method="nonequilibrium_regularized",
            prescribed_value=(ulb, 0.0, 0.0),
            indices=left_indices,
            )
        
        

    bc_outlet = DoNothingBC(indices=right_indices)

    # Setup walls moving, static of fall back to FullBounce
    if jsonfile['BCtypes']['walls'] == "moving":
        bc_top =HybridBC(bc_method="nonequilibrium_regularized", prescribed_value=(ulb, 0.0, 0.0), indices=top_indices)     
        bc_front =HybridBC(bc_method="nonequilibrium_regularized", prescribed_value=(ulb, 0.0, 0.0), indices=filtered_front_indices) 
        bc_back =HybridBC(bc_method="nonequilibrium_regularized", prescribed_value=(ulb, 0.0, 0.0), indices=filtered_back_indices)   
    elif jsonfile['BCtypes']['walls'] == "static":
        bc_top =HybridBC(bc_method="nonequilibrium_regularized", indices=top_indices)     
        bc_front =HybridBC(bc_method="nonequilibrium_regularized", indices=filtered_front_indices) 
        bc_back =HybridBC(bc_method="nonequilibrium_regularized", indices=filtered_back_indices)     
    else:
        bc_top = FullwayBounceBackBC(indices=top_indices)     
        bc_front = FullwayBounceBackBC(indices=filtered_front_indices)
        bc_back = FullwayBounceBackBC(indices=filtered_back_indices)
    # Setup ground moving, static or fall back to FullBounce
    if jsonfile['BCtypes']['ground'] == "moving":
        bc_bottom =HybridBC(bc_method="nonequilibrium_regularized", prescribed_value=(ulb, 0.0, 0.0), indices=bottom_indices)   
    elif jsonfile['BCtypes']['ground'] == "static":        
        bc_bottom =HybridBC(bc_method="nonequilibrium_regularized", indices=bottom_indices)     
    else:        
        bc_bottom = FullwayBounceBackBC(indices=bottom_indices)

    if jsonfile['BCtypes']['car_voxelization'] == "RAY":
        CarvoxelizationMethod = MeshVoxelizationMethod("RAY")
    else:
        CarvoxelizationMethod = MeshVoxelizationMethod("AABB_CLOSE", close_voxels=jsonfile['mesher']['close_voxels'])
    if jsonfile['BCtypes']['wheel_voxelization'] == "RAY":
        WheelvoxelizationMethod = MeshVoxelizationMethod("RAY")
    else:
        WheelvoxelizationMethod = MeshVoxelizationMethod("AABB_CLOSE", close_voxels=jsonfile['mesher']['close_voxels'])
    #Setup car bc
    if jsonfile['BCtypes']['car'] == "nonequilibrium_regularized":
        bc_body = HybridBC(
            bc_method="nonequilibrium_regularized",
            mesh_vertices=body_vertices,
            voxelization_method=CarvoxelizationMethod,
            use_mesh_distance=True,      
            use_wall_model=True,
            kinematic_viscosity=lbm_visc,
        )
    elif jsonfile['BCtypes']['car'] == "bounceback_regularized":
        bc_body = HybridBC(
            bc_method="bounceback_regularized",
            mesh_vertices=body_vertices,
            voxelization_method=CarvoxelizationMethod,
            use_mesh_distance=True,     
            use_wall_model=True,
            kinematic_viscosity=lbm_visc
        )    
    elif jsonfile['BCtypes']['car'] == "bounceback_grads":
        bc_body = HybridBC(
            bc_method="bounceback_grads",
            mesh_vertices=body_vertices,
            voxelization_method=CarvoxelizationMethod,
            use_mesh_distance=True,     
            use_wall_model=True,
            kinematic_viscosity=lbm_visc
        )
    else:
        bc_body = FullwayBounceBackBC(            
            mesh_vertices=body_vertices,
            voxelization_method=CarvoxelizationMethod,
        )

    if wheel_vertices is not None:
        # Define rotating boundary profile
        def wheel_profile(origin_np, rot_rate):
            dtype = precision_policy.compute_precision.wp_dtype
            _u_vec = wp.vec(3, dtype=dtype)
            angular_velocity = _u_vec(0.0, rot_rate, 0.0)
            origin_wp = _u_vec(origin_np[0], origin_np[1], origin_np[2])

            @wp.func
            def bc_profile_warp(index: wp.vec3i):
                x = dtype(index[0])
                y = dtype(index[1])
                z = dtype(index[2])
                surface_coord = _u_vec(x, y, z) - origin_wp
                return wp.cross(angular_velocity, surface_coord)

            return bc_profile_warp
        
        if jsonfile['BCtypes']['wheels'] == "nonequilibrium_regularized": 
            wheel_method="nonequilibrium_regularized"
        elif jsonfile['BCtypes']['wheels'] == "bounceback_grads": 
            wheel_method="bounceback_grads"
        elif jsonfile['BCtypes']['wheels'] == "bounceback_regularized": 
            wheel_method="bounceback_regularized"
        else: 
            wheel_method=None

        wheel_bc = []
        for wheel_vertice, (wheel_center, wheel_dia) in zip(wheel_vertices, wheel_centers):

            if wheel_method is not None: 
                rot_rate = -2.0 * ulb / wheel_dia
                if jsonfile['BCtypes']['wheel_motion']:
                    wheel_bc.append(HybridBC(
                    bc_method=wheel_method,
                    mesh_vertices=wheel_vertice,
                    voxelization_method=WheelvoxelizationMethod,
                    use_mesh_distance=True,                
                    profile=wheel_profile(wheel_center, rot_rate)
                        ))
                else:
                    wheel_bc.append(HybridBC(
                    bc_method=wheel_method,
                    mesh_vertices=wheel_vertice,
                    voxelization_method=WheelvoxelizationMethod,
                    use_mesh_distance=True,                
                        ))
            else:
                wheel_bc.append(FullwayBounceBackBC(
                mesh_vertices=wheel_vertice,
                voxelization_method=WheelvoxelizationMethod,           
                    ))
                
        return [bc_body, bc_outlet, bc_top, bc_bottom, bc_front, bc_back, bc_inlet] + wheel_bc # Body must be last. Outlet must be second to last
    else:
        return [bc_body, bc_outlet, bc_top, bc_bottom, bc_front, bc_back, bc_inlet] # Body must be last. Outlet must be second to last

# Utility Functions
# =================
def print_lift_drag(sim, step, momentum_transfer, wheel_momentum, ulb, reference_area, voxel_size, drag_values):
    """
    Calculate and print lift and drag coefficients.
    """
    boundary_force = momentum_transfer(sim.f_0, sim.f_1, sim.bc_mask, sim.missing_mask, sim.rho0, sim.u0, sim.relax, sim.normal_vector, sim.normal_distance)
    wheel_force = [0.0, 0.0, 0.0]
    if wheel_momentum is not None:
        for i in range(len(wheel_momentum)):
            wheel_force += wheel_momentum[i](sim.f_0, sim.f_1, sim.bc_mask, sim.missing_mask, sim.rho0, sim.u0, sim.relax, sim.normal_vector, sim.normal_distance)
    drag = boundary_force[0] + wheel_force[0]
    lift = boundary_force[2] + wheel_force[2]
    cd = 2.0 * drag / (ulb**2 * reference_area)
    cl = 2.0 * lift / (ulb**2 * reference_area)
    if np.isnan(cd) or np.isnan(cl):
        raise ValueError(f"NaN detected in coefficients at step {step}: Cd={cd}, Cl={cl}")
    drag_values.append([step, cd, cl])    
    return cd, cl, drag

def plot_drag_lift(drag_values, output_dir, script_name, percentile_range=(15, 85), use_log_scale=False):
    """
    Plot CD and CL over time and save the plot to the output directory.
    """
    drag_values_array = np.array(drag_values)
    steps = drag_values_array[:, 0]
    cd_values = drag_values_array[:, 1]
    cl_values = drag_values_array[:, 2]
    p_lower = float(percentile_range[0])    
    p_upper = float(percentile_range[1])
    y_min = min(np.percentile(cd_values,p_lower), np.percentile(cl_values, p_lower))
    y_max = max(np.percentile(cd_values, p_upper), np.percentile(cl_values, p_upper))
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

def compute_voxel_statistics_and_reference_area( jsonfile, sim, h5exporter, level_data, actual_num_levels, sparsity_pattern, boundary_conditions, voxel_size, wheel_ids=[]):
    """
    Compute active/solid voxels, totals, lattice updates, and reference area based on simulation data.
    """
    # Compute macro fields
    sim.macro(sim.f_0, sim.bc_mask, sim.rho, sim.u, streamId=0)
    fields_data = h5exporter.get_fields_data({"bc_mask": sim.bc_mask})
    bc_mask_data = fields_data["bc_mask_0"]
    level_id_field = h5exporter.level_id_field

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

    if jsonfile["BCtypes"]["car_voxelization"] == "RAY":
        target_ids = [bc.id for bc in wheel_ids]
        target_ids.append(boundary_conditions[0].id)
        is_target_voxel = np.isin(bc_mask_finest, target_ids)
    else:
        is_target_voxel = bc_mask_finest == 255

    ref_area_start = time.time()
    solid_voxels_indices = active_indices_finest[is_target_voxel]
    # Project onto YZ plane using a 2D boolean grid — O(N), avoids the
    # lexicographic sort that np.unique(..., axis=0) does on (j,k) pairs.
    finest_shape = level_data[0][0].shape
    yz_hit = np.zeros((finest_shape[1], finest_shape[2]), dtype=bool)
    yz_hit[solid_voxels_indices[:, 1], solid_voxels_indices[:, 2]] = True
    reference_area = int(yz_hit.sum())
    reference_area_physical = reference_area * (voxel_size ** 2)
    print(f"Reference area computation took {time.time() - ref_area_start:.3f} s")

    return {
        "active_voxels": active_voxels,
        "solid_voxels": solid_voxels,
        "total_voxels": total_voxels,
        "total_lattice_updates_per_step": total_lattice_updates_per_step,
        "reference_area": reference_area,
        "reference_area_physical": reference_area_physical
    }

def vehicle_slice_region_bounds(jsonfile, partSize, shift, grid_shape_zip, voxel_size):
    """Axis-aligned box around the vehicle matching the region swept by save_slices().

    Reconstructs the vehicle position in the domain and applies the same
    upstream/downstream/side/height factors used for the result slices, then
    clips to the simulation domain. Returns ((x0, y0, z0), (x1, y1, z1)) in the
    exporter's physical/model coordinate frame (the same frame slice origins use).
    """
    partSize = np.array(partSize, dtype=float)
    L, W, H = map(float, partSize)

    # Same window factors as save_slices()
    UPSTREAM_L = 0.5
    DOWNSTREAM_L = 2.0
    SIDE_W = 0.9
    HEIGHT_H = 1.9
    Z_START_OFFSET = 0.0

    shift = np.array(shift, dtype=float)
    domainSize = np.array(grid_shape_zip, dtype=float) * float(voxel_size)
    domain_min = -shift
    domain_max = domain_min + domainSize

    mesher = jsonfile.get("mesher", {})
    mesher_type = mesher.get("type", "mres")
    if mesher_type == "mres":
        dom = mesher.get("mres", {}).get("domain", {})
        mxm = float(dom.get("-x", 0.0))
        mym = float(dom.get("-y", 0.0))
        mzm = float(dom.get("-z", 0.0))
    else:
        dm0 = mesher.get("cuboid", [[0, 0, 0, 0, 0, 0]])[0]
        mxm, _mxp, mym, _myp, mzm, _mzp = map(float, dm0)

    # "-x/-y/-z" are negative extents: vehicle_min = domain_min - (-m*size)
    domain_min_rel = np.array([-mxm * L, -mym * W, -mzm * H], dtype=float)
    veh_min = domain_min - domain_min_rel
    veh_max = veh_min + partSize
    veh_center = 0.5 * (veh_min + veh_max)

    x0 = veh_min[0] - UPSTREAM_L * L
    x1 = veh_min[0] + DOWNSTREAM_L * L
    y0 = veh_center[1] - SIDE_W * W
    y1 = veh_center[1] + SIDE_W * W
    z0 = veh_min[2] + Z_START_OFFSET
    z1 = veh_min[2] + HEIGHT_H * H

    lo = np.maximum(np.array([x0, y0, z0], dtype=float), domain_min)
    hi = np.minimum(np.array([x1, y1, z1], dtype=float), domain_max)
    return (tuple(lo), tuple(hi))


def save_slices(output_dir, grid_shape_zip, shift, h5exporter, delta_x_coarse, voxel_size, jsonfile, partSize):

    # -----------------------------
    # Settings / constants
    # -----------------------------
    settings = jsonfile.get("settings", {})
    default_num_slices = int(settings.get("numSlices", 51))
    default_num_slices = max(1, default_num_slices)

    grid_res = settings["grid_res"]
    cmap = settings["sliceColorMap"]
    normalize = jsonfile["InletBC"]["x"] * settings["sliceFactor"]

    # Coefficient slices (Cp, CpTotal): proper derived fields synthesized from the
    # time-averaged density/velocity (accumulated with derived=[...] in solve()).
    # Each is linearly mapped from [lo, hi] onto the colormap's [0, 1] range and
    # rendered on the same slice geometry as velocity. This replaces the previous
    # manual density "pressure-analog" hack.
    coeff_slice_specs = (
        ("Cp",
         settings.get("cpColorMap", "bwr"),
         float(settings.get("cpMin", -1.0)),
         float(settings.get("cpMax", 1.0))),
        ("CpTotal",
         settings.get("cpTotalColorMap", "bwr"),
         float(settings.get("cpTotalMin", -1.0)),
         float(settings.get("cpTotalMax", 1.0))),
    )

    partSize = np.array(partSize, dtype=float)
    L, W, H = map(float, partSize)

    # "manual-style" factors (match the example you posted very closely)
    UPSTREAM_L = 0.5   # -0.5L
    DOWNSTREAM_L = 2.0 # +2.0L
    SIDE_W = 0.9       # ~ +/-0.9W (gives ~4.0 wide for your W=2.205)
    HEIGHT_H = 1.9     # ~1.9H (gives ~3.0 high for your H=1.581)
    Z_START_OFFSET = 0.0  # manual starts at 0.1 above ground

    # -----------------------------
    # True simulation domain bounds
    # -----------------------------
    shift = np.array(shift, dtype=float)
    domainSize = np.array(grid_shape_zip, dtype=float) * float(voxel_size)

    domain_min = -shift
    domain_max = domain_min + domainSize
    dom_extent = domain_max - domain_min

    # -----------------------------
    # Parse domain multipliers
    # -----------------------------
    mesher = jsonfile.get("mesher", {})
    mesher_type = mesher.get("type", "mres")

    if mesher_type == "mres":
        dom = mesher.get("mres", {}).get("domain", {})
        mxm = float(dom.get("-x", 0.0))
        mym = float(dom.get("-y", 0.0))
        mzm = float(dom.get("-z", 0.0))
        mxp = float(dom.get("+x", dom.get("x", 0.0)))
        myp = float(dom.get("+y", dom.get("y", 0.0)))
        mzp = float(dom.get("+z", dom.get("z", 0.0)))
    else:
        dm0 = mesher.get("cuboid", [[0, 0, 0, 0, 0, 0]])[0]
        mxm, mxp, mym, myp, mzm, mzp = map(float, dm0)

    # -----------------------------
    # CRITICAL FIX: "-x/-y/-z" are NEGATIVE extents, not positive offsets
    # domain_min = vehicle_min + (-mxm*L, -mym*W, -mzm*H)
    # => vehicle_min = domain_min - (-mxm*L, -mym*W, -mzm*H)
    # -----------------------------
    domain_min_rel = np.array([-mxm * L, -mym * W, -mzm * H], dtype=float)
    domain_max_rel = np.array([ mxp * L,  myp * W,  mzp * H], dtype=float)

    veh_min = domain_min - domain_min_rel
    veh_max = veh_min + partSize
    veh_center = 0.5 * (veh_min + veh_max)

    # -----------------------------
    # Diagnostics
    # -----------------------------
    # print("\n=== SAVE_SLICES DIAGNOSTICS ===")
    # print(f"domain_min: {domain_min}, domain_max: {domain_max}, dom_extent: {dom_extent}")
    # print(f"partSize (L,W,H): {(L, W, H)}")
    # print(f"multipliers: -x={mxm}, +x={mxp}, -y={mym}, +y={myp}, -z={mzm}, +z={mzp}")
    # print(f"domain_min_rel (signed): {domain_min_rel}")
    # print(f"domain_max_rel: {domain_max_rel}")
    # print(f"veh_min: {veh_min}, veh_max: {veh_max}, veh_center: {veh_center}")

    # Vehicle-referenced "manual-style" windows
    # X window uses vehicle_min.x as the reference (like your manual slices assume ~0)
    x0 = float(veh_min[0] - UPSTREAM_L * L)
    x1 = float(veh_min[0] + DOWNSTREAM_L * L)

    # y centered around vehicle center (manual: symmetric around 0)
    y0 = float(veh_center[1] - SIDE_W * W)
    y1 = float(veh_center[1] + SIDE_W * W)

    # z from ground (veh_min.z) + 0.1 to ~1.9H above ground
    z0 = float(veh_min[2] + Z_START_OFFSET)
    z1 = float(veh_min[2] + HEIGHT_H * H)

    # print(f"Target windows (pre-clip): X[{x0:.6g},{x1:.6g}]  Y[{y0:.6g},{y1:.6g}]  Z[{z0:.6g},{z1:.6g}]")

    # Helpers
    def clamp01(v):
        return 0.0 if v < 0.0 else (1.0 if v > 1.0 else float(v))

    def clip_range(axis, a0, a1):
        if axis == "X":
            lo, hi = domain_min[0], domain_max[0]
        elif axis == "Y":
            lo, hi = domain_min[1], domain_max[1]
        else:
            lo, hi = domain_min[2], domain_max[2]
        return float(max(a0, lo)), float(min(a1, hi))

    def bounds_axis_aligned(axis, origin, width, height):
        ox, oy, oz = map(float, origin)
        eps = 1e-12
        if axis == "X":
            u0 = (oy - domain_min[1]) / (dom_extent[1] + eps)
            u1 = (oy + width - domain_min[1]) / (dom_extent[1] + eps)
            v0 = (oz - domain_min[2]) / (dom_extent[2] + eps)
            v1 = (oz + height - domain_min[2]) / (dom_extent[2] + eps)
        elif axis == "Y":
            u0 = (ox - domain_min[0]) / (dom_extent[0] + eps)
            u1 = (ox + width - domain_min[0]) / (dom_extent[0] + eps)
            v0 = (oz - domain_min[2]) / (dom_extent[2] + eps)
            v1 = (oz + height - domain_min[2]) / (dom_extent[2] + eps)
        else:  # Z
            u0 = (ox - domain_min[0]) / (dom_extent[0] + eps)
            u1 = (ox + width - domain_min[0]) / (dom_extent[0] + eps)
            v0 = (oy - domain_min[1]) / (dom_extent[1] + eps)
            v1 = (oy + height - domain_min[1]) / (dom_extent[1] + eps)

        u0, u1, v0, v1 = clamp01(u0), clamp01(u1), clamp01(v0), clamp01(v1)
        if (u1 - u0) <= 1e-6 or (v1 - v0) <= 1e-6:
            return None
        return (u0, u1, v0, v1)

    axis_to_normal = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}

    # -----------------------------
    # Main loop
    # -----------------------------
    outputSlices = jsonfile.get("outputSlices", [])
    tic = time.time()

    for slice_group in outputSlices:
        axis = slice_group.get("axis", "X")
        plane_normal = axis_to_normal.get(axis, (1, 0, 0))
        n = max(1, int(slice_group.get("numSlices", default_num_slices)))

        if axis == "X":
            # Sweep X, plane is YZ
            sx0, sx1 = clip_range("X", x0, x1)
            width = float(y1 - y0)
            height = float(z1 - z0)
            width_vec = np.array([0.0, 1.0, 0.0], dtype=float)
            height_vec = np.array([0.0, 0.0, 1.0], dtype=float)

            xs = np.linspace(sx0, sx1, n, dtype=float)
            origins = np.column_stack((xs, np.full(n, y0), np.full(n, z0)))

            print(f"\n--- X: sweep raw[{x0:.6g},{x1:.6g}] clipped[{sx0:.6g},{sx1:.6g}]  width={width:.6g} height={height:.6g}")

        elif axis == "Y":
            # Sweep Y, plane is XZ
            sy0, sy1 = clip_range("Y", y0, y1)
            width = float(x1 - x0)
            height = float(z1 - z0)
            width_vec = np.array([1.0, 0.0, 0.0], dtype=float)
            height_vec = np.array([0.0, 0.0, 1.0], dtype=float)

            ys = np.linspace(sy0, sy1, n, dtype=float)
            origins = np.column_stack((np.full(n, x0), ys, np.full(n, z0)))

            print(f"\n--- Y: sweep raw[{y0:.6g},{y1:.6g}] clipped[{sy0:.6g},{sy1:.6g}]  width={width:.6g} height={height:.6g}")

        else:  # Z
            # Sweep Z, plane is XY
            sz0, sz1 = clip_range("Z", z0, z1)
            width = float(x1 - x0)
            height = float(y1 - y0)
            width_vec = np.array([1.0, 0.0, 0.0], dtype=float)
            height_vec = np.array([0.0, 1.0, 0.0], dtype=float)

            zs = np.linspace(sz0, sz1, n, dtype=float)
            origins = np.column_stack((np.full(n, x0), np.full(n, y0), zs))

            print(f"\n--- Z: sweep raw[{z0:.6g},{z1:.6g}] clipped[{sz0:.6g},{sz1:.6g}]  width={width:.6g} height={height:.6g}")

        # write back to JSON for downstream use
        slice_group["width"] = width
        slice_group["height"] = height
        slice_group["widthVec"] = {"x": float(width_vec[0]), "y": float(width_vec[1]), "z": float(width_vec[2])}
        slice_group["heightVec"] = {"x": float(height_vec[0]), "y": float(height_vec[1]), "z": float(height_vec[2])}
        slice_group["origin"] = [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in origins]

        # print a few origins so you can verify quickly
        for i in range(min(10, origins.shape[0])):
            p = origins[i]
            print(f"  {axis}[{i:03d}] origin=({p[0]:.6g},{p[1]:.6g},{p[2]:.6g})")

        # render
        prefix = os.path.join(output_dir, f"{axis}_slice_")
        rendered = 0
        skipped = 0

        # Extract field data ONCE before parallelization
        avg_fields = h5exporter.finalize_time_average(keep_state=True)
        comp_keys = [k for k in avg_fields.keys() if k.startswith("velocity" + "_")]
        comp_keys = sorted(comp_keys, key=lambda k: int(k.split("_")[-1]))

        if len(comp_keys) > 1:
            comps = [avg_fields[k].astype(np.float64) for k in comp_keys]
            cell_data = np.sqrt(np.sum([c**2 for c in comps], axis=0)).astype(np.float32)
            field_name = "velocity_magnitude"
        else:
            cell_data = avg_fields[comp_keys[0]]
            field_name = comp_keys[0]

        # Apply normalization clipping (CRITICAL for correct color range)
        if normalize != 1.0:
            cell_data = np.clip(cell_data / normalize, 0, 1)

        # Coefficient fields (Cp, CpTotal): proper derived fields from the averaged
        # density/velocity, linearly mapped from [lo, hi] onto the colormap's [0, 1].
        # Rendered on the same averaged data / slice geometry as the velocity slices.
        coeff_slices = []  # list of (field_name, cell_data, cmap)
        for base_name, coeff_cmap, lo, hi in coeff_slice_specs:
            keys = sorted(
                [k for k in avg_fields.keys() if k.startswith(base_name + "_")],
                key=lambda k: int(k.split("_")[-1]),
            )
            if not keys:
                continue
            coeff_data = avg_fields[keys[0]].astype(np.float64)
            coeff_data = np.clip((coeff_data - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
            coeff_slices.append((base_name, coeff_data, coeff_cmap))

        max_workers = os.cpu_count() or 1

        def render_single_slice_unlocked(slice_info):
            idx, plane_point, plane_normal, output_filename = slice_info
            try:
                # Pin the image extent to the published slice rectangle so every
                # slice in this sweep renders on an identical pixel grid.
                h5exporter._to_slice_image_single_field(
                    f"{output_filename}_{field_name}",
                    cell_data,
                    plane_point,
                    plane_normal,
                    slice_thickness=voxel_size,
                    bounds=[0, 1, 0, 1],
                    grid_res=grid_res,
                    cmap=cmap,
                    show_axes=False,
                    show_colorbar=False,
                    normalize=normalize,
                    width=width,
                    height=height,
                    width_vec=width_vec,
                    height_vec=height_vec,
                    query_workers=1,  # one core per slice; pool parallelizes across slices
                )
                # Coefficient slices (Cp, CpTotal) on the identical pixel grid
                # (different values + filenames). Data is pre-clipped to [0, 1];
                # normalize != 1.0 locks the PNG color range to vmin=0, vmax=1.
                for coeff_name, coeff_data, coeff_cmap in coeff_slices:
                    h5exporter._to_slice_image_single_field(
                        f"{output_filename}_{coeff_name}",
                        coeff_data,
                        plane_point,
                        plane_normal,
                        slice_thickness=voxel_size,
                        bounds=[0, 1, 0, 1],
                        grid_res=grid_res,
                        cmap=coeff_cmap,
                        show_axes=False,
                        show_colorbar=False,
                        normalize=2.0,   # any value != 1.0 -> fixed [0,1] color range
                        width=width,
                        height=height,
                        width_vec=width_vec,
                        height_vec=height_vec,
                        query_workers=1,  # one core per slice; pool parallelizes across slices
                    )
                return (idx, True, None)
            except Exception as e:
                return (idx, False, str(e))

        slice_tasks = []
        for idx in range(origins.shape[0]):
            plane_point = origins[idx]
            output_filename = prefix + f"{idx:03d}"
            slice_tasks.append((idx, plane_point, plane_normal, output_filename))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(render_single_slice_unlocked, task): task[0] for task in slice_tasks}
            for future in as_completed(futures):
                idx, success, error = future.result()
                if success:
                    rendered += 1
                else:
                    skipped += 1
                    if error:
                        print(f"  Warning: slice {axis}[{idx:03d}] failed: {error}")

        wp.synchronize()
        print(f"Rendered {rendered}, skipped {skipped} for axis {axis}")

    print(f"\nTime to save all images {time.time() - tic} seconds.")


def save_slices0(output_dir, grid_shape_zip, shift, h5exporter, delta_x_coarse,voxel_size, jsonfile):
    domainSize = np.array(grid_shape_zip) * voxel_size
    outputSlices = jsonfile['outputSlices']
    # Map axis to plane normal
    axis_to_normal = {
        'X': [1, 0, 0],
        'Y': [0, 1, 0],
        'Z': [0, 0, 1]
    }
    domain_min = -shift
    domain_max = domain_min + domainSize
    tic = time.time()
    def compute_slice_bounds_relative_to_domain(origin, width, height, width_vec, height_vec, domain_min, domain_max, plane_normal):        
        # Build in-plane basis
        n = np.array(plane_normal) / np.linalg.norm(plane_normal)
        if np.allclose(n, [1, 0, 0]):
            u1 = np.array([0, 1, 0])
        else:
            u1 = np.array([1, 0, 0])
        u1 = u1 / np.linalg.norm(u1)
        u2 = np.cross(n, u1)
        u2 = u2 / np.linalg.norm(u2)
        width_vec_norm = width_vec / np.linalg.norm(width_vec)
        height_vec_norm = height_vec / np.linalg.norm(height_vec)
        if np.dot(u1, width_vec_norm) < 0:
            u1 = -u1
        if np.dot(u2, height_vec_norm) < 0:
            u2 = -u2

        # Use the lower-left corner of the slice as the reference point
        ref_point = origin

        # Project domain corners onto the plane and compute in-plane coordinates
        domain_corners = np.array([
            [domain_min[0], domain_min[1], domain_min[2]],
            [domain_max[0], domain_min[1], domain_min[2]],
            [domain_min[0], domain_max[1], domain_min[2]],
            [domain_max[0], domain_max[1], domain_min[2]],
            [domain_min[0], domain_min[1], domain_max[2]],
            [domain_max[0], domain_min[1], domain_max[2]],
            [domain_min[0], domain_max[1], domain_max[2]],
            [domain_max[0], domain_max[1], domain_max[2]]
        ])
        local_corners = []
        for corner in domain_corners:
            # Project corner onto the plane
            proj = corner - np.dot(corner - ref_point, n) * n
            local_x = np.dot(proj - ref_point, u1)
            local_y = np.dot(proj - ref_point, u2)
            local_corners.append([local_x, local_y])
        local_corners = np.array(local_corners)
        domain_u_min, domain_u_max = local_corners[:, 0].min(), local_corners[:, 0].max()
        domain_v_min, domain_v_max = local_corners[:, 1].min(), local_corners[:, 1].max()

        # Project slice corners onto the plane and compute in-plane coordinates
        slice_corners = [
            origin,
            origin + width * width_vec,
            origin + height * height_vec,
            origin + width * width_vec + height * height_vec
        ]
        slice_local = []
        for corner in slice_corners:
            proj = corner - np.dot(corner - ref_point, n) * n
            local_x = np.dot(proj - ref_point, u1)
            local_y = np.dot(proj - ref_point, u2)
            slice_local.append([local_x, local_y])
        slice_local = np.array(slice_local)
        slice_u_min, slice_u_max = slice_local[:, 0].min(), slice_local[:, 0].max()
        slice_v_min, slice_v_max = slice_local[:, 1].min(), slice_local[:, 1].max()

        # Convert to fractions
        u_min = (slice_u_min - domain_u_min) / (domain_u_max - domain_u_min)
        u_max = (slice_u_max - domain_u_min) / (domain_u_max - domain_u_min)
        v_min = (slice_v_min - domain_v_min) / (domain_v_max - domain_v_min)
        v_max = (slice_v_max - domain_v_min) / (domain_v_max - domain_v_min)

        return [max(0,u_min), min(1,u_max), max(0,v_min), min(1,v_max)]
    
    for slice_group in outputSlices:
        field_name = slice_group['field']
        axis = slice_group['axis']
        height = slice_group['height']
        width = slice_group['width']
    # Extract vectors
        height_vec = np.array([
            slice_group['heightVec']['x'],
            slice_group['heightVec']['y'],
            slice_group['heightVec']['z']
        ])
        width_vec = np.array([
            slice_group['widthVec']['x'],
            slice_group['widthVec']['y'],
            slice_group['widthVec']['z']
        ])
        
        # Get plane normal
        plane_normal = axis_to_normal[axis]
        
        # Process each origin
        for idx, origin_dict in enumerate(slice_group['origin']):
            # The origin / plane point is the lower-left corner of the slice
            plane_point = np.array([
                origin_dict['x'],
                origin_dict['y'],
                origin_dict['z']
            ])
            
            # Calculate bounds in model units
            
            # Calculate the bounds
            # Since we're given absolute dimensions, we need to compute
            # the bounds relative to the full domain extent in the plane
            # For now, we'll use bounds [0, 1, 0, 1] to capture the full slice
            # as defined by the width and height
            bounds = [0, 1, 0, 1]
            bounds_x, bounds_x2, bounds_y, bounds_y2 = compute_slice_bounds_relative_to_domain(plane_point, width, height, width_vec, height_vec, domain_min, domain_max, plane_normal)
            print(f'bounds {bounds_x}, {bounds_x2}, {bounds_y}, {bounds_y2}')
            # Alternatively, if you want to compute bounds relative to domain:
            # You would need to:
            # 1. Project domain extents onto the plane
            # 2. Calculate where this slice sits within those extents
            # 3. Set bounds accordingly
            
            # Generate output filename
            output_filename = os.path.join(
                output_dir,
                f"{axis}_slice_{idx:03d}"
            )
            
            print(f"Generating slice: {output_filename}")
            print(f"  Axis: {axis}, Normal: {plane_normal}")
            print(f"  Plane point: {plane_point}")
            print(f"  Width: {width}, Height: {height}")
            

            # h5exporter.to_slice_image(
            #     output_filename,
            #     {"velocity": sim.u},
            #     plane_point=plane_point,
            #     plane_normal=plane_normal,
            #     grid_res=jsonfile['settings']['grid_res'],
            #     bounds=(bounds_x, bounds_x2, bounds_y, bounds_y2),
            #     show_axes=False,
            #     show_colorbar=False,
            #     cmap=jsonfile['settings']['sliceColorMap'],
            #     normalize=jsonfile['InletBC']['x'] * jsonfile['settings']['sliceFactor'],
            #     slice_thickness=delta_x_coarse #needed when using model units                
            # )
            # output_filename = os.path.join(
            #     output_dir,
            #     f"{axis}_slice_{idx:03d}_avg"
            # )
            h5exporter.to_slice_image_time_average(
                output_filename,
                field_base_name="velocity",
                plane_point=plane_point,
                plane_normal=plane_normal,
                grid_res=jsonfile['settings']['grid_res'],
                bounds=(bounds_x, bounds_x2, bounds_y, bounds_y2),
                show_axes=False,
                show_colorbar=False,
                cmap=jsonfile['settings']['sliceColorMap'],
                normalize=jsonfile['InletBC']['x'] * jsonfile['settings']['sliceFactor'],
                slice_thickness=voxel_size, #finest-cell thickness; per-level scaling handled in mesher
                keep_state=True,
            )
            wp.synchronize()
    print(f"Time to save all images {time.time()-tic} seconds. ")
  
  
def solve(
        sim, 
        ulb,
        num_steps, 
        h5exporter, 
        output_dir, 
        grid_shape_zip,
        grid_shape_x_coarsest, 
        delta_x_coarse, 
        shift,
        momentum_transfer,
        wheel_momentum,
        reference_area,
        reference_area_physical,
        voxel_size,
        prescribed_velocity_phys,
        total_lattice_updates_per_step,
        jsonfile,
        partSize,
        surface_mesh_for_vtk
        ):
    
    # -------------------------- Simulation Loop --------------------------
    wp.synchronize()
    print(f"\n*******\nSolver Started\n*******\n")
    start_time = time.time()
    solve_start = start_time
    compute_time = 0.0
    steps_since_last_print = 0
    time_out = False
    drag_values = []
    
    # Calculate print and file output intervals
    print_interval = max(1, int(num_steps * (jsonfile['settings']['solutionPrintFreq'] / 100.0)))
    crossover_step = int(num_steps * (jsonfile['settings']['crossover'] / 100.0))
    file_output_interval_pre_crossover = max(1, int(crossover_step / jsonfile['settings']['preCrossover_frames'])) if jsonfile['settings']['preCrossover_frames'] > 0 else num_steps + 1
    file_output_interval_post_crossover = max(1, int((num_steps - crossover_step) / jsonfile['settings']['postCrossover_frames'])) if jsonfile['settings']['postCrossover_frames'] > 0 else num_steps + 1
    final_print_interval = max(1, int((num_steps-crossover_step) * (jsonfile['settings']['solutionPrintFreq']  / 100.0)))
    h5exporter.start_time_average()
    if jsonfile['settings']['debug']:
        for step in range(num_steps):
            solution_time =(time.time()-solve_start)/60
            step_start = time.time()
            sim.step()
            compute_time += time.time() - step_start
            steps_since_last_print += 1
            percent_complete = (step + 1) / num_steps * 100
            scm_progress(np.floor(percent_complete))
            end_time = time.time()
            elapsed = end_time - start_time
            
            #Assess Timeout            
            if elapsed/60 >= jsonfile['settings']['limit']:
                time_out = True

            print_output_interval = print_interval if step < crossover_step else final_print_interval
            if step % print_output_interval == 0 or step == num_steps - 1 or time_out:
                sim.macro(sim.f_0, sim.bc_mask, sim.rho, sim.u, streamId=0)                
                cd, cl, drag = print_lift_drag(sim, step, momentum_transfer, wheel_momentum, ulb, reference_area, voxel_size, drag_values)
                filename = os.path.join(output_dir, f"{jsonfile['outputName']}_{step:04d}")                
                h5exporter.to_slice_image(
                    filename,
                    {"velocity": sim.u},
                    plane_point=(1, 0, 0),
                    plane_normal=(0, 1, 0),
                    grid_res=jsonfile['settings']['grid_res'],
                    bounds=(0, 1, 0, 1),
                    show_axes=False,
                    show_colorbar=False,
                    slice_thickness=voxel_size, #finest-cell thickness; per-level scaling handled in mesher
                    normalize=jsonfile['InletBC']['x'] * jsonfile['settings']['sliceFactor'],
                )
                if step >= crossover_step:
                    # Derived fields (pressure/Cp/CpTotal) are synthesized per-step
                    # before averaging so nonlinear coefficients average correctly.
                    h5exporter.accumulate_time_average(
                        {"velocity": sim.u, "density": sim.rho},
                        weight=1.0,
                        derived=["pressure", "Cp", "CpTotal"],
                    )
                wp.synchronize()
                total_lattice_updates = total_lattice_updates_per_step * steps_since_last_print
                MLUPS = total_lattice_updates / compute_time / 1e6 if compute_time > 0 else 0.0
                current_flow_passes = step * ulb / grid_shape_x_coarsest
                remaining_steps = num_steps - step - 1
                time_remaining = 0.0 if MLUPS == 0 else (total_lattice_updates_per_step * remaining_steps) / (MLUPS * 1e6)
                hours, rem = divmod(time_remaining, 3600)
                minutes, seconds = divmod(rem, 60)
                time_remaining_str = f"{int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s"
                
                print(f"Completed step {step}/{num_steps} ({percent_complete:.2f}% complete)")
                print(f"  Flow Passes: {current_flow_passes:.2f}")
                print(f"  Time elapsed: {elapsed:.1f}s, Compute time: {compute_time:.1f}s, ETA: {time_remaining_str}")
                print(f"  MLUPS: {MLUPS:.1f}")
                print(f"  Cd={cd:.3f}, Cl={cl:.3f}, Drag Force (lattice units)={drag:.3f}")
                compute_time = 0.0
                steps_since_last_print = 0
                scm_results_available() 
                
                
            file_output_interval = file_output_interval_pre_crossover if step < crossover_step else file_output_interval_post_crossover
            if step % file_output_interval == 0 or step == num_steps - 1 or time_out:
                sim.macro(sim.f_0, sim.bc_mask, sim.rho, sim.u, streamId=0)
                filename = os.path.join(output_dir, f"{jsonfile['outputName']}_{step:04d}")
                # Derived fields (pressure [Pa], Cp, CpTotal) are synthesized from the
                # base velocity/density -- no extra NEON fields needed.
                h5exporter.to_hdf5(
                    filename,
                    {"velocity": sim.u, "density": sim.rho, "bc_mask": sim.bc_mask},
                    compression="gzip",
                    compression_opts=0,
                    derived=["pressure", "Cp", "CpTotal"],
                )
                wp.synchronize()
            if time_out:
                break
            
                
        # Save drag and lift data to CSV
        if len(drag_values) > 0:
            with open(os.path.join(output_dir, "drag_lift.csv"), 'w') as fd:
                fd.write("Step,Cd,Cl\n")
                for step, cd, cl in drag_values:
                    fd.write(f"{step},{cd},{cl}\n")

            plot_drag_lift(drag_values, output_dir, jsonfile['outputName'])

            # Calculate and print average Cd and Cl for the last cutover%
            drag_values_array = np.array(drag_values)
            # Filter to only include steps >= crossover_step
            mask = drag_values_array[:, 0] >= crossover_step
            post_crossover = drag_values_array[mask, :]
            if len(post_crossover) <=1:
                post_crossover = drag_values_array
            avg_cd = np.mean(post_crossover[:, 1])
            avg_cl = np.mean(post_crossover[:, 2])
            epsilon = 1e-8
            target_cd = jsonfile['vehicle']['targets']['cd'] + epsilon
            target_cl = jsonfile['vehicle']['targets']['cl'] + epsilon
            print(f"\nExperimental Drag Coefficient (Cd): {target_cd}\n" 
                f"Averages over last {100-jsonfile['settings']['crossover']}% of run:\n"
                f"Cd: {avg_cd:.4f}\n"
                f"Cl: {avg_cl:.4f}\n"
                f"CdA: {avg_cd*reference_area_physical:.4f}\n"
                f"ClA: {avg_cl*reference_area_physical:.4f}\n"
                f"Aero Power (kW): {0.5*jsonfile['fluid']['density'] * (prescribed_velocity_phys**3)*avg_cd*reference_area_physical /1000:.4f}\n"                
                f"Aero Power (hp): {0.5*jsonfile['fluid']['density'] * (prescribed_velocity_phys**3)*avg_cd*reference_area_physical /746:.4f}\n"                
                f"Error Drag Coefficient (Cd): {((avg_cd-target_cd)/target_cd)*100:.2f}%\n" 
                f"Error Lift Coefficient (Cl): {((avg_cl-target_cl)/target_cl)*100:.2f}%\n"
                )
            
            with open(os.path.join(output_dir, "project.log"),'a') as fd:
                fd.write(f"Averages over last {100-jsonfile['settings']['crossover']}% of run:\n")
                fd.write(f"Cd: {avg_cd:.4f}\n")
                fd.write(f"Cl: {avg_cl:.4f}\n")
                fd.write(f"CdA (m2): {avg_cd*reference_area_physical:.4f}\n")
                fd.write(f"ClA (m2): {avg_cl*reference_area_physical:.4f}\n")
                fd.write(f"Aero Power (kW): {0.5*jsonfile['fluid']['density'] * (prescribed_velocity_phys**3)*avg_cd*reference_area_physical/1000:.4f}\n")
                fd.write(f"Aero Power (hp): {0.5*jsonfile['fluid']['density'] * (prescribed_velocity_phys**3)*avg_cd*reference_area_physical / 746:.4f}\n")
                fd.write(f"Error Drag Coefficient (Cd): {((avg_cd-target_cd)/target_cd)*100:.2f}%\n")
                fd.write(f"Error Lift Coefficient (Cl): {((avg_cl-target_cl)/target_cl)*100:.2f}%\n")
                fd.write(f'Total Solution Time:     {(time.time()-solve_start)/60:.3f} min\n')
                
        save_slices(output_dir, grid_shape_zip, shift, h5exporter, delta_x_coarse, voxel_size,jsonfile, partSize) 
            
        filename = os.path.join(output_dir, f"{jsonfile['outputName']}_average")
        h5exporter.to_hdf5_time_average(filename, compression="gzip", compression_opts=0, keep_state=True)
        filename = os.path.join(output_dir, f"{jsonfile['outputName']}_average_density")
        h5exporter.to_surface_vtk_time_average(
            output_filename=filename,
            surface_mesh_filename=surface_mesh_for_vtk,
            field_base_name="density",
            component=None,
            keep_state=True,
            sample_dx=voxel_size,
            shell_factors=(2.20, ),
            k=8,
            power=2.0,
            max_distance=1.0 * voxel_size,
            half_space_tolerance=0.15,
            aggregate="median",
            smooth_iterations=2,
            smooth_relaxation=0.20,
        )

        filename = os.path.join(output_dir, f"{jsonfile['outputName']}_average_vel")
        h5exporter.to_surface_vtk_time_average(
            output_filename=filename,
            surface_mesh_filename=surface_mesh_for_vtk,
            field_base_name="velocity",
            component=None,                 # magnitude
            keep_state=True,
            sample_dx=voxel_size,
            shell_factors=(2.20,),
            k=8,
            power=2.0,
            max_distance=1.0 * voxel_size,
            half_space_tolerance=0.15,
            aggregate="median",
            smooth_iterations=2,
            smooth_relaxation=0.20,
        )

        # Surface pressure-coefficient maps on the body (Cp, CpTotal) from the
        # time-averaged derived fields accumulated during the run. field_base_name
        # selects the stored derived key ("Cp" -> Cp_0, "CpTotal" -> CpTotal_0).
        for surf_field in ("Cp", "CpTotal"):
            filename = os.path.join(output_dir, f"{jsonfile['outputName']}_average_{surf_field}")
            h5exporter.to_surface_vtk_time_average(
                output_filename=filename,
                surface_mesh_filename=surface_mesh_for_vtk,
                field_base_name=surf_field,
                component=None,
                keep_state=True,
                sample_dx=voxel_size,
                shell_factors=(2.20,),
                k=8,
                power=2.0,
                max_distance=1.0 * voxel_size,
                half_space_tolerance=0.15,
                aggregate="median",
                smooth_iterations=2,
                smooth_relaxation=0.20,
            )

        # Total-pressure-coefficient (CpT) iso-surface as STL (replaces the ParaView pipeline).
        # Limited to the same vehicle region swept for the result slices.
        try:
            cpt_region = vehicle_slice_region_bounds(jsonfile, partSize, shift, grid_shape_zip, voxel_size)
            h5exporter.to_cpt_isosurface_stl_time_average(
                output_filename=os.path.join(output_dir, f"{jsonfile['outputName']}_CpT_iso"),
                iso_value=0.0,
                bc_mask_neon=sim.bc_mask,
                u_inf=prescribed_velocity_phys,
                rho_air=jsonfile['fluid']['density'],
                p_inf=101325.0,
                keep_state=True,
                bounds=cpt_region,
                grid_resolution=1024,
            )
        except Exception as e:
            print(f"CpT iso-surface export failed: {e}")

        with open(os.path.join(output_dir, "source.json"), 'w') as file:
            json.dump(jsonfile, file, indent=4) # indent for pretty-printing
            print(f"Source Json written to {os.path.join(output_dir, 'source.json')} successfully.")
            
        scm_results_available(True)
    else:
        print_interval=max(1, int((num_steps-crossover_step) * (jsonfile['settings']['solutionPrintFreq'] / 100.0)))        
        for step in range(num_steps):
            sim.step()
            percent_complete = (step + 1) / num_steps * 100
            scm_progress(np.floor(percent_complete))
            end_time = time.time()
            elapsed = end_time - start_time
            if elapsed/60 >= jsonfile['settings']['limit']:
                time_out = True               
                
            if step % int(num_steps / 100) == 0:
                print(f"Step {step} completed out of {num_steps}") 

            if (step >= crossover_step and (step % print_interval == 0 or step == num_steps - 1)) or time_out:
                    print(f"Step {step} completed out of {num_steps}")
                    sim.macro(sim.f_0, sim.bc_mask, sim.rho, sim.u, streamId=0)                    
                    cd, cl, drag = print_lift_drag(sim, step, momentum_transfer, wheel_momentum, ulb, reference_area, voxel_size, drag_values)
                    # Derived fields (pressure/Cp/CpTotal) are synthesized per-step
                    # before averaging so nonlinear coefficients average correctly.
                    h5exporter.accumulate_time_average(
                        {"velocity": sim.u, "density": sim.rho},
                        weight=1.0,
                        derived=["pressure", "Cp", "CpTotal"],
                    )
                    wp.synchronize()
                    scm_results_available() 

            if time_out:
                with open(os.path.join(output_dir, "project.log"),'a') as fd:
                    fd.write(f"*** Solution Timed out ***\n")
                    fd.write(f"Actual iterations: {step}\n")
                print('Time limit reached')                
                break
                
        if (jsonfile['settings']['fullData']==True):   
            sim.macro(sim.f_0, sim.bc_mask, sim.rho, sim.u, streamId=0)
            #filename = os.path.join(output_dir, f"{jsonfile['outputName']}_{step:04d}")
            #h5exporter.to_hdf5(filename, {"velocity": sim.u, "density": sim.rho}, compression="gzip", compression_opts=0)
            filename = os.path.join(output_dir, f"{jsonfile['outputName']}_average")
            h5exporter.to_hdf5_time_average(filename, compression="gzip", compression_opts=0, keep_state=True)
            wp.synchronize()
            scm_results_available() 

        # Save drag and lift data to CSV
        if len(drag_values) > 0:
            with open(os.path.join(output_dir, "drag_lift.csv"), 'w') as fd:
                fd.write("Step,Cd,Cl\n")
                for step, cd, cl in drag_values:
                    fd.write(f"{step},{cd},{cl}\n")
            plot_drag_lift(drag_values, output_dir, jsonfile['outputName'])

            # Calculate and print average Cd and Cl after crossover step
            drag_values_array = np.array(drag_values)
            # Filter to only include steps >= crossover_step
            mask = drag_values_array[:, 0] >= crossover_step
            post_crossover = drag_values_array[mask, :]
            if len(post_crossover) <=1:
                post_crossover = drag_values_array
            avg_cd = np.mean(post_crossover[:, 1])
            avg_cl = np.mean(post_crossover[:, 2])
            print(f"\nAverages over last {100-jsonfile['settings']['crossover']}% of run:\n"
                f"Cd: {avg_cd:.4f}\n"
                f"Cl: {avg_cl:.4f}\n"
                f"CdA: {avg_cd*reference_area_physical:.4f}\n"
                f"ClA: {avg_cl*reference_area_physical:.4f}\n"
                f"Aero Power (kW): {0.5*jsonfile['fluid']['density'] * (prescribed_velocity_phys*prescribed_velocity_phys*prescribed_velocity_phys)*avg_cd*reference_area_physical /1000:.4f}\n"                
                f"Aero Power (hp): {0.5*jsonfile['fluid']['density'] * (prescribed_velocity_phys*prescribed_velocity_phys*prescribed_velocity_phys)*avg_cd*reference_area_physical /746:.4f}\n"                
                )
            
            with open(os.path.join(output_dir, "project.log"),'a') as fd:
                fd.write(f"Averages over last {100-jsonfile['settings']['crossover']}% of run:\n")
                fd.write(f"Cd: {avg_cd:.4f}\n")
                fd.write(f"Cl: {avg_cl:.4f}\n")
                fd.write(f"CdA: {avg_cd*reference_area_physical:.4f}\n")
                fd.write(f"ClA: {avg_cl*reference_area_physical:.4f}\n")
                fd.write(f"Aero Power (kW): {0.5*jsonfile['fluid']['density'] * (prescribed_velocity_phys*prescribed_velocity_phys*prescribed_velocity_phys)*avg_cd*reference_area_physical/1000:.4f}\n")
                fd.write(f"Aero Power (hp): {0.5*jsonfile['fluid']['density'] * (prescribed_velocity_phys*prescribed_velocity_phys*prescribed_velocity_phys)*avg_cd*reference_area_physical / 746:.4f}\n")
                fd.write(f'Total Solution Time:     {(time.time()-solve_start)/60:.3f} min\n')
        save_slices(output_dir, grid_shape_zip, shift, h5exporter, delta_x_coarse,voxel_size, jsonfile, partSize)

        # Total-pressure-coefficient (CpT) iso-surface as STL (replaces the ParaView pipeline).
        # Limited to the same vehicle region swept for the result slices.
        try:
            cpt_region = vehicle_slice_region_bounds(jsonfile, partSize, shift, grid_shape_zip, voxel_size)
            h5exporter.to_cpt_isosurface_stl_time_average(
                output_filename=os.path.join(output_dir, f"{jsonfile['outputName']}_CpT_iso"),
                iso_value=0.0,
                bc_mask_neon=sim.bc_mask,
                u_inf=prescribed_velocity_phys,
                rho_air=jsonfile['fluid']['density'],
                p_inf=101325.0,
                keep_state=True,
                bounds=cpt_region,
                grid_resolution=1024,
            )
        except Exception as e:
            print(f"CpT iso-surface export failed: {e}")

        with open(os.path.join(output_dir, "Results.json"), 'w') as file:
            json.dump({"outputSlices": jsonfile['outputSlices']}, file, indent=4) # indent for pretty-printing
            print(f"Source Json written to {os.path.join(output_dir, 'project.json')} successfully.")        
            
        scm_results_available(True)


def main(argv):
    """
    Main entry point for the Studio Wind Tunnel Solver.

    Parses command-line arguments to obtain the input JSON file, initializes the simulation environment,
    cleans up previous output files, and runs the wind tunnel simulation. Handles errors and reports
    progress and completion status via SCM events.

    Args:
        argv (list): List of command-line arguments.

    Returns:
        int: Exit code. Returns 0 on success, 64 on argument/input errors, or 1 on simulation failure.
    """

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    input_file = ''
    usage = 'windtunnel_json.py -i <inputjson>'

    logging.info('Welcome to Studio Wind Tunnel Solver')  

    try:
        opts, _ = getopt.getopt(argv, "hi:o:", ["ifile="])
    except getopt.GetoptError:
        logging.error(usage)
        scm_set_error(64, 'Argument error')
        return 64

    for opt, arg in opts:
        if opt == '-h':
            logging.info(usage)
            return 64

        if opt in ("-i", "--ifile"):
            input_file = arg

    if not input_file:
        logging.error('Error: Input JSON file must be specified.\n' + usage)
        scm_set_error(64, 'Input file not specified')
        return 64

    try:
        if running_via_scm():
            log_file_scm = os.path.join(os.path.dirname(os.path.abspath(input_file)), 'solve.log')
            scm_log_handler = logging.FileHandler(log_file_scm, mode='w')
            scm_log_handler.setLevel(logging.INFO)
            scm_log_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
            logging.getLogger().addHandler(scm_log_handler)
            logging.info('SCM Log file: {}'.format(log_file_scm))

        logging.info('Input file: {}'.format(input_file))

        scm_init()
        
        prep_inputs(input_file)

        scm_complete()
    except Exception as e:
        logging.error(f'Exception occured: {e}')
        scm_set_error(1, f'Job failed: {e}')
        scm_cancel_heartbeat()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
