"""
Geometry-only variant of windtunnel_json.py: same input json, same output files, no solve.

It loads the CAD, runs the surfaceRemesh (wrap / resample / curvature-adaptive / snap) exactly like the solver does,
then writes the surface-field .usda with a DUMMY field in place of the mapped solution, plus Results.json with
placeholder numbers, so the downstream load (Alias / VRED) behaves as it does after a real run.

    XLB_FAST_MATH=1 python3 examples/windtunnel_geom_json.py -i examples/cfd/stl-files/project.json

Nothing here touches the solver: no voxelisation, grid, boundary conditions, time stepping or field mapping.
The dummy field is "front-facing = high, rear-facing = low" (from the outward vertex normals), so a wrongly
oriented patch shows up as inverted colour. Cd/Cl etc. in Results.json are the json's vehicle targets if present
(else 0) and are NOT results. Slice images listed in outputSlices are not produced.
"""
from __future__ import annotations
import os
import sys
import json
import time
import getopt
import logging

import numpy as np
import trimesh

# the existing solver script is imported, never modified: its OBJ converter and SCM/progress helpers are reused
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import windtunnel_json as wt


def build_surface_mesh(jsonfile, proj_path, output_dir, voxel_size):
    """Load body + wheels (OBJ in cm is converted to an m STL) and return the surface-field mesh, remeshed if enabled."""
    body_file = os.path.join(proj_path, str(jsonfile['vehicle']['body'][0]))
    filename, file_extension = os.path.splitext(body_file)
    print(f' STL path {body_file}')
    print(' Loading STL....')
    if file_extension == '.obj':
        print(' Loaded obj scaling to STL....')
        stl_path = os.path.join(output_dir, filename + '.stl')
        wt.obj_to_binary_stl_stream(body_file, stl_path, scale=0.01, assume_triangular_faces=True, compute_normals=False)
        body_mesh = trimesh.load_mesh(stl_path, process=False)
    else:
        body_mesh = trimesh.load_mesh(body_file, process=False)
    print(' Body Loaded....')

    if len(jsonfile['vehicle']['wheels']) > 0:
        print(' Loading Wheels...')
        wheel_meshes = []
        for w, wheel in enumerate(jsonfile['vehicle']['wheels'], start=1):
            wheel = os.path.join(proj_path, wheel)
            if file_extension == '.obj':
                stl_path = os.path.join(output_dir, 'wheel' + str(w) + '.stl')
                wt.obj_to_binary_stl_stream(wheel, stl_path, scale=0.01, assume_triangular_faces=True, compute_normals=False)
                wheel_meshes.append(trimesh.load_mesh(stl_path))
            else:
                wheel_meshes.append(trimesh.load_mesh(wheel, process=False))
        print(' Wheels Loaded....')
        car_mesh = trimesh.util.concatenate([body_mesh] + wheel_meshes)
    else:
        car_mesh = body_mesh.copy()

    surface_mesh = car_mesh.copy() if jsonfile.get("settings", {}).get("surfaceFieldScope", "car") == "car" else body_mesh.copy()

    # --- same remesh block as windtunnel_json.prep_inputs ---
    remesh_cfg = jsonfile.get("settings", {}).get("surfaceRemesh", {})
    if remesh_cfg.get("enabled", False) and str(jsonfile.get("settings", {}).get("surfaceField", "")).strip():
        from xlb.utils.surface_remesh import remesh_surface_isolated, adaptive_kwargs
        try:
            wrap_cfg = remesh_cfg.get("wrap", {})
            wrap_on = wrap_cfg.get("enabled", False)
            surface_mesh = remesh_surface_isolated(
                surface_mesh,
                target_edge=remesh_cfg.get("targetEdge", None if wrap_on else voxel_size),
                max_faces=remesh_cfg.get("maxFaces"),
                wrap_resolution=wrap_cfg.get("resolution", voxel_size / 2) if wrap_on else None,
                wrap_offset=wrap_cfg.get("offset") if wrap_on else None,
                gap_closure=wrap_cfg.get("gapClosure", 0.0) if wrap_on else 0.0,
                snap_offset=(lambda v: float(v) if v is not None and v is not False and float(v) >= 0 else None)(
                    wrap_cfg.get("snapOffset")) if wrap_on else None,
                relax_iters=int(wrap_cfg.get("relaxIterations", 3)) if wrap_on else 0,
                use_gpu=bool(remesh_cfg.get("useGpu", True)),
                # finer triangles where the wrapped surface curves tightly (see surfaceRemesh.curvatureAdaptive)
                **adaptive_kwargs(remesh_cfg.get("curvatureAdaptive")),
            )
            if wrap_on:
                surface_mesh.metadata["outward_normals"] = True
        except Exception as e:
            print(f" WARNING: surfaceRemesh failed ({e}); using the original surface mesh. "
                  f"(missing dependency? pip install pyacvd)")
    return surface_mesh


def dummy_surface_field(mesh, clim):
    """Front-facing high, rear-facing low, in [clim[0], clim[1]]; also a quick visual check of the normals."""
    n = np.asarray(mesh.vertex_normals, dtype=np.float64)
    t = 0.5 - 0.5 * n[:, 0]            # flow is +x: a normal facing -x (upstream) is the stagnation side
    return (clim[0] + (clim[1] - clim[0]) * t).astype(np.float32)


def run(input_file):
    t0 = time.time()
    with open(input_file) as f:
        jsonfile = json.load(f)
    proj_path = os.path.dirname(os.path.abspath(input_file))
    jsonfile['projPath'] = proj_path
    settings = jsonfile['settings']
    voxel_size = settings['voxelSize']

    if wt.running_via_scm():
        output_dir = proj_path
    else:
        output_dir = os.path.join(proj_path, jsonfile['outputName'])
        os.makedirs(output_dir, exist_ok=True)
        for fx in [os.path.join(output_dir, f) for f in os.listdir(output_dir)]:
            os.remove(fx)
    with open(os.path.join(output_dir, "project.log"), 'w') as fd:
        fd.write("***  Studio Wind Tunnel Solver Log File (GEOMETRY ONLY - no solve, dummy field) ***\n\n")

    surface_mesh = build_surface_mesh(jsonfile, proj_path, output_dir, voxel_size)
    wt.scm_progress(50)
    print("Progress 50%")

    # --- surface field USD, same name / colour range as the solver writes ---
    surface_field = settings.get("surfaceField", "")
    if isinstance(surface_field, str) and surface_field.strip():
        cmap = settings['surfaceFieldColorMap']
        if surface_field == "velocity":
            clim = (0.0, jsonfile['InletBC']['x'] * jsonfile['slices']['velocityFactor'])
        else:
            clim = (settings['surfaceFieldMin'], settings['surfaceFieldMax'])
        usd_path = os.path.join(output_dir, f"{jsonfile['outputName']}_average_{surface_field}_{cmap}.usda")
        values = dummy_surface_field(surface_mesh, clim)

        from xlb.utils.mesher import MultiresIO
        writer = MultiresIO.__new__(MultiresIO)          # the USD writer only uses static helpers; no grid needed
        writer._write_polydata_usd(
            usd_path,
            np.asarray(surface_mesh.vertices, dtype=np.float32),
            np.asarray(surface_mesh.faces, dtype=np.int32),
            point_data={surface_field: values},
            cell_data=None,
            color_field=surface_field,
            cmap=cmap,
            clim=clim,
        )
        wt.scm_results_available()

    # --- placeholder Results.json (same structure as the solver's) ---
    targets = jsonfile.get('vehicle', {}).get('targets', {})
    cd = float(targets.get('cd', 0.0))
    cl = float(targets.get('cl', 0.0))
    results = {'cd': cd, 'avg_cl': cl, 'cda': 0.0, 'cla': 0.0, 'aero_power_kW': 0.0, 'aero_power_hp': 0.0}
    with open(os.path.join(output_dir, "Results.json"), 'w') as fh:
        json.dump({"results": results, "outputName": jsonfile['outputName'],
                   "outputSlices": jsonfile.get('outputSlices', [])}, fh, indent=4)
        print(f"Results Json written to {os.path.join(output_dir, 'Results.json')} (placeholder values)")
    print(f"Geometry-only run finished in {time.time() - t0:.1f} s")
    print("Files in the output folder (the .usda references its *_cmap.png by relative path, so both must be delivered):")
    for fn in sorted(os.listdir(output_dir)):
        fp = os.path.join(output_dir, fn)
        if os.path.isfile(fp):
            print(f"    {fn}  ({os.path.getsize(fp):,} bytes)")
    wt.scm_progress(95)
    wt.scm_results_available(True)


def main(argv):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])
    usage = 'windtunnel_geom_json.py -i <inputjson>'
    input_file = ''
    try:
        opts, _ = getopt.getopt(argv, "hi:o:", ["ifile="])
    except getopt.GetoptError:
        logging.error(usage)
        wt.scm_set_error(64, 'Argument error')
        return 64
    for opt, arg in opts:
        if opt == '-h':
            logging.info(usage)
            return 64
        if opt in ("-i", "--ifile"):
            input_file = arg
    if not input_file:
        logging.error('Error: Input JSON file must be specified.\n' + usage)
        wt.scm_set_error(64, 'Input file not specified')
        return 64
    try:
        if wt.running_via_scm():
            h = logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(input_file)), 'solve.log'), mode='w')
            h.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
            logging.getLogger().addHandler(h)
        logging.info('Geometry-only run, input file: {}'.format(input_file))
        wt.scm_init()
        run(input_file)
        wt.scm_complete()
    except Exception as e:
        logging.error(f'Exception occured: {e}')
        wt.scm_set_error(1, f'Job failed: {e}')
        wt.scm_cancel_heartbeat()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
