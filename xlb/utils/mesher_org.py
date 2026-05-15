import numpy as np
import trimesh
from typing import Any, Optional, Dict
import time
import neon
import warp as wp
from xlb.utils.utils import UnitConvertor
from scipy.spatial import cKDTree
import gc


def adjust_bbox(cuboid_max, cuboid_min, voxel_size_up):
    """
    Adjust the bounding box to the nearest points of one level finer grid that encloses the desired region.

    Args:
        cuboid_min (np.ndarray): Desired minimum coordinates of the bounding box.
        cuboid_max (np.ndarray): Desired maximum coordinates of the bounding box.
        voxel_size_up (float): Voxel size of one level higher (finer) grid.

    Returns:
        tuple: (adjusted_min, adjusted_max) snapped to grid points of one level higher.
    """
    adjusted_min = np.round(cuboid_min / voxel_size_up) * voxel_size_up
    adjusted_max = np.round(cuboid_max / voxel_size_up) * voxel_size_up
    return adjusted_min, adjusted_max


def prepare_sparsity_pattern(level_data):
    """
    Prepare the sparsity pattern for the multiresolution grid based on the level data. "level_data" is expected to be formatted as in
    the output of "make_cuboid_mesh".
    """
    num_levels = len(level_data)
    level_origins = []
    sparsity_pattern = []
    for lvl in range(num_levels):
        # Get the level mask from the level data
        level_mask = level_data[lvl][0]

        # Ensure level_0 is contiguous int32
        level_mask = np.ascontiguousarray(level_mask, dtype=np.int32)

        # Append the padded level mask to the sparsity pattern
        sparsity_pattern.append(level_mask)

        # Get the origin for this level
        level_origins.append(level_data[lvl][2])

    return sparsity_pattern, level_origins


def make_cuboid_mesh(voxel_size, cuboids, stl_filename):
    """
    Create a strongly-balanced multi-level cuboid mesh with a sequence of bounding boxes.
    Outputs mask arrays that are set to True only in regions not covered by finer levels.

    Args:
        voxel_size (float): Voxel size of the finest grid .
        cuboids (list): List of multipliers defining each level's domain.
        stl_name (str): Path to the STL file.

    Returns:
        list: Level data with mask arrays, voxel sizes, origins, and levels.
    """
    # Load the mesh and get its bounding box
    mesh = trimesh.load_mesh(stl_filename, process=False)
    assert not mesh.is_empty, ValueError("Loaded mesh is empty or invalid.")

    mesh_vertices = mesh.vertices
    min_bound = mesh_vertices.min(axis=0)
    max_bound = mesh_vertices.max(axis=0)
    partSize = max_bound - min_bound

    level_data = []
    adjusted_bboxes = []
    max_voxel_size = voxel_size * pow(2, (len(cuboids) - 1))
    # Step 1: Generate all levels and store their data
    for level in range(len(cuboids)):
        # Compute desired bounding box for this level
        cuboid_min = np.array(
            [
                min_bound[0] - cuboids[level][0] * partSize[0],
                min_bound[1] - cuboids[level][2] * partSize[1],
                min_bound[2] - cuboids[level][4] * partSize[2],
            ],
            dtype=float,
        )

        cuboid_max = np.array(
            [
                max_bound[0] + cuboids[level][1] * partSize[0],
                max_bound[1] + cuboids[level][3] * partSize[1],
                max_bound[2] + cuboids[level][5] * partSize[2],
            ],
            dtype=float,
        )

        # Set voxel size for this level
        voxel_size_level = max_voxel_size / pow(2, level)

        # Adjust bounding box to align with one level up (finer grid)
        if level > 0:
            voxel_level_up = max_voxel_size / pow(2, level - 1)
        else:
            voxel_level_up = voxel_size_level
        adjusted_min, adjusted_max = adjust_bbox(cuboid_max, cuboid_min, voxel_level_up)

        xmin, ymin, zmin = adjusted_min
        xmax, ymax, zmax = adjusted_max

        # Compute number of voxels based on level-specific voxel size
        nx = int(np.round((xmax - xmin) / voxel_size_level))
        ny = int(np.round((ymax - ymin) / voxel_size_level))
        nz = int(np.round((zmax - zmin) / voxel_size_level))
        print(f"Domain {nx}, {ny}, {nz}  Origin {adjusted_min}  Voxel Size {voxel_size_level} Voxel Level Up {voxel_level_up}")

        voxel_matrix = np.ones((nx, ny, nz), dtype=bool)

        origin = adjusted_min
        level_data.append((voxel_matrix, voxel_size_level, origin, level))
        adjusted_bboxes.append((adjusted_min, adjusted_max))

    # Step 2: Adjust coarser levels to exclude regions covered by finer levels
    for k in range(len(level_data) - 1):  # Exclude the finest level
        # Current level's data
        voxel_matrix_k = level_data[k][0]
        origin_k = level_data[k][2]
        voxel_size_k = level_data[k][1]
        nx, ny, nz = voxel_matrix_k.shape

        # Next finer level's bounding box
        adjusted_min_k1, adjusted_max_k1 = adjusted_bboxes[k + 1]

        # Compute index ranges in level k that overlap with level k+1's bounding box
        # Use epsilon (1e-10) to handle floating-point precision
        i_start = max(0, int(np.ceil((adjusted_min_k1[0] - origin_k[0] - 1e-10) / voxel_size_k)))
        i_end = min(nx, int(np.floor((adjusted_max_k1[0] - origin_k[0] + 1e-10) / voxel_size_k)))
        j_start = max(0, int(np.ceil((adjusted_min_k1[1] - origin_k[1] - 1e-10) / voxel_size_k)))
        j_end = min(ny, int(np.floor((adjusted_max_k1[1] - origin_k[1] + 1e-10) / voxel_size_k)))
        k_start = max(0, int(np.ceil((adjusted_min_k1[2] - origin_k[2] - 1e-10) / voxel_size_k)))
        k_end = min(nz, int(np.floor((adjusted_max_k1[2] - origin_k[2] + 1e-10) / voxel_size_k)))

        # Set overlapping region to zero
        voxel_matrix_k[i_start:i_end, j_start:j_end, k_start:k_end] = 0

    # Step 3 Convert to Indices from STL units
    num_levels = len(level_data)
    level_data = [(dr, int(v / voxel_size), np.round(dOrigin / v).astype(int), num_levels - 1 - l) for dr, v, dOrigin, l in level_data]

    return list(reversed(level_data))


class MultiresIO(object):
    def __init__(
        self,
        field_name_cardinality_dict,
        levels_data,
        unit_convertor: UnitConvertor = None,
        offset: Optional[tuple] = (0.0, 0.0, 0.0),
        store_precision=None,
    ):
        """
        Initialize the MultiresIO object.

        Parameters
        ----------
        field_name_cardinality_dict : dict
            A dictionary mapping field names to their cardinalities.
            Example: {'velocity_x': 1, 'velocity_y': 1, 'velocity': 3, 'density': 1}
        levels_data : list of tuples
            Each tuple contains (data, voxel_size, origin, level).
        unit_convertor : UnitConvertor
            An instance of the UnitConvertor class for unit conversions.
        offset : tuple, optional
            Offset to be applied to the coordinates.
        store_precision : str, optional
            The precision policy for storing data.
        """
        # Set the unit convertor object
        self.unit_convertor = unit_convertor
        start_time = time.time()
        # Process the multires geometry and extract coordinates and connectivity in the coordinate system of the finest level
        coordinates, connectivity, level_id_field, total_cells = self.process_geometry(levels_data)

        # Ensure that coordinates and connectivity are not empty
        assert coordinates.size != 0, "Error: No valid data to process. Check the input levels_data."

        # Merge duplicate points
        coordinates, connectivity = self._merge_duplicates(coordinates, connectivity, levels_data)

        # Transform coordinates to physical units and apply offset if provided
        coordinates = self._transform_coordinates(coordinates, offset)

        # Assign to self
        self.field_name_cardinality_dict = field_name_cardinality_dict
        self.levels_data = levels_data
        self.coordinates = coordinates
        self.connectivity = connectivity
        self.level_id_field = level_id_field
        self.total_cells = total_cells
        self.centroids = np.mean(coordinates[connectivity], axis=1)
        self.kd_tree = cKDTree(self.centroids)
        

        # Set the default precision policy if not provided
        from xlb import DefaultConfig

        if store_precision is None:
            self.store_precision = DefaultConfig.default_precision_policy.store_precision
            self.store_dtype = DefaultConfig.default_precision_policy.store_precision.wp_dtype

        # Prepare and allocate the inputs for the NEON container
        self.field_warp_dict, self.origin_list = self._prepare_container_inputs()

        # Construct the NEON container for exporting multi-resolution data
        self.container = self._construct_neon_container()
        # ---- Time-averaging state ----
        self._avg_sum: Dict[str, np.ndarray] = {}
        self._avg_weight: float = 0.0
        self._avg_active: bool = False
        # Cached finalized averages (to avoid re-dividing for every export)
        self._avg_final_cache: Optional[Dict[str, np.ndarray]] = None
        self._avg_cache_weight: float = 0.0
        self._avg_cache: bool = False

        # Optional cache for derived quantities (e.g. velocity_magnitude)
        self._avg_derived_cache: Dict[str, np.ndarray] = {}

        print(f"MutliResIO initialized in {time.time()-start_time}sec ")

    def process_geometry(self, levels_data):
        num_voxels_per_level = [np.sum(data) for data, _, _, _ in levels_data]
        num_points_per_level = [8 * nv for nv in num_voxels_per_level]
        point_id_offsets = np.cumsum([0] + num_points_per_level[:-1])

        all_corners = []
        all_connectivity = []
        level_id_field = []
        total_cells = 0

        for level_idx, (data, voxel_size, origin, level) in enumerate(levels_data):
            origin = origin * voxel_size
            corners_list, conn_list = self._process_level(data, voxel_size, origin, point_id_offsets[level_idx])

            if corners_list:
                print(f"\tProcessing level {level}: Voxel size {voxel_size}, Origin {origin}, Shape {data.shape}")
                all_corners.extend(corners_list)
                all_connectivity.extend(conn_list)
                num_cells = sum(c.shape[0] for c in conn_list)
                level_id_field.extend([level] * num_cells)
                total_cells += num_cells
            else:
                print(f"\tSkipping level {level} (no unique data)")

        # Stacking coordinates and connectivity
        coordinates = np.concatenate(all_corners, axis=0).astype(np.float32)
        connectivity = np.concatenate(all_connectivity, axis=0).astype(np.int32)
        level_id_field = np.array(level_id_field, dtype=np.uint8)

        return coordinates, connectivity, level_id_field, total_cells

    def _process_level(self, data, voxel_size, origin, point_id_offset):
        """
        Given a voxel grid, returns all corners and connectivity in NumPy for this resolution level.
        """
        true_indices = np.argwhere(data)
        if true_indices.size == 0:
            return [], []

        max_voxels_per_chunk = 268_435_450
        chunks = np.array_split(true_indices, max(1, (len(true_indices) + max_voxels_per_chunk - 1) // max_voxels_per_chunk))

        all_corners = []
        all_connectivity = []
        pid_offset = point_id_offset

        for chunk in chunks:
            if chunk.size == 0:
                continue
            corners, connectivity = self._process_voxel_chunk(chunk, np.asarray(origin, dtype=np.float32), voxel_size, pid_offset)
            all_corners.append(corners)
            all_connectivity.append(connectivity)
            pid_offset += len(chunk) * 8

        return all_corners, all_connectivity

    def _process_voxel_chunk(self, true_indices, origin, voxel_size, point_id_offset):
        """
        Given a set of voxel indices, returns 8 corners and connectivity for each cube using NumPy.
        """
        true_indices = np.asarray(true_indices, dtype=np.float32)
        mins = origin + true_indices * voxel_size
        offsets = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ],
            dtype=np.float32,
        )

        corners = (mins[:, None, :] + offsets[None, :, :] * voxel_size).reshape(-1, 3).astype(np.float32)
        base_ids = point_id_offset + np.arange(len(true_indices), dtype=np.int32) * 8
        connectivity = (base_ids[:, None] + np.arange(8, dtype=np.int32)).astype(np.int32)

        return corners, connectivity

    def save_xdmf(self, h5_filename, xmf_filename, total_cells, num_points, fields={}):
        # Generate an XDMF file to accompany the HDF5 file
        print(f"\tGenerating XDMF file: {xmf_filename}")
        hdf5_rel_path = h5_filename.split("/")[-1]
        with open(xmf_filename, "w") as xmf:
            xmf.write(f'''<?xml version="1.0" ?>
    <!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
    <Xdmf Version="3.0">
        <Domain>
            <Grid Name="VoxelMesh" GridType="Uniform">
                <Topology TopologyType="Hexahedron" NumberOfElements="{total_cells}">
                    <DataItem Dimensions="{total_cells} 8" NumberType="Int" Format="HDF">
                        {hdf5_rel_path}:/Mesh/Connectivity
                    </DataItem>
                </Topology>
                <Geometry GeometryType="XYZ">
                    <DataItem Dimensions="{num_points} 3" NumberType="Float" Precision="4" Format="HDF">
                        {hdf5_rel_path}:/Mesh/Points
                    </DataItem>
                </Geometry>
                <Attribute Name="Level" AttributeType="Scalar" Center="Cell">
                    <DataItem Dimensions="{total_cells}" NumberType="UInt8" Format="HDF">
                        {hdf5_rel_path}:/Mesh/Level
                    </DataItem>
                </Attribute>
        ''')
            for field_name in fields.keys():
                xmf.write(f'''
            <Attribute Name="{field_name}" AttributeType="Scalar" Center="Cell">
                <DataItem Dimensions="{total_cells}" NumberType="Float" Precision="4" Format="HDF">
                {hdf5_rel_path}:/Fields/{field_name}
                </DataItem>
            </Attribute>
            ''')
            xmf.write("""
                </Grid>
            </Domain>
        </Xdmf>
        """)
        print("\tXDMF file written successfully")
        return

    def save_hdf5_file(self, filename, coordinates, connectivity, level_id_field, fields_data, compression="gzip", compression_opts=0):
        """Write the processed mesh data to an HDF5 file.
        Parameters
        ----------
        filename : str
            The name of the output HDF5 file.
        coordinates : numpy.ndarray
            An array of all coordinates.
        connectivity : numpy.ndarray
            An array of all connectivity data.
        level_id_field : numpy.ndarray
            An array of all level data.
        fields_data : dict
            A dictionary of all field data.
        compression : str, optional
            The compression method to use for the HDF5 file.
        compression_opts : int, optional
            The compression options to use for the HDF5 file.
        """
        import h5py

        with h5py.File(filename + ".h5", "w") as f:
            f.create_dataset("/Mesh/Points", data=coordinates, compression=compression, compression_opts=compression_opts, chunks=True)
            f.create_dataset(
                "/Mesh/Connectivity",
                data=connectivity,
                compression=compression,
                compression_opts=compression_opts,
                chunks=True,
            )
            f.create_dataset("/Mesh/Level", data=level_id_field, compression=compression, compression_opts=compression_opts)
            fg = f.create_group("/Fields")
            for fname, fdata in fields_data.items():
                fg.create_dataset(fname, data=fdata.astype(np.float32), compression=compression, compression_opts=compression_opts, chunks=True)

    def _merge_duplicates0(self, coordinates, connectivity, levels_data):
        """Merge duplicate points using np.unique on quantized int32 grid.
        Optimal balance for your 362M-point AMR mesh (finest grid (4256,2432,1280)).
        """
        
        tic = time.perf_counter()
        n = coordinates.shape[0]
        finest_voxel = float(min(voxel_size for (_, voxel_size, _, _) in levels_data))
        print(f"\tMerging duplicates from {n:,} points...")

        # Quantize directly to int32 — guaranteed safe for your grid
        inv = 1.0 / finest_voxel
        qi = np.rint(coordinates * inv).astype(np.int32)          # ~4.3 GB temp

        # np.unique does everything we need in one optimized call
        _, unique_idx, inverse = np.unique(
            qi, axis=0, return_index=True, return_inverse=True
        )

        unique_coordinates = coordinates[unique_idx].astype(np.float32, copy=False)
        new_connectivity = inverse.astype(np.int32)[connectivity]   # safe cast

        # Aggressive cleanup — important at this scale
        del qi, inverse
        gc.collect()

        reduction = 100.0 * (1.0 - len(unique_coordinates) / n)
        toc = time.perf_counter()
        print(f"\tMerged to {len(unique_coordinates):,} unique points "
            f"({reduction:.1f}% reduction) in {toc - tic:.2f} seconds")

        return unique_coordinates, new_connectivity

    def _merge_duplicates(self, coordinates, connectivity, levels_data):
        """
        Merge duplicate points using parallel sorting.
        ~3-4x faster than np.unique for large arrays.
        """
        import time
        tic = time.perf_counter()
        
        num_points = coordinates.shape[0]
        finest_voxel = min(voxel_size for (_, voxel_size, _, _) in levels_data)
        print(f"\tMerging duplicates from {num_points:,} points...")
        
        # Quantize coordinates to finest voxel grid
        inv = 1.0 / finest_voxel
        qi = np.rint(coordinates * inv).astype(np.int64)  # Removed unnecessary float64 cast
        
        # Compute tight bounding box
        mins = qi.min(axis=0)
        spans = qi.max(axis=0) - mins + 1
        shifts = qi - mins
        
        # Create hash keys
        hash_keys = shifts[:, 0] + spans[0] * (shifts[:, 1] + spans[1] * shifts[:, 2])        
        
        # Use stable sort (preserves first occurrence ordering)
        sort_idx = np.argsort(hash_keys, kind='stable')
        sorted_keys = hash_keys[sort_idx]
        
        # Find unique boundaries (vectorized)
        unique_mask = np.concatenate([[True], sorted_keys[1:] != sorted_keys[:-1]])
        unique_indices = sort_idx[unique_mask]
        
        # Build inverse mapping efficiently
        unique_ranks = np.cumsum(unique_mask) - 1
        inverse_indices = np.empty(num_points, dtype=np.int32)
        inverse_indices[sort_idx] = unique_ranks
        
        # Preserve original coordinates
        unique_coordinates = coordinates[unique_indices]
        new_connectivity = inverse_indices[connectivity]
        
        toc = time.perf_counter()
        reduction = 100 * (1 - len(unique_coordinates) / num_points)
        print(f"\tMerged to {len(unique_coordinates):,} unique points ({reduction:.1f}% reduction) in {toc - tic:.2f} seconds")
        
        return unique_coordinates, new_connectivity

    def _transform_coordinates(self, coordinates, offset):
        offset = np.array(offset, dtype=np.float32)
        if self.unit_convertor is not None:
            coordinates = self.unit_convertor.length_to_physical(coordinates)
        return coordinates + offset

    def _prepare_container_inputs(self):
        # load necessary modules
        from xlb.compute_backend import ComputeBackend
        from xlb.grid import grid_factory

        # Get the number of levels from the levels_data
        num_levels = len(self.levels_data)

        # Prepare lists to hold warp fields and origins allocated for each level
        field_warp_dict = {}
        origin_list = []
        for field_name, cardinality in self.field_name_cardinality_dict.items():
            field_warp_dict[field_name] = []
            for level in range(num_levels):
                # get the shape of the grid at this level
                box_shape = self.levels_data[level][0].shape

                # Use the warp backend to create dense fields to be written in multi-res NEON fields
                grid_dense = grid_factory(box_shape, compute_backend=ComputeBackend.WARP)
                field_warp_dict[field_name].append(grid_dense.create_field(cardinality=cardinality, dtype=self.store_precision))
                origin_list.append(wp.vec3i(*([int(x) for x in self.levels_data[level][2]])))

        return field_warp_dict, origin_list

    def _construct_neon_container(self):
        """
        Constructs a NEON container for exporting multi-resolution data to HDF5.
        This container will be used to transfer multi-resolution NEON fields into stacked warp fields.
        """

        @neon.Container.factory(name="HDF5MultiresExporter")
        def container(
            field_neon: Any,
            field_warp: Any,
            origin: Any,
            level: Any,
        ):
            def launcher(loader: neon.Loader):
                loader.set_mres_grid(field_neon.get_grid(), level)
                field_neon_hdl = loader.get_mres_read_handle(field_neon)
                refinement = 2**level

                @wp.func
                def kernel(index: Any):
                    cIdx = wp.neon_global_idx(field_neon_hdl, index)
                    # Get local indices by dividing the global indices (associated with the finest level) by 2^level
                    # Subtract the origin to get the local indices in the warp field
                    lx = wp.neon_get_x(cIdx) // refinement - origin[0]
                    ly = wp.neon_get_y(cIdx) // refinement - origin[1]
                    lz = wp.neon_get_z(cIdx) // refinement - origin[2]

                    # write the values to the warp field
                    cardinality = field_warp.shape[0]
                    for card in range(cardinality):
                        field_warp[card, lx, ly, lz] = self.store_dtype(wp.neon_read(field_neon_hdl, index, card))

                loader.declare_kernel(kernel)

            return launcher

        return container

    def get_fields_data(self, field_neon_dict):
        """
        Extracts and prepares the fields data from the NEON fields for export.
        """
        # Check if the field_neon_dict is empty
        if not field_neon_dict:
            return {}

        # Ensure that this operator is called on multires grids
        grid_mres = next(iter(field_neon_dict.values())).get_grid()
        assert grid_mres.name == "mGrid", f"Operation {self.__class__.__name} is only applicable to multi-resolution cases!"

        for field_name in field_neon_dict.keys():
            assert field_name in self.field_name_cardinality_dict.keys(), (
                f"Field {field_name} is not provided in the instantiation of the MultiresIO class!"
            )

        # number of levels
        num_levels = grid_mres.num_levels
        assert num_levels == len(self.levels_data), "Error: Inconsistent number of levels!"

        # Prepare the fields dictionary to be written by transfering multi-res NEON fields into stacked warp fields and then numpy arrays
        fields_data = {}
        for field_name, cardinality in self.field_name_cardinality_dict.items():
            if field_name not in field_neon_dict:
                continue
            for card in range(cardinality):
                fields_data[f"{field_name}_{card}"] = []

        # Iterate over each field and level to fill the dictionary with numpy fields
        for field_name, cardinality in self.field_name_cardinality_dict.items():
            if field_name not in field_neon_dict:
                continue
            for level in range(num_levels):
                # Create the container and run it to fill the warp fields
                c = self.container(field_neon_dict[field_name], self.field_warp_dict[field_name][level], self.origin_list[level], level)
                c.run(0, container_runtime=neon.Container.ContainerRuntime.neon)

                # Ensure all operations are complete before converting to JAX and Numpy arrays
                wp.synchronize()

                # Convert the warp fields to numpy arrays and use level's mask to filter the data
                mask = self.levels_data[level][0]
                field_np = self.field_warp_dict[field_name][level].numpy()
                for card in range(cardinality):
                    field_np_card = field_np[card][mask]
                    fields_data[f"{field_name}_{card}"].append(field_np_card)

        # Concatenate all field data
        for field_name in fields_data.keys():
            fields_data[field_name] = np.concatenate(fields_data[field_name])
            assert fields_data[field_name].size == self.total_cells, f"Error: Field {field_name} size mismatch!"

            # Unit conversion if applicable
            if self.unit_convertor is not None:
                if "velocity" in field_name.lower():
                    fields_data[field_name] = self.unit_convertor.velocity_to_physical(fields_data[field_name])
                elif "density" in field_name.lower():
                    fields_data[field_name] = self.unit_convertor.density_to_physical(fields_data[field_name])
                elif "pressure" in field_name.lower():
                    fields_data[field_name] = self.unit_convertor.pressure_to_physical(fields_data[field_name])
                # Add more physical quantities as needed

        return fields_data

    def to_hdf5(self, output_filename, field_neon_dict, compression="gzip", compression_opts=0):
        """
        Export the multi-resolution mesh data to an HDF5 file.
        Parameters
        ----------
        output_filename : str
            The name of the output HDF5 file (without extension).
        field_neon_dict : a dictionary of neon mGrid Fields
            Eg. The NEON fields containing velocity and density data as { "velocity": velocity_neon, "density": density_neon}
        compression : str, optional
            The compression method to use for the HDF5 file.
        compression_opts : int, optional
            The compression options to use for the HDF5 file.
        """
        import time

        # Get the fields data from the NEON fields
        fields_data = self.get_fields_data(field_neon_dict)

        # Save XDMF file
        self.save_xdmf(output_filename + ".h5", output_filename + ".xmf", self.total_cells, len(self.coordinates), fields_data)

        # Writing HDF5 file
        print("\tWriting HDF5 file")
        tic_write = time.perf_counter()
        self.save_hdf5_file(output_filename, self.coordinates, self.connectivity, self.level_id_field, fields_data, compression, compression_opts)
        toc_write = time.perf_counter()
        print(f"\tHDF5 file written in {toc_write - tic_write:0.1f} seconds")

    def to_slice_image(
        self,
        output_filename,
        field_neon_dict,
        plane_point,
        plane_normal,
        slice_thickness=1.0,
        bounds=[0, 1, 0, 1],
        grid_res=512,
        cmap=None,
        component=None,
        show_axes=False,
        show_colorbar=False,        
        normalize=1.0,
        output=None,
        **kwargs,
    ):
        """
        Export an arbitrary-plane slice from unstructured point data to PNG.

        Parameters
        ----------
        output_filename : str
            Output PNG filename (without extension).
        field_neon_dict : dict
            A dictionary of NEON fields containing the data to be plotted.
            Example: {"velocity": velocity_neon, "density": density_neon}
        plane_point : array_like
            A point [x, y, z] on the plane.
        plane_normal : array_like
            Plane normal vector [nx, ny, nz].
        slice_thickness : float
            How thick (in units of the coordinate system) the slice should be.
        grid_resolution : tuple
            Resolution of output image (pixels in plane u, v directions).
        grid_size : tuple
            Physical size of slice grid (width, height).
        cmap : str
            Matplotlib colormap.
        normalize : float
            Factor to scale and normalize data to ensure consistent images
        """
        # Get the fields data from the NEON fields
        assert len(field_neon_dict.keys()) == 1, "Error: This function is designed to plot a single field at a time."
        fields_data = self.get_fields_data(field_neon_dict)

        # Check if the component is within the valid range
        if component is None:
            print("\tCreating slice image of the field magnitude!")
            cell_data = list(fields_data.values())
            squared = [comp**2 for comp in cell_data]
            cell_data = np.sqrt(sum(squared))
            field_name = list(fields_data.keys())[0].split("_")[0] + "_magnitude"
        else:
            assert component < max(self.field_name_cardinality_dict.values()), (
                f"Error: Component {component} is out of range for the provided fields."
            )
            print(f"\tCreating slice image for component {component} of the input field!")
            field_name = list(fields_data.keys())[component]
            cell_data = fields_data[field_name]

        if normalize != 1.0:  
            cell_data = np.clip((cell_data / normalize),0,1)
        else:   
            cell_data = cell_data      

        # Plot each field in the dictionary
        self._to_slice_image_single_field(
            f"{output_filename}_{field_name}",
            cell_data,
            plane_point,
            plane_normal,
            slice_thickness=slice_thickness,
            bounds=bounds,
            grid_res=grid_res,
            cmap=cmap,
            show_axes=show_axes,
            show_colorbar=show_colorbar,
            normalize=normalize,
            **kwargs,
        )
        print(f"\tSlice image for field {field_name} saved as {output_filename}.png")

    def _to_slice_image_single_field(
        self,
        output_filename,
        field_data,
        plane_point,
        plane_normal,
        slice_thickness,
        bounds,
        grid_res,
        cmap,
        show_axes,
        show_colorbar,
        normalize,
        **kwargs,
    ):
        """
        Helper function to create a slice image for a single field.
        """
        from matplotlib import cm
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.spatial import cKDTree

        # field data are associated with the cells centers
        cell_values = field_data

        # get the normalized plane normal
        plane_normal = np.asarray(np.abs(plane_normal))
        n = plane_normal / np.linalg.norm(plane_normal)

        # Compute signed distances of each cell center to the plane
        plane_point *= plane_normal
        sdf = np.dot(self.centroids - plane_point, n)

        # Filter: cells with centroid near plane
        mask = np.abs(sdf) <= slice_thickness / 2
        if not np.any(mask):
            raise ValueError("No cells intersect the plane within thickness.")

        # Project centroids to plane
        centroids_slice = self.centroids[mask]
        sdf_slice = sdf[mask]
        proj = centroids_slice - np.outer(sdf_slice, n)

        values = cell_values[mask]

        # Build in-plane basis
        if np.allclose(n, [1, 0, 0]):
            u1 = np.array([0, 1, 0])
        else:
            u1 = np.array([1, 0, 0])
        u2 = np.abs(np.cross(n, u1))

        local_x = np.dot(proj - plane_point, u1)
        local_y = np.dot(proj - plane_point, u2)

        # Define extent of the plot
        xmin, xmax, ymin, ymax = local_x.min(), local_x.max(), local_y.min(), local_y.max()
        Lx = xmax - xmin
        Ly = ymax - ymin
        extent = np.array([xmin + bounds[0] * Lx, xmin + bounds[1] * Lx, ymin + bounds[2] * Ly, ymin + bounds[3] * Ly])
        mask_bounds = (extent[0] <= local_x) & (local_x <= extent[1]) & (extent[2] <= local_y) & (local_y <= extent[3])

        if cmap is None:
            cmap = cm.nipy_spectral

        # Adjust vertical resolution based on bounds
        bounded_x_min = local_x[mask_bounds].min()
        bounded_x_max = local_x[mask_bounds].max()
        bounded_y_min = local_y[mask_bounds].min()
        bounded_y_max = local_y[mask_bounds].max()
        width_x = bounded_x_max - bounded_x_min
        height_y = bounded_y_max - bounded_y_min
        aspect_ratio = height_y / width_x
        grid_resY = max(1, int(np.round(grid_res * aspect_ratio)))

        # Create grid
        grid_x = np.linspace(bounded_x_min, bounded_x_max, grid_res)
        grid_y = np.linspace(bounded_y_min, bounded_y_max, grid_resY)
        xv, yv = np.meshgrid(grid_x, grid_y, indexing="xy")

        # Fast KDTree-based interpolation
        points = np.column_stack((local_x[mask_bounds], local_y[mask_bounds]))
        tree = cKDTree(points)

        # Query points
        query_points = np.column_stack((xv.ravel(), yv.ravel()))

        # Find k nearest neighbors for smoother interpolation
        k = min(4, len(points))  # Use 4 neighbors or less if not enough points
        distances, indices = tree.query(query_points, k=k, workers=-1)  # -1 uses all cores

        # Inverse distance weighting
        epsilon = 1e-10
        weights = 1.0 / (distances + epsilon)
        weights /= weights.sum(axis=1, keepdims=True)

        # Interpolate values
        neighbor_values = values[mask_bounds][indices]
        grid_field = (neighbor_values * weights).sum(axis=1).reshape(grid_resY, grid_res)

        # Plot
        if show_colorbar or show_axes:
            dpi = 300
            plt.imshow(
                grid_field,
                extent=[bounded_x_min, bounded_x_max, bounded_y_min, bounded_y_max],
                cmap=cmap,
                origin="lower",
                aspect="equal",
                **kwargs,
            )
            if show_colorbar:
                plt.colorbar()
            if not show_axes:
                plt.axis("off")
            plt.savefig(output_filename + ".png", dpi=dpi, bbox_inches="tight", pad_inches=0)
            plt.close()
        else:
            if normalize != 1.0:
                plt.imsave(output_filename + ".png", grid_field, cmap=cmap, origin="lower", vmin=0, vmax=1)
            else:
                plt.imsave(output_filename + ".png", grid_field, cmap=cmap, origin="lower")

    def to_line(
        self,
        output_filename,
        field_neon_dict,
        start_point,
        end_point,
        resolution,
        component=None,
        radius=1.0,
        **kwargs,
    ):
        """
        Extract field data along a line between start_point and end_point and save to a CSV file.

        This function performs two main steps:
        1. Extracts field data from field_neon_dict, handling components or computing magnitude.
        2. Interpolates the field values along a line defined by start_point and end_point,
        then saves the results (coordinates and field values) to a CSV file.

        Parameters
        ----------
        output_filename : str
            The name of the output CSV file (without extension). Example: "velocity_profile".
        field_neon_dict : dict
            A dictionary containing the field data to extract, with a single key-value pair.
            The key is the field name (e.g., "velocity"), and the value is the NEON data object
            containing the field values. Example: {"velocity": velocity_neon}.
        start_point : array_like
            The starting point of the line in 3D space (e.g., [x0, y0, z0]).
            Units must match the coordinate system used in the class (voxel units if untransformed,
            or model units if scale/offset are applied).
        end_point : array_like
            The ending point of the line in 3D space (e.g., [x1, y1, z1]).
            Units must match the coordinate system used in the class.
        resolution : int
            The number of points along the line where the field will be interpolated.
            Example: 100 for 100 evenly spaced points.
        component : int, optional
            The specific component of the field to extract (e.g., 0 for x-component, 1 for y-component).
            If None, the magnitude of the field is computed. Default is None.
        radius : int
            The specified distance (in units of the coordinate system) to prefilter and query for line plot

        Returns
        -------
        None
            The function writes the output to a CSV file and prints a confirmation message.

        Notes
        -----
        - The output CSV file will contain columns: 'x', 'y', 'z', and the value of the field name (e.g., 'velocity_x' or 'velocity_magnitude').
        """

        # Get the fields data from the NEON fields
        assert len(field_neon_dict.keys()) == 1, "Error: This function is designed to plot a single field at a time."
        fields_data = self.get_fields_data(field_neon_dict)

        # Check if the component is within the valid range
        if component is None:
            print("\tCreating csv plot of the field magnitude!")
            cell_data = list(fields_data.values())
            squared = [comp**2 for comp in cell_data]
            cell_data = np.sqrt(sum(squared))
            field_name = list(fields_data.keys())[0].split("_")[0] + "_magnitude"

        else:
            assert component < max(self.field_name_cardinality_dict.values()), (
                f"Error: Component {component} is out of range for the provided fields."
            )
            print(f"\tCreating csv plot for component {component} of the input field!")
            field_name = list(fields_data.keys())[component]
            cell_data = fields_data[field_name]

        # Plot each field in the dictionary
        self._to_line_field(
            f"{output_filename}_{field_name}",
            cell_data,
            start_point,
            end_point,
            resolution,
            radius=radius,
            **kwargs,
        )
        print(f"\tLine Plot for field {field_name} saved as {output_filename}.csv")

    def _to_line_field(
        self,
        output_filename,
        cell_data,
        start_point,
        end_point,
        resolution,
        radius,
        **kwargs,
    ):
        """
        Helper function to create a line plot for a single field.
        """
        import numpy as np

        # cell_points = self.coordinates[self.connectivity]  # Shape: (M, K, 3), where M is num cells, K is nodes per cell
        # centroids = np.mean(cell_points, axis=1)  # Shape: (M, 3)
        centroids = self.centroids
        p0 = np.array(start_point, dtype=np.float32)
        p1 = np.array(end_point, dtype=np.float32)

        # direction and parameter t for each centroid
        d = p1 - p0
        L = np.linalg.norm(d)
        d_unit = d / L
        v = centroids - p0
        t = v.dot(d_unit)
        closest = p0 + np.outer(t, d_unit)
        perp_dist = np.linalg.norm(centroids - closest, axis=1)

        # optionally mask to [0,L] or a small perp-radius
        mask = (t >= 0) & (t <= L) & (perp_dist <= radius)
        t, data = t[mask], cell_data[mask]

        # sort by t
        idx = np.argsort(t)
        t_sorted = t[idx]
        data_sorted = data[idx]

        # target samples
        t_line = np.linspace(0, L, resolution)

        # 1D linear interpolation
        vals_line = np.interp(t_line, t_sorted, data_sorted, left=np.nan, right=np.nan)

        # reconstruct (x,y,z)
        line_xyz = p0[None, :] + t_line[:, None] * d_unit[None, :]

        # vectorized CSV dump
        out = np.hstack([line_xyz, vals_line[:, None]])
        np.savetxt(output_filename + ".csv", out, delimiter=",", header="x,y,z,value", comments="")
    
    def start_time_average(self):
        """
        Reset time-averaging accumulators.
        Call this once before your time loop.
        """
        self._avg_sum = {}
        self._avg_weight = 0.0
        self._avg_active = True
        self._avg_final_cache = None
        self._avg_cache_weight = 0.0
        self._avg_derived_cache = {}

    def accumulate_time_average(self, field_neon_dict: Dict[str, Any], weight: float = 1.0):
        """
        Accumulate a timestep into the running time-average using extracted cell-centered data.

        Parameters
        ----------
        field_neon_dict : dict
            Same as to_hdf5 / to_slice_image input (e.g. {"velocity": sim.u}).
            You can call this multiple times with different field keys over time;
            the accumulator stores by extracted component keys (e.g. "velocity_0").
        weight : float
            Typically dt for dt-weighted averaging, or 1.0 for simple arithmetic mean.
        """
        start_time = time.time()
        assert self._avg_active, "Call start_time_average() before accumulate_time_average()."
        fields_data = self.get_fields_data(field_neon_dict)
        self.accumulate_time_average_from_fields_data(fields_data, weight=weight)  
        print(f"Time Avg Accumulation in {time.time()-start_time}sec ")

    def accumulate_time_average_from_fields_data(self, fields_data: Dict[str, np.ndarray], weight: float = 1.0):
        """
        Same as accumulate_time_average(), but lets you pass already-extracted numpy arrays.
        Useful if you want to do custom pre-processing before accumulation.
        """
        assert self._avg_active, "Call start_time_average() before accumulating."
        w = float(weight)
        for k, v in fields_data.items():
            v64 = np.asarray(v, dtype=np.float64)
            if k not in self._avg_sum:
                self._avg_sum[k] = np.zeros_like(v64, dtype=np.float64)
            self._avg_sum[k] += w * v64
        self._avg_weight += w

    def finalize_time_average(self, keep_state: bool = True) -> Dict[str, np.ndarray]:
        assert self._avg_weight > 0.0, "No samples accumulated. Call accumulate_time_average() first."
        start_time = time.time()
        # Cache hit: only valid if no new accumulation since caching
        if self._avg_final_cache is None or self._avg_cache_weight != self._avg_weight:
            self._avg_final_cache = {
                k: (s / self._avg_weight).astype(np.float32)
                for k, s in self._avg_sum.items()
            }
            self._avg_cache_weight = self._avg_weight

        avg_fields = self._avg_final_cache

        if not keep_state:
            self._avg_active = False
            self._avg_sum = {}
            self._avg_weight = 0.0
            # Clear cache too (important)
            self._avg_final_cache = None
            self._avg_cache_weight = 0.0
        print(f"Finalize Time Avg in {time.time()-start_time}sec ")

        return avg_fields
    
    def to_hdf5_time_average(
        self,
        output_filename: str,
        compression: str = "gzip",
        compression_opts: int = 0,
        keep_state: bool = True,
    ):
        """
        Write an averaged HDF5/XDMF from the current accumulated average.
        """
        avg_fields = self.finalize_time_average(keep_state=keep_state)

        self.save_xdmf(
            output_filename + ".h5",
            output_filename + ".xmf",
            self.total_cells,
            len(self.coordinates),
            avg_fields,
        )

        print("\tWriting time-averaged HDF5 file")
        tic_write = time.perf_counter()
        self.save_hdf5_file(
            output_filename,
            self.coordinates,
            self.connectivity,
            self.level_id_field,
            avg_fields,
            compression,
            compression_opts,
        )
        toc_write = time.perf_counter()
        print(f"\tTime-averaged HDF5 file written in {toc_write - tic_write:0.1f} seconds")
        
    def to_slice_image_time_average(
        self,
        output_filename: str,
        field_base_name: str,
        plane_point,
        plane_normal,
        slice_thickness=1.0,
        bounds=[0, 1, 0, 1],
        grid_res=512,
        cmap=None,
        component=None,
        show_axes=False,
        show_colorbar=False,
        normalize=1.0,
        keep_state: bool = True,
        **kwargs,
    ):
        """
        Export a slice image from the accumulated time-averaged data.

        Parameters match to_slice_image(), except:
        - field_base_name: e.g. "velocity" (used to find keys "velocity_0", "velocity_1", ...)
        """
        avg_fields = self.finalize_time_average(keep_state=keep_state)

        # Find all components for this base field
        comp_keys = [k for k in avg_fields.keys() if k.startswith(field_base_name + "_")]
        assert len(comp_keys) > 0, f"No averaged components found for base field '{field_base_name}'."

        # Sort by component index (expects suffix _0, _1, _2, ...)
        def _comp_index(k: str) -> int:
            try:
                return int(k.split("_")[-1])
            except Exception:
                return 0

        comp_keys = sorted(comp_keys, key=_comp_index)

        if component is None and len(comp_keys) > 1:
            print("\tCreating time-averaged slice image of the field magnitude!")
            comps = [avg_fields[k].astype(np.float64) for k in comp_keys]
            cell_data = np.sqrt(np.sum([c**2 for c in comps], axis=0)).astype(np.float32)
            field_name = field_base_name + "_magnitude"
        elif component is None and len(comp_keys) == 1:
            print("\tCreating time-averaged slice image of the scalar field!")
            field_name = comp_keys[0]
            cell_data = avg_fields[field_name]
        else:
            assert 0 <= int(component) < len(comp_keys), "Requested component out of range for averaged field."
            field_name = comp_keys[int(component)]
            print(f"\tCreating time-averaged slice image for component {component} of the field!")
            cell_data = avg_fields[field_name]

        if normalize != 1.0:
            cell_data = np.clip(cell_data / normalize, 0, 1)

        self._to_slice_image_single_field(
            f"{output_filename}_{field_name}",
            cell_data,
            plane_point,
            plane_normal,
            slice_thickness=slice_thickness,
            bounds=bounds,
            grid_res=grid_res,
            cmap=cmap,
            show_axes=show_axes,
            show_colorbar=show_colorbar,
            normalize=normalize,
            **kwargs,
        )
        print(f"\tTime-averaged slice image for field {field_name} saved as {output_filename}.png")

    def to_line_time_average(
        self,
        output_filename: str,
        field_base_name: str,
        start_point,
        end_point,
        resolution: int,
        component: Optional[int] = None,
        radius: float = 1.0,
        keep_state: bool = True,
        **kwargs,
    ):
        """
        Extract time-averaged field data along a line and save to CSV.

        Parameters match to_line(), except:
        - field_base_name: e.g. "velocity" (used to find keys "velocity_0", "velocity_1", ...)
        - keep_state: if True, does not clear the time-average accumulators/cache
        """
        avg_fields = self.finalize_time_average(keep_state=keep_state)

        # Find all components for this base field
        comp_keys = [k for k in avg_fields.keys() if k.startswith(field_base_name + "_")]
        assert len(comp_keys) > 0, f"No averaged components found for base field '{field_base_name}'."

        # Sort by component index (expects suffix _0, _1, _2, ...)
        def _comp_index(k: str) -> int:
            try:
                return int(k.split("_")[-1])
            except Exception:
                return 0

        comp_keys = sorted(comp_keys, key=_comp_index)

        # Build the cell_data to sample along the line
        if component is None and len(comp_keys) > 1:
            print("\tCreating time-averaged line CSV of the field magnitude!")
            comps = [avg_fields[k].astype(np.float64, copy=False) for k in comp_keys]
            cell_data = np.sqrt(np.sum([c * c for c in comps], axis=0)).astype(np.float32)
            field_name = field_base_name + "_magnitude"

        elif component is None and len(comp_keys) == 1:
            print("\tCreating time-averaged line CSV of the scalar field!")
            field_name = comp_keys[0]  # e.g. density_0
            cell_data = avg_fields[field_name]

        else:
            ci = int(component)
            assert 0 <= ci < len(comp_keys), (
                f"Requested component {component} out of range for averaged field '{field_base_name}' "
                f"(has {len(comp_keys)} components)."
            )
            field_name = comp_keys[ci]
            print(f"\tCreating time-averaged line CSV for component {component} of the field!")
            cell_data = avg_fields[field_name]

        # Reuse your existing instantaneous line sampler
        self._to_line_field(
            f"{output_filename}_{field_name}",
            cell_data,
            start_point,
            end_point,
            resolution,
            radius=radius,
            **kwargs,
        )
        print(f"\tTime-averaged line CSV for field {field_name} saved as {output_filename}.csv")

    def _orient_normals_away_from_point(self, surface_points, normal_axes, reference_point):
        """
        Orient sign-ambiguous normal axes so they point away from a reference point.

        Parameters
        ----------
        surface_points : (N,3) float array
            Surface sample positions (usually vertices)
        normal_axes : (N,3) float array
            Local normal axes; sign may be arbitrary
        reference_point : array_like, shape (3,)
            A point assumed to lie inside the body

        Returns
        -------
        oriented_normals : (N,3) float32 array
            Normals with sign chosen so dot(n, x - p_ref) >= 0
        """
        surface_points = np.asarray(surface_points, dtype=np.float32)
        normal_axes = np.asarray(normal_axes, dtype=np.float32)
        reference_point = np.asarray(reference_point, dtype=np.float32)

        dirs = surface_points - reference_point[None, :]
        dots = np.einsum("ij,ij->i", normal_axes, dirs)

        oriented = normal_axes.copy()
        flip = dots < 0.0
        oriented[flip] *= -1.0

        oriented /= np.maximum(np.linalg.norm(oriented, axis=1, keepdims=True), 1e-12)
        return oriented.astype(np.float32)

    def _select_surface_field(self, fields_data, field_base_name, component=None):
        comp_keys = [k for k in fields_data.keys() if k.startswith(field_base_name + "_")]
        if len(comp_keys) == 0:
            raise KeyError(f"No field components found for base field '{field_base_name}'")

        comp_keys = sorted(comp_keys, key=lambda k: int(k.rsplit("_", 1)[1]))

        if component is None:
            if len(comp_keys) == 1:
                return comp_keys[0], np.asarray(fields_data[comp_keys[0]], dtype=np.float32)
            comps = np.stack([np.asarray(fields_data[k], dtype=np.float64) for k in comp_keys], axis=1)
            mag = np.linalg.norm(comps, axis=1).astype(np.float32)
            return f"{field_base_name}_magnitude", mag

        ci = int(component)
        if ci < 0 or ci >= len(comp_keys):
            raise ValueError(
                f"Component {component} out of range for '{field_base_name}' "
                f"(has {len(comp_keys)} components)"
            )

        key = comp_keys[ci]
        return key, np.asarray(fields_data[key], dtype=np.float32)

    def _load_surface_mesh(self, surface_mesh_filename):
        import trimesh

        mesh = trimesh.load(surface_mesh_filename, force="mesh", process=False)

        if isinstance(mesh, trimesh.Scene):
            meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if len(meshes) == 0:
                raise ValueError(f"No triangle mesh found in '{surface_mesh_filename}'")
            mesh = trimesh.util.concatenate(meshes)

        if mesh.is_empty:
            raise ValueError(f"Loaded mesh '{surface_mesh_filename}' is empty")

        try:
            mesh.remove_unreferenced_vertices()
        except Exception:
            pass
        try:
            mesh.remove_duplicate_faces()
        except Exception:
            pass
        try:
            mesh.remove_degenerate_faces()
        except Exception:
            pass
        try:
            trimesh.repair.fix_normals(mesh, multibody=True)
        except Exception:
            pass

        _ = mesh.vertex_normals
        _ = mesh.face_normals
        return mesh

    def _build_vertex_neighbors(self, faces, n_vertices):
        neighbors = [set() for _ in range(n_vertices)]
        for tri in np.asarray(faces, dtype=np.int32):
            i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
            neighbors[i].update((j, k))
            neighbors[j].update((i, k))
            neighbors[k].update((i, j))
        return [
            np.fromiter(nbrs, dtype=np.int32) if len(nbrs) else np.empty(0, dtype=np.int32)
            for nbrs in neighbors
        ]

    def _repair_and_smooth_vertex_normals(self, mesh):
        """
        
        """
        normals = np.asarray(mesh.vertex_normals, dtype=np.float32).copy()
        verts = np.asarray(mesh.vertices, dtype=np.float32)

        # Check for bad normals
        nrm = np.linalg.norm(normals, axis=1)
        bad = nrm < 1e-12
        
        if np.any(bad):
            # Fallback: use vertex position relative to center
            center = verts.mean(axis=0)
            fallback = verts - center
            fallback_norm = np.linalg.norm(fallback, axis=1, keepdims=True)
            fallback /= np.maximum(fallback_norm, 1e-12)
            normals[bad] = fallback[bad]
        
        return normals.astype(np.float32)

    def _build_face_normals_from_vertex_normals(self, faces, vertex_normals):
        f = np.asarray(faces, dtype=np.int32)
        vn = np.asarray(vertex_normals, dtype=np.float32)
        face_normals = vn[f].mean(axis=1)
        face_normals /= np.maximum(np.linalg.norm(face_normals, axis=1, keepdims=True), 1e-20)
        return face_normals.astype(np.float32)

    def _sample_surface_scalar_one_side(
        self,
        surface_points,
        surface_normals,
        cell_values,
        sample_dx,
        shell_factors=(0.75, 1.25, 1.75),
        k=24,
        power=2.0,
        max_distance=None,
        half_space_tolerance=0.25,
        aggregate="median",
    ):
        
        sample_dx = float(sample_dx)
        surface_points = np.asarray(surface_points, dtype=np.float32)
        surface_normals = np.asarray(surface_normals, dtype=np.float32)
        cell_values = np.asarray(cell_values, dtype=np.float32)

        surface_normals /= np.maximum(np.linalg.norm(surface_normals, axis=1, keepdims=True), 1e-20)

        shell_factors = np.asarray(shell_factors, dtype=np.float32)
        shell_offsets = shell_factors * sample_dx

        queries = surface_points[:, None, :] + surface_normals[:, None, :] * shell_offsets[None, :, None]
        n_points = queries.shape[0]
        n_shells = queries.shape[1]

        queries_flat = queries.reshape(-1, 3)
        base_points_flat = np.repeat(surface_points, n_shells, axis=0)
        normals_flat = np.repeat(surface_normals, n_shells, axis=0)

        tree = self.kd_tree 
        kk = min(int(k), len(self.centroids))
        distances, indices = tree.query(queries_flat, k=kk)

        if kk == 1:
            distances = distances[:, None]
            indices = indices[:, None]

        neighbor_points = self.centroids[indices]
        neighbor_values = cell_values[indices]

        signed = np.einsum("qki,qi->qk", neighbor_points - base_points_flat[:, None, :], normals_flat)
        valid = signed >= (-half_space_tolerance * sample_dx)

        if max_distance is not None:
            valid &= distances <= float(max_distance)

        weights = 1.0 / np.maximum(distances, 1e-12) ** float(power)
        weights *= valid

        dead = np.sum(weights, axis=1) <= 0.0
        if np.any(dead):
            fallback = 1.0 / np.maximum(distances[dead], 1e-12) ** float(power)
            weights[dead] = fallback

        wsum = np.sum(weights, axis=1)
        sampled = np.sum(weights * neighbor_values, axis=1) / np.maximum(wsum, 1e-20)

        local_mean = sampled[:, None]
        local_spread = np.sum(weights * np.abs(neighbor_values - local_mean), axis=1) / np.maximum(wsum, 1e-20)

        sampled = sampled.reshape(n_points, n_shells)
        support = wsum.reshape(n_points, n_shells)
        spread = local_spread.reshape(n_points, n_shells)

        if aggregate == "mean":
            mapped = np.mean(sampled, axis=1)
        else:
            mapped = np.median(sampled, axis=1)

        score = np.median(support / np.maximum(spread, 1e-6), axis=1)

        return mapped.astype(np.float32), score.astype(np.float32)

    def _sample_surface_scalar_bidirectional(
        self,
        surface_points,
        surface_normals,
        cell_values,
        sample_dx,
        shell_factors=(0.75, 1.25, 1.75),
        k=24,
        power=2.0,
        max_distance=None,
        half_space_tolerance=0.25,
        aggregate="median",
    ):
        plus_vals, plus_score = self._sample_surface_scalar_one_side(
            surface_points=surface_points,
            surface_normals=surface_normals,
            cell_values=cell_values,
            sample_dx=sample_dx,
            shell_factors=shell_factors,
            k=k,
            power=power,
            max_distance=max_distance,
            half_space_tolerance=half_space_tolerance,
            aggregate=aggregate,
        )

        minus_vals, minus_score = self._sample_surface_scalar_one_side(
            surface_points=surface_points,
            surface_normals=-surface_normals,
            cell_values=cell_values,
            sample_dx=sample_dx,
            shell_factors=shell_factors,
            k=k,
            power=power,
            max_distance=max_distance,
            half_space_tolerance=half_space_tolerance,
            aggregate=aggregate,
        )

        use_minus = ~(minus_score > plus_score)

        mapped = plus_vals.copy()
        mapped[use_minus] = minus_vals[use_minus] 

        return mapped.astype(np.float32)

    def _smooth_surface_scalar(self, values, faces, iterations=2, relaxation=0.2):
        """
        Vectorized Laplacian smoothing via sparse matrix multiply.
        ~100x faster than per-vertex loops.
        """
        from scipy.sparse import csr_matrix       
        
        if iterations <= 0:
            return values
        
        n_verts = len(values)
        
        # Build adjacency matrix (sparse)
        rows, cols, data = [], [], []
        for tri in faces:
            i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
            # Each edge contributes to the Laplacian
            for (a, b) in [(i, j), (j, k), (k, i)]:
                rows.extend([a, b])
                cols.extend([b, a])
                data.extend([1.0, 1.0])
        
        # Normalize: each vertex's neighbors
        L = csr_matrix((data, (rows, cols)), shape=(n_verts, n_verts))
        L.data /= np.asarray(L.sum(axis=1)).flatten()[L.nonzero()[0]]
        
        # Apply smoothing
        v = values.copy()
        for _ in range(iterations):
            v = (1.0 - relaxation) * v + relaxation * (L @ v)
        
        return v.astype(np.float32)

    def _weld_vertices(
        self,
        vertices,
        faces,
        vertex_rgb=None,
        vertex_values=None,
        tolerance=0.0,
    ):
        """
        Weld duplicate / near-duplicate vertices by position.

        Parameters
        ----------
        vertices : (N,3) array
        faces : (M,3) int array
        vertex_rgb : (N,3) uint8 array, optional
            Averaged across welded duplicates.
        vertex_values : (N,) float array, optional
            Averaged across welded duplicates.
        tolerance : float
            0.0 means exact-position weld.
            >0 means weld positions after quantization by this tolerance.

        Returns
        -------
        welded_vertices, welded_faces, welded_rgb, welded_values
        """
        vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int32)

        if len(vertices) == 0:
            return (
                vertices.astype(np.float32),
                faces,
                None if vertex_rgb is None else np.asarray(vertex_rgb),
                None if vertex_values is None else np.asarray(vertex_values),
            )

        tol = float(tolerance)

        if tol <= 0.0:
            # exact weld
            vv = np.ascontiguousarray(vertices)
            key_view = vv.view(np.dtype((np.void, vv.dtype.itemsize * vv.shape[1]))).ravel()
            _, unique_idx, inverse = np.unique(key_view, return_index=True, return_inverse=True)
        else:
            # tolerance-based weld
            q = np.rint(vertices / tol).astype(np.int64)
            _, unique_idx, inverse = np.unique(q, axis=0, return_index=True, return_inverse=True)

        # preserve first-occurrence order rather than np.unique key order
        order = np.argsort(unique_idx)
        unique_idx = unique_idx[order]
        remap = np.empty(len(order), dtype=np.int32)
        remap[order] = np.arange(len(order), dtype=np.int32)
        inverse = remap[inverse]

        welded_vertices = vertices[unique_idx].astype(np.float32, copy=False)

        n_unique = len(unique_idx)
        counts = np.bincount(inverse, minlength=n_unique).astype(np.float64)

        welded_rgb = None
        if vertex_rgb is not None:
            rgb = np.asarray(vertex_rgb, dtype=np.float64)
            acc = np.zeros((n_unique, 3), dtype=np.float64)
            np.add.at(acc, inverse, rgb)
            welded_rgb = np.round(acc / np.maximum(counts[:, None], 1.0)).astype(np.uint8)

        welded_values = None
        if vertex_values is not None:
            vals = np.asarray(vertex_values, dtype=np.float64)
            acc = np.zeros(n_unique, dtype=np.float64)
            np.add.at(acc, inverse, vals)
            welded_values = (acc / np.maximum(counts, 1.0)).astype(np.float32)

        welded_faces = inverse[faces]

        # drop degenerate triangles introduced by welding
        keep = (
            (welded_faces[:, 0] != welded_faces[:, 1]) &
            (welded_faces[:, 1] != welded_faces[:, 2]) &
            (welded_faces[:, 0] != welded_faces[:, 2])
        )
        welded_faces = welded_faces[keep]

        return welded_vertices, welded_faces, welded_rgb, welded_values
    
    def _write_mtl_for_color_bins(
        self,
        mtl_filename,
        material_rgb,
        material_prefix="field_bin",
    ):
        """
        Write one diffuse material per color bin.
        """
        from pathlib import Path

        mtl_path = Path(mtl_filename)
        mtl_path.parent.mkdir(parents=True, exist_ok=True)

        material_rgb = np.asarray(material_rgb, dtype=np.uint8)
        material_names = []

        with open(mtl_path, "w") as f:
            f.write("# MTL written by MultiresIO\n\n")

            for i, rgb in enumerate(material_rgb):
                r, g, b = (rgb.astype(np.float32) / 255.0).tolist()
                name = f"{material_prefix}_{i:03d}"
                material_names.append(name)

                f.write(f"newmtl {name}\n")
                f.write(f"Ka {r:.6f} {g:.6f} {b:.6f}\n")
                f.write(f"Kd {r:.6f} {g:.6f} {b:.6f}\n")
                f.write("Ks 0.000000 0.000000 0.000000\n")
                f.write("Ns 1.000000\n")
                f.write("illum 1\n\n")

        print(f"\tMTL written: {mtl_path}")
        return material_names, str(mtl_path)

    def _write_polydata_vtk(self, vtk_filename, vertices, faces, point_data=None, cell_data=None):
        """
        Fast VTK ASCII write using batch numpy formatting.
        """
        import io
        
        vertices = np.asarray(vertices, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.int32)
        
        buf = io.StringIO()
        
        # Header
        buf.write("# vtk DataFile Version 3.0\n")
        buf.write("surface_field_map\n")
        buf.write("ASCII\n")
        buf.write("DATASET POLYDATA\n")
        
        # Points — write entire array at once via numpy
        buf.write(f"POINTS {len(vertices)} float\n")
        np.savetxt(buf, vertices, fmt='%.6f', delimiter=' ')
        
        # Polygons — prepend "3" column, write at once
        buf.write(f"POLYGONS {len(faces)} {len(faces) * 4}\n")
        poly_block = np.column_stack([np.full(len(faces), 3, dtype=np.int32), faces])
        np.savetxt(buf, poly_block, fmt='%d', delimiter=' ')
        
        # Point data
        if point_data:
            buf.write(f"POINT_DATA {len(vertices)}\n")
            for name, arr in point_data.items():
                arr = np.asarray(arr, dtype=np.float32)
                if arr.ndim == 1:
                    buf.write(f"SCALARS {name} float 1\n")
                    buf.write("LOOKUP_TABLE default\n")
                    np.savetxt(buf, arr, fmt='%.6f')
                elif arr.ndim == 2 and arr.shape[1] == 3:
                    buf.write(f"VECTORS {name} float\n")
                    np.savetxt(buf, arr, fmt='%.6f', delimiter=' ')
                else:
                    raise ValueError(f"Unsupported point_data shape for '{name}': {arr.shape}")
        
        # Cell data
        if cell_data:
            buf.write(f"CELL_DATA {len(faces)}\n")
            for name, arr in cell_data.items():
                arr = np.asarray(arr, dtype=np.float32)
                if arr.ndim == 1:
                    buf.write(f"SCALARS {name} float 1\n")
                    buf.write("LOOKUP_TABLE default\n")
                    np.savetxt(buf, arr, fmt='%.6f')
                elif arr.ndim == 2 and arr.shape[1] == 3:
                    buf.write(f"VECTORS {name} float\n")
                    np.savetxt(buf, arr, fmt='%.6f', delimiter=' ')
                else:
                    raise ValueError(f"Unsupported cell_data shape for '{name}': {arr.shape}")
        
        # Single disk write
        with open(vtk_filename, "w") as f:
            f.write(buf.getvalue())
            
    def _map_field_to_surface_mesh(
        self,        
        surface_mesh,        
        fields_data,
        field_base_name,
        center_point=None,
        component=None,
        sample_dx=None,
        shell_factors=(1.0, 1.5),
        k=10,
        power=2.0,
        max_distance=None,
        half_space_tolerance=0.1,
        aggregate="median",
        smooth_iterations=2,
        smooth_relaxation=0.2,
    ):
  
        field_name, cell_values = self._select_surface_field(fields_data, field_base_name, component=component)

        mesh = surface_mesh
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)
   
        if sample_dx is None:
            sample_dx = min(float(vs) for (_, vs, _, _) in self.levels_data)

        if max_distance is None:
            max_distance = 3.0 * float(sample_dx)

        vertex_normals = self._repair_and_smooth_vertex_normals(mesh)
        if center_point is None:
            mapped = self._sample_surface_scalar_bidirectional(
                            surface_points=vertices,
                            surface_normals=vertex_normals,
                            cell_values=cell_values,
                            sample_dx=sample_dx,
                            shell_factors=shell_factors,
                            k=k,
                            power=power,
                            max_distance=max_distance,
                            half_space_tolerance=half_space_tolerance,
                            aggregate=aggregate,
                        )        
        else:
            chosen_normals = self._orient_normals_away_from_point(
                            surface_points=vertices,
                            normal_axes=vertex_normals,
                            reference_point=center_point,
                        )
            mapped, _ = self._sample_surface_scalar_one_side(
                        surface_points=vertices,
                        surface_normals=chosen_normals,
                        cell_values=cell_values,
                        sample_dx=sample_dx,
                        shell_factors=shell_factors,
                        k=k,
                        power=power,
                        max_distance=max_distance,
                        half_space_tolerance=half_space_tolerance,
                        aggregate=aggregate,
                    )

        
      
        if smooth_iterations > 0:
            mapped = self._smooth_surface_scalar(
                mapped,
                faces,
                iterations=smooth_iterations,
                relaxation=smooth_relaxation,
            )
 
        return mesh, vertices, faces, field_name, mapped.astype(np.float32)

    def _fields_data_to_surface_vtk(
        self,
        output_filename,
        surface_mesh,
        fields_data,
        field_base_name,
        center_point=None,
        component=None,
        sample_dx=None,
        shell_factors=(1.0, 1.5),
        k=10,
        power=2.0,
        max_distance=None,
        half_space_tolerance=0.1,
        aggregate="median",
        smooth_iterations=2,
        smooth_relaxation=0.2,
    ):
        _, vertices, faces, field_name, mapped = self._map_field_to_surface_mesh(
            surface_mesh=surface_mesh,
            fields_data=fields_data,
            field_base_name=field_base_name,
            center_point=center_point,
            component=component,
            sample_dx=sample_dx,
            shell_factors=shell_factors,
            k=k,
            power=power,
            max_distance=max_distance,
            half_space_tolerance=half_space_tolerance,
            aggregate=aggregate,
            smooth_iterations=smooth_iterations,
            smooth_relaxation=smooth_relaxation,
        )

        vtk_filename = output_filename if output_filename.endswith(".vtk") else output_filename + ".vtk"

        self._write_polydata_vtk(
            vtk_filename,
            vertices,
            faces,
            point_data={field_name: mapped},
            cell_data=None,
        )
    
    def _scalar_to_rgb(self, values, cmap="nipy_spectral", vmin=None, vmax=None):
        """
        Convert scalar values to uint8 RGB colors.
        """
        from matplotlib import cm

        values = np.asarray(values, dtype=np.float32)
        finite = np.isfinite(values)
        if not np.any(finite):
            raise ValueError("No finite values available for color mapping.")

        if vmin is None:
            vmin = float(np.nanmin(values))
        if vmax is None:
            vmax = float(np.nanmax(values))

        if vmax <= vmin:
            vmax = vmin + 1.0

        t = np.zeros_like(values, dtype=np.float32)
        t[finite] = np.clip((values[finite] - vmin) / (vmax - vmin), 0.0, 1.0)

        rgba = cm.get_cmap(cmap)(t)
        rgb = np.round(255.0 * rgba[:, :3]).astype(np.uint8)
        rgb[~finite] = 0

        return rgb, vmin, vmax

    def _write_obj_with_vertex_colors(
        self,
        obj_filename,
        vertices,
        faces,
        vertex_rgb,
        value_name=None,
        vmin=None,
        vmax=None,
        vertex_values=None,
        cmap="nipy_spectral",
        weld_vertices=True,
        weld_tolerance=0.0001,
        write_mtl=True,
        mtl_bin_count=32,
    ):
        """
        Write OBJ with:
        - welded vertices
        - per-vertex RGB on v-lines
        - MTL fallback with quantized face materials for Alias / standard OBJ readers

        Returns
        -------
        obj_path : str
        mtl_path : Optional[str]
        """
        from pathlib import Path

        vertices = np.asarray(vertices, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.int32)
        vertex_rgb = np.asarray(vertex_rgb, dtype=np.uint8)

        if vertex_values is not None:
            vertex_values = np.asarray(vertex_values, dtype=np.float32)

        # Normalize output path
        obj_path = Path(obj_filename)
        if obj_path.suffix.lower() != ".obj":
            obj_path = obj_path.with_suffix(".obj")
        obj_path.parent.mkdir(parents=True, exist_ok=True)

        # Weld first
        if weld_vertices:
            vertices, faces, vertex_rgb, vertex_values = self._weld_vertices(
                vertices=vertices,
                faces=faces,
                vertex_rgb=vertex_rgb,
                vertex_values=vertex_values,
                tolerance=weld_tolerance,
            )

        rgb01 = vertex_rgb.astype(np.float32) / 255.0

        # Build optional MTL / per-face material assignment
        mtl_path = None
        material_names = None
        face_material_ids = None

        if write_mtl and len(faces) > 0:
            n_bins = max(2, int(mtl_bin_count))

            if vertex_values is not None:
                # Preferred path: bin by scalar field
                if vmin is None:
                    vmin = float(np.nanmin(vertex_values))
                if vmax is None:
                    vmax = float(np.nanmax(vertex_values))
                if vmax <= vmin:
                    vmax = vmin + 1.0

                bin_centers = np.linspace(vmin, vmax, n_bins, dtype=np.float32)
                material_rgb, _, _ = self._scalar_to_rgb(
                    bin_centers, cmap=cmap, vmin=vmin, vmax=vmax
                )

                face_values = vertex_values[faces].mean(axis=1)
                t = np.clip((face_values - vmin) / (vmax - vmin), 0.0, 1.0)
                face_material_ids = np.minimum(
                    np.floor(t * n_bins).astype(np.int32),
                    n_bins - 1,
                )
            else:
                # Fallback: bin directly from averaged face RGB
                face_rgb = np.round(vertex_rgb[faces].mean(axis=1)).astype(np.uint8)
                # Quantize RGB to reduce material count
                qstep = max(1, int(np.ceil(256 / n_bins)))
                face_rgb_q = (face_rgb // qstep) * qstep
                material_rgb, inverse = np.unique(face_rgb_q, axis=0, return_inverse=True)
                face_material_ids = inverse.astype(np.int32)

            mtl_path = obj_path.with_suffix(".mtl")
            material_names, _ = self._write_mtl_for_color_bins(
                mtl_filename=str(mtl_path),
                material_rgb=material_rgb,
                material_prefix=(value_name or "field"),
            )

        with open(obj_path, "w") as f:
            f.write("# Colored OBJ written by Studio Wind Tunnel\n")
            if value_name is not None:
                f.write(f"# field_name {value_name}\n")
            if vmin is not None and vmax is not None:
                f.write(f"# color_range {vmin:.9g} {vmax:.9g}\n")
            if weld_vertices:
                f.write("# welded_vertices 1\n")
                f.write(f"# weld_tolerance {float(weld_tolerance):.9g}\n")
            else:
                f.write("# welded_vertices 0\n")

            if mtl_path is not None:
                f.write(f"mtllib {mtl_path.name}\n")

            f.write("# vertex format: v x y z r g b\n")

            # vertices with RGB
            for (x, y, z), (r, g, b) in zip(vertices, rgb01):
                f.write(f"v {x:.6f} {y:.6f} {z:.6f} {r:.6f} {g:.6f} {b:.6f}\n")

            # faces
            faces1 = faces + 1
            if face_material_ids is None:
                for i, j, k in faces1:
                    f.write(f"f {i} {j} {k}\n")
            else:
                current_mid = -1
                for (i, j, k), mid in zip(faces1, face_material_ids):
                    if mid != current_mid:
                        f.write(f"usemtl {material_names[mid]}\n")
                        current_mid = mid
                    f.write(f"f {i} {j} {k}\n")

        print(f"\tOBJ written: {obj_path}")
        if mtl_path is not None:
            print(f"\tOBJ references MTL: {mtl_path.name}")
        else:
            print("\tOBJ written without MTL")

        return str(obj_path), (None if mtl_path is None else str(mtl_path))

    def to_surface_vtk(
        self,
        output_filename,
        surface_mesh,
        field_neon_dict,
        field_base_name,
        center_point=None,
        component=None,
        sample_dx=None,
        shell_factors=(1.0, 1.5),
        k=10,
        power=2.0,
        max_distance=None,
        half_space_tolerance=0.1,
        aggregate="median",
        smooth_iterations=2,
        smooth_relaxation=0.2,
    ):
        tic_write = time.perf_counter()
        fields_data = self.get_fields_data(field_neon_dict)
        self._fields_data_to_surface_vtk(
            output_filename=output_filename,
            surface_mesh=surface_mesh,
            fields_data=fields_data,
            field_base_name=field_base_name,
            center_point=center_point,
            component=component,
            sample_dx=sample_dx,
            shell_factors=shell_factors,
            k=k,
            power=power,
            max_distance=max_distance,
            half_space_tolerance=half_space_tolerance,
            aggregate=aggregate,
            smooth_iterations=smooth_iterations,
            smooth_relaxation=smooth_relaxation,
        )
        print(f"\tSurface field written to {output_filename} in {time.perf_counter() - tic_write:0.1f} seconds")

    def to_surface_vtk_time_average(
        self,
        output_filename,
        surface_mesh,
        field_base_name,
        center_point=None,
        component=None,
        keep_state=True,
        sample_dx=None,
        shell_factors=(1.0, 1.5),
        k=10,
        power=2.0,
        max_distance=None,
        half_space_tolerance=0.1,
        aggregate="median",
        smooth_iterations=2,
        smooth_relaxation=0.2,
    ):
        tic_write = time.perf_counter()
        avg_fields = self.finalize_time_average(keep_state=keep_state)
        self._fields_data_to_surface_vtk(
            output_filename=output_filename,
            surface_mesh=surface_mesh,
            fields_data=avg_fields,
            field_base_name=field_base_name,
            center_point=center_point,
            component=component,
            sample_dx=sample_dx,
            shell_factors=shell_factors,
            k=k,
            power=power,
            max_distance=max_distance,
            half_space_tolerance=half_space_tolerance,
            aggregate=aggregate,
            smooth_iterations=smooth_iterations,
            smooth_relaxation=smooth_relaxation,
        )
        print(f"\tSurface field written to {output_filename} in {time.perf_counter() - tic_write:0.1f} seconds")

    def _fields_data_to_surface_obj(
        self,
        output_filename,
        surface_mesh,
        fields_data,
        field_base_name,
        center_point=None,
        component=None,
        sample_dx=None,
        shell_factors=(1.0, 1.5),
        k=10,
        power=2.0,
        max_distance=None,
        half_space_tolerance=0.1,
        aggregate="median",
        smooth_iterations=2,
        smooth_relaxation=0.2,
        cmap="viridis",
        vmin=None,
        vmax=None,
    ):
        _, vertices, faces, field_name, mapped = self._map_field_to_surface_mesh(
            surface_mesh=surface_mesh,
            fields_data=fields_data,
            field_base_name=field_base_name,
            center_point=center_point,
            component=component,
            sample_dx=sample_dx,
            shell_factors=shell_factors,
            k=k,
            power=power,
            max_distance=max_distance,
            half_space_tolerance=half_space_tolerance,
            aggregate=aggregate,
            smooth_iterations=smooth_iterations,
            smooth_relaxation=smooth_relaxation,
        )

        vertex_rgb, vmin, vmax = self._scalar_to_rgb(
            mapped, cmap=cmap, vmin=vmin, vmax=vmax
        )

        obj_filename = output_filename if output_filename.endswith(".obj") else output_filename + ".obj"

        self._write_obj_with_vertex_colors(
            obj_filename=obj_filename,
            vertices=vertices,
            faces=faces,
            vertex_rgb=vertex_rgb,
            value_name=field_name,
            vmin=vmin,
            vmax=vmax,
        )

    def to_surface_obj(
        self,
        output_filename,
        surface_mesh,
        field_neon_dict,
        field_base_name,
        center_point=None,
        component=None,
        sample_dx=None,
        shell_factors=(1.0, 1.5),
        k=10,
        power=2.0,
        max_distance=None,
        half_space_tolerance=0.1,
        aggregate="median",
        smooth_iterations=2,
        smooth_relaxation=0.2,
        cmap="nipy_spectral",
        vmin=None,
        vmax=None,
    ):
        tic_write = time.perf_counter()

        fields_data = self.get_fields_data(field_neon_dict)

        self._fields_data_to_surface_obj(
            output_filename=output_filename,
            surface_mesh=surface_mesh,
            fields_data=fields_data,
            field_base_name=field_base_name,
            center_point=center_point,
            component=component,
            sample_dx=sample_dx,
            shell_factors=shell_factors,
            k=k,
            power=power,
            max_distance=max_distance,
            half_space_tolerance=half_space_tolerance,
            aggregate=aggregate,
            smooth_iterations=smooth_iterations,
            smooth_relaxation=smooth_relaxation,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        print(f"\tSurface OBJ written to {output_filename} in {time.perf_counter() - tic_write:0.1f} seconds")

    def to_surface_obj_time_average(
        self,
        output_filename,
        surface_mesh,
        field_base_name,
        center_point=None,
        component=None,
        keep_state=True,
        sample_dx=None,
        shell_factors=(1.0, 1.5),
        k=10,
        power=2.0,
        max_distance=None,
        half_space_tolerance=0.1,
        aggregate="median",
        smooth_iterations=2,
        smooth_relaxation=0.2,
        cmap="nipy_spectral",
        vmin=None,
        vmax=None,
    ):
        tic_write = time.perf_counter()

        avg_fields = self.finalize_time_average(keep_state=keep_state)

        self._fields_data_to_surface_obj(
            output_filename=output_filename,
            surface_mesh=surface_mesh,
            fields_data=avg_fields,
            field_base_name=field_base_name,
            center_point=center_point,
            component=component,
            sample_dx=sample_dx,
            shell_factors=shell_factors,
            k=k,
            power=power,
            max_distance=max_distance,
            half_space_tolerance=half_space_tolerance,
            aggregate=aggregate,
            smooth_iterations=smooth_iterations,
            smooth_relaxation=smooth_relaxation,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        print(f"\tTime-averaged surface OBJ written to {output_filename} in {time.perf_counter() - tic_write:0.1f} seconds")
