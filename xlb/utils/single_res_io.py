"""
Single-resolution IO: HDF5+XDMF, time-averaging, and slice-image export.

:class:`SingleResIO` is a plain-dense-grid analogue of
:class:`~xlb.utils.mesher.MultiresIO`, reusing the same output conventions
(HDF5 dataset naming, XDMF companion file, derived-field formulas, PNG slice
look) so existing downstream tooling built against the multires H5+XDMF
output keeps working. It's much simpler internally because a single-res
``dGrid`` is a plain regular Cartesian array -- no unstructured hex-cell
mesh, no per-level block structure, no KDTree-based slicing.

Deferred (not implemented -- these depend on a body-fitted surface mesh
sampled against the voxel field, a different and more involved problem than
what's needed here): ``to_surface_vtk_time_average``,
``to_isosurface_stl_time_average``. Calling them prints a warning and
returns without writing anything, so a script written against the
MultiresIO API doesn't crash if those options are enabled.
"""

import os
from typing import Any
import numpy as np
import warp as wp


def neon_field_to_numpy(neon_field, cardinality, grid_shape, wp_dtype):
    """Read a plain (single-resolution) Neon field back to a numpy array.

    Plain ``dField`` objects have no ``.numpy()``/``.update_host()`` (only
    ``.export_vti()``) -- this bounces the field through a small Warp array
    via one Neon container, then reads that back with ``.numpy()``.

    Returns an array of shape ``(cardinality, nx, ny, nz)``.
    """
    import neon

    nx, ny, nz = grid_shape
    wp_buf = wp.zeros((cardinality, nx, ny, nz), dtype=wp_dtype)

    @neon.Container.factory(name="bounce_to_warp")
    def container(src_field: Any, dst_buf: Any):
        def ll(loader: neon.Loader):
            loader.set_grid(src_field.get_grid())
            src_pn = loader.get_read_handle(src_field)

            @wp.func
            def kern(index: Any):
                cIdx = wp.neon_global_idx(src_pn, index)
                gx = wp.neon_get_x(cIdx)
                gy = wp.neon_get_y(cIdx)
                gz = wp.neon_get_z(cIdx)
                for c in range(cardinality):
                    dst_buf[c, gx, gy, gz] = wp_dtype(wp.neon_read(src_pn, index, c))

            loader.declare_kernel(kern)

        return ll

    c = container(neon_field, wp_buf)
    c.run(0, container_runtime=neon.Container.ContainerRuntime.neon)
    wp.synchronize()
    return wp_buf.numpy()


class SingleResIO(object):
    # Multi-plane outputSlices rendering (arbitrary width/height-vector
    # planes, thread-pooled) is not implemented -- only the simple
    # axis-aligned to_slice_image/to_slice_image_time_average paths are.
    SUPPORTS_MULTI_PLANE_SLICES = False

    DERIVED_FIELD_DEPS = {
        "pressure": ["density"],
        "Cp": ["density"],
        "CpTotal": ["density", "velocity"],
        "CpTotalLoss": ["density", "velocity"],
        "qdyn": ["density", "velocity"],
    }

    def __init__(
        self,
        field_name_cardinality_dict,
        grid_shape,
        voxel_size,
        offset=(0.0, 0.0, 0.0),
        unit_convertor=None,
        store_precision=None,
    ):
        self.field_name_cardinality_dict = field_name_cardinality_dict
        self.grid_shape = tuple(int(s) for s in grid_shape)
        self.voxel_size = voxel_size
        self.offset = tuple(offset)
        self.unit_convertor = unit_convertor

        from xlb import DefaultConfig

        if store_precision is None:
            self.store_precision = DefaultConfig.default_precision_policy.store_precision
        else:
            self.store_precision = store_precision
        self.store_dtype = self.store_precision.wp_dtype

        self._avg_sum = {}
        self._avg_weight = 0.0
        self._avg_active = False
        self._avg_final_cache = None
        self._avg_cache_weight = 0.0
        self._avg_cache = False

    # ------------------------------------------------------------------
    # Field readback + derived quantities
    # ------------------------------------------------------------------
    def get_fields_data(self, field_neon_dict, derived=None):
        """Read requested Neon fields back to numpy, keyed ``{name}_{card}``.

        Matches MultiresIO.get_fields_data's naming convention exactly (e.g.
        a cardinality-3 ``velocity`` becomes ``velocity_0``/``_1``/``_2``).
        """
        fields_data = {}
        for name, field in field_neon_dict.items():
            cardinality = self.field_name_cardinality_dict[name]
            arr = neon_field_to_numpy(field, cardinality, self.grid_shape, self.store_dtype)
            for c in range(cardinality):
                fields_data[f"{name}_{c}"] = arr[c]

        if derived:
            self._compute_derived_fields(fields_data, derived)
        return fields_data

    def _compute_derived_fields(self, fields_data, derived):
        if self.unit_convertor is None:
            raise ValueError("derived fields require a unit_convertor")

        cs2 = 1.0 / 3.0
        ulb = self.unit_convertor.velocity_lbm_unit
        rho_ref = 1.0
        q_dyn = 0.5 * rho_ref * ulb**2

        for dep_name, deps in self.DERIVED_FIELD_DEPS.items():
            if dep_name not in derived:
                continue
            for dep in deps:
                assert f"{dep}_0" in fields_data, f"derived field '{dep_name}' needs '{dep}' in the requested fields"

            rho = fields_data["density_0"]
            if dep_name == "pressure":
                p_lattice = cs2 * rho
                fields_data["pressure_0"] = self.unit_convertor.pressure_to_physical(p_lattice).astype(np.float32)
            elif dep_name == "Cp":
                fields_data["Cp_0"] = ((cs2 * (rho - rho_ref)) / q_dyn).astype(np.float32)
            elif dep_name in ("CpTotal", "CpTotalLoss", "qdyn"):
                u0, u1, u2 = fields_data["velocity_0"], fields_data["velocity_1"], fields_data["velocity_2"]
                u_sq = u0**2 + u1**2 + u2**2
                if dep_name == "qdyn":
                    pressure_scale = self.unit_convertor.reference_density * self.unit_convertor.reference_velocity**2
                    fields_data["qdyn_0"] = (0.5 * rho * u_sq * pressure_scale).astype(np.float32)
                else:
                    p_static = cs2 * (rho - 1.0)
                    p_total = p_static + 0.5 * rho * u_sq
                    cp_total = p_total / q_dyn
                    if dep_name == "CpTotal":
                        fields_data["CpTotal_0"] = cp_total.astype(np.float32)
                    else:
                        fields_data["CpTotalLoss_0"] = (1.0 - cp_total).astype(np.float32)

    # ------------------------------------------------------------------
    # HDF5 + XDMF
    # ------------------------------------------------------------------
    def _save_xdmf(self, h5_filename, xdmf_filename, fields_data):
        nx, ny, nz = self.grid_shape
        ox, oy, oz = self.offset
        dx = dy = dz = self.voxel_size
        h5_basename = os.path.basename(h5_filename)

        attrs = []
        for field_name in fields_data:
            attrs.append(
                f'            <Attribute Name="{field_name}" AttributeType="Scalar" Center="Cell">\n'
                f'                <DataItem Dimensions="{nz} {ny} {nx}" NumberType="Float" Precision="4" Format="HDF">\n'
                f"                    {h5_basename}:/Fields/{field_name}\n"
                f"                </DataItem>\n"
                f"            </Attribute>\n"
            )

        # Topology is declared in POINT counts (one more per axis than the
        # per-voxel/cell field data, whose shape is the actual grid_shape).
        xdmf = (
            '<?xml version="1.0" ?>\n'
            '<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n'
            '<Xdmf Version="3.0">\n'
            "    <Domain>\n"
            '        <Grid Name="VoxelGrid" GridType="Uniform">\n'
            f'            <Topology TopologyType="3DCoRectMesh" Dimensions="{nz + 1} {ny + 1} {nx + 1}"/>\n'
            '            <Geometry GeometryType="ORIGIN_DXDYDZ">\n'
            '                <DataItem Dimensions="3" NumberType="Float" Precision="8" Format="XML">\n'
            f"                    {oz} {oy} {ox}\n"
            "                </DataItem>\n"
            '                <DataItem Dimensions="3" NumberType="Float" Precision="8" Format="XML">\n'
            f"                    {dz} {dy} {dx}\n"
            "                </DataItem>\n"
            "            </Geometry>\n"
            + "".join(attrs)
            + "        </Grid>\n"
            "    </Domain>\n"
            "</Xdmf>\n"
        )
        with open(xdmf_filename, "w") as f:
            f.write(xdmf)

    def _save_hdf5_file(self, output_filename, fields_data, compression, compression_opts):
        import h5py

        with h5py.File(output_filename + ".h5", "w") as hf:
            grp = hf.create_group("Fields")
            for field_name, arr in fields_data.items():
                grp.create_dataset(
                    field_name,
                    data=arr.astype(np.float32),
                    compression=compression,
                    compression_opts=compression_opts,
                    chunks=True,
                )

    def to_hdf5(self, output_filename, field_neon_dict, compression="gzip", compression_opts=0, derived=None):
        fields_data = self.get_fields_data(field_neon_dict, derived=derived)
        self._save_hdf5_file(output_filename, fields_data, compression, compression_opts)
        self._save_xdmf(output_filename + ".h5", output_filename + ".xmf", fields_data)

    # ------------------------------------------------------------------
    # Time averaging (simple weighted mean, mirrors MultiresIO exactly)
    # ------------------------------------------------------------------
    def start_time_average(self):
        self._avg_sum = {}
        self._avg_weight = 0.0
        self._avg_active = True
        self._avg_final_cache = None
        self._avg_cache_weight = 0.0
        self._avg_cache = False

    def accumulate_time_average(self, field_neon_dict, weight=1.0, derived=None):
        fields_data = self.get_fields_data(field_neon_dict, derived=derived)
        for k, v in fields_data.items():
            v64 = v.astype(np.float64)
            if k not in self._avg_sum:
                self._avg_sum[k] = weight * v64
            else:
                self._avg_sum[k] += weight * v64
        self._avg_weight += weight

    def finalize_time_average(self, keep_state=True):
        if self._avg_cache and self._avg_cache_weight == self._avg_weight and self._avg_final_cache is not None:
            result = self._avg_final_cache
        else:
            result = {k: (v / max(self._avg_weight, 1e-12)).astype(np.float32) for k, v in self._avg_sum.items()}
            self._avg_final_cache = result
            self._avg_cache_weight = self._avg_weight
            self._avg_cache = True

        if not keep_state:
            self._avg_sum = {}
            self._avg_weight = 0.0
            self._avg_active = False
            self._avg_final_cache = None
            self._avg_cache_weight = 0.0
            self._avg_cache = False

        return result

    def to_hdf5_time_average(self, output_filename, compression="gzip", compression_opts=0, keep_state=True):
        fields_data = self.finalize_time_average(keep_state=keep_state)
        self._save_hdf5_file(output_filename, fields_data, compression, compression_opts)
        self._save_xdmf(output_filename + ".h5", output_filename + ".xmf", fields_data)

    # ------------------------------------------------------------------
    # Slice images (axis-aligned only -- direct indexing, no KDTree needed)
    # ------------------------------------------------------------------
    def _axis_from_normal(self, plane_normal):
        n = np.asarray(plane_normal, dtype=float)
        axis = int(np.argmax(np.abs(n)))
        if not np.isclose(np.abs(n[axis]), np.linalg.norm(n), atol=1e-6):
            raise NotImplementedError("SingleResIO slice export only supports axis-aligned planes (single non-zero normal component).")
        return axis

    def _slice_scalar_field(self, fields_data, field_name, component):
        if component is not None:
            return fields_data[f"{field_name}_{component}"]
        keys = [k for k in fields_data if k.startswith(field_name + "_")]
        if len(keys) == 1:
            return fields_data[keys[0]]
        stacked = np.stack([fields_data[k] for k in sorted(keys)], axis=0)
        return np.sqrt(np.sum(stacked**2, axis=0))

    def _save_slice_png(self, output_filename, field_name, plane_2d, show_axes, show_colorbar, normalize, cmap):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cmap = cmap or matplotlib.cm.nipy_spectral

        data = plane_2d
        vmin, vmax = None, None
        if normalize is not None:
            if isinstance(normalize, (tuple, list)):
                vmin, vmax = normalize
            else:
                vmin, vmax = 0.0, float(normalize)
            data = np.clip((plane_2d - vmin) / max(vmax - vmin, 1e-12), 0.0, 1.0)
            vmin, vmax = 0.0, 1.0

        path = f"{output_filename}_{field_name}.png"
        if show_colorbar or show_axes:
            fig = plt.figure()
            plt.imshow(data, cmap=cmap, origin="lower", aspect="equal", vmin=vmin, vmax=vmax)
            if show_colorbar:
                plt.colorbar()
            if not show_axes:
                plt.axis("off")
            plt.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
        else:
            if vmin is not None:
                plt.imsave(path, data, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
            else:
                plt.imsave(path, data, cmap=cmap, origin="lower")

    def _slice_at(self, fields_data, field_name, component, axis, index_frac):
        scalar = self._slice_scalar_field(fields_data, field_name, component)
        idx = int(round(index_frac * (scalar.shape[axis] - 1)))
        idx = max(0, min(scalar.shape[axis] - 1, idx))
        # Transpose so the first remaining axis (image rows/vertical) is the
        # "taller" spatial axis and the second (columns/horizontal) is the
        # other -- e.g. an X-normal slice keeps (Y, Z) as sliced but we want
        # (Z, Y) so it displays as a normal wide side-on image, not rotated.
        if axis == 0:
            return scalar[idx, :, :].T
        elif axis == 1:
            return scalar[:, idx, :].T
        else:
            return scalar[:, :, idx].T

    def to_slice_image(
        self,
        output_filename,
        field_neon_dict,
        plane_point,
        plane_normal,
        grid_res=None,
        bounds=None,
        show_axes=False,
        show_colorbar=False,
        slice_thickness=None,
        normalize=None,
        cmap=None,
    ):
        """Axis-aligned slice through the dense grid, saved as PNG.

        ``plane_point``'s component along the normal axis is treated as a
        0-1 fraction of that axis's extent (matching how the caller in
        windtunnel_single_res.py already uses it, e.g. plane_point=(1,0,0)
        with plane_normal=(0,1,0) means "slice at 100% along Y").
        """
        axis = self._axis_from_normal(plane_normal)
        index_frac = float(np.asarray(plane_point)[axis])
        fields_data = self.get_fields_data(field_neon_dict)
        for field_name in field_neon_dict:
            plane_2d = self._slice_at(fields_data, field_name, None, axis, index_frac)
            self._save_slice_png(output_filename, f"{field_name}_magnitude", plane_2d, show_axes, show_colorbar, normalize, cmap)

    def to_slice_image_time_average(
        self,
        output_filename,
        field_base_name,
        plane_point,
        plane_normal,
        component=None,
        grid_res=None,
        bounds=None,
        show_axes=False,
        show_colorbar=False,
        slice_thickness=None,
        normalize=1.0,
        cmap=None,
        keep_state=True,
    ):
        axis = self._axis_from_normal(plane_normal)
        index_frac = float(np.asarray(plane_point)[axis])
        fields_data = self.finalize_time_average(keep_state=keep_state)
        plane_2d = self._slice_at(fields_data, field_base_name, component, axis, index_frac)
        self._save_slice_png(output_filename, field_base_name, plane_2d, show_axes, show_colorbar, normalize, cmap)

    def _to_slice_image_single_field(self, *args, **kwargs):
        raise NotImplementedError("Use to_slice_image/to_slice_image_time_average directly for single-resolution.")

    # ------------------------------------------------------------------
    # Deferred: body-fitted surface / isosurface export
    # ------------------------------------------------------------------
    @property
    def centroids(self):
        raise NotImplementedError(
            "SingleResIO has no unstructured centroids -- a dense grid's cell centers are implicit "
            "(origin + (i+0.5,j+0.5,k+0.5)*voxel_size). Not needed by to_hdf5/to_slice_image."
        )

    def to_surface_vtk_time_average(self, *args, **kwargs):
        print("WARNING: SingleResIO.to_surface_vtk_time_average is not implemented for single-resolution -- skipping surface field export.")

    def to_isosurface_stl_time_average(self, *args, **kwargs):
        print("WARNING: SingleResIO.to_isosurface_stl_time_average is not implemented for single-resolution -- skipping isosurface export.")
