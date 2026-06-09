"""
Post-processing helpers for streamwise-binned boundary forces.

Consumes the ``(num_bins, d)`` force distribution produced by
:class:`~xlb.operator.force.binned_momentum_transfer.MultiresBinnedMomentumTransfer`
(in lattice units) and turns it into cumulative aerodynamic-coefficient curves
along the vehicle length, a CSV, and styled PNG plots.

All forces are in lattice units; coefficients use the same normalisation as the
scalar Cd/Cl in the solver:  C = 2 * F / (ulb**2 * A_ref).
"""

import csv

import numpy as np

# Force-vector component indices (XLB convention: x = streamwise/drag, z = vertical/lift)
_DRAG_AXIS = 0
_LIFT_AXIS = 2


def _bin_centers_physical(num_bins, bin_width_cells, voxel_size):
    """Physical streamwise position (m) of each bin centre, measured from the
    start of the binned region (the vehicle nose)."""
    idx = np.arange(num_bins, dtype=np.float64)
    return (idx + 0.5) * bin_width_cells * voxel_size


def compute_cumulative_coefficients(force_bins, ulb, reference_area, num_samples=1):
    """Convert binned lattice forces into cumulative Cd(x) and Cl(x).

    Parameters
    ----------
    force_bins : np.ndarray, shape (num_bins, d)
        Accumulated lattice force per streamwise bin (summed over ``num_samples``
        timesteps).
    ulb : float
        Lattice inlet velocity.
    reference_area : float
        Reference (projected frontal) area in lattice cells.
    num_samples : int
        Number of timesteps accumulated into ``force_bins`` (for time-averaging).

    Returns
    -------
    dict with keys: cd_per_bin, cl_per_bin, cd_cumulative, cl_cumulative.
    """
    force_bins = np.asarray(force_bins, dtype=np.float64)
    if num_samples > 1:
        force_bins = force_bins / float(num_samples)

    norm = 2.0 / (ulb**2 * reference_area)
    cd_per_bin = force_bins[:, _DRAG_AXIS] * norm
    cl_per_bin = force_bins[:, _LIFT_AXIS] * norm

    return {
        "cd_per_bin": cd_per_bin,
        "cl_per_bin": cl_per_bin,
        "cd_cumulative": np.cumsum(cd_per_bin),
        "cl_cumulative": np.cumsum(cl_per_bin),
    }


def _plot_cumulative(x, y, ylabel, outfile, label="XLB", color="tab:red"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, y, color=color, linewidth=2.0, label=label)
    ax.set_xlabel("Length [m]")
    ax.set_ylabel(ylabel)
    ax.legend(title="Legend:", frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(left=0.0)
    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)
    return outfile


def plot_cumulative_cl(
    force_bins,
    ulb,
    reference_area,
    voxel_size,
    bin_width_cells,
    output_prefix,
    num_samples=1,
    label="XLB",
):
    """Write ``{output_prefix}_cumulative_cl.png`` and ``..._cd.png`` styled to
    match the streamwise force-development plots.

    Returns the computed coefficient dict (see
    :func:`compute_cumulative_coefficients`).
    """
    num_bins = np.asarray(force_bins).shape[0]
    x = _bin_centers_physical(num_bins, bin_width_cells, voxel_size)
    coeffs = compute_cumulative_coefficients(force_bins, ulb, reference_area, num_samples)

    _plot_cumulative(
        x,
        coeffs["cl_cumulative"],
        r"Lift force development along the vehicle, $C_L$ [-]",
        f"{output_prefix}_cumulative_cl.png",
        label=label,
        color="tab:red",
    )
    _plot_cumulative(
        x,
        coeffs["cd_cumulative"],
        r"Drag force development along the vehicle, $C_D$ [-]",
        f"{output_prefix}_cumulative_cd.png",
        label=label,
        color="tab:red",
    )
    return coeffs


def save_force_distribution_csv(
    force_bins,
    ulb,
    reference_area,
    voxel_size,
    bin_width_cells,
    outfile,
    num_samples=1,
):
    """Write per-bin and cumulative coefficients to ``outfile`` as CSV."""
    num_bins = np.asarray(force_bins).shape[0]
    x = _bin_centers_physical(num_bins, bin_width_cells, voxel_size)
    coeffs = compute_cumulative_coefficients(force_bins, ulb, reference_area, num_samples)

    with open(outfile, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Length_m", "Cd_per_bin", "Cl_per_bin", "Cd_cumulative", "Cl_cumulative"])
        for i in range(num_bins):
            writer.writerow(
                [
                    x[i],
                    coeffs["cd_per_bin"][i],
                    coeffs["cl_per_bin"][i],
                    coeffs["cd_cumulative"][i],
                    coeffs["cl_cumulative"][i],
                ]
            )
    return outfile


def compute_force_statistics(force_bins, ulb, reference_area, num_samples=1):
    """Return summary stats: total Cd/Cl and the streamwise location of peak
    local drag/lift contribution."""
    coeffs = compute_cumulative_coefficients(force_bins, ulb, reference_area, num_samples)
    return {
        "total_cd": float(coeffs["cd_cumulative"][-1]),
        "total_cl": float(coeffs["cl_cumulative"][-1]),
        "peak_cd_bin": int(np.argmax(coeffs["cd_per_bin"])),
        "peak_cl_bin": int(np.argmax(np.abs(coeffs["cl_per_bin"]))),
    }


def print_force_statistics(stats, logger=print):
    """Pretty-print the dict from :func:`compute_force_statistics`."""
    logger(
        "Cumulative force distribution: "
        f"total Cd = {stats['total_cd']:.4f}, total Cl = {stats['total_cl']:.4f} "
        f"(peak drag bin {stats['peak_cd_bin']}, peak lift bin {stats['peak_cl_bin']})"
    )
