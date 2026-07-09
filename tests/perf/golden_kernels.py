"""
Golden numerical harness for the hot-path Warp functionals.

Purpose
-------
The performance work on the collision / equilibrium / moment operators consists
mostly of *algebra-preserving* refactors (sparse/static unrolling, reciprocal
multiplies, vector-lifetime reduction, dead-code removal). These must not change
the numerical result. This harness pins down the current outputs on a fixed set
of pseudo-random inputs so a refactor can be validated in seconds without a full
end-to-end Cd/Cl run.

The WARP backend kernels exercised here share the identical ``@wp.func``
functional with the NEON production path (``_construct_neon`` reuses
``_construct_warp``), so validating the Warp functional validates the math that
runs in the real solver.

Usage
-----
    # 1. On known-good code, capture the reference outputs:
    python tests/perf/golden_kernels.py --generate

    # 2. After a refactor, check the outputs still match:
    python tests/perf/golden_kernels.py --check

    # Tolerances (rel/abs) can be loosened for intentionally non-bit-exact
    # changes (e.g. reciprocal-multiply, reordered summation):
    python tests/perf/golden_kernels.py --check --rtol 1e-4 --atol 1e-6

    # Register / spill baseline (best-effort; prints ptxas -v if Warp exposes it):
    python tests/perf/golden_kernels.py --reginfo

Notes
-----
* Runs in FP32/FP32 by default so the comparison reflects the compute functional
  only, not FP16 store rounding. Use --policy FP32FP16 to additionally stress the
  store path.
* The reference file is written next to this script as ``golden_kernels_<policy>.npz``.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import xlb
from xlb.compute_backend import ComputeBackend
from xlb import DefaultConfig
from xlb.grid import grid_factory
from xlb.operator.equilibrium import QuadraticEquilibrium
from xlb.operator.macroscopic import SecondMoment
from xlb.operator.collision.smagorinsky_les_kbc import SmagorinskyLESKBC


HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20240708
GRID_SHAPE = (16, 16, 16)  # 4096 cells; enough spread to hit clamp/floor/bounds branches
OMEGA = 1.9  # near the stability edge, exercises the Smagorinsky/KBC gamma path


def _policy_from_name(name: str) -> "xlb.PrecisionPolicy":
    return getattr(xlb.PrecisionPolicy, name)


def init_env(policy_name: str):
    policy = _policy_from_name(policy_name)
    vel_set = xlb.velocity_set.D3Q27(precision_policy=policy, compute_backend=ComputeBackend.WARP)
    xlb.init(
        default_precision_policy=policy,
        default_backend=ComputeBackend.WARP,
        velocity_set=vel_set,
    )
    return policy


def make_inputs():
    """Deterministic pseudo-random rho/u and a perturbed, near-equilibrium f.

    The velocity range intentionally spans the equilibrium clamp (|u|=0.7) and the
    per-cell perturbation scale is swept so KBC gamma feasibility bounds and the
    feq floor are exercised somewhere on the grid.
    """
    rng = np.random.default_rng(SEED)
    n = int(np.prod(GRID_SHAPE))
    q = DefaultConfig.velocity_set.q
    d = DefaultConfig.velocity_set.d

    # Density: 1.0 +/- 5%
    rho = (1.0 + 0.05 * rng.standard_normal(n)).astype(np.float64)

    # Velocity: mostly small, but a fraction pushed toward/over the 0.7 clamp.
    u = 0.15 * rng.standard_normal((d, n))
    hot = rng.random(n) < 0.15
    u[:, hot] *= 5.0  # some cells to |u| ~ 0.7-1.0 to hit clamps
    u = u.astype(np.float64)

    rho_field = np.ascontiguousarray(rho.reshape((1,) + GRID_SHAPE))
    u_field = np.ascontiguousarray(u.reshape((d,) + GRID_SHAPE))
    return rho_field, u_field, q, d


def _to_field(grid, cardinality, np_data):
    """Create a store-precision field and load np_data, cast to the field's dtype."""
    field = grid.create_field(cardinality=cardinality)
    target_dtype = field.numpy().dtype  # actual storage dtype (FP32 or FP16)
    field.assign(np.ascontiguousarray(np_data.astype(target_dtype)))
    return field


def run_operators():
    """Run equilibrium -> perturb -> second-moment + KBC collision; return numpy outputs.

    Assumes xlb.init() has already set the default backend/precision/velocity set.

    We drive each operator's shared ``warp_functional`` through purpose-built
    kernels here rather than the operators' own WARP ``warp_implementation``.
    This is deliberate: the NEON production path reuses ``warp_functional`` (via
    ``_construct_neon``), so this validates the exact math the solver runs, and it
    sidesteps the standalone WARP kernels (some of which reference ``self`` and
    won't even compile — they are unused dead paths in the Neon build).
    """
    import warp as wp

    policy = DefaultConfig.default_precision_policy
    cdt = policy.compute_precision.wp_dtype
    sdt = policy.store_precision.wp_dtype
    q = DefaultConfig.velocity_set.q
    d = DefaultConfig.velocity_set.d
    pi_dim = d * (d + 1) // 2

    _f_vec = wp.vec(q, dtype=cdt)
    _u_vec = wp.vec(d, dtype=cdt)

    eq = QuadraticEquilibrium()
    sm = SecondMoment()
    kbc = SmagorinskyLESKBC(smagorinsky_constant=0.1)

    eq_func = eq.warp_functional
    sm_func = sm.warp_functional
    kbc_func = kbc.warp_functional

    @wp.kernel
    def eq_kernel(rho: wp.array4d(dtype=sdt), u: wp.array4d(dtype=sdt), out: wp.array4d(dtype=sdt)):
        i, j, k = wp.tid()
        _u = _u_vec()
        for l in range(d):
            _u[l] = cdt(u[l, i, j, k])
        _rho = cdt(rho[0, i, j, k])
        feq = eq_func(_rho, _u)
        for l in range(q):
            out[l, i, j, k] = sdt(feq[l])

    @wp.kernel
    def sm_kernel(fneq: wp.array4d(dtype=sdt), out: wp.array4d(dtype=sdt)):
        i, j, k = wp.tid()
        _f = _f_vec()
        for l in range(q):
            _f[l] = cdt(fneq[l, i, j, k])
        pi = sm_func(_f)
        for c in range(pi_dim):
            out[c, i, j, k] = sdt(pi[c])

    @wp.kernel
    def kbc_kernel(
        f: wp.array4d(dtype=sdt),
        feq: wp.array4d(dtype=sdt),
        rho: wp.array4d(dtype=sdt),
        u: wp.array4d(dtype=sdt),
        out: wp.array4d(dtype=sdt),
        omega: cdt,
    ):
        i, j, k = wp.tid()
        _f = _f_vec()
        _feq = _f_vec()
        for l in range(q):
            _f[l] = cdt(f[l, i, j, k])
            _feq[l] = cdt(feq[l, i, j, k])
        _u = _u_vec()
        for l in range(d):
            _u[l] = cdt(u[l, i, j, k])
        _rho = cdt(rho[0, i, j, k])
        fo = kbc_func(_f, _feq, _rho, _u, omega)
        for l in range(q):
            out[l, i, j, k] = sdt(fo[l])

    grid = grid_factory(GRID_SHAPE)
    rho_np, u_np, _, _ = make_inputs()
    rho = _to_field(grid, 1, rho_np)
    u = _to_field(grid, d, u_np)

    # --- Equilibrium ---
    feq = grid.create_field(cardinality=q)
    wp.launch(eq_kernel, dim=GRID_SHAPE, inputs=[rho, u, feq])
    feq_np = feq.numpy().astype(np.float64)

    # --- Build a perturbed f around feq (per-cell varying perturbation) ---
    rng = np.random.default_rng(SEED + 1)
    w = np.asarray(DefaultConfig.velocity_set.w, dtype=np.float64).reshape((q,) + (1,) * len(GRID_SHAPE))
    scale = (0.05 + 0.45 * rng.random((1,) + GRID_SHAPE))  # 5%..50% perturbation, per cell
    noise = rng.standard_normal((q,) + GRID_SHAPE)
    f_np = feq_np + scale * noise * w
    f = _to_field(grid, q, f_np)

    # --- Second moment of fneq = f - feq (as used inside KBC shear decomposition) ---
    fneq_np = f_np - feq_np
    fneq = _to_field(grid, q, fneq_np)
    pi = grid.create_field(cardinality=pi_dim)
    wp.launch(sm_kernel, dim=GRID_SHAPE, inputs=[fneq, pi])
    pi_np = pi.numpy().astype(np.float64)

    # --- Smagorinsky-LES-KBC collision ---
    fout = grid.create_field(cardinality=q)
    wp.launch(kbc_kernel, dim=GRID_SHAPE, inputs=[f, feq, rho, u, fout, cdt(OMEGA)])
    fout_np = fout.numpy().astype(np.float64)

    return {
        "feq": feq_np,
        "second_moment": pi_np,
        "kbc_fout": fout_np,
    }


def _ref_path(policy_name: str) -> str:
    return os.path.join(HERE, f"golden_kernels_{policy_name}.npz")


def cmd_generate(policy_name: str):
    init_env(policy_name)
    outputs = run_operators()
    path = _ref_path(policy_name)
    np.savez_compressed(path, **outputs)
    print(f"[generate] policy={policy_name}  wrote {path}")
    for k, v in outputs.items():
        print(f"    {k:16s} shape={v.shape} finite={np.isfinite(v).all()} "
              f"min={v.min():.4e} max={v.max():.4e}")


def cmd_check(policy_name: str, rtol: float, atol: float) -> int:
    path = _ref_path(policy_name)
    if not os.path.exists(path):
        print(f"[check] ERROR: no reference at {path}. Run --generate on known-good code first.")
        return 2
    ref = np.load(path)
    init_env(policy_name)
    cur = run_operators()

    ok = True
    print(f"[check] policy={policy_name}  rtol={rtol} atol={atol}")
    for k in ref.files:
        a, b = ref[k], cur[k]
        if a.shape != b.shape:
            print(f"    {k:16s} SHAPE MISMATCH ref={a.shape} cur={b.shape}")
            ok = False
            continue
        adiff = np.abs(a - b)
        denom = np.maximum(np.abs(a), 1e-30)
        rdiff = adiff / denom
        max_abs = float(adiff.max())
        max_rel = float(rdiff.max())
        passed = np.allclose(b, a, rtol=rtol, atol=atol) and np.isfinite(b).all()
        ok = ok and passed
        print(f"    {k:16s} {'PASS' if passed else 'FAIL'}  "
              f"max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  finite={np.isfinite(b).all()}")
    print("[check] RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def cmd_reginfo(policy_name: str):
    """Best-effort register / spill report via ptxas -v, no profiler needed.

    Works inside Docker without GPU performance-counter permissions (unlike ncu).
    Warp's exact knob varies by version, so we try the known ones and, failing
    that, print instructions.
    """
    import warp as wp

    tried = []
    for attr, val in (("verbose", True), ("verbose_warnings", True)):
        if hasattr(wp.config, attr):
            setattr(wp.config, attr, val)
            tried.append(attr)
    # Newer Warp exposes ptxas passthrough; try a couple of spellings.
    for attr in ("ptxas_options", "cuda_ptxas_options"):
        if hasattr(wp.config, attr):
            try:
                setattr(wp.config, attr, ["-v"])
                tried.append(attr)
            except Exception:
                pass
    print(f"[reginfo] enabled Warp config flags: {tried or 'none found'}")
    print("[reginfo] compiling kernels; look for ptxas lines like "
          "'Used N registers, M bytes spill stores' in the output below.\n")
    wp.clear_kernel_cache()
    init_env(policy_name)
    run_operators()
    print("\n[reginfo] done. If no register lines appeared, set the environment "
          "variable WARP_VERBOSE=1 and/or check your Warp version's ptxas passthrough.")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--generate", action="store_true", help="capture reference outputs")
    p.add_argument("--check", action="store_true", help="compare current outputs to reference")
    p.add_argument("--reginfo", action="store_true", help="print best-effort register/spill info")
    p.add_argument("--policy", default="FP32FP32", help="PrecisionPolicy name (default FP32FP32)")
    p.add_argument("--rtol", type=float, default=1e-5)
    p.add_argument("--atol", type=float, default=1e-6)
    args = p.parse_args(argv)

    if not (args.generate or args.check or args.reginfo):
        p.error("choose one of --generate / --check / --reginfo")

    if args.generate:
        cmd_generate(args.policy)
    if args.reginfo:
        cmd_reginfo(args.policy)
    if args.check:
        return cmd_check(args.policy, args.rtol, args.atol)
    return 0


if __name__ == "__main__":
    sys.exit(main())
