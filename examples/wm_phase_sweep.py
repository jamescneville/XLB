"""
Wall-model phase-sensitivity sweep (TEST HARNESS).

Runs the windtunnel_json.py solver repeatedly on one project.json while sliding the geometry through
sub-voxel offsets (settings.testPhaseOffset), for several wall-model variants and voxelization methods,
then summarises how much Cd / Cl / Cy move with grid phase.

    python3 examples/wm_phase_sweep.py -i examples/cfd/stl-files/project.json
    python3 examples/wm_phase_sweep.py -i ... --methods AABB_CLOSE --variants base,fix1,fix2 --nphase 8
    python3 examples/wm_phase_sweep.py -i ... --dry-run          # list the runs only
    python3 examples/wm_phase_sweep.py -i ... --analyze-only     # rebuild summary/plots from stored results

Variant spec:  name[:key=value[:key=value...]]
    base   wallModelFix = 0 (original behaviour)
    fix1   wallModelFix = 1 (imposed wall velocity evaluated at a fixed height)
    fix2   wallModelFix = 2 (fix1 plus u_F sampled at a fixed height)
  keys:  h  = settings.wallModelRefHeight   (voxels; default |n|_1)
         s  = settings.wallModelRefScale
         sh = settings.wallModelSampleHeight (voxels; default 1.5)
  e.g.  fix1:h=0.75   fix2:h=1.0:sh=1.0

Every run is a normal solver launch with settings.forceOnly = true (forces only, no fields/slices/iso).
Run folders and per-run json/log files are created next to the project json; they are all named
wmsweep_<method>_<variant>_pNN*. Results accumulate in wm_sweep_results.json (re-running resumes).
"""

import argparse
import copy
import csv
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN = 0.6180339887498949
VOX_METHOD_JSON = {"AABB_CLOSE": "AABB", "RAY": "RAY"}
DEFAULT_VARIANTS = "base,fix1,fix2"
DEFAULT_RAY_VARIANTS = "base,fix1:h=0.75,fix1:h=1.0"


def parse_variant(spec):
    parts = spec.split(":")
    name = parts[0].strip().lower()
    mode = {"base": 0, "fix1": 1, "fix2": 2}[name]
    st = {"wallModelFix": mode}
    keymap = {"h": "wallModelRefHeight", "s": "wallModelRefScale", "sh": "wallModelSampleHeight"}
    for kv in parts[1:]:
        k, v = kv.split("=")
        st[keymap[k.strip()]] = float(v)
    label = spec.replace(":", "_").replace("=", "").replace(".", "p")
    return label, st


def make_offsets(n, custom=None):
    if custom:
        return [tuple(float(x) for x in o.split(",")) for o in custom.split(";")]
    # x sweeps one voxel evenly; y is decorrelated with the golden ratio so x and y phases are not locked.
    return [(k / n, (k * GOLDEN) % 1.0, 0.0) for k in range(n)]


def read_force_series(run_dir, crossover_pct):
    """Post-crossover mean / std / block standard error for each coefficient from drag_lift.csv."""
    path = os.path.join(run_dir, "drag_lift.csv")
    if not os.path.exists(path):
        return {}
    rows = []
    with open(path) as fh:
        rd = csv.reader(fh)
        header = next(rd)
        for r in rd:
            try:
                rows.append([float(x) if x not in ("None", "") else math.nan for x in r])
            except ValueError:
                continue
    if not rows:
        return {}
    a = np.array(rows)
    steps = a[:, 0]
    thresh = crossover_pct / 100.0 * (steps.max() + 1)
    sel = a[steps >= thresh]
    if len(sel) <= 1:
        sel = a
    out = {}
    for j, name in enumerate(header):
        if j == 0:
            continue
        col = sel[:, j]
        col = col[~np.isnan(col)]
        if len(col) < 5:
            continue
        blocks = np.array_split(col, 5)
        bm = np.array([b.mean() for b in blocks])
        out[name.lower()] = {
            "mean": float(col.mean()),
            "std": float(col.std()),
            "se": float(bm.std(ddof=1) / math.sqrt(len(bm))),
            "n": int(len(col)),
        }
    return out


_TAG = ""


def run_id(method, label, k):
    return f"wmsweep{'_' + _TAG if _TAG else ''}_{method}_{label}_p{k:02d}"


def build_run_json(base, method, vstate, offset, rid, args):
    j = copy.deepcopy(base)
    st = j["settings"]
    st["forceOnly"] = not args.full_output
    for kv in args.set:
        k, v = kv.split("=", 1)
        try:
            st[k] = json.loads(v)
        except ValueError:
            st[k] = v
    st["testPhaseOffset"] = list(offset)
    for k in ("wallModelFix", "wallModelRefHeight", "wallModelRefScale", "wallModelSampleHeight"):
        st.pop(k, None)
    st.update(vstate)
    if args.flow_passes is not None:
        st["flowPasses"] = args.flow_passes
    if args.iterations is not None:
        st["iterations"] = args.iterations
        st["flowPasses"] = 0
    if args.crossover is not None:
        st["crossover"] = args.crossover
    st["forcePrintFreq"] = args.force_print_freq
    if "liveView" in st:
        st["liveView"]["enabled"] = False
    j["BCtypes"]["car_voxelization"] = VOX_METHOD_JSON[method]
    j["outputName"] = rid
    return j


def summarize(results, out_dir, ref_variant="base"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = [r for r in results if r.get("returncode") == 0 and r.get("results")]
    if not ok:
        print("No completed runs to summarise.")
        return
    keys = sorted({(r["method"], r["variant"]) for r in ok})
    methods = sorted({m for m, _ in keys})
    rows = []
    base_mean = {}
    for m, v in keys:
        rs = sorted([r for r in ok if r["method"] == m and r["variant"] == v], key=lambda r: r["phase"])
        cd = np.array([r["results"]["cd"] for r in rs])
        cl = np.array([r["results"]["cl"] for r in rs])
        cy = np.array([r["results"].get("cy", np.nan) for r in rs])
        se = [r["series"].get("cd", {}).get("se", np.nan) for r in rs]
        row = dict(
            method=m,
            variant=v,
            n=len(rs),
            cd_mean=cd.mean(),
            cd_std=cd.std(ddof=1) if len(cd) > 1 else np.nan,
            cd_p2p=np.ptp(cd),
            cd_se=float(np.nanmean(se)),
            cl_mean=cl.mean(),
            cl_std=cl.std(ddof=1) if len(cl) > 1 else np.nan,
            cl_p2p=np.ptp(cl),
            cy_abs_mean=np.nanmean(np.abs(cy)),
            cy_std=np.nanstd(cy, ddof=1) if len(cy) > 1 else np.nan,
            cy_abs_max=np.nanmax(np.abs(cy)),
        )
        rows.append(row)
        if v == ref_variant:
            base_mean[m] = row["cd_mean"]
    for r in rows:
        b = base_mean.get(r["method"])
        r["cd_shift_vs_base"] = (r["cd_mean"] - b) if b is not None else np.nan

    cols = [
        "method",
        "variant",
        "n",
        "cd_mean",
        "cd_shift_vs_base",
        "cd_std",
        "cd_p2p",
        "cd_se",
        "cl_mean",
        "cl_std",
        "cl_p2p",
        "cy_abs_mean",
        "cy_std",
        "cy_abs_max",
    ]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "wm_sweep_summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] if not isinstance(r[c], float) else f"{r[c]:.5f}" for c in cols])

    print("\n=== Phase-sweep summary (std/p2p are across phases; se = within-run block standard error of Cd) ===")
    print(f"{'method':11s} {'variant':16s} {'n':>2s} {'Cd':>7s} {'dCd/base':>9s} {'Cd std':>7s} {'Cd p2p':>7s} {'Cd se':>7s} {'Cl std':>7s} {'Cl p2p':>7s} {'|Cy|':>7s} {'Cy std':>7s}")
    for r in rows:
        print(
            f"{r['method']:11s} {r['variant']:16s} {r['n']:2d} {r['cd_mean']:7.4f} {r['cd_shift_vs_base']:+9.4f} {r['cd_std']:7.4f} "
            f"{r['cd_p2p']:7.4f} {r['cd_se']:7.4f} {r['cl_std']:7.4f} {r['cl_p2p']:7.4f} {r['cy_abs_mean']:7.4f} {r['cy_std']:7.4f}"
        )
    print("\nRead it as: a variant helps if Cd std / Cl std / |Cy| fall versus base at a similar mean Cd, and the phase")
    print("std is well above the se (otherwise the sweep is noise-limited; run more flow passes).")

    fig, axes = plt.subplots(3, len(methods), figsize=(6.5 * len(methods), 9), squeeze=False, sharex="col")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ci, m in enumerate(methods):
        vs = sorted({v for mm, v in keys if mm == m}, key=lambda v: (v != "base", v))
        for vi, v in enumerate(vs):
            rs = sorted([r for r in ok if r["method"] == m and r["variant"] == v], key=lambda r: r["phase"])
            x = [r["offset"][0] for r in rs]
            for ri, (key, lab) in enumerate((("cd", "Cd"), ("cl", "Cl"), ("cy", "Cy (side force)"))):
                y = [r["results"].get(key, np.nan) for r in rs]
                o = np.argsort(x)
                axes[ri][ci].plot(np.array(x)[o], np.array(y)[o], "o-", color=colors[vi % len(colors)], label=v)
                axes[ri][ci].set_ylabel(lab)
        axes[0][ci].set_title(m)
        axes[2][ci].set_xlabel("x phase offset (voxels; y offset decorrelated)")
        axes[0][ci].legend(fontsize=8)
        for ri in range(3):
            axes[ri][ci].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "wm_sweep_summary.png"), dpi=130)
    plt.close(fig)
    print(f"\nWrote {os.path.join(out_dir, 'wm_sweep_summary.csv')} and wm_sweep_summary.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--ifile", required=True, help="base project.json")
    ap.add_argument("--methods", default="AABB_CLOSE,RAY", help="comma list of AABB_CLOSE, RAY")
    ap.add_argument("--variants", default=DEFAULT_VARIANTS, help="variants for AABB_CLOSE (and RAY unless --variants-ray)")
    ap.add_argument("--variants-ray", default=DEFAULT_RAY_VARIANTS, help="variants for RAY")
    ap.add_argument("--nphase", type=int, default=6)
    ap.add_argument("--offsets", default=None, help='custom offsets "dx,dy,dz;dx,dy,dz;..." in voxels')
    ap.add_argument("--flow-passes", type=float, default=None, help="override settings.flowPasses")
    ap.add_argument("--iterations", type=int, default=None, help="override settings.iterations (and disable flowPasses)")
    ap.add_argument("--crossover", type=float, default=None, help="override settings.crossover (percent)")
    ap.add_argument("--force-print-freq", type=float, default=0.5, help="settings.forcePrintFreq (percent of run) for the sweep")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="extra settings override applied to every run (JSON value), e.g. --set deterministicAccumulation=true; repeatable")
    ap.add_argument("--full-output", action="store_true", help="do not use forceOnly: write fields/slices/iso/surface outputs for every run (large; use with few phases)")
    ap.add_argument("--tag", default="", help="name for this batch; keeps run folders, results and summary separate from other batches (so different settings never get mixed by resume)")
    ap.add_argument("--timeout-min", type=float, default=None)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--solver", default=os.path.join(REPO_ROOT, "examples", "windtunnel_json.py"))
    ap.add_argument("--clear-cache", action="store_true", help="do not reuse the Warp kernel cache between runs")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    global _TAG
    _TAG = args.tag.strip()
    proj_json = os.path.abspath(args.ifile)
    proj_dir = os.path.dirname(proj_json)
    with open(proj_json) as fh:
        base = json.load(fh)
    sfx = f"_{_TAG}" if _TAG else ""
    results_path = os.path.join(proj_dir, f"wm_sweep_results{sfx}.json")
    summary_dir = os.path.join(proj_dir, f"wm_sweep_summary{sfx}")
    results = []
    if os.path.exists(results_path) and not args.no_resume:
        with open(results_path) as fh:
            results = json.load(fh)
    if args.analyze_only:
        summarize(results, summary_dir)
        return 0

    offsets = make_offsets(args.nphase, args.offsets)
    plan = []
    for method in [m.strip().upper() for m in args.methods.split(",")]:
        spec = args.variants_ray if method == "RAY" else args.variants
        for vs in [s.strip() for s in spec.split(",") if s.strip()]:
            label, vstate = parse_variant(vs)
            for k, off in enumerate(offsets):
                plan.append((method, label, vstate, k, off))
    done = {(r["method"], r["variant"], r["phase"]) for r in results if r.get("returncode") == 0}
    todo = [p for p in plan if (p[0], p[1], p[3]) not in done]
    print(f"{len(plan)} runs planned, {len(plan) - len(todo)} already complete, {len(todo)} to run.")
    for method, label, vstate, k, off in todo:
        print(f"  {run_id(method, label, k)}  offset={tuple(round(o, 3) for o in off)}  {vstate}")
    if args.dry_run or not todo:
        summarize(results, summary_dir) if not args.dry_run else None
        return 0

    env = os.environ.copy()
    if not args.clear_cache:
        env["XLB_KEEP_KERNEL_CACHE"] = "1"
    t_all = time.time()
    for n, (method, label, vstate, k, off) in enumerate(todo):
        rid = run_id(method, label, k)
        run_json = build_run_json(base, method, vstate, off, rid, args)
        jpath = os.path.join(proj_dir, rid + ".json")
        lpath = os.path.join(proj_dir, rid + ".log")
        with open(jpath, "w") as fh:
            json.dump(run_json, fh, indent=2)
        t0 = time.time()
        print(f"[{n + 1}/{len(todo)}] {rid} ...", flush=True)
        try:
            with open(lpath, "w") as lf:
                cp = subprocess.run(
                    [args.python, args.solver, "-i", jpath],
                    cwd=REPO_ROOT,
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    env=env,
                    timeout=args.timeout_min * 60 if args.timeout_min else None,
                )
            rc = cp.returncode
        except subprocess.TimeoutExpired:
            rc = -9
        run_dir = os.path.join(proj_dir, rid)
        rec = dict(id=rid, method=method, variant=label, phase=k, offset=list(off), settings=vstate, returncode=rc, minutes=(time.time() - t0) / 60)
        rjson = os.path.join(run_dir, "Results.json")
        if rc == 0 and os.path.exists(rjson):
            with open(rjson) as fh:
                rec["results"] = json.load(fh)["results"]
            rec["series"] = read_force_series(run_dir, run_json["settings"].get("crossover", 60))
            print(f"    Cd={rec['results']['cd']:.4f} Cl={rec['results']['cl']:.4f} Cy={rec['results'].get('cy', float('nan')):+.4f}  ({rec['minutes']:.1f} min)")
        else:
            print(f"    FAILED rc={rc}; see {lpath}")
        results = [r for r in results if r["id"] != rid] + [rec]
        with open(results_path, "w") as fh:
            json.dump(results, fh, indent=1)
        remaining = len(todo) - (n + 1)
        if remaining:
            per = (time.time() - t_all) / (n + 1)
            print(f"    ~{per * remaining / 60:.0f} min remaining", flush=True)
    summarize(results, summary_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
