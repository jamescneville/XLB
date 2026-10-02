# surfaceRemesh

Builds the mesh used for the **USD/VTK surface-field export**. Nothing else uses it: the solver, voxelization and force calculations still use the raw CAD.

- Code: [`xlb/utils/surface_remesh.py`](../xlb/utils/surface_remesh.py)
- Called from: [`examples/windtunnel_json.py`](../examples/windtunnel_json.py) via `remesh_surface_isolated`, which runs the work in a **child process**.
- Failure behaviour: if anything raises, the run continues with the raw mesh and prints a warning.
- When the wrap succeeds, the mesh is flagged `outward_normals`, which enables the `outward` / `outward_check` side selectors.

---

## Pipeline order

| # | Stage | What it does |
|---|---|---|
| 1 | **Weld and clean input** | Merges coincident vertices, drops degenerate faces. |
| 2 | **Wrap** *(optional)* | Closed shell at a fixed offset outside the CAD: exact distance field on a sparse uniform grid, then marching cubes at the offset. Enclosed air pockets and shells inside them are dropped. |
| 3 | **Resample** | Splits into shells, then uniform vertex clustering (pyacvd) to `targetEdge`, then projects back onto the source and matches shell winding. |
| 4 | **Global relax** *(wrap only)* | Laplacian steps, re-projected onto the wrap each time. Evens out triangle shape. Replaced by stage 4b when `curvatureAdaptive` is on. |
| 4b | **Curvature-adaptive refinement** *(optional, wrap only)* | Splits edges where the wrap curves tightly, then Delaunay flips and size-aware smoothing back onto the wrap. See below. |
| 5 | **Snap** *(optional, wrap only)* | Pulls vertices toward the original CAD, leaving `snapOffset` of stand-off so the two skins of a thin sheet stay apart. |

---

## Options (`settings.surfaceRemesh`)

| Key | Default | Effect |
|---|---|---|
| `enabled` | off | Master switch. Also requires `settings.surfaceField` to be set. |
| `targetEdge` (m) | `voxelSize` without wrap; with wrap and no value, the raw wrapped mesh is used | Final edge length. |
| `maxFaces` | none | Coarsens the edge to stay under this face cap. |
| `curvatureAdaptive.enabled` | off | Finer triangles where the wrapped surface curves tightly (needs the wrap). |
| `curvatureAdaptive.maxEdge` (m) | `targetEdge` | Edge length on flat areas. This is also the edge of the uniform base mesh, so keep it near the value that already works (about 8 mm with a 4 mm wrap; a 12 mm base produced hundreds of non-manifold edges). |
| `curvatureAdaptive.minEdge` (m) | `maxEdge / 2` | Smallest edge at tight curvature. Below about the wrap resolution it cannot add real detail. |
| `curvatureAdaptive.tolerance` (m) | `minEdge / 4` | Chord error: an edge on a radius R is kept under `sqrt(8 * tolerance * R)`. Larger means fewer refined areas. |
| `curvatureAdaptive.gradation` | `0.4` | Largest size change per unit distance. Smaller means smoother size transitions and more triangles. |
| `useGpu` | `true` | GPU (Warp) distance queries; falls back to CPU automatically. `false` forces CPU. |
| `wrap.enabled` | off | Turns on the boundary wrap. |
| `wrap.resolution` (m) | `voxelSize / 2` | Wrap grid cell size. |
| `wrap.offset` (m) | 0.75 × resolution | How far outside the CAD the wrap sits. Needs to be at least about 0.5 × resolution. |
| `wrap.gapClosure` (m) | `0` (off) | Seals leaks up to about twice this when deciding what is interior. |
| `wrap.snapOffset` (m) | off (missing, `null`, `false` or negative) | Stand-off from the CAD after the snap. |
| `wrap.relaxIterations` | `3` | Global relax steps (0 disables). |

Related setting, outside `surfaceRemesh`:

| Key | Default | Effect |
|---|---|---|
| `settings.surfaceSideSelector` | `velocity` | `velocity`, `outward` (probe the +normal side only) or `outward_check` (velocity selection plus a report of how often it disagrees with the normals). The outward modes need the wrap to have run, otherwise they fall back to `velocity` with a warning. |

Other switches: environment variable `XLB_SURFACE_GPU=0` and the CLI flag `--no-gpu` force the CPU path.

### Curvature-adaptive refinement

The uniform resample is made at `maxEdge`; edges are then split where the wrap's curvature asks for something smaller, down to `minEdge`. The sizing field comes from the wrap alone (the CAD plays no part).

1. Curvature per wrap vertex: the largest normal turn per unit length towards any neighbour, with the length floored at half the mean edge (the wrap contains near-zero-length edges, which otherwise read as enormous curvature and spray dense blobs over flat panels). The curved zone is widened by two rings, converted to a size, clamped to `[minEdge, maxEdge]` and graded.
2. Edges longer than 4/3 of the local size are split (1->2, 1->3 or 1->4, no hanging nodes); new vertices go back onto the wrap.
3. Delaunay edge flips, restricted to the neighbourhood of what changed, and size-aware spring smoothing re-projected onto the wrap.

`maxFaces` doubles as a budget: if the sizing field would give more triangles than that, `tolerance` is raised until it fits (the log says so).

GPU: curvature, sizing, gradation, smoothing, projection and size lookup run in Warp. Splits and flips are vectorised numpy because they are irregular. A CPU fallback gives the same meshes to within a few percent.

GTU, wrap 4 mm / offset 3 mm, snap 1 mm (uniform 8 mm = 2.45M triangles, 55 s):

| Setting | Triangles | Adaptive stage | Centre-to-CAD p99 | Min angle p1 |
|---|---|---|---|---|
| uniform 8 mm | 2.45M | - | 3.03 mm | 32 deg |
| 3-8 mm, tol 1.5 mm | 3.30M (+35%) | 21 s | 2.59 mm | 26 deg |
| 3-8 mm, tol 0.75 mm | 4.03M (+65%) | 36 s | 2.47 mm | 24 deg |

It concentrates triangles along creases and curved edges and leaves flat panels alone, but the measured error reduction is modest for the extra triangles. The snap to the CAD already puts most of the mesh within 1 mm. Topology gets slightly worse (boundary/non-manifold edges 6/6 -> 11/12), because splitting a non-manifold edge from the resample splits it for both sheets.

### Couplings that are easy to miss
- `relaxIterations` only applies when the wrap is on. On the plain CAD path it would wear away the edges the projection preserves.
- `outward` falls back to `velocity` unless the wrap actually ran.

---

## Paths

| Path | Setup | Result |
|---|---|---|
| **Plain resample** | `wrap.enabled` false | Edges kept as projected, but irregular triangles (originally 22% under 10°, longest edge 62 mm). |
| **Wrap + resample** | `wrap.enabled` true, `targetEdge` set | Uniform, oriented, mostly closed mesh. Rounded edges, stands off the CAD. |
| **Wrap only** | `wrap.enabled` true, no `targetEdge` | Raw wrapped mesh. About 9.7M triangles on GTU at 4 mm, so mostly for inspection. |
| **+ snap** | `snapOffset` set | Brings the mesh close to the CAD. |

---

## Performance (GTU, 4.3M triangles, wrap 4 mm, edge 8 mm, snap 1 mm°)

| Setup | Total |
|---|---|
| Original CPU | 304 s (with feature recovery, since removed) |
| GPU (current) | 70 s |
| CPU fallback (current code) | 185 s |

Current GPU stage breakdown: weld 9 s, wrap 25 s (of which mesh extraction about 10 s), shell split 4 s, clustering 10 s, project and assemble about 1 s, relax 6 s, snap under 1 s.

Quality of the current GPU pipeline versus the original slow one (equivalent): worst-5% triangle quality 0.83 against 0.82, minimum-angle p1 28.9° against 29.1°. 

---

## Known downsides

### Wrap
- Offset must be at least about 0.5 × grid size; a tiny offset on a coarse grid breaks.
- Details closer than `2 × offset` merge (the grille slats came out fatter).
- Convex edges round by roughly the offset.
- Resolution is uniform, with no adaptivity; memory grows with the cube of extent over resolution.
- Manifoldness is **not guaranteed**: the final GTU mesh had 6 boundary and 88 non-manifold edges out of 3.7M, and 0.01% of faces flipped. Normals point away from the CAD on about 99.99% of faces.
- **`gapClosure` is effectively unusable as it stands:** slow (255–1,100 s), it trimmed only 6–22% of the area, and it bridged surfaces up to 30 mm off the CAD. It also can't tell real engine-bay flow passages from leaks.

### Resample, relax, snap
- pyacvd has no awareness of creases or sharp edges.
- Relaxation softens edges slightly: the sharp-edge p90 loosens by about 0.8 mm.
- Snapping cannot recover detail the wrap already merged, so the sharp-edge p90 stays around 5 mm. Sharp edges are no longer re-created separately: feature recovery was tried and removed. The curvature-adaptive refinement only puts more triangles where the wrap curves.

### Operational
- The GPU path was tested with PyPI Warp 1.17 on Windows only. The server uses the Warp fork bundled with Neon; a start-up self-test falls back to CPU and logs why if anything fails.
- The remesh runs before the simulation, so it delays startup by about 70 s (GPU) with no caching between runs.
- Validation rests on my own metrics and a few renders; there are no unit tests in the repo.

---

## Where it could improve

### Wrap and speed
- A signed field from the flood fill would allow a small offset on a coarse grid.
- Do gap closure on a coarser, GPU-accelerated grid.
- Use Warp's marching cubes instead of the block-by-block VTK loop, and skip welding when the input is already welded.
- **Run the remesh in the background while the simulation runs.** The mesh is only needed at export time, so the startup delay could disappear.
- Cache results by an input-and-options hash, stored outside the run folder that gets wiped.

### Robustness
- Add synthetic unit tests: sphere, thin sheet, nested boxes, leaky box.
- Validate on OBJ input and the `car` scope with wheels.
