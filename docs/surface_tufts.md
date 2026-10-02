# surfaceTufts

Flow tufts on the export surface: short filaments, tubes or ribbons, rooted on the surface and bent along the time-averaged near-wall flow. Written as plain triangle meshes to `<outputName>_tufts.usda` (Alias and VRED both imported tubes and ribbons; `BasisCurves` did not import, so curves are not offered).

- Code: [`xlb/utils/tufts.py`](../xlb/utils/tufts.py) (pure numpy, no Neon), `MultiresIO.to_surface_tufts_time_average` in [`mesher.py`](../xlb/utils/mesher.py), wired in by `export_surface_tufts` in [`windtunnel_json.py`](../examples/windtunnel_json.py).
- Needs `surfaceRemesh` with `wrap.enabled` (outward normals); otherwise it is skipped with a warning.
- A failure never stops the run.

## Options (`settings.surfaceTufts`)

| Key | Default | Effect |
|---|---|---|
| `enabled` | off | Master switch. Also makes the remesh run when `surfaceField` is empty. |
| `spacing` (m) | `0.03` | Minimum distance between roots (Poisson-disk, area-weighted). |
| `length` (m) | `0.025` | Tuft length. A warning is printed below 3 voxels, where the near-wall velocity is not resolved. |
| `width` (m) | `0.0015` | Tube diameter / ribbon width; tapers to 60% at the tip. |
| `shape` | `tube` | `tube` (3-sided) or `ribbon`. |
| `color` | `[1.0, 0.45, 0.0]` | `[r, g, b]` constant diffuse colour (plus 15% emission), or `"velocity"` to colour by the sampled near-wall speed through a colormap texture (same mechanism as the surface field). |
| `colorMap` | `surfaceFieldColorMap`, else `turbo` | Colormap for `"velocity"`. |
| `velocityMin` / `velocityMax` | `0` / `InletBC.x * slices.velocityFactor` | Colour range for `"velocity"` (same default as the velocity slices). |
| `tape` | `true` | Set `false` to leave the tape off (same as `tapeSize` 0). |
| `tapeSize` (m) | `0.010` | Square tape patch at each root (0 = none), lying on the surface with its normal along the surface normal. Aligned with the tuft's first segment, root a quarter of the way in from the downstream edge. Written as a second prim, `tape`, in the same `.usda`. |
| `tapeOffset` (m) | `0.0005` | Height of the tape above the export surface, so it never coincides with it (z-fighting). |
| `tapeColor` | `[0.2, 0.55, 1.0]` | Constant tape colour. |
| `rootInset` (m) | `0.0008` | Sinks the root into the surface so it looks attached; the snapped wrap sits `snapOffset` off the CAD. |
| `maxTufts` | `150000` | Spacing is raised automatically to stay under this. |

## How it works

1. Roots: dart-throwing on the remeshed surface. Two roots exclude each other only if their normals agree, so both faces of a thin sheet can carry tufts.
2. Six segments are marched from each root. For each one, velocity is sampled by inverse-distance weighting over fluid cells only, at the same probe height as the surface field (1.5 voxels plus `surfaceProbeExtra`), at least that far above the root plane.
3. Direction = tangential flow plus a wall-normal lift from `u.n` (clamped 4-15 deg minimum to 60 deg maximum), blended 40% with the previous segment for stiffness. Reversed near-wall flow therefore points the tuft upstream.
4. Each joint is pushed out of the body using the nearest surface vertex and its outward normal.
5. Roots with no fluid cell nearby, or no flow, are dropped.

## Limits
- The velocity is time-averaged, so there is no flutter; separated regions show as steady, possibly upstream-pointing tufts.
- Collision uses the nearest export vertex (about one edge length), so tufts can still graze tight concave corners.
- Tested on a synthetic sphere only. Not yet run on a solver result.

## Previewing without a solve

`examples/windtunnel_geom_json.py -i project.json` runs the remesh and writes the surface field and the tufts from synthetic data, using the same `settings.surfaceTufts` options. The dummy near-wall velocity is attached flow along the surface on front and side faces (about 1.3 x `InletBC.x`, slowing to zero at the stagnation point), and reversed, lifting and swirling flow on rear-facing surfaces (normal.x > 0). It exercises the geometry, collision, direction and colouring, not any real flow.
