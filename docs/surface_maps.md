# surfaceMaps

Three extra surface visualisations from the time-averaged solution, each written next to the other results as `<outputName>_<name>.usda` on the remeshed export surface (the same mesh as the Cp surface field). Like the tufts they need `surfaceRemesh` with `wrap.enabled` (outward normals) and are skipped with a warning otherwise; a failure never stops the run.

- Code: [`xlb/utils/surface_maps.py`](../xlb/utils/surface_maps.py) (pure numpy), `MultiresIO.to_surface_maps_time_average` in [`mesher.py`](../xlb/utils/mesher.py), `export_surface_maps` in [`windtunnel_json.py`](../examples/windtunnel_json.py). `windtunnel_geom_json.py` writes the same files from dummy fields.
- Needs the averaged `Cp` field for the drag/lift map (the same field the Cp surface field uses) and `velocity_0..2` for the other two.
- Flow is +x, up is +z. Near-wall velocity is read at the same probe distance as the surface field (1.5 voxels plus `surfaceProbeExtra`), from fluid cells only.

```json
"surfaceMaps": {
    "dragContribution": { "enabled": true, "component": "drag", "range": 0.5, "colorMap": "RdBu_r" },
    "separation":       { "enabled": true, "colorMap": "RdYlGn" },
    "wallStreamlines":  { "enabled": true, "spacing": 0.06, "length": 0.3, "width": 0.002, "color": "velocity", "colorMap": "turbo" }
}
```

A block is only active with `"enabled": true`.

## dragContribution -> `<name>_drag_contribution.usda` (or `_lift_contribution`)

Per-vertex pressure force density, `-Cp * n_x` (drag) or `-Cp * n_z` (lift), in units of q-infinity per unit area. The force on the body is `-p n dA` with n pointing out of the body, so:

- **red** = pushes the car back (drag) / up (lift): upstream-facing high pressure and downstream-facing suction both read red;
- **blue** = pulls it forward / presses it down.

It is a density, so it does not depend on mesh size; the whole map integrates to the pressure drag of the exported surface. The log prints that integral and, divided by the reference area, a pressure-only coefficient for the exported surface. It excludes friction and anything not in the export mesh (`surfaceFieldScope: "body"` leaves the wheels out), so it will not match the solver's Cd. Check on a potential-flow sphere: the integral is 0, as d'Alembert says.

| Key | Default | Effect |
|---|---|---|
| `component` | `drag` | `drag` or `lift`. |
| `range` | `0.5` | Symmetric colour range (+/-). Full scale is about 1 at a stagnation point; most of a body is well under 0.2. |
| `colorMap` | `RdBu_r` | Any matplotlib colormap; keep it diverging. |

## separation -> `<name>_separation.usda`

Index in [-1, 1] per vertex: the cosine between the near-wall velocity and the velocity a little further out, faded to 0 where the near-wall flow is almost stagnant (its direction means nothing there).

- **green (+1)** attached: the flow at the wall follows the flow just outside it;
- **yellow (0)** stagnant or undetermined;
- **red (-1)** reversed: the flow at the wall runs against the outer flow, i.e. separated.

It does not use the freestream direction, so it is defined on front and rear faces too.

| Key | Default | Effect |
|---|---|---|
| `outerOffset` (m) | 3 voxels | How far beyond the near-wall probe the reference velocity is read. |
| `slowFraction` | `0.05` | Near-wall speed below this fraction of the outer speed fades the index to 0. |
| `colorMap` | `RdYlGn` | Low = red = reversed. |

The log prints the share of vertices reversed (index < -0.2) and neutral (|index| <= 0.2).

## wallStreamlines -> `<name>_wall_streamlines.usda`

Tubes that follow the near-wall flow, hugging the surface (1 mm off it by default), traced both ways from Poisson-spaced seeds, like flow-vis paint. They bunch up at separation lines and fan out where the flow spreads. A line stops where the near-wall flow stalls, reverses sharply (convergence or stagnation lines), or would jump onto another surface.

| Key | Default | Effect |
|---|---|---|
| `spacing` (m) | `0.06` | Distance between seeds. |
| `length` (m) | `0.3` | Total length (half each way from the seed). |
| `step` (m) | 1.5 voxels | Integration step. |
| `width` (m) | `0.002` | Tube diameter (3-sided). |
| `lift` (m) | `0.001` | Height off the surface. |
| `maxLines` | `60000` | Cap on seeds. |
| `color` | `"velocity"` | `"velocity"` (near-wall speed through the colormap) or `[r, g, b]`. |
| `colorMap`, `velocityMin`, `velocityMax` | as for the tufts | Colour range for `"velocity"`. |

Geometry is triangle meshes (not curves), the form that imports into Alias.

## Limits
- Maps inherit the 8 mm-voxel, ~15 mm probe resolution: small separation bubbles and thin features are not resolved.
- Time-averaged only: no unsteadiness.
- Tested on synthetic flow only (potential-flow sphere, a rounded box); not yet run on a solver result.
