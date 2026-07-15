# Sail: Reusable 3D Backbone Glyph

This repository provides a reusable 3D Backbone Glyph component for **[Sail](https://github.com/ScheWann/Sail)**, designed to read many scalar signals *in place* on a 3D backbone in a single view.

![3D Backbone Glyph](Image/Glyph_General.png)

## What Data Is Demonstrated

This repository currently includes two integrated non-biology-related demo datasets, both rendered unchanged by the same glyph:

1. TURB-Lagr (Lagrangian turbulence)  
	Source: https://smart-turb.roma2.infn.it
- Built from a direct numerical simulation of homogeneous isotropic turbulence (`1024³`, `Re_λ = 2310`); the notebook lazily fetches a ~140 MB subset of `full_traj_tracers.h5`.
- Backbone: a tracer particle's trajectory. Channels: `u_mag`, `a_mag`, `omega_mag`, `eps`, `vortex_fraction`, `helicity`.
- Focus: show that a bead's *spatial* neighbours — not its trajectory-index neighbours — share glyph shape, so the field forms 3D domains.

2. Intracranial Aneurysm (CFD hemodynamics)  
	Source: https://zenodo.org/records/19455127
- Built from CFD hemodynamic fields on real Aneurisk geometry `C0099.vtp` plus its VMTK `centerlines.vtp` (DOI `10.5281/zenodo.19455127`, CC BY 4.0).
- Backbone: the vessel centerline. Channels: `TAWSS`, `OSI`, `TAWSSG`, `Lambda2`, curvature, and related wall fields.
- Focus: read multiple hemodynamic metrics along the artery, with the aneurysm sac highlighted, on the geometry rather than an arc-length plot.

Glyph encoding overview:

- Backbone: the 3D curve itself (trajectory / centerline), rendered as a Catmull-Rom tube with labeled start `s` and end `e` markers.
- Fins: at each bead, one ribbon fin per channel. Fin length encodes the channel value (optionally remapped as `v ^ gamma`); the fin polygon is oriented perpendicular to the backbone tangent.
- Interaction: per-channel toggles, gamma remap, tube / vessel-overlay toggles, and box-select to hide backbones.

## Requirements For Using This Glyph

### 1. Runtime Requirements

- Node.js 18+ (LTS recommended)
- npm 9+
- Dependencies: React 19, Three.js, and [`@react-three/fiber`](https://github.com/pmndrs/react-three-fiber) / `@react-three/drei`

Run locally:

```bash
cd Frontend
npm install
npm run dev
```

### 2. Data Input Check

For data input, just check these files:

- Frontend/public/turb_glyph.json
- Frontend/public/aneurysm_glyph.json
- Frontend/src/GlyphView3D.tsx (see the `GlyphDataset` interface and `AVAILABLE_DATASETS`)

If your data follows the same structure as those JSON files, it can be rendered directly by the glyph component:

```jsonc
{
  "meta":     { "title": "...", "description": "...", "unit": "..." },
  "channels": ["u_mag", "a_mag", "..."],
  "objects":  [
    { "objectId": 0, "label": "...",
      "points": [ { "x": 0, "y": 0, "z": 0, "values": [0.1, 0.4, "..."] } ] }
  ]
}
```

Each notebook in [`Scripts/`](Scripts/) ends with an export cell that writes this JSON into `Frontend/public/`. To add a dataset, drop its JSON there and register it in `AVAILABLE_DATASETS`.

### 3. Minimal Usage Example

```tsx
import GlyphView3D from "./GlyphView3D";

// GlyphView3D fetches the datasets registered in AVAILABLE_DATASETS
// (e.g. /turb_glyph.json, /aneurysm_glyph.json) from Frontend/public.
export default function Example() {
	return (
		<div style={{ width: "100vw", height: "100vh" }}>
			<GlyphView3D />
		</div>
	);
}
```

## Typical Use Cases

- Reading multivariate signals in place on a 3D backbone, with no dimensionality reduction
- Comparing spatial neighbours that belong to different curves in a shared 3D space
- Analysis scenarios that need both the geometry and its per-node metrics in one glyph
