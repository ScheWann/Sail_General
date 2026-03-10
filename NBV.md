# NBV (Next Best View) – Function Reference

## Geometry / data helpers

| Function | Input | Output | Method |
|----------|--------|--------|--------|
| **getCenterAndRadius** | `beads` | `{ center, dataRadius }` | Center = mean of bead positions; dataRadius = max distance from center to any bead. |
| **generateViewpoints** | `center`, `dataRadius` | `Viewpoint[]` | Sample camera positions on a sphere (5° step in azimuth/elevation), radius = 2 × dataRadius. |
| **getTrackQuads** | `beads`, `trackIdx`, `numTracks` | `Vec3[][]` (quads) | For each consecutive node pair, build a quad: two vertices on the path, two offset perpendicular by the track's normalized value. |
| **projectedArea** | `quad`, `viewPos`, `center` | `number` | Orthographic projection of the quad onto the view plane; 2D area via shoelace formula. |
| **angleDiffDeg** | `posA`, `posB`, `center` | `number` (degrees) | Angle between the two view directions (from center): dot product of normalized vectors → arccos. |

## Visibility & probabilities (entropy pipeline)

| Function | Input | Output | Method |
|----------|--------|--------|--------|
| **computeVisibility** | `views`, `tracks`, `beads`, `center` | `visibility[viewId][trackId]` | For each view and each active track: sum of projected areas of that track's quads (no occlusion). |
| **computePCond** | `views`, `tracks`, `visibility` | `pCond[viewId][trackId]` | p(object \| view): normalize visibility per view so that per-view sum over objects = 1. |
| **computePMarginal** | `views`, `tracks`, `pCond` | `pMarginal[trackId]` | p(object): average of p(object \| view) over all views. |
| **computePTarget** | `tracks`, `pMarginal` | `pTarget[trackId]` | Importance-weighted target: p'(o) ∝ p(o) × importance(o), then renormalize (active = 100, inactive = 1). |
| **computeVMI** | `views`, `tracks`, `pCond`, `pTarget` | `scores[viewId]` | Viewpoint Mutual Information: for each view, Σ p(o\|v) log(p(o\|v) / p'(o)); lower = better. |
| **selectTop10** | `views`, `scores`, `center` | `Viewpoint[]` (length ≤ 10) | Sort views by score (ascending), then greedily take views that are at least 45° apart (by angleDiffDeg). |

## UI / pipeline

| Function | Input | Output | Method |
|----------|--------|--------|--------|
| **handleTrackToggle** | track index `i` | — | Toggle index `i` in the active set and call `onNbvActiveTracksChange` with the new list (only affects NBV window). |
| **runPipeline** | (uses beads, tracks, nbvActiveTrackIndices) | — | 1) Get center & radius. 2) Generate viewpoints. 3) Compute visibility → pCond → pMarginal → pTarget → VMI. 4) selectTop10. 5) Set first view and call `onApplyView`. |
| **goPrev** / **goNext** | — | — | Decrease/increase current index in the top-10 list and call `onApplyView` with that view's position and center. |

**Overall flow:** Build track-surface quads → sample views on a sphere → for each view compute visibility (projected area per track) → turn visibility into p(o\|v), then p(o) and p'(o) → score views with VMI → keep 10 views with ≥45° separation → show them in the NBV window and apply the chosen view there.
