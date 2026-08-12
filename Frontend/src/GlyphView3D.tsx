import { useEffect, useState, useMemo, useCallback, useRef } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Html, OrbitControls } from "@react-three/drei";
import * as THREE from "three";

// drei forwards the underlying three-stdlib controls instance.
type OrbitControlsHandle = React.ComponentRef<typeof OrbitControls>;

// ── Data types ──────────────────────────────────────────────────────────────

interface GlyphPoint {
  x: number;
  y: number;
  z: number;
  values: number[];
  // The same channels in physical units, where the export ships them. `values` is a
  // within-export percentile and deliberately flattens magnitude; `raw` is what the
  // log encoding is computed from. See `LogBounds`.
  raw?: number[];
  // Per-channel sign for channels whose `values` entry is a magnitude (see
  // `meta.signedChannels`). +1/-1, absent or 1 for ordinary channels.
  sgn?: number[];
  attributes?: Record<string, unknown>;
}

// Per-channel bounds for mapping log10(raw) onto [0,1], as published in
// `meta.logNormalization`. The bounds are percentiles of the raw distribution rather
// than its min/max, so a single extreme node cannot compress every other fin.
interface LogBounds {
  log10Lo: number;
  log10Hi: number;
  clipPercentiles?: number[];
  formula?: string;
}

interface GlyphObject {
  objectId: number;
  label?: string;
  points: GlyphPoint[];
}

interface GlyphDataset {
  meta?: Record<string, unknown> & {
    caseType?: string;
    title?: string;
    description?: string;
    interpretation?: string;
    limitation?: string;
    unit?: string;
    // Channels encoded as |value| in `values`, with the sign in `point.sgn`.
    signedChannels?: string[];
    // Bounds for the alternative log encoding, keyed by channel name. Present only
    // where the export also ships `point.raw`.
    logNormalization?: Record<string, LogBounds>;
    channelUnits?: Record<string, string>;
    // What each fin means, written by the notebook that produced the file. This is
    // the authority for legend text: the same channel name means different things in
    // different cases ("curvature" is a particle path's bending in one export and a
    // fiber bundle's in another), so a name-keyed table in the viewer cannot be right
    // for both.
    channelInfo?: Record<
      string,
      { label?: string; raw?: string; description?: string }
    >;
    // Channels the producing notebook measured but deliberately did not give a fin,
    // with the reason. Keyed by channel name; the value is the explanation.
    diagnosticChannels?: Record<string, string>;
  };
  channels: string[];
  objects: GlyphObject[];
}

interface SurfaceMesh {
  vertices: number[][];
  faces: number[][];
}

// White-matter case: one translucent envelope per fiber bundle plus a brain-mask
// isosurface for anatomical context. Bundle envelopes are keyed to the glyph
// objects by `objectId`, so hiding a bundle hides its tube too.
interface TractGeometry {
  meta?: Record<string, unknown> & {
    caseType?: string;
    title?: string;
    description?: string;
  };
  bundles: {
    objectId: number;
    key?: string;
    label?: string;
    nStreamlines?: number;
    mesh: SurfaceMesh;
  }[];
  context?: {
    description?: string;
    // Outer brain-mask hull: an envelope, no internal structure.
    mesh?: SurfaceMesh;
    // White-matter boundary: the folded surface the bundles actually terminate on.
    whiteMatter?: {
      description?: string;
      mesh?: SurfaceMesh;
    };
  };
}

interface CoordinateTransform {
  center: [number, number, number];
  scale: number;
}

// Internal per-point representation consumed by the 3D pipeline.
interface NodeData {
  position: [number, number, number];
  values: number[];
  // Parallel to `values`; 1 for channels that carry no sign.
  signs: number[];
}

interface SelectionRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

// ── Configuration ───────────────────────────────────────────────────────────

interface DatasetConfig {
  name: string;
  path: string;
  geometryPath?: string;
  // Label of the checkbox that toggles the surrounding anatomical context.
  contextLabel?: string;
  defaultSampleCount?: number;
  // Point order is a time series (tracer paths), so playback along it is meaningful.
  supportsAnimation?: boolean;
  // Rubber-band selection of individual objects. Only worth offering where the objects
  // are an interchangeable population; the other cases have a handful of named paths.
  supportsSelection?: boolean;
  defaultGamma?: number;
  // Initial camera placement and orbit axis; see `CameraSetup`.
  camera?: CameraSetup;
}

// OrbitControls turns horizontal dragging into rotation about the camera's up vector.
// With the default +Y up, dragging left/right on RAS anatomical data (superior = +Z)
// rolls the volume about its front-back axis instead of spinning it about the vertical
// axis, which is close to unusable on a brain. Cases with a real up axis declare it.
interface CameraSetup {
  position: [number, number, number];
  up: [number, number, number];
}

const DEFAULT_CAMERA: CameraSetup = { position: [0, 0, 500], up: [0, 1, 0] };
// Posterior-superior three-quarter view, superior axis up.
const RAS_CAMERA: CameraSetup = { position: [0, -470, 170], up: [0, 0, 1] };

// Playback state is shared through a ref so advancing time never re-renders the
// React tree — the reveal is applied straight to the three.js objects.
interface AnimationConfig {
  progressRef: React.MutableRefObject<number>;
  active: boolean;
}

// World-space extent the sampled point clouds are scaled to, so camera framing
// and glyph sizes work while preserving their relative source positions.
const TARGET_EXTENT = 200;
const RANDOM_SEED = 3601;
const DEFAULT_SAMPLE_COUNT = 4;
// How a channel value becomes a fin length. The two encodings answer different questions
// and the export ships the inputs for both, so the viewer switches between them:
//
//   percentile — `point.values`, the rank of the measurement among all exported nodes.
//                Comparable across channels (0.9 is "top 10 %" on every fin) but it
//                compresses the tail: the 99th percentile and the maximum both sit near
//                1.0 even when they differ by a factor of five.
//   log        — log10(`point.raw`) mapped between `meta.logNormalization` bounds. Keeps
//                ratios visible, but a value only means something within its own channel.
//
// A linear map of `raw` is not offered: these channels are heavy-tailed enough that it
// would leave most fins too short to see.
type Encoding = "percentile" | "log";

// Percentile values are uniform on [0,1], so v^gamma is what pushes everything below the
// top decile down and leaves the extremes long.
const DEFAULT_GAMMA = 20;
// The log encoding already carries magnitude in the value itself, so the default leaves it
// alone: fin length is exactly `meta.logNormalization.formula`. Gamma stays adjustable.
const DEFAULT_LOG_GAMMA = 1;

const AVAILABLE_DATASETS: DatasetConfig[] = [
  {
    name: "Turbulence tracers",
    path: "/turb_glyph_full.json",
    defaultSampleCount: DEFAULT_SAMPLE_COUNT,
    supportsAnimation: true,
    supportsSelection: true,
    defaultGamma: DEFAULT_GAMMA,
  },
  {
    name: "White-matter bundle profiles",
    path: "/wm_tract_glyph.json",
    geometryPath: "/wm_tract_geometry.json",
    contextLabel: "Brain",
    defaultSampleCount: 8,
    defaultGamma: DEFAULT_GAMMA,
    camera: RAS_CAMERA,
  },
];

const CHANNEL_COLORS = [
  "#ff6b6b", // Red
  "#45b7d1", // Blue
  "#4daf4a", // Green
  "#6c5ce7", // Purple
  "#f3722c", // Orange
  "#f9ca24", // Yellow
  "#ff9ff3", // Pink
  "#a29bfe", // Lavender
  "#00d2d3", // Teal
  "#54a0ff", // Light Blue
];

function getChannelColor(index: number): string {
  return CHANNEL_COLORS[index % CHANNEL_COLORS.length];
}

// Last resort only, for exports that carry no `meta.channelInfo`. Deliberately free
// of case-specific wording: a name-keyed table is shared by every dataset, so it can
// only be trusted to expand an abbreviation, never to say what a channel measures.
const FALLBACK_CHANNEL_NAMES: Record<string, string> = {
  a_mag: "acceleration",
  fiber_density: "fiber density",
  support_fraction: "support fraction",
};

function channelLabel(name: string, meta?: GlyphDataset["meta"]): string {
  const published = meta?.channelInfo?.[name]?.label;
  // The published label names the encoding as well ("FA percentile"), but the
  // encoding is already shown once by the fin-length toggle, so the per-channel
  // label drops it rather than repeating it on every fin -- and rather than
  // asserting "percentile" while the log encoding is active.
  if (published) return published.replace(/\s*percentile\s*$/i, "");
  return FALLBACK_CHANNEL_NAMES[name] ?? name;
}

function channelTooltip(
  name: string,
  meta: GlyphDataset["meta"] | undefined,
  encoding: Encoding,
): string {
  const info = meta?.channelInfo?.[name];
  const unit = info?.raw ?? meta?.channelUnits?.[name];
  const encodes =
    encoding === "log"
      ? "fin length = log of the measurement, scaled between the published bounds"
      : "fin length = percentile of the measurement among all exported nodes";
  return [info?.description, unit && `measured in ${unit}`, encodes]
    .filter(Boolean)
    .join(" — ");
}

// Channels the export itself marks as measured-but-not-drawn. `meta.diagnosticChannels`
// is the producing notebook's own record of what it chose not to give a fin, and why;
// the fallback covers older exports that predate that field.
const LEGACY_HIDDEN_CHANNELS = new Set(["vortex_fraction", "helicity"]);

function hiddenChannelSet(data: GlyphDataset | null): Set<string> {
  const declared = Object.keys(data?.meta?.diagnosticChannels ?? {});
  return new Set([...declared, ...LEGACY_HIDDEN_CHANNELS]);
}

// Signed channels (`meta.signedChannels`) put |value| in the fin length, so the
// direction has to come from somewhere else. Negative beads keep the channel's
// hue — the fin must stay identifiable as that channel — and drop saturation and
// lightness. Hue is already spent on channel identity and lightness alone on the
// per-node highlight patch (l * 1.5 below), so a negative fin still reads as
// "darker version of this channel" under its own highlight.
const NEGATIVE_SATURATION = 0.7;
const NEGATIVE_LIGHTNESS = 0.42;

function scaleHSL(base: THREE.Color, sFactor: number, lFactor: number) {
  const hsl = { h: 0, s: 0, l: 0 };
  base.getHSL(hsl);
  return new THREE.Color().setHSL(hsl.h, hsl.s * sFactor, hsl.l * lFactor);
}

function getNegativeChannelColor(index: number): string {
  const c = scaleHSL(
    new THREE.Color(getChannelColor(index)),
    NEGATIVE_SATURATION,
    NEGATIVE_LIGHTNESS,
  );
  return `#${c.getHexString()}`;
}

const BACKBONE_COLOR = "#FFFFFF";
const BEAD_COLOR = "#FFFFFF";
const BACKBONE_RADIUS = 0.25;
const ENDPOINT_MARKER_COLOR = "#999";
const ENDPOINT_MARKER_SIZE = 5;

// Seconds one full pass over a trajectory takes at 1x speed.
const ANIMATION_DURATION = 12;
const HEAD_COLOR = "#ffd166";

function clamp01(value: number) {
  return value < 0 ? 0 : value > 1 ? 1 : value;
}

function createSeededRandom(seed: number) {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

function sampleObjects(objects: GlyphObject[], count: number, seed: number) {
  const rand = createSeededRandom(seed);
  const shuffled = [...objects];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled.slice(0, count);
}

function getCoordinateTransform(
  objects: GlyphObject[],
): CoordinateTransform | null {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];

  for (const object of objects) {
    for (const p of object.points) {
      const c = [p.x, p.y, p.z];
      for (let k = 0; k < 3; k++) {
        if (Number.isFinite(c[k])) {
          if (c[k] < min[k]) min[k] = c[k];
          if (c[k] > max[k]) max[k] = c[k];
        }
      }
    }
  }

  if (!min.every(Number.isFinite) || !max.every(Number.isFinite)) {
    return null;
  }

  const center = [
    (min[0] + max[0]) / 2,
    (min[1] + max[1]) / 2,
    (min[2] + max[2]) / 2,
  ];
  const extent =
    Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2]) || 1;
  return {
    center: center as [number, number, number],
    scale: TARGET_EXTENT / extent,
  };
}

function transformPoint(
  point: { x: number; y: number; z: number },
  transform: CoordinateTransform,
): [number, number, number] {
  const { center, scale } = transform;
  return [
    ((Number.isFinite(point.x) ? point.x : center[0]) - center[0]) * scale,
    ((Number.isFinite(point.y) ? point.y : center[1]) - center[1]) * scale,
    ((Number.isFinite(point.z) ? point.z : center[2]) - center[2]) * scale,
  ];
}

// `logBounds` is parallel to `channels`; a null entry means that channel has no published
// bounds and falls back to its percentile, so a partially annotated dataset degrades
// per channel instead of failing.
function encodeValue(
  point: GlyphPoint,
  channelIndex: number,
  encoding: Encoding,
  logBounds: (LogBounds | null)[] | null,
): number {
  if (encoding === "log") {
    const bounds = logBounds?.[channelIndex];
    const raw = point.raw?.[channelIndex];
    if (bounds && Number.isFinite(raw) && (raw as number) > 0) {
      const span = bounds.log10Hi - bounds.log10Lo;
      if (span > 0)
        return clamp01((Math.log10(raw as number) - bounds.log10Lo) / span);
    }
  }
  const v = point.values?.[channelIndex];
  return Number.isFinite(v) ? v : 0;
}

function objectsToNodeGroups(
  objects: GlyphObject[],
  channels: string[],
  transform: CoordinateTransform | null,
  encoding: Encoding = "percentile",
  logBounds: (LogBounds | null)[] | null = null,
): NodeData[][] {
  if (!transform) return objects.map(() => []);

  return objects.map((object) =>
    object.points.map((p) => ({
      position: transformPoint(p, transform),
      values: channels.map((_, i) => encodeValue(p, i, encoding, logBounds)),
      signs: channels.map((_, i) => ((p.sgn?.[i] ?? 1) < 0 ? -1 : 1)),
    })),
  );
}

function makeSurfaceGeometry(
  mesh: { vertices: number[][]; faces: number[][] } | undefined,
  transform: CoordinateTransform,
) {
  if (!mesh?.vertices?.length || !mesh.faces?.length) return null;

  const positions: number[] = [];
  for (const vertex of mesh.vertices) {
    if (vertex.length < 3) continue;
    const [x, y, z] = transformPoint(
      { x: vertex[0], y: vertex[1], z: vertex[2] },
      transform,
    );
    positions.push(x, y, z);
  }

  const indices: number[] = [];
  for (const face of mesh.faces) {
    if (face.length < 3) continue;
    indices.push(face[0], face[1], face[2]);
  }

  const buffer = new THREE.BufferGeometry();
  buffer.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(positions, 3),
  );
  buffer.setIndex(indices);
  buffer.computeVertexNormals();
  return buffer;
}

// ── 3D glyph pipeline ─────────────────────────────────────────────────────────

function GlyphPipeline({
  nodes,
  enabledChannelIndices,
  gamma = 1,
  opacity = 1,
  showTube = false,
  animation,
}: {
  nodes: NodeData[];
  enabledChannelIndices?: number[];
  gamma?: number;
  opacity?: number;
  showTube?: boolean;
  animation?: AnimationConfig;
}) {
  const totalChannels = nodes[0]?.values.length || 1;
  const activeChannelIndices =
    enabledChannelIndices ?? Array.from({ length: totalChannels }, (_, i) => i);
  const numChannels = activeChannelIndices.length;
  const nodeRadius = 0.8;
  const maxRadarRadius = 5;
  const scale = 1;

  if (nodes.length < 2 || numChannels === 0) return null;

  const hasValid = nodes.every(
    (b) =>
      Number.isFinite(b.position[0]) &&
      Number.isFinite(b.position[1]) &&
      Number.isFinite(b.position[2]),
  );
  if (!hasValid) return null;

  // Per-node baseline + actual radar vertices
  const nodeVertices = nodes.map((node, nodeIndex) => {
    const pos = [
      node.position[0] * scale,
      node.position[1] * scale,
      node.position[2] * scale,
    ];

    let tangent = new THREE.Vector3(0, 0, 1);
    if (nodeIndex > 0 && nodeIndex < nodes.length - 1) {
      const prev = nodes[nodeIndex - 1].position;
      const next = nodes[nodeIndex + 1].position;
      tangent = new THREE.Vector3(
        next[0] - prev[0],
        next[1] - prev[1],
        next[2] - prev[2],
      );
      if (tangent.lengthSq() > 1e-10) tangent.normalize();
    } else if (nodeIndex === 0 && nodes.length > 1) {
      const next = nodes[1].position;
      tangent = new THREE.Vector3(
        next[0] - pos[0],
        next[1] - pos[1],
        next[2] - pos[2],
      );
      if (tangent.lengthSq() > 1e-10) tangent.normalize();
    } else if (nodeIndex === nodes.length - 1 && nodes.length > 1) {
      const prev = nodes[nodeIndex - 1].position;
      tangent = new THREE.Vector3(
        pos[0] - prev[0],
        pos[1] - prev[1],
        pos[2] - prev[2],
      );
      if (tangent.lengthSq() > 1e-10) tangent.normalize();
    }

    const globalUp = new THREE.Vector3(0, 0, 1);
    const right = new THREE.Vector3();
    const up = new THREE.Vector3();
    right.crossVectors(tangent, globalUp);
    if (right.lengthSq() < 1e-6)
      right.crossVectors(tangent, new THREE.Vector3(1, 0, 0));
    if (right.lengthSq() > 1e-10) right.normalize();
    up.crossVectors(right, tangent);
    if (up.lengthSq() > 1e-10) up.normalize();

    const baselineVertices: THREE.Vector3[] = [];
    const actualVertices: THREE.Vector3[] = [];
    const bRadius = nodeRadius * 1.05;

    for (let i = 0; i < numChannels; i++) {
      const angle = (i / numChannels) * Math.PI * 2;
      const cIdx = activeChannelIndices[i];
      const rawVal = node.values?.[cIdx];
      const tv = Number.isFinite(rawVal)
        ? Math.pow(Math.max(0, rawVal), gamma)
        : 0;

      const bx = Math.cos(angle) * bRadius;
      const by = Math.sin(angle) * bRadius;
      baselineVertices.push(
        new THREE.Vector3(
          pos[0] + right.x * bx + up.x * by,
          pos[1] + right.y * bx + up.y * by,
          pos[2] + right.z * bx + up.z * by,
        ),
      );

      const ar = bRadius + tv * maxRadarRadius;
      const ax = Math.cos(angle) * ar;
      const ay = Math.sin(angle) * ar;
      actualVertices.push(
        new THREE.Vector3(
          pos[0] + right.x * ax + up.x * ay,
          pos[1] + right.y * ax + up.y * ay,
          pos[2] + right.z * ax + up.z * ay,
        ),
      );
    }

    return { baseline: baselineVertices, actual: actualVertices };
  });

  // Backbone connection skeleton
  const centerPts = nodes.map(
    (b) =>
      new THREE.Vector3(
        b.position[0] * scale,
        b.position[1] * scale,
        b.position[2] * scale,
      ),
  );
  const backboneCurve = new THREE.CatmullRomCurve3(
    centerPts,
    false,
    "catmullrom",
    0.3,
  );
  // 8 segments per node still follows the curve at this radius, and costs a third of
  // the triangles the backbone used to spend across every object in the scene.
  const tubeTubularSegments = nodes.length * 8;
  const tubeRadialSegments = 6;

  // The default arc-length table has 200 divisions — about two samples per node,
  // too coarse both for TubeGeometry's own sampling and for tToU below.
  backboneCurve.arcLengthDivisions = tubeTubularSegments;
  backboneCurve.updateArcLengths();

  // Playback progress is a fraction of the *parameter*, which for a Catmull-Rom
  // through equally-spaced-in-time samples is a fraction of elapsed time. The
  // ribbon and the head both live in that space. TubeGeometry does not: it
  // samples via getPointAt, so its segments are spaced by arc length. Points are
  // not equally spaced (spacing tracks particle speed), so the two parameters
  // drift apart — on this data by up to 17% of the path. Convert before setting
  // the tube's draw range.
  const arcLengths = backboneCurve.getLengths(tubeTubularSegments);
  const totalArcLength = arcLengths[arcLengths.length - 1] || 1;
  const tToU = (t: number) => {
    const x = clamp01(t) * tubeTubularSegments;
    const i = Math.min(tubeTubularSegments - 1, Math.floor(x));
    return (
      (arcLengths[i] + (arcLengths[i + 1] - arcLengths[i]) * (x - i)) /
      totalArcLength
    );
  };

  const tubeGeo = new THREE.TubeGeometry(
    backboneCurve,
    tubeTubularSegments,
    BACKBONE_RADIUS,
    tubeRadialSegments,
    false,
  );

  // Continuous channel ribbon: one Catmull-Rom curve per channel through every
  // node (no per-node slicing → ribbon never breaks at node boundaries), plus a
  // darker highlight patch on the same surface at each node.
  const triVerts: number[] = [];
  const triColors: number[] = [];
  const triIdx: number[] = [];
  let vi = 0;

  const hlVerts: number[] = [];
  const hlColors: number[] = [];
  const hlIdx: number[] = [];
  let hi = 0;

  // subDiv must be even so each node index lands exactly on a sample.
  const subDiv = 8;
  // One quad on each side of the node (node sits on the boundary between them).
  const highlightHalfWidth = 1;
  const totalSamples = (nodes.length - 1) * subDiv + 1;

  // One continuous curve per channel through every node.
  const channelCurves = activeChannelIndices.map((origIdx, ti) => {
    const allBaseline = nodes.map((_, i) => nodeVertices[i].baseline[ti]);
    const allActual = nodes.map((_, i) => nodeVertices[i].actual[ti]);
    const bCurve = new THREE.CatmullRomCurve3(
      allBaseline,
      false,
      "catmullrom",
      0.3,
    );
    const aCurve = new THREE.CatmullRomCurve3(
      allActual,
      false,
      "catmullrom",
      0.3,
    );

    const tc = new THREE.Color(getChannelColor(origIdx));
    const nc = scaleHSL(tc, NEGATIVE_SATURATION, NEGATIVE_LIGHTNESS);

    return {
      bPts: bCurve.getPoints(totalSamples - 1),
      aPts: aCurve.getPoints(totalSamples - 1),
      color: tc,
      darkColor: scaleHSL(tc, 1, 1.5),
      // Same pair for beads where this channel's value is negative.
      negColor: nc,
      negDarkColor: scaleHSL(nc, 1, 1.5),
    };
  });

  // A bead's sign applies to the bead, so a ribbon quad takes the sign of the
  // nearest bead: the colour flips midway between the two beads that differ.
  const signAt = (nodeIndex: number, ci: number) =>
    nodes[nodeIndex]?.signs?.[activeChannelIndices[ci]] ?? 1;

  // Full ribbon (normal channel color). Emitted sample-major — all channels for
  // sample i before sample i+1 — so an index draw range is a prefix in time.
  for (let i = 0; i < totalSamples - 1; i++) {
    const nearestNode = Math.min(
      nodes.length - 1,
      Math.round((i + 0.5) / subDiv),
    );
    for (let ci = 0; ci < channelCurves.length; ci++) {
      const { bPts, aPts, negColor } = channelCurves[ci];
      const color =
        signAt(nearestNode, ci) < 0 ? negColor : channelCurves[ci].color;
      const bl1 = bPts[i],
        bl2 = bPts[i + 1];
      const ac1 = aPts[i],
        ac2 = aPts[i + 1];
      triVerts.push(
        bl1.x,
        bl1.y,
        bl1.z,
        bl2.x,
        bl2.y,
        bl2.z,
        ac1.x,
        ac1.y,
        ac1.z,
        ac2.x,
        ac2.y,
        ac2.z,
      );
      for (let j = 0; j < 4; j++) triColors.push(color.r, color.g, color.b);
      triIdx.push(vi, vi + 1, vi + 2, vi + 1, vi + 3, vi + 2);
      vi += 4;
    }
  }

  // Darker highlight patch on the same ribbon surface at each node, emitted
  // node-major for the same reason. Node bi sits at sample index bi * subDiv.
  const highlightIndexEnds: number[] = [];
  for (let bi = 0; bi < nodes.length; bi++) {
    const center = bi * subDiv;
    const lo = Math.max(0, center - highlightHalfWidth);
    const high = Math.min(totalSamples - 2, center + highlightHalfWidth - 1);
    for (let ci = 0; ci < channelCurves.length; ci++) {
      const { bPts, aPts, negDarkColor } = channelCurves[ci];
      const darkColor =
        signAt(bi, ci) < 0 ? negDarkColor : channelCurves[ci].darkColor;
      for (let i = lo; i <= high; i++) {
        const bl1 = bPts[i],
          bl2 = bPts[i + 1];
        const ac1 = aPts[i],
          ac2 = aPts[i + 1];
        hlVerts.push(
          bl1.x,
          bl1.y,
          bl1.z,
          bl2.x,
          bl2.y,
          bl2.z,
          ac1.x,
          ac1.y,
          ac1.z,
          ac2.x,
          ac2.y,
          ac2.z,
        );
        for (let j = 0; j < 4; j++)
          hlColors.push(darkColor.r, darkColor.g, darkColor.b);
        hlIdx.push(hi, hi + 1, hi + 2, hi + 1, hi + 3, hi + 2);
        hi += 4;
      }
    }
    highlightIndexEnds.push(hlIdx.length);
  }

  const triGeo = new THREE.BufferGeometry();
  triGeo.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(triVerts, 3),
  );
  triGeo.setAttribute("color", new THREE.Float32BufferAttribute(triColors, 3));
  triGeo.setIndex(triIdx);
  triGeo.computeVertexNormals();

  const hlGeo = new THREE.BufferGeometry();
  hlGeo.setAttribute("position", new THREE.Float32BufferAttribute(hlVerts, 3));
  hlGeo.setAttribute("color", new THREE.Float32BufferAttribute(hlColors, 3));
  hlGeo.setIndex(hlIdx);
  hlGeo.computeVertexNormals();

  const animating = Boolean(animation?.active);
  if (animating) {
    // Start hidden so the first frame never flashes the finished trajectory.
    triGeo.setDrawRange(0, 0);
    hlGeo.setDrawRange(0, 0);
    tubeGeo.setDrawRange(0, 0);
  }

  return (
    <group>
      {/* Full ribbon — normal channel color */}
      <mesh geometry={triGeo}>
        <meshBasicMaterial
          vertexColors
          transparent
          opacity={0.8 * opacity}
          side={THREE.DoubleSide}
        />
      </mesh>
      {/* Per-node highlight — same ribbon surface, darker color */}
      <mesh geometry={hlGeo}>
        <meshBasicMaterial
          vertexColors
          transparent={opacity < 1}
          opacity={opacity}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      </mesh>
      {showTube && (
        <mesh geometry={tubeGeo}>
          <meshBasicMaterial
            color={BACKBONE_COLOR}
            transparent={opacity < 1}
            opacity={opacity}
          />
        </mesh>
      )}
      {animating && animation && (
        <>
          <TrailReveal
            animation={animation}
            triGeo={triGeo}
            hlGeo={hlGeo}
            tubeGeo={tubeGeo}
            reveal={{
              totalSteps: totalSamples - 1,
              ribbonIndicesPerStep: numChannels * 6,
              highlightIndexEnds,
              subDiv,
              tubeIndicesPerStep: tubeRadialSegments * 6,
              tubeTubularSegments,
              tToU,
            }}
          />
          <TrailHead
            animation={animation}
            curve={backboneCurve}
            opacity={opacity}
          />
        </>
      )}
    </group>
  );
}

// Reveals the trajectory up to the current playback position by shrinking the
// index draw range of each geometry — no rebuild, no React re-render per frame.
function TrailReveal({
  animation,
  triGeo,
  hlGeo,
  tubeGeo,
  reveal,
}: {
  animation: AnimationConfig;
  triGeo: THREE.BufferGeometry;
  hlGeo: THREE.BufferGeometry;
  tubeGeo: THREE.BufferGeometry;
  reveal: {
    totalSteps: number;
    ribbonIndicesPerStep: number;
    highlightIndexEnds: number[];
    subDiv: number;
    tubeIndicesPerStep: number;
    tubeTubularSegments: number;
    // Parameter fraction -> arc-length fraction, the space TubeGeometry samples in.
    tToU: (t: number) => number;
  };
}) {
  useFrame(() => {
    const progress = clamp01(animation.progressRef.current);
    const steps = Math.round(progress * reveal.totalSteps);

    triGeo.setDrawRange(0, steps * reveal.ribbonIndicesPerStep);

    // A node's highlight patch straddles its sample, so it only appears once
    // the head has moved one sample past it.
    const nodesShown = Math.max(
      0,
      Math.min(
        reveal.highlightIndexEnds.length,
        Math.floor((steps - 1) / reveal.subDiv) + 1,
      ),
    );
    hlGeo.setDrawRange(
      0,
      nodesShown > 0 ? reveal.highlightIndexEnds[nodesShown - 1] : 0,
    );

    tubeGeo.setDrawRange(
      0,
      Math.floor(reveal.tToU(progress) * reveal.tubeTubularSegments) *
        reveal.tubeIndicesPerStep,
    );
  });

  return null;
}

// Bright marker riding the head of the revealed trajectory.
function TrailHead({
  animation,
  curve,
  opacity = 1,
}: {
  animation: AnimationConfig;
  curve: THREE.CatmullRomCurve3;
  opacity?: number;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const scratch = useRef(new THREE.Vector3());

  useFrame(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    curve.getPoint(clamp01(animation.progressRef.current), scratch.current);
    mesh.position.copy(scratch.current);
  });

  return (
    <mesh ref={meshRef}>
      <sphereGeometry args={[2, 20, 20]} />
      <meshStandardMaterial
        color={HEAD_COLOR}
        emissive={HEAD_COLOR}
        emissiveIntensity={0.75}
        transparent={opacity < 1}
        opacity={opacity}
        roughness={0.3}
      />
    </mesh>
  );
}

function EndpointMarker({
  label,
  position,
  targetPosition,
  opacity = 1,
  showCone = true,
  showLabel = true,
}: {
  label: string;
  position: [number, number, number];
  targetPosition?: [number, number, number];
  opacity?: number;
  showCone?: boolean;
  showLabel?: boolean;
}) {
  const markerHeight = ENDPOINT_MARKER_SIZE * 1.2;
  const labelY = -(markerHeight + ENDPOINT_MARKER_SIZE * 0.6);

  const rotation = useMemo(() => {
    if (!targetPosition) return new THREE.Euler(Math.PI / 2, 0, 0);

    const direction = new THREE.Vector3(
      targetPosition[0] - position[0],
      targetPosition[1] - position[1],
      targetPosition[2] - position[2],
    );
    if (direction.lengthSq() < 1e-10) return new THREE.Euler(Math.PI / 2, 0, 0);
    direction.normalize();

    const quaternion = new THREE.Quaternion();
    quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);

    const euler = new THREE.Euler();
    euler.setFromQuaternion(quaternion);
    return euler;
  }, [position, targetPosition]);

  return (
    <group position={position}>
      <group rotation={rotation}>
        {showLabel && (
          <Html
            position={[0, labelY, 0]}
            center
            sprite
            zIndexRange={[220, 120]}
            style={{ pointerEvents: "none" }}
          >
            <div style={endpointLabelStyle}>{label}</div>
          </Html>
        )}
        {showCone && (
          <mesh position={[0, -markerHeight / 2, 0]}>
            <coneGeometry
              args={[ENDPOINT_MARKER_SIZE * 0.5, markerHeight, 6]}
            />
            <meshStandardMaterial
              color={ENDPOINT_MARKER_COLOR}
              transparent={opacity < 1}
              opacity={opacity}
              metalness={0.4}
              roughness={0.2}
              emissive={ENDPOINT_MARKER_COLOR}
              emissiveIntensity={0.3}
            />
          </mesh>
        )}
      </group>
    </group>
  );
}

function ParticleSpheres({
  nodes,
  opacity = 1,
  showTube = false,
  showLabels = true,
  animation,
}: {
  nodes: NodeData[];
  opacity?: number;
  showTube?: boolean;
  showLabels?: boolean;
  animation?: AnimationConfig;
}) {
  const lastIndex = nodes.length - 1;
  const nodeGroups = useRef<(THREE.Group | null)[]>([]);

  // While animating, a node only exists once the head has reached it.
  useFrame(() => {
    if (!animation?.active) return;
    const progress = clamp01(animation.progressRef.current);
    for (let i = 0; i < nodeGroups.current.length; i++) {
      const group = nodeGroups.current[i];
      if (group)
        group.visible = lastIndex <= 0 || i / lastIndex <= progress + 1e-6;
    }
  });

  const animating = Boolean(animation?.active);

  return (
    <group>
      {nodes.map((node, i) => {
        let content: React.ReactNode = null;

        if (i === 0) {
          content = (
            <EndpointMarker
              label="s"
              position={node.position}
              targetPosition={nodes[1]?.position}
              opacity={opacity}
              showCone={!showTube}
              showLabel={showLabels && !animating}
            />
          );
        } else if (i === lastIndex) {
          content = (
            <EndpointMarker
              label="e"
              position={node.position}
              targetPosition={nodes[i - 1]?.position}
              opacity={opacity}
              showCone={!showTube}
              showLabel={showLabels && !animating}
            />
          );
        } else if (!showTube) {
          content = (
            <mesh position={node.position}>
              <sphereGeometry args={[1, 16, 16]} />
              <meshStandardMaterial
                color={BEAD_COLOR}
                emissive={BEAD_COLOR}
                emissiveIntensity={0.18}
                transparent
                opacity={0.92 * opacity}
              />
            </mesh>
          );
        }

        return (
          <group
            key={i}
            ref={(el) => {
              nodeGroups.current[i] = el;
            }}
            visible={!animating || i === 0}
          >
            {content}
          </group>
        );
      })}
    </group>
  );
}

// Applies the case's camera placement and orbit axis, and re-frames when the case
// changes. `setup` comes from the dataset table, so it only changes with the dataset.
function CameraRig({
  setup,
  controlsRef,
}: {
  setup: CameraSetup;
  controlsRef: React.MutableRefObject<OrbitControlsHandle | null>;
}) {
  const camera = useThree((state) => state.camera);

  useEffect(() => {
    camera.up.set(...setup.up);
    camera.position.set(...setup.position);
    camera.lookAt(0, 0, 0);
    camera.updateProjectionMatrix();
    // OrbitControls caches the spherical coordinates it derived from the old camera.
    const controls = controlsRef.current;
    if (controls) {
      controls.target.set(0, 0, 0);
      controls.update();
    }
  }, [camera, controlsRef, setup]);

  return null;
}

// Advances playback time. Lives inside the Canvas so it rides the render loop.
function AnimationDriver({
  progressRef,
  playing,
  speed,
  loop,
  onFinish,
}: {
  progressRef: React.MutableRefObject<number>;
  playing: boolean;
  speed: number;
  loop: boolean;
  onFinish: () => void;
}) {
  useFrame((_, delta) => {
    if (!playing) return;
    const next = progressRef.current + (delta * speed) / ANIMATION_DURATION;
    if (next >= 1) {
      if (loop) {
        progressRef.current = next % 1;
      } else {
        progressRef.current = 1;
        onFinish();
      }
      return;
    }
    progressRef.current = next;
  });

  return null;
}

// Translucent bundle envelopes plus the brain-mask isosurface. The envelope is
// the spatial extent the glyph's core path summarizes, so a tube is drawn only
// for bundles whose glyph is currently visible.
function TractOverlay({
  geometry,
  transform,
  showContext,
  visibleObjectIds,
}: {
  geometry: TractGeometry;
  transform: CoordinateTransform;
  showContext: boolean;
  visibleObjectIds: Set<number>;
}) {
  const contextGeometry = useMemo(
    () => makeSurfaceGeometry(geometry.context?.mesh, transform),
    [geometry.context?.mesh, transform],
  );

  const whiteMatterGeometry = useMemo(
    () => makeSurfaceGeometry(geometry.context?.whiteMatter?.mesh, transform),
    [geometry.context?.whiteMatter?.mesh, transform],
  );

  const bundleGeometries = useMemo(
    () =>
      (geometry.bundles ?? []).map((bundle) => ({
        objectId: bundle.objectId,
        geometry: makeSurfaceGeometry(bundle.mesh, transform),
      })),
    [geometry.bundles, transform],
  );

  return (
    <group>
      {showContext && contextGeometry && (
        <mesh geometry={contextGeometry} renderOrder={-2}>
          <meshStandardMaterial
            color="#9fb4c7"
            emissive="#2a3946"
            emissiveIntensity={0.1}
            transparent
            opacity={0.05}
            roughness={0.85}
            metalness={0.02}
            side={THREE.BackSide}
            depthWrite={false}
          />
        </mesh>
      )}
      {/* The hull alone reads as a featureless blob. The white-matter boundary is what
          carries recognizable anatomy — gyral folding, the interhemispheric fissure, the
          callosal arch — so it is drawn solidly enough to be read. Front faces only: the
          near-side folds are the recognizable ones, and one layer of translucency keeps
          the bundles inside legible. It writes no depth and renders first, so the glyphs
          are never occluded by it. */}
      {showContext && whiteMatterGeometry && (
        <mesh geometry={whiteMatterGeometry} renderOrder={-1}>
          <meshStandardMaterial
            color="#b9c9d6"
            emissive="#1b2732"
            emissiveIntensity={0.25}
            transparent
            opacity={0.3}
            roughness={0.72}
            metalness={0.04}
            side={THREE.FrontSide}
            depthWrite={false}
            flatShading
          />
        </mesh>
      )}
      {bundleGeometries.map(({ objectId, geometry: mesh }) =>
        mesh && visibleObjectIds.has(objectId) ? (
          <mesh key={objectId} geometry={mesh}>
            <meshStandardMaterial
              color="#7fd1c4"
              emissive="#7fd1c4"
              emissiveIntensity={0.12}
              transparent
              opacity={0.16}
              roughness={0.55}
              metalness={0.05}
              side={THREE.DoubleSide}
              depthWrite={false}
            />
          </mesh>
        ) : null,
      )}
    </group>
  );
}

// ── Main exported viewer ────────────────────────────────────────────────────

export default function GlyphView3D() {
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const cameraRef = useRef<THREE.Camera | null>(null);
  const orbitControlsRef = useRef<OrbitControlsHandle | null>(null);
  const dragStartRef = useRef<{ x: number; y: number } | null>(null);
  const [datasetIdx, setDatasetIdx] = useState(0);
  const [data, setData] = useState<GlyphDataset | null>(null);
  const [geometry, setGeometry] = useState<TractGeometry | null>(null);
  const [channels, setChannels] = useState<string[]>([]);
  const [enabledChannels, setEnabledChannels] = useState<Set<number>>(
    new Set(),
  );
  const [sampleCount, setSampleCount] = useState(DEFAULT_SAMPLE_COUNT);
  const [gamma, setGamma] = useState(DEFAULT_GAMMA);
  const [encoding, setEncoding] = useState<Encoding>("percentile");
  const [showTube] = useState(true);
  const [showLabels] = useState(false);
  const [showContext, setShowContext] = useState(true);
  const [selectMode, setSelectMode] = useState(false);
  const [hiddenObjectIds, setHiddenObjectIds] = useState<Set<number>>(
    new Set(),
  );
  const [selectedObjectIds, setSelectedObjectIds] = useState<Set<number>>(
    new Set(),
  );
  const [selectionRect, setSelectionRect] = useState<SelectionRect | null>(
    null,
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const progressRef = useRef(0);
  const [animate, setAnimate] = useState(false);
  const [playing, setPlaying] = useState(false);

  const dataset = AVAILABLE_DATASETS[datasetIdx];

  // Channels whose fin length is a magnitude; the toggle row shows both tones so
  // the dark ribbon segments are readable as "negative" rather than as a bug.
  const hiddenChannels = useMemo(() => hiddenChannelSet(data), [data]);

  const signedChannelSet = useMemo(
    () => new Set(data?.meta?.signedChannels ?? []),
    [data],
  );

  // Load the dataset JSON whenever the selected dataset changes.
  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      setLoading(true);
      setError(null);
      // Drop the previous dataset's geometry so it is not drawn against the new
      // glyphs while the fetch is in flight.
      setGeometry(null);
      try {
        const res = await fetch(dataset.path);
        if (!res.ok) throw new Error(`Dataset not found (${res.status})`);
        const json: GlyphDataset = await res.json();
        let geometryJson: TractGeometry | null = null;
        if (dataset.geometryPath) {
          const geometryRes = await fetch(dataset.geometryPath);
          if (!geometryRes.ok)
            throw new Error(`Geometry not found (${geometryRes.status})`);
          geometryJson = await geometryRes.json();
        }
        if (cancelled) return;

        setData(json);
        setGeometry(geometryJson);
        setChannels(json.channels);
        setEnabledChannels(
          new Set(
            json.channels
              .map((_, i) => i)
              .filter((i) => !hiddenChannelSet(json).has(json.channels[i])),
          ),
        );
        setSampleCount(
          Math.min(
            dataset.defaultSampleCount ?? DEFAULT_SAMPLE_COUNT,
            json.objects.length,
          ),
        );
        setGamma(dataset.defaultGamma ?? DEFAULT_GAMMA);
        setEncoding("percentile");
        setShowContext(true);
        setSelectMode(false);
        setHiddenObjectIds(new Set());
        setSelectedObjectIds(new Set());
        progressRef.current = 0;
        setAnimate(false);
        setPlaying(false);
      } catch (err) {
        if (!cancelled) {
          setGeometry(null);
          setError(err instanceof Error ? err.message : String(err));
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    load();
    return () => {
      cancelled = true;
    };
  }, [dataset]);

  const sampledObjects = useMemo(
    () => sampleObjects(data?.objects ?? [], sampleCount, RANDOM_SEED),
    [data, sampleCount],
  );

  const coordinateTransform = useMemo(
    () => getCoordinateTransform(sampledObjects),
    [sampledObjects],
  );

  // The log encoding needs both halves of the export: published bounds for every channel
  // and a physical value on the points themselves. Without both, the toggle is not shown
  // rather than silently drawing percentiles under a "log" label.
  const logBounds = useMemo(() => {
    const published = data?.meta?.logNormalization;
    if (!published || !channels.length) return null;
    const bounds = channels.map((name) => published[name] ?? null);
    return bounds.some((b) => b) ? bounds : null;
  }, [data, channels]);

  const supportsLogEncoding = useMemo(
    () => Boolean(logBounds && data?.objects?.[0]?.points?.[0]?.raw?.length),
    [logBounds, data],
  );

  const sampledNodes = useMemo(
    () =>
      objectsToNodeGroups(
        sampledObjects,
        channels,
        coordinateTransform,
        encoding,
        logBounds,
      ),
    [coordinateTransform, sampledObjects, channels, encoding, logBounds],
  );

  const visibleItems = useMemo(
    () =>
      sampledObjects
        .map((object, i) => ({ object, nodes: sampledNodes[i] ?? [] }))
        .filter(({ object }) => !hiddenObjectIds.has(object.objectId)),
    [hiddenObjectIds, sampledNodes, sampledObjects],
  );

  const enabledChannelIndices = useMemo(
    () => [...enabledChannels].sort((a, b) => a - b),
    [enabledChannels],
  );

  const tractGeometry = geometry;

  const hasContextMesh = Boolean(
    tractGeometry?.context?.mesh?.vertices?.length,
  );
  const visibleObjectIds = useMemo(
    () => new Set(visibleItems.map(({ object }) => object.objectId)),
    [visibleItems],
  );
  // Sampling how many objects to draw is useful everywhere; picking individual ones out of
  // the scene is only offered where the dataset opts in.
  const supportsSampling = (data?.objects.length ?? 0) > 1;
  // Gamma is calibrated to the encoding it acts on — v^20 keeps the top decile of a
  // uniform percentile, and would erase a log encoding whose median already sits near
  // 0.5 — so it follows the switch instead of carrying over.
  const selectEncoding = useCallback(
    (next: Encoding) => {
      if (next === encoding) return;
      setEncoding(next);
      setGamma(
        next === "log"
          ? DEFAULT_LOG_GAMMA
          : (dataset.defaultGamma ?? DEFAULT_GAMMA),
      );
    },
    [encoding, dataset],
  );

  const supportsSelection = Boolean(dataset.supportsSelection && supportsSampling);
  const supportsAnimation = Boolean(dataset.supportsAnimation && data);

  const animation = useMemo<AnimationConfig>(
    () => ({ progressRef, active: animate }),
    [animate],
  );

  const toggleAnimate = useCallback((enabled: boolean) => {
    progressRef.current = 0;
    setAnimate(enabled);
    setPlaying(enabled);
  }, []);

  const togglePlay = useCallback(() => {
    setPlaying((prev) => {
      // Restarting from the end rather than resuming a finished run.
      if (!prev && progressRef.current >= 1) progressRef.current = 0;
      return !prev;
    });
  }, []);

  const toggleChannel = useCallback((idx: number) => {
    setEnabledChannels((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx);
      else next.add(idx);
      return next;
    });
  }, []);

  const updateSampleCount = useCallback(
    (value: number) => {
      if (!Number.isFinite(value)) return;
      const maxCount = data?.objects.length ?? DEFAULT_SAMPLE_COUNT;
      setSampleCount(Math.max(1, Math.min(maxCount, Math.floor(value))));
    },
    [data],
  );

  const hideSelected = useCallback(() => {
    setHiddenObjectIds((prev) => {
      const next = new Set(prev);
      selectedObjectIds.forEach((id) => next.add(id));
      return next;
    });
    setSelectedObjectIds(new Set());
  }, [selectedObjectIds]);

  const showAllObjects = useCallback(() => {
    setHiddenObjectIds(new Set());
    setSelectedObjectIds(new Set());
  }, []);

  const updateDragRect = useCallback((x: number, y: number) => {
    const start = dragStartRef.current;
    if (!start) return;
    setSelectionRect({
      x: Math.min(start.x, x),
      y: Math.min(start.y, y),
      width: Math.abs(x - start.x),
      height: Math.abs(y - start.y),
    });
  }, []);

  const selectItemsInRect = useCallback(
    (rect: SelectionRect, additive: boolean) => {
      const camera = cameraRef.current;
      const viewport = viewportRef.current;
      if (!camera || !viewport || rect.width < 4 || rect.height < 4) return;

      camera.updateMatrixWorld();
      const point = new THREE.Vector3();
      const nextSelected = additive
        ? new Set(selectedObjectIds)
        : new Set<number>();

      for (const { object, nodes } of visibleItems) {
        let hits = 0;
        let endpointHit = false;

        for (let i = 0; i < nodes.length; i++) {
          const [x, y, z] = nodes[i].position;
          point.set(x, y, z).project(camera);
          if (point.z < -1 || point.z > 1) continue;

          const sx = ((point.x + 1) / 2) * viewport.clientWidth;
          const sy = ((-point.y + 1) / 2) * viewport.clientHeight;
          const inside =
            sx >= rect.x &&
            sx <= rect.x + rect.width &&
            sy >= rect.y &&
            sy <= rect.y + rect.height;

          if (inside) {
            hits += 1;
            if (i === 0 || i === nodes.length - 1) endpointHit = true;
          }
        }

        if (endpointHit || hits >= 3) nextSelected.add(object.objectId);
      }

      setSelectedObjectIds(nextSelected);
    },
    [selectedObjectIds, visibleItems],
  );

  const startSelection = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      if (event.button !== 0) return;
      const viewport = viewportRef.current;
      if (!viewport) return;

      const bounds = viewport.getBoundingClientRect();
      const x = event.clientX - bounds.left;
      const y = event.clientY - bounds.top;
      dragStartRef.current = { x, y };
      setSelectionRect({ x, y, width: 0, height: 0 });
      event.currentTarget.setPointerCapture(event.pointerId);
    },
    [],
  );

  const moveSelection = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      if (!dragStartRef.current) return;
      const viewport = viewportRef.current;
      if (!viewport) return;

      const bounds = viewport.getBoundingClientRect();
      updateDragRect(event.clientX - bounds.left, event.clientY - bounds.top);
    },
    [updateDragRect],
  );

  const finishSelection = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      const start = dragStartRef.current;
      const viewport = viewportRef.current;
      if (!start || !viewport) return;

      const bounds = viewport.getBoundingClientRect();
      const x = event.clientX - bounds.left;
      const y = event.clientY - bounds.top;
      const finalRect = {
        x: Math.min(start.x, x),
        y: Math.min(start.y, y),
        width: Math.abs(x - start.x),
        height: Math.abs(y - start.y),
      };

      selectItemsInRect(finalRect, event.shiftKey);
      dragStartRef.current = null;
      setSelectionRect(null);
      event.currentTarget.releasePointerCapture(event.pointerId);
    },
    [selectItemsInRect],
  );

  // ── Render ──

  return (
    <div
      style={{
        width: "100vw",
        height: "100vh",
        display: "flex",
        flexDirection: "column",
        background: "#0a1929",
      }}
    >
      {/* ── Control bar ── */}
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          alignItems: "center",
          gap: 12,
          padding: "10px 16px",
          background: "rgba(255,255,255,0.06)",
          borderBottom: "1px solid rgba(255,255,255,0.1)",
          color: "#e0e0e0",
          fontSize: 13,
          fontFamily: "system-ui, sans-serif",
        }}
      >
        {/* Dataset selector */}
        <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
          Dataset
          <select
            value={datasetIdx}
            onChange={(e) => setDatasetIdx(Number(e.target.value))}
            style={selectStyle}
          >
            {AVAILABLE_DATASETS.map((d, i) => (
              <option key={i} value={i}>
                {d.name}
              </option>
            ))}
          </select>
        </label>

        {supportsSampling && (
          <>
            <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
              Counts
              <input
                type="number"
                min={1}
                max={data?.objects.length ?? undefined}
                step={1}
                value={sampleCount}
                onChange={(e) => updateSampleCount(Number(e.target.value))}
                style={numberInputStyle}
              />
            </label>

            <div
              style={{
                width: 1,
                height: 20,
                background: "rgba(255,255,255,0.15)",
              }}
            />
          </>
        )}

        {/* Channel toggles */}
        {channels.map((name, i) =>
          hiddenChannels.has(name) ? null : (
            <label
              key={name}
              title={channelTooltip(name, data?.meta, encoding)}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 4,
                cursor: "pointer",
                opacity: enabledChannels.has(i) ? 1 : 0.4,
              }}
            >
              <input
                type="checkbox"
                checked={enabledChannels.has(i)}
                onChange={() => toggleChannel(i)}
                style={{ accentColor: getChannelColor(i) }}
              />
              {channelLabel(name, data?.meta)}
              {signedChannelSet.has(name) && (
                <span
                  title={`${channelLabel(name, data?.meta)}: fin length is |value|; bright = positive, dark = negative`}
                  style={{ display: "inline-flex", alignItems: "center", gap: 2 }}
                >
                  <span
                    style={{
                      width: 9,
                      height: 9,
                      borderRadius: 2,
                      background: getChannelColor(i),
                    }}
                  />
                  <span
                    style={{
                      width: 9,
                      height: 9,
                      borderRadius: 2,
                      background: getNegativeChannelColor(i),
                    }}
                  />
                  <span style={{ fontSize: 9, opacity: 0.6 }}>+/-</span>
                </span>
              )}
            </label>
          ),
        )}

        <div
          style={{ width: 1, height: 20, background: "rgba(255,255,255,0.15)" }}
        />

        {/* <label style={{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer" }}>
          <input
            type="checkbox"
            checked={showTube}
            onChange={(e) => setShowTube(e.target.checked)}
            style={{ accentColor: "#45b7d1" }}
          />
          Tube
        </label>

        <label style={{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer" }}>
          <input
            type="checkbox"
            checked={showLabels}
            onChange={(e) => setShowLabels(e.target.checked)}
            style={{ accentColor: "#45b7d1" }}
          />
          Labels
        </label> */}

        {hasContextMesh && (
          <label
            style={{
              display: "flex",
              alignItems: "center",
              gap: 6,
              cursor: "pointer",
            }}
          >
            <input
              type="checkbox"
              checked={showContext}
              onChange={(e) => setShowContext(e.target.checked)}
              style={{ accentColor: "#8ca3ad" }}
            />
            {dataset.contextLabel ?? "Context"}
          </label>
        )}

        {supportsSelection && (
          <>
            <button
              type="button"
              onClick={() => setSelectMode((value) => !value)}
              style={selectMode ? activeButtonStyle : buttonStyle}
            >
              Select
            </button>

            <button
              type="button"
              onClick={hideSelected}
              disabled={selectedObjectIds.size === 0}
              style={selectedObjectIds.size === 0 ? disabledButtonStyle : buttonStyle}
            >
              Hide selected
            </button>

            <button
              type="button"
              onClick={showAllObjects}
              disabled={hiddenObjectIds.size === 0 && selectedObjectIds.size === 0}
              style={hiddenObjectIds.size === 0 && selectedObjectIds.size === 0 ? disabledButtonStyle : buttonStyle}
            >
              Show all
            </button>

            <span style={{ color: "rgba(224,224,224,0.72)", fontVariantNumeric: "tabular-nums" }}>
              {selectedObjectIds.size} selected / {hiddenObjectIds.size} hidden
            </span>
          </>
        )}

        <div
          style={{ width: 1, height: 20, background: "rgba(255,255,255,0.15)" }}
        />

        {/* What a fin length means. Only offered where the export ships both the
            percentile and a physical value with published log bounds. */}
        {supportsLogEncoding && (
          <>
            <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
              Fin length
              <span style={{ display: "inline-flex", gap: 4 }}>
                <button
                  type="button"
                  aria-pressed={encoding === "percentile"}
                  onClick={() => selectEncoding("percentile")}
                  title="Rank of the measurement among all exported nodes. Comparable across channels; compresses the extreme tail."
                  style={
                    encoding === "percentile" ? activeButtonStyle : buttonStyle
                  }
                >
                  Percentile
                </button>
                <button
                  type="button"
                  aria-pressed={encoding === "log"}
                  onClick={() => selectEncoding("log")}
                  title="log10 of the physical value, mapped between the bounds in meta.logNormalization. Keeps magnitude ratios visible; comparable only within a channel."
                  style={encoding === "log" ? activeButtonStyle : buttonStyle}
                >
                  log(raw)
                </button>
              </span>
            </label>

            <div
              style={{
                width: 1,
                height: 20,
                background: "rgba(255,255,255,0.15)",
              }}
            />
          </>
        )}

        {/* Gamma control — remaps each channel value as v^gamma */}
        <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
          Gamma
          <input
            type="range"
            min={1}
            max={100}
            step={1}
            value={gamma}
            onChange={(e) => setGamma(Number(e.target.value))}
            style={{ width: 120, accentColor: "#45b7d1" }}
          />
          <span
            style={{
              width: 28,
              textAlign: "right",
              fontVariantNumeric: "tabular-nums",
            }}
          >
            {gamma.toFixed(1)}
          </span>
        </label>

        {supportsAnimation && (
          <>
            <button
              type="button"
              aria-label={playing ? "Pause animation" : "Play animation"}
              aria-pressed={playing}
              title={playing ? "Pause" : "Play"}
              onClick={() => (animate ? togglePlay() : toggleAnimate(true))}
              style={{
                width: 24,
                height: 24,
                padding: 0,
                display: "inline-flex",
                alignItems: "center",
                justifyContent: "center",
                cursor: "pointer",
                color: playing ? "#fff" : "rgba(255,255,255,0.75)",
                background: playing ? "#1677ff" : "transparent",
                border: playing
                  ? "1px solid #1677ff"
                  : "1px solid rgba(255,255,255,0.3)",
                borderRadius: 4,
                transition: "all 0.2s",
              }}
            >
              {playing ? (
                <svg
                  viewBox="0 0 16 16"
                  width="14"
                  height="14"
                  aria-hidden="true"
                  fill="currentColor"
                >
                  <path d="M4.25 3h2.5v10h-2.5zM9.25 3h2.5v10h-2.5z" />
                </svg>
              ) : (
                <svg
                  viewBox="0 0 16 16"
                  width="14"
                  height="14"
                  aria-hidden="true"
                  fill="currentColor"
                >
                  <path d="M5 3.25a.75.75 0 0 1 1.14-.64l6 3.75a.75.75 0 0 1 0 1.28l-6 3.75A.75.75 0 0 1 5 10.75v-7.5Z" />
                </svg>
              )}
            </button>
          </>
        )}
      </div>

      {/* ── 3D viewport ── */}
      <div ref={viewportRef} style={{ flex: 1, position: "relative" }}>
        {loading && (
          <div style={overlayStyle}>
            <div style={{ fontSize: 14, color: "rgba(255,255,255,0.7)" }}>
              Loading 3D structure...
            </div>
          </div>
        )}
        {error && (
          <div style={overlayStyle}>
            <div style={{ color: "#ff6b6b", fontSize: 14 }}>{error}</div>
          </div>
        )}
        <Canvas
          camera={{ position: [0, 0, 500], fov: 60, near: 0.1, far: 10000 }}
          onCreated={({ camera }) => {
            cameraRef.current = camera;
          }}
          style={{ background: "#0a1929", width: "100%", height: "100%" }}
        >
          <ambientLight intensity={0.6} />
          <directionalLight position={[100, 100, 100]} intensity={0.8} />
          <CameraRig
            setup={dataset.camera ?? DEFAULT_CAMERA}
            controlsRef={orbitControlsRef}
          />
          {animate && (
            <AnimationDriver
              progressRef={progressRef}
              playing={playing}
              speed={1}
              loop={false}
              onFinish={() => setPlaying(false)}
            />
          )}
          {visibleItems.map(({ object, nodes }) => {
            if (nodes.length <= 1) return null;
            const isSelected = selectedObjectIds.has(object.objectId);
            const opacity = isSelected ? 0.42 : 1;
            return (
              <group key={object.objectId}>
                <GlyphPipeline
                  nodes={nodes}
                  enabledChannelIndices={enabledChannelIndices}
                  gamma={gamma}
                  opacity={opacity}
                  showTube={showTube}
                  animation={animation}
                />
                <ParticleSpheres
                  nodes={nodes}
                  opacity={opacity}
                  showTube={showTube}
                  showLabels={showLabels}
                  animation={animation}
                />
              </group>
            );
          })}
          {tractGeometry && coordinateTransform && (
            <TractOverlay
              geometry={tractGeometry}
              transform={coordinateTransform}
              showContext={showContext}
              visibleObjectIds={visibleObjectIds}
            />
          )}
          <OrbitControls
            ref={orbitControlsRef}
            enableZoom={!supportsSelection || !selectMode}
            enablePan={!supportsSelection || !selectMode}
            enableRotate={!supportsSelection || !selectMode}
          />
        </Canvas>
        {supportsSelection && selectMode && (
          <div
            style={selectionLayerStyle}
            onPointerDown={startSelection}
            onPointerMove={moveSelection}
            onPointerUp={finishSelection}
            onPointerCancel={() => {
              dragStartRef.current = null;
              setSelectionRect(null);
            }}
          >
            {selectionRect && <div style={selectionBoxStyle(selectionRect)} />}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Shared inline styles ────────────────────────────────────────────────────

const selectStyle: React.CSSProperties = {
  background: "#1a2a3e",
  color: "#e0e0e0",
  border: "1px solid rgba(255,255,255,0.2)",
  borderRadius: 4,
  padding: "3px 6px",
  fontSize: 13,
};

const numberInputStyle: React.CSSProperties = {
  ...selectStyle,
  width: 64,
};

const buttonStyle: React.CSSProperties = {
  background: "#1a2a3e",
  color: "#e0e0e0",
  border: "1px solid rgba(255,255,255,0.2)",
  borderRadius: 4,
  padding: "4px 8px",
  fontSize: 13,
  cursor: "pointer",
};

const activeButtonStyle: React.CSSProperties = {
  ...buttonStyle,
  background: "#256d85",
  borderColor: "rgba(69,183,209,0.85)",
};

const disabledButtonStyle: React.CSSProperties = {
  ...buttonStyle,
  opacity: 0.45,
  cursor: "not-allowed",
};

const overlayStyle: React.CSSProperties = {
  position: "absolute",
  inset: 0,
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  zIndex: 10,
  background: "rgba(10,25,41,0.8)",
};

const selectionLayerStyle: React.CSSProperties = {
  position: "absolute",
  inset: 0,
  zIndex: 6,
  cursor: "crosshair",
  touchAction: "none",
};

function selectionBoxStyle(rect: SelectionRect): React.CSSProperties {
  return {
    position: "absolute",
    left: rect.x,
    top: rect.y,
    width: rect.width,
    height: rect.height,
    border: "1px solid rgba(69,183,209,0.95)",
    background: "rgba(69,183,209,0.16)",
    boxShadow: "0 0 0 1px rgba(0,0,0,0.35)",
    pointerEvents: "none",
  };
}

const endpointLabelStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  background: "rgba(0, 0, 0, 0.55)",
  color: "#ffffff",
  border: "1px solid rgba(255, 255, 255, 0.2)",
  borderRadius: 6,
  padding: "2px 6px 1px 7px",
  fontSize: 12,
  fontWeight: 600,
  lineHeight: 1,
  letterSpacing: "0.3px",
  textIndent: "0.3px",
  textTransform: "uppercase",
  whiteSpace: "nowrap",
};
