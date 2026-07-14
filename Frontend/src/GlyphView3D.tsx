import { useEffect, useState, useMemo, useCallback, useRef } from "react";
import { Canvas } from "@react-three/fiber";
import { Html, OrbitControls } from "@react-three/drei";
import * as THREE from "three";

// ── Data types ──────────────────────────────────────────────────────────────

interface GlyphPoint {
  x: number;
  y: number;
  z: number;
  values: number[];
  attributes?: Record<string, unknown>;
}

interface GlyphObject {
  objectId: number;
  label?: string;
  points: GlyphPoint[];
}

interface GlyphDataset {
  meta?: Record<string, unknown> & {
    title?: string;
    description?: string;
    unit?: string;
  };
  channels: string[];
  objects: GlyphObject[];
}

interface AneurysmGeometry {
  meta?: Record<string, unknown>;
  center: { x: number; y: number; z: number };
  size: {
    bboxMin?: number[];
    bboxMax?: number[];
    extent?: number[];
    maxRadiusFromCenter?: number;
    meanRadiusFromCenter?: number;
    equivalentDiameterApprox?: number;
  };
  neck?: {
    x: number;
    y: number;
    z: number;
    node?: number;
    arclen?: number;
    dist_to_sac?: number;
  };
  mesh?: {
    vertices: number[][];
    faces: number[][];
    originalVertexIds?: number[];
    distanceToNeck?: number[];
  };
  vessel?: {
    description?: string;
    center?: { x: number; y: number; z: number };
    size?: {
      bboxMin?: number[];
      bboxMax?: number[];
      extent?: number[];
    };
    mesh?: {
      vertices: number[][];
      faces: number[][];
      originalVertexIds?: number[];
      distanceToNeck?: number[];
    };
    fields?: string[];
    distanceToNeckRange?: number[];
    nWallPoints?: number;
    nWallFaces?: number;
    nVesselMeshVertices?: number;
    nVesselMeshFaces?: number;
  };
  nSacWallPoints?: number;
}

interface CoordinateTransform {
  center: [number, number, number];
  scale: number;
}

// Internal per-point representation consumed by the 3D pipeline.
interface NodeData {
  position: [number, number, number];
  values: number[];
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
  defaultSampleCount?: number;
}

// World-space extent the sampled point clouds are scaled to, so camera framing
// and glyph sizes work while preserving their relative source positions.
const TARGET_EXTENT = 200;
const RANDOM_SEED = 3601;
const DEFAULT_SAMPLE_COUNT = 4;

const AVAILABLE_DATASETS: DatasetConfig[] = [
  { name: "Turbulence tracers", path: "/turb_glyph.json", defaultSampleCount: DEFAULT_SAMPLE_COUNT },
  {
    name: "Aneurysm",
    path: "/aneurysm_glyph.json",
    geometryPath: "/aneurysm_geometry.json",
    defaultSampleCount: 1,
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

const BACKBONE_COLOR = "#FFFFFF";
const BEAD_COLOR = "#FFFFFF";
const BACKBONE_RADIUS = 0.25;
const ENDPOINT_MARKER_COLOR = "#999";
const ENDPOINT_MARKER_SIZE = 5;

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

function getCoordinateTransform(objects: GlyphObject[]): CoordinateTransform | null {
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

  const center = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2];
  const extent = Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2]) || 1;
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

function objectsToNodeGroups(
  objects: GlyphObject[],
  channels: string[],
  transform: CoordinateTransform | null,
): NodeData[][] {
  if (!transform) return objects.map(() => []);

  return objects.map((object) =>
    object.points.map((p) => ({
      position: transformPoint(p, transform),
      values: channels.map((_, i) => {
        const v = p.values?.[i];
        return Number.isFinite(v) ? v : 0;
      }),
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
    const [x, y, z] = transformPoint({ x: vertex[0], y: vertex[1], z: vertex[2] }, transform);
    positions.push(x, y, z);
  }

  const indices: number[] = [];
  for (const face of mesh.faces) {
    if (face.length < 3) continue;
    indices.push(face[0], face[1], face[2]);
  }

  const buffer = new THREE.BufferGeometry();
  buffer.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
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
}: {
  nodes: NodeData[];
  enabledChannelIndices?: number[];
  gamma?: number;
  opacity?: number;
  showTube?: boolean;
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
      tangent = new THREE.Vector3(next[0] - pos[0], next[1] - pos[1], next[2] - pos[2]);
      if (tangent.lengthSq() > 1e-10) tangent.normalize();
    } else if (nodeIndex === nodes.length - 1 && nodes.length > 1) {
      const prev = nodes[nodeIndex - 1].position;
      tangent = new THREE.Vector3(pos[0] - prev[0], pos[1] - prev[1], pos[2] - prev[2]);
      if (tangent.lengthSq() > 1e-10) tangent.normalize();
    }

    const globalUp = new THREE.Vector3(0, 0, 1);
    const right = new THREE.Vector3();
    const up = new THREE.Vector3();
    right.crossVectors(tangent, globalUp);
    if (right.lengthSq() < 1e-6) right.crossVectors(tangent, new THREE.Vector3(1, 0, 0));
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
      const tv = Number.isFinite(rawVal) ? Math.pow(Math.max(0, rawVal), gamma) : 0;

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
    (b) => new THREE.Vector3(b.position[0] * scale, b.position[1] * scale, b.position[2] * scale),
  );
  const backboneCurve = new THREE.CatmullRomCurve3(centerPts, false, "catmullrom", 0.3);
  const tubeGeo = new THREE.TubeGeometry(backboneCurve, nodes.length * 20, BACKBONE_RADIUS, 8, false);

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

  for (let ti = 0; ti < numChannels; ti++) {
    const allBaseline = nodes.map((_, i) => nodeVertices[i].baseline[ti]);
    const allActual = nodes.map((_, i) => nodeVertices[i].actual[ti]);

    // One continuous curve through every node — no per-segment slicing needed.
    const bCurve = new THREE.CatmullRomCurve3(allBaseline, false, "catmullrom", 0.3);
    const aCurve = new THREE.CatmullRomCurve3(allActual, false, "catmullrom", 0.3);

    const totalSamples = (nodes.length - 1) * subDiv + 1;
    const bPts = bCurve.getPoints(totalSamples - 1);
    const aPts = aCurve.getPoints(totalSamples - 1);

    const origIdx = activeChannelIndices[ti];
    const tc = new THREE.Color(getChannelColor(origIdx));
    const hsl = { h: 0, s: 0, l: 0 };
    tc.getHSL(hsl);
    const darkColor = new THREE.Color().setHSL(hsl.h, hsl.s, hsl.l * 1.5);

    // Full ribbon (normal channel color)
    for (let i = 0; i < bPts.length - 1; i++) {
      const bl1 = bPts[i], bl2 = bPts[i + 1];
      const ac1 = aPts[i], ac2 = aPts[i + 1];
      triVerts.push(
        bl1.x, bl1.y, bl1.z, bl2.x, bl2.y, bl2.z,
        ac1.x, ac1.y, ac1.z, ac2.x, ac2.y, ac2.z,
      );
      for (let j = 0; j < 4; j++) triColors.push(tc.r, tc.g, tc.b);
      triIdx.push(vi, vi + 1, vi + 2, vi + 1, vi + 3, vi + 2);
      vi += 4;
    }

    // Darker highlight patch on the same ribbon surface at each node.
    // Node bi sits at sample index (bi * subDiv) in the array.
    for (let bi = 0; bi < nodes.length; bi++) {
      const center = bi * subDiv;
      const lo = Math.max(0, center - highlightHalfWidth);
      const high = Math.min(bPts.length - 2, center + highlightHalfWidth - 1);
      for (let i = lo; i <= high; i++) {
        const bl1 = bPts[i], bl2 = bPts[i + 1];
        const ac1 = aPts[i], ac2 = aPts[i + 1];
        hlVerts.push(
          bl1.x, bl1.y, bl1.z, bl2.x, bl2.y, bl2.z,
          ac1.x, ac1.y, ac1.z, ac2.x, ac2.y, ac2.z,
        );
        for (let j = 0; j < 4; j++) hlColors.push(darkColor.r, darkColor.g, darkColor.b);
        hlIdx.push(hi, hi + 1, hi + 2, hi + 1, hi + 3, hi + 2);
        hi += 4;
      }
    }
  }

  const triGeo = new THREE.BufferGeometry();
  triGeo.setAttribute("position", new THREE.Float32BufferAttribute(triVerts, 3));
  triGeo.setAttribute("color", new THREE.Float32BufferAttribute(triColors, 3));
  triGeo.setIndex(triIdx);
  triGeo.computeVertexNormals();

  const hlGeo = new THREE.BufferGeometry();
  hlGeo.setAttribute("position", new THREE.Float32BufferAttribute(hlVerts, 3));
  hlGeo.setAttribute("color", new THREE.Float32BufferAttribute(hlColors, 3));
  hlGeo.setIndex(hlIdx);
  hlGeo.computeVertexNormals();

  return (
    <group>
      {/* Full ribbon — normal channel color */}
      <mesh geometry={triGeo}>
        <meshBasicMaterial vertexColors transparent opacity={0.8 * opacity} side={THREE.DoubleSide} />
      </mesh>
      {/* Per-node highlight — same ribbon surface, darker color */}
      <mesh geometry={hlGeo}>
        <meshBasicMaterial vertexColors transparent={opacity < 1} opacity={opacity} side={THREE.DoubleSide} depthWrite={false} />
      </mesh>
      {showTube && (
        <mesh geometry={tubeGeo}>
          <meshBasicMaterial color={BACKBONE_COLOR} transparent={opacity < 1} opacity={opacity} />
        </mesh>
      )}
    </group>
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
            <coneGeometry args={[ENDPOINT_MARKER_SIZE * 0.5, markerHeight, 6]} />
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
}: {
  nodes: NodeData[];
  opacity?: number;
  showTube?: boolean;
  showLabels?: boolean;
}) {
  const lastIndex = nodes.length - 1;

  return (
    <group>
      {nodes.map((node, i) => {
        if (i === 0) {
          return (
            <EndpointMarker
              key={i}
              label="s"
              position={node.position}
              targetPosition={nodes[1]?.position}
              opacity={opacity}
              showCone={!showTube}
              showLabel={showLabels}
            />
          );
        }

        if (i === lastIndex) {
          return (
            <EndpointMarker
              key={i}
              label="e"
              position={node.position}
              targetPosition={nodes[i - 1]?.position}
              opacity={opacity}
              showCone={!showTube}
              showLabel={showLabels}
            />
          );
        }

        if (showTube) return null;

        return (
          <mesh key={i} position={node.position}>
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
      })}
    </group>
  );
}

function AneurysmOverlay({
  geometry,
  transform,
  showVessel,
}: {
  geometry: AneurysmGeometry;
  transform: CoordinateTransform;
  showVessel: boolean;
}) {
  const center = transformPoint(geometry.center, transform);
  const radius =
    (geometry.size.maxRadiusFromCenter ??
      (geometry.size.equivalentDiameterApprox ? geometry.size.equivalentDiameterApprox / 2 : 2)) *
    transform.scale;

  const sacMeshGeometry = useMemo(
    () => makeSurfaceGeometry(geometry.mesh, transform),
    [geometry.mesh, transform],
  );

  const vesselMeshGeometry = useMemo(
    () => makeSurfaceGeometry(geometry.vessel?.mesh, transform),
    [geometry.vessel?.mesh, transform],
  );

  const neck = geometry.neck ? transformPoint(geometry.neck, transform) : null;

  return (
    <group>
      {showVessel && vesselMeshGeometry && (
        <mesh geometry={vesselMeshGeometry}>
          <meshStandardMaterial
            color="#8ca3ad"
            emissive="#32424a"
            emissiveIntensity={0.08}
            transparent
            opacity={0.18}
            roughness={0.62}
            metalness={0.03}
            side={THREE.DoubleSide}
            depthWrite={false}
          />
        </mesh>
      )}
      {sacMeshGeometry ? (
        <mesh geometry={sacMeshGeometry}>
          <meshStandardMaterial
            color="#e84a5f"
            emissive="#e84a5f"
            emissiveIntensity={0.18}
            transparent
            opacity={0.38}
            roughness={0.42}
            metalness={0.05}
            side={THREE.DoubleSide}
            depthWrite={false}
          />
        </mesh>
      ) : (
        <mesh position={center}>
          <sphereGeometry args={[Math.max(radius, 1.5), 32, 18]} />
          <meshStandardMaterial
            color="#e84a5f"
            emissive="#e84a5f"
            emissiveIntensity={0.2}
            transparent
            opacity={0.22}
            roughness={0.35}
            depthWrite={false}
          />
        </mesh>
      )}
      {neck && (
        <group position={neck}>
          <mesh>
            <sphereGeometry args={[3, 18, 18]} />
            <meshStandardMaterial
              color="#ffffff"
              emissive="#ffffff"
              emissiveIntensity={0.45}
              roughness={0.3}
            />
          </mesh>
        </group>
      )}
    </group>
  );
}

// ── Main exported viewer ────────────────────────────────────────────────────

export default function GlyphView3D() {
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const cameraRef = useRef<THREE.Camera | null>(null);
  const dragStartRef = useRef<{ x: number; y: number } | null>(null);
  const [datasetIdx, setDatasetIdx] = useState(0);
  const [data, setData] = useState<GlyphDataset | null>(null);
  const [aneurysmGeometry, setAneurysmGeometry] = useState<AneurysmGeometry | null>(null);
  const [channels, setChannels] = useState<string[]>([]);
  const [enabledChannels, setEnabledChannels] = useState<Set<number>>(new Set());
  const [sampleCount, setSampleCount] = useState(DEFAULT_SAMPLE_COUNT);
  const [gamma, setGamma] = useState(20);
  const [showTube, setShowTube] = useState(true);
  const [showLabels, setShowLabels] = useState(true);
  const [showVessel, setShowVessel] = useState(true);
  const [selectMode, setSelectMode] = useState(false);
  const [hiddenObjectIds, setHiddenObjectIds] = useState<Set<number>>(new Set());
  const [selectedObjectIds, setSelectedObjectIds] = useState<Set<number>>(new Set());
  const [selectionRect, setSelectionRect] = useState<SelectionRect | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const dataset = AVAILABLE_DATASETS[datasetIdx];

  // Load the dataset JSON whenever the selected dataset changes.
  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(dataset.path);
        if (!res.ok) throw new Error(`Dataset not found (${res.status})`);
        const json: GlyphDataset = await res.json();
        let geometryJson: AneurysmGeometry | null = null;
        if (dataset.geometryPath) {
          const geometryRes = await fetch(dataset.geometryPath);
          if (!geometryRes.ok) throw new Error(`Geometry not found (${geometryRes.status})`);
          geometryJson = await geometryRes.json();
        }
        if (cancelled) return;

        setData(json);
        setAneurysmGeometry(geometryJson);
        setChannels(json.channels);
        setEnabledChannels(new Set(json.channels.map((_, i) => i)));
        setSampleCount(Math.min(dataset.defaultSampleCount ?? DEFAULT_SAMPLE_COUNT, json.objects.length));
        setShowVessel(true);
        setSelectMode(false);
        setHiddenObjectIds(new Set());
        setSelectedObjectIds(new Set());
      } catch (err) {
        if (!cancelled) {
          setAneurysmGeometry(null);
          setError(err instanceof Error ? err.message : String(err));
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    load();
    return () => { cancelled = true; };
  }, [dataset]);

  const sampledObjects = useMemo(
    () => sampleObjects(data?.objects ?? [], sampleCount, RANDOM_SEED),
    [data, sampleCount],
  );

  const coordinateTransform = useMemo(
    () => getCoordinateTransform(sampledObjects),
    [sampledObjects],
  );

  const sampledNodes = useMemo(
    () => objectsToNodeGroups(sampledObjects, channels, coordinateTransform),
    [coordinateTransform, sampledObjects, channels],
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

  const hasVesselMesh = Boolean(aneurysmGeometry?.vessel?.mesh?.vertices?.length);
  const supportsObjectSelection = (data?.objects.length ?? 0) > 1;

  const toggleChannel = useCallback((idx: number) => {
    setEnabledChannels((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx);
      else next.add(idx);
      return next;
    });
  }, []);

  const updateSampleCount = useCallback((value: number) => {
    if (!Number.isFinite(value)) return;
    const maxCount = data?.objects.length ?? DEFAULT_SAMPLE_COUNT;
    setSampleCount(Math.max(1, Math.min(maxCount, Math.floor(value))));
  }, [data]);

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

  const selectItemsInRect = useCallback((rect: SelectionRect, additive: boolean) => {
    const camera = cameraRef.current;
    const viewport = viewportRef.current;
    if (!camera || !viewport || rect.width < 4 || rect.height < 4) return;

    camera.updateMatrixWorld();
    const point = new THREE.Vector3();
    const nextSelected = additive ? new Set(selectedObjectIds) : new Set<number>();

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
  }, [selectedObjectIds, visibleItems]);

  const startSelection = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) return;
    const viewport = viewportRef.current;
    if (!viewport) return;

    const bounds = viewport.getBoundingClientRect();
    const x = event.clientX - bounds.left;
    const y = event.clientY - bounds.top;
    dragStartRef.current = { x, y };
    setSelectionRect({ x, y, width: 0, height: 0 });
    event.currentTarget.setPointerCapture(event.pointerId);
  }, []);

  const moveSelection = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    if (!dragStartRef.current) return;
    const viewport = viewportRef.current;
    if (!viewport) return;

    const bounds = viewport.getBoundingClientRect();
    updateDragRect(event.clientX - bounds.left, event.clientY - bounds.top);
  }, [updateDragRect]);

  const finishSelection = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
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
  }, [selectItemsInRect]);

  // ── Render ──

  return (
    <div style={{ width: "100vw", height: "100vh", display: "flex", flexDirection: "column", background: "#0a1929" }}>
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

        {supportsObjectSelection && (
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

            <div style={{ width: 1, height: 20, background: "rgba(255,255,255,0.15)" }} />
          </>
        )}

        {/* Channel toggles */}
        {channels.map((name, i) => (
          <label
            key={name}
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
            {name}
          </label>
        ))}

        <div style={{ width: 1, height: 20, background: "rgba(255,255,255,0.15)" }} />

        <label style={{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer" }}>
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
        </label>

        {hasVesselMesh && (
          <label style={{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer" }}>
            <input
              type="checkbox"
              checked={showVessel}
              onChange={(e) => setShowVessel(e.target.checked)}
              style={{ accentColor: "#8ca3ad" }}
            />
            Vessel
          </label>
        )}

        {supportsObjectSelection && (
          <>
            <div style={{ width: 1, height: 20, background: "rgba(255,255,255,0.15)" }} />

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

        <div style={{ width: 1, height: 20, background: "rgba(255,255,255,0.15)" }} />

        {/* Gamma control — remaps each channel value as v^gamma */}
        <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
          Gamma
          <input
            type="range"
            min={1}
            max={50}
            step={1}
            value={gamma}
            onChange={(e) => setGamma(Number(e.target.value))}
            style={{ width: 120, accentColor: "#45b7d1" }}
          />
          <span style={{ width: 28, textAlign: "right", fontVariantNumeric: "tabular-nums" }}>
            {gamma.toFixed(1)}
          </span>
        </label>
      </div>

      {/* ── 3D viewport ── */}
      <div ref={viewportRef} style={{ flex: 1, position: "relative" }}>
        {loading && (
          <div style={overlayStyle}>
            <div style={{ fontSize: 14, color: "rgba(255,255,255,0.7)" }}>Loading 3D structure...</div>
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
                />
                <ParticleSpheres
                  nodes={nodes}
                  opacity={opacity}
                  showTube={showTube}
                  showLabels={showLabels}
                />
              </group>
            );
          })}
          {aneurysmGeometry && coordinateTransform && (
            <AneurysmOverlay
              geometry={aneurysmGeometry}
              transform={coordinateTransform}
              showVessel={showVessel}
            />
          )}
          <OrbitControls
            enableZoom={!supportsObjectSelection || !selectMode}
            enablePan={!supportsObjectSelection || !selectMode}
            enableRotate={!supportsObjectSelection || !selectMode}
          />
        </Canvas>
        {supportsObjectSelection && selectMode && (
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
