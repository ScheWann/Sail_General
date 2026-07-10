import { useEffect, useState, useMemo, useCallback } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";

// ── Data types ──────────────────────────────────────────────────────────────

interface GlyphPoint {
  x: number;
  y: number;
  z: number;
  values: number[];
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

// Internal per-point representation consumed by the 3D pipeline.
interface NodeData {
  position: [number, number, number];
  values: number[];
}

// ── Configuration ───────────────────────────────────────────────────────────

interface DatasetConfig {
  name: string;
  path: string;
}

const AVAILABLE_DATASETS: DatasetConfig[] = [
  { name: "Turbulence tracers", path: "/Data/turb_glyph.json" },
];

const CHANNEL_COLORS = [
  "#ff6b6b", "#bf812d", "#45b7d1",
  "#f9ca24", "#6c5ce7", "#00d2d3",
  "#ff9ff3", "#54a0ff", "#a29bfe",
];

function getChannelColor(index: number): string {
  return CHANNEL_COLORS[index % CHANNEL_COLORS.length];
}

// World-space extent each object's point cloud is scaled to, so the camera
// framing and glyph sizes work regardless of the source data's units.
const TARGET_EXTENT = 200;

// ── 3D glyph pipeline ─────────────────────────────────────────────────────────

function GlyphPipeline({
  nodes,
  enabledChannelIndices,
  gamma = 1,
  opacity = 1,
}: {
  nodes: NodeData[];
  enabledChannelIndices?: number[];
  gamma?: number;
  opacity?: number;
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

  // Backbone tube
  const centerPts = nodes.map(
    (b) => new THREE.Vector3(b.position[0] * scale, b.position[1] * scale, b.position[2] * scale),
  );
  const backboneCurve = new THREE.CatmullRomCurve3(centerPts, false, "catmullrom", 0.3);
  const tubeGeo = new THREE.TubeGeometry(backboneCurve, nodes.length * 20, 0.5, 8, false);

  // Radar polygon outline (white baseline)
  const rpVerts: number[] = [];
  const rpColors: number[] = [];
  for (let bi = 0; bi < nodes.length; bi++) {
    const bl = nodeVertices[bi].baseline;
    for (let ti = 0; ti < numChannels; ti++) {
      const nti = (ti + 1) % numChannels;
      rpVerts.push(bl[ti].x, bl[ti].y, bl[ti].z, bl[nti].x, bl[nti].y, bl[nti].z);
      rpColors.push(1, 1, 1, 1, 1, 1);
    }
  }
  const rpGeo = new THREE.BufferGeometry();
  rpGeo.setAttribute("position", new THREE.Float32BufferAttribute(rpVerts, 3));
  rpGeo.setAttribute("color", new THREE.Float32BufferAttribute(rpColors, 3));

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
      <mesh geometry={tubeGeo}>
        <meshStandardMaterial color="#ffffff" transparent opacity={0.9 * opacity} metalness={0.3} roughness={0.4} />
      </mesh>
      <lineSegments geometry={rpGeo}>
        <lineBasicMaterial vertexColors linewidth={3} transparent opacity={0.9 * opacity} />
      </lineSegments>
    </group>
  );
}

// ── Main exported viewer ────────────────────────────────────────────────────

export default function GlyphView3D() {
  const [datasetIdx, setDatasetIdx] = useState(0);
  const [objectId, setObjectId] = useState(0);
  const [data, setData] = useState<GlyphDataset | null>(null);
  const [channels, setChannels] = useState<string[]>([]);
  const [enabledChannels, setEnabledChannels] = useState<Set<number>>(new Set());
  const [objectIds, setObjectIds] = useState<number[]>([]);
  const [gamma, setGamma] = useState(1);
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
        if (cancelled) return;

        const ids = json.objects.map((o) => o.objectId).sort((a, b) => a - b);
        setData(json);
        setChannels(json.channels);
        setEnabledChannels(new Set(json.channels.map((_, i) => i)));
        setObjectIds(ids);
        setObjectId((prev) => (ids.includes(prev) ? prev : ids[0] ?? 0));
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    load();
    return () => { cancelled = true; };
  }, [dataset]);

  const currentObject = useMemo(
    () => data?.objects.find((o) => o.objectId === objectId) ?? data?.objects[0] ?? null,
    [data, objectId],
  );

  // Center + scale the selected object into a canonical box so camera framing
  // and glyph sizes are independent of the source data's units.
  const nodes = useMemo<NodeData[]>(() => {
    if (!data || !currentObject) return [];
    const pts = currentObject.points;
    if (pts.length === 0) return [];

    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    for (const p of pts) {
      const c = [p.x, p.y, p.z];
      for (let k = 0; k < 3; k++) {
        if (Number.isFinite(c[k])) {
          if (c[k] < min[k]) min[k] = c[k];
          if (c[k] > max[k]) max[k] = c[k];
        }
      }
    }
    const center = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2];
    const extent = Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2]) || 1;
    const s = TARGET_EXTENT / extent;

    return pts.map((p) => ({
      position: [
        ((Number.isFinite(p.x) ? p.x : center[0]) - center[0]) * s,
        ((Number.isFinite(p.y) ? p.y : center[1]) - center[1]) * s,
        ((Number.isFinite(p.z) ? p.z : center[2]) - center[2]) * s,
      ] as [number, number, number],
      values: data.channels.map((_, i) => {
        const v = p.values?.[i];
        return Number.isFinite(v) ? v : 0;
      }),
    }));
  }, [data, currentObject]);

  const enabledChannelIndices = useMemo(
    () => [...enabledChannels].sort((a, b) => a - b),
    [enabledChannels],
  );

  const toggleChannel = useCallback((idx: number) => {
    setEnabledChannels((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx);
      else next.add(idx);
      return next;
    });
  }, []);

  const title = (data?.meta?.title as string | undefined) ?? dataset.name;

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

        {/* Object selector */}
        <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
          Object
          <select
            value={objectId}
            onChange={(e) => setObjectId(Number(e.target.value))}
            style={selectStyle}
          >
            {objectIds.map((id) => {
              const obj = data?.objects.find((o) => o.objectId === id);
              return (
                <option key={id} value={id}>
                  {obj?.label ?? id}
                </option>
              );
            })}
          </select>
        </label>

        <div style={{ width: 1, height: 20, background: "rgba(255,255,255,0.15)" }} />

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

        {/* Gamma control — remaps each channel value as v^gamma */}
        <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
          Gamma
          <input
            type="range"
            min={1}
            max={20}
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
      <div style={{ flex: 1, position: "relative" }}>
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
          style={{ background: "#0a1929", width: "100%", height: "100%" }}
        >
          <ambientLight intensity={0.6} />
          <directionalLight position={[100, 100, 100]} intensity={0.8} />
          {nodes.length > 1 && (
            <GlyphPipeline
              nodes={nodes}
              enabledChannelIndices={enabledChannelIndices}
              gamma={gamma}
            />
          )}
          <OrbitControls enableZoom enablePan enableRotate />
        </Canvas>
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

const overlayStyle: React.CSSProperties = {
  position: "absolute",
  inset: 0,
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  zIndex: 10,
  background: "rgba(10,25,41,0.8)",
};
