import { useEffect, useState, useMemo, useCallback } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";

// ── Data types ──────────────────────────────────────────────────────────────

interface SampleData {
  pid: number;
  cell_line: string;
  chrId: string;
  sampleId: number;
  start_value: number;
  end_value: number;
  x: number;
  y: number;
  z: number;
}

interface BeadData {
  position: [number, number, number];
  trackValues: number[];
  sampleData: SampleData;
}

interface TracksJson {
  region: { chromosome: string; start: number; end: number; bin_size: number };
  tracks: Record<string, { raw: number[]; normalized: number[] }>;
}

// ── Configuration ───────────────────────────────────────────────────────────

interface DatasetConfig {
  cellLine: string;
  chrId: string;
  start: number;
  end: number;
  positionPrefix?: string;
}

const AVAILABLE_DATASETS: DatasetConfig[] = [
  { cellLine: "Calu3", chrId: "chr8", start: 127200000, end: 127750000 },
  { cellLine: "GM12878", chrId: "chr8", start: 127200000, end: 127750000 },
  { cellLine: "Monocytes", chrId: "chr8", start: 127200000, end: 127750000, positionPrefix: "monocytes" },
];

const TRACK_COLORS = [
  "#ff6b6b", "#bf812d", "#45b7d1",
  "#f9ca24", "#6c5ce7", "#00d2d3",
  "#ff9ff3", "#54a0ff", "#a29bfe",
];

function getTrackColor(index: number): string {
  return TRACK_COLORS[index % TRACK_COLORS.length];
}

// ── Data helpers ────────────────────────────────────────────────────────────

function parsePositionCsv(text: string): SampleData[] {
  const lines = text.trim().split("\n");
  const result: SampleData[] = [];
  for (let i = 1; i < lines.length; i++) {
    const v = lines[i].split(",");
    if (v.length < 9) continue;
    result.push({
      pid: parseInt(v[0], 10),
      cell_line: v[1],
      chrId: v[2],
      sampleId: parseInt(v[3], 10),
      start_value: parseInt(v[4], 10),
      end_value: parseInt(v[5], 10),
      x: parseFloat(v[6]) || 0,
      y: parseFloat(v[7]) || 0,
      z: parseFloat(v[8]) || 0,
    });
  }
  return result;
}

function matchBeadsToTracks(
  samples: SampleData[],
  tracksJson: TracksJson,
  trackNames: string[],
  normalize: boolean,
): BeadData[] {
  return samples.map((sample, idx) => {
    const trackValues = trackNames.map((name) => {
      const track = tracksJson.tracks[name];
      if (!track) return 0;
      const arr = normalize ? track.normalized : track.raw;
      if (!arr || idx >= arr.length) return 0;
      const v = arr[idx];
      return Number.isFinite(v) ? v : 0;
    });
    const x = Number.isFinite(sample.x) ? sample.x : 0;
    const y = Number.isFinite(sample.y) ? sample.y : 0;
    const z = Number.isFinite(sample.z) ? sample.z : 0;
    return { position: [x, y, z], trackValues, sampleData: { ...sample, x, y, z } };
  });
}

// ── 3D track pipeline ─────────

function ChromosomePipeline({
  beads,
  enabledTrackIndices,
  trackNames: _trackNames,
  opacity = 1,
}: {
  beads: BeadData[];
  enabledTrackIndices?: number[];
  trackNames: string[];
  opacity?: number;
}) {
  const totalTracks = beads[0]?.trackValues.length || 1;
  const activeTrackIndices =
    enabledTrackIndices ?? Array.from({ length: totalTracks }, (_, i) => i);
  const numTracks = activeTrackIndices.length;
  const beadRadius = 0.8;
  const maxRadarRadius = 20;
  const scale = 1;

  if (beads.length < 2 || numTracks === 0) return null;

  const hasValid = beads.every(
    (b) =>
      Number.isFinite(b.position[0]) &&
      Number.isFinite(b.position[1]) &&
      Number.isFinite(b.position[2]),
  );
  if (!hasValid) return null;

  // Per-bead baseline + actual radar vertices
  const beadBaselineVertices = beads.map((bead, beadIndex) => {
    const pos = [
      bead.position[0] * scale,
      bead.position[1] * scale,
      bead.position[2] * scale,
    ];

    let tangent = new THREE.Vector3(0, 0, 1);
    if (beadIndex > 0 && beadIndex < beads.length - 1) {
      const prev = beads[beadIndex - 1].position;
      const next = beads[beadIndex + 1].position;
      tangent = new THREE.Vector3(
        next[0] - prev[0],
        next[1] - prev[1],
        next[2] - prev[2],
      );
      if (tangent.lengthSq() > 1e-10) tangent.normalize();
    } else if (beadIndex === 0 && beads.length > 1) {
      const next = beads[1].position;
      tangent = new THREE.Vector3(next[0] - pos[0], next[1] - pos[1], next[2] - pos[2]);
      if (tangent.lengthSq() > 1e-10) tangent.normalize();
    } else if (beadIndex === beads.length - 1 && beads.length > 1) {
      const prev = beads[beadIndex - 1].position;
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
    const bRadius = beadRadius * 1.05;

    for (let i = 0; i < numTracks; i++) {
      const angle = (i / numTracks) * Math.PI * 2;
      const tIdx = activeTrackIndices[i];
      const rawVal = bead.trackValues?.[tIdx];
      const tv = Number.isFinite(rawVal) ? rawVal : 0;

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
  const centerPts = beads.map(
    (b) => new THREE.Vector3(b.position[0] * scale, b.position[1] * scale, b.position[2] * scale),
  );
  const backboneCurve = new THREE.CatmullRomCurve3(centerPts, false, "catmullrom", 0.3);
  const tubeGeo = new THREE.TubeGeometry(backboneCurve, beads.length * 20, 0.5, 8, false);

  // Radar polygon outline (white baseline)
  const rpVerts: number[] = [];
  const rpColors: number[] = [];
  for (let bi = 0; bi < beads.length; bi++) {
    const bl = beadBaselineVertices[bi].baseline;
    for (let ti = 0; ti < numTracks; ti++) {
      const nti = (ti + 1) % numTracks;
      rpVerts.push(bl[ti].x, bl[ti].y, bl[ti].z, bl[nti].x, bl[nti].y, bl[nti].z);
      rpColors.push(1, 1, 1, 1, 1, 1);
    }
  }
  const rpGeo = new THREE.BufferGeometry();
  rpGeo.setAttribute("position", new THREE.Float32BufferAttribute(rpVerts, 3));
  rpGeo.setAttribute("color", new THREE.Float32BufferAttribute(rpColors, 3));

  // Triangular track fill between beads
  const triVerts: number[] = [];
  const triColors: number[] = [];
  const triIdx: number[] = [];
  let vi = 0;
  const subDiv = 8;

  for (let bi = 0; bi < beads.length - 1; bi++) {
    for (let ti = 0; ti < numTracks; ti++) {
      const cb = beadBaselineVertices[bi].baseline[ti];
      const nb = beadBaselineVertices[bi + 1].baseline[ti];
      const ca = beadBaselineVertices[bi].actual[ti];
      const na = beadBaselineVertices[bi + 1].actual[ti];

      let prevB = cb, prevA = ca;
      if (bi > 0) {
        prevB = beadBaselineVertices[bi - 1].baseline[ti];
        prevA = beadBaselineVertices[bi - 1].actual[ti];
      }
      let nnB = nb, nnA = na;
      if (bi < beads.length - 2) {
        nnB = beadBaselineVertices[bi + 2].baseline[ti];
        nnA = beadBaselineVertices[bi + 2].actual[ti];
      }

      const bCurve = new THREE.CatmullRomCurve3([prevB, cb, nb, nnB], false, "catmullrom", 0.3);
      const aCurve = new THREE.CatmullRomCurve3([prevA, ca, na, nnA], false, "catmullrom", 0.3);

      const startT = bi > 0 ? 0.33 : 0;
      const endT = bi < beads.length - 2 ? 0.67 : 1.0;
      const totalPts = subDiv * 3;
      const bPts = bCurve.getPoints(totalPts);
      const aPts = aCurve.getPoints(totalPts);
      const si = Math.floor(startT * bPts.length);
      const ei = Math.floor(endT * bPts.length);
      const segB = bPts.slice(si, ei);
      const segA = aPts.slice(si, ei);

      const origIdx = activeTrackIndices[ti];
      const tc = new THREE.Color(getTrackColor(origIdx));

      for (let i = 0; i < segB.length - 1; i++) {
        triVerts.push(
          segB[i].x, segB[i].y, segB[i].z,
          segB[i + 1].x, segB[i + 1].y, segB[i + 1].z,
          segA[i].x, segA[i].y, segA[i].z,
          segA[i + 1].x, segA[i + 1].y, segA[i + 1].z,
        );
        for (let j = 0; j < 4; j++) triColors.push(tc.r, tc.g, tc.b);
        triIdx.push(vi, vi + 1, vi + 2, vi + 1, vi + 3, vi + 2);
        vi += 4;
      }
    }
  }

  const triGeo = new THREE.BufferGeometry();
  triGeo.setAttribute("position", new THREE.Float32BufferAttribute(triVerts, 3));
  triGeo.setAttribute("color", new THREE.Float32BufferAttribute(triColors, 3));
  triGeo.setIndex(triIdx);
  triGeo.computeVertexNormals();

  return (
    <group>
      <mesh geometry={triGeo}>
        <meshBasicMaterial vertexColors transparent opacity={0.8 * opacity} side={THREE.DoubleSide} />
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

export default function ChromosomeTrack3D() {
  const [datasetIdx, setDatasetIdx] = useState(0);
  const [sampleId, setSampleId] = useState(0);
  const [beads, setBeads] = useState<BeadData[]>([]);
  const [trackNames, setTrackNames] = useState<string[]>([]);
  const [enabledTracks, setEnabledTracks] = useState<Set<number>>(new Set());
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [availableSampleIds, setAvailableSampleIds] = useState<number[]>([]);

  const dataset = AVAILABLE_DATASETS[datasetIdx];

  // Load data whenever dataset / sample / normalize changes
  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const { cellLine, chrId, start, end, positionPrefix } = dataset;
        const posName = positionPrefix ?? cellLine;
        const posPath = `/Data/Example/${posName}_${chrId}_${start}_${end}_original_position.csv`;
        const trkPath = `/Data/Tracks/${cellLine}_${chrId}_${start}_${end}_tracks_data.json`;

        const [posRes, trkRes] = await Promise.all([fetch(posPath), fetch(trkPath)]);
        if (!posRes.ok) throw new Error(`Position CSV not found (${posRes.status})`);
        if (!trkRes.ok) throw new Error(`Tracks JSON not found (${trkRes.status})`);

        const posText = await posRes.text();
        const trkJson: TracksJson = await trkRes.json();

        const allSamples = parsePositionCsv(posText);
        const sampleIds = [...new Set(allSamples.map((s) => s.sampleId))].sort((a, b) => a - b);
        const names = Object.keys(trkJson.tracks);

        if (cancelled) return;
        setAvailableSampleIds(sampleIds);
        setTrackNames(names);
        setEnabledTracks(new Set(names.map((_, i) => i)));

        const targetSid = sampleIds.includes(sampleId) ? sampleId : sampleIds[0] ?? 0;
        if (targetSid !== sampleId) setSampleId(targetSid);

        const filtered = allSamples
          .filter((s) => s.sampleId === targetSid)
          .sort((a, b) => a.start_value - b.start_value);

        if (filtered.length === 0) throw new Error("No position data for selected sample");

        setBeads(matchBeadsToTracks(filtered, trkJson, names, true));
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    load();
    return () => { cancelled = true; };
  }, [dataset, sampleId]);

  const enabledTrackIndices = useMemo(
    () => [...enabledTracks].sort((a, b) => a - b),
    [enabledTracks],
  );

  const toggleTrack = useCallback((idx: number) => {
    setEnabledTracks((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx);
      else next.add(idx);
      return next;
    });
  }, []);

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
          Cell Line
          <select
            value={datasetIdx}
            onChange={(e) => setDatasetIdx(Number(e.target.value))}
            style={selectStyle}
          >
            {AVAILABLE_DATASETS.map((d, i) => (
              <option key={i} value={i}>
                {d.cellLine}
              </option>
            ))}
          </select>
        </label>

        {/* Sample selector */}
        <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
          Sample
          <select
            value={sampleId}
            onChange={(e) => setSampleId(Number(e.target.value))}
            style={selectStyle}
          >
            {availableSampleIds.slice(0, 5000).map((sid) => (
              <option key={sid} value={sid}>
                {sid}
              </option>
            ))}
          </select>
        </label>

        <div style={{ width: 1, height: 20, background: "rgba(255,255,255,0.15)" }} />

        {/* Track toggles */}
        {trackNames.map((name, i) => (
          <label
            key={name}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 4,
              cursor: "pointer",
              opacity: enabledTracks.has(i) ? 1 : 0.4,
            }}
          >
            <input
              type="checkbox"
              checked={enabledTracks.has(i)}
              onChange={() => toggleTrack(i)}
              style={{ accentColor: getTrackColor(i) }}
            />
            {name}
          </label>
        ))}
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
          {beads.length > 1 && (
            <ChromosomePipeline
              beads={beads}
              enabledTrackIndices={enabledTrackIndices}
              trackNames={trackNames}
            />
          )}
          <OrbitControls enableZoom enablePan enableRotate />
        </Canvas>
      </div>

      {/* ── Legend ── */}
      <div
        style={{
          position: "absolute",
          bottom: 16,
          left: 16,
          background: "rgba(0,0,0,0.55)",
          backdropFilter: "blur(8px)",
          borderRadius: 8,
          padding: "10px 14px",
          color: "#e0e0e0",
          fontSize: 12,
          fontFamily: "system-ui, sans-serif",
          lineHeight: 1.8,
        }}
      >
        <div style={{ fontWeight: 600, marginBottom: 2 }}>
          {dataset.cellLine} &middot; {dataset.chrId}:{dataset.start.toLocaleString()}-{dataset.end.toLocaleString()}
        </div>
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
