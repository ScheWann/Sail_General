import { useMemo } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Html, Line } from "@react-three/drei";
import * as THREE from "three";

// Hardcoded 5 bead positions (simple S-shaped curve)
const DEMO_BEADS: [number, number, number][] = [
  [-60, 0, 0],
  [-30, 15, 10],
  [0, 0, 0],
  [30, -15, -10],
  [60, 0, 0],
];

// Simulated values for 3 tracks (normalized 0-1)
const DEMO_TRACK_VALUES = [
  [0.8, 0.4, 0.3],  // bead 0
  [0.5, 0.7, 0.4],  // bead 1
  [0.3, 0.5, 0.8],  // bead 2
  [0.6, 0.3, 0.5],  // bead 3
  [0.4, 0.6, 0.4],  // bead 4
];

const TRACK_NAMES = ["Data 1", "Data 2", "Data 3"];
const TRACK_COLORS = ["#ff6b6b", "#bf812d", "#45b7d1"];

const BASE_RADIUS = 4;
const MAX_EXTENSION = 15;

// Compute local coordinate system and vertices for each bead
function computeBeadGeometry(beadPositions: [number, number, number][], trackValues: number[][]) {
  const numTracks = trackValues[0].length;
  
  return beadPositions.map((pos, beadIdx) => {
    const center = new THREE.Vector3(...pos);
    
    // Compute tangent
    let tangent = new THREE.Vector3(1, 0, 0);
    if (beadIdx > 0 && beadIdx < beadPositions.length - 1) {
      const prev = new THREE.Vector3(...beadPositions[beadIdx - 1]);
      const next = new THREE.Vector3(...beadPositions[beadIdx + 1]);
      tangent = next.clone().sub(prev).normalize();
    } else if (beadIdx === 0) {
      const next = new THREE.Vector3(...beadPositions[1]);
      tangent = next.clone().sub(center).normalize();
    } else {
      const prev = new THREE.Vector3(...beadPositions[beadIdx - 1]);
      tangent = center.clone().sub(prev).normalize();
    }
    
    // Compute right and up vectors
    const globalUp = new THREE.Vector3(0, 0, 1);
    const right = new THREE.Vector3().crossVectors(tangent, globalUp).normalize();
    const up = new THREE.Vector3().crossVectors(right, tangent).normalize();
    
    // Compute baseline and actual vertices
    const baselineVertices: THREE.Vector3[] = [];
    const actualVertices: THREE.Vector3[] = [];
    
    for (let ti = 0; ti < numTracks; ti++) {
      const angle = (ti / numTracks) * Math.PI * 2;
      const trackVal = trackValues[beadIdx][ti];
      
      // Baseline vertex (fixed radius)
      const bx = Math.cos(angle) * BASE_RADIUS;
      const by = Math.sin(angle) * BASE_RADIUS;
      baselineVertices.push(
        center.clone()
          .add(right.clone().multiplyScalar(bx))
          .add(up.clone().multiplyScalar(by))
      );
      
      // Actual vertex (extended based on track value)
      const actualRadius = BASE_RADIUS + trackVal * MAX_EXTENSION;
      const ax = Math.cos(angle) * actualRadius;
      const ay = Math.sin(angle) * actualRadius;
      actualVertices.push(
        center.clone()
          .add(right.clone().multiplyScalar(ax))
          .add(up.clone().multiplyScalar(ay))
      );
    }
    
    return { center, tangent, right, up, baselineVertices, actualVertices };
  });
}

// Bead sphere component
function BeadSphere({ position }: { position: [number, number, number] }) {
  return (
    <group position={position}>
      <mesh>
        <sphereGeometry args={[3, 32, 32]} />
        <meshStandardMaterial color="#888" transparent opacity={0.6} />
      </mesh>
    </group>
  );
}

// Backbone curve
function BackboneCurve({ positions }: { positions: [number, number, number][] }) {
  const points = useMemo(() => {
    const pts = positions.map(p => new THREE.Vector3(...p));
    const curve = new THREE.CatmullRomCurve3(pts, false, "catmullrom", 0.3);
    return curve.getPoints(100);
  }, [positions]);
  
  return (
    <Line
      points={points}
      color="#ffffff"
      lineWidth={4}
      dashed={false}
    />
  );
}

// Baseline polygons (reference circle for each bead)
function BaselinePolygons({ beadGeometry }: { beadGeometry: ReturnType<typeof computeBeadGeometry> }) {
  const lines = useMemo(() => {
    const result: THREE.Vector3[][] = [];
    beadGeometry.forEach(({ baselineVertices }) => {
      const pts = [...baselineVertices, baselineVertices[0]];
      result.push(pts);
    });
    return result;
  }, [beadGeometry]);
  
  return (
    <>
      {lines.map((pts, i) => (
        <Line key={i} points={pts} color="#ffffff" lineWidth={2} dashed dashSize={1} gapSize={0.5} />
      ))}
    </>
  );
}

// Single track strip (quadrilaterals connecting adjacent bead vertices, precisely aligned at beads)
function TrackStrip({ 
  beadGeometry, 
  trackIndex, 
  color 
}: { 
  beadGeometry: ReturnType<typeof computeBeadGeometry>; 
  trackIndex: number; 
  color: string;
}) {
  const geometry = useMemo(() => {
    const vertices: number[] = [];
    const indices: number[] = [];
    let vi = 0;
    
    for (let bi = 0; bi < beadGeometry.length - 1; bi++) {
      const currBase = beadGeometry[bi].baselineVertices[trackIndex];
      const currActual = beadGeometry[bi].actualVertices[trackIndex];
      const nextBase = beadGeometry[bi + 1].baselineVertices[trackIndex];
      const nextActual = beadGeometry[bi + 1].actualVertices[trackIndex];
      
      // Quadrilateral: currBase → nextBase → nextActual → currActual (adjacent segments share curr/next vertices)
      vertices.push(
        currBase.x, currBase.y, currBase.z,
        nextBase.x, nextBase.y, nextBase.z,
        nextActual.x, nextActual.y, nextActual.z,
        currActual.x, currActual.y, currActual.z
      );
      indices.push(vi, vi + 1, vi + 2, vi, vi + 2, vi + 3);
      vi += 4;
    }
    
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
    geo.setIndex(indices);
    geo.computeVertexNormals();
    return geo;
  }, [beadGeometry, trackIndex]);
  
  return (
    <mesh geometry={geometry}>
      <meshBasicMaterial color={color} transparent opacity={0.7} side={THREE.DoubleSide} />
    </mesh>
  );
}

// Extension lines from baseline to actual (showing track value extension)
function ExtensionLines({ beadGeometry, showLabels = true }: { 
  beadGeometry: ReturnType<typeof computeBeadGeometry>; 
  showLabels?: boolean;
}) {
  return (
    <>
      {beadGeometry.map((bead, bi) => (
        <group key={bi}>
          {bead.baselineVertices.map((baseV, ti) => {
            const actualV = bead.actualVertices[ti];
            return (
              <group key={ti}>
                <Line
                  points={[baseV, actualV]}
                  color={TRACK_COLORS[ti]}
                  lineWidth={2}
                />
                {showLabels && bi === 2 && (
                  <Html position={[actualV.x, actualV.y, actualV.z]} center>
                    <div style={{
                      background: TRACK_COLORS[ti],
                      color: "#fff",
                      padding: "2px 6px",
                      borderRadius: 4,
                      fontSize: 10,
                      fontWeight: "bold",
                      whiteSpace: "nowrap",
                    }}>
                      {TRACK_NAMES[ti]}
                    </div>
                  </Html>
                )}
              </group>
            );
          })}
        </group>
      ))}
    </>
  );
}

// Actual polygon outlines (actual value polygon for each bead)
function ActualPolygons({ beadGeometry }: { beadGeometry: ReturnType<typeof computeBeadGeometry> }) {
  return (
    <>
      {beadGeometry.map(({ actualVertices }, bi) => {
        const pts = [...actualVertices, actualVertices[0]];
        return <Line key={bi} points={pts} color="#d4a574" lineWidth={2} />;
      })}
    </>
  );
}

// Main demo component
export default function TrackMappingDemo() {
  const beadGeometry = useMemo(() => computeBeadGeometry(DEMO_BEADS, DEMO_TRACK_VALUES), []);
  
  return (
    <div style={{ width: "100vw", height: "100vh", background: "#0a1929" }}>
      {/* Legend */}
      <div style={{
        position: "absolute",
        top: 16,
        left: 16,
        background: "rgba(0,0,0,0.7)",
        padding: "12px 16px",
        borderRadius: 8,
        color: "#fff",
        fontSize: 13,
        zIndex: 100,
        fontFamily: "system-ui, sans-serif",
      }}>
        <div style={{ fontWeight: "bold", marginBottom: 8, fontSize: 15 }}>Track Mapping Demo</div>
        <div style={{ marginBottom: 8 }}>5 Beads × 3 Tracks</div>
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          {TRACK_NAMES.map((name, i) => (
            <div key={name} style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <div style={{ width: 16, height: 16, background: TRACK_COLORS[i], borderRadius: 2 }} />
              <span>{name}</span>
            </div>
          ))}
        </div>
      </div>
      
      {/* Formula description */}
      <div style={{
        position: "absolute",
        bottom: 16,
        left: 16,
        background: "rgba(0,0,0,0.7)",
        padding: "12px 16px",
        borderRadius: 8,
        color: "#fff",
        fontSize: 12,
        zIndex: 100,
        fontFamily: "monospace",
      }}>
        <div style={{ marginBottom: 4 }}>θ = (trackIndex / numTracks) × 2π</div>
        <div style={{ marginBottom: 4 }}>r = baseRadius + trackValue × maxExtension</div>
        <div>vertex = center + R×cos(θ)×r + U×sin(θ)×r</div>
      </div>
      
      <Canvas camera={{ position: [0, 80, 120], fov: 50 }}>
        <ambientLight intensity={0.6} />
        <directionalLight position={[50, 50, 50]} intensity={0.8} />
        
        {/* Backbone curve */}
        <BackboneCurve positions={DEMO_BEADS} />
        
        {/* Bead spheres */}
        {DEMO_BEADS.map((pos, i) => (
          <BeadSphere key={i} position={pos} />
        ))}
        
        {/* Baseline polygons */}
        <BaselinePolygons beadGeometry={beadGeometry} />
        
        {/* Actual polygon outlines */}
        <ActualPolygons beadGeometry={beadGeometry} />
        
        {/* Extension lines */}
        <ExtensionLines beadGeometry={beadGeometry} />
        
        {/* Track strips */}
        {TRACK_COLORS.map((color, ti) => (
          <TrackStrip key={ti} beadGeometry={beadGeometry} trackIndex={ti} color={color} />
        ))}
        
        <OrbitControls enableZoom enablePan enableRotate />
      </Canvas>
    </div>
  );
}
