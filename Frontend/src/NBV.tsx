import { useCallback, useState, useEffect } from "react";
import "./NBV.css";

const EPS = 1e-12;
const STEP_DEG = 5;
const MIN_ANGLE_DIFF_DEG = 45;
const TOP_COUNT = 10;
const SPHERE_RADIUS_FACTOR = 2;
const TRACK_VALUE_SCALE = 20;

export interface NBVBead {
  position: [number, number, number];
  trackValues: number[];
}

export interface NBVTrack {
  id: string;
  name: string;
  color: string;
  active: boolean;
}

export interface NBVProps {
  beads: NBVBead[];
  tracks: NBVTrack[];
  /** Indices of tracks active in NBV window only (controls checkboxes and preview). */
  nbvActiveTrackIndices: number[];
  onApplyView?: (position: [number, number, number], target: [number, number, number]) => void;
  onNbvActiveTracksChange?: (indices: number[]) => void;
  /** When user runs NBV with a range, call this so parent can filter displayed beads to only that range. */
  onBeadRangeApply?: (start: number, end: number) => void;
  /** Toggle brush/select mode: user goes to main scene and clicks two beads to set range. */
  onToggleSelectMode?: () => void;
  /** Whether brush/select mode is active (main scene shows selectable beads). */
  isSelectMode?: boolean;
  /** Range set from scene selection (1-based); syncs to range inputs when set. */
  selectedRange?: [number, number] | null;
  /** Toggle 2D planar projection view (algorithm from 2D.tsx). */
  onToggle2D?: () => void;
  /** Whether 2D projection is shown in the preview area. */
  show2D?: boolean;
  onClose?: () => void;
  onMinimize?: () => void;
  minimized?: boolean;
  /** Called when user presses pointer on the bar (for drag-to-move). */
  onBarPointerDown?: (e: React.PointerEvent) => void;
}

type Vec3 = [number, number, number];

interface Viewpoint {
  id: string;
  position: Vec3;
}

// Center of sample = centroid of its beads; dataRadius = distance from center to farthest bead/node
function getCenterAndRadius(beads: NBVBead[]): { center: Vec3; dataRadius: number } {
  const n = beads.length;
  let cx = 0, cy = 0, cz = 0;
  for (const b of beads) {
    cx += b.position[0]; cy += b.position[1]; cz += b.position[2];
  }
  const center: Vec3 = [cx / n, cy / n, cz / n];
  let maxD = 0;
  for (const b of beads) {
    const d = Math.hypot(b.position[0] - center[0], b.position[1] - center[1], b.position[2] - center[2]);
    if (d > maxD) maxD = d;
  }
  return { center, dataRadius: Math.max(maxD, 1) };
}

// Viewpoints on sphere, step 5°, radius 1.5 * dataRadius
function generateViewpoints(center: Vec3, dataRadius: number): Viewpoint[] {
  const r = SPHERE_RADIUS_FACTOR * dataRadius;
  const views: Viewpoint[] = [];
  let idx = 0;
  for (let el = -90; el <= 90; el += STEP_DEG) {
    for (let az = 0; az < 360; az += STEP_DEG) {
      const elRad = (el * Math.PI) / 180;
      const azRad = (az * Math.PI) / 180;
      const cosEl = Math.cos(elRad);
      views.push({
        id: `v_${idx++}`,
        position: [
          center[0] + r * cosEl * Math.cos(azRad),
          center[1] + r * cosEl * Math.sin(azRad),
          center[2] + r * Math.sin(elRad),
        ],
      });
    }
  }
  return views;
}

// One quad = 4 corners; for segment i: p0, p1, p1+perp*v1, p0+perp*v0. Perp from tangent and track angle.
function getTrackQuads(beads: NBVBead[], trackIdx: number, numTracks: number): Vec3[][] {
  const quads: Vec3[][] = [];
  const worldUp: Vec3 = [0, 0, 1];
  const angle = (trackIdx / numTracks) * Math.PI * 2;

  for (let i = 0; i < beads.length - 1; i++) {
    const p0 = beads[i].position;
    const p1 = beads[i + 1].position;
    const v0 = beads[i].trackValues[trackIdx] ?? 0;
    const v1 = beads[i + 1].trackValues[trackIdx] ?? 0;

    const dx = p1[0] - p0[0], dy = p1[1] - p0[1], dz = p1[2] - p0[2];
    const lenSeg = Math.hypot(dx, dy, dz) || 1;
    const tx = dx / lenSeg, ty = dy / lenSeg, tz = dz / lenSeg;

    let rx = ty * worldUp[2] - tz * worldUp[1];
    let ry = tz * worldUp[0] - tx * worldUp[2];
    let rz = tx * worldUp[1] - ty * worldUp[0];
    const lenR = Math.hypot(rx, ry, rz) || 1;
    rx /= lenR; ry /= lenR; rz /= lenR;

    let ux = ry * tz - rz * ty, uy = rz * tx - rx * tz, uz = rx * ty - ry * tx;
    const lenU = Math.hypot(ux, uy, uz) || 1;
    ux /= lenU; uy /= lenU; uz /= lenU;

    const perpX = rx * Math.cos(angle) + ux * Math.sin(angle);
    const perpY = ry * Math.cos(angle) + uy * Math.sin(angle);
    const perpZ = rz * Math.cos(angle) + uz * Math.sin(angle);
    const s = TRACK_VALUE_SCALE;

    quads.push([
      [p0[0], p0[1], p0[2]],
      [p1[0], p1[1], p1[2]],
      [p1[0] + perpX * v1 * s, p1[1] + perpY * v1 * s, p1[2] + perpZ * v1 * s],
      [p0[0] + perpX * v0 * s, p0[1] + perpY * v0 * s, p0[2] + perpZ * v0 * s],
    ]);
  }
  return quads;
}

// Project quad onto view plane (orthographic), return 2D area
function projectedArea(quad: Vec3[], viewPos: Vec3, center: Vec3): number {
  const vx = viewPos[0] - center[0], vy = viewPos[1] - center[1], vz = viewPos[2] - center[2];
  const lenV = Math.hypot(vx, vy, vz) || 1;
  const vnx = vx / lenV, vny = vy / lenV, vnz = vz / lenV;

  const right: Vec3 = [1, 0, 0];
  let rx = right[0] - vnx * (right[0] * vnx + right[1] * vny + right[2] * vnz);
  let ry = right[1] - vny * (right[0] * vnx + right[1] * vny + right[2] * vnz);
  let rz = right[2] - vnz * (right[0] * vnx + right[1] * vny + right[2] * vnz);
  const lenR = Math.hypot(rx, ry, rz) || 1;
  rx /= lenR; ry /= lenR; rz /= lenR;

  const ux = vny * rz - vnz * ry, uy = vnz * rx - vnx * rz, uz = vnx * ry - vny * rx;

  const proj = (p: Vec3) => [
    (p[0] - center[0]) * rx + (p[1] - center[1]) * ry + (p[2] - center[2]) * rz,
    (p[0] - center[0]) * ux + (p[1] - center[1]) * uy + (p[2] - center[2]) * uz,
  ] as [number, number];
  const a = proj(quad[0]), b = proj(quad[1]), c = proj(quad[2]), d = proj(quad[3]);
  return Math.abs((a[0] * b[1] + b[0] * c[1] + c[0] * d[1] + d[0] * a[1]) - (a[1] * b[0] + b[1] * c[0] + c[1] * d[0] + d[1] * a[0])) * 0.5;
}

// visibility[viewId][trackId] = sum of projected areas for that track
function computeVisibility(views: Viewpoint[], tracks: NBVTrack[], beads: NBVBead[], center: Vec3): Record<string, Record<string, number>> {
  const vis: Record<string, Record<string, number>> = {};
  const active = tracks.filter((t) => t.active);
  const numTracks = tracks.length;

  for (const v of views) {
    vis[v.id] = {};
    for (const t of active) {
      const trackIdx = tracks.findIndex((x) => x.id === t.id);
      const quads = getTrackQuads(beads, trackIdx >= 0 ? trackIdx : 0, numTracks);
      let sum = 0;
      for (const q of quads) sum += projectedArea(q, v.position, center);
      vis[v.id][t.id] = Math.max(0, sum);
    }
  }
  return vis;
}

/**
 * Score views by visible track surface. For one track: pick view that maximizes that track's
 * visible area. For multiple tracks: maximize sum of per-track normalized visibility so
 * all tracks are shown as much as possible in the same ratio (balanced).
 * Returns viewId -> score (higher = more visible surface).
 */
function computeVisibleSurfaceScores(
  views: Viewpoint[],
  tracks: NBVTrack[],
  visibility: Record<string, Record<string, number>>,
): Record<string, number> {
  const active = tracks.filter((t) => t.active);
  if (active.length === 0) return {};

  // Per track: max visibility over all views (best possible for that track)
  const maxVis: Record<string, number> = {};
  for (const t of active) {
    let m = 0;
    for (const v of views) m = Math.max(m, visibility[v.id]?.[t.id] ?? 0);
    maxVis[t.id] = m;
  }

  // Per view: sum of (visibility / maxVis) so each track contributes 0..1; same ratio = balanced
  const scores: Record<string, number> = {};
  for (const v of views) {
    let s = 0;
    for (const t of active) {
      const vis = visibility[v.id]?.[t.id] ?? 0;
      const mx = maxVis[t.id];
      if (mx > EPS) s += vis / mx;
    }
    scores[v.id] = s;
  }
  return scores;
}

// Angular difference in degrees between two view directions (from center)
function angleDiffDeg(posA: Vec3, posB: Vec3, center: Vec3): number {
  const ax = posA[0] - center[0], ay = posA[1] - center[1], az = posA[2] - center[2];
  const bx = posB[0] - center[0], by = posB[1] - center[1], bz = posB[2] - center[2];
  const la = Math.hypot(ax, ay, az) || 1;
  const lb = Math.hypot(bx, by, bz) || 1;
  const dot = (ax * bx + ay * by + az * bz) / (la * lb);
  const rad = Math.acos(Math.max(-1, Math.min(1, dot)));
  return (rad * 180) / Math.PI;
}

// Top 10 views with at least 45° between any two (higher score = better; best view first)
function selectTop10(views: Viewpoint[], scores: Record<string, number>, center: Vec3): Viewpoint[] {
  const sorted = [...views].sort((a, b) => (scores[b.id] ?? 0) - (scores[a.id] ?? 0));
  const out: Viewpoint[] = [];
  for (const v of sorted) {
    let ok = true;
    for (const u of out) {
      if (angleDiffDeg(v.position, u.position, center) < MIN_ANGLE_DIFF_DEG) {
        ok = false;
        break;
      }
    }
    if (ok) out.push(v);
    if (out.length >= TOP_COUNT) break;
  }
  return out;
}

const DEFAULT_MAX_BEADS = 110;

export default function NBV({ beads, tracks, nbvActiveTrackIndices, onApplyView, onNbvActiveTracksChange, onBeadRangeApply, onToggleSelectMode, isSelectMode, selectedRange, onToggle2D, show2D, onClose, onMinimize, minimized, onBarPointerDown }: NBVProps) {
  const [busy, setBusy] = useState(false);
  const [topViews, setTopViews] = useState<Viewpoint[]>([]);
  const [center, setCenter] = useState<Vec3>([0, 0, 0]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const maxBeads = beads.length > 0 ? beads.length : DEFAULT_MAX_BEADS;
  const [rangeStartInput, setRangeStartInput] = useState("");
  const [rangeEndInput, setRangeEndInput] = useState("");

  useEffect(() => {
    if (selectedRange) {
      setRangeStartInput(String(selectedRange[0]));
      setRangeEndInput(String(selectedRange[1]));
    }
  }, [selectedRange]);

  const activeSet = new Set(nbvActiveTrackIndices);

  // Resolve bead range: default 1 to maxBeads; clamp and validate inputs
  const getBeadRange = useCallback((): [number, number] => {
    const start = rangeStartInput.trim() === "" ? 1 : Math.max(1, Math.min(maxBeads, parseInt(rangeStartInput, 10) || 1));
    const end = rangeEndInput.trim() === "" ? maxBeads : Math.max(1, Math.min(maxBeads, parseInt(rangeEndInput, 10) || maxBeads));
    return [Math.min(start, end), Math.max(start, end)];
  }, [rangeStartInput, rangeEndInput, maxBeads]);

  const handleTrackToggle = useCallback(
    (i: number) => {
      const next = activeSet.has(i)
        ? nbvActiveTrackIndices.filter((idx) => idx !== i)
        : [...nbvActiveTrackIndices, i].sort((a, b) => a - b);
      onNbvActiveTracksChange?.(next);
    },
    [nbvActiveTrackIndices, onNbvActiveTracksChange]
  );

  // Recalculate NBV from current node range (start/end inputs). First view in list is the best view.
  const runPipeline = useCallback(() => {
    const [start, end] = getBeadRange();
    const beadsInRange = beads.slice(start - 1, end);
    const set = new Set(nbvActiveTrackIndices);
    const forCalc = tracks.map((t, i) => ({ ...t, active: set.has(i) }));
    const activeTracks = forCalc.filter((t) => t.active);
    if (beadsInRange.length < 2 || activeTracks.length === 0) return;
    onBeadRangeApply?.(start, end);
    setBusy(true);
    requestAnimationFrame(() => {
      const { center: c, dataRadius } = getCenterAndRadius(beadsInRange);
      setCenter(c);
      const views = generateViewpoints(c, dataRadius);
      const visibility = computeVisibility(views, forCalc, beadsInRange, c);
      const scores = computeVisibleSurfaceScores(views, forCalc, visibility);
      const top = selectTop10(views, scores, c);
      setTopViews(top);
      setCurrentIndex(0);
      setBusy(false);
      // Apply best view first (index 0 is highest visible-surface score)
      if (top.length > 0) onApplyView?.(top[0].position, c);
    });
  }, [beads, tracks, nbvActiveTrackIndices, onApplyView, onBeadRangeApply, getBeadRange]);

  const goPrev = useCallback(() => {
    const prevIdx = Math.max(0, currentIndex - 1);
    setCurrentIndex(prevIdx);
    if (topViews[prevIdx]) onApplyView?.(topViews[prevIdx].position, center);
  }, [currentIndex, topViews, center, onApplyView]);
  const goNext = useCallback(() => {
    const nextIdx = Math.min(topViews.length - 1, currentIndex + 1);
    setCurrentIndex(nextIdx);
    if (topViews[nextIdx]) onApplyView?.(topViews[nextIdx].position, center);
  }, [currentIndex, topViews, center, onApplyView]);

  return (
    <div className="nbv">
      <div
          className={`nbv__bar ${minimized ? "nbv__bar--minimized" : ""}`}
          onPointerDown={onBarPointerDown}
        >
        <span className="nbv__title">NBV</span>
        <div className="nbv__tracks">
          {tracks.map((t, i) => (
            <label key={t.id} className="nbv__track" title={t.name} style={{ ["--track-color" as string]: t.color }}>
              <input
                type="checkbox"
                checked={activeSet.has(i)}
                onChange={() => handleTrackToggle(i)}
                className="nbv__track-cb"
              />
              <span className="nbv__track-dot" style={{ background: t.color }} />
              <span className="nbv__track-name">{t.name}</span>
            </label>
          ))}
        </div>
        {onToggleSelectMode && (
          <button
            type="button"
            className={`nbv__btn nbv__btn--select ${isSelectMode ? "nbv__btn--select-active" : ""}`}
            onClick={onToggleSelectMode}
            title={isSelectMode ? "Click two beads in main scene to set range (click again to cancel)" : "Select range in main scene: click first bead, then second bead"}
          >
            {isSelectMode ? "Selecting…" : "Select"}
          </button>
        )}
        <div className="nbv__range">
          <input
            type="number"
            className="nbv__range-input"
            placeholder="1"
            min={1}
            max={maxBeads}
            value={rangeStartInput}
            onChange={(e) => setRangeStartInput(e.target.value)}
            title={`Bead range start (1–${maxBeads})`}
            aria-label="Bead range from"
          />
          <span className="nbv__range-sep">–</span>
          <input
            type="number"
            className="nbv__range-input"
            placeholder={String(maxBeads)}
            min={1}
            max={maxBeads}
            value={rangeEndInput}
            onChange={(e) => setRangeEndInput(e.target.value)}
            title={`Bead range end (1–${maxBeads})`}
            aria-label="Bead range to"
          />
        </div>
        <button type="button" className="nbv__btn nbv__btn--primary" onClick={runPipeline} disabled={busy || beads.length < 2}>
          {busy ? "…" : "NBV"}
        </button>
        {onToggle2D && (
          <button
            type="button"
            className={`nbv__btn nbv__btn--2d ${show2D ? "nbv__btn--select-active" : ""}`}
            onClick={onToggle2D}
            title={show2D ? "Show 3D preview" : "Show 2D planar projection"}
          >
            2D
          </button>
        )}
        {topViews.length > 0 && (
          <div className="nbv__nav">
            <button type="button" className="nbv__btn" onClick={goPrev} disabled={currentIndex <= 0}>‹</button>
            <span className="nbv__nav-label">{currentIndex + 1}/{topViews.length}</span>
            <button type="button" className="nbv__btn" onClick={goNext} disabled={currentIndex >= topViews.length - 1}>›</button>
          </div>
        )}
        <div className="nbv__spacer" />
        {onMinimize && (
          <button type="button" className="nbv__btn" onClick={onMinimize} title="Minimize">{minimized ? "▢" : "−"}</button>
        )}
        {onClose && (
          <button type="button" className="nbv__btn" onClick={onClose} title="Close">✕</button>
        )}
      </div>
    </div>
  );
}
