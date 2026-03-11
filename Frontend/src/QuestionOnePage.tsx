import { useEffect, useRef, useState, useMemo, useCallback } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as d3 from "d3";
import {
  ChromosomePipeline,
  parsePositionCsv,
  matchBeadsToTracks,
  getTrackColor,
} from "./ChromosomeTrack3D";
import type { BeadData, TracksJson } from "./ChromosomeTrack3D";

// ── Constants ──────────────────────────────────────────────────────────────

const DUMMY_TRACKS_PATH = "/Data/Tracks/dummy_30tracks_chr8_127200000_127750000.json";
const POSITION_PATH = "/Data/Example/Calu3_chr8_127200000_127750000_original_position.csv";
const REGION_START = 127_200_000;
const BIN_SIZE = 5_000;
const NUM_QUESTIONS = 3;
const NUM_PHASE2_QUESTIONS = 3;
const NUM_PHASE3_QUESTIONS = 3;
const PHASE4_TRACK_COUNTS = [2, 3, 5, 7, 8]; // fixed track count per question
const NUM_PHASE4_QUESTIONS = PHASE4_TRACK_COUNTS.length;
const PHASE5_TRACK_COUNTS = [2, 4, 6]; // fixed track count per question
const NUM_PHASE5_QUESTIONS = PHASE5_TRACK_COUNTS.length;
const NUM_TUTORIAL_QUESTIONS = 5; // Fixed tutorial: Q1(phase1), Q2(phase2), Q3(phase3), Q4(phase4), Q5(phase5)
const TUTORIAL_Q4_TRACK_COUNT = 5;
const TUTORIAL_Q5_TRACK_COUNT = 4;
const COMMON_PEAK_TOLERANCE = 2; // bins: peaks within ±N count as same location
const PHASE4_TRACK_COLORS = [
  "#ff6b6b", "#bf812d", "#45b7d1",
  "#f9ca24", "#6c5ce7", "#00d2d3",
  "#ff9ff3", "#54a0ff", "#a29bfe",
];
const PEAK_MIN_HEIGHT = 0.25;
const PEAK_MIN_DISTANCE = 3;
const PEAK_LABELS = "ABCDEF".split("");

/** Build and download results as a readable JSON file, organized by Phase-Question */
function downloadResultsFile(data: {
  answers: Array<{ similarity: number | null; confidence: number | null }>;
  phase1Times: number[];
  phase2Results: Array<{ correctAnswer: string; userAnswer: string | null; correct: boolean; timeSpentMs: number }>;
  phase3Results: Array<{ correctBeadIndex: number; userBeadIndex: number | null; correct: boolean; timeSpentMs: number }>;
  phase4Results: Array<{ userAnswer: number | null; timeSpentMs: number }>;
  phase5Results: Array<{ correctAnswer: number; userAnswer: number | null; correct: boolean; timeSpentMs: number; confidence: number | null }>;
}) {
  const { answers, phase1Times, phase2Results, phase3Results, phase4Results, phase5Results } = data;

  const results: Record<string, unknown> = {};

  // Phase 1: similarity & confidence (no correct answer)
  for (let i = 0; i < NUM_QUESTIONS; i++) {
    const key = `Phase1-Question${i + 1}`;
    results[key] = {
      timeSpentMs: phase1Times[i] ?? null,
      timeSpentSeconds: phase1Times[i] != null ? (phase1Times[i] / 1000).toFixed(2) : null,
      similarity: answers[i]?.similarity ?? null,
      confidence: answers[i]?.confidence ?? null,
    };
  }

  // Phase 2: has correct answer
  for (let i = 0; i < phase2Results.length; i++) {
    const r = phase2Results[i];
    const key = `Phase2-Question${i + 1}`;
    results[key] = {
      timeSpentMs: r.timeSpentMs,
      timeSpentSeconds: (r.timeSpentMs / 1000).toFixed(2),
      correct: r.correct,
      userAnswer: r.userAnswer,
      correctAnswer: r.correctAnswer,
    };
  }

  // Phase 3: has correct answer (bead index)
  for (let i = 0; i < phase3Results.length; i++) {
    const r = phase3Results[i];
    const key = `Phase3-Question${i + 1}`;
    results[key] = {
      timeSpentMs: r.timeSpentMs,
      timeSpentSeconds: (r.timeSpentMs / 1000).toFixed(2),
      correct: r.correct,
      userBeadIndex: r.userBeadIndex,
      correctBeadIndex: r.correctBeadIndex,
    };
  }

  // Phase 4: confidence only (no correct answer)
  for (let i = 0; i < phase4Results.length; i++) {
    const r = phase4Results[i];
    const key = `Phase4-Question${i + 1}`;
    results[key] = {
      timeSpentMs: r.timeSpentMs,
      timeSpentSeconds: (r.timeSpentMs / 1000).toFixed(2),
      userAnswer: r.userAnswer,
    };
  }

  // Phase 5: has correct answer
  for (let i = 0; i < phase5Results.length; i++) {
    const r = phase5Results[i];
    const key = `Phase5-Question${i + 1}`;
    results[key] = {
      timeSpentMs: r.timeSpentMs,
      timeSpentSeconds: (r.timeSpentMs / 1000).toFixed(2),
      correct: r.correct,
      userAnswer: r.userAnswer,
      correctAnswer: r.correctAnswer,
      confidence: r.confidence,
    };
  }

  const exportData = {
    exportedAt: new Date().toISOString(),
    summary: {
      totalQuestions: NUM_QUESTIONS + NUM_PHASE2_QUESTIONS + NUM_PHASE3_QUESTIONS + NUM_PHASE4_QUESTIONS + NUM_PHASE5_QUESTIONS,
      phase2Correct: phase2Results.filter((r) => r.correct).length,
      phase3Correct: phase3Results.filter((r) => r.correct).length,
      phase5Correct: phase5Results.filter((r) => r.correct).length,
    },
    results,
  };

  const json = JSON.stringify(exportData, null, 2);
  const blob = new Blob([json], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `chromvis-results-${new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19)}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

function colorToRgba(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

// ── Peak detection ───────────────────────────────────────────────────────

interface Peak {
  index: number;
  value: number;
  label: string;
}

function detectPeaks(values: number[], maxPeaks = 6): Peak[] {
  const candidates: { index: number; value: number }[] = [];
  for (let i = 1; i < values.length - 1; i++) {
    if (
      values[i] >= PEAK_MIN_HEIGHT &&
      values[i] >= values[i - 1] &&
      values[i] >= values[i + 1]
    ) {
      candidates.push({ index: i, value: values[i] });
    }
  }
  candidates.sort((a, b) => b.value - a.value);
  const selected: { index: number; value: number }[] = [];
  for (const c of candidates) {
    if (selected.length >= maxPeaks) break;
    const tooClose = selected.some((p) => Math.abs(p.index - c.index) < PEAK_MIN_DISTANCE);
    if (!tooClose) selected.push(c);
  }
  selected.sort((a, b) => a.index - b.index);
  return selected.map((p, i) => ({
    ...p,
    label: PEAK_LABELS[i] ?? String(i + 1),
  }));
}

/** Count locations where ≥minTracks tracks have a peak (within COMMON_PEAK_TOLERANCE bins) */
function countCommonPeakLocations(trackValuesList: number[][], minTracks = 2): number {
  const peakIndicesByTrack: number[][] = [];
  for (let t = 0; t < trackValuesList.length; t++) {
    const peaks = detectPeaks(trackValuesList[t]);
    peakIndicesByTrack.push(peaks.map((p) => p.index));
  }

  const n = trackValuesList[0]?.length ?? 0;
  const inCommon: boolean[] = [];
  for (let i = 0; i < n; i++) {
    let trackCount = 0;
    for (const indices of peakIndicesByTrack) {
      if (indices.some((idx) => Math.abs(idx - i) <= COMMON_PEAK_TOLERANCE)) trackCount++;
    }
    inCommon[i] = trackCount >= minTracks;
  }

  let regionCount = 0;
  let inRegion = false;
  for (let i = 0; i < n; i++) {
    if (inCommon[i]) {
      if (!inRegion) {
        inRegion = true;
        regionCount++;
      }
    } else {
      inRegion = false;
    }
  }
  return regionCount;
}

/** Generate 5 options including correctAnswer, asymmetric (not symmetrically centered) */
function generatePhase5Options(correctAnswer: number, seed: number): number[] {
  const below = [correctAnswer - 2, correctAnswer - 1].filter((x) => x >= 0);
  const above = [correctAnswer + 1, correctAnswer + 2, correctAnswer + 3, correctAnswer + 4];
  const opts: number[] = [correctAnswer];
  if (seed % 2 === 0) {
    while (opts.length < 5 && above.length > 0) opts.push(above.shift()!);
    while (opts.length < 5 && below.length > 0) opts.push(below.pop()!);
  } else {
    while (opts.length < 5 && below.length > 0) opts.push(below.shift()!);
    while (opts.length < 5 && above.length > 0) opts.push(above.shift()!);
  }
  while (opts.length < 5) opts.push(Math.max(...opts) + 1);
  return [...new Set(opts)].sort((a, b) => a - b).slice(0, 5);
}

// ── Rating scale 1–5 ───────────────────────────────────────────────────────

function RatingScale({
  value,
  onChange,
  labelLow,
  labelHigh,
}: {
  value: number | null;
  onChange: (v: number) => void;
  labelLow: string;
  labelHigh: string;
}) {
  return (
    <div style={ratingRowStyle}>
      <div style={ratingScaleStyle}>
        {[1, 2, 3, 4, 5].map((n) => (
          <button
            key={n}
            type="button"
            onClick={() => onChange(n)}
            style={{
              ...ratingBtnStyle,
              ...(value === n ? ratingBtnActiveStyle : {}),
            }}
          >
            {n}
          </button>
        ))}
      </div>
      <div style={ratingLabelsStyle}>
        <span>{labelLow}</span>
        <span>{labelHigh}</span>
      </div>
    </div>
  );
}

// ── D3 Line Chart ──────────────────────────────────────────────────────────

function LineChart({
  values,
  trackName,
  color = "#45b7d1",
  hideAxisLabels = false,
}: {
  values: number[];
  trackName: string;
  color?: string;
  hideAxisLabels?: boolean;
}) {
  const svgRef = useRef<SVGSVGElement>(null);

  const draw = useCallback(() => {
    if (!svgRef.current || values.length === 0) return;
    const container = svgRef.current.parentElement;
    if (!container) return;
    const W = Math.max(container.clientWidth || 300, 100);
    const H = Math.max(container.clientHeight || 120, 80);
    const margin = hideAxisLabels
      ? { top: 20, right: 6, bottom: 8, left: 25 }
      : { top: 40, right: 30, bottom: 60, left: 60 };
    const width = W - margin.left - margin.right;
    const height = H - margin.top - margin.bottom;
    if (width <= 0 || height <= 0) return;
    d3.select(svgRef.current).selectAll("*").remove();
    const svg = d3.select(svgRef.current).attr("width", W).attr("height", H);
    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
    const genomicPositions = values.map((_, i) => REGION_START + i * BIN_SIZE);
    const xScale = d3.scaleLinear().domain([genomicPositions[0], genomicPositions[genomicPositions.length - 1]]).range([0, width]);
    const yScale = d3.scaleLinear().domain([0, 1]).range([height, 0]);
    const area = d3.area<number>().x((_, i) => xScale(genomicPositions[i])).y0(height).y1((d) => yScale(d)).curve(d3.curveCatmullRom.alpha(0.5));
    const fillRgba = colorToRgba(color, 0.15);
    g.append("path").datum(values).attr("fill", fillRgba).attr("d", area);
    const line = d3.line<number>().x((_, i) => xScale(genomicPositions[i])).y((d) => yScale(d)).curve(d3.curveCatmullRom.alpha(0.5));
    g.append("path").datum(values).attr("fill", "none").attr("stroke", color).attr("stroke-width", 2).attr("d", line);
    const xAxis = d3.axisBottom(xScale).ticks(5).tickFormat((d) => `${(+d / 1_000_000).toFixed(2)} Mb`);
    g.append("g").attr("transform", `translate(0,${height})`).call(xAxis).call((ax) => {
      ax.selectAll("text").attr("fill", "#b0bec5").attr("font-size", 11);
      ax.selectAll("line, path").attr("stroke", "rgba(255,255,255,0.2)");
    });
    const yAxis = d3.axisLeft(yScale).ticks(5);
    g.append("g").call(yAxis).call((ax) => {
      ax.selectAll("text").attr("fill", "#b0bec5").attr("font-size", 11);
      ax.selectAll("line, path").attr("stroke", "rgba(255,255,255,0.2)");
    });
    g.append("g").call(d3.axisLeft(yScale).ticks(5).tickSize(-width).tickFormat(() => "")).call((ax) => {
      ax.selectAll("line").attr("stroke", "rgba(255,255,255,0.06)");
      ax.select(".domain").remove();
    });
    if (!hideAxisLabels) {
      svg.append("text").attr("x", W / 2).attr("y", H - 8).attr("text-anchor", "middle").attr("fill", "#b0bec5").attr("font-size", 12).text("Genomic Position (chr8)");
      svg.append("text").attr("transform", "rotate(-90)").attr("x", -(H / 2)).attr("y", 14).attr("text-anchor", "middle").attr("fill", "#b0bec5").attr("font-size", 12).text("Normalized Signal");
    }
  }, [values, trackName, color, hideAxisLabels]);

  useEffect(() => {
    draw();
    const container = svgRef.current?.parentElement;
    if (!container) return;
    const ro = new ResizeObserver(() => draw());
    ro.observe(container);
    return () => ro.disconnect();
  }, [draw]);

  return (
    <svg
      ref={svgRef}
      style={{ width: "100%", height: "100%", display: "block" }}
    />
  );
}

// ── Multi-track line charts (one per track, column layout, for Phase 4 & 5) ─────

const PHASE4_CHART_MIN_HEIGHT = 150;

function MultiTrackLineChartColumn({
  tracks,
  compact,
}: {
  tracks: Phase4TrackData[];
  compact?: boolean;
}) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 2,
        height: "100%",
        minHeight: 0,
        overflow: compact ? "hidden" : "auto",
        padding: "4px",
      }}
    >
      {tracks.map((t, i) => (
        <div
          key={t.trackName}
          style={{
            flex: compact ? "1 1 0" : `0 0 ${PHASE4_CHART_MIN_HEIGHT}px`,
            height: compact ? undefined : PHASE4_CHART_MIN_HEIGHT,
            minHeight: compact ? 0 : PHASE4_CHART_MIN_HEIGHT,
            minWidth: 0,
          }}
        >
          <div style={{ width: "100%", height: "100%" }}>
            <LineChart
              values={t.trackValues}
              trackName={t.trackName}
              color={PHASE4_TRACK_COLORS[i % PHASE4_TRACK_COLORS.length]}
              hideAxisLabels
            />
          </div>
        </div>
      ))}
    </div>
  );
}

// ── D3 Line Chart with peak labels ─────────────────────────────────────────

function LineChartWithPeaks({
  values,
  trackName,
  peaks,
  color = "#45b7d1",
}: {
  values: number[];
  trackName: string;
  peaks: Peak[];
  color?: string;
}) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || values.length === 0) return;

    const container = svgRef.current.parentElement;
    const W = container?.clientWidth ?? 500;
    const H = container?.clientHeight ?? 400;
    const margin = { top: 40, right: 30, bottom: 60, left: 60 };
    const width = W - margin.left - margin.right;
    const height = H - margin.top - margin.bottom;

    d3.select(svgRef.current).selectAll("*").remove();

    const svg = d3.select(svgRef.current).attr("width", W).attr("height", H);
    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const genomicPositions = values.map((_, i) => REGION_START + i * BIN_SIZE);
    const xScale = d3
      .scaleLinear()
      .domain([genomicPositions[0], genomicPositions[genomicPositions.length - 1]])
      .range([0, width]);
    const yScale = d3.scaleLinear().domain([0, 1]).range([height, 0]);

    const fillRgba = colorToRgba(color, 0.15);
    const area = d3
      .area<number>()
      .x((_, i) => xScale(genomicPositions[i]))
      .y0(height)
      .y1((d) => yScale(d))
      .curve(d3.curveCatmullRom.alpha(0.5));
    g.append("path").datum(values).attr("fill", fillRgba).attr("d", area);

    const line = d3
      .line<number>()
      .x((_, i) => xScale(genomicPositions[i]))
      .y((d) => yScale(d))
      .curve(d3.curveCatmullRom.alpha(0.5));
    g.append("path").datum(values).attr("fill", "none").attr("stroke", color).attr("stroke-width", 2).attr("d", line);

    const xAxis = d3.axisBottom(xScale).ticks(5).tickFormat((d) => `${(+d / 1_000_000).toFixed(2)} Mb`);
    g.append("g").attr("transform", `translate(0,${height})`).call(xAxis).call((ax) => {
      ax.selectAll("text").attr("fill", "#b0bec5").attr("font-size", 11);
      ax.selectAll("line, path").attr("stroke", "rgba(255,255,255,0.2)");
    });

    const yAxis = d3.axisLeft(yScale).ticks(5);
    g.append("g").call(yAxis).call((ax) => {
      ax.selectAll("text").attr("fill", "#b0bec5").attr("font-size", 11);
      ax.selectAll("line, path").attr("stroke", "rgba(255,255,255,0.2)");
    });

    g.append("g")
      .call(d3.axisLeft(yScale).ticks(5).tickSize(-width).tickFormat(() => ""))
      .call((ax) => {
        ax.selectAll("line").attr("stroke", "rgba(255,255,255,0.06)");
        ax.select(".domain").remove();
      });

    svg.append("text").attr("x", W / 2).attr("y", H - 8).attr("text-anchor", "middle").attr("fill", "#b0bec5").attr("font-size", 12).text("Genomic Position (chr8)");
    svg.append("text").attr("transform", "rotate(-90)").attr("x", -(H / 2)).attr("y", 14).attr("text-anchor", "middle").attr("fill", "#b0bec5").attr("font-size", 12).text("Normalized Signal");

    peaks.forEach((peak) => {
      const x = xScale(genomicPositions[peak.index]);
      const y = yScale(peak.value);
      g.append("text")
        .attr("x", x)
        .attr("y", y - 10)
        .attr("text-anchor", "middle")
        .attr("fill", "#f59e0b")
        .attr("font-size", 14)
        .attr("font-weight", "700")
        .text(peak.label);
    });
  }, [values, trackName, peaks, color]);

  return <svg ref={svgRef} style={{ width: "100%", height: "100%", display: "block" }} />;
}

// ── D3 Line Chart with single peak highlight (for Phase 3) ───────────────────

function LineChartWithSinglePeakHighlight({
  values,
  trackName,
  peak,
  color = "#45b7d1",
}: {
  values: number[];
  trackName: string;
  peak: Peak;
  color?: string;
}) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || values.length === 0) return;

    const container = svgRef.current.parentElement;
    const W = container?.clientWidth ?? 500;
    const H = container?.clientHeight ?? 400;
    const margin = { top: 40, right: 30, bottom: 60, left: 60 };
    const width = W - margin.left - margin.right;
    const height = H - margin.top - margin.bottom;

    d3.select(svgRef.current).selectAll("*").remove();

    const svg = d3.select(svgRef.current).attr("width", W).attr("height", H);
    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const genomicPositions = values.map((_, i) => REGION_START + i * BIN_SIZE);
    const xScale = d3
      .scaleLinear()
      .domain([genomicPositions[0], genomicPositions[genomicPositions.length - 1]])
      .range([0, width]);
    const yScale = d3.scaleLinear().domain([0, 1]).range([height, 0]);

    const fillRgba = colorToRgba(color, 0.15);
    const area = d3
      .area<number>()
      .x((_, i) => xScale(genomicPositions[i]))
      .y0(height)
      .y1((d) => yScale(d))
      .curve(d3.curveCatmullRom.alpha(0.5));
    g.append("path").datum(values).attr("fill", fillRgba).attr("d", area);

    const line = d3
      .line<number>()
      .x((_, i) => xScale(genomicPositions[i]))
      .y((d) => yScale(d))
      .curve(d3.curveCatmullRom.alpha(0.5));
    g.append("path").datum(values).attr("fill", "none").attr("stroke", color).attr("stroke-width", 2).attr("d", line);

    const xAxis = d3.axisBottom(xScale).ticks(5).tickFormat((d) => `${(+d / 1_000_000).toFixed(2)} Mb`);
    g.append("g").attr("transform", `translate(0,${height})`).call(xAxis).call((ax) => {
      ax.selectAll("text").attr("fill", "#b0bec5").attr("font-size", 11);
      ax.selectAll("line, path").attr("stroke", "rgba(255,255,255,0.2)");
    });

    const yAxis = d3.axisLeft(yScale).ticks(5);
    g.append("g").call(yAxis).call((ax) => {
      ax.selectAll("text").attr("fill", "#b0bec5").attr("font-size", 11);
      ax.selectAll("line, path").attr("stroke", "rgba(255,255,255,0.2)");
    });

    g.append("g")
      .call(d3.axisLeft(yScale).ticks(5).tickSize(-width).tickFormat(() => ""))
      .call((ax) => {
        ax.selectAll("line").attr("stroke", "rgba(255,255,255,0.06)");
        ax.select(".domain").remove();
      });

    svg.append("text").attr("x", W / 2).attr("y", H - 8).attr("text-anchor", "middle").attr("fill", "#b0bec5").attr("font-size", 12).text("Genomic Position (chr8)");
    svg.append("text").attr("transform", "rotate(-90)").attr("x", -(H / 2)).attr("y", 14).attr("text-anchor", "middle").attr("fill", "#b0bec5").attr("font-size", 12).text("Normalized Signal");

    // Highlight only this peak: glowing circle + label
    const x = xScale(genomicPositions[peak.index]);
    const y = yScale(peak.value);
    g.append("circle")
      .attr("cx", x)
      .attr("cy", y)
      .attr("r", 14)
      .attr("fill", "rgba(245,158,11,0.25)")
      .attr("stroke", "#f59e0b")
      .attr("stroke-width", 3);
    g.append("text")
      .attr("x", x)
      .attr("y", y - 22)
      .attr("text-anchor", "middle")
      .attr("fill", "#f59e0b")
      .attr("font-size", 16)
      .attr("font-weight", "700")
      .text(peak.label);
  }, [values, trackName, peak, color]);

  return <svg ref={svgRef} style={{ width: "100%", height: "100%", display: "block" }} />;
}

// ── Main component ─────────────────────────────────────────────────────────

interface BlockData {
  trackName: string;
  trackValues: number[];
  beads: BeadData[];
  sampleId: number;
}

interface Phase2BlockData {
  trackName: string;
  trackValues: number[];
  beads: BeadData[];
  sampleId: number;
  peaks: Peak[];
  highlightedPeakIndex: number;
  highlightedBeadIndex: number;
  correctAnswer: string;
}

interface Phase4TrackData {
  trackName: string;
  trackValues: number[];
}

interface Phase4BlockData {
  tracks: Phase4TrackData[];
  beads: BeadData[];
  trackNames: string[];
  sampleId: number;
}

interface Phase5BlockData {
  tracks: Phase4TrackData[];
  beads: BeadData[];
  trackNames: string[];
  sampleId: number;
  correctAnswer: number;
  options: number[];
}

/** Fixed tutorial blocks: [phase1, phase2, phase3, phase4, phase5] */
interface TutorialBlocks {
  block1: BlockData;
  block2: Phase2BlockData;
  block3: Phase2BlockData;
  block4: Phase4BlockData;
  block5: Phase5BlockData;
}

export default function QuestionOnePage() {
  const [phase, setPhase] = useState<0 | 1 | 2 | 3 | 4 | 5>(0);
  const [blocks, setBlocks] = useState<BlockData[]>([]);
  const [phase2Blocks, setPhase2Blocks] = useState<Phase2BlockData[]>([]);
  const [phase3Blocks, setPhase3Blocks] = useState<Phase2BlockData[]>([]);
  const [phase4Blocks, setPhase4Blocks] = useState<Phase4BlockData[]>([]);
  const [phase5Blocks, setPhase5Blocks] = useState<Phase5BlockData[]>([]);
  const [currentBlock, setCurrentBlock] = useState(0);
  const [answers, setAnswers] = useState<Array<{ similarity: number | null; confidence: number | null }>>(
    Array(NUM_QUESTIONS).fill(null).map(() => ({ similarity: null, confidence: null })),
  );
  const [phase2Answers, setPhase2Answers] = useState<(string | null)[]>(
    Array(NUM_PHASE2_QUESTIONS).fill(null),
  );
  const [phase2Results, setPhase2Results] = useState<
    Array<{ correctAnswer: string; userAnswer: string | null; correct: boolean; timeSpentMs: number }>
  >([]);
  const [phase3Answers, setPhase3Answers] = useState<(number | null)[]>(
    Array(NUM_PHASE3_QUESTIONS).fill(null),
  );
  const [phase3Results, setPhase3Results] = useState<
    Array<{ correctBeadIndex: number; userBeadIndex: number | null; correct: boolean; timeSpentMs: number }>
  >([]);
  const [phase3HoveredBead, setPhase3HoveredBead] = useState<number | null>(null);
  const [phase4Answers, setPhase4Answers] = useState<(number | null)[]>(
    Array(NUM_PHASE4_QUESTIONS).fill(null),
  );
  const [phase4Results, setPhase4Results] = useState<
    Array<{ userAnswer: number | null; timeSpentMs: number }>
  >([]);
  const [phase5Answers, setPhase5Answers] = useState<(number | null)[]>(
    Array(NUM_PHASE5_QUESTIONS).fill(null),
  );
  const [phase5Confidence, setPhase5Confidence] = useState<(number | null)[]>(
    Array(NUM_PHASE5_QUESTIONS).fill(null),
  );
  const [phase5Results, setPhase5Results] = useState<
    Array<{ correctAnswer: number; userAnswer: number | null; correct: boolean; timeSpentMs: number; confidence: number | null }>
  >([]);
  const [phase1Times, setPhase1Times] = useState<number[]>([]);
  const [phase1Q1TimerStarted, setPhase1Q1TimerStarted] = useState(false);
  const [tutorialBlocks, setTutorialBlocks] = useState<TutorialBlocks | null>(null);
  const [tutorialPhase1Answer, setTutorialPhase1Answer] = useState<{ similarity: number | null; confidence: number | null }>({ similarity: null, confidence: null });
  const [tutorialPhase2Answer, setTutorialPhase2Answer] = useState<string | null>(null);
  const [tutorialPhase3Answer, setTutorialPhase3Answer] = useState<number | null>(null);
  const [tutorialPhase3HoveredBead, setTutorialPhase3HoveredBead] = useState<number | null>(null);
  const [tutorialPhase4Answer, setTutorialPhase4Answer] = useState<number | null>(null);
  const [tutorialPhase5Answer, setTutorialPhase5Answer] = useState<number | null>(null);
  const [tutorialPhase5Confidence, setTutorialPhase5Confidence] = useState<number | null>(null);
  const [gamma, setGamma] = useState(1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const questionStartTimeRef = useRef<number>(Date.now());
  const phase5AnswersRef = useRef<(number | null)[]>(Array(NUM_PHASE5_QUESTIONS).fill(null));
  const phase5ConfidenceRef = useRef<(number | null)[]>(Array(NUM_PHASE5_QUESTIONS).fill(null));

  useEffect(() => {
    phase5AnswersRef.current = phase5Answers;
  }, [phase5Answers]);
  useEffect(() => {
    phase5ConfidenceRef.current = phase5Confidence;
  }, [phase5Confidence]);

  // Load data and prepare 5 random (track, sample) blocks
  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const [trkRes, posRes] = await Promise.all([
          fetch(DUMMY_TRACKS_PATH),
          fetch(POSITION_PATH),
        ]);
        if (!trkRes.ok) throw new Error(`Cannot load tracks JSON (${trkRes.status})`);
        if (!posRes.ok) throw new Error(`Cannot load position CSV (${posRes.status})`);

        const trkJson: TracksJson = await trkRes.json();
        const posText = await posRes.text();

        const allTrackNames = Object.keys(trkJson.tracks);
        const allSamples = parsePositionCsv(posText);
        const sampleIds = [...new Set(allSamples.map((s) => s.sampleId))].sort(
          (a, b) => a - b,
        );

        const built: BlockData[] = [];
        const builtPhase2: Phase2BlockData[] = [];
        const builtPhase3: Phase2BlockData[] = [];

        for (let i = 0; i < NUM_QUESTIONS; i++) {
          const trackIdx = Math.floor(Math.random() * allTrackNames.length);
          const chosenName = allTrackNames[trackIdx];
          const sampleIdx = Math.floor(Math.random() * sampleIds.length);
          const chosenSampleId = sampleIds[sampleIdx];

          const singleTrackJson: TracksJson = {
            region: trkJson.region,
            tracks: { [chosenName]: trkJson.tracks[chosenName] },
          };

          const filtered = allSamples
            .filter((s) => s.sampleId === chosenSampleId)
            .sort((a, b) => a.start_value - b.start_value);

          const beadsData = matchBeadsToTracks(
            filtered,
            singleTrackJson,
            [chosenName],
            true,
          );
          built.push({
            trackName: chosenName,
            trackValues: trkJson.tracks[chosenName].normalized,
            beads: beadsData,
            sampleId: chosenSampleId,
          });
        }

        for (let i = 0; i < NUM_PHASE2_QUESTIONS; i++) {
          let attempts = 0;
          const maxAttempts = 20;
          while (attempts < maxAttempts) {
            const trackIdx = Math.floor(Math.random() * allTrackNames.length);
            const chosenName = allTrackNames[trackIdx];
            const trackVals = trkJson.tracks[chosenName].normalized;
            const peaks = detectPeaks(trackVals);
            if (peaks.length < 2) {
              attempts++;
              continue;
            }
            const sampleIdx = Math.floor(Math.random() * sampleIds.length);
            const chosenSampleId = sampleIds[sampleIdx];
            const singleTrackJson: TracksJson = {
              region: trkJson.region,
              tracks: { [chosenName]: trkJson.tracks[chosenName] },
            };
            const filtered = allSamples
              .filter((s) => s.sampleId === chosenSampleId)
              .sort((a, b) => a.start_value - b.start_value);
            const beadsData = matchBeadsToTracks(
              filtered,
              singleTrackJson,
              [chosenName],
              true,
            );
            const highlightIdx = Math.floor(Math.random() * peaks.length);
            const highlightedPeak = peaks[highlightIdx];
            const beadIndex = Math.min(highlightedPeak.index, beadsData.length - 1);
            builtPhase2.push({
              trackName: chosenName,
              trackValues: trackVals,
              beads: beadsData,
              sampleId: chosenSampleId,
              peaks,
              highlightedPeakIndex: highlightIdx,
              highlightedBeadIndex: beadIndex,
              correctAnswer: highlightedPeak.label,
            });
            break;
          }
          if (builtPhase2.length <= i) {
            const fallbackBeads = matchBeadsToTracks(
              allSamples
                .filter((s) => s.sampleId === sampleIds[0])
                .sort((a, b) => a.start_value - b.start_value),
              {
                region: trkJson.region,
                tracks: { [allTrackNames[0]]: trkJson.tracks[allTrackNames[0]] },
              },
              [allTrackNames[0]],
              true,
            );
            const fallbackPeaks = [{ index: 10, value: 0.5, label: "A" }, { index: Math.min(50, fallbackBeads.length - 1), value: 0.5, label: "B" }];
            builtPhase2.push({
              trackName: allTrackNames[0],
              trackValues: trkJson.tracks[allTrackNames[0]].normalized,
              beads: fallbackBeads,
              sampleId: sampleIds[0],
              peaks: fallbackPeaks,
              highlightedPeakIndex: 0,
              highlightedBeadIndex: Math.min(10, fallbackBeads.length - 1),
              correctAnswer: "A",
            });
          }
        }

        for (let i = 0; i < NUM_PHASE3_QUESTIONS; i++) {
          let attempts = 0;
          const maxAttempts = 20;
          while (attempts < maxAttempts) {
            const trackIdx = Math.floor(Math.random() * allTrackNames.length);
            const chosenName = allTrackNames[trackIdx];
            const trackVals = trkJson.tracks[chosenName].normalized;
            const peaks = detectPeaks(trackVals);
            if (peaks.length < 2) {
              attempts++;
              continue;
            }
            const sampleIdx = Math.floor(Math.random() * sampleIds.length);
            const chosenSampleId = sampleIds[sampleIdx];
            const singleTrackJson: TracksJson = {
              region: trkJson.region,
              tracks: { [chosenName]: trkJson.tracks[chosenName] },
            };
            const filtered = allSamples
              .filter((s) => s.sampleId === chosenSampleId)
              .sort((a, b) => a.start_value - b.start_value);
            const beadsData = matchBeadsToTracks(
              filtered,
              singleTrackJson,
              [chosenName],
              true,
            );
            const highlightIdx = Math.floor(Math.random() * peaks.length);
            const highlightedPeak = peaks[highlightIdx];
            const beadIndex = Math.min(highlightedPeak.index, beadsData.length - 1);
            builtPhase3.push({
              trackName: chosenName,
              trackValues: trackVals,
              beads: beadsData,
              sampleId: chosenSampleId,
              peaks,
              highlightedPeakIndex: highlightIdx,
              highlightedBeadIndex: beadIndex,
              correctAnswer: highlightedPeak.label,
            });
            break;
          }
          if (builtPhase3.length <= i) {
            const fallbackBeads = matchBeadsToTracks(
              allSamples
                .filter((s) => s.sampleId === sampleIds[0])
                .sort((a, b) => a.start_value - b.start_value),
              {
                region: trkJson.region,
                tracks: { [allTrackNames[0]]: trkJson.tracks[allTrackNames[0]] },
              },
              [allTrackNames[0]],
              true,
            );
            const fallbackPeaks = [{ index: 10, value: 0.5, label: "A" }, { index: Math.min(50, fallbackBeads.length - 1), value: 0.5, label: "B" }];
            builtPhase3.push({
              trackName: allTrackNames[0],
              trackValues: trkJson.tracks[allTrackNames[0]].normalized,
              beads: fallbackBeads,
              sampleId: sampleIds[0],
              peaks: fallbackPeaks,
              highlightedPeakIndex: 0,
              highlightedBeadIndex: Math.min(10, fallbackBeads.length - 1),
              correctAnswer: "A",
            });
          }
        }

        const builtPhase4: Phase4BlockData[] = [];
        for (let i = 0; i < NUM_PHASE4_QUESTIONS; i++) {
          const numTracks = PHASE4_TRACK_COUNTS[i];
          const shuffled = [...allTrackNames].sort(() => Math.random() - 0.5);
          const chosenNames = shuffled.slice(0, numTracks);
          const sampleIdx = Math.floor(Math.random() * sampleIds.length);
          const chosenSampleId = sampleIds[sampleIdx];

          const multiTrackTracks: Record<string, { raw: number[]; normalized: number[] }> = {};
          for (const name of chosenNames) {
            multiTrackTracks[name] = trkJson.tracks[name];
          }
          const multiTrackJson: TracksJson = {
            region: trkJson.region,
            tracks: multiTrackTracks,
          };

          const filtered = allSamples
            .filter((s) => s.sampleId === chosenSampleId)
            .sort((a, b) => a.start_value - b.start_value);

          const beadsData = matchBeadsToTracks(
            filtered,
            multiTrackJson,
            chosenNames,
            true,
          );

          builtPhase4.push({
            tracks: chosenNames.map((name) => ({
              trackName: name,
              trackValues: trkJson.tracks[name].normalized,
            })),
            beads: beadsData,
            trackNames: chosenNames,
            sampleId: chosenSampleId,
          });
        }

        const builtPhase5: Phase5BlockData[] = [];
        for (let i = 0; i < NUM_PHASE5_QUESTIONS; i++) {
          const numTracks = PHASE5_TRACK_COUNTS[i];
          let attempts = 0;
          const maxAttempts = 50;
          while (attempts < maxAttempts) {
            const shuffled = [...allTrackNames].sort(() => Math.random() - 0.5);
            const chosenNames = shuffled.slice(0, numTracks);
            const trackValuesList = chosenNames.map((name) => trkJson.tracks[name].normalized);
            const correctAnswer = countCommonPeakLocations(trackValuesList, numTracks);
            if (correctAnswer < 1) {
              attempts++;
              continue;
            }
            const sampleIdx = Math.floor(Math.random() * sampleIds.length);
            const chosenSampleId = sampleIds[sampleIdx];
            const multiTrackTracks: Record<string, { raw: number[]; normalized: number[] }> = {};
            for (const name of chosenNames) {
              multiTrackTracks[name] = trkJson.tracks[name];
            }
            const multiTrackJson: TracksJson = {
              region: trkJson.region,
              tracks: multiTrackTracks,
            };
            const filtered = allSamples
              .filter((s) => s.sampleId === chosenSampleId)
              .sort((a, b) => a.start_value - b.start_value);
            const beadsData = matchBeadsToTracks(
              filtered,
              multiTrackJson,
              chosenNames,
              true,
            );
            const options = generatePhase5Options(correctAnswer, i);
            if (!options.includes(correctAnswer)) {
              attempts++;
              continue;
            }
            builtPhase5.push({
              tracks: chosenNames.map((name) => ({
                trackName: name,
                trackValues: trkJson.tracks[name].normalized,
              })),
              beads: beadsData,
              trackNames: chosenNames,
              sampleId: chosenSampleId,
              correctAnswer,
              options,
            });
            break;
          }
          if (builtPhase5.length <= i) {
            const fallbackNames = allTrackNames.slice(0, numTracks);
            const fallbackVals = fallbackNames.map((n) => trkJson.tracks[n].normalized);
            const correctAnswer = Math.max(1, countCommonPeakLocations(fallbackVals, numTracks));
            const multiTrackJson: TracksJson = {
              region: trkJson.region,
              tracks: Object.fromEntries(fallbackNames.map((n) => [n, trkJson.tracks[n]])),
            };
            const filtered = allSamples
              .filter((s) => s.sampleId === sampleIds[0])
              .sort((a, b) => a.start_value - b.start_value);
            builtPhase5.push({
              tracks: fallbackNames.map((name) => ({
                trackName: name,
                trackValues: trkJson.tracks[name].normalized,
              })),
              beads: matchBeadsToTracks(filtered, multiTrackJson, fallbackNames, true),
              trackNames: fallbackNames,
              sampleId: sampleIds[0],
              correctAnswer,
              options: generatePhase5Options(correctAnswer, i),
            });
          }
        }

        // Build fixed tutorial blocks (deterministic, no random)
        const sortedTrackNames = [...allTrackNames].sort();
        let tutorialTrackName: string = sortedTrackNames[0];
        let maxPeakCount = 0;
        for (const name of sortedTrackNames) {
          const peaks = detectPeaks(trkJson.tracks[name].normalized);
          if (peaks.length > 3 && peaks.length > maxPeakCount) {
            tutorialTrackName = name;
            maxPeakCount = peaks.length;
            if (peaks.length >= 5) break;
          }
        }
        if (maxPeakCount <= 3) {
          for (const name of sortedTrackNames) {
            const peaks = detectPeaks(trkJson.tracks[name].normalized);
            if (peaks.length > maxPeakCount) {
              tutorialTrackName = name;
              maxPeakCount = peaks.length;
            }
          }
        }
        const tutorialSampleId = sampleIds[0];
        const tutorialTrackVals = trkJson.tracks[tutorialTrackName].normalized;
        const tutorialPeaks = detectPeaks(tutorialTrackVals);
        const tutorialSingleJson: TracksJson = {
          region: trkJson.region,
          tracks: { [tutorialTrackName]: trkJson.tracks[tutorialTrackName] },
        };
        const tutorialFiltered = allSamples
          .filter((s) => s.sampleId === tutorialSampleId)
          .sort((a, b) => a.start_value - b.start_value);
        const tutorialBeads = matchBeadsToTracks(tutorialFiltered, tutorialSingleJson, [tutorialTrackName], true);
        const highlightIdx2 = Math.min(0, tutorialPeaks.length - 1);
        const highlightIdx3 = tutorialPeaks.length > 1 ? Math.min(1, tutorialPeaks.length - 1) : highlightIdx2;
        const builtTutorial: TutorialBlocks = {
          block1: {
            trackName: tutorialTrackName,
            trackValues: tutorialTrackVals,
            beads: tutorialBeads,
            sampleId: tutorialSampleId,
          },
          block2: {
            trackName: tutorialTrackName,
            trackValues: tutorialTrackVals,
            beads: tutorialBeads,
            sampleId: tutorialSampleId,
            peaks: tutorialPeaks,
            highlightedPeakIndex: highlightIdx2,
            highlightedBeadIndex: Math.min(tutorialPeaks[highlightIdx2].index, tutorialBeads.length - 1),
            correctAnswer: tutorialPeaks[highlightIdx2].label,
          },
          block3: {
            trackName: tutorialTrackName,
            trackValues: tutorialTrackVals,
            beads: tutorialBeads,
            sampleId: tutorialSampleId,
            peaks: tutorialPeaks,
            highlightedPeakIndex: highlightIdx3,
            highlightedBeadIndex: Math.min(tutorialPeaks[highlightIdx3].index, tutorialBeads.length - 1),
            correctAnswer: tutorialPeaks[highlightIdx3].label,
          },
          block4: (() => {
            const q4Names = sortedTrackNames.slice(0, TUTORIAL_Q4_TRACK_COUNT);
            const q4Tracks: Record<string, { raw: number[]; normalized: number[] }> = {};
            for (const n of q4Names) q4Tracks[n] = trkJson.tracks[n];
            const q4Json: TracksJson = { region: trkJson.region, tracks: q4Tracks };
            const q4Filtered = allSamples.filter((s) => s.sampleId === tutorialSampleId).sort((a, b) => a.start_value - b.start_value);
            return {
              tracks: q4Names.map((n) => ({ trackName: n, trackValues: trkJson.tracks[n].normalized })),
              beads: matchBeadsToTracks(q4Filtered, q4Json, q4Names, true),
              trackNames: q4Names,
              sampleId: tutorialSampleId,
            };
          })(),
          block5: (() => {
            let q5Names: string[] = [];
            let correctAnswer = 0;
            for (let offset = 0; offset <= sortedTrackNames.length - TUTORIAL_Q5_TRACK_COUNT; offset++) {
              const names = sortedTrackNames.slice(offset, offset + TUTORIAL_Q5_TRACK_COUNT);
              const vals = names.map((n) => trkJson.tracks[n].normalized);
              const count = countCommonPeakLocations(vals, TUTORIAL_Q5_TRACK_COUNT);
              if (count >= 1) {
                q5Names = names;
                correctAnswer = count;
                break;
              }
            }
            if (q5Names.length === 0) {
              q5Names = sortedTrackNames.slice(0, TUTORIAL_Q5_TRACK_COUNT);
              correctAnswer = Math.max(1, countCommonPeakLocations(q5Names.map((n) => trkJson.tracks[n].normalized), TUTORIAL_Q5_TRACK_COUNT));
            }
            const q5Tracks: Record<string, { raw: number[]; normalized: number[] }> = {};
            for (const n of q5Names) q5Tracks[n] = trkJson.tracks[n];
            const q5Json: TracksJson = { region: trkJson.region, tracks: q5Tracks };
            const q5Filtered = allSamples.filter((s) => s.sampleId === tutorialSampleId).sort((a, b) => a.start_value - b.start_value);
            const options = generatePhase5Options(correctAnswer, 0);
            return {
              tracks: q5Names.map((n) => ({ trackName: n, trackValues: trkJson.tracks[n].normalized })),
              beads: matchBeadsToTracks(q5Filtered, q5Json, q5Names, true),
              trackNames: q5Names,
              sampleId: tutorialSampleId,
              correctAnswer,
              options: options.includes(correctAnswer) ? options : [correctAnswer, ...options.filter((x) => x !== correctAnswer)].slice(0, 5),
            };
          })(),
        };

        if (!cancelled) {
          setTutorialBlocks(builtTutorial);
          setBlocks(built);
          setPhase2Blocks(builtPhase2);
          setPhase3Blocks(builtPhase3);
          setPhase4Blocks(builtPhase4);
          setPhase5Blocks(builtPhase5);
        }
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    load();
    return () => { cancelled = true; };
  }, []);

  const block = phase === 0 && tutorialBlocks
    ? (currentBlock === 0 ? tutorialBlocks.block1 : null)
    : blocks[currentBlock];
  const phase2Block = phase === 0 && tutorialBlocks
    ? (currentBlock === 1 ? tutorialBlocks.block2 : null)
    : phase2Blocks[currentBlock];
  const phase3Block = phase === 0 && tutorialBlocks
    ? (currentBlock === 2 ? tutorialBlocks.block3 : null)
    : phase3Blocks[currentBlock];
  const phase4Block = phase === 0 && tutorialBlocks
    ? (currentBlock === 3 ? tutorialBlocks.block4 : null)
    : phase4Blocks[currentBlock];
  const phase5Block = phase === 0 && tutorialBlocks
    ? (currentBlock === 4 ? tutorialBlocks.block5 : null)
    : phase5Blocks[currentBlock];
  const enabledTrackIndices = useMemo(() => [0], []);
  const phase4EnabledTrackIndices = useMemo(
    () => phase4Block?.trackNames.map((_, i) => i) ?? [],
    [phase4Block],
  );

  const handleSimilarity = (v: number) => {
    if (phase === 0) {
      setTutorialPhase1Answer((prev) => ({ ...prev, similarity: v }));
      return;
    }
    setAnswers((prev) => {
      const next = [...prev];
      next[currentBlock] = { ...next[currentBlock], similarity: v };
      return next;
    });
  };
  const handleConfidence = (v: number) => {
    if (phase === 0) {
      setTutorialPhase1Answer((prev) => ({ ...prev, confidence: v }));
      return;
    }
    setAnswers((prev) => {
      const next = [...prev];
      next[currentBlock] = { ...next[currentBlock], confidence: v };
      return next;
    });
  };

  const handlePrevious = () => {
    if (phase === 0 && currentBlock > 0) {
      setCurrentBlock((c) => c - 1);
      setTutorialPhase3HoveredBead(null);
    }
  };

  const handleNext = () => {
    if (phase === 0) {
      if (currentBlock < NUM_TUTORIAL_QUESTIONS - 1) {
        setCurrentBlock((c) => c + 1);
        setTutorialPhase3HoveredBead(null);
      } else {
        setPhase(1);
        setCurrentBlock(0);
      }
      return;
    }
    // Validate: current question must be answered before proceeding
    if (phase === 1) {
      if (currentBlock === 0 && !phase1Q1TimerStarted) {
        window.alert("Please click Start to begin the timer before proceeding.");
        return;
      }
      const a = answers[currentBlock];
      if (a == null || a.similarity == null || a.confidence == null) {
        window.alert("Please complete both similarity and confidence ratings for this question before proceeding.");
        return;
      }
    } else if (phase === 2) {
      if (phase2Answers[currentBlock] == null) {
        window.alert("Please select your answer for this question before proceeding.");
        return;
      }
    } else if (phase === 3) {
      if (phase3Answers[currentBlock] == null) {
        window.alert("Please click the corresponding bead on the 3D chromosome to complete this question before proceeding.");
        return;
      }
    } else if (phase === 4) {
      if (phase4Answers[currentBlock] == null) {
        window.alert("Please complete the confidence rating for this question before proceeding.");
        return;
      }
    } else if (phase === 5) {
      const ans = phase5AnswersRef.current[currentBlock];
      const conf = phase5ConfidenceRef.current[currentBlock];
      if (ans == null || conf == null) {
        window.alert("Please complete both value selection and confidence rating for this question before proceeding.");
        return;
      }
    }

    const timeSpentMs = Date.now() - questionStartTimeRef.current;
    questionStartTimeRef.current = Date.now();
    setGamma(1); // Reset gamma when moving to next question

    if (phase === 1) {
      setPhase1Times((prev) => [...prev, timeSpentMs]);
      console.log(`[Phase 1] Question ${currentBlock + 1} completed in ${(timeSpentMs / 1000).toFixed(2)} seconds`);
      if (currentBlock < NUM_QUESTIONS - 1) {
        setCurrentBlock((c) => c + 1);
      } else {
        setPhase(2);
        setCurrentBlock(0);
      }
    } else if (phase === 2) {
      if (phase2Block && phase2Results.length <= currentBlock) {
        const correct = phase2Answers[currentBlock] === phase2Block.correctAnswer;
        console.log(`[Phase 2] Question ${currentBlock + 1} completed in ${(timeSpentMs / 1000).toFixed(2)} seconds | Your answer "${phase2Answers[currentBlock]}", correct answer "${phase2Block.correctAnswer}" → ${correct ? "✓ Correct" : "✗ Incorrect"}`);
        setPhase2Results((prev) => [
          ...prev,
          {
            correctAnswer: phase2Block.correctAnswer,
            userAnswer: phase2Answers[currentBlock],
            correct,
            timeSpentMs,
          },
        ]);
      }
      if (currentBlock < NUM_PHASE2_QUESTIONS - 1) {
        setCurrentBlock((c) => c + 1);
      } else {
        setPhase(3);
        setCurrentBlock(0);
        setPhase3HoveredBead(null);
      }
    } else if (phase === 3) {
      if (phase3Block && phase3Results.length <= currentBlock) {
        const userIdx = phase3Answers[currentBlock];
        const targetIdx = phase3Block.highlightedBeadIndex;
        const correct = userIdx != null && Math.abs(userIdx - targetIdx) <= 1;
        console.log(`[Phase 3] Question ${currentBlock + 1} completed in ${(timeSpentMs / 1000).toFixed(2)} seconds | Your selected bead index ${userIdx}, correct bead index ${targetIdx} (tolerance ±1) → ${correct ? "✓ Correct" : "✗ Incorrect"}`);
        setPhase3Results((prev) => [
          ...prev,
          {
            correctBeadIndex: phase3Block.highlightedBeadIndex,
            userBeadIndex: phase3Answers[currentBlock],
            correct,
            timeSpentMs,
          },
        ]);
      }
      if (currentBlock < NUM_PHASE3_QUESTIONS - 1) {
        setCurrentBlock((c) => c + 1);
        setPhase3HoveredBead(null);
      } else {
        setPhase(4);
        setCurrentBlock(0);
      }
    } else if (phase === 4) {
      if (phase4Block && phase4Results.length <= currentBlock) {
        console.log(`[Phase 4] Question ${currentBlock + 1} completed in ${(timeSpentMs / 1000).toFixed(2)} seconds | Confidence: ${phase4Answers[currentBlock]}`);
        setPhase4Results((prev) => [
          ...prev,
          {
            userAnswer: phase4Answers[currentBlock],
            timeSpentMs,
          },
        ]);
      }
      if (currentBlock < NUM_PHASE4_QUESTIONS - 1) {
        setCurrentBlock((c) => c + 1);
      } else {
        setPhase(5);
        setCurrentBlock(0);
      }
    } else {
      let lastPhase5Result: { correctAnswer: number; userAnswer: number | null; correct: boolean; timeSpentMs: number; confidence: number | null } | null = null;
      if (phase5Block && phase5Results.length <= currentBlock) {
        const userAns = phase5AnswersRef.current[currentBlock];
        const correct = userAns === phase5Block.correctAnswer;
        const conf = phase5ConfidenceRef.current[currentBlock];
        lastPhase5Result = {
          correctAnswer: phase5Block.correctAnswer,
          userAnswer: userAns,
          correct,
          timeSpentMs,
          confidence: conf,
        };
        console.log(`[Phase 5] Question ${currentBlock + 1} completed in ${(timeSpentMs / 1000).toFixed(2)} seconds | Your answer ${userAns}, correct ${phase5Block.correctAnswer}, confidence ${conf} → ${correct ? "✓" : "✗"}`);
        setPhase5Results((prev) => [...prev, lastPhase5Result!]);
      }
      if (currentBlock < NUM_PHASE5_QUESTIONS - 1) {
        setCurrentBlock((c) => c + 1);
      } else {
        // All questions complete — export results to file
        const phase5Full = lastPhase5Result != null ? [...phase5Results, lastPhase5Result] : phase5Results;
        downloadResultsFile({
          answers,
          phase1Times,
          phase2Results,
          phase3Results,
          phase4Results,
          phase5Results: phase5Full,
        });
      }
    }
  };

  useEffect(() => {
    if (phase1Times.length > 0) {
      (window as unknown as { __phase1Times?: number[] }).__phase1Times = phase1Times;
    }
  }, [phase1Times]);

  useEffect(() => {
    if (phase2Results.length > 0) {
      (window as unknown as { __phase2Results?: unknown }).__phase2Results = phase2Results;
    }
  }, [phase2Results]);

  useEffect(() => {
    if (phase3Results.length > 0) {
      (window as unknown as { __phase3Results?: unknown }).__phase3Results = phase3Results;
    }
  }, [phase3Results]);

  useEffect(() => {
    if (phase4Results.length > 0) {
      (window as unknown as { __phase4Results?: unknown }).__phase4Results = phase4Results;
    }
  }, [phase4Results]);

  useEffect(() => {
    if (phase5Results.length > 0) {
      (window as unknown as { __phase5Results?: unknown }).__phase5Results = phase5Results;
    }
  }, [phase5Results]);

  useEffect(() => {
    // Don't start timer for Phase 1 Q1 until user clicks Start
    if (phase === 1 && currentBlock === 0 && !phase1Q1TimerStarted) return;
    questionStartTimeRef.current = Date.now();
  }, [phase, currentBlock, phase1Q1TimerStarted]);

  useEffect(() => {
    const hovered = phase === 0 ? tutorialPhase3HoveredBead : phase3HoveredBead;
    const inPhase3 = phase === 3 || (phase === 0 && currentBlock === 2);
    if (inPhase3 && hovered != null) {
      document.body.style.cursor = "pointer";
    } else {
      document.body.style.cursor = "auto";
    }
    return () => {
      document.body.style.cursor = "auto";
    };
  }, [phase, currentBlock, phase3HoveredBead, tutorialPhase3HoveredBead]);

  const handlePhase2Answer = (label: string) => {
    if (phase === 0) {
      setTutorialPhase2Answer(label);
      return;
    }
    setPhase2Answers((prev) => {
      const next = [...prev];
      next[currentBlock] = label;
      return next;
    });
  };

  const handlePhase4Confidence = (v: number) => {
    if (phase === 0) {
      setTutorialPhase4Answer(v);
      return;
    }
    setPhase4Answers((prev) => {
      const next = [...prev];
      next[currentBlock] = v;
      return next;
    });
  };

  const handlePhase5Answer = (v: number) => {
    if (phase === 0) {
      setTutorialPhase5Answer(v);
      return;
    }
    const next = [...phase5Answers];
    next[currentBlock] = v;
    phase5AnswersRef.current = next;
    setPhase5Answers(next);
  };

  const handlePhase5Confidence = (v: number) => {
    if (phase === 0) {
      setTutorialPhase5Confidence(v);
      return;
    }
    const next = [...phase5Confidence];
    next[currentBlock] = v;
    phase5ConfidenceRef.current = next;
    setPhase5Confidence(next);
  };

  const handlePhase3BeadClick = (beadIndex: number) => {
    if (phase === 0) {
      setTutorialPhase3Answer(beadIndex);
      return;
    }
    setPhase3Answers((prev) => {
      const next = [...prev];
      next[currentBlock] = beadIndex;
      return next;
    });
  };

  return (
    <div style={rootStyle}>
      {/* ── Header ── */}
      <div style={headerStyle}>
        <div style={progressStyle}>
          <span style={progressLabelStyle}>
            {phase === 0
              ? `Tutorial — Question ${currentBlock + 1} / ${NUM_TUTORIAL_QUESTIONS}`
              : `${phase === 1 ? "Phase 1" : phase === 2 ? "Phase 2" : phase === 3 ? "Phase 3" : phase === 4 ? "Phase 4" : "Phase 5"} — Question ${currentBlock + 1} / ${phase === 1 ? NUM_QUESTIONS : phase === 2 ? NUM_PHASE2_QUESTIONS : phase === 3 ? NUM_PHASE3_QUESTIONS : phase === 4 ? NUM_PHASE4_QUESTIONS : NUM_PHASE5_QUESTIONS}`}
          </span>
          <div style={progressBarTrackStyle}>
            <div
              style={{
                ...progressBarFillStyle,
                width: `${
                  (phase === 0
                    ? ((currentBlock + 1) / NUM_TUTORIAL_QUESTIONS) * 100
                    : ((currentBlock + 1) /
                        (phase === 1 ? NUM_QUESTIONS : phase === 2 ? NUM_PHASE2_QUESTIONS : phase === 3 ? NUM_PHASE3_QUESTIONS : phase === 4 ? NUM_PHASE4_QUESTIONS : NUM_PHASE5_QUESTIONS)) *
                      100)
                }%`,
              }}
            />
          </div>
        </div>
      </div>

      {/* ── Body: split panel ── */}
      <div style={bodyStyle}>
        {loading && (
          <div style={fullOverlayStyle}>
            <div style={spinnerStyle} />
            <span style={{ color: "rgba(255,255,255,0.7)", marginTop: 12, fontSize: 14 }}>
              Loading data…
            </span>
          </div>
        )}
        {error && (
          <div style={fullOverlayStyle}>
            <span style={{ color: "#ff6b6b", fontSize: 14 }}>{error}</span>
          </div>
        )}

        {!loading && !error && (phase === 1 || phase === 0) && block && (
          <>
            {/* Left: D3 line chart */}
            <div style={panelStyle}>
              <div style={panelLabelStyle}>1D Signal Track</div>
              <div style={chartContainerStyle}>
                <LineChart
                  values={block.trackValues}
                  trackName={block.trackName}
                  color={getTrackColor(0)}
                />
              </div>
            </div>

            {/* Divider */}
            <div style={dividerStyle} />

            {/* Right: 3D view */}
            <div style={panelStyle}>
              <div style={panelLabelStyle}>3D Chromatin Structure</div>
              <div style={canvasWrapperStyle}>
                <Canvas
                  camera={{ position: [0, 0, 500], fov: 60, near: 0.1, far: 10000 }}
                  style={{ width: "100%", height: "100%", background: "#060f1a" }}
                >
                  <ambientLight intensity={0.6} />
                  <directionalLight position={[100, 100, 100]} intensity={0.8} />
                  {block.beads.length > 1 && (
                    <ChromosomePipeline
                      beads={block.beads}
                      enabledTrackIndices={enabledTrackIndices}
                      trackNames={[block.trackName]}
                      highlightStartEnd
                      gamma={gamma}
                    />
                  )}
                  <OrbitControls enableZoom enablePan enableRotate />
                </Canvas>
                <div style={legendStyle}>
                  <div style={gammaControlStyle}>
                    <span style={{ whiteSpace: "nowrap", fontSize: 12 }}>Gamma {gamma}</span>
                    <input
                      type="range"
                      min={1}
                      max={20}
                      step={0.1}
                      value={gamma}
                      onChange={(e) => setGamma(Number(e.target.value))}
                      style={gammaSliderStyle}
                    />
                  </div>
                  <div style={legendItemStyle}>
                    <span style={{ ...legendDotStyle, background: "#22c55e" }} />
                    Start
                  </div>
                  <div style={legendItemStyle}>
                    <span style={{ ...legendDotStyle, background: "#ef4444" }} />
                    End
                  </div>
                </div>
              </div>
            </div>
          </>
        )}

        {!loading && !error && (phase === 2 || phase === 0) && phase2Block && (
          <>
            {/* Left: D3 line chart with peak labels */}
            <div style={panelStyle}>
              <div style={panelLabelStyle}>1D Signal Track</div>
              <div style={chartContainerStyle}>
                <LineChartWithPeaks
                  values={phase2Block.trackValues}
                  trackName={phase2Block.trackName}
                  peaks={phase2Block.peaks}
                  color={getTrackColor(0)}
                />
              </div>
            </div>

            {/* Divider */}
            <div style={dividerStyle} />

            {/* Right: 3D view with one peak bead highlighted */}
            <div style={panelStyle}>
              <div style={panelLabelStyle}>3D Chromatin Structure</div>
              <div style={canvasWrapperStyle}>
                <Canvas
                  camera={{ position: [0, 0, 500], fov: 60, near: 0.1, far: 10000 }}
                  style={{ width: "100%", height: "100%", background: "#060f1a" }}
                >
                  <ambientLight intensity={0.6} />
                  <directionalLight position={[100, 100, 100]} intensity={0.8} />
                  {phase2Block.beads.length > 1 && (
                    <ChromosomePipeline
                      beads={phase2Block.beads}
                      enabledTrackIndices={enabledTrackIndices}
                      trackNames={[phase2Block.trackName]}
                      highlightStartEnd
                      highlightedBeadIndex={phase2Block.highlightedBeadIndex}
                      gamma={gamma}
                    />
                  )}
                  <OrbitControls enableZoom enablePan enableRotate />
                </Canvas>
                <div style={legendStyle}>
                  <div style={gammaControlStyle}>
                    <span style={{ whiteSpace: "nowrap", fontSize: 12 }}>Gamma {gamma}</span>
                    <input
                      type="range"
                      min={1}
                      max={20}
                      step={0.1}
                      value={gamma}
                      onChange={(e) => setGamma(Number(e.target.value))}
                      style={gammaSliderStyle}
                    />
                  </div>
                  <div style={legendItemStyle}>
                    <span style={{ ...legendDotStyle, background: "#22c55e" }} />
                    Start
                  </div>
                  <div style={legendItemStyle}>
                    <span style={{ ...legendDotStyle, background: "#ef4444" }} />
                    End
                  </div>
                  <div style={legendItemStyle}>
                    <span style={{ ...legendDotStyle, background: "#00d4ff" }} />
                    Highlighted
                  </div>
                </div>
              </div>
            </div>
          </>
        )}

        {!loading && !error && (phase === 3 || phase === 0) && phase3Block && (
          <>
            {/* Left: D3 line chart with single peak highlight */}
            <div style={panelStyle}>
              <div style={panelLabelStyle}>1D Signal Track</div>
              <div style={chartContainerStyle}>
                <LineChartWithSinglePeakHighlight
                  values={phase3Block.trackValues}
                  trackName={phase3Block.trackName}
                  peak={phase3Block.peaks[phase3Block.highlightedPeakIndex]}
                  color={getTrackColor(0)}
                />
              </div>
            </div>

            {/* Divider */}
            <div style={dividerStyle} />

            {/* Right: 3D view with interactive beads */}
            <div style={panelStyle}>
              <div style={panelLabelStyle}>3D Chromatin Structure</div>
              <div style={canvasWrapperStyle}>
                <Canvas
                  camera={{ position: [0, 0, 500], fov: 60, near: 0.1, far: 10000 }}
                  style={{ width: "100%", height: "100%", background: "#060f1a" }}
                >
                  <ambientLight intensity={0.6} />
                  <directionalLight position={[100, 100, 100]} intensity={0.8} />
                    {phase3Block.beads.length > 1 && (
                    <ChromosomePipeline
                      beads={phase3Block.beads}
                      enabledTrackIndices={enabledTrackIndices}
                      trackNames={[phase3Block.trackName]}
                      highlightStartEnd
                      interactiveMode
                      hoveredBeadIndex={phase === 0 ? tutorialPhase3HoveredBead : phase3HoveredBead}
                      selectedBeadIndex={phase === 0 ? tutorialPhase3Answer : phase3Answers[currentBlock]}
                      onBeadHover={phase === 0 ? setTutorialPhase3HoveredBead : setPhase3HoveredBead}
                      onBeadClick={handlePhase3BeadClick}
                      gamma={gamma}
                    />
                  )}
                  <OrbitControls enableZoom enablePan enableRotate />
                </Canvas>
                <div style={legendStyle}>
                  <div style={gammaControlStyle}>
                    <span style={{ whiteSpace: "nowrap", fontSize: 12 }}>Gamma {gamma}</span>
                    <input
                      type="range"
                      min={1}
                      max={20}
                      step={0.1}
                      value={gamma}
                      onChange={(e) => setGamma(Number(e.target.value))}
                      style={gammaSliderStyle}
                    />
                  </div>
                  <div style={legendItemStyle}>
                    <span style={{ ...legendDotStyle, background: "#22c55e" }} />
                    Start
                  </div>
                  <div style={legendItemStyle}>
                    <span style={{ ...legendDotStyle, background: "#ef4444" }} />
                    End
                  </div>
                  <div style={legendItemStyle}>
                    <span style={{ ...legendDotStyle, background: "#f59e0b" }} />
                    Selected bead
                  </div>
                </div>
              </div>
            </div>
          </>
        )}

        {!loading && !error && (phase === 4 || phase === 0) && phase4Block && (
          <>
            {/* Left: Multi-track line charts (one per track, column) — compact so all fit without scroll */}
            <div style={panelStyle}>
              <div style={panelLabelStyle}>1D Signal Tracks</div>
              <div style={chartContainerStyle}>
                <MultiTrackLineChartColumn tracks={phase4Block.tracks} compact />
              </div>
            </div>

            {/* Divider */}
            <div style={dividerStyle} />

            {/* Right: 3D view with multiple tracks */}
            <div style={panelStyle}>
              <div style={{ ...panelLabelStyle, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span>3D Chromatin Structure</span>
                <span style={{ fontSize: 11, fontWeight: 500, color: "rgba(255,255,255,0.5)" }}>
                  {phase4Block.trackNames.length} track{phase4Block.trackNames.length !== 1 ? "s" : ""}
                </span>
              </div>
              <div style={canvasWrapperStyle}>
                <Canvas
                  camera={{ position: [0, 0, 500], fov: 60, near: 0.1, far: 10000 }}
                  style={{ width: "100%", height: "100%", background: "#060f1a" }}
                >
                  <ambientLight intensity={0.6} />
                  <directionalLight position={[100, 100, 100]} intensity={0.8} />
                  {phase4Block.beads.length > 1 && phase4EnabledTrackIndices.length > 0 && (
                    <ChromosomePipeline
                      beads={phase4Block.beads}
                      enabledTrackIndices={phase4EnabledTrackIndices}
                      trackNames={phase4Block.trackNames}
                      highlightStartEnd
                      gamma={gamma}
                    />
                  )}
                  <OrbitControls enableZoom enablePan enableRotate />
                </Canvas>
                <div style={legendStyle}>
                  <div style={gammaControlStyle}>
                    <span style={{ whiteSpace: "nowrap", fontSize: 12 }}>Gamma {gamma}</span>
                    <input
                      type="range"
                      min={1}
                      max={20}
                      step={0.1}
                      value={gamma}
                      onChange={(e) => setGamma(Number(e.target.value))}
                      style={gammaSliderStyle}
                    />
                  </div>
                  <div style={legendItemStyle}>
                    <span style={{ ...legendDotStyle, background: "#22c55e" }} />
                    Start
                  </div>
                  <div style={legendItemStyle}>
                    <span style={{ ...legendDotStyle, background: "#ef4444" }} />
                    End
                  </div>
                </div>
              </div>
            </div>
          </>
        )}

        {!loading && !error && (phase === 5 || phase === 0) && phase5Block && (
          <>
            {/* Left panel (1D line charts) hidden for Phase 5
            <div style={panelStyle}>
              <div style={panelLabelStyle}>1D Signal Tracks</div>
              <div style={chartContainerStyle}>
                <MultiTrackLineChartColumn tracks={phase5Block.tracks} compact />
              </div>
            </div>
            <div style={dividerStyle} /> */}
            <div style={panelStyle}>
              <div style={{ ...panelLabelStyle, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span>3D Chromatin Structure</span>
                <span style={{ fontSize: 11, fontWeight: 500, color: "rgba(255,255,255,0.5)" }}>
                  {phase5Block.trackNames.length} track{phase5Block.trackNames.length !== 1 ? "s" : ""}
                </span>
              </div>
              <div style={canvasWrapperStyle}>
                <Canvas
                  camera={{ position: [0, 0, 500], fov: 60, near: 0.1, far: 10000 }}
                  style={{ width: "100%", height: "100%", background: "#060f1a" }}
                >
                  <ambientLight intensity={0.6} />
                  <directionalLight position={[100, 100, 100]} intensity={0.8} />
                  {phase5Block.beads.length > 1 && phase5Block.trackNames.length > 0 && (
                    <ChromosomePipeline
                      beads={phase5Block.beads}
                      enabledTrackIndices={phase5Block.trackNames.map((_, i) => i)}
                      trackNames={phase5Block.trackNames}
                      highlightStartEnd
                      gamma={gamma}
                    />
                  )}
                  <OrbitControls enableZoom enablePan enableRotate />
                </Canvas>
                <div style={legendStyle}>
                  <div style={gammaControlStyle}>
                    <span style={{ whiteSpace: "nowrap", fontSize: 12 }}>Gamma {gamma}</span>
                    <input
                      type="range"
                      min={1}
                      max={20}
                      step={0.1}
                      value={gamma}
                      onChange={(e) => setGamma(Number(e.target.value))}
                      style={gammaSliderStyle}
                    />
                  </div>
                  <div style={legendItemStyle}>
                    <span style={{ ...legendDotStyle, background: "#22c55e" }} />
                    Start
                  </div>
                  <div style={legendItemStyle}>
                    <span style={{ ...legendDotStyle, background: "#ef4444" }} />
                    End
                  </div>
                </div>
              </div>
            </div>
          </>
        )}
      </div>

      {/* ── Questions panel ── */}
      {!loading && !error && (phase === 1 || phase === 0) && block && (
        <div key={`p1-b${currentBlock}`} style={questionsPanelStyle}>
          <div style={questionBlockStyle}>
            <div style={questionLabelStyle}>
              Q1. How similar are the peak distribution and pattern of the 1D track when mapped onto the 3D structure?
            </div>
            <RatingScale
              value={(phase === 0 ? tutorialPhase1Answer.similarity : answers[currentBlock]?.similarity) ?? null}
              onChange={handleSimilarity}
              labelLow="1 = most dissimilar"
              labelHigh="5 = very similar"
            />
          </div>
          <div style={questionBlockStyle}>
            <div style={questionLabelStyle}>
              Q2. How confident are you about your result?
            </div>
            <RatingScale
              value={(phase === 0 ? tutorialPhase1Answer.confidence : answers[currentBlock]?.confidence) ?? null}
              onChange={handleConfidence}
              labelLow="1 = least confident"
              labelHigh="5 = very confident"
            />
          </div>
        </div>
      )}

      {/* ── Phase 2: Multiple choice ── */}
      {!loading && !error && (phase === 2 || phase === 0) && phase2Block && (
        <div key={`p2-b${currentBlock}`} style={questionsPanelStyle}>
          <div style={{ ...questionBlockStyle, flex: 1 }}>
            <div style={questionLabelStyle}>
              Which peak has been highlighted?
            </div>
            <div style={peakOptionsStyle}>
              {phase2Block.peaks.map((p) => (
                <button
                  key={p.label}
                  type="button"
                  onClick={() => handlePhase2Answer(p.label)}
                  style={{
                    ...peakOptionBtnStyle,
                    ...((phase === 0 ? tutorialPhase2Answer : phase2Answers[currentBlock]) === p.label
                      ? peakOptionBtnActiveStyle
                      : {}),
                  }}
                >
                  {p.label}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ── Phase 3: Click bead (no options) ── */}
      {!loading && !error && (phase === 3 || phase === 0) && phase3Block && (
        <div key={`p3-b${currentBlock}`} style={questionsPanelStyle}>
          <div style={{ ...questionBlockStyle, flex: 1 }}>
            <div style={questionLabelStyle}>
              Select the bead in 3D that corresponds to the highlighted peak. Hover to preview, click to confirm.
              {(phase === 0 ? tutorialPhase3Answer : phase3Answers[currentBlock]) != null && (
                <span style={{ marginLeft: 8, color: "#22c55e" }}>✓</span>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ── Phase 4: Confidence rating ── */}
      {!loading && !error && (phase === 4 || phase === 0) && phase4Block && (
        <div key={`p4-b${currentBlock}`} style={questionsPanelStyle}>
          <div style={{ ...questionBlockStyle, flex: 1 }}>
            <div style={questionLabelStyle}>
              How confident are you in recognizing the changing patterns of all the tracks here?
            </div>
            <RatingScale
              value={(phase === 0 ? tutorialPhase4Answer : phase4Answers[currentBlock]) ?? null}
              onChange={handlePhase4Confidence}
              labelLow="1 = least confident"
              labelHigh="5 = very confident"
            />
          </div>
        </div>
      )}

      {/* ── Phase 5: Common peak count + confidence ── */}
      {!loading && !error && (phase === 5 || phase === 0) && phase5Block && (
        <div key={`p5-b${currentBlock}`} style={questionsPanelStyle}>
          <div style={{ ...questionBlockStyle, flex: 1 }}>
            <div style={questionLabelStyle}>
              Q1. How many locations exhibit co-occurring peaks across <strong>ALL</strong> tracks?
            </div>
            <div style={peakOptionsStyle}>
              {phase5Block.options.map((opt) => (
                <button
                  key={opt}
                  type="button"
                  onClick={() => handlePhase5Answer(opt)}
                  style={{
                    ...peakOptionBtnStyle,
                    ...((phase === 0 ? tutorialPhase5Answer : phase5Answers[currentBlock]) === opt ? peakOptionBtnActiveStyle : {}),
                  }}
                >
                  {opt}
                </button>
              ))}
            </div>
          </div>
          <div style={questionBlockStyle}>
            <div style={questionLabelStyle}>
              Q2. How confident are you about your result?
            </div>
            <RatingScale
              value={(phase === 0 ? tutorialPhase5Confidence : phase5Confidence[currentBlock]) ?? null}
              onChange={handlePhase5Confidence}
              labelLow="1 = least confident"
              labelHigh="5 = very confident"
            />
          </div>
        </div>
      )}

      {/* ── Footer ── */}
      <div style={{
        ...footerStyle,
        ...((phase === 1 && currentBlock === 0 && !phase1Q1TimerStarted) || (phase === 0 && currentBlock > 0) ? { justifyContent: "space-between" } : {}),
      }}>
        {phase === 1 && currentBlock === 0 && !phase1Q1TimerStarted && (
          <button style={btnStyle} onClick={() => { setPhase1Q1TimerStarted(true); questionStartTimeRef.current = Date.now(); }}>
            Start
          </button>
        )}
        {phase === 0 && currentBlock > 0 && (
          <button style={btnStyle} onClick={handlePrevious}>
            ← Previous
          </button>
        )}
        <button
          style={btnStyle}
          onClick={handleNext}
          disabled={
            phase !== 0 &&
            ((phase === 2 &&
              currentBlock >= NUM_PHASE2_QUESTIONS - 1 &&
              phase2Results.length >= NUM_PHASE2_QUESTIONS) ||
            (phase === 3 &&
              currentBlock >= NUM_PHASE3_QUESTIONS - 1 &&
              phase3Results.length >= NUM_PHASE3_QUESTIONS) ||
            (phase === 4 &&
              currentBlock >= NUM_PHASE4_QUESTIONS - 1 &&
              phase4Results.length >= NUM_PHASE4_QUESTIONS) ||
            (phase === 5 &&
              currentBlock >= NUM_PHASE5_QUESTIONS - 1 &&
              phase5Results.length >= NUM_PHASE5_QUESTIONS))
          }
        >
          {phase === 0 && currentBlock >= NUM_TUTORIAL_QUESTIONS - 1
            ? "Start Phase 1 →"
            : phase === 1 && currentBlock >= NUM_QUESTIONS - 1
              ? "Next phase →"
              : phase === 2 && currentBlock >= NUM_PHASE2_QUESTIONS - 1
                ? "Next phase →"
                : phase === 3 && currentBlock >= NUM_PHASE3_QUESTIONS - 1
                  ? "Next phase →"
                  : phase === 4 && currentBlock >= NUM_PHASE4_QUESTIONS - 1
                    ? "Next phase →"
                    : phase === 5 && currentBlock >= NUM_PHASE5_QUESTIONS - 1
                      ? "Complete"
                      : "Next →"}
        </button>
      </div>
    </div>
  );
}

// ── Styles ─────────────────────────────────────────────────────────────────

const rootStyle: React.CSSProperties = {
  width: "100vw",
  height: "100vh",
  display: "flex",
  flexDirection: "column",
  background: "#0a1929",
  fontFamily: "system-ui, -apple-system, sans-serif",
  color: "#e0e0e0",
  overflow: "hidden",
};

const headerStyle: React.CSSProperties = {
  padding: "16px 32px 12px",
  background: "rgba(255,255,255,0.04)",
  borderBottom: "1px solid rgba(255,255,255,0.08)",
  flexShrink: 0,
};

const progressStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 12,
  marginBottom: 10,
};

const progressLabelStyle: React.CSSProperties = {
  fontSize: 12,
  color: "rgba(255,255,255,0.45)",
  whiteSpace: "nowrap",
};

const progressBarTrackStyle: React.CSSProperties = {
  flex: 1,
  height: 4,
  borderRadius: 2,
  background: "rgba(255,255,255,0.1)",
};

const progressBarFillStyle: React.CSSProperties = {
  height: "100%",
  borderRadius: 2,
  background: "#45b7d1",
  transition: "width 0.4s ease",
};

const bodyStyle: React.CSSProperties = {
  flex: 1,
  display: "flex",
  position: "relative",
  overflow: "hidden",
};

const panelStyle: React.CSSProperties = {
  flex: 1,
  display: "flex",
  flexDirection: "column",
  overflow: "hidden",
};

const panelLabelStyle: React.CSSProperties = {
  padding: "8px 20px",
  fontSize: 11,
  fontWeight: 600,
  letterSpacing: "0.08em",
  textTransform: "uppercase",
  color: "rgba(255,255,255,0.35)",
  background: "rgba(255,255,255,0.03)",
  borderBottom: "1px solid rgba(255,255,255,0.06)",
  flexShrink: 0,
};

const chartContainerStyle: React.CSSProperties = {
  flex: 1,
  padding: "12px 8px 8px",
  overflow: "hidden",
};

const dividerStyle: React.CSSProperties = {
  width: 1,
  background: "rgba(255,255,255,0.1)",
  flexShrink: 0,
};

const canvasWrapperStyle: React.CSSProperties = {
  flex: 1,
  position: "relative",
  overflow: "hidden",
};

const legendStyle: React.CSSProperties = {
  position: "absolute",
  top: 12,
  right: 12,
  display: "flex",
  flexDirection: "column",
  gap: 8,
  padding: "10px 14px",
  background: "rgba(6,15,26,0.85)",
  borderRadius: 8,
  border: "1px solid rgba(255,255,255,0.12)",
  fontSize: 12,
  color: "#e0e0e0",
  zIndex: 10,
};

const legendItemStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 8,
};

const legendDotStyle: React.CSSProperties = {
  width: 10,
  height: 10,
  borderRadius: "50%",
  flexShrink: 0,
};

const gammaControlStyle: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: 4,
};

const gammaSliderStyle: React.CSSProperties = {
  width: 80,
  accentColor: "#45b7d1",
};

const questionsPanelStyle: React.CSSProperties = {
  display: "flex",
  gap: 32,
  padding: "16px 32px",
  background: "rgba(255,255,255,0.03)",
  borderTop: "1px solid rgba(255,255,255,0.08)",
  flexShrink: 0,
};

const questionBlockStyle: React.CSSProperties = {
  flex: 1,
  display: "flex",
  flexDirection: "column",
  gap: 10,
};

const questionLabelStyle: React.CSSProperties = {
  fontSize: 13,
  fontWeight: 500,
  color: "#e0e0e0",
};

const ratingRowStyle: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: 6,
};

const ratingScaleStyle: React.CSSProperties = {
  display: "flex",
  gap: 8,
};

const ratingBtnStyle: React.CSSProperties = {
  width: 40,
  height: 40,
  borderRadius: 8,
  border: "1px solid rgba(255,255,255,0.2)",
  background: "rgba(255,255,255,0.05)",
  color: "#b0bec5",
  fontSize: 15,
  fontWeight: 600,
  cursor: "pointer",
  transition: "all 0.2s",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
};

const ratingBtnActiveStyle: React.CSSProperties = {
  borderColor: "#45b7d1",
  background: "rgba(69,183,209,0.2)",
  color: "#45b7d1",
};

const ratingLabelsStyle: React.CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  fontSize: 11,
  color: "rgba(255,255,255,0.4)",
};

const peakOptionsStyle: React.CSSProperties = {
  display: "flex",
  gap: 12,
  flexWrap: "wrap",
};

const peakOptionBtnStyle: React.CSSProperties = {
  width: 48,
  height: 48,
  borderRadius: 8,
  border: "1px solid rgba(255,255,255,0.2)",
  background: "rgba(255,255,255,0.05)",
  color: "#b0bec5",
  fontSize: 18,
  fontWeight: 700,
  cursor: "pointer",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  transition: "all 0.2s",
};

const peakOptionBtnActiveStyle: React.CSSProperties = {
  borderColor: "#f59e0b",
  background: "rgba(245,158,11,0.2)",
  color: "#f59e0b",
};

const footerStyle: React.CSSProperties = {
  display: "flex",
  justifyContent: "flex-end",
  alignItems: "center",
  padding: "12px 32px",
  background: "rgba(255,255,255,0.04)",
  borderTop: "1px solid rgba(255,255,255,0.08)",
  flexShrink: 0,
};

const btnStyle: React.CSSProperties = {
  padding: "8px 24px",
  borderRadius: 6,
  border: "1px solid rgba(69,183,209,0.5)",
  background: "rgba(69,183,209,0.1)",
  color: "#45b7d1",
  fontSize: 14,
  cursor: "pointer",
  transition: "all 0.2s",
};

const fullOverlayStyle: React.CSSProperties = {
  position: "absolute",
  inset: 0,
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  justifyContent: "center",
  background: "rgba(10,25,41,0.85)",
  zIndex: 20,
};

const spinnerStyle: React.CSSProperties = {
  width: 32,
  height: 32,
  borderRadius: "50%",
  border: "3px solid rgba(69,183,209,0.2)",
  borderTopColor: "#45b7d1",
  animation: "spin 0.8s linear infinite",
};
