import { useEffect, useRef, useState, useMemo, useCallback } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as d3 from "d3";
import {
  ChromosomePipeline,
  parsePositionCsv,
  matchBeadsToTracks,
} from "./ChromosomeTrack3D";
import type { BeadData, TracksJson } from "./ChromosomeTrack3D";

// ── Constants ──────────────────────────────────────────────────────────────

const DUMMY_TRACKS_PATH = "/Data/Tracks/dummy_30tracks_chr8_127200000_127750000.json";
const POSITION_PATH = "/Data/Example/Calu3_chr8_127200000_127750000_original_position.csv";
const REGION_START = 127_200_000;
const BIN_SIZE = 5_000;
const NUM_QUESTIONS = 5;
const NUM_PHASE2_QUESTIONS = 5;
const NUM_PHASE3_QUESTIONS = 5;
const NUM_PHASE4_QUESTIONS = 7;

const PHASE4_TRACK_COLORS = [
  "#ff6b6b", "#bf812d", "#45b7d1",
  "#f9ca24", "#6c5ce7", "#00d2d3",
  "#ff9ff3", "#54a0ff", "#a29bfe",
];
const PEAK_MIN_HEIGHT = 0.25;
const PEAK_MIN_DISTANCE = 3;
const PEAK_LABELS = "ABCDEF".split("");

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

// ── Multi-track line charts (one per track, column layout, for Phase 4) ─────

const PHASE4_CHART_MIN_HEIGHT = 150;

function MultiTrackLineChartColumn({
  tracks,
}: {
  tracks: Phase4TrackData[];
}) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 2,
        height: "100%",
        minHeight: 0,
        overflow: "auto",
        padding: "4px",
      }}
    >
      {tracks.map((t, i) => (
        <div
          key={t.trackName}
          style={{
            flex: `0 0 ${PHASE4_CHART_MIN_HEIGHT}px`,
            height: PHASE4_CHART_MIN_HEIGHT,
            minHeight: PHASE4_CHART_MIN_HEIGHT,
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
}: {
  values: number[];
  trackName: string;
  peaks: Peak[];
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

    const area = d3
      .area<number>()
      .x((_, i) => xScale(genomicPositions[i]))
      .y0(height)
      .y1((d) => yScale(d))
      .curve(d3.curveCatmullRom.alpha(0.5));
    g.append("path").datum(values).attr("fill", "rgba(69,183,209,0.15)").attr("d", area);

    const line = d3
      .line<number>()
      .x((_, i) => xScale(genomicPositions[i]))
      .y((d) => yScale(d))
      .curve(d3.curveCatmullRom.alpha(0.5));
    g.append("path").datum(values).attr("fill", "none").attr("stroke", "#45b7d1").attr("stroke-width", 2).attr("d", line);

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
  }, [values, trackName, peaks]);

  return <svg ref={svgRef} style={{ width: "100%", height: "100%", display: "block" }} />;
}

// ── D3 Line Chart with single peak highlight (for Phase 3) ───────────────────

function LineChartWithSinglePeakHighlight({
  values,
  trackName,
  peak,
}: {
  values: number[];
  trackName: string;
  peak: Peak;
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

    const area = d3
      .area<number>()
      .x((_, i) => xScale(genomicPositions[i]))
      .y0(height)
      .y1((d) => yScale(d))
      .curve(d3.curveCatmullRom.alpha(0.5));
    g.append("path").datum(values).attr("fill", "rgba(69,183,209,0.15)").attr("d", area);

    const line = d3
      .line<number>()
      .x((_, i) => xScale(genomicPositions[i]))
      .y((d) => yScale(d))
      .curve(d3.curveCatmullRom.alpha(0.5));
    g.append("path").datum(values).attr("fill", "none").attr("stroke", "#45b7d1").attr("stroke-width", 2).attr("d", line);

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
  }, [values, trackName, peak]);

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

export default function QuestionOnePage() {
  const [phase, setPhase] = useState<1 | 2 | 3 | 4>(1);
  const [blocks, setBlocks] = useState<BlockData[]>([]);
  const [phase2Blocks, setPhase2Blocks] = useState<Phase2BlockData[]>([]);
  const [phase3Blocks, setPhase3Blocks] = useState<Phase2BlockData[]>([]);
  const [phase4Blocks, setPhase4Blocks] = useState<Phase4BlockData[]>([]);
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
  const [phase1Times, setPhase1Times] = useState<number[]>([]);
  const [gamma, setGamma] = useState(1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const questionStartTimeRef = useRef<number>(Date.now());

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
          const numTracks = i + 2;
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

        if (!cancelled) {
          setBlocks(built);
          setPhase2Blocks(builtPhase2);
          setPhase3Blocks(builtPhase3);
          setPhase4Blocks(builtPhase4);
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

  const block = blocks[currentBlock];
  const phase2Block = phase2Blocks[currentBlock];
  const phase3Block = phase3Blocks[currentBlock];
  const phase4Block = phase4Blocks[currentBlock];
  const enabledTrackIndices = useMemo(() => [0], []);
  const phase4EnabledTrackIndices = useMemo(
    () => phase4Block?.trackNames.map((_, i) => i) ?? [],
    [phase4Block],
  );

  const handleSimilarity = (v: number) => {
    setAnswers((prev) => {
      const next = [...prev];
      next[currentBlock] = { ...next[currentBlock], similarity: v };
      return next;
    });
  };
  const handleConfidence = (v: number) => {
    setAnswers((prev) => {
      const next = [...prev];
      next[currentBlock] = { ...next[currentBlock], confidence: v };
      return next;
    });
  };

  const handlePrevious = () => {
    if (phase === 1 && currentBlock > 0) {
      setCurrentBlock((c) => c - 1);
    } else if (phase === 2 && currentBlock > 0) {
      setCurrentBlock((c) => c - 1);
    } else if (phase === 2 && currentBlock === 0) {
      setPhase(1);
      setCurrentBlock(NUM_QUESTIONS - 1);
    } else if (phase === 3 && currentBlock > 0) {
      setCurrentBlock((c) => c - 1);
      setPhase3HoveredBead(null);
    } else if (phase === 3 && currentBlock === 0) {
      setPhase(2);
      setCurrentBlock(NUM_PHASE2_QUESTIONS - 1);
      setPhase3HoveredBead(null);
    } else if (phase === 4 && currentBlock > 0) {
      setCurrentBlock((c) => c - 1);
    } else if (phase === 4 && currentBlock === 0) {
      setPhase(3);
      setCurrentBlock(NUM_PHASE3_QUESTIONS - 1);
      setPhase3HoveredBead(null);
    }
  };

  const handleNext = () => {
    const timeSpentMs = Date.now() - questionStartTimeRef.current;
    questionStartTimeRef.current = Date.now();

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
    } else {
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
    questionStartTimeRef.current = Date.now();
  }, [phase, currentBlock]);

  useEffect(() => {
    if (phase === 3 && phase3HoveredBead != null) {
      document.body.style.cursor = "pointer";
    } else {
      document.body.style.cursor = "auto";
    }
    return () => {
      document.body.style.cursor = "auto";
    };
  }, [phase, phase3HoveredBead]);

  const handlePhase2Answer = (label: string) => {
    setPhase2Answers((prev) => {
      const next = [...prev];
      next[currentBlock] = label;
      return next;
    });
  };

  const handlePhase4Confidence = (v: number) => {
    setPhase4Answers((prev) => {
      const next = [...prev];
      next[currentBlock] = v;
      return next;
    });
  };

  const handlePhase3BeadClick = (beadIndex: number) => {
    if (phase3Answers[currentBlock] != null) return;
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
            {phase === 1 ? "Phase 1" : phase === 2 ? "Phase 2" : phase === 3 ? "Phase 3" : "Phase 4"} — Question {currentBlock + 1} /{" "}
            {phase === 1 ? NUM_QUESTIONS : phase === 2 ? NUM_PHASE2_QUESTIONS : phase === 3 ? NUM_PHASE3_QUESTIONS : NUM_PHASE4_QUESTIONS}
          </span>
          <div style={progressBarTrackStyle}>
            <div
              style={{
                ...progressBarFillStyle,
                width: `${
                  ((currentBlock + 1) /
                    (phase === 1 ? NUM_QUESTIONS : phase === 2 ? NUM_PHASE2_QUESTIONS : phase === 3 ? NUM_PHASE3_QUESTIONS : NUM_PHASE4_QUESTIONS)) *
                  100
                }%`,
              }}
            />
          </div>
        </div>
        <h2 style={questionTitleStyle}>
          {phase === 1
            ? "How similar is the 1D track to its 3D mapping?"
            : phase === 2
              ? "Which peak has been highlighted?"
              : phase === 3
                ? "Click on the bead that corresponds to the highlighted peak"
                : "How confident are you in recognizing the changing patterns of all the tracks here?"}
        </h2>
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

        {!loading && !error && phase === 1 && block && (
          <>
            {/* Left: D3 line chart */}
            <div style={panelStyle}>
              <div style={panelLabelStyle}>1D Signal Track</div>
              <div style={chartContainerStyle}>
                <LineChart values={block.trackValues} trackName={block.trackName} />
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

        {!loading && !error && phase === 2 && phase2Block && (
          <>
            {/* Left: D3 line chart with peak labels */}
            <div style={panelStyle}>
              <div style={panelLabelStyle}>1D Signal Track</div>
              <div style={chartContainerStyle}>
                <LineChartWithPeaks
                  values={phase2Block.trackValues}
                  trackName={phase2Block.trackName}
                  peaks={phase2Block.peaks}
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

        {!loading && !error && phase === 3 && phase3Block && (
          <>
            {/* Left: D3 line chart with single peak highlight */}
            <div style={panelStyle}>
              <div style={panelLabelStyle}>1D Signal Track</div>
              <div style={chartContainerStyle}>
                <LineChartWithSinglePeakHighlight
                  values={phase3Block.trackValues}
                  trackName={phase3Block.trackName}
                  peak={phase3Block.peaks[phase3Block.highlightedPeakIndex]}
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
                      hoveredBeadIndex={phase3HoveredBead}
                      selectedBeadIndex={phase3Answers[currentBlock]}
                      onBeadHover={setPhase3HoveredBead}
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

        {!loading && !error && phase === 4 && phase4Block && (
          <>
            {/* Left: Multi-track line charts (one per track, column) */}
            <div style={panelStyle}>
              <div style={panelLabelStyle}>1D Signal Tracks</div>
              <div style={chartContainerStyle}>
                <MultiTrackLineChartColumn tracks={phase4Block.tracks} />
              </div>
            </div>

            {/* Divider */}
            <div style={dividerStyle} />

            {/* Right: 3D view with multiple tracks */}
            <div style={panelStyle}>
              <div style={panelLabelStyle}>3D Chromatin Structure</div>
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
      </div>

      {/* ── Questions panel ── */}
      {!loading && !error && phase === 1 && block && (
        <div key={`p1-b${currentBlock}`} style={questionsPanelStyle}>
          <div style={questionBlockStyle}>
            <div style={questionLabelStyle}>
              Q1. How similar is the 1D track to its 3D mapping?
            </div>
            <RatingScale
              value={answers[currentBlock]?.similarity ?? null}
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
              value={answers[currentBlock]?.confidence ?? null}
              onChange={handleConfidence}
              labelLow="1 = least confident"
              labelHigh="5 = very confident"
            />
          </div>
        </div>
      )}

      {/* ── Phase 2: Multiple choice ── */}
      {!loading && !error && phase === 2 && phase2Block && (
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
                    ...(phase2Answers[currentBlock] === p.label
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
      {!loading && !error && phase === 3 && phase3Block && (
        <div key={`p3-b${currentBlock}`} style={questionsPanelStyle}>
          <div style={{ ...questionBlockStyle, flex: 1 }}>
            <div style={questionLabelStyle}>
              Select the bead in 3D that corresponds to the highlighted peak. Hover to preview, click to confirm.
              {phase3Answers[currentBlock] != null && (
                <span style={{ marginLeft: 8, color: "#22c55e" }}>✓</span>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ── Phase 4: Confidence rating ── */}
      {!loading && !error && phase === 4 && phase4Block && (
        <div key={`p4-b${currentBlock}`} style={questionsPanelStyle}>
          <div style={{ ...questionBlockStyle, flex: 1 }}>
            <div style={questionLabelStyle}>
              How confident are you in recognizing the changing patterns of all the tracks here?
            </div>
            <RatingScale
              value={phase4Answers[currentBlock] ?? null}
              onChange={handlePhase4Confidence}
              labelLow="1 = least confident"
              labelHigh="5 = very confident"
            />
          </div>
        </div>
      )}

      {/* ── Footer ── */}
      <div style={footerStyle}>
        <button
          style={{
            ...btnStyle,
            ...(phase === 1 && currentBlock === 0
              ? { opacity: 0.5, cursor: "not-allowed" }
              : phase === 2 && currentBlock === 0
                ? { opacity: 0.5, cursor: "not-allowed" }
                : {}),
          }}
          onClick={handlePrevious}
          disabled={(phase === 1 && currentBlock === 0) || (phase === 2 && currentBlock === 0)}
        >
          ← Previous
        </button>
        <button
          style={btnStyle}
          onClick={handleNext}
          disabled={
            (phase === 2 &&
              currentBlock >= NUM_PHASE2_QUESTIONS - 1 &&
              phase2Results.length >= NUM_PHASE2_QUESTIONS) ||
            (phase === 3 &&
              currentBlock >= NUM_PHASE3_QUESTIONS - 1 &&
              phase3Results.length >= NUM_PHASE3_QUESTIONS) ||
            (phase === 4 &&
              currentBlock >= NUM_PHASE4_QUESTIONS - 1 &&
              phase4Results.length >= NUM_PHASE4_QUESTIONS)
          }
        >
          {phase === 1 && currentBlock >= NUM_QUESTIONS - 1
            ? "Next phase →"
            : phase === 2 && currentBlock >= NUM_PHASE2_QUESTIONS - 1
              ? "Next phase →"
              : phase === 3 && currentBlock >= NUM_PHASE3_QUESTIONS - 1
                ? "Next phase →"
                : phase === 4 && currentBlock >= NUM_PHASE4_QUESTIONS - 1
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

const questionTitleStyle: React.CSSProperties = {
  margin: 0,
  fontSize: 16,
  fontWeight: 500,
  lineHeight: 1.5,
  color: "#e8f4f8",
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
  justifyContent: "space-between",
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
