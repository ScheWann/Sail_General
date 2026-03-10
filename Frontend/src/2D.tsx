/**
 * 3D Tree Skeleton → 2D Planar Layout
 * Algorithm from "Planar Visualization of Treelike Structures" (orthographic projection + radial embedding + shape recovery).
 */

// ─── Types ───────────────────────────────────────────────────────────────────

export type Vec3 = [number, number, number];
export type Vec2 = [number, number];

/** Orthonormal projection frame: u,v = plane basis, w = view normal. */
export interface ProjectionFrame {
  u: Vec3;
  v: Vec3;
  w: Vec3;
}

/** Tree node with 3D position and computed fields through the pipeline. */
export interface TreeNode {
  id: number;
  position: Vec3;
  parentId: number | null;
  childrenIds: number[];
  /** 3D edge length to parent (undefined for root). */
  edgeLength?: number;
  /** Projected 2D position on view plane. */
  xProj?: Vec2;
  /** Depth along w (for ordering). */
  depth?: number;
  /** Target turning angle θ_n ∈ [0, 2π) from projection. */
  targetAngle?: number;
  /** Leaf count in subtree. */
  leafCount?: number;
  /** Placement weight α for sibling ordering (only for non-root). */
  placementWeight?: number;
  /** Angular span ω of subtree in radial layout. */
  omega?: number;
  /** Start angle τ of subtree in radial layout. */
  tau?: number;
  /** Current 2D position in embedding. */
  x?: Vec2;
  /** Current turning angle φ in embedded layout. */
  currentAngle?: number;
}

/** Input: list of 3D positions forming a path (bead chain). Output: tree nodes array. */
export interface TreeInput {
  positions: Vec3[];
  /** Optional: parent index per node; if omitted, node i has parent i-1 (path). */
  parentIndices?: (number | null)[];
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/** Subtract two 3D vectors. Input: a, b. Output: a - b. */
export function vec3Subtract(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

/** Euclidean norm of a 3D vector. Input: v. Output: ‖v‖. */
export function vec3Norm(v: Vec3): number {
  return Math.hypot(v[0], v[1], v[2]);
}

/** Dot product of two 2D vectors. Input: a, b. Output: ⟨a,b⟩. */
export function vec2Dot(a: Vec2, b: Vec2): number {
  return a[0] * b[0] + a[1] * b[1];
}

/** 2D determinant det(a,b) = a_x b_y - a_y b_x. Input: a, b. Output: det(a,b). */
export function vec2Det(a: Vec2, b: Vec2): number {
  return a[0] * b[1] - a[1] * b[0];
}

/** CCW angle from vector a to b in [0, 2π). Input: a, b. Output: angle in radians. */
export function ccwAngle(a: Vec2, b: Vec2): number {
  const det = vec2Det(a, b);
  const dot = vec2Dot(a, b);
  let angle = Math.atan2(det, dot);
  if (angle < 0) angle += 2 * Math.PI;
  return angle;
}

/** Wrap angle to (-π, π]. Input: angle in rad. Output: wrapped angle. */
export function wrapToPi(angle: number): number {
  let a = angle % (2 * Math.PI);
  if (a > Math.PI) a -= 2 * Math.PI;
  if (a <= -Math.PI) a += 2 * Math.PI;
  return a;
}

/** Subtract two 2D vectors. Input: a, b. Output: a - b. */
export function vec2Subtract(a: Vec2, b: Vec2): Vec2 {
  return [a[0] - b[0], a[1] - b[1]];
}

/** Apply 3D→2D orthographic projection: p · u, p · v, p · w. Input: p, frame. Output: [u', v', depth]. */
function projectPoint(p: Vec3, frame: ProjectionFrame): Vec2 & { depth: number } {
  const u = frame.u, v = frame.v, w = frame.w;
  const uPrime = p[0] * u[0] + p[1] * u[1] + p[2] * u[2];
  const vPrime = p[0] * v[0] + p[1] * v[1] + p[2] * v[2];
  const depth = p[0] * w[0] + p[1] * w[1] + p[2] * w[2];
  return [uPrime, vPrime, depth] as Vec2 & { depth: number };
}

/** Norm of 2D vector. Input: v. Output: ‖v‖. */
function vec2Norm(v: Vec2): number {
  return Math.hypot(v[0], v[1]);
}

/** 2D rotation matrix R(γ) applied to vector v. Input: angle γ, vector v. Output: R(γ) v. */
function rotate2D(gamma: number, v: Vec2): Vec2 {
  const c = Math.cos(gamma), s = Math.sin(gamma);
  return [c * v[0] - s * v[1], s * v[0] + c * v[1]];
}

// ─── Step 1: Build tree data ─────────────────────────────────────────────────

/**
 * Build rooted tree with parent/children and 3D edge lengths.
 * Input: raw positions (and optional parent indices). Output: array of TreeNode with edgeLength set.
 */
export function buildTreeData(input: TreeInput): TreeNode[] {
  const { positions, parentIndices } = input;
  const n = positions.length;
  const nodes: TreeNode[] = positions.map((position, id) => ({
    id,
    position,
    parentId: parentIndices ? parentIndices[id] : (id === 0 ? null : id - 1),
    childrenIds: [],
  }));

  for (let i = 0; i < n; i++) {
    const parentId = nodes[i].parentId;
    if (parentId != null) {
      nodes[parentId].childrenIds!.push(i);
      const pChild = nodes[i].position;
      const pParent = nodes[parentId].position;
      nodes[i].edgeLength = vec3Norm(vec3Subtract(pChild, pParent));
    }
  }
  return nodes;
}

// ─── Step 2: Orthographic projection ─────────────────────────────────────────

/**
 * Project every 3D node onto the view plane (u,v); store 2D point and depth.
 * Input: tree nodes, projection frame. Output: nodes with xProj and depth set.
 */
export function orthographicProject(nodes: TreeNode[], frame: ProjectionFrame): void {
  for (const node of nodes) {
    const proj = projectPoint(node.position, frame);
    node.xProj = [proj[0], proj[1]];
    node.depth = proj.depth;
  }
}

// ─── Step 3: Compute target turning angles ───────────────────────────────────

/**
 * Compute target angle θ_n at each non-root node from projected edges (ccw angle from parent edge to current edge).
 * Input: tree with xProj set. Output: nodes with targetAngle set.
 */
export function computeTargetAngles(nodes: TreeNode[]): void {
  const root = nodes.find((n) => n.parentId === null);
  if (!root || root.xProj == null) return;

  for (const node of nodes) {
    if (node.parentId === null) continue;
    const parent = nodes[node.parentId];
    const grandparent = node.parentId !== null ? nodes[parent.parentId!] : null;

    const eCur = vec2Subtract(node.xProj!, parent.xProj!);
    if (vec2Norm(eCur) < 1e-10) {
      node.targetAngle = Math.PI;
      continue;
    }

    let ePrev: Vec2;
    if (grandparent && parent.xProj && grandparent.xProj) {
      ePrev = vec2Subtract(parent.xProj, grandparent.xProj);
      if (vec2Norm(ePrev) < 1e-10) {
        let walk = parent;
        while (walk.parentId != null && vec2Norm(vec2Subtract(walk.xProj!, nodes[walk.parentId].xProj!)) < 1e-10) {
          walk = nodes[walk.parentId];
        }
        ePrev = walk.parentId != null
          ? vec2Subtract(walk.xProj!, nodes[walk.parentId].xProj!)
          : [1, 0];
        if (vec2Norm(ePrev) < 1e-10) ePrev = [1, 0];
      }
    } else {
      ePrev = [1, 0];
    }
    node.targetAngle = ccwAngle(ePrev, eCur);
  }
}

// ─── Step 4: Compute leaf counts ─────────────────────────────────────────────

/**
 * For each node, set leafCount = number of leaves in subtree (leaf = 1, internal = sum of children).
 * Input: tree structure. Output: nodes with leafCount set.
 */
export function computeLeafCounts(nodes: TreeNode[]): void {
  function dfs(id: number): number {
    const node = nodes[id];
    if (node.childrenIds.length === 0) {
      node.leafCount = 1;
      return 1;
    }
    let sum = 0;
    for (const c of node.childrenIds) sum += dfs(c);
    node.leafCount = sum;
    return sum;
  }
  const rootId = nodes.findIndex((n) => n.parentId === null);
  if (rootId >= 0) dfs(rootId);
}

// ─── Step 5: Placement weights (longest path, cutoff, weighted center, α) ─────

/** Path length in 2D projected edges from node to descendant (sum of projected edge lengths). */
function pathLength2D(nodes: TreeNode[], fromId: number, toId: number): number {
  let len = 0;
  let cur = toId;
  while (cur !== fromId && nodes[cur].parentId != null) {
    const p = nodes[cur].parentId!;
    const a = nodes[cur].xProj!, b = nodes[p].xProj!;
    len += vec2Norm(vec2Subtract(a, b));
    cur = p;
  }
  return cur === fromId ? len : 0;
}

/** Graph distance (hop count) from node c to node i within subtree of c. */
function graphDist(nodes: TreeNode[], cId: number, iId: number): number {
  let d = 0;
  let cur = iId;
  while (cur !== cId && nodes[cur].parentId != null) {
    d++;
    cur = nodes[cur].parentId!;
  }
  return cur === cId ? d : -1;
}

/** All node ids in subtree rooted at cId. */
function subtreeIds(nodes: TreeNode[], cId: number): number[] {
  const out: number[] = [];
  function collect(id: number) {
    out.push(id);
    for (const ch of nodes[id].childrenIds) collect(ch);
  }
  collect(cId);
  return out;
}

/**
 * Compute placement weight α_c for each child; sort children by α to preserve left/right order.
 * Input: tree with xProj, leafCount. Output: each node (as child) has placementWeight set.
 */
export function computePlacementWeights(nodes: TreeNode[]): void {
  const rootId = nodes.findIndex((n) => n.parentId === null);
  if (rootId < 0) return;

  for (const node of nodes) {
    if (node.childrenIds.length < 2) continue;
    const nId = node.id;
    const xN = node.xProj!;

    const delta: number[] = [];
    for (const cId of node.childrenIds) {
      const leaves = subtreeIds(nodes, cId).filter((id) => nodes[id].childrenIds.length === 0);
      let maxLen = 0;
      for (const q of leaves) {
        const L = pathLength2D(nodes, cId, q);
        if (L > maxLen) maxLen = L;
      }
      delta.push(maxLen);
    }
    const d = Math.min(...delta) || 1;

    const weightedCenters: Vec2[] = [];
    for (const cId of node.childrenIds) {
      const within = subtreeIds(nodes, cId).filter((i) => graphDist(nodes, cId, i) >= 0);
      const withDist = within.map((i) => ({ i, dist: graphDist(nodes, cId, i) }));
      const withinCutoff = withDist.filter(({ dist }) => dist <= d).map(({ i }) => i);
      let sumW = 0;
      let sumWx = [0, 0] as Vec2;
      for (const i of withinCutoff) {
        const dist = graphDist(nodes, cId, i);
        const w = 1 - dist / d;
        sumW += w;
        const xi = nodes[i].xProj!;
        sumWx[0] += w * xi[0];
        sumWx[1] += w * xi[1];
      }
      if (sumW < 1e-10) {
        weightedCenters.push(nodes[cId].xProj!);
      } else {
        weightedCenters.push([sumWx[0] / sumW, sumWx[1] / sumW]);
      }
    }

    const qN: Vec2 = node.parentId == null
      ? [1, 0]
      : vec2Subtract(node.xProj!, nodes[node.parentId].xProj!);
    const qNorm = vec2Norm(qN) || 1;
    const qDir: Vec2 = [qN[0] / qNorm, qN[1] / qNorm];

    node.childrenIds.forEach((cId, idx) => {
      const mc = weightedCenters[idx];
      const toChild = vec2Subtract(mc, xN);
      const angle = ccwAngle(qDir, toChild);
      nodes[cId].placementWeight = angle;
    });
    node.childrenIds.sort((a, b) => (nodes[a].placementWeight ?? 0) - (nodes[b].placementWeight ?? 0));
  }
}

// ─── Step 6: Radial planar embedding ────────────────────────────────────────

/**
 * Initial 2D planar positions by radial embedding with fixed edge lengths; children sorted by α.
 * Input: tree with edgeLength, leafCount, placementWeight. Output: nodes with x, omega, tau set.
 */
export function radialEmbedding(nodes: TreeNode[]): void {
  const rootId = nodes.findIndex((n) => n.parentId === null);
  if (rootId < 0) return;
  assignRadialPositions(nodes, rootId);
}

/** Recursive helper for Step 6: set x, omega, tau for subtree; children sorted by placementWeight. */
function assignRadialPositions(nodes: TreeNode[], rootId: number): void {
  const root = nodes[rootId];
  root.x = [0, 0];
  root.omega = 2 * Math.PI;
  root.tau = 0;
  const lr = root.leafCount ?? 1;
  const rootChildren = root.childrenIds.slice().sort((a, b) => (nodes[a].placementWeight ?? 0) - (nodes[b].placementWeight ?? 0));

  let eta = 0;
  for (const cId of rootChildren) {
    const child = nodes[cId];
    child.omega = 2 * Math.PI * ((child.leafCount ?? 1) / lr);
    child.tau = eta;
    const eps = child.edgeLength ?? 0;
    const mid = child.tau + 0.5 * child.omega!;
    child.x = [
      root.x![0] + eps * Math.cos(mid),
      root.x![1] + eps * Math.sin(mid),
    ];
    eta += child.omega!;
  }

  function visit(nId: number) {
    const node = nodes[nId];
    if (!node.x) return;
    const children = node.childrenIds.slice().sort((a, b) => (nodes[a].placementWeight ?? 0) - (nodes[b].placementWeight ?? 0));
    let eta = node.tau ?? 0;
    for (const cId of children) {
      const child = nodes[cId];
      child.omega = 2 * Math.PI * ((child.leafCount ?? 1) / lr);
      child.tau = eta;
      const eps = child.edgeLength ?? 0;
      const mid = child.tau + 0.5 * child.omega!;
      child.x = [
        node.x[0] + eps * Math.cos(mid),
        node.x[1] + eps * Math.sin(mid),
      ];
      eta += child.omega!;
      visit(cId);
    }
  }
  for (const cId of rootChildren) visit(cId);
}

// ─── Step 7: Compute current angles in embedded layout ───────────────────────

/**
 * Compute current turning angle φ_n in the 2D layout (ccw from parent edge to current edge).
 * Input: tree with x set. Output: nodes with currentAngle set.
 */
export function computeCurrentAngles(nodes: TreeNode[]): void {
  for (const node of nodes) {
    if (node.parentId === null) continue;
    const parent = nodes[node.parentId];
    const grandparent = parent.parentId != null ? nodes[parent.parentId] : null;

    const eCur = vec2Subtract(node.x!, parent.x!);
    let ePrev: Vec2;
    if (grandparent && parent.x && grandparent.x) {
      ePrev = vec2Subtract(parent.x, grandparent.x);
    } else {
      ePrev = [1, 0];
    }
    if (vec2Norm(eCur) < 1e-10) {
      node.currentAngle = Math.PI;
    } else {
      node.currentAngle = ccwAngle(ePrev, eCur);
    }
  }
}

// ─── Step 8 & 9: Shape recovery + intersection rejection ──────────────────────

/** Test if segments (a1,a2) and (b1,b2) intersect (excluding shared endpoints). */
export function segmentIntersect(a1: Vec2, a2: Vec2, b1: Vec2, b2: Vec2, excludeShared: boolean): boolean {
  const dax = a2[0] - a1[0], day = a2[1] - a1[1];
  const dbx = b2[0] - b1[0], dby = b2[1] - b1[1];
  const den = dax * dby - day * dbx;
  if (Math.abs(den) < 1e-12) return false;
  const t = ((b1[0] - a1[0]) * dby - (b1[1] - a1[1]) * dbx) / den;
  const s = ((b1[0] - a1[0]) * day - (b1[1] - a1[1]) * dax) / den;
  if (t < 0 || t > 1 || s < 0 || s > 1) return false;
  if (excludeShared) {
    const tol = 1e-8;
    if (t < tol && (Math.hypot(b1[0] - a1[0], b1[1] - a1[1]) < tol || Math.hypot(b2[0] - a1[0], b2[1] - a1[1]) < tol)) return false;
    if (t > 1 - tol && (Math.hypot(b1[0] - a2[0], b1[1] - a2[1]) < tol || Math.hypot(b2[0] - a2[0], b2[1] - a2[1]) < tol)) return false;
  }
  return true;
}

/** All edges as [nodeId, parentId] (excluding root). */
function listEdges(nodes: TreeNode[]): [number, number][] {
  const edges: [number, number][] = [];
  for (const node of nodes) {
    if (node.parentId != null) edges.push([node.id, node.parentId]);
  }
  return edges;
}

/** Check if tree has any edge-edge intersection (non-adjacent). */
function hasIntersection(nodes: TreeNode[], edges: [number, number][]): boolean {
  for (let i = 0; i < edges.length; i++) {
    const [a, b] = edges[i];
    const a1 = nodes[a].x!, a2 = nodes[b].x!;
    for (let j = i + 1; j < edges.length; j++) {
      const [c, d] = edges[j];
      if (a === c || a === d || b === c || b === d) continue;
      const b1 = nodes[c].x!, b2 = nodes[d].x!;
      if (segmentIntersect(a1, a2, b1, b2, true)) return true;
    }
  }
  return false;
}

/**
 * One step of shape recovery: compute angle error, propose rotation γ = κ Δ, rotate subtree, reject if intersection.
 * Input: tree (x, targetAngle, currentAngle), κ. Output: nodes with updated x (if accepted).
 */
export function shapeRecoveryStep(nodes: TreeNode[], kappa: number): boolean {
  computeCurrentAngles(nodes);
  const rootId = nodes.findIndex((n) => n.parentId === null);
  if (rootId < 0) return false;

  const bfOrder: number[] = [];
  const queue = [rootId];
  while (queue.length > 0) {
    const nId = queue.shift()!;
    bfOrder.push(nId);
    queue.push(...nodes[nId].childrenIds);
  }

  let accepted = false;
  for (const nId of bfOrder) {
    const node = nodes[nId];
    if (node.parentId === null) continue;
    const theta = node.targetAngle ?? 0;
    const phi = node.currentAngle ?? 0;
    const delta = wrapToPi(theta - phi);
    const gamma = kappa * delta;

    const parentId = node.parentId;
    const pivot = nodes[parentId].x!;
    const backup = new Map<number, Vec2>();
    const subtreeIds: number[] = [];
    function collect(id: number) {
      subtreeIds.push(id);
      backup.set(id, [...nodes[id].x!]);
      nodes[id].x = rotate2D(gamma, vec2Subtract(nodes[id].x!, pivot));
      nodes[id].x![0] += pivot[0];
      nodes[id].x![1] += pivot[1];
      for (const c of nodes[id].childrenIds) collect(c);
    }
    collect(nId);

    const edges = listEdges(nodes);
    if (hasIntersection(nodes, edges)) {
      for (const id of subtreeIds) nodes[id].x = backup.get(id);
    } else {
      accepted = true;
    }
  }
  return accepted;
}

// ─── Step 10: Iterate until convergence ─────────────────────────────────────

/**
 * Repeat steps 7–9 until max |Δ_n| < ε or max iterations or no accepted update.
 * Input: tree, κ, ε, maxIter. Output: tree with final x.
 */
export function iterateConvergence(nodes: TreeNode[], kappa: number, epsilon: number, maxIter: number): void {
  for (let iter = 0; iter < maxIter; iter++) {
    computeCurrentAngles(nodes);
    let maxDelta = 0;
    for (const node of nodes) {
      if (node.parentId === null) continue;
      const d = wrapToPi((node.targetAngle ?? 0) - (node.currentAngle ?? 0));
      if (Math.abs(d) > maxDelta) maxDelta = Math.abs(d);
    }
    if (maxDelta < epsilon) break;
    const updated = shapeRecoveryStep(nodes, kappa);
    if (!updated) break;
  }
}

// ─── Step 11: Normalize to widget ───────────────────────────────────────────

/**
 * Map final 2D positions to widget coordinates with margin and uniform scale.
 * Input: nodes with x, widget W,H, margin m. Output: array of Vec2 in widget space.
 */
export function normalizeToWidget(
  nodes: TreeNode[],
  width: number,
  height: number,
  margin: number
): Vec2[] {
  const xs = nodes.map((n) => n.x!).filter(Boolean);
  if (xs.length === 0) return [];
  let uMin = xs[0][0], uMax = xs[0][0], vMin = xs[0][1], vMax = xs[0][1];
  for (const [u, v] of xs) {
    if (u < uMin) uMin = u;
    if (u > uMax) uMax = u;
    if (v < vMin) vMin = v;
    if (v > vMax) vMax = v;
  }
  const rangeU = uMax - uMin || 1;
  const rangeV = vMax - vMin || 1;
  const scale = Math.min((width - 2 * margin) / rangeU, (height - 2 * margin) / rangeV);
  return nodes.map((n) => {
    const [u, v] = n.x ?? [0, 0];
    return [margin + scale * (u - uMin), margin + scale * (v - vMin)];
  });
}

// ─── Full pipeline ────────────────────────────────────────────────────────────

/**
 * Build projection frame from view direction (camera looking at center); u,v span the view plane.
 * Input: viewNormal w (unit). Output: orthonormal { u, v, w }.
 */
export function makeProjectionFrameFromView(w: Vec3): ProjectionFrame {
  const wn = Math.hypot(w[0], w[1], w[2]) || 1;
  const wu: Vec3 = [w[0] / wn, w[1] / wn, w[2] / wn];
  let u: Vec3 = [1, 0, 0];
  if (Math.abs(wu[0]) > 0.9) u = [0, 1, 0];
  let ux = u[0] - wu[0] * (u[0] * wu[0] + u[1] * wu[1] + u[2] * wu[2]);
  let uy = u[1] - wu[1] * (u[0] * wu[0] + u[1] * wu[1] + u[2] * wu[2]);
  let uz = u[2] - wu[2] * (u[0] * wu[0] + u[1] * wu[1] + u[2] * wu[2]);
  const ul = Math.hypot(ux, uy, uz) || 1;
  u = [ux / ul, uy / ul, uz / ul];
  const v: Vec3 = [
    wu[1] * u[2] - wu[2] * u[1],
    wu[2] * u[0] - wu[0] * u[2],
    wu[0] * u[1] - wu[1] * u[0],
  ];
  return { u, v, w: wu };
}

/**
 * Continuous path projection: project each 3D point onto the view plane in order (first → last, node by node).
 * Use this when the structure is a single thread so it never breaks; output order is always 0,1,…,n-1.
 */
export function projectPathTo2D(
  positions: Vec3[],
  frame: ProjectionFrame,
  width: number,
  height: number,
  margin: number = 20
): Vec2[] {
  if (positions.length === 0) return [];
  const u = frame.u, v = frame.v;
  const points: Vec2[] = positions.map((p) => [
    p[0] * u[0] + p[1] * u[1] + p[2] * u[2],
    p[0] * v[0] + p[1] * v[1] + p[2] * v[2],
  ]);
  let uMin = points[0][0], uMax = points[0][0], vMin = points[0][1], vMax = points[0][1];
  for (const [a, b] of points) {
    if (a < uMin) uMin = a;
    if (a > uMax) uMax = a;
    if (b < vMin) vMin = b;
    if (b > vMax) vMax = b;
  }
  const rangeU = uMax - uMin || 1;
  const rangeV = vMax - vMin || 1;
  const scale = Math.min((width - 2 * margin) / rangeU, (height - 2 * margin) / rangeV);
  return points.map(([a, b]) => [
    margin + scale * (a - uMin),
    margin + scale * (b - vMin),
  ]);
}

/**
 * Full pipeline: 3D positions + view frame → 2D planar layout (no crossings, shape preserved).
 * Input: positions, frame, widget size, margin, options. Output: 2D coordinates per node.
 */
/**
 * Full pipeline (Planar Visualization of Treelike Structures):
 * orthographic projection → target angles → leaf counts → sibling order (placement weights)
 * → radial planar embedding → iterative shape recovery (κ·Δ, reject if intersection) → normalize.
 */
export function tree3DTo2D(
  positions: Vec3[],
  frame: ProjectionFrame,
  width: number,
  height: number,
  margin: number = 20,
  kappa: number = 0.1,
  epsilon: number = 1e-4,
  maxIter: number = 50
): Vec2[] {
  const nodes = buildTreeData({ positions });
  orthographicProject(nodes, frame);
  computeTargetAngles(nodes);
  computeLeafCounts(nodes);
  computePlacementWeights(nodes);
  radialEmbedding(nodes);
  iterateConvergence(nodes, kappa, epsilon, maxIter);
  return normalizeToWidget(nodes, width, height, margin);
}
