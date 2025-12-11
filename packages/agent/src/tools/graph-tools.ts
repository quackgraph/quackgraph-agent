import type { QuackGraph } from '@quackgraph/graph';
import type { SectorSummary, LabyrinthContext } from '../types';
// import type { JsPatternEdge } from '@quackgraph/native';

export interface JsPatternEdge {
  srcVar: number;
  tgtVar: number;
  edgeType: string;
  direction: string;
}

export class GraphTools {
  constructor(private graph: QuackGraph) { }

  private resolveAsOf(contextOrAsOf?: LabyrinthContext | number): number | undefined {
    let ms: number | undefined;
    if (typeof contextOrAsOf === 'number') {
      ms = contextOrAsOf;
    } else if (contextOrAsOf?.asOf) {
      ms = contextOrAsOf.asOf instanceof Date ? contextOrAsOf.asOf.getTime() : typeof contextOrAsOf.asOf === 'number' ? contextOrAsOf.asOf : undefined;
    }
    
    // Native Rust layer expects milliseconds (f64) and converts to microseconds internally.
    // We default to Date.now() if no time is provided, to ensure "present" physics by default.
    // This prevents future edges from leaking into implicit queries.
    return ms ?? Date.now();
  }

  /**
   * LOD 0: Sector Scan / Satellite View
   * Returns a summary of available moves from the current nodes.
   */
  async getSectorSummary(currentNodes: string[], contextOrAsOf?: LabyrinthContext | number, allowedEdgeTypes?: string[]): Promise<SectorSummary[]> {
    if (currentNodes.length === 0) return [];

    const asOf = this.resolveAsOf(contextOrAsOf);

    // 1. Get Sector Stats (Count + Heat) in a single Rust call (O(1))
    const results = await this.graph.native.getSectorStats(currentNodes, asOf, allowedEdgeTypes);

    // 2. Filter if explicit allowed list provided (double check)
    // Native usually handles this, but if we have complex registry logic (e.g. exclusions), we filter here too
    if (allowedEdgeTypes && allowedEdgeTypes.length > 0) {
      return results.filter((r: SectorSummary) => allowedEdgeTypes.includes(r.edgeType)).sort((a: SectorSummary, b: SectorSummary) => b.count - a.count);
    }

    // 3. Sort by count (descending)
    return results.sort((a: SectorSummary, b: SectorSummary) => b.count - a.count);
  }

  /**
   * LOD 1.5: Ghost Map / Navigational Map
   * Generates an ASCII tree of the topology up to a certain depth.
   * Uses geometric pruning and token budgeting to keep the map readable.
   */
  async getNavigationalMap(rootId: string, depth: number = 1, contextOrAsOf?: LabyrinthContext | number): Promise<{ map: string, truncated: boolean }> {
    const maxDepth = Math.min(depth, 3); // Hard cap at depth 3 for safety
    const treeLines: string[] = [`[ROOT] ${rootId}`];
    let isTruncated = false;
    let totalLines = 0;
    const MAX_LINES = 40; // Prevent context window explosion

    // Helper for recursion
    const buildTree = async (currentId: string, currentDepth: number, prefix: string) => {
      if (currentDepth >= maxDepth) return;
      if (totalLines >= MAX_LINES) {
        if (!isTruncated) {
          treeLines.push(`${prefix}... (truncated)`);
          isTruncated = true;
        }
        return;
      }

      // Geometric pruning: 8 -> 4 -> 2
      const branchLimit = Math.max(2, Math.floor(8 / 2 ** currentDepth));
      let branchesCount = 0;

      // 1. Get stats to find "hot" edges
      const stats = await this.getSectorSummary([currentId], contextOrAsOf);

      // Sort by Heat first, then Count to prioritize "Hot" paths in the view
      stats.sort((a, b) => (b.avgHeat || 0) - (a.avgHeat || 0) || b.count - a.count);

      for (const stat of stats) {
        if (branchesCount >= branchLimit) break;
        if (totalLines >= MAX_LINES) break;

        const edgeType = stat.edgeType;
        const heatVal = stat.avgHeat || 0;
        let heatMarker = '';
        if (heatVal > 80) heatMarker = ' 🔥🔥🔥';
        else if (heatVal > 50) heatMarker = ' 🔥';
        else if (heatVal > 20) heatMarker = ' ♨️';

        // 2. Traverse to get samples (fetch just enough to display)
        const neighbors = await this.topologyScan([currentId], edgeType, contextOrAsOf);

        // Pruning neighbor display based on depth
        const neighborLimit = Math.max(1, Math.floor(branchLimit / (stats.length || 1)) + 1);
        const displayNeighbors = neighbors.slice(0, neighborLimit);

        for (let i = 0; i < displayNeighbors.length; i++) {
          if (branchesCount >= branchLimit) break;
          if (totalLines >= MAX_LINES) { isTruncated = true; break; }

          const neighborId = displayNeighbors[i];
          if (!neighborId) continue;

          // Check if this is the last item to choose the connector symbol
          const isLast = (i === displayNeighbors.length - 1) && (stats.indexOf(stat) === stats.length - 1 || branchesCount === branchLimit - 1);
          const connector = isLast ? '└──' : '├──';

          treeLines.push(`${prefix}${connector} [${edgeType}]──> (${neighborId})${heatMarker}`);
          totalLines++;

          const nextPrefix = prefix + (isLast ? '    ' : '│   ');
          await buildTree(neighborId, currentDepth + 1, nextPrefix);
          branchesCount++;
        }
      }
    };

    await buildTree(rootId, 0, ' ');

    return {
      map: treeLines.join('\n'),
      truncated: isTruncated
    };
  }

  /**
   * LOD 1: Topology Scan
   * Returns the IDs of neighbors reachable via a specific edge type.
   */
  async topologyScan(currentNodes: string[], edgeType?: string, contextOrAsOf?: LabyrinthContext | number, _minValidFrom?: number): Promise<string[]> {
    if (currentNodes.length === 0) return [];
    const asOf = this.resolveAsOf(contextOrAsOf);
    return this.graph.native.traverse(currentNodes, edgeType, 'out', asOf);
  }

  /**
   * LOD 1: Temporal Interval Scan
   * Finds neighbors connected via edges overlapping/contained in the window.
   */
  async intervalScan(currentNodes: string[], windowStart: number, windowEnd: number, constraint: 'overlaps' | 'contains' | 'during' | 'meets' = 'overlaps'): Promise<string[]> {
    return this.graph.native.traverseInterval(currentNodes, undefined, 'out', windowStart, windowEnd, constraint);
  }

  /**
   * LOD 1: Temporal Scan (Wrapper for intervalScan with edge type filtering)
   */
  async temporalScan(currentNodes: string[], windowStart: number, windowEnd: number, edgeType?: string, constraint: 'overlaps' | 'contains' | 'during' | 'meets' = 'overlaps'): Promise<string[]> {
    if (currentNodes.length === 0) return [];
    // We use the native traverseInterval which accepts edgeType
    return this.graph.native.traverseInterval(currentNodes, edgeType, 'out', windowStart, windowEnd, constraint);
  }

  /**
   * LOD 1.5: Pattern Matching (Structural Inference)
   * Finds subgraphs matching a specific shape.
   */
  async findPattern(startNodes: string[], pattern: Partial<JsPatternEdge>[], contextOrAsOf?: LabyrinthContext | number): Promise<string[][]> {
    if (startNodes.length === 0) return [];
    const nativePattern = pattern.map(p => ({
      srcVar: p.srcVar || 0,
      tgtVar: p.tgtVar || 0,
      edgeType: p.edgeType || '',
      direction: p.direction || 'out'
    }));
    const asOf = this.resolveAsOf(contextOrAsOf);
    return this.graph.native.matchPattern(startNodes, nativePattern, asOf);
  }

  /**
   * LOD 2: Content Retrieval with "Virtual Spine" Expansion.
   * If nodes are part of a document chain (NEXT/PREV), fetch context.
   */
  // biome-ignore lint/suspicious/noExplicitAny: Generic node content
  async contentRetrieval(nodeIds: string[]): Promise<any[]> {
    if (nodeIds.length === 0) return [];

    // 1. Fetch Primary Content
    const primaryNodes = await this.graph.match([])
      .where({ id: nodeIds })
      .select();

    // 2. Virtual Spine Expansion
    // Check for "NEXT" or "PREV" connections to provide document flow context.
    const spineContextIds = new Set<string>();

    for (const id of nodeIds) {
      // Look ahead
      const next = await this.graph.native.traverse([id], 'NEXT', 'out');
      next.forEach((nid: string) => { spineContextIds.add(nid); });

      // Look back
      const incomingNext = await this.graph.native.traverse([id], 'NEXT', 'in');
      incomingNext.forEach((nid: string) => { spineContextIds.add(nid); });

      const explicitPrev = await this.graph.native.traverse([id], 'PREV', 'out');
      explicitPrev.forEach((nid: string) => { spineContextIds.add(nid); });
    }

    // Remove duplicates (original nodes)
    nodeIds.forEach(id => { spineContextIds.delete(id); });

    if (spineContextIds.size > 0) {
      const contextNodes = await this.graph.match([])
        .where({ id: Array.from(spineContextIds) })
        .select();

      // Merge and Annotate
      // Create a map for fast lookup
      const contextMap = new Map(contextNodes.map(n => [n.id, n]));

      return primaryNodes.map(node => {
        // Find connected context nodes?
        // For simplicity, we just attach all found spine context, 
        // ideally we would link specific context to specific nodes but that requires tracking edges again.
        // We will just return the primary node and let the LLM see the expanded content if requested separately
        // or attach generic context.
        return {
          ...node,
          _isPrimary: true,
          _context: Array.from(contextMap.values()).map(c => ({ id: c.id, ...c.properties }))
        };
      });
    }

    return primaryNodes;
  }

  /**
   * Pheromones: Reinforce a successful path by increasing edge heat.
   */
  async reinforcePath(nodes: string[], edges: (string | undefined)[], qualityScore: number = 1.0) {
    if (nodes.length < 2) return;

    // Base increment is 50 for a perfect score. 
    const heatDelta = Math.floor(qualityScore * 50);

    for (let i = 0; i < nodes.length - 1; i++) {
      const source = nodes[i];
      const target = nodes[i + 1];
      const edge = edges[i + 1]; // edges[0] is undefined (start)

      if (source && target && edge) {
        // Call native update
        try {
          await this.graph.native.updateEdgeHeat(source, target, edge, heatDelta);
        } catch (e) {
          console.warn(`[Pheromones] Failed to update heat for ${source}->${target}:`, e);
        }
      }
    }
  }
}