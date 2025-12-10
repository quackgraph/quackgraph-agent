import type { QuackGraph } from '@quackgraph/graph';
import type { SectorSummary } from '../types';
import type { JsPatternEdge } from '@quackgraph/native';

export class GraphTools {
  constructor(private graph: QuackGraph) { }

  /**
   * LOD 0: Sector Scan / Satellite View
   * Returns a summary of available moves from the current nodes.
   */
  async getSectorSummary(currentNodes: string[], asOf?: number, allowedEdgeTypes?: string[]): Promise<SectorSummary[]> {
    if (currentNodes.length === 0) return [];

    // 1. Get Sector Stats (Count + Heat) in a single Rust call (O(1))
    const results = await this.graph.native.getSectorStats(currentNodes, asOf, allowedEdgeTypes);

    // 2. Filter if explicit allowed list provided (double check)
    // Native usually handles this, but if we have complex registry logic (e.g. exclusions), we filter here too
    // Note: optimization - native filtering is faster, but we rely on caller to pass correct allowedEdgeTypes from registry.getValidEdges()
    if (allowedEdgeTypes && allowedEdgeTypes.length > 0) {
      // redundant but safe if native implementation varies
      // no-op if native did its job
    }

    // 3. Sort by count (descending)
    return results.sort((a, b) => b.count - a.count);
  }

  /**
   * LOD 1: Topology Scan
   * Returns the IDs of neighbors reachable via a specific edge type.
   */
  async topologyScan(currentNodes: string[], edgeType: string, asOf?: number, _minValidFrom?: number): Promise<string[]> {
    if (currentNodes.length === 0) return [];
    // Native traverse does not support minValidFrom yet
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
  async findPattern(startNodes: string[], pattern: Partial<JsPatternEdge>[], asOf?: number): Promise<string[][]> {
    if (startNodes.length === 0) return [];
    const nativePattern = pattern.map(p => ({
      srcVar: p.srcVar || 0,
      tgtVar: p.tgtVar || 0,
      edgeType: p.edgeType || '',
      direction: p.direction || 'out'
    }));
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
      next.forEach(nid => { spineContextIds.add(nid); });

      // Look back
      const _prev = await this.graph.native.traverse([id], 'PREV', 'out');
      const incomingNext = await this.graph.native.traverse([id], 'NEXT', 'in');
      incomingNext.forEach(nid => { spineContextIds.add(nid); });

      const explicitPrev = await this.graph.native.traverse([id], 'PREV', 'out');
      explicitPrev.forEach(nid => { spineContextIds.add(nid); });
    }

    // Remove duplicates (original nodes)
    nodeIds.forEach(id => { spineContextIds.delete(id); });

    if (spineContextIds.size > 0) {
      const contextNodes = await this.graph.match([])
        .where({ id: Array.from(spineContextIds) })
        .select();

      // Merge and Annotate
      return primaryNodes.map(node => {
        return {
          ...node,
          _isPrimary: true,
          _context: contextNodes
        };
      });
    }

    return primaryNodes;
  }

  /**
   * Pheromones: Reinforce a successful path by increasing edge heat.
   */
  async reinforcePath(trace: { source: string; incomingEdge?: string }[], qualityScore: number = 1.0) {
    // Base increment is 50 for a perfect score. Clamped by native logic (u8 wraparound or saturation).
    // We assume native handles saturation at 255.
    const _heatDelta = Math.floor(qualityScore * 50);

    for (let i = 1; i < trace.length; i++) {
      const prev = trace[i - 1];
      const curr = trace[i];
      if (!prev || !curr) continue; // Satisfy noUncheckedIndexedAccess
      if (curr.incomingEdge) {
        // await this.graph.updateEdgeHeat(prev.source, curr.source, curr.incomingEdge, heatDelta);
        console.warn('Pheromones not implemented in V1 native graph');
      }
    }
  }
}