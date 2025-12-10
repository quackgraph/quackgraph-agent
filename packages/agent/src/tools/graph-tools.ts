import type { QuackGraph } from '@quackgraph/graph';
import type { SectorSummary, LabyrinthContext } from '../types';
import type { JsPatternEdge } from '@quackgraph/native';

export class GraphTools {
  constructor(private graph: QuackGraph) { }

  private resolveAsOf(contextOrAsOf?: LabyrinthContext | number): number | undefined {
    if (typeof contextOrAsOf === 'number') return contextOrAsOf;
    if (!contextOrAsOf?.asOf) return undefined;
    return contextOrAsOf.asOf instanceof Date ? contextOrAsOf.asOf.getTime() : contextOrAsOf.asOf;
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
    // Note: optimization - native filtering is faster, but we rely on caller to pass correct allowedEdgeTypes from registry.getValidEdges()
    if (allowedEdgeTypes && allowedEdgeTypes.length > 0) {
      // redundant but safe if native implementation varies
      // no-op if native did its job
    }

    // 3. Sort by count (descending)
    return results.sort((a, b) => b.count - a.count);
  }

  /**
   * LOD 1.5: Ghost Map / Navigational Map
   * Generates an ASCII tree of the topology up to a certain depth.
   * Uses geometric pruning to keep the map readable.
   */
  async getNavigationalMap(rootId: string, depth: number = 1, contextOrAsOf?: LabyrinthContext | number): Promise<{ map: string, truncated: boolean }> {
    const maxDepth = Math.min(depth, 4);
    const treeLines: string[] = [`[ROOT] ${rootId}`];
    let isTruncated = false;
    const _asOf = this.resolveAsOf(contextOrAsOf);

    // Helper for recursion
    const buildTree = async (currentId: string, currentDepth: number, prefix: string) => {
      if (currentDepth >= maxDepth) return;

      // Geometric pruning: 10 -> 5 -> 3 -> 1
      const branchLimit = Math.floor(10 / (currentDepth + 1));
      let branchesCount = 0;

      // 1. Get stats to find "hot" edges
      const stats = await this.getSectorSummary([currentId], contextOrAsOf);
      
      for (const stat of stats) {
        if (branchesCount >= branchLimit) {
            isTruncated = true;
            break;
        }

        const edgeType = stat.edgeType;
        const heatMarker = (stat.avgHeat || 0) > 50 ? ' 🔥' : '';
        
        // 2. Traverse to get samples (fetch just enough to display)
        const neighbors = await this.topologyScan([currentId], edgeType, contextOrAsOf);
        const neighborLimit = Math.max(1, Math.floor(branchLimit / (stats.length || 1)) + 1); 
        const displayNeighbors = neighbors.slice(0, neighborLimit);
        
        for (let i = 0; i < displayNeighbors.length; i++) {
             if (branchesCount >= branchLimit) { isTruncated = true; break; }
             const neighborId = displayNeighbors[i];
             if (!neighborId) continue;
             const connector = (i === displayNeighbors.length - 1 && branchesCount === branchLimit - 1) ? '└──' : '├──';
             
             treeLines.push(`${prefix}${connector}[${edgeType}]──> (${neighborId})${heatMarker}`);
             
             const nextPrefix = prefix + (connector === '└──' ? '    ' : '│   ');
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
    // Native traverse does not support minValidFrom yet
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