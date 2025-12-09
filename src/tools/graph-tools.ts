import type { QuackGraph } from '../../packages/quackgraph/packages/quack-graph/src/graph';
import type { SectorSummary } from '../types';

export class GraphTools {
  constructor(private graph: QuackGraph) {}

  /**
   * LOD 0: Sector Scan / Satellite View
   * Returns a summary of available moves from the current nodes.
   */
  async getSectorSummary(currentNodes: string[], asOf?: number): Promise<SectorSummary[]> {
    if (currentNodes.length === 0) return [];
    
    // 1. Get Sector Stats (Count + Heat) in a single Rust call (O(1))
    // @ts-ignore: Native binding update
    const results = await this.graph.native.getSectorStats(currentNodes, asOf);
    
    // 2. Sort by count (descending)
    return results.sort((a, b) => b.count - a.count);
  }

  /**
   * LOD 1: Topology Scan
   * Returns the IDs of neighbors reachable via a specific edge type.
   */
  async topologyScan(currentNodes: string[], edgeType: string, asOf?: number): Promise<string[]> {
    return this.graph.native.traverse(currentNodes, edgeType, 'out', asOf);
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
      next.forEach(nid => spineContextIds.add(nid));
      
      // Look back
      const prev = await this.graph.native.traverse([id], 'PREV', 'out'); // Assuming bidirectional NEXT usually implies PREV or we traverse NEXT 'in'
      // If schema uses NEXT for both directions (A)-[NEXT]->(B), then previous is incoming NEXT.
      // Let's check incoming NEXT as well to be safe for "PREV" logic.
      const incomingNext = await this.graph.native.traverse([id], 'NEXT', 'in');
      incomingNext.forEach(nid => spineContextIds.add(nid));
      
      // Also check explicit PREV if it exists in schema
      const explicitPrev = await this.graph.native.traverse([id], 'PREV', 'out');
      explicitPrev.forEach(nid => spineContextIds.add(nid));
    }

    // Remove duplicates (original nodes)
    nodeIds.forEach(id => spineContextIds.delete(id));

    if (spineContextIds.size > 0) {
      const contextNodes = await this.graph.match([])
        .where({ id: Array.from(spineContextIds) })
        .select();

      // Merge and Annotate
      return primaryNodes.map(node => {
        // Naive attachment of context for now. 
        // ideally we map specific neighbors to specific nodes, but for LOD 2 batch retrieval, 
        // we return the bag of content.
        return {
          ...node,
          _isPrimary: true,
          _context: contextNodes // The Judge agent can filter this list based on IDs if needed, but usually just having the text in the prompt is enough.
        };
      });
    }

    return primaryNodes;
  }

  /**
   * Pheromones: Reinforce a successful path by increasing edge heat.
   */
  async reinforcePath(trace: { source: string; incomingEdge?: string }[]) {
    for (let i = 1; i < trace.length; i++) {
      const prev = trace[i - 1];
      const curr = trace[i];
      if (curr.incomingEdge) {
        // Boost heat by 200 (max is 255 usually, V1 logic)
        await this.graph.updateEdgeHeat(prev.source, curr.source, curr.incomingEdge, 200);
      }
    }
  }
}