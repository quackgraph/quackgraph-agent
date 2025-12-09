import type { QuackGraph } from '../../packages/quackgraph/packages/quack-graph/src/graph';
import type { CorrelationResult, EvolutionResult, SectorSummary, TimeStepDiff } from '../types';
import type { GraphTools } from '../tools/graph-tools';

export class Chronos {
  constructor(private graph: QuackGraph, private tools: GraphTools) {}

  /**
   * Finds events connected to the anchor node that occurred or overlapped
   * with the specified time window.
   */
  async findEventsDuring(
    anchorNodeId: string, 
    windowStart: Date, 
    windowEnd: Date, 
    edgeType?: string,
    direction: 'out' | 'in' = 'out'
  ): Promise<string[]> {
    return await this.graph.traverseInterval(
      [anchorNodeId], 
      edgeType, 
      direction, 
      windowStart, 
      windowEnd
    );
  }

  /**
   * Analyze correlation between an anchor node and a target label within a time window.
   * Uses DuckDB SQL window functions.
   */
  async analyzeCorrelation(
    anchorNodeId: string, 
    targetLabel: string, 
    windowMinutes: number
  ): Promise<CorrelationResult> {
    // 1. Get Anchor Node Timestamp (valid_from)
    const anchorRows = await this.graph.db.query(
      "SELECT valid_from FROM nodes WHERE id = ?", 
      [anchorNodeId]
    );
    
    if (anchorRows.length === 0) {
      throw new Error(`Anchor node ${anchorNodeId} not found`);
    }
    
    // DuckDB returns TIMESTAMP
    
    const sql = `
      WITH Anchor AS (
        SELECT valid_from as t_anchor 
        FROM nodes 
        WHERE id = ?
      ),
      Targets AS (
        SELECT id, valid_from as t_target 
        FROM nodes 
        WHERE list_contains(labels, ?)
      )
      SELECT count(*) as count
      FROM Targets, Anchor
      WHERE t_target >= (t_anchor - INTERVAL ${windowMinutes} MINUTE)
        AND t_target <= t_anchor
    `;
    
    // biome-ignore lint/suspicious/noExplicitAny: SQL result
    const result = await this.graph.db.query(sql, [anchorNodeId, targetLabel]);
    const count = Number(result[0]?.count || 0);
    
    return {
      anchorLabel: 'Unknown', 
      targetLabel,
      windowSizeMinutes,
      correlationScore: count > 0 ? 1.0 : 0.0, // Simplified boolean correlation
      sampleSize: count,
      description: `Found ${count} instances of ${targetLabel} in the ${windowMinutes}m window.`
    };
  }

  /**
   * Evolutionary Diffing: Watches how the topology around a node changes over time.
   * Returns a diff of edges (Added, Removed, Persisted) between time snapshots.
   */
  async evolutionaryDiff(anchorNodeId: string, timestamps: Date[]): Promise<EvolutionResult> {
    const sortedTimes = timestamps.sort((a, b) => a.getTime() - b.getTime());
    const timeline: TimeStepDiff[] = [];

    // Initial state (baseline)
    // We scan effectively "Before Time" or just use the first timestamp as baseline
    let prevSummary: Map<string, number> = new Map();

    for (const ts of sortedTimes) {
      const micros = ts.getTime() * 1000;
      const currentSummaryList = await this.tools.getSectorSummary([anchorNodeId], micros);
      
      const currentSummary = new Map<string, number>();
      currentSummaryList.forEach(s => currentSummary.set(s.edgeType, s.count));

      const addedEdges: SectorSummary[] = [];
      const removedEdges: SectorSummary[] = [];
      const persistedEdges: SectorSummary[] = [];

      // Compare Current vs Prev
      // 1. Check for Added or Persisted
      for (const [type, count] of currentSummary) {
        if (prevSummary.has(type)) {
          persistedEdges.push({ edgeType: type, count });
        } else {
          addedEdges.push({ edgeType: type, count });
        }
      }

      // 2. Check for Removed
      for (const [type, count] of prevSummary) {
        if (!currentSummary.has(type)) {
          removedEdges.push({ edgeType: type, count });
        }
      }

      // Calculate density change (total edges)
      const prevTotal = Array.from(prevSummary.values()).reduce((a, b) => a + b, 0);
      const currTotal = Array.from(currentSummary.values()).reduce((a, b) => a + b, 0);
      
      const densityChange = prevTotal === 0 ? (currTotal > 0 ? 100 : 0) : ((currTotal - prevTotal) / prevTotal) * 100;

      timeline.push({
        timestamp: ts,
        addedEdges,
        removedEdges,
        persistedEdges,
        densityChange
      });

      // Update state
      prevSummary = currentSummary;
    }

    return { anchorNodeId, timeline };
  }
}