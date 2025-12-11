import type { QuackGraph } from '@quackgraph/graph';
import type { CorrelationResult, EvolutionResult, SectorSummary, TimeStepDiff } from '../types';
import type { GraphTools } from '../tools/graph-tools';

export class Chronos {
  constructor(private graph: QuackGraph, private tools: GraphTools) { }

  /**
   * Finds events connected to the anchor node that occurred or overlapped
   * with the specified time window.
   */
  async findEventsDuring(
    anchorNodeId: string,
    windowStart: Date,
    windowEnd: Date,
    constraint: 'overlaps' | 'contains' | 'during' | 'meets' = 'overlaps'
  ): Promise<string[]> {
    // Use native directly for granular control
    return await this.graph.native.traverseInterval(
      [anchorNodeId],
      undefined,
      'out',
      windowStart.getTime(),
      windowEnd.getTime(),
      constraint
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
    const anchorRows = await this.graph.db.query(
      "SELECT valid_from FROM nodes WHERE id = ?",
      [anchorNodeId]
    );

    if (anchorRows.length === 0) {
      throw new Error(`Anchor node ${anchorNodeId} not found`);
    }

    const sql = `
      WITH Anchor AS (
        SELECT valid_from::TIMESTAMPTZ as t_anchor 
        FROM nodes 
        WHERE id = ?
      ),
      Targets AS (
        SELECT id, valid_from::TIMESTAMPTZ as t_target 
        FROM nodes 
        WHERE list_contains(labels, ?)
      )
      SELECT count(*) as count
      FROM Targets, Anchor
      WHERE t_target >= (t_anchor - (INTERVAL 1 MINUTE * ${Math.floor(windowMinutes)}))
        AND t_target <= t_anchor
    `;

    const result = await this.graph.db.query(sql, [anchorNodeId, targetLabel]);
    const count = Number(result[0]?.count || 0);

    return {
      anchorLabel: 'Unknown',
      targetLabel,
      windowSizeMinutes: windowMinutes,
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
    let prevSummary: Map<string, number> = new Map();

    for (const ts of sortedTimes) {
      // Use standard JS timestamps (ms) to be consistent with GraphTools and native bindings
      const currentSummaryList = await this.tools.getSectorSummary([anchorNodeId], ts.getTime());

      const currentSummary = new Map<string, number>();
      for (const s of currentSummaryList) {
        currentSummary.set(s.edgeType, s.count);
      }

      const addedEdges: SectorSummary[] = [];
      const removedEdges: SectorSummary[] = [];
      const persistedEdges: SectorSummary[] = [];

      // Compare Current vs Prev
      for (const [type, count] of currentSummary) {
        if (prevSummary.has(type)) {
          persistedEdges.push({ edgeType: type, count });
        } else {
          addedEdges.push({ edgeType: type, count });
        }
      }

      for (const [type, count] of prevSummary) {
        if (!currentSummary.has(type)) {
          removedEdges.push({ edgeType: type, count });
        }
      }

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

      prevSummary = currentSummary;
    }

    return { anchorNodeId, timeline };
  }
}