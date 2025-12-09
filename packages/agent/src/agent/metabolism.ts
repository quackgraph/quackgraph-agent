import { randomUUID } from 'crypto';
import type { QuackGraph } from '@quackgraph/graph';
import type { Judge } from './judge';
import type { JudgePrompt } from '../types';

export class Metabolism {
  constructor(
    private graph: QuackGraph,
    private judge: Judge
  ) { }

  /**
   * Graph Metabolism: Summarize and Prune.
   * Identifies old, dense clusters and summarizes them into a single high-level node.
   */
  async dream(criteria: { minAgeDays: number; targetLabel: string }) {
    // 1. Identify Candidates (Nodes older than X days)
    // We process in batches to allow iterative cleanup
    const sql = `
      SELECT id, properties 
      FROM nodes 
      WHERE list_contains(labels, ?) 
        AND valid_from < (current_timestamp - INTERVAL ${criteria.minAgeDays} DAY)
        AND valid_to IS NULL -- Active nodes only
      LIMIT 100 -- Batch size
    `;

    const candidates = await this.graph.db.query(sql, [criteria.targetLabel]);
    if (candidates.length === 0) return;

    // 2. Synthesize (Judge)
    const judgePrompt: JudgePrompt = {
      goal: `Metabolism/Dreaming: Summarize these ${candidates.length} ${criteria.targetLabel} logs into a single concise insight node. Focus on patterns and key events.`,
      nodeContent: candidates.map((c) =>
        typeof c.properties === 'string' ? JSON.parse(c.properties) : c.properties
      ),
    };

    const artifact = await this.judge.evaluate(judgePrompt);

    if (!artifact) return; // Judge failed

    // 3. Identification of Anchor (Parent)
    const candidateIds = candidates.map((c) => c.id);
    const potentialParents = await this.graph.native.traverse(
      candidateIds,
      undefined,
      'in',
      undefined,
      undefined
    );

    if (potentialParents.length === 0) return; // Orphaned

    // Use the first parent found as the anchor for the summary
    const anchorId = potentialParents[0];
    if (!anchorId) return; // Satisfy noUncheckedIndexedAccess

    // 4. Rewire & Prune
    const summaryId = `summary:${randomUUID()}`;
    const summaryProps = {
      content: artifact.answer,
      source_count: candidates.length,
      generated_at: new Date().toISOString(),
      period_end: new Date().toISOString()
    };

    await this.graph.addNode(summaryId, ['Summary', 'Insight'], summaryProps);
    await this.graph.addEdge(anchorId, summaryId, 'HAS_SUMMARY');

    // Soft delete raw nodes
    for (const id of candidateIds) {
      await this.graph.deleteNode(id);
    }
  }
}