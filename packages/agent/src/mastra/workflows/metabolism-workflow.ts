import { createStep, createWorkflow } from '@mastra/core/workflows';
import { z } from 'zod';
import { getGraphInstance } from '../../lib/graph-instance';
import { randomUUID } from 'crypto';
import { JudgeDecisionSchema } from '../../agent-schemas';

// Step 1: Identify Candidates
const identifyCandidates = createStep({
  id: 'identify-candidates',
  description: 'Finds old nodes suitable for summarization',
  inputSchema: z.object({
    minAgeDays: z.number(),
    targetLabel: z.string(),
  }),
  outputSchema: z.object({
    candidateIds: z.array(z.string()),
    candidatesContent: z.array(z.record(z.any())),
  }),
  execute: async ({ inputData }) => {
    const graph = getGraphInstance();
    // Raw SQL for efficiency
    // Ensure we don't accidentally wipe recent data if minAgeDays is too small
    const safeDays = Math.max(inputData.minAgeDays, 1);
    const sql = `
      SELECT id, properties 
      FROM nodes 
      WHERE list_contains(labels, ?) 
        AND valid_from < (current_timestamp - INTERVAL ${safeDays} DAY)
        AND valid_to IS NULL
      LIMIT 100
    `;
    const rows = await graph.db.query(sql, [inputData.targetLabel]);

    const candidatesContent = rows.map((c) =>
      typeof c.properties === 'string' ? JSON.parse(c.properties) : c.properties
    );
    const candidateIds = rows.map(c => c.id);

    return { candidateIds, candidatesContent };
  },
});

// Step 2: Synthesize Insight (using Judge Agent)
const synthesizeInsight = createStep({
  id: 'synthesize-insight',
  description: 'Uses LLM to summarize the candidates',
  inputSchema: z.object({
    candidateIds: z.array(z.string()),
    candidatesContent: z.array(z.record(z.any())),
  }),
  outputSchema: z.object({
    summaryText: z.string().optional(),
    candidateIds: z.array(z.string()),
  }),
  execute: async ({ inputData, mastra }) => {
    if (inputData.candidateIds.length === 0) return { candidateIds: [] };

    const judge = mastra?.getAgent('judgeAgent');
    if (!judge) throw new Error('Judge Agent not found');

    const prompt = `
      Goal: Metabolism/Dreaming: Summarize these ${inputData.candidatesContent.length} logs into a single concise insight node. Focus on patterns and key events.
      Data: ${JSON.stringify(inputData.candidatesContent)}
    `;

    const response = await judge.generate(prompt, {
      structuredOutput: {
        schema: JudgeDecisionSchema
      }
    });
    let summaryText = '';

    try {
      const result = response.object;
      if (result && (result.isAnswer || result.answer)) {
        summaryText = result.answer;
      }
    } catch (e) {
      // Fallback or just log, but don't crash workflow if one synthesis fails
      console.error("Metabolism synthesis failed parsing", e);
    }

    return { summaryText, candidateIds: inputData.candidateIds };
  },
});

// Step 3: Apply Summary (Rewire Graph)
const applySummary = createStep({
  id: 'apply-summary',
  description: 'Writes the summary node and prunes old nodes',
  inputSchema: z.object({
    summaryText: z.string().optional(),
    candidateIds: z.array(z.string()),
  }),
  outputSchema: z.object({
    success: z.boolean(),
  }),
  execute: async ({ inputData }) => {
    if (!inputData.summaryText || inputData.candidateIds.length === 0) return { success: false };

    const graph = getGraphInstance();

    // Find parents
    const allParents = await graph.native.traverse(
      inputData.candidateIds,
      undefined,
      'in',
      undefined,
      undefined
    );

    const candidateSet = new Set(inputData.candidateIds);
    const externalParents = allParents.filter(p => !candidateSet.has(p));

    if (externalParents.length === 0) return { success: false };

    const summaryId = `summary:${randomUUID()}`;
    const summaryProps = {
      content: inputData.summaryText,
      source_count: inputData.candidateIds.length,
      generated_at: new Date().toISOString(),
      period_end: new Date().toISOString()
    };

    await graph.addNode(summaryId, ['Summary', 'Insight'], summaryProps);

    for (const parentId of externalParents) {
      await graph.addEdge(parentId, summaryId, 'HAS_SUMMARY');
    }

    for (const id of inputData.candidateIds) {
      await graph.deleteNode(id);
    }

    return { success: true };
  },
});

import { Workflow } from '@mastra/core/workflows';

const workflow = createWorkflow({
  id: 'metabolism-workflow',
  inputSchema: z.object({
    minAgeDays: z.number(),
    targetLabel: z.string(),
  }),
  outputSchema: z.object({
    success: z.boolean(),
  }),
})
  .then(identifyCandidates)
  .then(synthesizeInsight)
  .then(applySummary);

workflow.commit();

export { workflow as metabolismWorkflow };