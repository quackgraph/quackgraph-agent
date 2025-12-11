import { createStep, createWorkflow } from '@mastra/core/workflows';
import { z } from 'zod';
import { getGraphInstance } from '../../lib/graph-instance';
import { ScribeDecisionSchema } from '../../agent-schemas';

// Input Schema
const MutationInputSchema = z.object({
  query: z.string(),
  traceId: z.string().optional(),
  userId: z.string().optional().default('Me'),
  asOf: z.number().optional()
});

// Step 1: Scribe Analysis (Intent -> Operations)
const analyzeIntent = createStep({
  id: 'analyze-intent',
  inputSchema: MutationInputSchema,
  outputSchema: z.object({
    operations: z.array(z.any()),
    reasoning: z.string(),
    requiresClarification: z.string().optional()
  }),
  execute: async ({ inputData, mastra }) => {
    const scribe = mastra?.getAgent('scribeAgent');
    if (!scribe) throw new Error("Scribe Agent not found");

    const now = inputData.asOf ? new Date(inputData.asOf) : new Date();
    
    // Prompt Scribe
    const prompt = `
      User Query: "${inputData.query}"
      Context User ID: "${inputData.userId}"
      System Time: ${now.toISOString()}
    `;

    const res = await scribe.generate(prompt, {
      structuredOutput: { schema: ScribeDecisionSchema },
      // Inject context for tools (Time Travel & Governance)
      // @ts-expect-error - Mastra context injection
      runtimeContext: { asOf: inputData.asOf } 
    });

    const decision = res.object;
    if (!decision) throw new Error("Scribe returned no structured decision");

    return {
      operations: decision.operations,
      reasoning: decision.reasoning,
      requiresClarification: decision.requiresClarification
    };
  }
});

// Step 2: Apply Mutations (Batch Execution)
const applyMutations = createStep({
  id: 'apply-mutations',
  inputSchema: z.object({
    operations: z.array(z.any()),
    reasoning: z.string(),
    requiresClarification: z.string().optional()
  }),
  outputSchema: z.object({
    success: z.boolean(),
    summary: z.string()
  }),
  execute: async ({ inputData }) => {
    if (inputData.requiresClarification) {
      return { success: false, summary: `Clarification needed: ${inputData.requiresClarification}` };
    }

    const graph = getGraphInstance();
    const ops = inputData.operations;

    if (!ops || !Array.isArray(ops)) {
        return { success: false, summary: "No operations returned by agent." };
    }
    
    // Arrays for Batching
    // biome-ignore lint/suspicious/noExplicitAny: Batch types
    const nodesToAdd: any[] = [];
    // biome-ignore lint/suspicious/noExplicitAny: Batch types
    const edgesToAdd: any[] = [];
    
    const summaryLines: string[] = [];
    
    for (const op of ops) {
      const validFrom = op.validFrom ? new Date(op.validFrom) : undefined;
      const validTo = op.validTo ? new Date(op.validTo) : undefined;

      try {
        switch (op.op) {
          case 'CREATE_NODE': {
            const id = op.id || crypto.randomUUID();
            nodesToAdd.push({
              id,
              labels: op.labels,
              properties: op.properties,
              validFrom,
              validTo
            });
            summaryLines.push(`Created Node ${id} (${op.labels.join(',')})`);
            break;
          }
          case 'CREATE_EDGE': {
            edgesToAdd.push({
              source: op.source,
              target: op.target,
              type: op.type,
              properties: op.properties || {},
              validFrom,
              validTo
            });
            summaryLines.push(`Created Edge ${op.source}->${op.target} [${op.type}]`);
            break;
          }
          case 'UPDATE_NODE': {
            // Fetch label if needed for optimization, or pass generic
            // For now, we assume simple properties update.
            // If the schema requires label, we find it.
            let label = 'Entity'; // Fallback
            const existing = await graph.db.query('SELECT labels FROM nodes WHERE id = ?', [op.match.id]);
            if (existing.length > 0 && existing[0].labels && existing[0].labels.length > 0) {
                label = existing[0].labels[0];
            }

            await graph.mergeNode(
              label, 
              op.match, 
              op.set, 
              { validFrom }
            );
            summaryLines.push(`Updated Node ${op.match.id}`);
            break;
          }
          case 'DELETE_NODE': {
              // Direct DB manipulation for retroactive delete if needed
              if (validTo) {
                   await graph.db.execute(
                       `UPDATE nodes SET valid_to = ? WHERE id = ? AND valid_to IS NULL`, 
                       [validTo.toISOString(), op.id]
                   );
                   // Update RAM
                   graph.native.removeNode(op.id);
              } else {
                  await graph.deleteNode(op.id);
              }
              summaryLines.push(`Deleted Node ${op.id}`);
              break;
          }
          case 'CLOSE_EDGE': {
              if (validTo) {
                   await graph.db.execute(
                       `UPDATE edges SET valid_to = ? WHERE source = ? AND target = ? AND type = ? AND valid_to IS NULL`, 
                       [validTo.toISOString(), op.source, op.target, op.type]
                   );
                   graph.native.removeEdge(op.source, op.target, op.type);
              } else {
                  await graph.deleteEdge(op.source, op.target, op.type);
              }
              summaryLines.push(`Closed Edge ${op.source}->${op.target} [${op.type}]`);
              break;
          }
        }
      } catch (e) {
          console.error(`Failed to apply operation ${op.op}:`, e);
          summaryLines.push(`FAILED: ${op.op} - ${(e as Error).message}`);
      }
    }

    // Execute Batches
    if (nodesToAdd.length > 0) {
        await graph.addNodes(nodesToAdd);
    }
    if (edgesToAdd.length > 0) {
        await graph.addEdges(edgesToAdd);
    }

    return { success: true, summary: summaryLines.join('\n') };
  }
});

export const mutationWorkflow = createWorkflow({
  id: 'mutation-workflow',
  inputSchema: MutationInputSchema,
  outputSchema: z.object({
    success: z.boolean(),
    summary: z.string()
  })
})
.then(analyzeIntent)
.then(applyMutations)
.commit();