import { createStep, createWorkflow } from '@mastra/core/workflows';
import { z } from 'zod';
import { randomUUID } from 'node:crypto';
import type { LabyrinthCursor, LabyrinthArtifact, ThreadTrace } from '../../types';
import { RouterDecisionSchema, ScoutDecisionSchema, JudgeDecisionSchema } from '../../agent-schemas';
import { getGraphInstance } from '../../lib/graph-instance';
import { GraphTools } from '../../tools/graph-tools';
import { getSchemaRegistry } from '../../governance/schema-registry';

// Types
const WorkflowInputSchema = z.object({
  goal: z.string(),
  start: z.union([z.string(), z.object({ query: z.string() })]),
  domain: z.string().optional(),
  maxHops: z.number().optional().default(10),
  maxCursors: z.number().optional().default(3),
  confidenceThreshold: z.number().optional().default(0.7),
  timeContext: z.object({
    asOf: z.number().optional(),
    windowStart: z.string().optional(),
    windowEnd: z.string().optional()
  }).optional()
});

// Step 1: Route
const routeDomain = createStep({
  id: 'route-domain',
  inputSchema: WorkflowInputSchema,
  outputSchema: z.object({
    domain: z.string(),
    governance: z.any(),
    // Pass-through
    goal: z.string(),
    start: z.union([z.string(), z.object({ query: z.string() })]),
    config: z.any() 
  }),
  execute: async ({ inputData, mastra }) => {
    const registry = getSchemaRegistry();
    const availableDomains = registry.getAllDomains();
    
    // Pass-through config
    const config = {
      maxHops: inputData.maxHops,
      maxCursors: inputData.maxCursors,
      confidenceThreshold: inputData.confidenceThreshold,
      timeContext: inputData.timeContext
    };

    let selectedDomain = inputData.domain || 'global';
    let reasoning = 'Default';
    let rejected: string[] = [];

    if (availableDomains.length > 1 && !inputData.domain) {
      const router = mastra?.getAgent('routerAgent');
      if (router) {
        const descriptions = availableDomains.map(d => `- ${d.name}: ${d.description}`).join('\n');
        const prompt = `Goal: "${inputData.goal}"\nAvailable Domains:\n${descriptions}`;
        try {
          const res = await router.generate(prompt, { structuredOutput: { schema: RouterDecisionSchema } });
          const decision = res.object;
          if (decision) {
             const valid = availableDomains.find(d => d.name.toLowerCase() === decision.domain.toLowerCase());
             if (valid) selectedDomain = decision.domain;
             reasoning = decision.reasoning;
             rejected = availableDomains.map(d => d.name).filter(n => n.toLowerCase() !== selectedDomain.toLowerCase());
          }
        } catch(e) { console.warn("Router failed", e); }
      }
    }

    return {
      domain: selectedDomain,
      governance: { query: inputData.goal, selected_domain: selectedDomain, rejected_domains: rejected, reasoning },
      goal: inputData.goal,
      start: inputData.start,
      config
    };
  }
});

// Step 2: Initialize
const initializeCursors = createStep({
  id: 'initialize-cursors',
  inputSchema: z.object({
    domain: z.string(),
    governance: z.any(),
    goal: z.string(),
    start: z.union([z.string(), z.object({ query: z.string() })]),
    config: z.any()
  }),
  outputSchema: z.object({
    cursors: z.array(z.custom<LabyrinthCursor>()),
    domain: z.string(),
    governance: z.any(),
    goal: z.string(),
    config: z.any()
  }),
  execute: async ({ inputData }) => {
    let startNodes: string[] = [];
    if (typeof inputData.start === 'string') {
      startNodes = [inputData.start];
    } else {
       // Vector search fallback logic would go here
       console.warn("Vector search not implemented in this workflow step yet.");
       startNodes = [];
    }

    const cursors: LabyrinthCursor[] = startNodes.map(nodeId => ({
      id: randomUUID().slice(0, 8),
      currentNodeId: nodeId,
      path: [nodeId],
      pathEdges: [undefined],
      stepHistory: [{
        step: 0,
        node_id: nodeId,
        action: 'START',
        reasoning: 'Initialized',
        ghost_view: 'N/A'
      }],
      stepCount: 0,
      confidence: 1.0
    }));

    return {
      cursors,
      domain: inputData.domain,
      governance: inputData.governance,
      goal: inputData.goal,
      config: inputData.config
    };
  }
});

// Step 3: Traverse
const speculativeTraversal = createStep({
  id: 'speculative-traversal',
  inputSchema: z.object({
    cursors: z.array(z.custom<LabyrinthCursor>()),
    domain: z.string(),
    governance: z.any(),
    goal: z.string(),
    config: z.any()
  }),
  outputSchema: z.object({
    winner: z.custom<LabyrinthArtifact | null>(),
    deadThreads: z.array(z.custom<ThreadTrace>()),
    tokensUsed: z.number(),
    governance: z.any()
  }),
  execute: async ({ inputData, mastra }) => {
    const graph = getGraphInstance();
    const tools = new GraphTools(graph);
    const registry = getSchemaRegistry();
    const scout = mastra?.getAgent('scoutAgent');
    const judge = mastra?.getAgent('judgeAgent');
    
    if (!scout || !judge) throw new Error("Missing agents");

    const { goal, domain, config } = inputData;
    let cursors = inputData.cursors;
    const deadThreads: ThreadTrace[] = [];
    let winner: LabyrinthArtifact | null = null;
    let tokensUsed = 0;
    
    const asOfTs = config.timeContext?.asOf;
    const timeDesc = asOfTs ? `As of: ${new Date(asOfTs).toISOString()}` : '';

    while (cursors.length > 0 && !winner) {
      const nextCursors: LabyrinthCursor[] = [];
      const promises = cursors.map(async (cursor) => {
        if (winner) return;
        if (cursor.stepCount >= config.maxHops) {
           deadThreads.push({ thread_id: cursor.id, status: 'KILLED', steps: cursor.stepHistory });
           return;
        }

        const nodeMeta = await graph.match([]).where({ id: cursor.currentNodeId }).select();
        if (!nodeMeta[0]) return;
        const currentNode = nodeMeta[0];

        const allowedEdges = registry.getValidEdges(domain);
        const sectorSummary = await tools.getSectorSummary([cursor.currentNodeId], asOfTs, allowedEdges);
        const summaryList = sectorSummary.map(s => `- ${s.edgeType}: ${s.count}`).join('\n');
        
        const prompt = `
          Goal: "${goal}"
          Domain: "${domain}"
          Node: "${cursor.currentNodeId}" (Labels: ${JSON.stringify(currentNode.labels)})
          Path: ${JSON.stringify(cursor.path)}
          Time: "${timeDesc}"
          Moves:
          ${summaryList}
        `;

        try {
            const res = await scout.generate(prompt, { 
                structuredOutput: { schema: ScoutDecisionSchema },
                memory: {
                    thread: cursor.id,
                    resource: inputData.governance?.query || 'global-query'
                },
                // @ts-expect-error - Injecting runtime context for tools (Mastra experimental/custom support)
                runtimeContext: { asOf: asOfTs, domain }
            });
            // @ts-expect-error usage
            if (res.usage) tokensUsed += (res.usage.promptTokens||0) + (res.usage.completionTokens||0);
            const decision = res.object;
            if (!decision) return;

            cursor.stepHistory.push({
                step: cursor.stepCount + 1,
                node_id: cursor.currentNodeId,
                action: decision.action,
                reasoning: decision.reasoning,
                ghost_view: sectorSummary.slice(0,3).map(s=>s.edgeType).join(',')
            });

            if (decision.action === 'CHECK') {
                const content = await tools.contentRetrieval([cursor.currentNodeId]);
                const jRes = await judge.generate(`Goal: ${goal}\nData: ${JSON.stringify(content)}`, { 
                    structuredOutput: { schema: JudgeDecisionSchema },
                    memory: {
                        thread: cursor.id,
                        resource: inputData.governance?.query || 'global-query'
                    }
                });
                 // @ts-expect-error usage
                if (jRes.object && jRes.usage) tokensUsed += (jRes.usage.promptTokens||0) + (jRes.usage.completionTokens||0);
                if (jRes.object?.isAnswer && jRes.object.confidence >= config.confidenceThreshold) {
                    winner = {
                        answer: jRes.object.answer,
                        confidence: jRes.object.confidence,
                        traceId: randomUUID(),
                        sources: [cursor.currentNodeId],
                        metadata: {
                            duration_ms: 0,
                            tokens_used: 0,
                            governance: inputData.governance,
                            execution: [],
                            judgment: { verdict: jRes.object.answer, confidence: jRes.object.confidence }
                        }
                    };
                     if (winner.metadata) winner.metadata.execution = [{ thread_id: cursor.id, status: 'COMPLETED', steps: cursor.stepHistory }];
                }
            } else if (decision.action === 'MOVE' && (decision.edgeType || decision.path)) {
                // Simplified move logic
                if (decision.path) {
                     const target = decision.path.length > 0 ? decision.path[decision.path.length-1] : undefined;
                     if (target) {
                        nextCursors.push({ ...cursor, id: randomUUID(), currentNodeId: target, path: [...cursor.path, ...decision.path], stepCount: cursor.stepCount + decision.path.length, confidence: cursor.confidence * decision.confidence });
                     }
                } else if (decision.edgeType) {
                     const neighbors = await tools.topologyScan([cursor.currentNodeId], decision.edgeType, asOfTs);
                     for (const t of neighbors.slice(0, 2)) {
                        nextCursors.push({ ...cursor, id: randomUUID(), currentNodeId: t, path: [...cursor.path, t], stepCount: cursor.stepCount+1, confidence: cursor.confidence * decision.confidence });
                     }
                }
            }
        } catch(_e) { return; }
      });

      await Promise.all(promises);
      if (winner) break;

      nextCursors.sort((a,b) => b.confidence - a.confidence);
      for(let i=config.maxCursors; i<nextCursors.length; i++) {
        const c = nextCursors[i];
        if (c) deadThreads.push({ thread_id: c.id, status: 'KILLED', steps: c.stepHistory });
      }
      cursors = nextCursors.slice(0, config.maxCursors);
    }
    
    if(!winner) {
        cursors.forEach(c => {
            deadThreads.push({ thread_id: c.id, status: 'KILLED', steps: c.stepHistory });
        });
    }

    return { winner, deadThreads, tokensUsed, governance: inputData.governance };
  }
});

// Step 4: Finalize
const finalizeArtifact = createStep({
  id: 'finalize-artifact',
  inputSchema: z.object({
    winner: z.custom<LabyrinthArtifact | null>(),
    deadThreads: z.array(z.custom<ThreadTrace>()),
    tokensUsed: z.number(),
    governance: z.any()
  }),
  outputSchema: z.object({
    artifact: z.custom<LabyrinthArtifact | null>()
  }),
  execute: async ({ inputData }) => {
    if (!inputData.winner || !inputData.winner.metadata) return { artifact: null };
    const w = inputData.winner;
    if (w.metadata) {
        w.metadata.tokens_used = inputData.tokensUsed;
        w.metadata.execution.push(...inputData.deadThreads.slice(-5));
    }
    return { artifact: w };
  }
});

export const labyrinthWorkflow = createWorkflow({
  id: 'labyrinth-workflow',
  description: 'Agentic Labyrinth Traversal',
  inputSchema: WorkflowInputSchema,
  outputSchema: z.object({ artifact: z.custom<LabyrinthArtifact | null>() })
})
  .then(routeDomain)
  .then(initializeCursors)
  .then(speculativeTraversal)
  .then(finalizeArtifact)
  .commit();