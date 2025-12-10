import { createStep, createWorkflow } from '@mastra/core/workflows';
import { AISpanType } from '@mastra/core/ai-tracing';
import { z } from 'zod';
import { randomUUID } from 'node:crypto';
import type { LabyrinthCursor, LabyrinthArtifact, ThreadTrace } from '../../types';
import { RouterDecisionSchema, ScoutDecisionSchema, JudgeDecisionSchema } from '../../agent-schemas';
import { getGraphInstance } from '../../lib/graph-instance';
import { GraphTools } from '../../tools/graph-tools';
import { getSchemaRegistry } from '../../governance/schema-registry';

// --- State Schema ---
// This tracks the "memory" of the entire traversal run
const LabyrinthStateSchema = z.object({
  // Traversal State
  cursors: z.array(z.custom<LabyrinthCursor>()).default([]),
  deadThreads: z.array(z.custom<ThreadTrace>()).default([]),
  winner: z.custom<LabyrinthArtifact | null>().optional(),
  tokensUsed: z.number().default(0),

  // Governance & Config State (Persisted from init)
  domain: z.string().default('global'),
  governance: z.any().default({}),
  config: z.object({
    maxHops: z.number(),
    maxCursors: z.number(),
    confidenceThreshold: z.number(),
    timeContext: z.any().optional()
  }).optional()
});

// --- Input Schemas ---

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

// --- Step 1: Route Domain ---
// Determines the "Ghost Earth" layer (Domain) and initializes state configuration
const routeDomain = createStep({
  id: 'route-domain',
  inputSchema: WorkflowInputSchema,
  outputSchema: z.object({
    selectedDomain: z.string(),
    goal: z.string(),
    start: z.union([z.string(), z.object({ query: z.string() })])
  }),
  stateSchema: LabyrinthStateSchema,
  execute: async ({ inputData, mastra, setState, state }) => {
    const registry = getSchemaRegistry();
    const availableDomains = registry.getAllDomains();

    // 1. Setup Configuration in State
    const config = {
      maxHops: inputData.maxHops,
      maxCursors: inputData.maxCursors,
      confidenceThreshold: inputData.confidenceThreshold,
      timeContext: inputData.timeContext
    };

    let selectedDomain = inputData.domain || 'global';
    let reasoning = 'Default';
    let rejected: string[] = [];

    // 2. AI Routing (if multiple domains exist and none specified)
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
        } catch (e) { console.warn("Router failed", e); }
      }
    }

    // 3. Update Global State
    setState({
      ...state,
      domain: selectedDomain,
      governance: { query: inputData.goal, selected_domain: selectedDomain, rejected_domains: rejected, reasoning },
      config,
      // Reset counters
      tokensUsed: 0,
      cursors: [],
      deadThreads: [],
      winner: undefined
    });

    // Pass-through essential inputs for the next step in the chain
    return { selectedDomain, goal: inputData.goal, start: inputData.start };
  }
});

// --- Step 2: Initialize Cursors ---
// Bootstraps the search threads
const initializeCursors = createStep({
  id: 'initialize-cursors',
  inputSchema: z.object({
    goal: z.string(),
    start: z.union([z.string(), z.object({ query: z.string() })]),
    selectedDomain: z.string()
  }),
  outputSchema: z.object({
    cursorCount: z.number(),
    goal: z.string()
  }),
  stateSchema: LabyrinthStateSchema,
  execute: async ({ inputData, state, setState }) => {
    let startNodes: string[] = [];
    if (typeof inputData.start === 'string') {
      startNodes = [inputData.start];
    } else {
      // Future: Vector search fallback logic
      console.warn("Vector search not implemented in this workflow step yet.");
      startNodes = [];
    }

    const initialCursors: LabyrinthCursor[] = startNodes.map(nodeId => ({
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

    setState({
      ...state,
      cursors: initialCursors
    });

    return { cursorCount: initialCursors.length, goal: inputData.goal };
  }
});

// --- Step 3: Speculative Traversal ---
// The Core Loop: Runs agents, branches threads, and updates state until a winner is found or hops exhausted
const speculativeTraversal = createStep({
  id: 'speculative-traversal',
  inputSchema: z.object({
    goal: z.string(),
    cursorCount: z.number()
  }),
  outputSchema: z.object({
    foundWinner: z.boolean()
  }),
  stateSchema: LabyrinthStateSchema,
  execute: async ({ inputData, mastra, state, setState, tracingContext }) => {
    // Agents & Tools
    const scout = mastra?.getAgent('scoutAgent');
    const judge = mastra?.getAgent('judgeAgent');
    if (!scout || !judge) throw new Error("Missing agents");

    const graph = getGraphInstance();
    const tools = new GraphTools(graph);
    const registry = getSchemaRegistry();

    // Load from State
    const { goal } = inputData;
    const { domain, config } = state;
    if (!config) throw new Error("Config missing in state");

    // Local mutable copies for the loop (will sync back to state at end)
    let cursors = [...state.cursors];
    const deadThreads = [...state.deadThreads];
    let winner: LabyrinthArtifact | null = state.winner || null;
    let tokensUsed = state.tokensUsed;

    const asOfTs = config.timeContext?.asOf;
    const timeDesc = asOfTs ? `As of: ${new Date(asOfTs).toISOString()}` : '';

    // --- The Loop ---
    // Note: We loop inside the step because Mastra workflows are currently DAGs.
    // Ideally, this would be a cyclic workflow, but loop-in-step is robust for now.
    while (cursors.length > 0 && !winner) {
      const nextCursors: LabyrinthCursor[] = [];

      // Parallel execution of all active cursors
      const promises = cursors.map(async (cursor) => {
        if (winner) return; // Short circuit

        // Trace this specific thread's execution for this step
        const threadSpan = tracingContext?.currentSpan?.createChildSpan({
          type: AISpanType.GENERIC,
          name: `thread-exec-${cursor.id}`,
          metadata: {
            thread_id: cursor.id,
            step_count: cursor.stepCount,
            current_node: cursor.currentNodeId
          }
        });

        try {
          // 1. Max Hops Check
          if (cursor.stepCount >= config.maxHops) {
            deadThreads.push({ thread_id: cursor.id, status: 'KILLED', steps: cursor.stepHistory });
            threadSpan?.end({ output: { result: 'max_hops' } });
            return;
          }

          // 2. Fetch Node Metadata (LOD 1)
          const nodeMeta = await graph.match([]).where({ id: cursor.currentNodeId }).select();
          if (!nodeMeta[0]) {
            threadSpan?.end({ output: { result: 'node_not_found' } });
            return;
          }
          const currentNode = nodeMeta[0];

          // 3. Sector Scan (LOD 0) - "Satellite View"
          const allowedEdges = registry.getValidEdges(domain);
          const sectorSummary = await tools.getSectorSummary([cursor.currentNodeId], asOfTs, allowedEdges);
          const summaryList = sectorSummary.map(s => `- ${s.edgeType}: ${s.count}`).join('\n');

          // 4. Scout Decision
          const prompt = `
            Goal: "${goal}"
            Domain: "${domain}"
            Node: "${cursor.currentNodeId}" (Labels: ${JSON.stringify(currentNode.labels)})
            Path: ${JSON.stringify(cursor.path)}
            Time: "${timeDesc}"
            Moves:
            ${summaryList}
          `;

          // Note: We inject runtimeContext here to ensure tools called by Scout respect "Time" and "Domain"
          const res = await scout.generate(prompt, {
            structuredOutput: { schema: ScoutDecisionSchema },
            memory: {
              thread: cursor.id,
              resource: state.governance?.query || 'global-query'
            },
            // Pass "Ghost Earth" context to the agent runtime
            // @ts-expect-error - Mastra experimental context injection
            runtimeContext: { asOf: asOfTs, domain: domain }
          });

          // @ts-expect-error usage tracking
          if (res.usage) tokensUsed += (res.usage.promptTokens || 0) + (res.usage.completionTokens || 0);

          const decision = res.object;
          if (!decision) {
            threadSpan?.end({ output: { error: 'no_decision' } });
            return;
          }

          // Log step
          cursor.stepHistory.push({
            step: cursor.stepCount + 1,
            node_id: cursor.currentNodeId,
            action: decision.action,
            reasoning: decision.reasoning,
            ghost_view: sectorSummary.slice(0, 3).map(s => s.edgeType).join(',')
          });

          // 5. Handle Actions
          if (decision.action === 'CHECK') {
            // Judge Agent: "Street View" (LOD 2 - Full Content)
            const content = await tools.contentRetrieval([cursor.currentNodeId]);
            const jRes = await judge.generate(`Goal: ${goal}\nData: ${JSON.stringify(content)}`, {
              structuredOutput: { schema: JudgeDecisionSchema },
              memory: {
                thread: cursor.id,
                resource: state.governance?.query || 'global-query'
              }
            });
            // @ts-expect-error usage
            if (jRes.object && jRes.usage) tokensUsed += (jRes.usage.promptTokens || 0) + (jRes.usage.completionTokens || 0);

            if (jRes.object?.isAnswer && jRes.object.confidence >= config.confidenceThreshold) {
              winner = {
                answer: jRes.object.answer,
                confidence: jRes.object.confidence,
                traceId: randomUUID(),
                sources: [cursor.currentNodeId],
                metadata: {
                  duration_ms: 0,
                  tokens_used: 0,
                  governance: state.governance,
                  execution: [],
                  judgment: { verdict: jRes.object.answer, confidence: jRes.object.confidence }
                }
              };
              if (winner.metadata) winner.metadata.execution = [{ thread_id: cursor.id, status: 'COMPLETED', steps: cursor.stepHistory }];
            }
          } else if (decision.action === 'MOVE') {
            // Collect all intended moves (Primary + Alternatives)
            const moves = [];

            // 1. Primary Move
            if (decision.edgeType || decision.path) {
              moves.push({
                edgeType: decision.edgeType,
                path: decision.path,
                confidence: decision.confidence,
                reasoning: decision.reasoning
              });
            }

            // 2. Alternative Moves (Semantic Forking)
            if (decision.alternativeMoves) {
              for (const alt of decision.alternativeMoves) {
                moves.push({
                  edgeType: alt.edgeType,
                  path: undefined,
                  confidence: alt.confidence,
                  reasoning: alt.reasoning
                });
              }
            }

            // Process all moves
            for (const move of moves) {
              if (move.path) {
                // Multi-hop jump (from Navigational Map)
                const target = move.path.length > 0 ? move.path[move.path.length - 1] : undefined;
                if (target) {
                  nextCursors.push({
                    ...cursor,
                    id: randomUUID(),
                    currentNodeId: target,
                    path: [...cursor.path, ...move.path],
                    pathEdges: [...cursor.pathEdges, ...new Array(move.path.length).fill(undefined)],
                    stepCount: cursor.stepCount + move.path.length,
                    confidence: cursor.confidence * move.confidence
                  });
                }
              } else if (move.edgeType) {
                // Single-hop move
                const neighbors = await tools.topologyScan([cursor.currentNodeId], move.edgeType, asOfTs);

                // Speculative Forking: Take top 3 paths per semantic branch to handle topological ambiguity
                for (const t of neighbors.slice(0, 3)) {
                  nextCursors.push({
                    ...cursor,
                    id: randomUUID(),
                    currentNodeId: t,
                    path: [...cursor.path, t],
                    pathEdges: [...cursor.pathEdges, move.edgeType],
                    stepCount: cursor.stepCount + 1,
                    confidence: cursor.confidence * move.confidence
                  });
                }
              }

              threadSpan?.end({ output: { action: 'MOVE', branches: moves.length } });
            }
          } else {
            threadSpan?.end({ output: { action: decision.action } });
          }
        } catch (e) {
          console.warn(`Thread ${cursor.id} failed:`, e);
          deadThreads.push({ thread_id: cursor.id, status: 'KILLED', steps: cursor.stepHistory });
          // @ts-expect-error - span.error exists in runtime but typing might be strict
          threadSpan?.error({ error: e });
        }
      });

      await Promise.all(promises);
      if (winner) break;

      // 6. Pruning (Survival of the Fittest)
      nextCursors.sort((a, b) => b.confidence - a.confidence);
      // Kill excess threads
      for (let i = config.maxCursors; i < nextCursors.length; i++) {
        const c = nextCursors[i];
        if (c) deadThreads.push({ thread_id: c.id, status: 'KILLED', steps: c.stepHistory });
      }
      cursors = nextCursors.slice(0, config.maxCursors);
    }

    // Cleanup if no winner
    if (!winner) {
      cursors.forEach(c => {
        deadThreads.push({ thread_id: c.id, status: 'KILLED', steps: c.stepHistory });
      });
      cursors = []; // Clear active
    }

    // 7. Update State
    setState({
      ...state,
      cursors, // Should be empty if no winner, or active if paused? Logic here assumes run-to-completion.
      deadThreads,
      winner: winner || undefined,
      tokensUsed
    });

    return { foundWinner: !!winner };
  }
});

// --- Step 4: Finalize Artifact ---
// Compiles the final report and metadata
const finalizeArtifact = createStep({
  id: 'finalize-artifact',
  inputSchema: z.object({
    foundWinner: z.boolean()
  }),
  stateSchema: LabyrinthStateSchema,
  outputSchema: z.object({
    artifact: z.custom<LabyrinthArtifact | null>()
  }),
  execute: async ({ state }) => {
    if (!state.winner || !state.winner.metadata) return { artifact: null };

    const w = state.winner;
    if (w.metadata) {
      w.metadata.tokens_used = state.tokensUsed;
      // Attach a few dead threads for debugging context
      w.metadata.execution.push(...state.deadThreads.slice(-5));
    }
    return { artifact: w };
  }
});

// --- Step 5: Pheromone Reinforcement ---
// Heats up the edges of the winning path to guide future agents
const reinforcePath = createStep({
  id: 'reinforce-path',
  inputSchema: z.object({
    artifact: z.custom<LabyrinthArtifact | null>()
  }),
  stateSchema: LabyrinthStateSchema,
  outputSchema: z.object({ success: z.boolean() }),
  execute: async ({ state, tracingContext }) => {
    if (!state.winner || !state.winner.sources) return { success: false };

    // Find the cursor that produced the winner
    const winningCursor = state.cursors.find(c => state.winner?.sources.includes(c.currentNodeId));
    if (winningCursor) {
      const graph = getGraphInstance();
      const tools = new GraphTools(graph);

      const span = tracingContext?.currentSpan?.createChildSpan({
        type: AISpanType.GENERIC,
        name: 'apply-pheromones',
        metadata: { path_length: winningCursor.path.length }
      });

      try {
        await tools.reinforcePath(winningCursor.path, winningCursor.pathEdges, state.winner.confidence);
        span?.end();
      } catch (e) {
        // @ts-expect-error - span.error usage
        span?.error({ error: e });
        throw e;
      }
      return { success: true };
    }

    return { success: false };
  }
});

// --- Workflow Definition ---

export const labyrinthWorkflow = createWorkflow({
  id: 'labyrinth-workflow',
  description: 'Agentic Labyrinth Traversal with Parallel Speculation',
  inputSchema: WorkflowInputSchema,
  outputSchema: z.object({ artifact: z.custom<LabyrinthArtifact | null>() }),
  stateSchema: LabyrinthStateSchema,
})
  .then(routeDomain)
  .then(initializeCursors)
  .then(speculativeTraversal)
  .then(finalizeArtifact)
  .then(reinforcePath)
  .commit();