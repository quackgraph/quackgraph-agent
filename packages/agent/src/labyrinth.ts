import type { QuackGraph } from '@quackgraph/graph';
import type {
  AgentConfig,
  LabyrinthArtifact,
  CorrelationResult,
  TimeContext,
  DomainConfig,
  MastraAgent
} from './types';
import { randomUUID } from 'node:crypto';
import { trace, type Span } from '@opentelemetry/api';

import { setGraphInstance } from './lib/graph-instance';
import { mastra } from './mastra';

// Tools / Helpers
import { Chronos } from './agent/chronos';
import { GraphTools } from './tools/graph-tools';
import { SchemaRegistry } from './governance/schema-registry';

// Schemas
import {
  RouterDecisionSchema,
  ScoutDecisionSchema,
  JudgeDecisionSchema
} from './agent-schemas';

interface Cursor {
  id: string;
  currentNodeId: string;
  path: string[]; // History of node IDs
  pathEdges: (string | undefined)[]; // History of edges taken (index 0 is undefined)
  traceHistory: string[]; // Summary of actions for context
  stepCount: number;
  confidence: number;
  lastEdgeType?: string; // Edge taken to reach currentNodeId
  lastTimestamp?: number; // Microseconds timestamp of the current node (for Causal enforcement)
}

export class Labyrinth {
  public chronos: Chronos;
  public tools: GraphTools;
  public registry: SchemaRegistry;
  private logger = mastra.getLogger();
  private tracer = trace.getTracer('quackgraph-agent');

  constructor(
    private graph: QuackGraph,
    private agents: {
      scout: MastraAgent;
      judge: MastraAgent;
      router: MastraAgent;
    },
    private config: AgentConfig
  ) {
    // Initialize Graph Singleton for tool access if needed
    setGraphInstance(graph);

    this.tools = new GraphTools(graph);
    this.chronos = new Chronos(graph, this.tools);

    // Governance
    this.registry = new SchemaRegistry();
  }

  /**
   * Register a new Domain (Semantic Lens) for the agent to use.
   */
  registerDomain(config: DomainConfig) {
    this.registry.register(config);
  }

  /**
   * Parallel Speculative Execution:
   * Main Entry Point: Orchestrates multiple Scouts and a Judge to find an answer.
   */
  async findPath(
    start: string | { query: string },
    goal: string,
    timeContext?: TimeContext
  ): Promise<LabyrinthArtifact | null> {
    const rootController = new AbortController();
    const rootSignal = rootController.signal;

    return this.tracer.startActiveSpan('labyrinth.find_path', async (span: Span) => {
      span.setAttribute('labyrinth.goal', goal);
      const traceId = span.spanContext().traceId;
      const parentSpanId = span.spanContext().spanId;

      this.logger.info(`Starting Labyrinth trace: ${goal}`, { traceId, goal });

      let foundArtifact: { artifact: LabyrinthArtifact; cursor: Cursor } | null = null;

      try {
        // --- Phase 0: Context Firewall (Routing) ---
        const availableDomains = this.registry.getAllDomains();
        let activeDomain = 'global';

        if (availableDomains.length > 1) {
          const domainDescriptions = availableDomains
            .map(d => `- "${d.name}": ${d.description}`)
            .join('\n');

          const routerPrompt = `
              Goal: "${goal}"
              Available Domains (Lenses):
              ${domainDescriptions}
            `;

          try {
            const res = await this.agents.router.generate(routerPrompt, {
              abortSignal: rootSignal,
              tracingOptions: { traceId, parentSpanId },
              structuredOutput: {
                schema: RouterDecisionSchema
              }
            });

            // Use structured output
            const decision = res.object;
            if (decision) {
              const valid = availableDomains.find(
                d => d.name.toLowerCase() === decision.domain.toLowerCase()
              );
              if (valid) activeDomain = decision.domain;
              this.logger.debug(`Router selected domain: ${activeDomain}`, { traceId, domain: activeDomain });
            }

            span.setAttribute('labyrinth.active_domain', activeDomain);
          } catch (e) {
            if (!rootSignal.aborted) this.logger.warn('Routing failed, defaulting to global', { traceId, error: e });
          }
        }

        // Resolve effective timestamp (passed context > graph context > current)
        const effectiveAsOf = timeContext?.asOf || this.graph.context.asOf;
        const asOfTs = effectiveAsOf ? effectiveAsOf.getTime() * 1000 : undefined;

        const timeDesc = effectiveAsOf
          ? `As of: ${effectiveAsOf.toISOString()}`
          : timeContext?.windowStart
            ? `Window: ${timeContext.windowStart.toISOString()} - ${timeContext.windowEnd?.toISOString()}`
            : undefined;

        let startNodes: string[] = [];

        // Vector Genesis
        if (typeof start === 'object' && 'query' in start) {
          if (!this.config.embeddingProvider) {
            throw new Error('Vector Genesis requires an embeddingProvider in AgentConfig.');
          }
          const vector = await this.config.embeddingProvider.embed(start.query);
          // Get top 3 relevant start nodes
          const matches = await this.graph.match([]).nearText(vector).limit(3).select();
          startNodes = matches.map(m => m.id);
        } else {
          startNodes = [start as string];
        }

        if (startNodes.length === 0) {
          span.setAttribute('labyrinth.outcome', 'EXHAUSTED');
          return null;
        }


        // Initialize Root Cursor
        let cursors: Cursor[] = startNodes.map(nodeId => ({
          id: randomUUID(),
          currentNodeId: nodeId,
          path: [nodeId],
          pathEdges: [undefined],
          traceHistory: [`[ROUTER] Selected domain: ${activeDomain}`],
          stepCount: 0,
          confidence: 1.0
        }));

        const maxHops = this.config.maxHops || 10;
        const maxCursors = this.config.maxCursors || 3;


        // Speculative Execution Loop
        while (cursors.length > 0 && !foundArtifact && !rootSignal.aborted) {
          const nextCursors: Cursor[] = [];
          const processingPromises: Promise<void>[] = [];

          for (const cursor of cursors) {
            if (foundArtifact || rootSignal.aborted) break;

            const task = async () => {
              if (foundArtifact || rootSignal.aborted) return;

              // Create a span for this cursor's step
              await this.tracer.startActiveSpan('labyrinth.cursor_step', async (cursorSpan) => {
                cursorSpan.setAttribute('cursor.id', cursor.id);
                cursorSpan.setAttribute('cursor.node_id', cursor.currentNodeId);
                cursorSpan.setAttribute('cursor.step_count', cursor.stepCount);

                try {
                  // Pruning: Max Depth
                  if (cursor.stepCount >= maxHops) {
                    cursorSpan.addEvent('Max hops reached');
                    return;
                  }

                  // 1. Context Awareness (LOD 1)
                  const nodeMeta = await this.graph
                    .match([])
                    .where({ id: cursor.currentNodeId })
                    .select(
                      "id, labels, date_diff('us', '1970-01-01'::TIMESTAMPTZ, valid_from)::DOUBLE as valid_from_micros"
                    );

                  if (nodeMeta.length === 0) return; // Node lost/deleted
                  // biome-ignore lint/suspicious/noExplicitAny: raw sql result
                  const currentNode = nodeMeta[0] as any;
                  if (!currentNode) return;

                  // 2. Sector Scan (LOD 0) - Enhanced Satellite View
                  // Governance: Get effective whitelist
                  const allowedEdges = this.registry.getValidEdges(activeDomain);

                  const sectorSummary = await this.tools.getSectorSummary(
                    [cursor.currentNodeId],
                    asOfTs,
                    allowedEdges
                  );
                  const summaryList = sectorSummary
                    .map(
                      s => `- ${s.edgeType}: ${s.count} nodes${(s.avgHeat ?? 0) > 50 ? ' 🔥' : ''}`
                    )
                    .join('\n');

                  const validMovesText = allowedEdges
                    ? `Valid Moves for ${activeDomain}: [${allowedEdges.join(', ')}]`
                    : 'Valid Moves: ALL';

                  // 3. Ask Scout
                  const scoutPrompt = `
                    Goal: "${goal}"
                    activeDomain: "${activeDomain}"
                    currentNodeId: "${currentNode.id}"
                    currentNodeLabels: ${JSON.stringify(currentNode.labels || [])}
                    pathHistory: ${JSON.stringify(cursor.path)}
                    timeContext: "${timeDesc || ''}"
                    ${validMovesText}
                    
                    Satellite View (Available Moves):
                    ${summaryList}
                  `;

                  // biome-ignore lint/suspicious/noExplicitAny: decision blob
                  let decision: any;
                  try {
                    const res = await this.agents.scout.generate(scoutPrompt, {
                      abortSignal: rootSignal,
                      tracingOptions: { traceId, parentSpanId: cursorSpan.spanContext().spanId },
                      structuredOutput: {
                        schema: ScoutDecisionSchema
                      }
                    });

                    decision = res.object;

                    // Trace tool results if any (post-hoc observability)
                    if (res.toolResults && res.toolResults.length > 0) {
                      cursorSpan.setAttribute('scout.tool_calls_count', res.toolResults.length);
                      cursorSpan.addEvent('scout_tool_execution', {
                        results: JSON.stringify(res.toolResults)
                      });
                    }

                  } catch (e) {
                    if (!rootSignal.aborted) this.logger.warn('Scout failed to decide', { traceId, cursorId: cursor.id, error: e });
                    return;
                  }

                  if (!decision) return;

                  cursor.traceHistory.push(`[${decision.action}] ${decision.reasoning}`);
                  cursorSpan.setAttribute('scout.action', decision.action);
                  cursorSpan.setAttribute('scout.reasoning', decision.reasoning);

                  // 4. Handle Decision
                  if (decision.action === 'CHECK') {
                    // LOD 2: Content Retrieval
                    const content = await this.tools.contentRetrieval([cursor.currentNodeId]);

                    // Ask Judge
                    const judgePrompt = `
                      Goal: "${goal}"
                      Data: ${JSON.stringify(content)}
                      Time Context: "${timeDesc || ''}"
                    `;

                    try {
                      const res = await this.agents.judge.generate(judgePrompt, {
                        abortSignal: rootSignal,
                        tracingOptions: { traceId, parentSpanId: cursorSpan.spanContext().spanId },
                        structuredOutput: {
                          schema: JudgeDecisionSchema
                        }
                      });

                      const artifact = res.object;

                      if (
                        artifact?.isAnswer &&
                        artifact.confidence >= (this.config.confidenceThreshold || 0.7)
                      ) {
                        const finalArtifact = {
                          ...artifact,
                          traceId,
                          sources: [cursor.currentNodeId]
                        };
                        foundArtifact = { artifact: finalArtifact, cursor };
                        span.setAttribute('labyrinth.outcome', 'FOUND');
                        rootController.abort(); // KILL SWITCH
                        return;
                      }
                    } catch (_e) {
                      /* ignore judge fail */
                    }

                    return;
                  } else if (decision.action === 'MATCH') {
                    // Structural Inference
                    if (!decision.pattern) return; // Should be enforced by Zod

                    const matches = await this.tools.findPattern(
                      [cursor.currentNodeId],
                      decision.pattern,
                      asOfTs
                    );

                    if (matches.length > 0) {
                      const foundPaths = matches.slice(0, 3);
                      for (const path of foundPaths) {
                        const endNode = path[path.length - 1];
                        if (endNode) {
                          // Note: path edges are not fully returned by matchPattern yet, simple appending
                          nextCursors.push({
                            id: randomUUID(),
                            currentNodeId: endNode,
                            path: [...cursor.path, ...path.slice(1)],
                            pathEdges: [...cursor.pathEdges, ...new Array(path.length - 1).fill('MATCH_JUMP')],
                            traceHistory: [...cursor.traceHistory, `[MATCH] Pattern Found`],
                            stepCount: cursor.stepCount + path.length - 1,
                            confidence: cursor.confidence * 0.9,
                            lastEdgeType: 'MATCH_JUMP'
                          });
                        }
                      }
                    }
                    return;
                  } else if (decision.action === 'MOVE') {
                    // biome-ignore lint/suspicious/noExplicitAny: flexible move structure
                    const moves: any[] = [];
                    // Using discriminated union access
                    if (decision.edgeType)
                      moves.push({ edge: decision.edgeType, conf: decision.confidence });
                    if (decision.alternativeMoves) {
                      for (const alt of decision.alternativeMoves) {
                        moves.push({ edge: alt.edgeType, conf: alt.confidence });
                      }
                    }

                    const isCausal = this.registry.isDomainCausal(activeDomain);
                    const minValidFrom = isCausal
                      ? cursor.lastTimestamp || currentNode.valid_from_micros
                      : undefined;

                    for (const move of moves) {
                      if (move.conf < (this.config.confidenceThreshold || 0.2)) continue;
                      if (!this.registry.isEdgeAllowed(activeDomain, move.edge)) continue;

                      const nextNodes = await this.tools.topologyScan(
                        [cursor.currentNodeId],
                        move.edge,
                        asOfTs,
                        minValidFrom
                      );

                      if (nextNodes.length > 0) {
                        const targets = nextNodes.slice(0, 3);
                        for (const target of targets) {
                          nextCursors.push({
                            id: randomUUID(),
                            currentNodeId: target,
                            path: [...cursor.path, target],
                            pathEdges: [...cursor.pathEdges, move.edge],
                            traceHistory: [...cursor.traceHistory],
                            stepCount: cursor.stepCount + 1,
                            confidence: cursor.confidence * move.conf,
                            lastEdgeType: move.edge
                          });
                        }
                      }
                    }
                    return;
                  }
                } finally {
                  cursorSpan.end();
                }
              });
            };

            processingPromises.push(task());
          }

          // Wait for batch to settle, but we might have aborted early
          await Promise.allSettled(processingPromises);

          if (foundArtifact) {
            break;
          }

          // Pruning
          nextCursors.sort((a, b) => b.confidence - a.confidence);
          cursors = nextCursors.slice(0, maxCursors);
        }

        if (foundArtifact) {
          const { artifact, cursor } = foundArtifact;

          // Reconstruct path for reinforcement using cursor history
          const pathTrace = cursor.path.map((nodeId, idx) => ({
            source: nodeId,
            incomingEdge: cursor.pathEdges[idx]
          }));

          await this.tools.reinforcePath(pathTrace, artifact.confidence);

          return artifact;
        }

        span.setAttribute('labyrinth.outcome', 'EXHAUSTED');
        return null;

      } finally {
        if (!foundArtifact && !rootSignal.aborted) {
          rootController.abort(); // Cleanup
        }
      }
    });
  }

  // --- Wrapper for Chronos ---
  async analyzeCorrelation(
    anchorNodeId: string,
    targetLabel: string,
    windowMinutes: number
  ): Promise<CorrelationResult> {
    return this.chronos.analyzeCorrelation(anchorNodeId, targetLabel, windowMinutes);
  }
}