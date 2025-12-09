import { QuackGraph } from '@quackgraph/graph';
import type {
  AgentConfig,
  TraceStep,
  TraceLog,
  LabyrinthArtifact,
  ScoutPrompt,
  JudgePrompt,
  CorrelationResult,
  TimeContext,
  DomainConfig,
  MastraAgent
} from './types';
import { randomUUID } from 'crypto';

import { setGraphInstance } from './lib/graph-instance';

// Tools / Helpers
import { Chronos } from './agent/chronos';
import { GraphTools } from './tools/graph-tools';
import { SchemaRegistry } from './governance/schema-registry';

interface Cursor {
  id: string;
  currentNodeId: string;
  path: string[]; // History of node IDs
  traceHistory: string[]; // Summary of actions for context
  stepCount: number;
  confidence: number;
  parentId?: number; // stepId of the action that led here
  lastEdgeType?: string; // Edge taken to reach currentNodeId
  lastTimestamp?: number; // Microseconds timestamp of the current node (for Causal enforcement)
}

export class Labyrinth {
  private traces = new Map<string, TraceLog>();

  public chronos: Chronos;
  public tools: GraphTools;
  public registry: SchemaRegistry;

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

    const traceId = randomUUID();
    const startTime = Date.now();

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
        const res = await this.agents.router.generate(routerPrompt, { abortSignal: rootSignal });
        const jsonStr = res.text.match(/\{[\s\S]*\}/)?.[0] || res.text;
        const decision = JSON.parse(jsonStr);
        const valid = availableDomains.find(
          d => d.name.toLowerCase() === decision.domain.toLowerCase()
        );
        if (valid) activeDomain = decision.domain;
      } catch (e) {
        if (!rootSignal.aborted) console.warn('Routing failed, defaulting to global', e);
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

    const log: TraceLog = {
      traceId,
      goal,
      activeDomain,
      startTime,
      steps: [],
      outcome: 'ABORTED' // Default
    };

    this.traces.set(traceId, log);

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
      log.outcome = 'EXHAUSTED';
      return null;
    }

    // Initialize Root Cursor
    let cursors: Cursor[] = startNodes.map(nodeId => ({
      id: randomUUID(),
      currentNodeId: nodeId,
      path: [nodeId],
      traceHistory: [`[ROUTER] Selected domain: ${activeDomain}`],
      stepCount: 0,
      confidence: 1.0
    }));

    const maxHops = this.config.maxHops || 10;
    const maxCursors = this.config.maxCursors || 3;
    let globalStepCounter = 0;
    // biome-ignore lint/suspicious/noExplicitAny: internal result carrier
    let foundArtifact: { type: 'FOUND'; artifact: LabyrinthArtifact; finalStepId: number } | null = null;

    // Speculative Execution Loop
    try {
      while (cursors.length > 0 && !foundArtifact && !rootSignal.aborted) {
        const nextCursors: Cursor[] = [];
        const processingPromises: Promise<void>[] = [];

        for (const cursor of cursors) {
          if (foundArtifact || rootSignal.aborted) break;

          const task = async () => {
            if (foundArtifact || rootSignal.aborted) return;

            // Pruning: Max Depth
            if (cursor.stepCount >= maxHops) return;

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
              const res = await this.agents.scout.generate(scoutPrompt, { abortSignal: rootSignal });
              const jsonStr = res.text.match(/\{[\s\S]*\}/)?.[0] || res.text;
              decision = JSON.parse(jsonStr);
            } catch (e) {
              if (!rootSignal.aborted) console.warn('Scout failed to decide', e);
              return;
            }

            // Register Step
            const currentStepId = globalStepCounter++;
            const step: TraceStep = {
              stepId: currentStepId,
              parentId: cursor.parentId,
              cursorId: cursor.id,
              nodeId: cursor.currentNodeId,
              source: cursor.currentNodeId,
              incomingEdge: cursor.lastEdgeType,
              action: decision.action,
              decision,
              reasoning: decision.reasoning,
              timestamp: Date.now()
            };
            log.steps.push(step);
            cursor.traceHistory.push(`[${decision.action}] ${decision.reasoning}`);

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
                const res = await this.agents.judge.generate(judgePrompt, { abortSignal: rootSignal });
                const jsonStr = res.text.match(/\{[\s\S]*\}/)?.[0] || res.text;
                // biome-ignore lint/suspicious/noExplicitAny: artifact shape
                const artifact = JSON.parse(jsonStr) as any;

                if (
                  artifact.isAnswer &&
                  artifact.confidence >= (this.config.confidenceThreshold || 0.7)
                ) {
                  const finalArtifact = {
                    ...artifact,
                    traceId,
                    sources: [cursor.currentNodeId]
                  };
                  foundArtifact = { type: 'FOUND', artifact: finalArtifact, finalStepId: currentStepId };
                  rootController.abort(); // KILL SWITCH
                  return;
                }
              } catch (e) {
                /* ignore judge fail */
              }

              return;
            } else if (decision.action === 'MATCH' && decision.pattern) {
              // Structural Inference
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
                    nextCursors.push({
                      id: randomUUID(),
                      currentNodeId: endNode,
                      path: [...cursor.path, ...path.slice(1)],
                      traceHistory: [...cursor.traceHistory, `[MATCH] Pattern Found`],
                      stepCount: cursor.stepCount + path.length - 1,
                      confidence: cursor.confidence * 0.9,
                      parentId: currentStepId,
                      lastEdgeType: 'MATCH_JUMP'
                    });
                  }
                }
              }
              return;
            } else if (decision.action === 'MOVE') {
              // biome-ignore lint/suspicious/noExplicitAny: flexible move structure
              const moves: any[] = [];
              if (decision.edgeType)
                moves.push({ edge: decision.edgeType, conf: decision.confidence });
              if (decision.alternativeMoves) {
                // biome-ignore lint/suspicious/noExplicitAny: flexible move structure
                for (const alt of decision.alternativeMoves as any[]) {
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
                      traceHistory: [...cursor.traceHistory],
                      stepCount: cursor.stepCount + 1,
                      confidence: cursor.confidence * move.conf,
                      parentId: currentStepId,
                      lastEdgeType: move.edge
                    });
                  }
                }
              }
              return;
            }
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
    } finally {
      if (!foundArtifact && !rootSignal.aborted) {
        rootController.abort(); // Cleanup
      }
    }

    if (foundArtifact) {
      log.outcome = 'FOUND';
      const fa = foundArtifact as { type: 'FOUND'; artifact: LabyrinthArtifact; finalStepId: number };
      log.finalArtifact = fa.artifact;

      if (fa.finalStepId !== undefined) {
        const winningTrace = this.reconstructPath(log.steps, fa.finalStepId);
        // Reinforce with quality score
        await this.tools.reinforcePath(winningTrace, fa.artifact.confidence);
      }
      return fa.artifact;
    }

    if (log.outcome !== 'FOUND') {
      log.outcome = 'EXHAUSTED';
    }

    return null;
  }

  getTrace(traceId: string): TraceLog | undefined {
    return this.traces.get(traceId);
  }

  /**
   * Returns a JSON-serializable version of the trace for debugging or frontend rendering.
   */
  getTraceJSON(traceId: string): string | undefined {
    const trace = this.traces.get(traceId);
    if (!trace) return undefined;
    return JSON.stringify(trace, null, 2);
  }

  private reconstructPath(allSteps: TraceStep[], finalStepId: number): TraceStep[] {
    const path: TraceStep[] = [];
    let currentId: number | undefined = finalStepId;

    const stepMap = new Map<number, TraceStep>();
    for (const s of allSteps) stepMap.set(s.stepId, s);

    while (currentId !== undefined) {
      const step = stepMap.get(currentId);
      if (step) {
        path.unshift(step);
        currentId = step.parentId;
      } else {
        break;
      }
    }
    return path;
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