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
  DomainConfig
} from './types';
import { randomUUID } from 'crypto';

import { Scout } from './agent/scout';
import { Judge } from './agent/judge';
import { Chronos } from './agent/chronos';
import { GraphTools } from './tools/graph-tools';
import { Metabolism } from './agent/metabolism';
import { Router } from './agent/router';
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
}

export class Labyrinth {
  private traces = new Map<string, TraceLog>();

  public scout: Scout;
  public judge: Judge;
  public chronos: Chronos;
  public tools: GraphTools;
  public metabolism: Metabolism;
  public router: Router;
  public registry: SchemaRegistry;

  constructor(
    private graph: QuackGraph,
    private config: AgentConfig
  ) {
    this.scout = new Scout(config);
    this.judge = new Judge(config);
    this.tools = new GraphTools(graph);
    this.chronos = new Chronos(graph, this.tools);
    this.metabolism = new Metabolism(graph, this.judge);

    // Governance
    this.router = new Router(config);
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
    // Before we start walking, decide WHICH domain we are walking in.
    const availableDomains = this.registry.getAllDomains();
    const route = await this.router.route(goal, availableDomains, rootSignal);
    const activeDomain = route.domain;

    // Resolve effective timestamp (passed context > graph context > current)
    const effectiveAsOf = timeContext?.asOf || this.graph.context.asOf;
    const asOfTs = effectiveAsOf ? effectiveAsOf.getTime() * 1000 : undefined;

    const timeDesc = effectiveAsOf
      ? `As of: ${effectiveAsOf.toISOString()}`
      : (timeContext?.windowStart ? `Window: ${timeContext.windowStart.toISOString()} - ${timeContext.windowEnd?.toISOString()}` : undefined);

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
        throw new Error("Vector Genesis requires an embeddingProvider in AgentConfig.");
      }
      const vector = await this.config.embeddingProvider.embed(start.query);
      // Get top 3 relevant start nodes
      const matches = await this.graph.match([]).nearText(vector).limit(3).select();
      startNodes = matches.map(m => m.id);
    } else {
      startNodes = [start as string];
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
    let foundArtifact: LabyrinthArtifact | null = null;

    // Speculative Execution: We use a local controller for the batch to kill peers if winner found
    try {
      // Main Loop: While we have active cursors and haven't found the answer
      while (cursors.length > 0 && !foundArtifact && !rootSignal.aborted) {
        const nextCursors: Cursor[] = [];
        const processingPromises: Promise<void>[] = [];

        // We wrap the iteration to allow for "Race" logic (checking foundArtifact)

        for (const cursor of cursors) {
          if (foundArtifact) break; // Early exit check

          const task = async () => {
            if (foundArtifact) return; // Double check inside async

            // Pruning: Max Depth
            if (cursor.stepCount >= maxHops) return;

            // 1. Context Awareness (LOD 1)
            const nodeMeta = await this.graph.match([])
              .where({ id: cursor.currentNodeId })
              .select(n => ({ id: n.id, labels: n.labels }));

            if (nodeMeta.length === 0) return; // Node lost/deleted
            const currentNode = nodeMeta[0];
            if (!currentNode) return; // Satisfy noUncheckedIndexedAccess

            // 2. Sector Scan (LOD 0) - Enhanced Satellite View

            // CONTEXT FIREWALL: Push filtering down to Rust
            const domainConfig = this.registry.getDomain(activeDomain);
            let allowedEdges: string[] | undefined;

            if (domainConfig && domainConfig.name !== 'global' && domainConfig.allowedEdges.length > 0) {
              allowedEdges = domainConfig.allowedEdges;
            }

            const sectorSummary = await this.tools.getSectorSummary([cursor.currentNodeId], asOfTs, allowedEdges);

            // 3. Ask Scout
            const prompt: ScoutPrompt = {
              goal,
              activeDomain,
              currentNodeId: currentNode.id,
              currentNodeLabels: currentNode.labels || [],
              sectorSummary,
              pathHistory: cursor.path,
              timeContext: timeDesc
            };

            const decision = await this.scout.decide(prompt, rootSignal);

            // Register Step
            const currentStepId = globalStepCounter++;
            const step: TraceStep = {
              stepId: currentStepId,
              parentId: cursor.parentId,
              cursorId: cursor.id,
              nodeId: cursor.currentNodeId,
              source: cursor.currentNodeId,
              incomingEdge: cursor.lastEdgeType,
              action: decision.action as 'MOVE' | 'CHECK' | 'MATCH',
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
              const judgePrompt: JudgePrompt = {
                goal,
                nodeContent: content,
                timeContext: timeDesc
              };
              const artifact = await this.judge.evaluate(judgePrompt, rootSignal);

              if (artifact && artifact.confidence >= (this.config.confidenceThreshold || 0.7)) {
                // Success found by this cursor!
                artifact.traceId = traceId;
                artifact.sources = [cursor.currentNodeId];

                // Set global found flag to stop other cursors
                foundArtifact = { type: 'FOUND', artifact, finalStepId: currentStepId } as any;
                rootController.abort(); // Kill other pending scouts in this batch
                return;
              }

              // If Judge rejects, this path ends here.
              return;

            } else if (decision.action === 'MATCH' && decision.pattern) {
              // Structural Inference
              const matches = await this.tools.findPattern([cursor.currentNodeId], decision.pattern, asOfTs);

              if (matches.length > 0) {
                // Pattern found! Jump to the end of the matched paths.
                // matches is Vec<Vec<string>> (list of paths)
                // We take up to 3 matches to spawn cursors
                const foundPaths = matches.slice(0, 3);

                for (const path of foundPaths) {
                  const endNode = path[path.length - 1]; // Jump to end of pattern
                  if (endNode) {
                    nextCursors.push({
                      id: randomUUID(),
                      currentNodeId: endNode,
                      path: [...cursor.path, ...path.slice(1)], // Append path (skipping start which is current)
                      traceHistory: [...cursor.traceHistory, `[MATCH] Pattern Found: ${JSON.stringify(decision.pattern)}`],
                      stepCount: cursor.stepCount + path.length - 1, // Advance step count
                      confidence: cursor.confidence * 0.9, // Slight penalty for jump
                      parentId: currentStepId,
                      lastEdgeType: 'MATCH_JUMP'
                    });
                  }
                }
              }
              return;

            } else if (decision.action === 'MOVE') {
              const moves = [];

              // Primary Move
              if (decision.edgeType) {
                moves.push({ edge: decision.edgeType, conf: decision.confidence });
              }

              // Alternative Moves (Speculative Execution)
              if (decision.alternativeMoves) {
                for (const alt of decision.alternativeMoves) {
                  moves.push({ edge: alt.edgeType, conf: alt.confidence });
                }
              }

              // Process Moves
              for (const move of moves) {
                if (move.conf < (this.config.confidenceThreshold || 0.2)) continue;

                // Double check governance (Scout hallucination check)
                if (!this.registry.isEdgeAllowed(activeDomain, move.edge)) {
                  // If Scout tries to move on a forbidden edge (despite firewall in prompt), block it.
                  continue;
                }

                // LOD 1: Topology Scan
                const nextNodes = await this.tools.topologyScan([cursor.currentNodeId], move.edge, asOfTs);

                if (nextNodes.length > 0) {
                  // If multiple targets, we branch, taking up to 3 to prevent explosion
                  const targets = nextNodes.slice(0, 3);

                  for (const target of targets) {
                    if (move.edge === decision.edgeType && target === nextNodes[0]) {
                      step.target = target;
                    }

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

        await Promise.all(processingPromises);

        // Check results
        // biome-ignore lint/suspicious/noExplicitAny: hack for race result
        const res = foundArtifact as any;
        if (res && res.type === 'FOUND') {
          log.outcome = 'FOUND';
          const artifact = res.artifact as LabyrinthArtifact;
          log.finalArtifact = artifact;

          if (res.finalStepId !== undefined) {
            const winningTrace = this.reconstructPath(log.steps, res.finalStepId);
            await this.tools.reinforcePath(winningTrace);
          }

          return artifact;
        }

        // Pruning / Management
        // We sort by confidence and take top N
        nextCursors.sort((a, b) => b.confidence - a.confidence);
        cursors = nextCursors.slice(0, maxCursors);
      }
    } finally {
      // Ensure we clean up if something throws
      if (!foundArtifact && !rootSignal.aborted) {
        rootController.abort();
      }
    }

    if (log.outcome !== 'FOUND') {
      log.outcome = 'EXHAUSTED';
    }

    return null;
  }

  getTrace(traceId: string): TraceLog | undefined {
    return this.traces.get(traceId);
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

  async analyzeCorrelation(anchorNodeId: string, targetLabel: string, windowMinutes: number): Promise<CorrelationResult> {
    return this.chronos.analyzeCorrelation(anchorNodeId, targetLabel, windowMinutes);
  }
}