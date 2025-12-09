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

// Mastra Imports
import { mastra } from './mastra';
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
    private config: AgentConfig
  ) {
    // Initialize Graph Singleton for Mastra
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
    // Using Mastra Router Agent
    const availableDomains = this.registry.getAllDomains();
    const routerAgent = mastra.getAgent('routerAgent');
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
            const res = await routerAgent.generate(routerPrompt);
            const jsonStr = res.text.match(/\{[\s\S]*\}/)?.[0] || res.text;
            const decision = JSON.parse(jsonStr);
            const valid = availableDomains.find(d => d.name.toLowerCase() === decision.domain.toLowerCase());
            if (valid) activeDomain = decision.domain;
        } catch (e) {
            console.warn("Routing failed, defaulting to global", e);
        }
    }

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

    // Mastra Agents
    const scoutAgent = mastra.getAgent('scoutAgent');
    const judgeAgent = mastra.getAgent('judgeAgent');

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
              .select('id, labels, date_diff(\'us\', \'1970-01-01\'::TIMESTAMPTZ, valid_from)::DOUBLE as valid_from_micros');

            if (nodeMeta.length === 0) return; // Node lost/deleted
            // biome-ignore lint/suspicious/noExplicitAny: raw sql result
            const currentNode = nodeMeta[0] as any;
            if (!currentNode) return;

            // 2. Sector Scan (LOD 0) - Enhanced Satellite View
            const domainConfig = this.registry.getDomain(activeDomain);
            let allowedEdges: string[] | undefined;

            if (domainConfig && domainConfig.name !== 'global' && domainConfig.allowedEdges.length > 0) {
              allowedEdges = domainConfig.allowedEdges;
            }

            const sectorSummary = await this.tools.getSectorSummary([cursor.currentNodeId], asOfTs, allowedEdges);
            const summaryList = sectorSummary
                .map(s => `- ${s.edgeType}: ${s.count} nodes${(s.avgHeat ?? 0) > 50 ? ' 🔥' : ''}`)
                .join('\n');

            // 3. Ask Scout (Mastra)
            const scoutPrompt = `
              Goal: "${goal}"
              activeDomain: "${activeDomain}"
              currentNodeId: "${currentNode.id}"
              currentNodeLabels: ${JSON.stringify(currentNode.labels || [])}
              pathHistory: ${JSON.stringify(cursor.path)}
              timeContext: "${timeDesc || ''}"
              
              Satellite View (Available Moves):
              ${summaryList}
            `;

            let decision: any;
            try {
                const res = await scoutAgent.generate(scoutPrompt);
                const jsonStr = res.text.match(/\{[\s\S]*\}/)?.[0] || res.text;
                decision = JSON.parse(jsonStr);
            } catch (e) {
                console.warn("Scout failed", e);
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

              // Ask Judge (Mastra)
              const judgePrompt = `
                Goal: "${goal}"
                Data: ${JSON.stringify(content)}
                Time Context: "${timeDesc || ''}"
              `;
              
              try {
                const res = await judgeAgent.generate(judgePrompt);
                const jsonStr = res.text.match(/\{[\s\S]*\}/)?.[0] || res.text;
                const artifact = JSON.parse(jsonStr) as LabyrinthArtifact & { isAnswer: boolean };

                if (artifact.isAnswer && artifact.confidence >= (this.config.confidenceThreshold || 0.7)) {
                    artifact.traceId = traceId;
                    artifact.sources = [cursor.currentNodeId];
                    foundArtifact = { type: 'FOUND', artifact, finalStepId: currentStepId } as any;
                    rootController.abort();
                    return;
                }
              } catch (e) { /* ignore judge fail */ }

              return;

            } else if (decision.action === 'MATCH' && decision.pattern) {
              // Structural Inference
              const matches = await this.tools.findPattern([cursor.currentNodeId], decision.pattern, asOfTs);

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
              const moves = [];
              if (decision.edgeType) moves.push({ edge: decision.edgeType, conf: decision.confidence });
              if (decision.alternativeMoves) {
                for (const alt of decision.alternativeMoves) {
                    moves.push({ edge: alt.edgeType, conf: alt.confidence });
                }
              }

              for (const move of moves) {
                if (move.conf < (this.config.confidenceThreshold || 0.2)) continue;
                if (!this.registry.isEdgeAllowed(activeDomain, move.edge)) continue;
                
                const isCausal = this.registry.isDomainCausal(activeDomain);
                const minValidFrom = isCausal ? (cursor.lastTimestamp || currentNode.valid_from_micros) : undefined;

                const nextNodes = await this.tools.topologyScan([cursor.currentNodeId], move.edge, asOfTs, minValidFrom);

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
                      lastEdgeType: move.edge,
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

        // Pruning
        nextCursors.sort((a, b) => b.confidence - a.confidence);
        cursors = nextCursors.slice(0, maxCursors);
      }
    } finally {
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

  // --- Wrapper for Metabolism Workflow ---
  async dream(criteria: { minAgeDays: number; targetLabel: string }) {
    const workflow = mastra.getWorkflow('metabolismWorkflow');
    if (!workflow) throw new Error('Metabolism workflow not found');
    return await workflow.execute({ triggerData: criteria });
  }
}