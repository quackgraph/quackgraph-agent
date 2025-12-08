import { QuackGraph } from '../packages/quackgraph/packages/quack-graph/src/graph';
import type { 
  AgentConfig, 
  TraceStep, 
  TraceLog, 
  LabyrinthArtifact, 
  ScoutDecision,
  ScoutPrompt,
  JudgePrompt,
  CorrelationResult
} from './types';
import { randomUUID } from 'crypto';

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

  constructor(
    private graph: QuackGraph, 
    private config: AgentConfig
  ) {}

  /**
   * Parallel Speculative Execution:
   * Main Entry Point: Orchestrates multiple Scouts and a Judge to find an answer.
   */
  async findPath(startNodeId: string, goal: string): Promise<LabyrinthArtifact | null> {
    const traceId = randomUUID();
    const startTime = Date.now();
    
    const log: TraceLog = {
      traceId,
      goal,
      startTime,
      steps: [],
      outcome: 'ABORTED' // Default
    };
    
    this.traces.set(traceId, log);

    // Initialize Root Cursor
    let cursors: Cursor[] = [{
      id: randomUUID(),
      currentNodeId: startNodeId,
      path: [startNodeId],
      traceHistory: [],
      stepCount: 0,
      confidence: 1.0
    }];

    const maxHops = this.config.maxHops || 10;
    const maxCursors = this.config.maxCursors || 3;
    let globalStepCounter = 0;

    // Main Loop: While we have active cursors
    while (cursors.length > 0) {
      const nextCursors: Cursor[] = [];
      
      const promises = cursors.map(async (cursor) => {
        // Pruning: Max Depth
        if (cursor.stepCount >= maxHops) {
          return null;
        }

        // 1. Context Awareness (LOD 1)
        const nodeMeta = await this.graph.match([])
          .where({ id: cursor.currentNodeId })
          .select(n => ({ id: n.id, labels: n.labels }));
        
        if (nodeMeta.length === 0) return null; // Node lost/deleted
        const currentNode = nodeMeta[0];

        // 2. Sector Scan (LOD 0)
        const edgeTypes = await this.sectorScan([cursor.currentNodeId]);

        // 3. Ask Scout
        const prompt: ScoutPrompt = {
          goal,
          currentNodeId: currentNode.id,
          currentNodeLabels: currentNode.labels || [],
          availableEdgeTypes: edgeTypes,
          pathHistory: cursor.path
        };

        const decision = await this.askScout(prompt);
        
        // Register Step
        const currentStepId = globalStepCounter++;
        const step: TraceStep = {
          stepId: currentStepId,
          parentId: cursor.parentId,
          cursorId: cursor.id,
          nodeId: cursor.currentNodeId,
          source: cursor.currentNodeId,
          incomingEdge: cursor.lastEdgeType,
          action: decision.action as 'MOVE' | 'CHECK',
          decision,
          reasoning: decision.reasoning,
          timestamp: Date.now()
        };
        log.steps.push(step);
        cursor.traceHistory.push(`[${decision.action}] ${decision.reasoning}`);

        // 4. Handle Decision
        if (decision.action === 'CHECK') {
          // LOD 2: Content Retrieval
          const content = await this.contentRetrieval([cursor.currentNodeId]);
          
          // Ask Judge
          const artifact = await this.askJudge({ goal, nodeContent: content });
          
          if (artifact && artifact.confidence >= (this.config.confidenceThreshold || 0.7)) {
            // Success found by this cursor!
            artifact.traceId = traceId;
            artifact.sources = [cursor.currentNodeId];
            return { type: 'FOUND', artifact, finalStepId: currentStepId };
          } 
          
          // If Judge rejects, this path ends here.
          return null;

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
          const generatedCursors: Cursor[] = [];

          for (const move of moves) {
            if (move.conf < (this.config.confidenceThreshold || 0.2)) continue;

            // LOD 1: Topology Scan
            const nextNodes = await this.topologyScan([cursor.currentNodeId], move.edge);
            
            if (nextNodes.length > 0) {
              // Naive selection: Pick first. Real logic might fork for targets too.
              const target = nextNodes[0]; 
              
              // Update step target if primary move (for legacy/simple visualization)
              if (move.edge === decision.edgeType) {
                 step.target = target;
              }

              generatedCursors.push({
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
          return { type: 'CONTINUE', newCursors: generatedCursors };
        }
        
        return null; // ABORT
      });

      const results = await Promise.all(promises);

      // Check results
      for (const res of results) {
        if (!res) continue;
        if (res.type === 'FOUND') {
          log.outcome = 'FOUND';
          const artifact = res.artifact as LabyrinthArtifact;
          log.finalArtifact = artifact;
          
          // Reconstruct Path & Reinforce
          if (res.finalStepId !== undefined) {
            const winningTrace = this.reconstructPath(log.steps, res.finalStepId);
            await this.reinforcePath(winningTrace);
          }
          
          return artifact;
        }
        if (res.type === 'CONTINUE' && res.newCursors) {
          nextCursors.push(...res.newCursors);
        }
      }

      // Pruning / Management
      // Sort by confidence and take top N
      nextCursors.sort((a, b) => b.confidence - a.confidence);
      cursors = nextCursors.slice(0, maxCursors);
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
    
    // Build map for O(1) lookup
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

  // --- LOD Implementations ---

  /**
   * LOD 0: Sector Scan (Satellite View)
   * The "Ghost Layer". The agent asks: "Where CAN I go from here?"
   */
  async sectorScan(currentNodes: string[]): Promise<string[]> {
    if (currentNodes.length === 0) return [];
    return await this.graph.getAvailableEdgeTypes(currentNodes);
  }

  /**
   * LOD 1: Topology Scan (Drone View)
   * The "Structural Layer". The agent moves blindly through the graph using only IDs and Types.
   */
  async topologyScan(currentNodes: string[], edgeType: string): Promise<string[]> {
    const asOfTs = this.graph.context.asOf ? this.graph.context.asOf.getTime() * 1000 : undefined;
    return this.graph.native.traverse(currentNodes, edgeType, 'out', asOfTs);
  }

  /**
   * LOD 2: Content Retrieval (Street View)
   * The "Data Layer". The agent has identified specific nodes of interest and now reads the text.
   */
  // biome-ignore lint/suspicious/noExplicitAny: Generic node content
  async contentRetrieval(nodeIds: string[]): Promise<any[]> {
    if (nodeIds.length === 0) return [];
    
    return await this.graph.match([])
      .where({ id: nodeIds })
      .select();
  }

  /**
   * Pheromones: Reinforce
   * Reinforces edges along the winning path.
   */
  async reinforcePath(trace: TraceStep[]) {
    // Traverse pairs of steps to find edges
    for (let i = 1; i < trace.length; i++) {
      const prev = trace[i - 1];
      const curr = trace[i];
      if (curr.incomingEdge) {
        await this.graph.updateEdgeHeat(prev.source, curr.source, curr.incomingEdge, 200);
      }
    }
  }

  /**
   * Pheromones: Decay
   * (Optional) Decay edges on failed paths.
   */
  async decayPath(trace: TraceStep[]) {
     // Naive decay logic for now
     for (let i = 1; i < trace.length; i++) {
      const prev = trace[i - 1];
      const curr = trace[i];
      if (curr.incomingEdge) {
        await this.graph.updateEdgeHeat(prev.source, curr.source, curr.incomingEdge, 10);
      }
    }
  }

  // --- Part 3: Temporal Algebra & Metabolism ---

  /**
   * Analyze correlation between an anchor node and a target label within a time window.
   * Uses DuckDB SQL window functions.
   */
  async analyzeCorrelation(
    anchorNodeId: string, 
    targetLabel: string, 
    windowMinutes: number
  ): Promise<CorrelationResult> {
    // 1. Get Anchor Node Timestamp (valid_from)
    const anchorRows = await this.graph.db.query(
      "SELECT valid_from FROM nodes WHERE id = ?", 
      [anchorNodeId]
    );
    
    if (anchorRows.length === 0) {
      throw new Error(`Anchor node ${anchorNodeId} not found`);
    }
    
    // DuckDB returns TIMESTAMP, which might be object or string depending on driver version.
    
    // Query: Find all nodes with targetLabel that overlap the [AnchorTime - Window, AnchorTime] interval
    const sql = `
      WITH Anchor AS (
        SELECT valid_from as t_anchor 
        FROM nodes 
        WHERE id = ?
      ),
      Targets AS (
        SELECT id, valid_from as t_target 
        FROM nodes 
        WHERE list_contains(labels, ?)
      )
      SELECT count(*) as count
      FROM Targets, Anchor
      WHERE t_target >= (t_anchor - INTERVAL ${windowMinutes} MINUTE)
        AND t_target <= t_anchor
    `;
    
    const result = await this.graph.db.query(sql, [anchorNodeId, targetLabel]);
    const count = Number(result[0]?.count || 0);
    
    return {
      anchorLabel: 'Unknown', 
      targetLabel,
      windowSizeMinutes,
      correlationScore: count > 0 ? 1.0 : 0.0, // Simplified boolean correlation
      sampleSize: count,
      description: `Found ${count} instances of ${targetLabel} in the ${windowMinutes}m window.`
    };
  }

  /**
   * Graph Metabolism: Summarize and Prune.
   * Identifies old, dense clusters and compresses them.
   */
  async dream(criteria: { minAgeDays: number, targetLabel: string }) {
    // 1. Identify Candidates (Nodes older than X days)
    const sql = `
      SELECT id, properties 
      FROM nodes 
      WHERE list_contains(labels, ?) 
        AND valid_from < (current_timestamp - INTERVAL ${criteria.minAgeDays} DAY)
        AND valid_to IS NULL -- Active nodes only
      LIMIT 50 -- Batch size
    `;
    
    const candidates = await this.graph.db.query(sql, [criteria.targetLabel]);
    if (candidates.length === 0) return;

    // 2. Synthesize (Judge)
    const prompt: JudgePrompt = {
      goal: `Summarize these ${criteria.targetLabel} logs into a single concise insight.`,
      nodeContent: candidates.map(c => typeof c.properties === 'string' ? JSON.parse(c.properties) : c.properties)
    };
    
    const artifact = await this.askJudge(prompt);
    
    if (!artifact) return; // Judge failed

    // 3. Identification of Anchor (Parent)
    const candidateIds = candidates.map(c => c.id);
    const potentialParents = await this.graph.native.traverse(candidateIds, undefined, 'in', undefined);
    
    if (potentialParents.length === 0) return; // Orphaned
    
    const anchorId = potentialParents[0];

    // 4. Rewire & Prune
    const summaryId = `summary:${randomUUID()}`;
    const summaryProps = {
      content: artifact.answer,
      source_count: candidates.length,
      generated_at: new Date().toISOString()
    };
    
    await this.graph.addNode(summaryId, ['Summary', 'Insight'], summaryProps);
    await this.graph.addEdge(anchorId, summaryId, 'HAS_SUMMARY');
    
    for (const id of candidateIds) {
      await this.graph.deleteNode(id);
    }
  }

  // --- LLM Interaction ---

  private async askScout(promptCtx: ScoutPrompt): Promise<ScoutDecision> {
    const prompt = `
      You are a Graph Scout navigating a blind topology.
      Goal: "${promptCtx.goal}"
      Current Node: ${promptCtx.currentNodeId} (Labels: ${promptCtx.currentNodeLabels.join(', ')})
      Path History: ${promptCtx.pathHistory.join(' -> ')}
      Available Edges: ${JSON.stringify(promptCtx.availableEdgeTypes)}
      
      Decide your next move.
      - If you strongly believe this current node contains the answer, action: "CHECK".
      - If you want to explore, action: "MOVE" and specify the "edgeType".
      - If stuck, "ABORT".
      - If you see multiple promising paths, you can provide "alternativeMoves".
      
      Return ONLY a JSON object:
      { 
        "action": "MOVE", 
        "edgeType": "KNOWS", 
        "confidence": 0.9, 
        "reasoning": "...",
        "alternativeMoves": [
           { "edgeType": "WORKS_WITH", "confidence": 0.5, "reasoning": "..." }
        ]
      }
    `;

    try {
      const raw = await this.config.llmProvider.generate(prompt);
      // Basic JSON extraction attempt
      const jsonStr = raw.match(/\{[\s\S]*\}/)?.[0] || raw;
      return JSON.parse(jsonStr) as ScoutDecision;
    } catch (e) {
      return { action: 'ABORT', confidence: 0, reasoning: 'LLM Parsing Error' };
    }
  }

  private async askJudge(promptCtx: JudgePrompt): Promise<LabyrinthArtifact | null> {
    const prompt = `
      You are a Judge evaluating data.
      Goal: "${promptCtx.goal}"
      Data: ${JSON.stringify(promptCtx.nodeContent)}
      
      Does this data answer the goal?
      Return ONLY a JSON object:
      { "isAnswer": true, "answer": "The user is...", "confidence": 0.95 }
    `;

    try {
      const raw = await this.config.llmProvider.generate(prompt);
      const jsonStr = raw.match(/\{[\s\S]*\}/)?.[0] || raw;
      // biome-ignore lint/suspicious/noExplicitAny: Weak schema for LLM response
      const result = JSON.parse(jsonStr) as any;
      
      if (result.isAnswer) {
        return {
          answer: result.answer,
          confidence: result.confidence,
          traceId: '', // Filled by caller
          sources: [] // Filled by caller
        };
      }
      return null;
    } catch (e) {
      return null;
    }
  }
}