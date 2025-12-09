# Directory Structure
```
packages/
  agent/
    src/
      mastra/
        agents/
          judge-agent.ts
          router-agent.ts
          scout-agent.ts
        workflows/
          metabolism-workflow.ts
      tools/
        graph-tools.ts
      labyrinth.ts
      types.ts
```

# Files

## File: packages/agent/src/mastra/agents/judge-agent.ts
```typescript
import { Agent } from '@mastra/core/agent';

export const judgeAgent = new Agent({
  name: 'Judge Agent',
  instructions: `
    You are a Judge evaluating data from a Knowledge Graph.
    
    Input provided:
    - Goal: The user's question.
    - Data: Content of the nodes found.
    - Time Context: Relevant timeframe.
    
    Task: Determine if the data answers the goal.
    
    Return ONLY a JSON object:
    { 
      "isAnswer": boolean, 
      "answer": string (The synthesized answer), 
      "confidence": number (0-1) 
    }
  `,
  model: {
    id: 'groq/llama-3.3-70b-versatile',
  }
});
```

## File: packages/agent/src/mastra/agents/router-agent.ts
```typescript
import { Agent } from '@mastra/core/agent';

export const routerAgent = new Agent({
  name: 'Router Agent',
  instructions: `
    You are a Semantic Router for a Knowledge Graph.
    
    Task: Select the single most relevant domain (lens) to conduct the search based on the user's goal.
    
    Input provided:
    - Goal: User query.
    - Available Domains: List of domains and descriptions.
    
    Return ONLY a JSON object:
    {
      "domain": string,
      "confidence": number,
      "reasoning": string
    }
  `,
  model: {
    id: 'groq/llama-3.3-70b-versatile',
  }
});
```

## File: packages/agent/src/mastra/agents/scout-agent.ts
```typescript
import { Agent } from '@mastra/core/agent';
import { sectorScanTool, topologyScanTool, temporalScanTool } from '../tools';

export const scoutAgent = new Agent({
  name: 'Scout Agent',
  instructions: `
    You are a Graph Scout navigating a topology.
    
    Your goal is to decide the next move based on the provided context.
    
    Context provided in user message:
    - Goal: The user's query.
    - Active Domain: The semantic lens (e.g., "medical", "supply-chain").
    - Current Node: ID and Labels.
    - Path History: Nodes visited so far.
    - Satellite View: A summary of outgoing edges (LOD 0).
    - Time Context: Relevant timestamps.

    Decide your next move:
    - **Pheromones:** Edges marked with 🔥 or ♨️ have been successfully traversed before.
    - **Exploration:** To explore, action: "MOVE" with "edgeType".
    - **Pattern Matching:** To find a structure, action: "MATCH" with "pattern".
    - **Goal Check:** If the current node likely contains the answer, action: "CHECK".
    - **Abort:** If stuck, action: "ABORT".
    
    Return ONLY a JSON object matching this schema:
    { 
      "action": "MOVE" | "CHECK" | "ABORT" | "MATCH", 
      "edgeType": string (optional), 
      "confidence": number (0-1), 
      "reasoning": string,
      "pattern": array (optional),
      "alternativeMoves": array (optional)
    }
  `,
  model: {
    id: 'groq/llama-3.3-70b-versatile',
  },
  tools: {
    sectorScanTool,
    topologyScanTool,
    temporalScanTool
  }
});
```

## File: packages/agent/src/mastra/workflows/metabolism-workflow.ts
```typescript
import { createStep, createWorkflow } from '@mastra/core/workflows';
import { z } from 'zod';
import { getGraphInstance } from '../../lib/graph-instance';
import { randomUUID } from 'crypto';

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

    const response = await judge.generate(prompt);
    let summaryText = '';

    try {
      const jsonStr = response.text.match(/\{[\s\S]*\}/)?.[0] || response.text;
      const result = JSON.parse(jsonStr);
      if (result.isAnswer || result.answer) {
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
```

## File: packages/agent/src/tools/graph-tools.ts
```typescript
import type { QuackGraph } from '@quackgraph/graph';
import type { SectorSummary } from '../types';
import type { JsPatternEdge } from '@quackgraph/native';

export class GraphTools {
  constructor(private graph: QuackGraph) { }

  /**
   * LOD 0: Sector Scan / Satellite View
   * Returns a summary of available moves from the current nodes.
   */
  async getSectorSummary(currentNodes: string[], asOf?: number, allowedEdgeTypes?: string[]): Promise<SectorSummary[]> {
    if (currentNodes.length === 0) return [];

    // 1. Get Sector Stats (Count + Heat) in a single Rust call (O(1))
    const results = await this.graph.native.getSectorStats(currentNodes, asOf, allowedEdgeTypes);
    
    // 2. Filter if explicit allowed list provided (double check)
    // Native usually handles this, but if we have complex registry logic (e.g. exclusions), we filter here too
    // Note: optimization - native filtering is faster, but we rely on caller to pass correct allowedEdgeTypes from registry.getValidEdges()
    if (allowedEdgeTypes && allowedEdgeTypes.length > 0) {
        // redundant but safe if native implementation varies
        // no-op if native did its job
    }
    
    // 3. Sort by count (descending)
    return results.sort((a, b) => b.count - a.count);
  }

  /**
   * LOD 1: Topology Scan
   * Returns the IDs of neighbors reachable via a specific edge type.
   */
  async topologyScan(currentNodes: string[], edgeType: string, asOf?: number, minValidFrom?: number): Promise<string[]> {
    if (currentNodes.length === 0) return [];
    return this.graph.native.traverse(currentNodes, edgeType, 'out', asOf, minValidFrom);
  }

  /**
   * LOD 1: Temporal Interval Scan
   * Finds neighbors connected via edges overlapping/contained in the window.
   */
  async intervalScan(currentNodes: string[], windowStart: number, windowEnd: number, constraint: 'overlaps' | 'contains' | 'during' | 'meets' = 'overlaps'): Promise<string[]> {
    return this.graph.native.traverseInterval(currentNodes, undefined, 'out', windowStart, windowEnd, constraint);
  }

  /**
   * LOD 1: Temporal Scan (Wrapper for intervalScan with edge type filtering)
   */
  async temporalScan(currentNodes: string[], windowStart: number, windowEnd: number, edgeType?: string, constraint: 'overlaps' | 'contains' | 'during' | 'meets' = 'overlaps'): Promise<string[]> {
    if (currentNodes.length === 0) return [];
    // We use the native traverseInterval which accepts edgeType
    return this.graph.native.traverseInterval(currentNodes, edgeType, 'out', windowStart, windowEnd, constraint);
  }

  /**
   * LOD 1.5: Pattern Matching (Structural Inference)
   * Finds subgraphs matching a specific shape.
   */
  async findPattern(startNodes: string[], pattern: Partial<JsPatternEdge>[], asOf?: number): Promise<string[][]> {
    if (startNodes.length === 0) return [];
    const nativePattern = pattern.map(p => ({
      srcVar: p.srcVar || 0,
      tgtVar: p.tgtVar || 0,
      edgeType: p.edgeType || '',
      direction: p.direction || 'out'
    }));
    return this.graph.native.matchPattern(startNodes, nativePattern, asOf);
  }

  /**
   * LOD 2: Content Retrieval with "Virtual Spine" Expansion.
   * If nodes are part of a document chain (NEXT/PREV), fetch context.
   */
  // biome-ignore lint/suspicious/noExplicitAny: Generic node content
  async contentRetrieval(nodeIds: string[]): Promise<any[]> {
    if (nodeIds.length === 0) return [];

    // 1. Fetch Primary Content
    const primaryNodes = await this.graph.match([])
      .where({ id: nodeIds })
      .select();

    // 2. Virtual Spine Expansion
    // Check for "NEXT" or "PREV" connections to provide document flow context.
    const spineContextIds = new Set<string>();

    for (const id of nodeIds) {
      // Look ahead
      const next = await this.graph.native.traverse([id], 'NEXT', 'out');
      next.forEach(nid => spineContextIds.add(nid));

      // Look back
      const prev = await this.graph.native.traverse([id], 'PREV', 'out');
      const incomingNext = await this.graph.native.traverse([id], 'NEXT', 'in');
      incomingNext.forEach(nid => spineContextIds.add(nid));

      const explicitPrev = await this.graph.native.traverse([id], 'PREV', 'out');
      explicitPrev.forEach(nid => spineContextIds.add(nid));
    }

    // Remove duplicates (original nodes)
    nodeIds.forEach(id => spineContextIds.delete(id));

    if (spineContextIds.size > 0) {
      const contextNodes = await this.graph.match([])
        .where({ id: Array.from(spineContextIds) })
        .select();

      // Merge and Annotate
      return primaryNodes.map(node => {
        return {
          ...node,
          _isPrimary: true,
          _context: contextNodes
        };
      });
    }

    return primaryNodes;
  }

  /**
   * Pheromones: Reinforce a successful path by increasing edge heat.
   */
  async reinforcePath(trace: { source: string; incomingEdge?: string }[], qualityScore: number = 1.0) {
    // Base increment is 50 for a perfect score. Clamped by native logic (u8 wraparound or saturation).
    // We assume native handles saturation at 255.
    const heatDelta = Math.floor(qualityScore * 50);
    
    for (let i = 1; i < trace.length; i++) {
      const prev = trace[i - 1];
      const curr = trace[i];
      if (!prev || !curr) continue; // Satisfy noUncheckedIndexedAccess
      if (curr.incomingEdge) {
        await this.graph.updateEdgeHeat(prev.source, curr.source, curr.incomingEdge, heatDelta);
      }
    }
  }
}
```

## File: packages/agent/src/types.ts
```typescript
export enum ZoomLevel {
  SECTOR = 0,    // Ghost/Satellite View: Available Moves (Schema)
  TOPOLOGY = 1,  // Drone View: Structural Hops (IDs only)
  CONTENT = 2    // Street View: Full JSON Data
}

// Type alias for Mastra Agent - imports the actual Agent type from @mastra/core
import type { Agent, ToolsInput } from '@mastra/core/agent';
import type { Metric } from '@mastra/core/eval';

// Re-export as an alias for cleaner internal usage
export type MastraAgent = Agent<string, ToolsInput, Record<string, Metric>>;

export interface AgentConfig {
  llmProvider: {
    generate: (prompt: string, signal?: AbortSignal) => Promise<string>;
  };
  maxHops?: number;
  // Max number of concurrent exploration threads
  maxCursors?: number;
  // Minimum confidence to pursue a path (0.0 - 1.0)
  confidenceThreshold?: number;
  // Vector Genesis: Optional embedding provider for start-from-text
  embeddingProvider?: {
    embed: (text: string) => Promise<number[]>;
  };
}

// --- Governance & Router ---

export interface DomainConfig {
  name: string;
  description: string;
  allowedEdges: string[]; // Whitelist of edge types (empty = all unless excluded)
  excludedEdges?: string[]; // Blacklist of edge types (overrides allowed)
  // If true, traversal enforces Monotonic Time (Next Event >= Current Event)
  isCausal?: boolean;
}

export interface RouterPrompt {
  goal: string;
  availableDomains: DomainConfig[];
}

export interface RouterDecision {
  domain: string;
  confidence: number;
  reasoning: string;
}

// --- Scout ---

export interface TimeContext {
  asOf?: Date;
  windowStart?: Date;
  windowEnd?: Date;
}

export interface SectorSummary {
  edgeType: string;
  count: number;
  avgHeat?: number;
}

export interface ScoutDecision {
  action: 'MOVE' | 'CHECK' | 'ABORT' | 'MATCH';
  edgeType?: string; // Required if action is MOVE
  pattern?: { srcVar: number; tgtVar: number; edgeType: string; direction?: string }[]; // Required if action is MATCH
  targetLabels?: string[]; // Optional filter for the move
  confidence: number;
  reasoning: string;
  alternativeMoves?: {
    edgeType: string;
    confidence: number;
    reasoning: string;
  }[];
}

export interface ScoutPrompt {
  goal: string;
  activeDomain: string; // The semantic domain grounding this search
  currentNodeId: string;
  currentNodeLabels: string[];
  sectorSummary: SectorSummary[];
  pathHistory: string[];
  timeContext?: string;
}

export interface JudgePrompt {
  goal: string;
  nodeContent: Record<string, any>[];
  timeContext?: string;
}

// --- Traces ---

export interface LabyrinthArtifact {
  answer: string;
  confidence: number;
  traceId: string;
  sources: string[];
  metadata?: Record<string, any>;
}

// Temporal Logic Types
export interface TemporalWindow {
  anchorNodeId: string;
  windowStart: number; // Unix timestamp
  windowEnd: number;   // Unix timestamp
}

export interface CorrelationResult {
  anchorLabel: string;
  targetLabel: string;
  windowSizeMinutes: number;
  correlationScore: number; // 0.0 - 1.0
  sampleSize: number;
  description: string;
}

export interface EvolutionResult {
  anchorNodeId: string;
  timeline: TimeStepDiff[];
}

export interface TimeStepDiff {
  timestamp: Date;
  // Comparison vs previous step (or baseline)
  addedEdges: SectorSummary[];
  removedEdges: SectorSummary[];
  persistedEdges: SectorSummary[];
  densityChange: number; // percentage
}
```

## File: packages/agent/src/labyrinth.ts
```typescript
import { QuackGraph } from '@quackgraph/graph';
import type {
  AgentConfig,
  LabyrinthArtifact,
  CorrelationResult,
  TimeContext,
  DomainConfig,
  MastraAgent
} from './types';
import { randomUUID } from 'crypto';
import { trace, type Span } from '@opentelemetry/api';

import { setGraphInstance } from './lib/graph-instance';
import { mastra } from './mastra';

// Tools / Helpers
import { Chronos } from './agent/chronos';
import { GraphTools } from './tools/graph-tools';
import { SchemaRegistry } from './governance/schema-registry';

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
              tracingOptions: { traceId, parentSpanId } 
            });
            const jsonStr = res.text.match(/\{[\s\S]*\}/)?.[0] || res.text;
            const decision = JSON.parse(jsonStr);
            const valid = availableDomains.find(
              d => d.name.toLowerCase() === decision.domain.toLowerCase()
            );
            if (valid) activeDomain = decision.domain;

            span.setAttribute('labyrinth.active_domain', activeDomain);
            this.logger.debug(`Router selected domain: ${activeDomain}`, { traceId, domain: activeDomain });
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
        
        let foundArtifact: { artifact: LabyrinthArtifact; cursor: Cursor } | null = null;

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
                      tracingOptions: { traceId, parentSpanId: cursorSpan.spanContext().spanId }
                    });
                    const jsonStr = res.text.match(/\{[\s\S]*\}/)?.[0] || res.text;
                    decision = JSON.parse(jsonStr);
                  } catch (e) {
                    if (!rootSignal.aborted) this.logger.warn('Scout failed to decide', { traceId, cursorId: cursor.id, error: e });
                    return;
                  }

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
                        tracingOptions: { traceId, parentSpanId: cursorSpan.spanContext().spanId }
                      });
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
                        foundArtifact = { artifact: finalArtifact, cursor };
                        span.setAttribute('labyrinth.outcome', 'FOUND');
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
```
