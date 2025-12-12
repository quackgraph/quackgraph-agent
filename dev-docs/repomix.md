# Directory Structure
```
packages/
  agent/
    src/
      agent/
        chronos.ts
      lib/
        config.ts
      mastra/
        tools/
          index.ts
      tools/
        graph-tools.ts
      utils/
        temporal.ts
      labyrinth.ts
      types.ts
    test/
      unit/
        chronos.test.ts
        graph-tools.test.ts
        temporal.test.ts
  quackgraph/
    packages/
      quack-graph/
        src/
          index.ts
    test/
      utils/
        helpers.ts
```

# Files

## File: packages/quackgraph/packages/quack-graph/src/index.ts
```typescript
export * from './db';
export * from './graph';
export * from './query';
export * from './schema';
```

## File: packages/quackgraph/test/utils/helpers.ts
```typescript
import { unlink } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { QuackGraph } from '../../packages/quack-graph/src/index';

export const getTempPath = (prefix = 'quack-test') => {
  const uuid = crypto.randomUUID();
  return join(tmpdir(), `${prefix}-${uuid}.duckdb`);
};

export const createGraph = async (mode: 'memory' | 'disk' = 'memory', dbName?: string) => {
  const path = mode === 'memory' ? ':memory:' : getTempPath(dbName);
  const graph = new QuackGraph(path);
  await graph.init();
  return { graph, path };
};

export const cleanupGraph = async (path: string) => {
  if (path === ':memory:') return;
  try {
    // Aggressively clean up main DB file and potential WAL/tmp files
    await unlink(path).catch(() => {});
    await unlink(`${path}.wal`).catch(() => {});
    await unlink(`${path}.tmp`).catch(() => {});
    // Snapshots are sometimes saved as .bin
    await unlink(`${path}.bin`).catch(() => {});
  } catch (_e) {
    // Ignore errors if file doesn't exist
  }
};

/**
 * Wait for a short duration. Useful if we need to ensure timestamps differ slightly
 * (though QuackGraph uses microsecond precision usually, node might be ms).
 */
export const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Seeds a basic graph with a few nodes and edges for testing traversals.
 * A -> B -> C
 *      |
 *      v
 *      D
 */
export const seedBasicGraph = async (g: QuackGraph) => {
  await g.addNode('a', ['Node']);
  await g.addNode('b', ['Node']);
  await g.addNode('c', ['Node']);
  await g.addNode('d', ['Node']);
  await g.addEdge('a', 'b', 'NEXT');
  await g.addEdge('b', 'c', 'NEXT');
  await g.addEdge('b', 'd', 'NEXT');
};
```

## File: packages/agent/src/lib/config.ts
```typescript
import { z } from 'zod';

const envSchema = z.object({
  // Server Config
  MASTRA_PORT: z.coerce.number().default(4111),
  LOG_LEVEL: z.enum(['debug', 'info', 'warn', 'error']).default('info'),

  // Agent Model Configuration (Granular)
  // Format: provider/model-name (e.g., 'groq/llama-3.3-70b-versatile', 'openai/gpt-4')
  AGENT_SCOUT_MODEL: z.string().default('groq/llama-3.3-70b-versatile'),
  AGENT_JUDGE_MODEL: z.string().default('groq/llama-3.3-70b-versatile'),
  AGENT_ROUTER_MODEL: z.string().default('groq/llama-3.3-70b-versatile'),
  AGENT_SCRIBE_MODEL: z.string().default('groq/llama-3.3-70b-versatile'),

  // API Keys (Validated for existence if required by selected models)
  GROQ_API_KEY: z.string().optional(),
  OPENAI_API_KEY: z.string().optional(),
  ANTHROPIC_API_KEY: z.string().optional(),
});

// Validate process.env
// Note: In Bun, process.env is automatically populated from .env files
const parsed = envSchema.parse(process.env);

export const config = {
  server: {
    port: parsed.MASTRA_PORT,
    logLevel: parsed.LOG_LEVEL,
  },
  agents: {
    scout: {
      model: { id: parsed.AGENT_SCOUT_MODEL as `${string}/${string}` },
    },
    judge: {
      model: { id: parsed.AGENT_JUDGE_MODEL as `${string}/${string}` },
    },
    router: {
      model: { id: parsed.AGENT_ROUTER_MODEL as `${string}/${string}` },
    },
    scribe: {
      model: { id: parsed.AGENT_SCRIBE_MODEL as `${string}/${string}` },
    },
  },
};
```

## File: packages/agent/src/utils/temporal.ts
```typescript
/**
 * Simple heuristic parser for relative time strings.
 * Used to ground natural language ("yesterday") into absolute ISO timestamps for the Graph.
 * 
 * In a production system, this would be replaced by a robust library like `chrono-node`.
 */
export function resolveRelativeTime(input: string, referenceDate: Date = new Date()): Date | null {
  const lower = input.toLowerCase().trim();
  const now = referenceDate.getTime();
  const ONE_MINUTE = 60 * 1000;
  const ONE_HOUR = 60 * ONE_MINUTE;
  const ONE_DAY = 24 * ONE_HOUR;

  // 1. Direct keywords
  if (lower === 'now' || lower === 'today') return new Date(now);
  if (lower === 'yesterday') return new Date(now - ONE_DAY);
  if (lower === 'tomorrow') return new Date(now + ONE_DAY);

  // 2. "X [unit] ago"
  const agoMatch = lower.match(/^(\d+)\s+(day|days|hour|hours|minute|minutes|week|weeks)\s+ago$/);
  if (agoMatch) {
    const amount = parseInt(agoMatch[1] || '0', 10);
    const unit = agoMatch[2] || '';
    if (unit.startsWith('day')) return new Date(now - amount * ONE_DAY);
    if (unit.startsWith('hour')) return new Date(now - amount * ONE_HOUR);
    if (unit.startsWith('minute')) return new Date(now - amount * ONE_MINUTE);
    if (unit.startsWith('week')) return new Date(now - amount * 7 * ONE_DAY);
  }

  // 3. "in X [unit]"
  const inMatch = lower.match(/^in\s+(\d+)\s+(day|days|hour|hours|minute|minutes|week|weeks)$/);
  if (inMatch) {
    const amount = parseInt(inMatch[1] || '0', 10);
    const unit = inMatch[2] || '';
    if (unit.startsWith('day')) return new Date(now + amount * ONE_DAY);
    if (unit.startsWith('hour')) return new Date(now + amount * ONE_HOUR);
    if (unit.startsWith('minute')) return new Date(now + amount * ONE_MINUTE);
    if (unit.startsWith('week')) return new Date(now + amount * 7 * ONE_DAY);
  }

  // 4. Fallback: Try native Date parse (e.g. "2023-01-01", "Oct 5 2024")
  const parsed = Date.parse(input);
  if (!Number.isNaN(parsed)) {
    return new Date(parsed);
  }

  return null;
}
```

## File: packages/agent/test/unit/graph-tools.test.ts
```typescript
import { describe, it, expect } from "bun:test";
import { GraphTools } from "../../src/tools/graph-tools";
import { runWithTestGraph } from "../utils/test-graph";

describe("Unit: GraphTools (Physics Layer)", () => {
  it("getNavigationalMap: handles cycles gracefully (Infinite Loop Protection)", async () => {
    await runWithTestGraph(async (graph) => {
      const tools = new GraphTools(graph);

      // Create a cycle: A <-> B
      // @ts-expect-error - dynamic graph
      await graph.addNode("A", ["Entity"], { name: "A" });
      // @ts-expect-error
      await graph.addNode("B", ["Entity"], { name: "B" });

      // @ts-expect-error
      await graph.addEdge("A", "B", "LOOP", {});
      // @ts-expect-error
      await graph.addEdge("B", "A", "LOOP", {});

      // Recursion depth 3
      const { map, truncated } = await tools.getNavigationalMap("A", 3);

      // Should show A -> B -> A -> B
      // The logic clamps at max depth, preventing infinite recursion
      expect(map).toContain("[ROOT] A");
      expect(map).toContain("(B)");
      
      // We expect some repetition due to depth=3, but it must not crash or hang
      const occurrencesOfB = map.match(/\(B\)/g)?.length || 0;
      expect(occurrencesOfB).toBeGreaterThanOrEqual(1);
      
      // Should not have exploded the line count limit immediately
      expect(truncated).toBe(false);
    });
  });

  it("reinforcePath: increments edge heat (Pheromones)", async () => {
    await runWithTestGraph(async (graph) => {
      const tools = new GraphTools(graph);

      // A -> B
      // @ts-expect-error
      await graph.addNode("start", ["Start"], {});
      // @ts-expect-error
      await graph.addNode("end", ["End"], {});
      // @ts-expect-error
      await graph.addEdge("start", "end", "PATH", { weight: 1 });

      // Initial state: heat is 0 (or default)
      const initialSummary = await tools.getSectorSummary(["start"]);
      const initialHeat = initialSummary.find(s => s.edgeType === "PATH")?.avgHeat || 0;

      // Reinforce
      await tools.reinforcePath(["start", "end"], [undefined, "PATH"], 1.0);

      // Check heat increase
      const newSummary = await tools.getSectorSummary(["start"]);
      const newHeat = newSummary.find(s => s.edgeType === "PATH")?.avgHeat || 0;

      // Note: In-memory mock might not implement the full u8 heat decay math, 
      // but we expect the command to have been issued and state updated if supported.
      // If the mock supports it:
      expect(newHeat).toBeGreaterThan(initialHeat);
    });
  });

  it("getSectorSummary: strictly enforces allowedEdgeTypes (Governance)", async () => {
    await runWithTestGraph(async (graph) => {
      const tools = new GraphTools(graph);

      // Root connected to Safe and Unsafe
      // @ts-expect-error
      await graph.addNode("root", ["Root"], {});
      // @ts-expect-error
      await graph.addNode("safe", ["Child"], {});
      // @ts-expect-error
      await graph.addNode("unsafe", ["Child"], {});

      // @ts-expect-error
      await graph.addEdge("root", "safe", "SAFE_LINK", {});
      // @ts-expect-error
      await graph.addEdge("root", "unsafe", "FORBIDDEN_LINK", {});

      // 1. Unrestricted
      const all = await tools.getSectorSummary(["root"]);
      expect(all.length).toBe(2);

      // 2. Restricted
      const restricted = await tools.getSectorSummary(["root"], undefined, ["SAFE_LINK"]);
      expect(restricted.length).toBe(1);
      expect(restricted[0].edgeType).toBe("SAFE_LINK");
      
      const forbidden = restricted.find(r => r.edgeType === "FORBIDDEN_LINK");
      expect(forbidden).toBeUndefined();
    });
  });
});
```

## File: packages/agent/test/unit/temporal.test.ts
```typescript
import { describe, it, expect } from "bun:test";
import { resolveRelativeTime } from "../../src/utils/temporal";

describe("Temporal Logic (resolveRelativeTime)", () => {
  // Fixed reference time: 2025-01-01T12:00:00Z
  // Timestamp: 1735732800000
  const refDate = new Date("2025-01-01T12:00:00Z");
  const ONE_DAY = 24 * 60 * 60 * 1000;
  const ONE_HOUR = 60 * 60 * 1000;

  it("resolves exact keywords", () => {
    expect(resolveRelativeTime("now", refDate)?.getTime()).toBe(refDate.getTime());
    expect(resolveRelativeTime("today", refDate)?.getTime()).toBe(refDate.getTime());
    
    const yesterday = resolveRelativeTime("yesterday", refDate);
    expect(yesterday?.getTime()).toBe(refDate.getTime() - ONE_DAY);

    const tomorrow = resolveRelativeTime("tomorrow", refDate);
    expect(tomorrow?.getTime()).toBe(refDate.getTime() + ONE_DAY);
  });

  it("resolves 'X time ago' patterns", () => {
    const twoDaysAgo = resolveRelativeTime("2 days ago", refDate);
    expect(twoDaysAgo?.getTime()).toBe(refDate.getTime() - (2 * ONE_DAY));

    const fiveHoursAgo = resolveRelativeTime("5 hours ago", refDate);
    expect(fiveHoursAgo?.getTime()).toBe(refDate.getTime() - (5 * ONE_HOUR));
  });

  it("resolves 'in X time' patterns", () => {
    const inThreeWeeks = resolveRelativeTime("in 3 weeks", refDate);
    // 3 weeks = 21 days
    expect(inThreeWeeks?.getTime()).toBe(refDate.getTime() + (21 * ONE_DAY));
  });

  it("resolves absolute dates", () => {
    const iso = "2023-10-05T00:00:00.000Z";
    const result = resolveRelativeTime(iso, refDate);
    expect(result?.toISOString()).toBe(iso);
  });

  it("returns null for garbage input", () => {
    expect(resolveRelativeTime("not a date", refDate)).toBeNull();
  });
});
```

## File: packages/agent/test/unit/chronos.test.ts
```typescript
import { describe, it, expect } from "bun:test";
import { Chronos } from "../../src/agent/chronos";
import { GraphTools } from "../../src/tools/graph-tools";
import { runWithTestGraph } from "../utils/test-graph";

describe("Unit: Chronos (Temporal Physics)", () => {
  it("evolutionaryDiff: detects addition, removal, and persistence", async () => {
    await runWithTestGraph(async (graph) => {
      const tools = new GraphTools(graph);
      const chronos = new Chronos(graph, tools);

      const t1 = new Date("2024-01-01T00:00:00Z");
      const t2 = new Date("2024-02-01T00:00:00Z");
      const t3 = new Date("2024-03-01T00:00:00Z");

      // Setup Anchor
      // @ts-expect-error
      await graph.addNode("anchor", ["Entity"], {});
      // @ts-expect-error
      await graph.addNode("target", ["Entity"], {});

      // T1: Edge exists (valid from T1 to T2)
      await graph.addEdge("anchor", "target", "CONN", {}, { validFrom: t1, validTo: t2 });

      // T3: Edge re-created (new instance, valid from T3 onwards)
      await graph.addEdge("anchor", "target", "CONN", {}, { validFrom: t3 });

      const result = await chronos.evolutionaryDiff("anchor", [t1, t2, t3]);
      
      // Snapshot 1 (T1): Added (Initial)
      expect(result.timeline[0].addedEdges[0].edgeType).toBe("CONN");
      
      // Snapshot 2 (T2): Removed (Compared to T1)
      // At T2, the edge is invalid (closed), so count is 0. 
      // Diff logic: T1(1) vs T2(0) -> Removed
      expect(result.timeline[1].removedEdges.length).toBeGreaterThan(0);
      expect(result.timeline[1].removedEdges[0].edgeType).toBe("CONN");
      
      // Snapshot 3 (T3): Added (Compared to T2)
      // T2(0) vs T3(1) -> Added
      expect(result.timeline[2].addedEdges.length).toBeGreaterThan(0);
      expect(result.timeline[2].addedEdges[0].edgeType).toBe("CONN");
    });
  });

  it("analyzeCorrelation: respects strict time windows", async () => {
    await runWithTestGraph(async (graph) => {
      const tools = new GraphTools(graph);
      const chronos = new Chronos(graph, tools);

      const anchorTime = new Date("2025-01-01T12:00:00Z");
      const windowMins = 60; // 1 hour window: 11:00 -> 12:00

      // Anchor Node
      // @ts-expect-error
      await graph.addNodes([{ id: "A", labels: ["Anchor"], properties: {}, validFrom: anchorTime }]);

      // Event 1: Inside Window (11:30)
      const tInside = new Date(anchorTime.getTime() - (30 * 60 * 1000));
      // @ts-expect-error
      await graph.addNodes([{ id: "E1", labels: ["Event"], properties: {}, validFrom: tInside }]);

      // Event 2: Outside Window (10:30)
      const tOutside = new Date(anchorTime.getTime() - (90 * 60 * 1000));
      // @ts-expect-error
      await graph.addNodes([{ id: "E2", labels: ["Event"], properties: {}, validFrom: tOutside }]);

      // Event 3: Future (12:30) - Causality check (should not be correlated if we look backwards)
      const tFuture = new Date(anchorTime.getTime() + (30 * 60 * 1000));
      // @ts-expect-error
      await graph.addNodes([{ id: "E3", labels: ["Event"], properties: {}, validFrom: tFuture }]);

      const result = await chronos.analyzeCorrelation("A", "Event", windowMins);

      // Should only find E1
      expect(result.sampleSize).toBe(1);
      expect(result.correlationScore).toBe(1.0);
    });
  });
});
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
import type { z } from 'zod';
import type { RouterDecisionSchema, ScoutDecisionSchema } from './agent-schemas';

// Re-export as an alias for cleaner internal usage
export type MastraAgent = Agent<string, ToolsInput, Record<string, Metric>>;

// Labyrinth Runtime Context (Injected via Mastra)
export interface LabyrinthContext {
  // Temporal: The "Now" for the graph traversal (Unix Timestamp in seconds or Date)
  // If undefined, defaults to real-time.
  asOf?: number | Date;

  // Governance: The semantic lens restricting traversal (e.g., "Medical", "Financial")
  domain?: string;

  // Traceability: The distributed trace ID for this execution
  traceId?: string;

  // Threading: The specific cursor/thread ID for parallel speculation
  threadId?: string;
}

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

export type RouterDecision = z.infer<typeof RouterDecisionSchema>;

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

export type ScoutDecision = z.infer<typeof ScoutDecisionSchema>;

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
  // biome-ignore lint/suspicious/noExplicitAny: generic content
  nodeContent: Record<string, any>[];
  timeContext?: string;
}

// --- Traces ---

export interface LabyrinthArtifact {
  answer: string;
  confidence: number;
  traceId: string;
  sources: string[];
  metadata?: LabyrinthMetadata;
}

export interface LabyrinthMetadata {
  duration_ms: number;
  tokens_used: number;
  governance: {
    query: string;
    selected_domain: string;
    rejected_domains: string[];
    reasoning: string;
  };
  execution: ThreadTrace[];
  judgment?: {
    verdict: string;
    confidence: number;
  };
}

export interface ThreadTrace {
  thread_id: string;
  status: 'COMPLETED' | 'KILLED' | 'ACTIVE';
  steps: {
    step: number;
    node_id: string;
    ghost_view?: string; // Snapshot of what the agent saw
    action: string;
    reasoning: string;
  }[];
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

export interface StepEvent {
  step: number;
  node_id: string;
  ghost_view?: string;
  action: string;
  reasoning: string;
}

export interface LabyrinthCursor {
  id: string;
  currentNodeId: string;
  path: string[];
  pathEdges: (string | undefined)[];
  stepHistory: StepEvent[];
  stepCount: number;
  confidence: number;
  lastEdgeType?: string;
  lastTimestamp?: number;
}
```

## File: packages/agent/src/agent/chronos.ts
```typescript
import type { QuackGraph } from '@quackgraph/graph';
import type { CorrelationResult, EvolutionResult, SectorSummary, TimeStepDiff } from '../types';
import type { GraphTools } from '../tools/graph-tools';

export class Chronos {
  constructor(private graph: QuackGraph, private tools: GraphTools) { }

  /**
   * Finds events connected to the anchor node that occurred or overlapped
   * with the specified time window.
   */
  async findEventsDuring(
    anchorNodeId: string,
    windowStart: Date,
    windowEnd: Date,
    constraint: 'overlaps' | 'contains' | 'during' | 'meets' = 'overlaps'
  ): Promise<string[]> {
    // Use native directly for granular control
    return await this.graph.native.traverseInterval(
      [anchorNodeId],
      undefined,
      'out',
      windowStart.getTime(),
      windowEnd.getTime(),
      constraint
    );
  }

  /**
   * Analyze correlation between an anchor node and a target label within a time window.
   * Uses DuckDB SQL window functions.
   */
  async analyzeCorrelation(
    anchorNodeId: string,
    targetLabel: string,
    windowMinutes: number
  ): Promise<CorrelationResult> {
    const count = await this.graph.getTemporalCorrelation(anchorNodeId, targetLabel, windowMinutes);

    return {
      anchorLabel: 'Unknown',
      targetLabel,
      windowSizeMinutes: windowMinutes,
      correlationScore: count > 0 ? 1.0 : 0.0, // Simplified boolean correlation
      sampleSize: count,
      description: `Found ${count} instances of ${targetLabel} in the ${windowMinutes}m window.`
    };
  }

  /**
   * Evolutionary Diffing: Watches how the topology around a node changes over time.
   * Returns a diff of edges (Added, Removed, Persisted) between time snapshots.
   */
  async evolutionaryDiff(anchorNodeId: string, timestamps: Date[]): Promise<EvolutionResult> {
    const sortedTimes = timestamps.sort((a, b) => a.getTime() - b.getTime());
    if (sortedTimes.length === 0) {
      return { anchorNodeId, timeline: [] };
    }

    const timeline: TimeStepDiff[] = [];

    // Initial state (baseline)
    let prevSummary: Map<string, number> = new Map();

    for (const ts of sortedTimes || []) {
      // Use standard JS timestamps (ms) to be consistent with GraphTools and native bindings
      const currentSummaryList = await this.tools.getSectorSummary([anchorNodeId], ts.getTime());
      
      const currentSummary = new Map<string, number>();
      for (const s of currentSummaryList) {
        currentSummary.set(s.edgeType, s.count);
      }

      const addedEdges: SectorSummary[] = [];
      const removedEdges: SectorSummary[] = [];
      const persistedEdges: SectorSummary[] = [];

      // Compare Current vs Prev
      for (const [type, count] of currentSummary) {
        if (prevSummary.has(type)) {
          persistedEdges.push({ edgeType: type, count });
        } else {
          addedEdges.push({ edgeType: type, count });
        }
      }

      for (const [type, count] of prevSummary) {
        if (!currentSummary.has(type)) {
          removedEdges.push({ edgeType: type, count });
        }
      }

      const prevTotal = Array.from(prevSummary.values()).reduce((a, b) => a + b, 0);
      const currTotal = Array.from(currentSummary.values()).reduce((a, b) => a + b, 0);

      const densityChange = prevTotal === 0 ? (currTotal > 0 ? 100 : 0) : ((currTotal - prevTotal) / prevTotal) * 100;

      timeline.push({
        timestamp: ts,
        addedEdges,
        removedEdges,
        persistedEdges,
        densityChange
      });

      prevSummary = currentSummary;
    }

    return { anchorNodeId, timeline };
  }
}
```

## File: packages/agent/src/mastra/tools/index.ts
```typescript
import { createTool } from '@mastra/core/tools';
import { z } from 'zod';
import { getGraphInstance } from '../../lib/graph-instance';
import { GraphTools } from '../../tools/graph-tools';
import { getSchemaRegistry } from '../../governance/schema-registry';
import { Chronos } from '../../agent/chronos';

// Helper to reliably extract context from Mastra's RuntimeContext
// biome-ignore lint/suspicious/noExplicitAny: RuntimeContext access
function extractContext(runtimeContext: any) {
  const asOf = runtimeContext?.get?.('asOf') as number | undefined;
  const domain = runtimeContext?.get?.('domain') as string | undefined;
  return { asOf, domain };
}

export const sectorScanTool = createTool({
  id: 'sector-scan',
  description: 'Get a summary of available moves (edge types) from the current nodes (LOD 0). Automatically filters by active governance domain.',
  inputSchema: z.object({
    nodeIds: z.array(z.string()),
    allowedEdgeTypes: z.array(z.string()).optional(),
  }),
  outputSchema: z.object({
    summary: z.array(z.object({
      edgeType: z.string(),
      count: z.number(),
      avgHeat: z.number().optional(),
    })),
  }),
  execute: async ({ context, runtimeContext }) => {
    const graph = getGraphInstance();
    const tools = new GraphTools(graph);
    const registry = getSchemaRegistry();

    // 1. Extract Environmental Context
    const { asOf, domain } = extractContext(runtimeContext);

    // 2. Apply Governance
    const domainEdges = domain ? registry.getValidEdges(domain) : undefined;
    
    // Merge explicit allowed types (if provided by agent) with domain restrictions
    let effectiveAllowed: string[] | undefined;
    
    if (context.allowedEdgeTypes && domainEdges) {
      // Intersection
      effectiveAllowed = context.allowedEdgeTypes.filter(e => domainEdges.includes(e));
    } else {
      effectiveAllowed = context.allowedEdgeTypes || domainEdges;
    }

    const summary = await tools.getSectorSummary(context.nodeIds, asOf, effectiveAllowed);
    return { summary };
  },
});

export const topologyScanTool = createTool({
  id: 'topology-scan',
  description: 'Get IDs of neighbors reachable via a specific edge type (LOD 1) or visualize structure (Ghost Map).',
  inputSchema: z.object({
    nodeIds: z.array(z.string()),
    edgeType: z.string().optional(),
    depth: z.number().min(1).max(3).optional(),
  }),
  outputSchema: z.object({
    neighborIds: z.array(z.string()).optional(),
    map: z.string().optional(),
    truncated: z.boolean().optional(),
  }),
  execute: async ({ context, runtimeContext }) => {
    const graph = getGraphInstance();
    const tools = new GraphTools(graph);
    const registry = getSchemaRegistry();
    const { asOf, domain } = extractContext(runtimeContext);

    // 1. Governance Check
    if (domain && context.edgeType) {
      if (!registry.isEdgeAllowed(domain, context.edgeType)) {
        return { neighborIds: [] }; // Silently block restricted edges
      }
    }

    // 2. Ghost Map Mode (LOD 1.5)
    if (context.depth && context.depth > 1) {
      const maps = [];
      let truncated = false;
      for (const id of context.nodeIds) {
        // Note: NavigationalMap respects asOf
        const res = await tools.getNavigationalMap(id, context.depth, { asOf });
        maps.push(res.map);
        if (res.truncated) truncated = true;
      }
      return { map: maps.join('\n\n'), truncated };
    }

    // Implicit map mode if no edgeType is provided
    if (!context.edgeType) {
        const maps = [];
        for (const id of context.nodeIds) {
            const res = await tools.getNavigationalMap(id, 1, { asOf });
            maps.push(res.map);
        }
        return { map: maps.join('\n\n') };
    }

    // 3. Standard Traversal
    const neighborIds = await tools.topologyScan(context.nodeIds, context.edgeType, { asOf });
    return { neighborIds };
  },
});

export const temporalScanTool = createTool({
  id: 'temporal-scan',
  description: 'Find neighbors connected via edges overlapping a specific time window.',
  inputSchema: z.object({
    nodeIds: z.array(z.string()),
    windowStart: z.string().describe('ISO Date String'),
    windowEnd: z.string().describe('ISO Date String'),
    edgeType: z.string().optional(),
    constraint: z.enum(['overlaps', 'contains', 'during', 'meets']).optional().default('overlaps'),
  }),
  outputSchema: z.object({
    neighborIds: z.array(z.string()),
  }),
  execute: async ({ context, runtimeContext }) => {
    const graph = getGraphInstance();
    const tools = new GraphTools(graph);
    const registry = getSchemaRegistry();
    const { domain } = extractContext(runtimeContext);

    // Governance Check
    if (domain && context.edgeType) {
       if (!registry.isEdgeAllowed(domain, context.edgeType)) {
         return { neighborIds: [] };
       }
    }
    
    const s = new Date(context.windowStart).getTime();
    const e = new Date(context.windowEnd).getTime();
    const neighborIds = await tools.temporalScan(context.nodeIds, s, e, context.edgeType, context.constraint);
    return { neighborIds };
  },
});

export const contentRetrievalTool = createTool({
  id: 'content-retrieval',
  description: 'Retrieve full content for nodes, including virtual spine expansion (LOD 2).',
  inputSchema: z.object({
    nodeIds: z.array(z.string()),
  }),
  outputSchema: z.object({
    content: z.array(z.record(z.any())),
  }),
  execute: async ({ context }) => {
    const graph = getGraphInstance();
    const tools = new GraphTools(graph);
    const content = await tools.contentRetrieval(context.nodeIds);
    return { content };
  },
});

export const evolutionaryScanTool = createTool({
  id: 'evolutionary-scan',
  description: 'Analyze how the topology around a node changed over specific timepoints (LOD 4 - Time).',
  inputSchema: z.object({
    nodeId: z.string(),
    timestamps: z.array(z.string()).describe('ISO Date Strings'),
  }),
  outputSchema: z.object({
    timeline: z.array(z.object({
      timestamp: z.string(),
      added: z.array(z.string()),
      removed: z.array(z.string()),
      persisted: z.array(z.string()),
      densityChange: z.string()
    })),
    summary: z.string()
  }),
  execute: async ({ context }) => {
    const graph = getGraphInstance();
    const tools = new GraphTools(graph);
    const chronos = new Chronos(graph, tools);

    const dates = context.timestamps.map(t => new Date(t));
    const result = await chronos.evolutionaryDiff(context.nodeId, dates);

    const timeline = result.timeline.map(t => ({
      timestamp: t.timestamp.toISOString(),
      added: t.addedEdges.map(e => `${e.edgeType} (${e.count})`),
      removed: t.removedEdges.map(e => `${e.edgeType} (${e.count})`),
      persisted: t.persistedEdges.map(e => `${e.edgeType} (${e.count})`),
      densityChange: `${t.densityChange.toFixed(1)}%`
    }));

    const summary = `Evolution of ${context.nodeId} across ${dates.length} points. Net density change: ${timeline[timeline.length - 1]?.densityChange || '0%'}.`;

    return { timeline, summary };
  }
});
```

## File: packages/agent/src/tools/graph-tools.ts
```typescript
import type { QuackGraph } from '@quackgraph/graph';
import type { SectorSummary, LabyrinthContext } from '../types';
// import type { JsPatternEdge } from '@quackgraph/native';

export interface JsPatternEdge {
  srcVar: number;
  tgtVar: number;
  edgeType: string;
  direction: string;
}

export class GraphTools {
  constructor(private graph: QuackGraph) { }

  private resolveAsOf(contextOrAsOf?: LabyrinthContext | number): number | undefined {
    let ms: number | undefined;
    if (typeof contextOrAsOf === 'number') {
      ms = contextOrAsOf;
    } else if (contextOrAsOf?.asOf) {
      ms = contextOrAsOf.asOf instanceof Date ? contextOrAsOf.asOf.getTime() : typeof contextOrAsOf.asOf === 'number' ? contextOrAsOf.asOf : undefined;
    }
    
    // Native Rust layer expects milliseconds (f64) and converts to microseconds internally.
    // We default to Date.now() if no time is provided, to ensure "present" physics by default.
    // This prevents future edges from leaking into implicit queries.
    return ms ?? Date.now();
  }

  /**
   * LOD 0: Sector Scan / Satellite View
   * Returns a summary of available moves from the current nodes.
   */
  async getSectorSummary(currentNodes: string[], contextOrAsOf?: LabyrinthContext | number, allowedEdgeTypes?: string[]): Promise<SectorSummary[]> {
    if (currentNodes.length === 0) return [];

    const asOf = this.resolveAsOf(contextOrAsOf);

    // 1. Get Sector Stats (Count + Heat) in a single Rust call (O(1))
    const results = await this.graph.native.getSectorStats(currentNodes, asOf, allowedEdgeTypes);

    // 2. Filter if explicit allowed list provided (double check)
    // Native usually handles this, but if we have complex registry logic (e.g. exclusions), we filter here too
    if (allowedEdgeTypes && allowedEdgeTypes.length > 0) {
      return results.filter((r: SectorSummary) => allowedEdgeTypes.includes(r.edgeType)).sort((a: SectorSummary, b: SectorSummary) => b.count - a.count);
    }

    // 3. Sort by count (descending)
    return results.sort((a: SectorSummary, b: SectorSummary) => b.count - a.count);
  }

  /**
   * LOD 1.5: Ghost Map / Navigational Map
   * Generates an ASCII tree of the topology up to a certain depth.
   * Uses geometric pruning and token budgeting to keep the map readable.
   */
  async getNavigationalMap(rootId: string, depth: number = 1, contextOrAsOf?: LabyrinthContext | number): Promise<{ map: string, truncated: boolean }> {
    const maxDepth = Math.min(depth, 3); // Hard cap at depth 3 for safety
    const treeLines: string[] = [`[ROOT] ${rootId}`];
    let isTruncated = false;
    let totalLines = 0;
    const MAX_LINES = 40; // Prevent context window explosion

    // Helper for recursion
    const buildTree = async (currentId: string, currentDepth: number, prefix: string) => {
      if (currentDepth >= maxDepth) return;
      if (totalLines >= MAX_LINES) {
        if (!isTruncated) {
          treeLines.push(`${prefix}... (truncated)`);
          isTruncated = true;
        }
        return;
      }

      // Geometric pruning: 8 -> 4 -> 2
      const branchLimit = Math.max(2, Math.floor(8 / 2 ** currentDepth));
      let branchesCount = 0;

      // 1. Get stats to find "hot" edges
      const stats = await this.getSectorSummary([currentId], contextOrAsOf);

      // Sort by Heat first, then Count to prioritize "Hot" paths in the view
      stats.sort((a, b) => (b.avgHeat || 0) - (a.avgHeat || 0) || b.count - a.count);

      for (const stat of stats) {
        if (branchesCount >= branchLimit) break;
        if (totalLines >= MAX_LINES) break;

        const edgeType = stat.edgeType;
        const heatVal = stat.avgHeat || 0;
        let heatMarker = '';
        if (heatVal > 80) heatMarker = ' 🔥🔥🔥';
        else if (heatVal > 50) heatMarker = ' 🔥';
        else if (heatVal > 20) heatMarker = ' ♨️';

        // 2. Traverse to get samples (fetch just enough to display)
        const neighbors = await this.topologyScan([currentId], edgeType, contextOrAsOf);

        // Pruning neighbor display based on depth
        const neighborLimit = Math.max(1, Math.floor(branchLimit / (stats.length || 1)) + 1);
        const displayNeighbors = neighbors.slice(0, neighborLimit);

        for (let i = 0; i < displayNeighbors.length; i++) {
          if (branchesCount >= branchLimit) break;
          if (totalLines >= MAX_LINES) { isTruncated = true; break; }

          const neighborId = displayNeighbors[i];
          if (!neighborId) continue;

          // Check if this is the last item to choose the connector symbol
          const isLast = (i === displayNeighbors.length - 1) && (stats.indexOf(stat) === stats.length - 1 || branchesCount === branchLimit - 1);
          const connector = isLast ? '└──' : '├──';

          treeLines.push(`${prefix}${connector} [${edgeType}]──> (${neighborId})${heatMarker}`);
          totalLines++;

          const nextPrefix = prefix + (isLast ? '    ' : '│   ');
          await buildTree(neighborId, currentDepth + 1, nextPrefix);
          branchesCount++;
        }
      }
    };

    await buildTree(rootId, 0, ' ');

    return {
      map: treeLines.join('\n'),
      truncated: isTruncated
    };
  }

  /**
   * LOD 1: Topology Scan
   * Returns the IDs of neighbors reachable via a specific edge type.
   */
  async topologyScan(currentNodes: string[], edgeType?: string, contextOrAsOf?: LabyrinthContext | number, _minValidFrom?: number): Promise<string[]> {
    if (currentNodes.length === 0) return [];
    const asOf = this.resolveAsOf(contextOrAsOf);
    return this.graph.native.traverse(currentNodes, edgeType, 'out', asOf);
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
  async findPattern(startNodes: string[], pattern: Partial<JsPatternEdge>[], contextOrAsOf?: LabyrinthContext | number): Promise<string[][]> {
    if (startNodes.length === 0) return [];
    const nativePattern = pattern.map(p => ({
      srcVar: p.srcVar || 0,
      tgtVar: p.tgtVar || 0,
      edgeType: p.edgeType || '',
      direction: p.direction || 'out'
    }));
    const asOf = this.resolveAsOf(contextOrAsOf);
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
      next.forEach((nid: string) => { spineContextIds.add(nid); });

      // Look back
      const incomingNext = await this.graph.native.traverse([id], 'NEXT', 'in');
      incomingNext.forEach((nid: string) => { spineContextIds.add(nid); });

      const explicitPrev = await this.graph.native.traverse([id], 'PREV', 'out');
      explicitPrev.forEach((nid: string) => { spineContextIds.add(nid); });
    }

    // Remove duplicates (original nodes)
    nodeIds.forEach(id => { spineContextIds.delete(id); });

    if (spineContextIds.size > 0) {
      const contextNodes = await this.graph.match([])
        .where({ id: Array.from(spineContextIds) })
        .select();

      // Merge and Annotate
      // Create a map for fast lookup
      const contextMap = new Map(contextNodes.map(n => [n.id, n]));

      return primaryNodes.map(node => {
        // Find connected context nodes?
        // For simplicity, we just attach all found spine context, 
        // ideally we would link specific context to specific nodes but that requires tracking edges again.
        // We will just return the primary node and let the LLM see the expanded content if requested separately
        // or attach generic context.
        return {
          ...node,
          _isPrimary: true,
          _context: Array.from(contextMap.values()).map(c => ({ id: c.id, ...c.properties }))
        };
      });
    }

    return primaryNodes;
  }

  /**
   * Pheromones: Reinforce a successful path by increasing edge heat.
   */
  async reinforcePath(nodes: string[], edges: (string | undefined)[], qualityScore: number = 1.0) {
    if (nodes.length < 2) return;

    // Base increment is 50 for a perfect score. 
    const heatDelta = Math.floor(qualityScore * 50);

    for (let i = 0; i < nodes.length - 1; i++) {
      const source = nodes[i];
      const target = nodes[i + 1];
      const edge = edges[i + 1]; // edges[0] is undefined (start)

      if (source && target && edge) {
        // Call native update
        try {
          await this.graph.native.updateEdgeHeat(source, target, edge, heatDelta);
        } catch (e) {
          console.warn(`[Pheromones] Failed to update heat for ${source}->${target}:`, e);
        }
      }
    }
  }
}
```

## File: packages/agent/src/labyrinth.ts
```typescript
import type { QuackGraph } from '@quackgraph/graph';
import type {
  AgentConfig,
  LabyrinthArtifact,
  CorrelationResult,
  TimeContext,
  DomainConfig,
  MastraAgent
} from './types';
import { trace, type Span } from '@opentelemetry/api';

// Core Dependencies
import { setGraphInstance } from './lib/graph-instance';
import { mastra } from './mastra';
import { Chronos } from './agent/chronos';
import { GraphTools } from './tools/graph-tools';
import { SchemaRegistry } from './governance/schema-registry';

/**
 * The QuackGraph Agent Facade.
 * 
 * A Native Mastra implementation.
 * This class acts as a thin client that orchestrates the `labyrinth-workflow` 
 * and injects the RuntimeContext (Time Travel & Governance).
 */
export class Labyrinth {
  public chronos: Chronos;
  public tools: GraphTools;
  public registry: SchemaRegistry;
  
  // Simulating persistence layer for traces (In production, use Redis/DB via Mastra Storage)
  private traceCache = new Map<string, LabyrinthArtifact>();
  private logger = mastra.getLogger();
  private tracer = trace.getTracer('quackgraph-agent');

  constructor(
    graph: QuackGraph,
    _agents: {
      scout: MastraAgent;
      judge: MastraAgent;
      router: MastraAgent;
    },
    private config: AgentConfig
  ) {
    // Bridge Pattern: Inject the graph instance into the global scope
    // so Mastra Tools can access it without passing it through every step.
    setGraphInstance(graph);

    // Utilities
    this.tools = new GraphTools(graph);
    this.chronos = new Chronos(graph, this.tools);
    this.registry = new SchemaRegistry();
  }

  /**
   * Registers a semantic domain (LOD 0 governance).
   * Direct proxy to the singleton registry used by tools.
   */
  registerDomain(config: DomainConfig) {
    this.registry.register(config);
  }

  /**
   * Main Entry Point: Finds a path through the Labyrinth.
   * 
   * @param start - Starting Node ID or natural language query
   * @param goal - The question to answer
   * @param timeContext - "Time Travel" parameters (asOf, window)
   */
  async findPath(
    start: string | { query: string },
    goal: string,
    timeContext?: TimeContext
  ): Promise<LabyrinthArtifact | null> {
    return this.tracer.startActiveSpan('labyrinth.findPath', async (span: Span) => {
        try {
            const workflow = mastra.getWorkflow('labyrinthWorkflow');
            if (!workflow) throw new Error("Labyrinth Workflow not registered in Mastra.");

            // 1. Prepare Input Data & Configuration
            const inputData = {
                goal,
                start,
                // Domain is left undefined here; the 'route-domain' step will decide it
                // unless we wanted to force it via config.
                maxHops: this.config.maxHops,
                maxCursors: this.config.maxCursors,
                confidenceThreshold: this.config.confidenceThreshold,
                timeContext: timeContext ? {
                    asOf: timeContext.asOf instanceof Date ? timeContext.asOf.getTime() : timeContext.asOf,
                    windowStart: timeContext.windowStart?.toISOString(),
                    windowEnd: timeContext.windowEnd?.toISOString()
                } : undefined
            };

            // 2. Execute Workflow
            const run = await workflow.createRunAsync();
            
            // The workflow steps are responsible for extracting timeContext from input
            // and passing it to agents via runtimeContext injection in the 'speculative-traversal' step.
            const result = await run.start({ inputData });
            
            // 3. Extract Result
            // @ts-expect-error - Result payload typing
            const payload = result.result || result;
            const artifact = payload?.artifact as LabyrinthArtifact | null;
            if (!artifact && result.status === 'failed') {
                 throw new Error(`Workflow failed: ${result.error?.message || 'Unknown error'}`);
            }

            if (artifact) {
              // Sync traceId with the actual Run ID for retrievability
              // @ts-expect-error - runId access
              const runId = run.runId || run.id;
              artifact.traceId = runId;

              span.setAttribute('labyrinth.confidence', artifact.confidence);
              span.setAttribute('labyrinth.traceId', artifact.traceId);

              // Cache the full artifact (with heavy execution trace)
              this.traceCache.set(runId, JSON.parse(JSON.stringify(artifact)));

              // Return "Executive Briefing" version (strip execution logs)
              if (artifact.metadata) {
                 artifact.metadata.execution = []; 
              }
            }

            return artifact;

        } catch (e) {
            this.logger.error("Labyrinth traversal failed", { error: e });
            span.recordException(e as Error);
            throw e;
        } finally {
            span.end();
        }
    });
  }

  /**
   * Retrieve the full reasoning trace for a specific run.
   * Useful for auditing or "Show your work" features.
   */
  async getTrace(traceId: string): Promise<LabyrinthArtifact | undefined> {
    // 1. Try Memory Cache
    if (this.traceCache.has(traceId)) {
        return this.traceCache.get(traceId);
    }

    // 2. Future: Try Mastra Storage (DB)
    // const run = await mastra.getRun(traceId);
    // return run?.result?.artifact;

    return undefined;
  }

  /**
   * Direct access to Chronos for temporal analytics.
   * Useful for "Life Coach" dashboards that need raw stats without full agent traversal.
   */
  async analyzeCorrelation(
    anchorNodeId: string,
    targetLabel: string,
    windowMinutes: number
  ): Promise<CorrelationResult> {
    return this.chronos.analyzeCorrelation(anchorNodeId, targetLabel, windowMinutes);
  }

  /**
   * Execute a Natural Language Mutation.
   * Uses the Scribe Agent to parse intent and apply graph operations.
   */
  async mutate(query: string, timeContext?: TimeContext): Promise<{ success: boolean; summary: string }> {
    return this.tracer.startActiveSpan('labyrinth.mutate', async (span: Span) => {
      try {
        const workflow = mastra.getWorkflow('mutationWorkflow');
        if (!workflow) throw new Error("Mutation Workflow not registered.");

        const inputData = {
          query,
          asOf: timeContext?.asOf instanceof Date ? timeContext.asOf.getTime() : timeContext?.asOf,
          userId: 'Me' // Default context
        };

        const run = await workflow.createRunAsync();
        const result = await run.start({ inputData });
        if (result.status === 'failed') {
          throw new Error(`Mutation failed: ${result.error.message}`);
        }
        if (result.status !== 'success') {
          throw new Error(`Mutation possibly failed with status: ${result.status}`);
        }
        const payload = result.result || result;
        return payload as { success: boolean; summary: string };
          } catch (e) {
            this.logger.error("Mutation failed", { error: e });
            span.recordException(e as Error);
            return { success: false, summary: `Mutation failed: ${(e as Error).message}` };
          } finally {
            span.end();
          }
    });
  }
}
```
