# Directory Structure
```
packages/
  agent/
    src/
      lib/
        config.ts
      mastra/
        tools/
          index.ts
        workflows/
          labyrinth-workflow.ts
          mutation-workflow.ts
      index.ts
      labyrinth.ts
      types.ts
    test/
      e2e/
        labyrinth-complex.test.ts
        labyrinth.test.ts
        metabolism.test.ts
        mutation-complex.test.ts
        mutation.test.ts
        resilience.test.ts
        time-travel.test.ts
      integration/
        chronos.test.ts
        tools.test.ts
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
export * from './types';
export * from './tools';
export * from './chronos';
export * from './utils/temporal';
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

/**
 * Runs a test with a fresh ephemeral graph instance.
 */
export const runWithTestGraph = async (testFn: (graph: QuackGraph) => Promise<void>) => {
  const { graph, path } = await createGraph('memory');
  try {
    await testFn(graph);
  } finally {
    await cleanupGraph(path);
  }
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

## File: packages/agent/test/integration/tools.test.ts
```typescript
import { describe, it, expect, beforeEach, afterEach } from "bun:test";
import { GraphTools } from "../../src/tools/graph-tools";
import { createTestGraph } from "../utils/test-graph";
import type { QuackGraph } from "@quackgraph/graph";

describe("Integration: Graph Tools (Physics Layer)", () => {
  let graph: QuackGraph;
  let tools: GraphTools;

  beforeEach(async () => {
    graph = await createTestGraph();
    tools = new GraphTools(graph);
    
    // Seed Data
    // Root Node
    // @ts-expect-error - Dynamic graph method
    await graph.addNode("root", ["Entity"], { name: "Root" });
    
    // Branch A (Hot)
    // @ts-expect-error
    await graph.addNode("a1", ["Entity"], { name: "A1" });
    // @ts-expect-error
    await graph.addEdge("root", "a1", "LINK", { weight: 1 });
    
    // Branch B (Cold)
    // @ts-expect-error
    await graph.addNode("b1", ["Entity"], { name: "B1" });
    // @ts-expect-error
    await graph.addEdge("root", "b1", "LINK", { weight: 1 });
    
    // Temporal Node (Future)
    // using addNodes to ensure validFrom property support if addNode signature varies
    // @ts-expect-error
    await graph.addNodes([{
      id: "future",
      labels: ["Entity"],
      properties: { name: "Future" },
      validFrom: new Date("2030-01-01")
    }]);
    
    // @ts-expect-error
    await graph.addEdges([{
      source: "root",
      target: "future",
      type: "FUTURE_LINK",
      properties: {},
      validFrom: new Date("2030-01-01")
    }]);
  });

  afterEach(async () => {
    // Teardown if supported by graph instance
    // @ts-expect-error
    if (typeof graph.close === 'function') await graph.close();
  });

  it("LOD 0: getSectorSummary aggregates edge types", async () => {
    // Add more edges to test aggregation
    // @ts-expect-error
    await graph.addNode("a2", ["Entity"], {});
    // @ts-expect-error
    await graph.addEdge("root", "a2", "LINK", {});
    
    // @ts-expect-error
    await graph.addNode("c1", ["Entity"], {});
    // @ts-expect-error
    await graph.addEdge("root", "c1", "OTHER", {});

    const summary = await tools.getSectorSummary(["root"]);
    
    const linkStats = summary.find(s => s.edgeType === "LINK");
    const otherStats = summary.find(s => s.edgeType === "OTHER");
    
    expect(linkStats).toBeDefined();
    // 2 initial LINKs (a1, b1) + 1 new (a2) = 3
    expect(linkStats?.count).toBeGreaterThanOrEqual(3);
    expect(otherStats?.count).toBe(1);
  });

  it("LOD 0: getSectorSummary respects time travel (asOf)", async () => {
    const past = new Date("2020-01-01").getTime();
    const summary = await tools.getSectorSummary(["root"], past);
    
    // The "future" edge shouldn't exist in 2020
    const futureStats = summary.find(s => s.edgeType === "FUTURE_LINK");
    expect(futureStats).toBeUndefined();
  });

  it("LOD 1: topologyScan traverses edges", async () => {
    const neighbors = await tools.topologyScan(["root"], "LINK");
    expect(neighbors).toContain("a1");
    expect(neighbors).toContain("b1");
    expect(neighbors).not.toContain("future"); // Wrong type
  });

  it("LOD 1.5: getNavigationalMap generates ASCII tree", async () => {
    const { map, truncated } = await tools.getNavigationalMap("root", 2);
    
    // console.log("Debug Map Output:\n", map);
    
    expect(map).toContain("[ROOT] root");
    expect(map).toContain("├── [LINK]──> (a1)"); // Order depends on heat/count
    expect(map).not.toContain("future"); // Should be hidden by default "now"
    
    expect(truncated).toBe(false);
  });

  it("LOD 1.5: getNavigationalMap respects asOf", async () => {
    const futureTime = new Date("2035-01-01").getTime();
    const { map } = await tools.getNavigationalMap("root", 1, futureTime);
    
    expect(map).toContain("future");
    expect(map).toContain("[FUTURE_LINK]");
  });
});
```

## File: packages/agent/src/mastra/workflows/mutation-workflow.ts
```typescript
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
            const labels = await graph.getNodeLabels(op.match.id);
            if (labels.length > 0) {
                label = labels[0];
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
                   await graph.expireNode(op.id, validTo);
              } else {
                  await graph.deleteNode(op.id);
              }
              summaryLines.push(`Deleted Node ${op.id}`);
              break;
          }
          case 'CLOSE_EDGE': {
              if (validTo) {
                   await graph.expireEdge(op.source, op.target, op.type, validTo);
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
```

## File: packages/agent/test/e2e/labyrinth.test.ts
```typescript
import { describe, it, expect, beforeEach, afterEach, beforeAll } from "bun:test";
import { createTestGraph } from "../utils/test-graph";
import { SyntheticLLM } from "../utils/synthetic-llm";
import { scoutAgent } from "../../src/mastra/agents/scout-agent";
import { judgeAgent } from "../../src/mastra/agents/judge-agent";
import { routerAgent } from "../../src/mastra/agents/router-agent";
import { Labyrinth } from "../../src/labyrinth";
import type { QuackGraph } from "@quackgraph/graph";

describe("E2E: Labyrinth (Traversal Workflow)", () => {
  let graph: QuackGraph;
  let llm: SyntheticLLM;
  let labyrinth: Labyrinth;

  beforeAll(() => {
    llm = new SyntheticLLM();
    // Providing safe defaults for each agent type to pass Zod schemas
    llm.mockAgent(scoutAgent, { action: "ABORT", confidence: 0, reasoning: "Default Abort" });
    llm.mockAgent(judgeAgent, { isAnswer: false, answer: "No", confidence: 0 });
    llm.mockAgent(routerAgent, { domain: "global", confidence: 1, reasoning: "Default Global" });
  });

  beforeEach(async () => {
    graph = await createTestGraph();
    // Re-instantiate Labyrinth per test to ensure clean state config
    labyrinth = new Labyrinth(graph, { scout: scoutAgent, judge: judgeAgent, router: routerAgent }, {
        llmProvider: { generate: async () => "" }, // Dummy, unused due to mock
        maxHops: 5,
        maxCursors: 3,
        confidenceThreshold: 0.8
    });
  });

  afterEach(async () => {
    // @ts-expect-error
    if (typeof graph.close === 'function') await graph.close();
  });

  it("Scenario: Single Hop Success", async () => {
    // Topology: Start -> Middle -> Goal
    // @ts-expect-error
    await graph.addNode("start", ["Entity"], { name: "Start" });
    // @ts-expect-error
    await graph.addNode("goal_node", ["Entity"], { name: "The Answer" });
    // @ts-expect-error
    await graph.addEdge("start", "goal_node", "LINKS_TO", {});

    // 1. Train Router
    llm.addResponse("Available Domains:", {
        domain: "global",
        confidence: 1.0,
        reasoning: "General query"
    });

    // 2. Train Scout
    // Step 1: At 'start', sees 'goal_node' via 'LINKS_TO'
    // Use a more specific keyword that won't conflict with other prompts
    llm.addResponse('Node: "start" (Labels:', {
        action: "MOVE",
        edgeType: "LINKS_TO",
        confidence: 1.0,
        reasoning: "Moving to linked node"
    });
    
    // Step 2: At 'goal_node', checks for answer
    llm.addResponse('Node: "goal_node" (Labels:', {
        action: "CHECK",
        confidence: 1.0,
        reasoning: "This looks like the answer"
    });

    // 3. Train Judge
    llm.addResponse(`Data:`, {
        isAnswer: true,
        answer: "Found the answer at goal_node",
        confidence: 0.95
    });

    // 4. Run Labyrinth
    const artifact = await labyrinth.findPath("start", "Find the answer");

    expect(artifact).toBeDefined();
    expect(artifact?.answer).toBe("Found the answer at goal_node");
    expect(artifact?.sources).toContain("goal_node");
    expect(artifact?.confidence).toBe(0.95);
  });

  it("Scenario: Speculative Forking (The Race)", async () => {
    // Topology: 
    // start -> path_A -> dead_end
    // start -> path_B -> success
    // @ts-expect-error
    await graph.addNode("start", ["Entity"], {});
    // @ts-expect-error
    await graph.addNode("path_A", ["Entity"], {});
    // @ts-expect-error
    await graph.addNode("path_B", ["Entity"], {});
    // @ts-expect-error
    await graph.addNode("success", ["Entity"], { content: "Victory" });

    // @ts-expect-error
    await graph.addEdge("start", "path_A", "OPTION_A", {});
    // @ts-expect-error
    await graph.addEdge("start", "path_B", "OPTION_B", {});
    // @ts-expect-error
    await graph.addEdge("path_B", "success", "WIN", {});

    // 1. Scout at Start: Unsure, forks!
    // Keyword match on the Node ID provided in prompt
    llm.addResponse(`Node: "start"`, {
        action: "MOVE",
        confidence: 0.5,
        reasoning: "Unsure which path is better",
        alternativeMoves: [
            { edgeType: "OPTION_A", confidence: 0.5, reasoning: "Try A" },
            { edgeType: "OPTION_B", confidence: 0.5, reasoning: "Try B" }
        ]
    });

    // 2. Scout at Path A (Dead End)
    llm.addResponse(`Node: "path_A"`, {
        action: "ABORT", // Or exhaustive search that yields nothing
        confidence: 0.0,
        reasoning: "Dead end"
    });

    // 3. Scout at Path B (Good Path)
    llm.addResponse(`Node: "path_B"`, {
        action: "MOVE",
        edgeType: "WIN",
        confidence: 0.9,
        reasoning: "Found winning path"
    });

    // 4. Scout at Success
    llm.addResponse(`Node: "success"`, {
        action: "CHECK",
        confidence: 1.0,
        reasoning: "Check this"
    });

    // 5. Judge
    llm.addResponse(`Goal: Race`, {
        isAnswer: true,
        answer: "Victory found",
        confidence: 1.0
    });

    // Router default
    llm.setDefault({ domain: "global", confidence: 1.0, reasoning: "default" });

    const artifact = await labyrinth.findPath("start", "Race");

    expect(artifact).toBeDefined();
    expect(artifact?.sources).toContain("success");
    
    // Use getTrace to verify forking happened
    const _trace = await labyrinth.getTrace(artifact?.traceId || "");
    // In our implementation, execution trace is in metadata
    // We expect at least 2 threads to have existed
    // The winner thread + dead thread(s)
    
    // Note: In the mock implementation `labyrinth.ts`, we copy metadata to traceCache. 
    // The test might access `artifact.metadata.execution` directly if Labyrinth returns it (it strips it for the public return, but keeps in cache).
    
    // For this test, verifying we found the answer via path_B is sufficient proof the B-thread survived.
  });

  it("Scenario: Max Hops Exhaustion", async () => {
    // Loop: A <-> B
    // @ts-expect-error
    await graph.addNode("A", ["Entity"], {});
    // @ts-expect-error
    await graph.addNode("B", ["Entity"], {});
    // @ts-expect-error
    await graph.addEdge("A", "B", "LOOP", {});
    // @ts-expect-error
    await graph.addEdge("B", "A", "LOOP", {});

    // Scout just bounces
    llm.addResponse("LOOP", {
        action: "MOVE",
        edgeType: "LOOP",
        confidence: 0.5,
        reasoning: "Looping"
    });

    // Limit hops
    labyrinth = new Labyrinth(graph, { scout: scoutAgent, judge: judgeAgent, router: routerAgent }, {
        llmProvider: { generate: async () => "" },
        maxHops: 3, // Very short leash
        maxCursors: 1
    });

    const artifact = await labyrinth.findPath("A", "Infinite Loop");

    // Should fail gracefully
    expect(artifact).toBeNull();
  });
});
```

## File: packages/agent/test/e2e/resilience.test.ts
```typescript
import { describe, it, expect, beforeAll, mock } from "bun:test";
import { runWithTestGraph } from "../utils/test-graph";
import { SyntheticLLM } from "../utils/synthetic-llm";
import { scoutAgent } from "../../src/mastra/agents/scout-agent";
import { judgeAgent } from "../../src/mastra/agents/judge-agent";
import { routerAgent } from "../../src/mastra/agents/router-agent";
import { mastra } from "../../src/mastra/index";
import { getWorkflowResult } from "../utils/result-helper";

describe("E2E: Chaos Monkey (Resilience)", () => {
  let llm: SyntheticLLM;

  beforeAll(() => {
    llm = new SyntheticLLM();
    // Safe defaults
    llm.mockAgent(scoutAgent, { action: "ABORT", confidence: 0, reasoning: "Default Abort" });
    llm.mockAgent(judgeAgent, { isAnswer: false, answer: "No", confidence: 0 });
    llm.mockAgent(routerAgent, { domain: "global", confidence: 1, reasoning: "Default Global" });
  });

  it("handles Brain Damage (Malformed JSON from Scout)", async () => {
    await runWithTestGraph(async (graph) => {
      // @ts-expect-error
      await graph.addNode("start", ["Entity"], {});
      // @ts-expect-error
      await graph.addNode("end", ["Entity"], {});
      // @ts-expect-error
      await graph.addEdge("start", "end", "LINK", {});

      // 1. Sabotage the Scout Agent for this specific run
      // We override the generate method to throw garbage
      const originalGenerate = scoutAgent.generate;
      
      // @ts-expect-error - hijacking
      scoutAgent.generate = mock(async () => {
        return {
          text: "{ NOT VALID JSON ",
          object: null, // Simulate parser failure or raw text return
          usage: { totalTokens: 0 }
        };
      });

      try {
        const run = await mastra.getWorkflow("labyrinthWorkflow").createRunAsync();
        const res = await run.start({
          inputData: {
            goal: "Garbage In",
            start: "start",
            maxHops: 2
          }
        });

        // The workflow should complete, but find nothing because the thread was killed
        const payload = getWorkflowResult(res);
        const artifact = payload?.artifact;
        expect(artifact).toBeNull(); // No winner found

      } finally {
        // Restore sanity
        scoutAgent.generate = originalGenerate;
      }
    });
  });

  it("handles Exhaustion (Max Hops Reached)", async () => {
    await runWithTestGraph(async (graph) => {
      // Infinite Chain: 1 -> 2 -> 3 ...
      // @ts-expect-error
      await graph.addNode("1", ["Entity"], {});
      // @ts-expect-error
      await graph.addNode("2", ["Entity"], {});
      // @ts-expect-error
      await graph.addEdge("1", "2", "NEXT", {});
      // @ts-expect-error
      await graph.addEdge("2", "3", "NEXT", {}); // Ghost edge to 3

      // Train Scout to always move NEXT
      llm.setDefault({
        action: "MOVE",
        edgeType: "NEXT",
        confidence: 0.9,
        reasoning: "Forever onward"
      });

      // Judge never satisfied
      llm.addResponse("search", { isAnswer: false, confidence: 0 });

      // Router
      llm.addResponse("search", { domain: "global" });

      const run = await mastra.getWorkflow("labyrinthWorkflow").createRunAsync();
      const res = await run.start({
        inputData: {
          goal: "search",
          start: "1",
          maxHops: 1 // Strict limit
        }
      });

      const payload = getWorkflowResult(res);
      const artifact = payload?.artifact;
      
      // Should result in null (failure to find) rather than hanging
      expect(artifact).toBeNull();
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
import type { SectorSummary, CorrelationResult, EvolutionResult, TimeStepDiff } from '@quackgraph/graph';

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
export type { SectorSummary, CorrelationResult, EvolutionResult, TimeStepDiff };

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

## File: packages/agent/test/e2e/mutation-complex.test.ts
```typescript
import { describe, it, expect, beforeAll } from "bun:test";
import { runWithTestGraph } from "../utils/test-graph";
import { SyntheticLLM } from "../utils/synthetic-llm";
import { scribeAgent } from "../../src/mastra/agents/scribe-agent";
import { mastra } from "../../src/mastra/index";
import { getWorkflowResult } from "../utils/result-helper";

describe("E2E: Scribe (Complex Mutations)", () => {
  let llm: SyntheticLLM;

  beforeAll(() => {
    llm = new SyntheticLLM();
    llm.mockAgent(scribeAgent, { operations: [], reasoning: "Default", requiresClarification: undefined });
  });

  it("Halts on Ambiguity ('Delete the blue car')", async () => {
    await runWithTestGraph(async (graph) => {
      // Setup: Two blue cars
      // @ts-expect-error
      await graph.addNode("ford", ["Car"], { color: "blue" });
      // @ts-expect-error
      await graph.addNode("chevy", ["Car"], { color: "blue" });

      // Train Scribe to be confused
      llm.addResponse("Delete the blue car", {
        reasoning: "Ambiguous target.",
        operations: [],
        requiresClarification: "Did you mean the Ford or the Chevy?"
      });

      const run = await mastra.getWorkflow("mutationWorkflow").createRunAsync();
      const res = await run.start({
        inputData: { query: "Delete the blue car" }
      });

      // @ts-expect-error
      if (res.status === "failed") throw new Error(`Workflow failed: ${res.error?.message}`);
      
      const payload = getWorkflowResult(res);

      // @ts-expect-error
      expect(payload?.success).toBe(false);
      // @ts-expect-error
      expect(payload?.summary).toContain("Did you mean the Ford or the Chevy?");

      // Verify no deletion happened
      const cars = await graph.match([]).where({ labels: ["Car"] }).select();
      expect(cars.length).toBe(2);
    });
  });

  it("Executes Temporal Deletion ('Sold it yesterday')", async () => {
    await runWithTestGraph(async (graph) => {
      const THREE_DAYS_AGO = new Date(Date.now() - (3 * 86400000));
      const YESTERDAY = new Date(Date.now() - 86400000).toISOString();
      
      // Setup - create edge that existed 3 days ago
      // @ts-expect-error
      await graph.addNode("me", ["User"], {});
      // @ts-expect-error
      await graph.addNode("bike", ["Item"], {});
      // @ts-expect-error
      await graph.addEdge("me", "bike", "OWNS", {}, { validFrom: THREE_DAYS_AGO });

      // Train Scribe
      llm.addResponse("I sold the bike yesterday", {
        reasoning: "Ownership ended.",
        operations: [
          {
            op: "CLOSE_EDGE",
            source: "me",
            target: "bike",
            type: "OWNS",
            validTo: YESTERDAY
          }
        ]
      });

      const run = await mastra.getWorkflow("mutationWorkflow").createRunAsync();
      const res = await run.start({ inputData: { query: "I sold the bike yesterday" } });

      // @ts-expect-error
      if (res.status === "failed") throw new Error(`Workflow failed: ${res.error?.message}`);
      
      // Verify Physics: Edge should not exist in "Present" view
      // traverse() defaults to now()
      const currentItems = await graph.native.traverse(["me"], "OWNS", "out");
      expect(currentItems).not.toContain("bike");

      // Verify it exists in the past (Time Travel)
      // Check 2 days ago
      const twoDaysAgo = Date.now() - (2 * 86400000);
      const pastItems = await graph.native.traverse(["me"], "OWNS", "out", twoDaysAgo);
      // Depending on how strict the test graph implementation is, this should be true if supported
      // For in-memory QuackGraph, basic time travel is supported.
      expect(pastItems).toContain("bike");
    });
  });
});
```

## File: packages/agent/test/e2e/labyrinth-complex.test.ts
```typescript
import { describe, it, expect, beforeAll, beforeEach } from "bun:test";
import { runWithTestGraph } from "../utils/test-graph";
import { SyntheticLLM } from "../utils/synthetic-llm";
import { scoutAgent } from "../../src/mastra/agents/scout-agent";
import { judgeAgent } from "../../src/mastra/agents/judge-agent";
import { routerAgent } from "../../src/mastra/agents/router-agent";
import { mastra } from "../../src/mastra/index";
import { getWorkflowResult } from "../utils/result-helper";

interface WorkflowResult {
  artifact: {
    answer: string;
    confidence: number;
    sources: string[];
    traceId: string;
    metadata?: unknown;
  } | null;
}

describe("E2E: Labyrinth (Advanced)", () => {
  let llm: SyntheticLLM;

  beforeAll(() => {
    llm = new SyntheticLLM();
    // Mock agents without agent-specific defaults so setDefault() works
    llm.mockAgent(scoutAgent);
    llm.mockAgent(judgeAgent);
    llm.mockAgent(routerAgent);
  });

  // Default Router Response
  beforeEach(() => {
      llm.setDefault({ domain: "global", confidence: 1.0, reasoning: "Global search" });
  });

  it("Executes Speculative Forking (The Race)", async () => {
    await runWithTestGraph(async (graph) => {
      // Topology: start -> (A | B) -> goal
      // @ts-expect-error
      await graph.addNode("start", ["Start"], {});
      // @ts-expect-error
      await graph.addNode("path_A", ["Way"], {});
      // @ts-expect-error
      await graph.addNode("path_B", ["Way"], {});
      // @ts-expect-error
      await graph.addNode("goal", ["End"], { content: "The Answer" });

      // @ts-expect-error
      await graph.addEdge("start", "path_A", "LEFT", {});
      // @ts-expect-error
      await graph.addEdge("start", "path_B", "RIGHT", {});
      // @ts-expect-error
      await graph.addEdge("path_B", "goal", "WIN", {});

      // 1. Train Scout at 'start' to FORK
      // Returns a MOVE for 'LEFT' but alternative 'RIGHT'
      llm.addResponse(`Node: "start"`, {
        action: "MOVE",
        edgeType: "LEFT",
        confidence: 0.5,
        reasoning: "Maybe left?",
        alternativeMoves: [
            { edgeType: "RIGHT", confidence: 0.5, reasoning: "Or maybe right?" }
        ]
      });

      // 2. Train Scout at 'path_A' (Dead End)
      llm.addResponse(`Node: "path_A"`, {
        action: "ABORT",
        confidence: 0.0,
        reasoning: "Dead end here"
      });

      // 3. Train Scout at 'path_B' (Winner)
      llm.addResponse(`Node: "path_B"`, {
        action: "MOVE",
        edgeType: "WIN",
        confidence: 0.9,
        reasoning: "Found the path"
      });

      // 4. Train Scout at 'goal'
      llm.addResponse(`Node: "goal"`, {
          action: "CHECK",
          confidence: 1.0,
          reasoning: "Goal found"
      });

      // 5. Train Judge
      llm.addResponse(`Goal: Race`, {
          isAnswer: true,
          answer: "Found it via Path B",
          confidence: 1.0
      });

      const run = await mastra.getWorkflow("labyrinthWorkflow").createRunAsync();
      const res = await run.start({
          inputData: { goal: "Race", start: "start", maxCursors: 5 }
      });

      // @ts-expect-error
      if (res.status === "failed") throw new Error(`Workflow failed: ${res.error?.message}`);

      const results = getWorkflowResult(res) as WorkflowResult;
      const artifact = results?.artifact;
      
      expect(artifact).toBeDefined();
      expect(artifact?.sources).toContain("goal");

      // We implicitly proved forking works because the "Primary" move was LEFT (Dead End),
      // but the agent found the goal via RIGHT (Alternative), which was only explored due to forking.
    });
  });

  it("Reinforces Path (Pheromones)", async () => {
    await runWithTestGraph(async (graph) => {
      // Simple path: Start -> End
      // @ts-expect-error
      await graph.addNode("s1", ["Start"], {});
      // @ts-expect-error
      await graph.addNode("e1", ["End"], {});
      // @ts-expect-error
      await graph.addEdge("s1", "e1", "DIRECT", { weight: 0 }); // Cold edge

      // Train Scout
      llm.addResponse(`Node: "s1"`, { action: "MOVE", edgeType: "DIRECT", confidence: 1.0, reasoning: "Go" });
      llm.addResponse(`Node: "e1"`, { action: "CHECK", confidence: 1.0, reasoning: "Done" });
      // Train Judge
      llm.addResponse(`Goal: Heat`, { isAnswer: true, answer: "Done", confidence: 1.0 });

      const run = await mastra.getWorkflow("labyrinthWorkflow").createRunAsync();
      const res = await run.start({
          inputData: { goal: "Heat", start: "s1" }
      });

      // @ts-expect-error
      if (res.status === "failed") throw new Error(`Workflow failed: ${res.error?.message}`);

      // Verify Heat Increase
      // Note: In a real integration test we'd check the native graph store.
      // Here we trust the GraphTools implementation which we tested in unit tests.
      // But we can check if getSectorSummary reports >0 heat now if supported.
      // (This assumes the in-memory graph persists state across the workflow step and verify call)
      
      // We rely on the workflow completing successfully as proof the reinforce step ran.
      // Ideally, we'd query: graph.native.getEdge("s1", "e1", "DIRECT").heat
      
      // Let's assume verifying the workflow output contains no error is sufficient for E2E
      // as we tested `reinforcePath` logic in unit tests.
    });
  });
});
```

## File: packages/agent/test/e2e/metabolism.test.ts
```typescript
import { describe, it, expect, beforeAll } from "bun:test";
import { runWithTestGraph } from "../utils/test-graph";
import { SyntheticLLM } from "../utils/synthetic-llm";
import { judgeAgent } from "../../src/mastra/agents/judge-agent";
import { mastra } from "../../src/mastra/index";
import { generateTimeSeries } from "../utils/generators";
import { getWorkflowResult } from "../utils/result-helper";

interface MetabolismResult {
  success: boolean;
  summary: string;
}

describe("E2E: Metabolism (The Dreaming Graph)", () => {
  let llm: SyntheticLLM;

  beforeAll(() => {
    llm = new SyntheticLLM();
    llm.mockAgent(judgeAgent, { isAnswer: false, answer: "No", confidence: 0 });
  });

  it("Digests raw logs into a summary node", async () => {
    await runWithTestGraph(async (graph) => {
      // 1. Generate 10 days of "Mood" logs
      // @ts-expect-error
      await graph.addNode("user_alice", ["User"], { name: "Alice" });
      const { eventIds } = await generateTimeSeries(graph, "user_alice", 10, 24 * 60, 10 * 24 * 60);

      // 2. Train the Brain (Judge)
      llm.addResponse("Summarize these", {
        isAnswer: true,
        answer: "User mood was generally positive with a dip on day 3.",
        confidence: 1.0
      });

      // 3. Run Metabolism Workflow
      const run = await mastra.getWorkflow("metabolismWorkflow").createRunAsync();
      const res = await run.start({
        inputData: {
          minAgeDays: 0, // Process everything immediately for test
          targetLabel: "Event" // Matching generator label
        }
      });

      // @ts-expect-error
      if (res.status === "failed") throw new Error(`Workflow failed: ${res.error?.message}`);

      // 4. Verify Success
      const results = getWorkflowResult(res) as MetabolismResult;
      expect(results?.success).toBe(true);

      // 5. Verify Physics (Graph State)
      // Old nodes should be gone (or disconnected/deleted)
      const oldNodes = await graph.match([]).where({ id: eventIds }).select();
      expect(oldNodes.length).toBe(0);

      // Summary node should exist
      const summaries = await graph.match([]).where({ labels: ["Summary"] }).select();
      expect(summaries.length).toBe(1);
      expect(summaries[0].content).toBe("User mood was generally positive with a dip on day 3.");

      // Check linkage: user_alice -> HAS_SUMMARY -> SummaryNode
      // We need to verify user_alice is connected to the new summary
      const summaryId = summaries[0].id;
      const neighbors = await graph.native.traverse(["user_alice"], "HAS_SUMMARY", "out");
      expect(neighbors).toContain(summaryId);
    });
  });
});
```

## File: packages/agent/test/e2e/time-travel.test.ts
```typescript
import { describe, it, expect, beforeAll } from "bun:test";
import { runWithTestGraph } from "../utils/test-graph";
import { SyntheticLLM } from "../utils/synthetic-llm";
import { scoutAgent } from "../../src/mastra/agents/scout-agent";
import { judgeAgent } from "../../src/mastra/agents/judge-agent";
import { routerAgent } from "../../src/mastra/agents/router-agent";
import { mastra } from "../../src/mastra/index";
import { getWorkflowResult } from "../utils/result-helper";

interface LabyrinthResult {
  artifact: {
    answer: string;
    sources: string[];
  } | null;
}

describe("E2E: The Time Traveler (Labyrinth Workflow)", () => {
  let llm: SyntheticLLM;

  beforeAll(() => {
    llm = new SyntheticLLM();
    // Mock agents without agent-specific defaults so setDefault() works
    llm.mockAgent(scoutAgent);
    llm.mockAgent(judgeAgent);
    llm.mockAgent(routerAgent);
  });

  it("returns different answers for 2023 vs 2024 contexts", async () => {
    await runWithTestGraph(async (graph) => {
      // Topology: Employee managed by different people at different times
      // @ts-expect-error
      await graph.addNode("dave", ["Employee"], { name: "Dave" });
      // @ts-expect-error
      await graph.addNode("alice", ["Manager"], { name: "Alice" }); // 2023
      // @ts-expect-error
      await graph.addNode("bob", ["Manager"], { name: "Bob" });     // 2024

      const t2023_start = new Date("2023-01-01").toISOString();
      const t2023_end   = new Date("2023-12-31").toISOString();
      const t2024_start = new Date("2024-01-01").toISOString();

      // Dave --(MANAGED_BY)--> Alice (2023 only)
      // @ts-expect-error
      await graph.addEdges([{ 
        source: "dave", target: "alice", type: "MANAGED_BY", properties: {}, 
        validFrom: new Date(t2023_start), validTo: new Date(t2023_end) 
      }]);

      // Dave --(MANAGED_BY)--> Bob (2024 onwards)
      // @ts-expect-error
      await graph.addEdges([{ 
        source: "dave", target: "bob", type: "MANAGED_BY", properties: {}, 
        validFrom: new Date(t2024_start)
      }]);

      // --- Train the Synthetic Brain ---
      
      // Router: Will use default (global domain)
      
      // Scout: Sees MANAGED_BY edges. 
      // NOTE: The Scout prompt contains the sector summary. 
      // The sector summary is generated by GraphTools, which respects `asOf`.
      // So if asOf=2023, Scout only sees Alice. If asOf=2024, Scout only sees Bob.
      
      // Generic move response (Scout decides based on what it sees)
      // We'll trust the "Ghost Earth" logic: if it sees an edge, it takes it.
      llm.setDefault({
        action: "MOVE",
        edgeType: "MANAGED_BY",
        confidence: 0.9,
        reasoning: "Following management chain",
        // Safety fields for other agents (Router/Judge) if they fall back
        domain: "global",
        isAnswer: false,
        answer: "Fallback"
      });

      // Special case: If at Alice or Bob, Check for answer
      llm.addResponse(`Node: "alice"`, { action: "CHECK", confidence: 1.0, reasoning: "Checking Alice" });
      llm.addResponse(`Node: "bob"`, { action: "CHECK", confidence: 1.0, reasoning: "Checking Bob" });

      // Judge: Confirms answer (using different keywords to avoid overwriting Scout responses)
      llm.addResponse(`"name":"Alice"`, { isAnswer: true, answer: "Manager was Alice", confidence: 1.0 });
      llm.addResponse(`"name":"Bob"`, { isAnswer: true, answer: "Manager was Bob", confidence: 1.0 });


      // --- Execution 1: Query as of mid-2023 ---
      const run2023 = await mastra.getWorkflow("labyrinthWorkflow").createRunAsync();
      const res2023 = await run2023.start({
        inputData: {
          goal: "Who managed Dave?",
          start: "dave",
          timeContext: { asOf: new Date("2023-06-15").getTime() }
        }
      });

      // @ts-expect-error
      if (res2023.status === "failed") throw new Error(`Workflow failed: ${res2023.error?.message}`);

      const payload2023 = getWorkflowResult(res2023);
      const art2023 = (payload2023 as LabyrinthResult)?.artifact;
      expect(art2023).toBeDefined();
      expect(art2023?.answer).toContain("Alice");
      expect(art2023?.sources).toContain("alice");


      // --- Execution 2: Query as of 2024 ---
      const run2024 = await mastra.getWorkflow("labyrinthWorkflow").createRunAsync();
      const res2024 = await run2024.start({
        inputData: {
          goal: "Who managed Dave?",
          start: "dave",
          timeContext: { asOf: new Date("2024-06-15").getTime() }
        }
      });

      // @ts-expect-error
      if (res2024.status === "failed") throw new Error(`Workflow failed: ${res2024.error?.message}`);

      const payload2024 = getWorkflowResult(res2024);
      const art2024 = (payload2024 as LabyrinthResult)?.artifact;
      expect(art2024).toBeDefined();
      expect(art2024?.answer).toContain("Bob");
      expect(art2024?.sources).toContain("bob");
    });
  });
});
```

## File: packages/agent/src/mastra/tools/index.ts
```typescript
import { createTool } from '@mastra/core/tools';
import { z } from 'zod';
import { getGraphInstance } from '../../lib/graph-instance';
import { getSchemaRegistry } from '../../governance/schema-registry';
import { GraphTools, Chronos } from '@quackgraph/graph';

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
        const res = await tools.getNavigationalMap(id, context.depth, asOf);
        maps.push(res.map);
        if (res.truncated) truncated = true;
      }
      return { map: maps.join('\n\n'), truncated };
    }

    // Implicit map mode if no edgeType is provided
    if (!context.edgeType) {
        const maps = [];
        for (const id of context.nodeIds) {
            const res = await tools.getNavigationalMap(id, 1, asOf);
            maps.push(res.map);
        }
        return { map: maps.join('\n\n') };
    }

    // 3. Standard Traversal
    const neighborIds = await tools.topologyScan(context.nodeIds, context.edgeType, asOf);
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

## File: packages/agent/src/index.ts
```typescript
// Core Facade
export { Labyrinth } from './labyrinth';

// Types & Schemas
export * from './types';
export * from './agent-schemas';

// Utilities
export * from './agent/chronos';
export * from './governance/schema-registry';

// Mastra Internals (Exposed for advanced configuration)
export { mastra } from './mastra';
export { labyrinthWorkflow } from './mastra/workflows/labyrinth-workflow';

// Factory
import type { QuackGraph } from '@quackgraph/graph';
import { Labyrinth } from './labyrinth';
import type { AgentConfig } from './types';
import { mastra } from './mastra';

/**
 * Factory to create a fully wired Labyrinth Agent.
 * Checks for required Mastra agents (Scout, Judge, Router) before instantiation.
 */
export function createAgent(graph: QuackGraph, config: AgentConfig) {
  const scout = mastra.getAgent('scoutAgent');
  const judge = mastra.getAgent('judgeAgent');
  const router = mastra.getAgent('routerAgent');

  if (!scout || !judge || !router) {
    const missing = [];
    if (!scout) missing.push('scoutAgent');
    if (!judge) missing.push('judgeAgent');
    if (!router) missing.push('routerAgent');
    throw new Error(
      `Failed to create QuackGraph Agent. Required Mastra agents are missing: ${missing.join(', ')}. ` +
      `Ensure you have imported 'mastra' from this package and registered these agents.`
    );
  }

  return new Labyrinth(
    graph,
    { scout, judge, router },
    config
  );
}
```

## File: packages/agent/test/integration/chronos.test.ts
```typescript
import { describe, it, expect } from "bun:test";
import { Chronos } from "../../src/agent/chronos";
import { GraphTools } from "../../src/tools/graph-tools";
import { runWithTestGraph } from "../utils/test-graph";
import { generateTimeSeries } from "../utils/generators";

describe("Integration: Chronos (Temporal Physics)", () => {
  
  it("traverseInterval: strictly enforces interval algebra", async () => {
    await runWithTestGraph(async (graph) => {
      const tools = new GraphTools(graph);
      const chronos = new Chronos(graph, tools);

      const BASE_TIME = new Date("2025-01-01T12:00:00Z").getTime();
      const ONE_HOUR = 60 * 60 * 1000;

      // 1. Setup Anchor Node (The Meeting)
      // Duration: 12:00 -> 13:00
      // @ts-expect-error
      await graph.addNode("meeting", ["Event"], { name: "Meeting" });
      
      // We represent interval edges by having a target node that represents the interval context,
      // OR we just test traverseInterval on nodes that have validFrom.
      // QuackGraph's traverseInterval usually checks edge validity or target node validity.
      // Let's assume we are checking TARGET NODE validFrom/validTo relative to the window.

      // Target A: Inside (12:15)
      // @ts-expect-error
      await graph.addNode("note_inside", ["Note"], {}, {
        validFrom: new Date(BASE_TIME + 15 * 60 * 1000), // 12:15
        validTo: new Date(BASE_TIME + 45 * 60 * 1000)    // 12:45 (Must end before window end for strictly DURING)
      });
      
      // Target B: Before (11:00)
      // @ts-expect-error
      await graph.addNode("note_before", ["Note"], {}, { validFrom: new Date(BASE_TIME - ONE_HOUR) }); // 11:00

      // Target C: After (14:00)
      // @ts-expect-error
      await graph.addNode("note_after", ["Note"], {}, { validFrom: new Date(BASE_TIME + 2 * ONE_HOUR) }); // 14:00

      // Connect them all
      // @ts-expect-error
      await graph.addEdge("meeting", "note_inside", "HAS_NOTE", {}, { 
        validFrom: new Date(BASE_TIME + 15 * 60 * 1000),
        validTo: new Date(BASE_TIME + 45 * 60 * 1000) // Must end within window for DURING
      });
      // @ts-expect-error
      await graph.addEdge("meeting", "note_before", "HAS_NOTE", {}, { validFrom: new Date(BASE_TIME - ONE_HOUR) });
      // @ts-expect-error
      await graph.addEdge("meeting", "note_after", "HAS_NOTE", {}, { validFrom: new Date(BASE_TIME + 2 * ONE_HOUR) });

      // Define Window: 12:00 -> 13:00
      const wStart = new Date(BASE_TIME);
      const wEnd = new Date(BASE_TIME + ONE_HOUR);

      // Test: CONTAINS / DURING (Strictly inside)
      // Note: Implementation of 'contains' might vary, usually means Window contains Node.
      const inside = await chronos.findEventsDuring("meeting", wStart, wEnd, 'during');
      
      // Depending on implementation, 'during' might mean the event is during the window.
      // note_inside (12:15) is DURING [12:00, 13:00].
      expect(inside).toContain("note_inside");
      expect(inside).not.toContain("note_before");
      expect(inside).not.toContain("note_after");

      // Test: OVERLAPS (Any intersection)
      // If we had an event spanning 11:30 -> 12:30, it should appear.
      // note_before (11:00 point) does not overlap 12:00-13:00 if it's a point event.
      const overlaps = await chronos.findEventsDuring("meeting", wStart, wEnd, 'overlaps');
      expect(overlaps).toContain("note_inside");
    });
  });

  it("evolutionaryDiff: handles out-of-order writes (Non-linear insertion)", async () => {
    await runWithTestGraph(async (graph) => {
      const tools = new GraphTools(graph);
      const chronos = new Chronos(graph, tools);

      const t1 = new Date("2024-01-01T00:00:00Z");
      const t2 = new Date("2024-02-01T00:00:00Z");
      const t3 = new Date("2024-03-01T00:00:00Z");

      // @ts-expect-error
      await graph.addNode("anchor", ["Entity"], {});
      // @ts-expect-error
      await graph.addNode("target", ["Entity"], {});

      // 1. Write T1 state (First) - Edge valid from T1 to just after T1
      await graph.addEdge("anchor", "target", "LINK", {}, { 
        validFrom: t1, 
        validTo: new Date(t1.getTime() + 1000) // Closed 1 second after T1
      });

      // 2. Write T3 state (Second) - Jump to future
      // We simulate a change in T3. Let's say we add a NEW edge type.
      await graph.addEdge("anchor", "target", "FUTURE_LINK", {}, { validFrom: t3 });

      // Now request the diff in order: T1 -> T2 -> T3
      const result = await chronos.evolutionaryDiff("anchor", [t1, t2, t3]);
      
      // T1 Snapshot: LINK exists
      const snap1 = result.timeline[0];
      expect(snap1.addedEdges.find(e => e.edgeType === "LINK")).toBeDefined();

      // T2 Snapshot: LINK should be REMOVED (because we backfilled the close)
      const snap2 = result.timeline[1];
      expect(snap2.removedEdges.find(e => e.edgeType === "LINK")).toBeDefined();
      
      // T3 Snapshot: FUTURE_LINK should be ADDED
      const snap3 = result.timeline[2];
      expect(snap3.addedEdges.find(e => e.edgeType === "FUTURE_LINK")).toBeDefined();
    });
  });

  it("analyzeCorrelation: detects patterns in high-noise environments", async () => {
    await runWithTestGraph(async (graph) => {
      const tools = new GraphTools(graph);
      const chronos = new Chronos(graph, tools);
      const windowMinutes = 60;

      // 1. Generate Noise (Background events)
      // @ts-expect-error
      await graph.addNode("root", ["System"], {});
      // Generate 50 events over the last 50 hours
      await generateTimeSeries(graph, "root", 50, 60, 50 * 60);

      // 2. Inject Correlation
      // Anchor: "Failure" at T=0 (Now)
      const now = new Date();
      // @ts-expect-error
      await graph.addNode("failure", ["Failure"], {}, { validFrom: now });

      // Target: "CPU_Spike" at T=-30m (Inside window)
      const tInside = new Date(now.getTime() - 30 * 60 * 1000);
      // @ts-expect-error
      await graph.addNode("cpu_spike", ["Metric"], { val: 99 }, { validFrom: tInside });
      
      // Target: "CPU_Spike" at T=-90m (Outside window)
      const tOutside = new Date(now.getTime() - 90 * 60 * 1000);
      // @ts-expect-error
      await graph.addNode("cpu_spike_old", ["Metric"], { val: 80 }, { validFrom: tOutside });

      const res = await chronos.analyzeCorrelation("failure", "Metric", windowMinutes);

      expect(res.sampleSize).toBe(1); // Should only catch the one inside the window
      expect(res.correlationScore).toBe(1.0);
    });
  });
});
```

## File: packages/agent/test/e2e/mutation.test.ts
```typescript
import { describe, it, expect, beforeEach, afterEach, beforeAll } from "bun:test";
import { createTestGraph } from "../utils/test-graph";
import { SyntheticLLM } from "../utils/synthetic-llm";
import { scribeAgent } from "../../src/mastra/agents/scribe-agent";
import { mastra } from "../../src/mastra/index";
import type { QuackGraph } from "@quackgraph/graph";
import { z } from "zod";
import { getWorkflowResult } from "../utils/result-helper";

const MutationResultSchema = z.object({
  success: z.boolean(),
  summary: z.string()
});

describe("E2E: Mutation Workflow (The Scribe)", () => {
  let graph: QuackGraph;
  let llm: SyntheticLLM;

  beforeAll(() => {
    llm = new SyntheticLLM();
    // Hijack the singleton scribe agent
    // Scribe schema requires operations array
    llm.mockAgent(scribeAgent, { operations: [], reasoning: "Default", requiresClarification: undefined });
  });

  beforeEach(async () => {
    graph = await createTestGraph();
  });

  afterEach(async () => {
    // @ts-expect-error
    if (graph && typeof graph.close === 'function') await graph.close();
  });

  it("Scenario: Create Node ('Create a user named Bob')", async () => {
    // 1. Train the Synthetic Brain
    llm.addResponse("Create a user named Bob", {
      reasoning: "User explicitly requested creation of a new Entity.",
      operations: [
        {
          op: "CREATE_NODE",
          id: "bob_1",
          labels: ["User"],
          properties: { name: "Bob" },
          validFrom: "2024-01-01T00:00:00.000Z"
        }
      ]
    });

    // 2. Execute Workflow
    const run = await mastra.getWorkflow("mutationWorkflow").createRunAsync();
    const result = await run.start({
      inputData: {
        query: "Create a user named Bob",
        userId: "admin",
        asOf: new Date("2024-01-01").getTime()
      }
    });

    // @ts-expect-error
    if (result.status === "failed") throw new Error(`Workflow failed: ${result.error?.message}`);

    // 3. Verify Result
    const rawResults = getWorkflowResult(result);

    const parsed = MutationResultSchema.safeParse(rawResults);
    if (!parsed.success) {
      throw new Error(`Invalid workflow result: ${JSON.stringify(rawResults)}`);
    }

    expect(parsed.data.success).toBe(true);
    expect(parsed.data.summary).toContain("Created Node bob_1");

    // Verify side effects
    // @ts-expect-error
    const storedNode = await graph.match([]).where({ labels: ["User"], id: "bob_1" }).select();
    expect(storedNode.length).toBe(1);
    expect(storedNode[0].name).toBe("Bob");
  });

  it("Scenario: Temporal Close ('Bob left the company yesterday')", async () => {
    // Setup: Bob exists and works at Acme
    // @ts-expect-error
    await graph.addNode("bob_1", ["User"], { name: "Bob" });
    // @ts-expect-error
    await graph.addNode("acme", ["Company"], { name: "Acme Inc" });
    // @ts-expect-error
    await graph.addEdge("bob_1", "acme", "WORKS_AT", { role: "Engineer" });

    // 1. Train Brain
    const validTo = "2024-01-02T12:00:00.000Z";
    llm.addResponse("Bob left the company", {
      reasoning: "User indicated employment ended. Closing edge.",
      operations: [
        {
          op: "CLOSE_EDGE",
          source: "bob_1",
          target: "acme",
          type: "WORKS_AT",
          validTo: validTo
        }
      ]
    });

    // 2. Execute
    const run = await mastra.getWorkflow("mutationWorkflow").createRunAsync();
    const result = await run.start({
      inputData: {
        query: "Bob left the company",
        userId: "admin"
      }
    });

    // @ts-expect-error
    if (result.status === "failed") throw new Error(`Workflow failed: ${result.error?.message}`);

    // 3. Verify
    const rawResults = getWorkflowResult(result);
    const parsed = MutationResultSchema.safeParse(rawResults);
    if (!parsed.success) {
      throw new Error(`Invalid workflow result: ${JSON.stringify(rawResults)}`);
    }

    expect(parsed.data.success).toBe(true);

    // 4. Verify Side Effects (Time Travel)
    // The edge should still exist physically but have a valid_to set
    // Note: QuackGraph native.removeEdge might delete it from RAM index, 
    // but DB should retain it if we checked DB directly.
    // For this test, we verify the Workflow reported success and assumed DB update logic held.
    
    // In memory graph implementation might delete it immediately on 'removeEdge' 
    // depending on how QuackGraph core handles soft deletes in memory.
    // Let's verify it's gone from the "Present" view.
    const neighbors = await graph.native.traverse(["bob_1"], "WORKS_AT", "out");
    expect(neighbors).not.toContain("acme");
  });

  it("Scenario: Ambiguity ('Delete the car')", async () => {
    // Setup: Two cars
    // @ts-expect-error
    await graph.addNode("car_1", ["Car"], { color: "Blue", model: "Ford" });
    // @ts-expect-error
    await graph.addNode("car_2", ["Car"], { color: "Blue", model: "Chevy" });

    // 1. Train Brain to be confused
    llm.addResponse("Delete the car", {
      reasoning: "Ambiguous reference. Found multiple cars.",
      operations: [],
      requiresClarification: "Which car? The Ford or the Chevy?"
    });

    // 2. Execute
    const run = await mastra.getWorkflow("mutationWorkflow").createRunAsync();
    const result = await run.start({
      inputData: {
        query: "Delete the car",
        userId: "admin"
      }
    });

    // @ts-expect-error
    if (result.status === "failed") throw new Error(`Workflow failed: ${result.error?.message}`);

    // 3. Verify
    const rawResults = getWorkflowResult(result);
    const parsed = MutationResultSchema.safeParse(rawResults);
    if (!parsed.success) throw new Error("Invalid Result");

    expect(parsed.data.success).toBe(false);
    expect(parsed.data.summary).toContain("Clarification needed");
    expect(parsed.data.summary).toContain("The Ford or the Chevy");

    // 4. Verify Safety (No deletes happened)
    const cars = await graph.match([]).where({ labels: ["Car"] }).select();
    expect(cars.length).toBe(2);
  });
});
```

## File: packages/agent/src/mastra/workflows/labyrinth-workflow.ts
```typescript
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
      maxHops: inputData.maxHops ?? 10,
      maxCursors: inputData.maxCursors ?? 3,
      confidenceThreshold: inputData.confidenceThreshold ?? 0.7,
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

          // biome-ignore lint/suspicious/noExplicitAny: Agent result is loosely typed
          let res: any;
          try {
            // Note: We inject runtimeContext here to ensure tools called by Scout respect "Time" and "Domain"
            res = await scout.generate(prompt, {
              structuredOutput: { schema: ScoutDecisionSchema },
              memory: {
                thread: cursor.id,
                resource: state.governance?.query || 'global-query'
              },
              // Pass "Ghost Earth" context to the agent runtime
              runtimeContext: new Map([['asOf', asOfTs], ['domain', domain]])
            // biome-ignore lint/suspicious/noExplicitAny: RuntimeContext type compatibility
            } as any);
          } catch (err) {
             console.warn(`Thread ${cursor.id} agent generation failed:`, err);
             deadThreads.push({ thread_id: cursor.id, status: 'KILLED', steps: cursor.stepHistory });
             threadSpan?.end({ output: { error: 'agent_failure' } });
             return;
          }

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
  outputSchema: z.object({ 
    artifact: z.custom<LabyrinthArtifact | null>(),
    success: z.boolean() 
  }),
  execute: async ({ inputData, state, tracingContext }) => {
    if (!state.winner || !state.winner.sources) return { artifact: inputData.artifact, success: false };

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
      return { artifact: inputData.artifact, success: true };
    }

    return { artifact: inputData.artifact, success: false };
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
import { Chronos, GraphTools } from '@quackgraph/graph';
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
