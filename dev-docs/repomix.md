# Directory Structure
```
dev-docs/
  MONOREPO.md
packages/
  agent/
    src/
      agent/
        chronos.ts
      governance/
        schema-registry.ts
      lib/
        graph-instance.ts
      mastra/
        agents/
          judge-agent.ts
          router-agent.ts
          scout-agent.ts
        tools/
          index.ts
        workflows/
          labyrinth-workflow.ts
          metabolism-workflow.ts
        index.ts
      tools/
        graph-tools.ts
      agent-schemas.ts
      index.ts
      labyrinth.ts
      types.ts
    biome.json
    package.json
    tsconfig.json
    tsup.config.ts
scripts/
  git-pull.ts
  git-sync.ts
.gitignore
LICENSE
package.json
README.md
relay.config.json
repomix.config.json
tsconfig.json
```

# Files

## File: dev-docs/MONOREPO.md
````markdown
# Monorepo Structure - Federated Repositories

This monorepo uses a **federated repository structure** where:

- **`quackgraph-agent`** (this repo) owns the high-level Agent logic
- **`packages/quackgraph`** is a nested Git repository containing the Core engine

## Repository Structure

```
quackgraph-agent/               # GitHub: quackgraph/quackgraph-agent
├── packages/
│   ├── agent/                  # Agent logic (owned by parent repo)
│   │   └── src/
│   └── quackgraph/             # GitHub: quackgraph/quackgraph
│       └── packages/
│           ├── native/         # Rust N-API bindings
│           └── quack-graph/    # Core graph TypeScript library
├── scripts/
│   └── git-sync.ts             # Federated push automation
├── package.json                # Root workspace config
└── tsconfig.json
```

## Workspace Configuration

The root `package.json` defines a Bun workspace that spans both repos:

```json
{
  "workspaces": [
    "packages/agent",
    "packages/quackgraph/packages/*"
  ]
}
```

This allows seamless dependency resolution:
- `@quackgraph/agent` → `packages/agent`
- `@quackgraph/graph` → `packages/quackgraph/packages/quack-graph`
- `@quackgraph/native` → `packages/quackgraph/packages/native`

## Git Workflow

### Synchronized Push

To push changes to both repositories atomically:

```bash
bun run push:all "your commit message"
```

This runs `scripts/git-sync.ts` which:
1. Checks if `packages/quackgraph` has uncommitted changes
2. Commits and pushes the inner repo first
3. Commits and pushes the parent repo

### Manual Push (Fine-Grained Control)

```bash
# Push inner repo only
cd packages/quackgraph
git add -A && git commit -m "message" && git push

# Push outer repo only (after inner)
cd ../..
git add -A && git commit -m "message" && git push
```

## Development Commands

| Command | Description |
|---------|-------------|
| `bun install` | Install all dependencies across workspaces |
| `bun run build` | Build core + agent packages |
| `bun run build:core` | Build only the quackgraph core |
| `bun run build:agent` | Build only the agent package |
| `bun run test` | Run tests |
| `bun run push:all` | Synchronized git push to both repos |
| `bun run clean` | Clean all build artifacts |

## Dependency Flow

```
@quackgraph/agent
    ├── depends on → @quackgraph/graph
    └── depends on → @quackgraph/native

@quackgraph/graph
    └── depends on → @quackgraph/native
```

The Agent extends and orchestrates the Core, not the other way around.
````

## File: packages/agent/src/lib/graph-instance.ts
````typescript
import type { QuackGraph } from '@quackgraph/graph';

let graphInstance: QuackGraph | null = null;

export function setGraphInstance(graph: QuackGraph) {
  graphInstance = graph;
}

export function getGraphInstance(): QuackGraph {
  if (!graphInstance) {
    throw new Error('Graph instance not initialized. Call setGraphInstance() first.');
  }
  return graphInstance;
}
````

## File: packages/agent/tsconfig.json
````json
{
  "extends": "../../tsconfig.json",
  "include": [
    "src/**/*"
  ],
  "compilerOptions": {
    "outDir": "dist",
    "rootDir": "src"
  }
}
````

## File: packages/agent/tsup.config.ts
````typescript
import { defineConfig } from 'tsup';

export default defineConfig({
  entry: ['src/index.ts'],
  format: ['cjs', 'esm'],
  dts: true,
  splitting: false,
  sourcemap: true,
  clean: true,
});
````

## File: scripts/git-sync.ts
````typescript
#!/usr/bin/env bun
/**
 * Git Sync Script - Federated Push for Nested Repositories
 * 
 * This script handles the synchronization of the nested git structure:
 * - quackgraph-agent (parent) -> Contains packages/agent
 * - packages/quackgraph (nested repo) -> The core engine
 * 
 * Usage:
 *   bun run scripts/git-sync.ts [message]
 *   bun run push:all
 * 
 * The script will:
 * 1. Check if the inner repo (packages/quackgraph) has changes
 * 2. Commit and push the inner repo if needed
 * 3. Update the parent repo with any changes (including submodule pointer if configured)
 * 4. Push the parent repo
 */

import { $ } from "bun";

const INNER_REPO_PATH = "packages/quackgraph";
const ROOT_DIR = import.meta.dir.replace("/scripts", "");

interface GitStatus {
    isDirty: boolean;
    isAhead: boolean;
    branch: string;
}

async function getGitStatus(cwd: string): Promise<GitStatus> {
    try {
        // Check for uncommitted changes
        const statusResult = await $`git -C ${cwd} status --porcelain`.text();
        const isDirty = statusResult.trim().length > 0;

        // Get current branch
        const branchResult = await $`git -C ${cwd} rev-parse --abbrev-ref HEAD`.text();
        const branch = branchResult.trim();

        // Check if ahead of remote
        let isAhead = false;
        try {
            const aheadResult = await $`git -C ${cwd} rev-list --count @{upstream}..HEAD 2>/dev/null`.text();
            isAhead = parseInt(aheadResult.trim(), 10) > 0;
        } catch {
            // No upstream configured, assume not ahead
            isAhead = false;
        }

        return { isDirty, isAhead, branch };
    } catch (error) {
        console.error(`Error getting git status for ${cwd}:`, error);
        throw error;
    }
}

async function commitAndPush(cwd: string, message: string, repoName: string): Promise<boolean> {
    const status = await getGitStatus(cwd);

    console.log(`\n📦 [${repoName}] Status:`);
    console.log(`   Branch: ${status.branch}`);
    console.log(`   Dirty: ${status.isDirty}`);
    console.log(`   Ahead of remote: ${status.isAhead}`);

    if (!status.isDirty && !status.isAhead) {
        console.log(`   ✅ Nothing to push for ${repoName}`);
        return false;
    }

    if (status.isDirty) {
        console.log(`\n   📝 Staging and committing changes in ${repoName}...`);
        await $`git -C ${cwd} add -A`.quiet();
        await $`git -C ${cwd} commit -m ${message}`.quiet();
        console.log(`   ✅ Committed: "${message}"`);
    }

    console.log(`\n   🚀 Pushing ${repoName} to remote...`);
    try {
        await $`git -C ${cwd} push`.quiet();
        console.log(`   ✅ Successfully pushed ${repoName}`);
        return true;
    } catch (error) {
        console.error(`   ❌ Failed to push ${repoName}:`, error);
        throw error;
    }
}

async function syncRepos(): Promise<void> {
    // Get commit message from args or use default
    const args = process.argv.slice(2);
    const commitMessage = args.join(" ") || `sync: ${new Date().toISOString()}`;

    console.log("🔄 Git Sync - Federated Repository Push");
    console.log("=========================================");
    console.log(`📝 Commit message: "${commitMessage}"`);

    const innerRepoPath = `${ROOT_DIR}/${INNER_REPO_PATH}`;

    // Step 1: Handle inner repository (quackgraph core)
    console.log("\n\n🔷 Step 1: Processing inner repository (quackgraph core)...");
    try {
        await commitAndPush(innerRepoPath, commitMessage, "quackgraph");
    } catch (error) {
        console.error("❌ Failed to sync inner repository");
        throw error;
    }

    // Step 2: Handle parent repository (quackgraph-agent)
    console.log("\n\n🔷 Step 2: Processing parent repository (quackgraph-agent)...");
    try {
        await commitAndPush(ROOT_DIR, commitMessage, "quackgraph-agent");
    } catch (error) {
        console.error("❌ Failed to sync parent repository");
        throw error;
    }

    console.log("\n\n=========================================");
    console.log("✅ Git sync completed successfully!");
    console.log("=========================================\n");
}

// Run the sync
syncRepos().catch((error) => {
    console.error("\n❌ Sync failed:", error);
    process.exit(1);
});
````

## File: LICENSE
````
MIT License

Copyright (c) 2025 quackgraph

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
````

## File: relay.config.json
````json
{
  "$schema": "https://relay.noca.pro/schema.json",
  "projectId": "quackgraph-agent",
  "core": {
    "logLevel": "info",
    "enableNotifications": false,
    "watchConfig": false
  },
  "watcher": {
    "clipboardPollInterval": 2000,
    "preferredStrategy": "auto"
  },
  "patch": {
    "approvalMode": "manual",
    "approvalOnErrorCount": 0,
    "linter": "",
    "preCommand": "",
    "postCommand": "",
    "minFileChanges": 0
  },
  "git": {
    "autoGitBranch": false,
    "gitBranchPrefix": "relay/",
    "gitBranchTemplate": "gitCommitMsg"
  }
}
````

## File: tsconfig.json
````json
{
    "compilerOptions": {
        // Environment setup & latest features
        "lib": [
            "ESNext"
        ],
        "target": "ESNext",
        "module": "Preserve",
        "moduleDetection": "force",
        "jsx": "react-jsx",
        "allowJs": true,
        // Bundler mode
        "moduleResolution": "bundler",
        "allowImportingTsExtensions": true,
        "verbatimModuleSyntax": true,
        "noEmit": true,
        // Best practices
        "strict": true,
        "skipLibCheck": true,
        "noFallthroughCasesInSwitch": true,
        "noUncheckedIndexedAccess": true,
        "noImplicitOverride": true,
        // Some stricter flags (disabled by default)
        "noUnusedLocals": false,
        "noUnusedParameters": false,
        "noPropertyAccessFromIndexSignature": false
    },
    "include": [
        "packages/agent/src/**/*",
        "packages/quackgraph/packages/*/src/**/*"
    ]
}
````

## File: packages/agent/src/agent/chronos.ts
````typescript
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
    const anchorRows = await this.graph.db.query(
      "SELECT valid_from FROM nodes WHERE id = ?",
      [anchorNodeId]
    );

    if (anchorRows.length === 0) {
      throw new Error(`Anchor node ${anchorNodeId} not found`);
    }

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
    const timeline: TimeStepDiff[] = [];

    // Initial state (baseline)
    let prevSummary: Map<string, number> = new Map();

    for (const ts of sortedTimes) {
      const micros = ts.getTime() * 1000;
      const currentSummaryList = await this.tools.getSectorSummary([anchorNodeId], micros);

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
````

## File: packages/agent/src/mastra/workflows/labyrinth-workflow.ts
````typescript
import { createStep, createWorkflow } from '@mastra/core/workflows';
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
        } catch(e) { console.warn("Router failed", e); }
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
    goal: z.string()
  }),
  outputSchema: z.object({
    foundWinner: z.boolean()
  }),
  stateSchema: LabyrinthStateSchema,
  execute: async ({ inputData, mastra, state, setState }) => {
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
        
        // 1. Max Hops Check
        if (cursor.stepCount >= config.maxHops) {
           deadThreads.push({ thread_id: cursor.id, status: 'KILLED', steps: cursor.stepHistory });
           return;
        }

        // 2. Fetch Node Metadata (LOD 1)
        const nodeMeta = await graph.match([]).where({ id: cursor.currentNodeId }).select();
        if (!nodeMeta[0]) return;
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

        try {
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
            if (res.usage) tokensUsed += (res.usage.promptTokens||0) + (res.usage.completionTokens||0);
            
            const decision = res.object;
            if (!decision) return;

            // Log step
            cursor.stepHistory.push({
                step: cursor.stepCount + 1,
                node_id: cursor.currentNodeId,
                action: decision.action,
                reasoning: decision.reasoning,
                ghost_view: sectorSummary.slice(0,3).map(s=>s.edgeType).join(',')
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
                            governance: state.governance,
                            execution: [],
                            judgment: { verdict: jRes.object.answer, confidence: jRes.object.confidence }
                        }
                    };
                     if (winner.metadata) winner.metadata.execution = [{ thread_id: cursor.id, status: 'COMPLETED', steps: cursor.stepHistory }];
                }
            } else if (decision.action === 'MOVE' && (decision.edgeType || decision.path)) {
                // Fork / Move Logic
                if (decision.path) {
                     // Multi-hop jump (from Navigational Map)
                     const target = decision.path.length > 0 ? decision.path[decision.path.length-1] : undefined;
                     if (target) {
                        nextCursors.push({ ...cursor, id: randomUUID(), currentNodeId: target, path: [...cursor.path, ...decision.path], stepCount: cursor.stepCount + decision.path.length, confidence: cursor.confidence * decision.confidence });
                     }
                } else if (decision.edgeType) {
                     // Single-hop move
                     const neighbors = await tools.topologyScan([cursor.currentNodeId], decision.edgeType, asOfTs);
                     // Speculative Forking: Take top 2 paths if ambiguous
                     for (const t of neighbors.slice(0, 2)) {
                        nextCursors.push({ ...cursor, id: randomUUID(), currentNodeId: t, path: [...cursor.path, t], stepCount: cursor.stepCount+1, confidence: cursor.confidence * decision.confidence });
                     }
                }
            }
        } catch(e) { 
           console.warn(`Thread ${cursor.id} failed:`, e);
           deadThreads.push({ thread_id: cursor.id, status: 'KILLED', steps: cursor.stepHistory });
        }
      });

      await Promise.all(promises);
      if (winner) break;

      // 6. Pruning (Survival of the Fittest)
      nextCursors.sort((a,b) => b.confidence - a.confidence);
      // Kill excess threads
      for(let i=config.maxCursors; i<nextCursors.length; i++) {
        const c = nextCursors[i];
        if (c) deadThreads.push({ thread_id: c.id, status: 'KILLED', steps: c.stepHistory });
      }
      cursors = nextCursors.slice(0, config.maxCursors);
    }
    
    // Cleanup if no winner
    if(!winner) {
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
  inputSchema: z.object({}),
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
  .commit();
````

## File: packages/agent/src/agent-schemas.ts
````typescript
import { z } from 'zod';

export const RouterDecisionSchema = z.object({
  domain: z.string(),
  confidence: z.number().min(0).max(1),
  reasoning: z.string(),
});

export const JudgeDecisionSchema = z.object({
  isAnswer: z.boolean(),
  answer: z.string(),
  confidence: z.number().min(0).max(1),
});

// Discriminated Union for Scout Actions
const MoveAction = z.object({
  action: z.literal('MOVE'),
  edgeType: z.string().optional().describe("The edge type to traverse (Single Hop)"),
  path: z.array(z.string()).optional().describe("Sequence of node IDs to traverse (Multi Hop)"),
  confidence: z.number().min(0).max(1),
  reasoning: z.string(),
  alternativeMoves: z.array(z.object({
    edgeType: z.string(),
    confidence: z.number(),
    reasoning: z.string()
  })).optional()
});

const CheckAction = z.object({
  action: z.literal('CHECK'),
  confidence: z.number().min(0).max(1),
  reasoning: z.string(),
});

const MatchAction = z.object({
  action: z.literal('MATCH'),
  pattern: z.array(z.object({
    srcVar: z.number(),
    tgtVar: z.number(),
    edgeType: z.string(),
    direction: z.string().optional()
  })),
  confidence: z.number().min(0).max(1),
  reasoning: z.string(),
});

const AbortAction = z.object({
  action: z.literal('ABORT'),
  confidence: z.number().min(0).max(1),
  reasoning: z.string(),
});

export const ScoutDecisionSchema = z.discriminatedUnion('action', [
  MoveAction,
  CheckAction,
  MatchAction,
  AbortAction
]);
````

## File: packages/agent/biome.json
````json
{
  "$schema": "https://biomejs.dev/schemas/2.3.8/schema.json",
  "vcs": {
    "enabled": true,
    "clientKind": "git",
    "useIgnoreFile": false
  },
  "files": {
    "ignoreUnknown": true,
    "includes": [
      "**",
      "!**/dist",
      "!**/node_modules"
    ]
  },
  "formatter": {
    "enabled": true,
    "indentStyle": "space",
    "indentWidth": 2,
    "lineWidth": 100
  },
  "linter": {
    "enabled": true,
    "rules": {
      "recommended": true
    }
  },
  "javascript": {
    "formatter": {
      "quoteStyle": "single",
      "trailingCommas": "es5"
    }
  }
}
````

## File: scripts/git-pull.ts
````typescript
#!/usr/bin/env bun
/**
 * Git Pull Script - Federated Pull for Nested Repositories
 * 
 * Usage:
 *   bun run scripts/git-pull.ts
 *   bun run pull:all
 */

import { $ } from "bun";

const INNER_REPO_PATH = "packages/quackgraph";
const ROOT_DIR = import.meta.dir.replace("/scripts", "");

async function pullRepo(cwd: string, repoName: string, repoUrl?: string): Promise<void> {
    console.log(`\n⬇️ [${repoName}] Processing...`);

    // Check if directory exists and has .git
    const fs = await import("node:fs/promises");
    const hasGit = await fs.exists(`${cwd}/.git`).catch(() => false);

    if (!hasGit && repoUrl) {
        console.log(`   ✨ Repository not found. Cloning from ${repoUrl}...`);
        try {
            // Ensure parent dir exists
            await $`mkdir -p ${cwd}`;
            // Remove the empty dir if it exists so clone works (or clone into it if empty)
            // Safest is to remove checking uniqueness or just run git clone
            // If cwd exists but is empty, git clone <url> <dir> works.

            await $`git clone ${repoUrl} ${cwd}`;
            console.log(`   ✅ Successfully cloned ${repoName}`);
            return;
        } catch (error) {
            console.error(`   ❌ Failed to clone ${repoName}:`, error);
            throw error;
        }
    }

    console.log(`   ⬇️ Pulling changes...`);
    try {
        await $`git -C ${cwd} pull`.quiet();
        console.log(`   ✅ Successfully pulled ${repoName}`);
    } catch (error) {
        console.error(`   ❌ Failed to pull ${repoName}:`, error);
        throw error;
    }
}

async function pullAll(): Promise<void> {
    console.log("🔄 Git Pull - Federated Repository Update");
    console.log("=========================================");

    // Pull parent first
    console.log("\n\n🔷 Step 1: Processing parent repository (quackgraph-agent)...");
    await pullRepo(ROOT_DIR, "quackgraph-agent");

    // Pull inner repo
    console.log("\n\n🔷 Step 2: Processing inner repository (quackgraph core)...");
    const innerRepoPath = `${ROOT_DIR}/${INNER_REPO_PATH}`;
    const innerRepoUrl = "https://github.com/quackgraph/quackgraph.git";

    // Custom logic to ensure 'agent' branch
    await pullRepo(innerRepoPath, "quackgraph", innerRepoUrl);
    // Force checkout agent branch if not already
    try {
        await $`git -C ${innerRepoPath} checkout agent`.quiet();
        await $`git -C ${innerRepoPath} pull origin agent`.quiet();
    } catch (e) {
        console.warn("   ⚠️ Could not checkout/pull agent branch explicitly:", e);
    }

    console.log("\n\n=========================================");
    console.log("✅ Git pull completed successfully!");
    console.log("=========================================\n");
}

pullAll().catch((error) => {
    console.error("\n❌ Pull failed:", error);
    process.exit(1);
});
````

## File: .gitignore
````
# Dependencies
node_modules/
bun.lock

# Build outputs
dist/
*.node
target/

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Relay state
/.relay/

# Logs
*.log
npm-debug.log*
````

## File: README.md
````markdown
# QuackGraph-Agent Labyrinth Report
**File ID:** `quackgraph-agent-labyrinth.md 83838384`
**Subject:** High-Performance Structural Inference & "Ghost Earth" Architecture
**Engine:** QuackGraph (Rust/Arrow) | **Orchestration:** Mastra AI | **Intelligence:** OpenRouter
**Date:** December 07, 2025

---

## 1. Executive Summary

This report defines the final architecture for **QuackLabyrinth**, an agentic retrieval system designed to obsolete "Flat RAG" (vector-only) and "Heavy Graph" (Graphiti/LangChain) approaches.

By decoupling **Topology (Structure)** from **Content (Data)**, QuackLabyrinth treats the LLM as a **Blind Pathfinder**—an engine that navigates a lightweight `u32` integer map in Rust without seeing the heavy textual content until the final moment of synthesis. This approach guarantees an **~82% reduction in token usage** and sub-millisecond graph traversals, enabling a new class of real-time, logic-heavy applications (e.g., Bearable-style Life Coaching, Cybersecurity, Supply Chain).

---

## 2. Core Architecture: The "Ghost Google Earth" Protocol

Standard RAG systems fail because they lack "Altitude." They search the entire database at ground level. QuackLabyrinth implements a **Semantic Level of Detail (S-LOD)** system, using ephemeral "Ghost Nodes" to guide the LLM from global context to specific data.

### 2.1 The Semantic Zoom (LODs)

1.  **LOD 0: Satellite View (The Ghost Layer)**
    *   **Data:** Dynamic Cluster Centroids (Virtual Nodes).
    *   **Action:** The Scout LLM selects a domain. "The user is asking about *Health*. Zoom into the Health Cluster."
    *   **Mechanism:** QuackGraph maintains background community detection. It exposes `GhostID`s representing entire subgraphs.

2.  **LOD 1: Drone View (The Structural Layer)**
    *   **Data:** The "Spine" (Entities & Relationships). No chunks.
    *   **Action:** The Scout navigates the topology. "Path: `(User) --[LOGGED]--> (Symptom: Headache) --[COINCIDES_WITH]--> (Diet: Caffeine)`."
    *   **Mechanism:** Integer-only traversal in Rust.

1.5 **LOD 1.5: The Ghost Map (Navigational Radar)**
    *   **Data:** ASCII Tree with geometric pruning (Depth 1-4).
    *   **Action:** The Scout requests `topology-scan(depth: 3)`.
    *   **Output:** `[ROOT] User:Alex ├──[HAS_SYMPTOM]──> (Migraine) 🔥 ...`
    *   **Benefit:** Enables multi-hop planning in a single inference step.

3.  **LOD 2: Street View (The Data Layer)**
    *   **Data:** Rich Text, PDF Chunks, JSON Blobs.
    *   **Action:** The Judge LLM reads the content.
    *   **Mechanism:** Zero-Copy Apache Arrow hydration from DuckDB.

---

## 3. Inference Logic: Dynamic Schema & Schema Injection

**Constraint:** Providing a massive edge schema (500+ types) to the LLM at the first call causes context bloat and confusion.

**Solution:** **Contextual Schema Injection.** The LLM is never provided with the full schema. It is provided with a **Local Routing Table**.

### 3.1 The Schema Protocol
1.  **Anchor Analysis:** Mastra AI identifies the domain (e.g., "Health").
2.  **Schema Pruning (Rust):** QuackGraph filters the schema registry. It retrieves only Edge Types valid for the "Health" cluster.
    *   *Included:* `CAUSED_BY`, `TREATED_WITH`, `OCCURRED_AT`.
    *   *Excluded:* `REPORTED_TO` (Corporate), `DEPLOYED_ON` (Tech).
3.  **Prompt Injection:** The Scout LLM receives:
    > "You are at Node [User]. Valid paths are: [CAUSED_BY, TREATED_WITH]. Which edge do you take?"

This ensures the LLM is not hallucinating relationships that don't exist in the current context, while keeping the input token count negligible.

---

## 4. The Parallel Labyrinth (Speculative Execution)

Since Rust traversals are effectively free (microseconds) compared to LLM generation (milliseconds), we utilize **Parallel Speculative Execution**.

### 4.1 The Forking Workflow
1.  **Ambiguity Detection:** If the Scout LLM assigns a 50/50 probability to two paths (e.g., "Was the headache caused by *Stress* or *Diet*?"), Mastra **forks** the process.
2.  **Parallel Threads:** Two Scout Agents run simultaneously on separate threads.
    *   *Thread A:* Explores the `(Stress)` subgraph.
    *   *Thread B:* Explores the `(Diet)` subgraph.
3.  **The Race:** The thread that finds a "Terminal Node" (a node with high relevance score or matching answer type) signals the Orchestrator to kill the other thread.
4.  **Result:** The user gets the answer twice as fast, effectively trading cheap CPU cycles for reduced user wait time.

---

## 5. Active Metabolism: The "Dreaming" State

To prevent the "Life Coach" graph from becoming a garbage dump of daily logs, the system implements an active maintenance cycle.

### 5.1 The Abstraction Ladder
When the system is idle (Dreaming), QuackGraph identifies dense clusters of low-value nodes (e.g., 30 days of "Mood: OK" logs).

1.  **Identification:** Rust scans for high-degree, low-centrality clusters.
2.  **Synthesis:** The Judge LLM reads the 30 logs and writes a single summary: "October was generally stable."
3.  **Rewiring:**
    *   Create new node: `[Summary: Oct 2025]`.
    *   Link to `[User]`.
    *   **Soft Delete** the 30 raw logs (Bitmasked `valid_to = false`).
    
**Benefit:** The "Parent Agent" (Life Coach) querying the past sees a clean, high-level timeline, not noise.

---

## 6. Integration: The "Life Coach" Parent Agent

How does a massive, multi-domain "Parent Agent" (like Bearable/Jarvis) utilize the QuackGraph-Agent?

### 6.1 The "Executive Briefing" Protocol
The Parent Agent does **not** see the raw graph traversal. Exposing the full trace (100+ hops and dead ends) would pollute the Parent's context window.

*   **Request:** Parent asks: *"Is there a correlation between my coffee intake and sleep?"*
*   **Quack Action:** The Labyrinth runs. It traverses `(Coffee) -> (Caffeine) -> (Sleep Latency)`.
*   **The Artifact:** QuackGraph returns a structured **Artifact Object**:
    ```json
    {
      "answer": "Yes, on days with >3 coffees, sleep latency increases by 40%.",
      "confidence": 0.92,
      "sources": [502, 891, 104], // Node IDs
      "trace_id": "uuid-trace-888"
    }
    ```
*   **Traceability:** If the Parent Agent doubts the answer (Self-Correction), it can call `getTrace("uuid-trace-888")`. Only then does QuackGraph render the full step-by-step reasoning tree for debugging or deep analysis.

---

## 7. Special Handling: PDFs & Unstructured Blobs

For large PDFs (Medical Reports, Manuals), we use the **Virtual Spine** topology.

1.  **The Spine:** A linear chain of nodes representing physical document flow. `[Page 1] --(NEXT)--> [Page 2]`.
2.  **The Ribs:** Entity Extraction links semantic concepts to specific Spine segments. `[Entity: "Insulin"] --(MENTIONED_IN)--> [Page 4]`.
3.  **Behavior:** The LLM traverses the semantic link to find "Insulin," then traverses the *Spine* to read the surrounding context (Page 3 and 5), reconstructing the narrative flow without embedding the whole document.

---

## 8. Resilience & "Pheromones"

To optimize efficiency over time without fine-tuning, the system uses **Ghost Traces**.

*   **Pheromones:** Every `u32` edge in Rust has a mutable `heat` counter.
*   **Reinforcement:** Successful paths (validated by user feedback) increment heat. Dead ends decrement heat.
*   **Heuristic:** The Scout LLM is prompted to prioritize "Hot" edges. The system effectively "learns" that `(Symptom) --(TREATED_BY)--> (Medication)` is a better path than `(Symptom) --(REPORTED_ON)--> (Date)` for medical queries.

---

## 9. Recommendations for Core QuackGraph (Rust)

To fully enable this architecture, the Rust core must implement:

1.  **Dynamic Bitmasking:** Support `layer_mask` in traversal to enable the Satellite/Drone views instantly.
2.  **Atomic Edge Metadata:** Allow `heat` (u8) to be updated atomically during read operations for the Pheromone system.
3.  **Schema Pruning API:** A fast method to return valid Edge Types for a given set of source Node IDs.

---

## 10. Conclusion

**QuackLabyrinth** is not just a database; it is a **Cognitive Operating System**.

*   **It forgets** (Dreaming/Pruning).
*   **It zooms** (Ghost Earth S-LOD).
*   **It learns** (Pheromone Traces).
*   **It specializes** (Dynamic Schema Injection).

By moving the complexity into the Rust/Architecture layer, we allow the LLM to remain small, fast, and focused, creating an agent that is orders of magnitude more efficient than current vector-based solutions.



# File: quackgraph-agent-labyrinth.md

*(Continuing from previous sections...)*

---

## 11. Complex Temporal Reasoning (The "Time Variance" Protocol)

**The Problem:** LLMs are notoriously bad at "Calendar Math."
*   *Query:* "Who was the project lead while the server was down?"
*   *LLM Struggle:* It has to compare Unix timestamps or distinct string dates (`2023-05-12` vs `May 12th`) across hundreds of nodes. It frequently hallucinates sequence (thinking 2022 happened after 2023 in long contexts).
*   *Vector Failure:* Embeddings capture semantic similarity, not temporal overlap. "Server Down" and "Project Lead" might be semantically close, but the vector doesn't know if they happened at the same minute.

**The QuackLabyrinth Solution:** We remove the concept of Time from the "Thinking Layer" (LLM) and push it entirely into the "Physics Layer" (Rust). The LLM does not calculate dates; it sets **Temporal Constraints** on the QuackGraph engine.

### 11.1 The "Time Travel" Slider (`asOf`)

QuackGraph treats the graph not as a static snapshot, but as a **4D Object**.

*   **Logic:** Every edge in the Rust core has `valid_from` and `valid_to` (u64 integers).
*   **The Protocol:**
    1.  **Query:** "Who was managing Bob *last September*?"
    2.  **Scout Action:** The Scout LLM extracts the target time: `Sep 2024`.
    3.  **Rust Execution:** `graph.traverse(source="Bob", edge="MANAGED_BY", asOf=1725148800)`.
    4.  **Physics:** The Rust engine applies a bitmask filter *during traversal*. It literally "hides" any edge that wasn't active at that second.
    5.  **Result:** The LLM receives only the manager valid at that instant (e.g., "Alice"). It never sees "Charlie" (who managed Bob in October).
    6.  **Token Savings:** 100% of irrelevant history is pruned before the LLM sees it.

### 11.2 Interval Algebra (The "During" Operator)

For queries involving duration overlap (e.g., "What errors occurred *during* the backup window?"), we implement **Allen’s Interval Algebra** natively in Rust.

*   **The Challenge:** A point-in-time check isn't enough. We need `Intersection(Window A, Window B) > 0`.
*   **The Data Structure:** QuackGraph uses an **Interval Tree** for edges with durations.
*   **The Workflow:**
    1.  **Scout:** Identifies the "Backup Window" node (Start: T1, End: T2).
    2.  **Instruction:** `graph.getEdges(type="ERROR", constraint="OVERLAPS", interval=[T1, T2])`.
    3.  **Rust Core:** Performs a specialized interval tree search ($O(\log N)$).
    4.  **Result:** Returns only errors that started, ended, or existed within that window.

### 11.3 Evolutionary Diffing (The "Movie Reel")

How do we answer: *"How has the team's focus changed since 2020?"*

Instead of feeding 5 years of logs to the LLM, we use **Temporal Sampling**.

1.  **Sampling:** Mastra requests "Ghost Earth" Satellite views at 3 intervals:
    *   `T1 (2020)`
    *   `T2 (2022)`
    *   `T3 (2024)`
2.  **Diffing:** The Scout LLM receives 3 small topology skeletons.
    *   *2020:* Focus -> (Legacy Code)
    *   *2022:* Focus -> (Migration)
    *   *2024:* Focus -> (AI Features)
3.  **Synthesis:** The Judge LLM narrates the evolution based on the changing topology.
4.  **Efficiency:** The LLM reads 3 summaries instead of 5,000 daily logs.

### 11.4 Causality Enforcement (The "Arrow of Time")

To prevent hallucinations where an effect precedes a cause.

*   **Mechanism:** When traversing a path defined as Causal (e.g., `CAUSED_BY`, `TRIGGERED`), the Rust engine enforces `Target.timestamp >= Source.timestamp`.
*   **Benefit:** If the LLM asks for a "Chain of Events," QuackGraph automatically filters out "Back to the Future" edges that would confuse the reasoning process.

### 11.5 Visualization: Temporal Filtering

```mermaid
graph LR
    subgraph "Full Database (The Mess)"
        A[Manager: Alice (2020-2022)]
        B[Manager: Bob (2023-Present)]
        C[Manager: Charlie (Acting, Jan 2023)]
        U[User: Dave]
        U --> A
        U --> B
        U --> C
    end

    subgraph "Query: 'Who managed Dave in 2021?'"
        direction TB
        Filter[Rust Time Filter: 2021]
        Result[User: Dave] -->|Visible Edge| Manager[Alice]
    end

    style B opacity:0.1
    style C opacity:0.1
```

### 11.6 Integration with Life Coaching (Bearable Example)

*   **Query:** *"Do my migraines happen after I eat sugar?"*
*   **Process:**
    1.  **Anchor:** Find all "Migraine" nodes.
    2.  **Lookback Window:** For each Migraine at $T$, query the graph for "Food" nodes in interval $[T - 4hours, T]$.
    3.  **Aggregation:** Rust counts the occurrences of "Sugar" in those windows.
    4.  **Judge LLM:** Receives the stats: "Sugar appeared in the 4-hour pre-window of 85% of migraines."
    5.  **Why it wins:** The LLM didn't have to look at 1,000 meal logs and calculate time deltas. Rust did the math; LLM did the storytelling.
````

## File: packages/agent/src/governance/schema-registry.ts
````typescript
import type { DomainConfig } from '../types';

export class SchemaRegistry {
  private domains = new Map<string, DomainConfig>();

  constructor() {
    // Default 'Global' domain that sees everything (fallback)
    this.register({
      name: 'global',
      description: 'Unrestricted access to the entire topology.',
      allowedEdges: [], // Empty means ALL allowed
      excludedEdges: []
    });
  }

  register(config: DomainConfig) {
    this.domains.set(config.name.toLowerCase(), config);
  }

  loadFromConfig(configs: DomainConfig[]) {
    configs.forEach(c => { this.register(c); });
  }

  getDomain(name: string): DomainConfig | undefined {
    return this.domains.get(name.toLowerCase());
  }

  getAllDomains(): DomainConfig[] {
    return Array.from(this.domains.values());
  }

  /**
   * Returns true if the edge type is allowed within the domain.
   * If domain is 'global' or not found, it defaults to true (permissive) 
   * unless strict mode is desired.
   */
  isEdgeAllowed(domainName: string, edgeType: string): boolean {
    const domain = this.domains.get(domainName.toLowerCase());
    if (!domain) return true;
    
    // 1. Check Exclusion (Blacklist)
    if (domain.excludedEdges?.includes(edgeType)) return false;

    // 2. Check Inclusion (Whitelist)
    if (domain.allowedEdges.length > 0) {
      return domain.allowedEdges.includes(edgeType);
    }

    // 3. Default Permissive
    return true;
  }

  getValidEdges(domainName: string): string[] | undefined {
    const domain = this.domains.get(domainName.toLowerCase());
    if (!domain || (domain.allowedEdges.length === 0 && (!domain.excludedEdges || domain.excludedEdges.length === 0))) {
      return undefined; // All allowed
    }
    return domain.allowedEdges;
  }

  /**
   * Returns true if the domain requires causal (monotonic) time traversal.
   */
  isDomainCausal(domainName: string): boolean {
    const domain = this.domains.get(domainName.toLowerCase());
    return !!domain?.isCausal;
  }
}

export const schemaRegistry = new SchemaRegistry();
export const getSchemaRegistry = () => schemaRegistry;
````

## File: packages/agent/src/mastra/agents/judge-agent.ts
````typescript
import { Agent } from '@mastra/core/agent';
import { Memory } from '@mastra/memory';
import { LibSQLStore } from '@mastra/libsql';

export const judgeAgent = new Agent({
  name: 'Judge Agent',
  instructions: `
    You are a Judge evaluating data from a Knowledge Graph.
    
    Input provided:
    - Goal: The user's question.
    - Data: Content of the nodes found.
    - Time Context: Relevant timeframe.
    
    Task: Determine if the data answers the goal.
  `,
  model: {
    id: 'groq/llama-3.3-70b-versatile',
  },
  memory: new Memory({
    storage: new LibSQLStore({
      url: ':memory:'
    })
  }),
});
````

## File: packages/agent/src/mastra/agents/router-agent.ts
````typescript
import { Agent } from '@mastra/core/agent';
import { Memory } from '@mastra/memory';
import { LibSQLStore } from '@mastra/libsql';

export const routerAgent = new Agent({
  name: 'Router Agent',
  instructions: `
    You are a Semantic Router for a Knowledge Graph.
    
    Task: Select the single most relevant domain (lens) to conduct the search based on the user's goal.
    
    Input provided:
    - Goal: User query.
    - Available Domains: List of domains and descriptions.
  `,
  model: {
    id: 'groq/llama-3.3-70b-versatile',
  },
  memory: new Memory({
    storage: new LibSQLStore({
      url: ':memory:'
    })
  }),
});
````

## File: packages/agent/src/mastra/tools/index.ts
````typescript
import { createTool } from '@mastra/core/tools';
import { z } from 'zod';
import { getGraphInstance } from '../../lib/graph-instance';
import { GraphTools } from '../../tools/graph-tools';
import { getSchemaRegistry } from '../../governance/schema-registry';

// We wrap the existing GraphTools logic to make it available to Mastra agents/workflows

export const sectorScanTool = createTool({
  id: 'sector-scan',
  description: 'Get a summary of available moves (edge types) from the current nodes (LOD 0). Context aware: filters by active domain.',
  inputSchema: z.object({
    nodeIds: z.array(z.string()),
    asOf: z.number().optional(),
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

    // 1. Resolve Context
    const ctxAsOf = runtimeContext?.get?.('asOf') as number | undefined;
    const ctxDomain = runtimeContext?.get?.('domain') as string | undefined;
    
    // Prioritize tool input (if agent explicitly sets it), fallback to runtime context
    const asOf = context.asOf ?? ctxAsOf;

    // 2. Resolve Governance
    const registry = getSchemaRegistry();
    const domainEdges = ctxDomain ? registry.getValidEdges(ctxDomain) : undefined;
    const effectiveAllowed = context.allowedEdgeTypes ?? domainEdges;

    const summary = await tools.getSectorSummary(context.nodeIds, asOf, effectiveAllowed);
    return { summary };
  },
});

export const topologyScanTool = createTool({
  id: 'topology-scan',
  description: 'Get IDs of neighbors reachable via a specific edge type (LOD 1)',
  inputSchema: z.object({
    nodeIds: z.array(z.string()),
    edgeType: z.string().optional(),
    asOf: z.number().optional(),
    minValidFrom: z.number().optional(),
    depth: z.number().min(1).max(4).optional(),
  }),
  outputSchema: z.object({
    neighborIds: z.array(z.string()).optional(),
    map: z.string().optional(),
    truncated: z.boolean().optional(),
  }),
  execute: async ({ context, runtimeContext }) => {
    const graph = getGraphInstance();
    const tools = new GraphTools(graph);
    
    // Resolve Context
    const ctxAsOf = runtimeContext?.get?.('asOf') as number | undefined;
    const ctxDomain = runtimeContext?.get?.('domain') as string | undefined;
    const asOf = context.asOf ?? ctxAsOf;
    
    // Enforce Domain Governance if implicit
    if (ctxDomain && context.edgeType) {
      const registry = getSchemaRegistry();
      if (!registry.isEdgeAllowed(ctxDomain, context.edgeType)) {
        return { neighborIds: [] }; // Silently block restricted edges
      }
    }

    if (context.depth && context.depth > 1) {
      // Ghost Map Mode (LOD 1.5)
      const maps = [];
      let truncated = false;
      for (const id of context.nodeIds) {
        // Note: NavigationalMap internal logic might need asOf update in future, currently uses standard scan
        const res = await tools.getNavigationalMap(id, context.depth, asOf);
        maps.push(res.map);
        if (res.truncated) truncated = true;
      }
      return { map: maps.join('\n\n'), truncated };
    }

    // Implicit map mode if no edgeType is provided, defaulting to depth 1 map
    if (!context.edgeType) {
        const maps = [];
        for (const id of context.nodeIds) {
            const res = await tools.getNavigationalMap(id, 1, asOf);
            maps.push(res.map);
        }
        return { map: maps.join('\n\n') };
    }

    const neighborIds = await tools.topologyScan(context.nodeIds, context.edgeType, asOf, context.minValidFrom);
    return { neighborIds };
  },
});

export const temporalScanTool = createTool({
  id: 'temporal-scan',
  description: 'Find neighbors connected via edges overlapping a specific time window',
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
    
    // Enforce Governance
    const ctxDomain = runtimeContext?.get?.('domain') as string | undefined;
    if (ctxDomain && context.edgeType) {
       const registry = getSchemaRegistry();
       if (!registry.isEdgeAllowed(ctxDomain, context.edgeType)) {
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
  description: 'Retrieve full content for nodes, including virtual spine expansion (LOD 2)',
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
````

## File: packages/agent/src/mastra/workflows/metabolism-workflow.ts
````typescript
import { createStep, createWorkflow } from '@mastra/core/workflows';
import { z } from 'zod';
import { getGraphInstance } from '../../lib/graph-instance';
import { randomUUID } from 'node:crypto';
import { JudgeDecisionSchema } from '../../agent-schemas';

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

    const response = await judge.generate(prompt, {
      structuredOutput: {
        schema: JudgeDecisionSchema
      }
    });
    let summaryText = '';

    try {
      const result = response.object;
      if (result && (result.isAnswer || result.answer)) {
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
````

## File: packages/agent/src/mastra/index.ts
````typescript
import { Mastra } from '@mastra/core/mastra';
import { LibSQLStore } from '@mastra/libsql';
// import { PinoLogger } from '@mastra/loggers';
import { scoutAgent } from './agents/scout-agent';
import { judgeAgent } from './agents/judge-agent';
import { routerAgent } from './agents/router-agent';
import { metabolismWorkflow } from './workflows/metabolism-workflow';
import { labyrinthWorkflow } from './workflows/labyrinth-workflow';

export const mastra = new Mastra({
  agents: { scoutAgent, judgeAgent, routerAgent },
  workflows: { metabolismWorkflow, labyrinthWorkflow },
  storage: new LibSQLStore({
    url: ':memory:',
  }),
  observability: {
    default: {
      enabled: true,

      // exporters: [new DefaultExporter()],
    },
  },
});
````

## File: package.json
````json
{
  "name": "quackgraph-agent",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "workspaces": [
    "packages/agent",
    "packages/quackgraph/packages/*"
  ],
  "scripts": {
    "format": "bun run --cwd packages/quackgraph format && bun run --cwd packages/agent format",
    "clean": "rm -rf node_modules && bun run --cwd packages/quackgraph clean && bun run --cwd packages/agent clean",
    "postinstall": "bun run build",
    "typecheck": "tsc --noEmit && bun run --cwd packages/agent typecheck && bun run --cwd packages/quackgraph typecheck",
    "lint": "bun run --cwd packages/quackgraph lint && bun run --cwd packages/agent lint",
    "check": "bun run typecheck && bun run lint",
    "dev": "bun test --watch",
    "test": "bun run build && bun test",
    "build": "bun run build:core && bun run build:agent",
    "build:core": "bun run --cwd packages/quackgraph build",
    "build:agent": "bun run --cwd packages/agent build",
    "build:watch": "bun run --cwd packages/agent build --watch",
    "push:all": "bun run scripts/git-sync.ts",
    "pull:all": "bun run scripts/git-pull.ts"
  },
  "devDependencies": {
    "@biomejs/biome": "latest",
    "@types/bun": "latest",
    "tsup": "^8.5.1",
    "typescript": "^5.0.0"
  }
}
````

## File: packages/agent/src/index.ts
````typescript
// Core Facade
export { Labyrinth } from './labyrinth';

// Types & Schemas
export * from './types';
export * from './agent-schemas';

// Utilities
export * from './agent/chronos';
export * from './governance/schema-registry';
export * from './tools/graph-tools';

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
    throw new Error('Required Mastra agents not found. Ensure scoutAgent, judgeAgent, and routerAgent are registered.');
  }

  return new Labyrinth(
    graph,
    { scout, judge, router },
    config
  );
}
````

## File: packages/agent/package.json
````json
{
  "name": "@quackgraph/agent",
  "version": "0.1.0",
  "main": "dist/index.js",
  "module": "dist/index.mjs",
  "types": "dist/index.d.ts",
  "exports": {
    ".": {
      "types": "./dist/index.d.ts",
      "import": "./dist/index.mjs",
      "require": "./dist/index.js"
    }
  },
  "scripts": {
    "build": "tsup",
    "dev": "tsup --watch",
    "clean": "rm -rf dist",
    "format": "biome format --write .",
    "lint": "biome lint .",
    "check": "biome check .",
    "typecheck": "tsc --noEmit"
  },
  "dependencies": {
    "@mastra/core": "^0.24.6",
    "@mastra/loggers": "^0.1.0",
    "@mastra/memory": "^0.15.12",
    "@mastra/libsql": "0.16.4",
    "@opentelemetry/api": "^1.8.0",
    "zod": "^3.23.0",
    "@quackgraph/graph": "workspace:*",
    "@quackgraph/native": "workspace:*"
  },
  "devDependencies": {
    "@biomejs/biome": "latest",
    "typescript": "^5.0.0",
    "tsup": "^8.0.0"
  }
}
````

## File: repomix.config.json
````json
{
  "$schema": "https://repomix.com/schemas/latest/schema.json",
  "input": {
    "maxFileSize": 52428800
  },
  "output": {
    "filePath": "dev-docs/repomix.md",
    "style": "markdown",
    "parsableStyle": true,
    "fileSummary": false,
    "directoryStructure": true,
    "files": true,
    "removeComments": false,
    "removeEmptyLines": false,
    "compress": false,
    "topFilesLength": 5,
    "showLineNumbers": false,
    "copyToClipboard": true,
    "git": {
      "sortByChanges": true,
      "sortByChangesMaxCommits": 100,
      "includeDiffs": false
    }
  },
  "include": [
//"README.md",
//"test-docs/"
//"test/",
//"test-docs/unit.test-plan.

],
  "ignore": {
    "useGitignore": true,
    "useDefaultPatterns": true,
    "customPatterns": [
            "packages/quackgraph",
      "dev-docs/flow.todo.md",
      "packages/quackgraph/.git",
      "packages/quackgraph/repomix.config.json",
      "packages/quackgraph/relay.config.json",
      "packages/quackgraph/.relay",
      "packages/quackgraph/dev-docs",
      "packages/quackgraph/LICENSE",
      "packages/quackgraph/.gitignore",
      "packages/quackgraph/tsconfig.tsbuildinfo",
"packages/quackgraph/packages/quack-graph/dist",
      "packages/quackgraph/test/",
      "packages/quackgraph/test-docs/",
      "packages/quackgraph/test/e2e/",
      "packages/quackgraph/src",
            "packages/quackgraph/RFC.README.md",
            "packages/quackgraph/README.md"
    ]
  },
  "security": {
    "enableSecurityCheck": true
  },
  "tokenCount": {
    "encoding": "o200k_base"
  }
}
````

## File: packages/agent/src/mastra/agents/scout-agent.ts
````typescript
import { Agent } from '@mastra/core/agent';
import { Memory } from '@mastra/memory';
import { LibSQLStore } from '@mastra/libsql';
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
    - **Radar Control (Depth):** You can request a "Ghost Map" (ASCII Tree) by using \`topology-scan\` with \`depth: 2\` or \`3\`.
      - Use Depth 1 to check immediate neighbors.
      - Use Depth 2-3 to explore structure without moving.
      - The map shows "🔥" for hot paths (high pheromones).

    - **Pheromones:** Edges marked with 🔥 or ♨️ have been successfully traversed before.
    - **Exploration:** 
      - Single Hop: Action "MOVE" with \`edgeType\`.
      - Multi Hop: If you see a path in the Ghost Map, Action "MOVE" with \`path: [id1, id2]\`.
    - **Pattern Matching:** To find a structure, action: "MATCH" with "pattern".
    - **Goal Check:** If the current node likely contains the answer, action: "CHECK".
    - **Abort:** If stuck or exhausted, action: "ABORT".
  `,
  model: {
    id: 'groq/llama-3.3-70b-versatile',
  },
  memory: new Memory({
    storage: new LibSQLStore({
      url: ':memory:'
    })
  }),
  tools: {
    sectorScanTool,
    topologyScanTool,
    temporalScanTool
  }
});
````

## File: packages/agent/src/tools/graph-tools.ts
````typescript
import type { QuackGraph } from '@quackgraph/graph';
import type { SectorSummary, LabyrinthContext } from '../types';
import type { JsPatternEdge } from '@quackgraph/native';

export class GraphTools {
  constructor(private graph: QuackGraph) { }

  private resolveAsOf(contextOrAsOf?: LabyrinthContext | number): number | undefined {
    if (typeof contextOrAsOf === 'number') return contextOrAsOf;
    if (!contextOrAsOf?.asOf) return undefined;
    return contextOrAsOf.asOf instanceof Date ? contextOrAsOf.asOf.getTime() : contextOrAsOf.asOf;
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
    // Note: optimization - native filtering is faster, but we rely on caller to pass correct allowedEdgeTypes from registry.getValidEdges()
    if (allowedEdgeTypes && allowedEdgeTypes.length > 0) {
      // redundant but safe if native implementation varies
      // no-op if native did its job
    }

    // 3. Sort by count (descending)
    return results.sort((a, b) => b.count - a.count);
  }

  /**
   * LOD 1.5: Ghost Map / Navigational Map
   * Generates an ASCII tree of the topology up to a certain depth.
   * Uses geometric pruning to keep the map readable.
   */
  async getNavigationalMap(rootId: string, depth: number = 1, contextOrAsOf?: LabyrinthContext | number): Promise<{ map: string, truncated: boolean }> {
    const maxDepth = Math.min(depth, 4);
    const treeLines: string[] = [`[ROOT] ${rootId}`];
    let isTruncated = false;
    const _asOf = this.resolveAsOf(contextOrAsOf);

    // Helper for recursion
    const buildTree = async (currentId: string, currentDepth: number, prefix: string) => {
      if (currentDepth >= maxDepth) return;

      // Geometric pruning: 10 -> 5 -> 3 -> 1
      const branchLimit = Math.floor(10 / (currentDepth + 1));
      let branchesCount = 0;

      // 1. Get stats to find "hot" edges
      const stats = await this.getSectorSummary([currentId], contextOrAsOf);
      
      for (const stat of stats) {
        if (branchesCount >= branchLimit) {
            isTruncated = true;
            break;
        }

        const edgeType = stat.edgeType;
        const heatMarker = (stat.avgHeat || 0) > 50 ? ' 🔥' : '';
        
        // 2. Traverse to get samples (fetch just enough to display)
        const neighbors = await this.topologyScan([currentId], edgeType, contextOrAsOf);
        const neighborLimit = Math.max(1, Math.floor(branchLimit / (stats.length || 1)) + 1); 
        const displayNeighbors = neighbors.slice(0, neighborLimit);
        
        for (let i = 0; i < displayNeighbors.length; i++) {
             if (branchesCount >= branchLimit) { isTruncated = true; break; }
             const neighborId = displayNeighbors[i];
             if (!neighborId) continue;
             const connector = (i === displayNeighbors.length - 1 && branchesCount === branchLimit - 1) ? '└──' : '├──';
             
             treeLines.push(`${prefix}${connector}[${edgeType}]──> (${neighborId})${heatMarker}`);
             
             const nextPrefix = prefix + (connector === '└──' ? '    ' : '│   ');
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
    // Native traverse does not support minValidFrom yet
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
      next.forEach(nid => { spineContextIds.add(nid); });

      // Look back
      const _prev = await this.graph.native.traverse([id], 'PREV', 'out');
      const incomingNext = await this.graph.native.traverse([id], 'NEXT', 'in');
      incomingNext.forEach(nid => { spineContextIds.add(nid); });

      const explicitPrev = await this.graph.native.traverse([id], 'PREV', 'out');
      explicitPrev.forEach(nid => { spineContextIds.add(nid); });
    }

    // Remove duplicates (original nodes)
    nodeIds.forEach(id => { spineContextIds.delete(id); });

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
    const _heatDelta = Math.floor(qualityScore * 50);

    for (let i = 1; i < trace.length; i++) {
      const prev = trace[i - 1];
      const curr = trace[i];
      if (!prev || !curr) continue; // Satisfy noUncheckedIndexedAccess
      if (curr.incomingEdge) {
        // await this.graph.updateEdgeHeat(prev.source, curr.source, curr.incomingEdge, heatDelta);
        console.warn('Pheromones not implemented in V1 native graph');
      }
    }
  }
}
````

## File: packages/agent/src/types.ts
````typescript
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
````

## File: packages/agent/src/labyrinth.ts
````typescript
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
 * In the new "Native Mastra" architecture, this class acts as a thin client
 * that orchestrates the `labyrinth-workflow` and injects the RuntimeContext
 * (Time Travel & Governance) into the Mastra execution environment.
 */
export class Labyrinth {
  public chronos: Chronos;
  public tools: GraphTools;
  public registry: SchemaRegistry;
  
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
    // Initialize Singleton for legacy tools support
    setGraphInstance(graph);
    
    // Utilities
    this.tools = new GraphTools(graph);
    this.chronos = new Chronos(graph, this.tools);
    this.registry = new SchemaRegistry();
  }

  /**
   * Registers a semantic domain (LOD 0 governance).
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

            // 1. Prepare Input Data
            const inputData = {
                goal,
                start,
                // Domain is left undefined here; the 'route-domain' step will decide it
                // unless we wanted to force it via config.
                maxHops: this.config.maxHops,
                maxCursors: this.config.maxCursors,
                confidenceThreshold: this.config.confidenceThreshold,
                timeContext: timeContext ? {
                    asOf: timeContext.asOf?.getTime(),
                    windowStart: timeContext.windowStart?.toISOString(),
                    windowEnd: timeContext.windowEnd?.toISOString()
                } : undefined
            };

            // 2. Execute Workflow
            const run = await workflow.createRunAsync();
            
            // Note: We pass strict inputs. The workflow steps handle State initialization.
            // RuntimeContext for 'asOf' is passed via the inputData.timeContext mostly,
            // but we can also inject it if the run start supported context overrides directly.
            // For now, the workflow steps extract timeContext from inputData and pass it 
            // to agents via runtimeContext.
            const result = await run.start({ inputData });
            
            // 3. Extract Result
            // @ts-expect-error - Result payload typing
            const artifact = result.results?.artifact as LabyrinthArtifact | null;
            
            if (artifact) {
              span.setAttribute('labyrinth.confidence', artifact.confidence);
              span.setAttribute('labyrinth.traceId', artifact.traceId);
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
}
````
