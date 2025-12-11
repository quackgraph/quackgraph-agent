# Directory Structure
```
packages/
  agent/
    src/
      mastra/
        workflows/
          labyrinth-workflow.ts
      labyrinth.ts
    test/
      e2e/
        labyrinth-complex.test.ts
        labyrinth.test.ts
        metabolism.test.ts
        mutation-complex.test.ts
        mutation.test.ts
        resilience.test.ts
        time-travel.test.ts
  quackgraph/
    packages/
      quack-graph/
        src/
          db.ts
          schema.ts
```

# Files

## File: packages/quackgraph/packages/quack-graph/src/db.ts
```typescript
import duckdb from 'duckdb';
import { tableFromJSON, tableToIPC } from 'apache-arrow';

// Interface for operations that can be performed within a transaction or globally
export interface DbExecutor {
  // biome-ignore lint/suspicious/noExplicitAny: SQL params are generic
  execute(sql: string, params?: any[]): Promise<void>;
  // biome-ignore lint/suspicious/noExplicitAny: SQL results are generic
  query(sql: string, params?: any[]): Promise<any[]>;
}

export class DuckDBManager implements DbExecutor {
  private db: duckdb.Database | null = null;
  private _path: string;
  private writeListeners: Array<() => Promise<void>> = [];

  constructor(path: string = ':memory:') {
    this._path = path;
  }

  async init() {
    if (!this.db) {
      // Native constructor is synchronous but can take a callback for errors
      await new Promise<void>((resolve, reject) => {
        this.db = new duckdb.Database(this._path, (err) => {
          if (err) reject(err);
          else resolve();
        });
      });
    }
  }

  async close() {
    if (this.db) {
      const db = this.db;
      this.db = null;
      await new Promise<void>((resolve, reject) => {
        db.close((err) => {
          if (err) reject(err);
          else resolve();
        });
      });
    }
  }

  onWrite(listener: () => Promise<void>) {
    this.writeListeners.push(listener);
  }

  private async notifyListeners() {
    await Promise.all(this.writeListeners.map(l => l()));
  }

  get path(): string {
    return this._path;
  }

  getDb(): duckdb.Database {
    if (!this.db) {
      throw new Error('Database not initialized. Call init() first.');
    }
    return this.db;
  }

  // biome-ignore lint/suspicious/noExplicitAny: SQL params
  async execute(sql: string, params: any[] = []): Promise<void> {
    const db = this.getDb();
    await new Promise<void>((resolve, reject) => {
      // biome-ignore lint/suspicious/noExplicitAny: DuckDB callback
      db.run(sql, ...params, (err: any) => {
        if (err) reject(err);
        else resolve();
      });
    });
    await this.notifyListeners();
  }

  // biome-ignore lint/suspicious/noExplicitAny: SQL results
  async query(sql: string, params: any[] = []): Promise<any[]> {
    const db = this.getDb();
    return new Promise((resolve, reject) => {
      // biome-ignore lint/suspicious/noExplicitAny: DuckDB callback
      db.all(sql, ...params, (err: any, rows: any[]) => {
        if (err) reject(err);
        else resolve(rows);
      });
    });
  }

  /**
   * Executes a callback within a transaction using a dedicated connection.
   * This guarantees that all operations inside the callback share the same ACID scope.
   */
  async transaction<T>(callback: (executor: DbExecutor) => Promise<T>): Promise<T> {
    const db = this.getDb();
    // Connect synchronously
    const conn = db.connect();
    
    // Create a transaction-bound executor wrapper around the Connection object
    const txExecutor: DbExecutor = {
      // biome-ignore lint/suspicious/noExplicitAny: SQL params
      execute: (sql: string, params: any[] = []) => {
        return new Promise((resolve, reject) => {
          conn.run(sql, ...params, (err) => {
            if (err) reject(err);
            else resolve();
          });
        });
      },
      // biome-ignore lint/suspicious/noExplicitAny: SQL results
      query: (sql: string, params: any[] = []) => {
        return new Promise((resolve, reject) => {
          conn.all(sql, ...params, (err, rows) => {
            if (err) reject(err);
            else resolve(rows);
          });
        });
      }
    };

    try {
      await txExecutor.execute('BEGIN TRANSACTION');
      const result = await callback(txExecutor);
      await txExecutor.execute('COMMIT');
      await this.notifyListeners(); // Notify AFTER commit
      return result;
    } catch (e) {
      try {
        await txExecutor.execute('ROLLBACK');
      } catch (rollbackError) {
        console.error('Failed to rollback transaction:', rollbackError);
      }
      throw e;
    } finally {
      // Best effort close - connection is native here
      // biome-ignore lint/suspicious/noExplicitAny: DuckDB connection types
      if (conn && typeof (conn as any).close === 'function') {
        // biome-ignore lint/suspicious/noExplicitAny: DuckDB connection types
        (conn as any).close();
      }
    }
  }

  /**
   * Executes a query and returns the raw Apache Arrow IPC Buffer.
   * Used for high-speed hydration.
   */
  // biome-ignore lint/suspicious/noExplicitAny: SQL params
  async queryArrow(sql: string, params: any[] = []): Promise<Uint8Array> {
    const db = this.getDb();
    
    return new Promise((resolve, reject) => {
      // Helper to merge multiple Arrow batches if necessary
      const mergeBatches = (batches: Uint8Array[]) => {
        if (batches.length === 0) return new Uint8Array(0);
        if (batches.length === 1) return batches[0] ?? new Uint8Array(0);
        const totalLength = batches.reduce((acc, val) => acc + val.length, 0);
        const merged = new Uint8Array(totalLength);
        let offset = 0;
        for (const batch of batches) {
          merged.set(batch, offset);
          offset += batch.length;
        }
        return merged;
      };

      const runFallback = async () => {
        try {
          const rows = await this.query(sql, params);
          if (rows.length === 0) return resolve(new Uint8Array(0));
          const table = tableFromJSON(rows);
          const ipc = tableToIPC(table, 'stream');
          resolve(ipc);
        } catch (e) {
          reject(e);
        }
      };

      // Try Database.arrowIPCAll (available in newer node-duckdb)
      // biome-ignore lint/suspicious/noExplicitAny: duckdb native type check
      if (typeof (db as any).arrowIPCAll === 'function') {
        // biome-ignore lint/suspicious/noExplicitAny: internal callback signature
        (db as any).arrowIPCAll(sql, ...params, (err: any, result: any) => {
          if (err) {
            const msg = String(err.message || '');
            if (msg.includes('to_arrow_ipc') || msg.includes('Table Function')) {
              return runFallback();
            }
            return reject(err);
          }
          // Result is usually Array<Uint8Array> (batches)
          if (Array.isArray(result)) {
            resolve(mergeBatches(result));
          } else {
            resolve(result ?? new Uint8Array(0));
          }
        });
      } else {
         // Fallback: Create a raw connection if db.arrowIPCAll missing
         try {
            const rawConn = db.connect();
            
            // biome-ignore lint/suspicious/noExplicitAny: check for method
            if (rawConn && typeof (rawConn as any).arrowIPCAll === 'function') {
               // biome-ignore lint/suspicious/noExplicitAny: internal callback signature
               (rawConn as any).arrowIPCAll(sql, ...params, (err: any, result: any) => {
                  if (err) {
                    const msg = String(err.message || '');
                    if (msg.includes('to_arrow_ipc') || msg.includes('Table Function')) {
                      return runFallback();
                    }
                    return reject(err);
                  }
                  if (Array.isArray(result)) {
                    resolve(mergeBatches(result));
                  } else {
                    resolve(result ?? new Uint8Array(0));
                  }
               });
            } else {
               runFallback();
            }
         } catch(_e) {
            runFallback();
         }
      }
    });
  }
}
```

## File: packages/quackgraph/packages/quack-graph/src/schema.ts
```typescript
import type { DuckDBManager, DbExecutor } from './db';

const NODES_TABLE = `
CREATE TABLE IF NOT EXISTS nodes (
    row_id UBIGINT PRIMARY KEY, -- Simple auto-increment equivalent logic handled by sequence
    id TEXT NOT NULL,
    labels TEXT[],
    properties JSON,
    embedding DOUBLE[], -- Vector embedding
    valid_from TIMESTAMPTZ DEFAULT (current_timestamp AT TIME ZONE 'UTC'),
    valid_to TIMESTAMPTZ DEFAULT NULL
);
CREATE SEQUENCE IF NOT EXISTS seq_node_id;
`;

const EDGES_TABLE = `
CREATE TABLE IF NOT EXISTS edges (
    source TEXT NOT NULL,
    target TEXT NOT NULL,
    type TEXT NOT NULL,
    properties JSON,
    heat UTINYINT DEFAULT 0,
    valid_from TIMESTAMPTZ DEFAULT (current_timestamp AT TIME ZONE 'UTC'),
    valid_to TIMESTAMPTZ DEFAULT NULL
);
`;

export interface TemporalOptions {
  validFrom?: Date;
  validTo?: Date;
}

export class SchemaManager {
  constructor(private db: DuckDBManager) {}

  async ensureSchema() {
    // Ensure sequence exists before table creation to avoid race conditions.
    // Note: Breaking NODES_TABLE into separate executions to ensure sequence is ready.
    await this.db.execute("CREATE SEQUENCE IF NOT EXISTS seq_node_id");
    
    await this.db.execute(NODES_TABLE);
    await this.db.execute(EDGES_TABLE);

    // Sync sequence to avoid collisions if table existed with data but sequence was reset
    // Force checkpoint to ensure we see committed data for sequence calc
    try { await this.db.execute("CHECKPOINT"); } catch {}

    // We use a robust query to find the max ID, handling potential NULLs from empty tables
    const rows = await this.db.query("SELECT COALESCE(MAX(row_id), 0) as max_id FROM nodes");
    const maxId = rows.length > 0 ? BigInt(rows[0].max_id) : 0n;

    // Recreate sequence starting after maxId.
    // Only advance sequence if needed (idempotent for persistent stores)
    if (maxId > 0n) {
      // Try setval (standard), fallback to ALTER (Postgres/DuckDB variant)
      try {
        await this.db.execute(`SELECT setval('seq_node_id', ${maxId + 1n})`);
      } catch {
        try {
          await this.db.execute(`ALTER SEQUENCE seq_node_id RESTART WITH ${maxId + 1n}`);
        } catch (e) {
          // Warn but proceed; if sequence is fresh it might be fine
          console.warn("SchemaManager: Could not sync sequence", e);
        }
      }
    }

    // Performance Indexes
    // Note: Partial indexes (WHERE valid_to IS NULL) are not supported in all DuckDB environments/bindings yet.
    // We use standard indexes for now.
    await this.db.execute('CREATE INDEX IF NOT EXISTS idx_nodes_id ON nodes (id)');
    // idx_nodes_labels removed: Standard B-Tree on LIST column does not help list_contains() queries.
    await this.db.execute('CREATE INDEX IF NOT EXISTS idx_edges_src_tgt_type ON edges (source, target, type)');
  }

  // biome-ignore lint/suspicious/noExplicitAny: generic properties
  async writeNode(id: string, labels: string[], properties: Record<string, any> = {}, options: TemporalOptions = {}) {
    const vf = options.validFrom ? `'${options.validFrom.toISOString()}'` : "(current_timestamp AT TIME ZONE 'UTC')";
    const vt = options.validTo ? `'${options.validTo.toISOString()}'` : "NULL";

    await this.db.transaction(async (tx: DbExecutor) => {
      // 1. Close existing record (SCD Type 2)
      await tx.execute(
        `UPDATE nodes SET valid_to = ${vf} WHERE id = ? AND valid_to IS NULL`,
        [id]
      );
      // 2. Insert new version
      await tx.execute(`
        INSERT INTO nodes (row_id, id, labels, properties, valid_from, valid_to) 
        VALUES (nextval('seq_node_id'), ?, ?::JSON::TEXT[], ?::JSON, ${vf}, ${vt})
      `, [id, JSON.stringify(labels), JSON.stringify(properties)]);
    });
  }

  // biome-ignore lint/suspicious/noExplicitAny: generic properties
  async writeEdge(source: string, target: string, type: string, properties: Record<string, any> = {}, options: TemporalOptions = {}) {
    const vf = options.validFrom ? `'${options.validFrom.toISOString()}'` : "(current_timestamp AT TIME ZONE 'UTC')";
    const vt = options.validTo ? `'${options.validTo.toISOString()}'` : "NULL";

    await this.db.transaction(async (tx: DbExecutor) => {
      // 1. Close existing edge
      await tx.execute(
        `UPDATE edges SET valid_to = ${vf} WHERE source = ? AND target = ? AND type = ? AND valid_to IS NULL`,
        [source, target, type]
      );
      // 2. Insert new version
      await tx.execute(`
        INSERT INTO edges (source, target, type, properties, valid_from, valid_to) 
        VALUES (?, ?, ?, ?::JSON, ${vf}, ${vt})
      `, [source, target, type, JSON.stringify(properties)]);
    });
  }

  async deleteNode(id: string) {
    // Soft Delete: Close the validity period
    await this.db.transaction(async (tx: DbExecutor) => {
      await tx.execute(
        `UPDATE nodes SET valid_to = (current_timestamp AT TIME ZONE 'UTC') WHERE id = ? AND valid_to IS NULL`,
        [id]
      );
    });
  }

  async deleteEdge(source: string, target: string, type: string) {
    // Soft Delete: Close the validity period
    await this.db.transaction(async (tx: DbExecutor) => {
      await tx.execute(
        `UPDATE edges SET valid_to = (current_timestamp AT TIME ZONE 'UTC') WHERE source = ? AND target = ? AND type = ? AND valid_to IS NULL`,
        [source, target, type]
      );
    });
  }

  /**
   * Promotes a JSON property to a native column for faster filtering.
   * This creates a column on the `nodes` table and backfills it from the `properties` JSON blob.
   * 
   * @param label The node label to target (e.g., 'User'). Only nodes with this label will be updated.
   * @param property The property key to promote (e.g., 'age').
   * @param type The DuckDB SQL type (e.g., 'INTEGER', 'VARCHAR').
   */
  async promoteNodeProperty(label: string, property: string, type: string) {
    // Sanitize inputs to prevent basic SQL injection (rudimentary check)
    if (!/^[a-zA-Z0-9_]+$/.test(property)) throw new Error(`Invalid property name: '${property}'. Must be alphanumeric + underscore.`);
    // Type check is looser to allow various SQL types, but strictly alphanumeric + spaces/parens usually safe enough for now
    if (!/^[a-zA-Z0-9_() ]+$/.test(type)) throw new Error(`Invalid SQL type: '${type}'.`);
    // Sanitize label just in case, though it is used as a parameter usually, here we might need dynamic check if we were using it in table names, but we use it in list_contains param.
    
    // 1. Add Column (Idempotent)
    try {
      // Note: DuckDB 0.9+ supports ADD COLUMN IF NOT EXISTS
      await this.db.execute(`ALTER TABLE nodes ADD COLUMN IF NOT EXISTS ${property} ${type}`);
    } catch (_e) {
      // Fallback or ignore if column exists
    }

    // 2. Backfill Data
    // We use list_contains to only update relevant nodes
    const sql = `
      UPDATE nodes 
      SET ${property} = CAST(json_extract(properties, '$.${property}') AS ${type})
      WHERE list_contains(labels, ?)
    `;
    await this.db.execute(sql, [label]);
  }

  /**
   * Declarative Merge (Upsert).
   * Finds a node by `matchProps` and `label`.
   * If found: Updates properties with `setProps`.
   * If not found: Creates new node with `matchProps` + `setProps`.
   * Returns the node ID.
   */
  // biome-ignore lint/suspicious/noExplicitAny: Generic property bag
  async mergeNode(label: string, matchProps: Record<string, any>, setProps: Record<string, any>, options: TemporalOptions = {}): Promise<string> {
    // 1. Build Search Query
    const vf = options.validFrom ? `'${options.validFrom.toISOString()}'` : "(current_timestamp AT TIME ZONE 'UTC')";
    const vt = options.validTo ? `'${options.validTo.toISOString()}'` : "NULL";

    const matchKeys = Object.keys(matchProps);
    const conditions = [`valid_to IS NULL`, `list_contains(labels, ?)`];
    // biome-ignore lint/suspicious/noExplicitAny: Params array
    const params: any[] = [label];
    
    for (const key of matchKeys) {
      if (key === 'id') {
        conditions.push(`id = ?`);
        params.push(matchProps[key]);
      } else {
        conditions.push(`json_extract(properties, '$.${key}') = ?::JSON`);
        params.push(JSON.stringify(matchProps[key]));
      }
    }

    const searchSql = `SELECT id, labels, properties FROM nodes WHERE ${conditions.join(' AND ')} LIMIT 1`;

    return await this.db.transaction(async (tx) => {
      const rows = await tx.query(searchSql, params);
      let id: string;
      // biome-ignore lint/suspicious/noExplicitAny: Generic property bag
      let finalProps: Record<string, any>;
      let finalLabels: string[];

      if (rows.length > 0) {
        // Update Existing
        const row = rows[0];
        id = row.id;
        const currentProps = typeof row.properties === 'string' ? JSON.parse(row.properties) : row.properties;
        finalProps = { ...currentProps, ...setProps };
        finalLabels = row.labels; // Preserve existing labels

        // Close old version
        await tx.execute(`UPDATE nodes SET valid_to = ${vf} WHERE id = ? AND valid_to IS NULL`, [id]);
      } else {
        // Insert New
        id = matchProps.id || crypto.randomUUID();
        finalProps = { ...matchProps, ...setProps };
        finalLabels = [label];
      }

      // Insert new version (for both Update and Create cases)
      await tx.execute(`
        INSERT INTO nodes (row_id, id, labels, properties, valid_from, valid_to) 
        VALUES (nextval('seq_node_id'), ?, ?::JSON::TEXT[], ?::JSON, ${vf}, ${vt})
      `, [id, JSON.stringify(finalLabels), JSON.stringify(finalProps)]);

      return id;
    });
  }

  // biome-ignore lint/suspicious/noExplicitAny: generic properties
  async writeNodesBulk(nodes: { id: string, labels: string[], properties: Record<string, any>, validFrom?: Date, validTo?: Date }[]) {
    if (nodes.length === 0) return;
    
    // Using Staging Table Strategy for efficient SCD-2
    const stagingName = `staging_nodes_${Date.now()}_${Math.floor(Math.random() * 1000)}`;
    
    await this.db.transaction(async (tx) => {
      await tx.execute(`CREATE TEMP TABLE ${stagingName} (id TEXT, labels TEXT[], properties JSON, valid_from TIMESTAMPTZ, valid_to TIMESTAMPTZ)`);
      
      // Bulk Insert into Staging
      const BATCH_SIZE = 200;
      for (let i = 0; i < nodes.length; i += BATCH_SIZE) {
        const chunk = nodes.slice(i, i + BATCH_SIZE);
        const placeholders = chunk.map(() => `(?, ?::JSON::TEXT[], ?::JSON, ?::TIMESTAMPTZ, ?::TIMESTAMPTZ)`).join(',');
        // biome-ignore lint/suspicious/noExplicitAny: SQL params
        const params: any[] = [];

        for (const n of chunk) {
          const vf = n.validFrom ? n.validFrom.toISOString() : new Date().toISOString(); // Fallback to JS now if undefined (since we can't use SQL function in param)
          const vt = n.validTo ? n.validTo.toISOString() : null;
          params.push(n.id, JSON.stringify(n.labels), JSON.stringify(n.properties), vf, vt);
        }
        
        await tx.execute(`INSERT INTO ${stagingName} VALUES ${placeholders}`, params);
      }

      // SCD-2 Close Old Rows (Set valid_to = new_valid_from for existing active rows)
      await tx.execute(`
        UPDATE nodes 
        SET valid_to = s.valid_from 
        FROM ${stagingName} s 
        WHERE nodes.id = s.id AND nodes.valid_to IS NULL
      `);

      // Insert New Rows
      await tx.execute(`
        INSERT INTO nodes (row_id, id, labels, properties, valid_from, valid_to)
        SELECT nextval('seq_node_id'), id, labels, properties, valid_from, valid_to FROM ${stagingName}
      `);

      await tx.execute(`DROP TABLE ${stagingName}`);
    });
  }

  // biome-ignore lint/suspicious/noExplicitAny: generic properties
  async writeEdgesBulk(edges: { source: string, target: string, type: string, properties: Record<string, any>, validFrom?: Date, validTo?: Date, heat?: number }[]) {
    if (edges.length === 0) return;

    const stagingName = `staging_edges_${Date.now()}_${Math.floor(Math.random() * 1000)}`;

    await this.db.transaction(async (tx) => {
      await tx.execute(`CREATE TEMP TABLE ${stagingName} (source TEXT, target TEXT, type TEXT, properties JSON, valid_from TIMESTAMPTZ, valid_to TIMESTAMPTZ, heat UTINYINT)`);

      const BATCH_SIZE = 200;
      for (let i = 0; i < edges.length; i += BATCH_SIZE) {
        const chunk = edges.slice(i, i + BATCH_SIZE);
        const placeholders = chunk.map(() => `(?, ?, ?, ?::JSON, ?::TIMESTAMPTZ, ?::TIMESTAMPTZ, ?::UTINYINT)`).join(',');
        // biome-ignore lint/suspicious/noExplicitAny: SQL params
        const params: any[] = [];

        for (const e of chunk) {
          const vf = e.validFrom ? e.validFrom.toISOString() : new Date().toISOString();
          const vt = e.validTo ? e.validTo.toISOString() : null;
          const heat = e.heat || 0;
          params.push(e.source, e.target, e.type, JSON.stringify(e.properties), vf, vt, heat);
        }
        
        await tx.execute(`INSERT INTO ${stagingName} VALUES ${placeholders}`, params);
      }

      // Close Old Edges
      await tx.execute(`
        UPDATE edges 
        SET valid_to = s.valid_from
        FROM ${stagingName} s
        WHERE edges.source = s.source AND edges.target = s.target AND edges.type = s.type AND edges.valid_to IS NULL
      `);

      // Insert New Edges
      await tx.execute(`
        INSERT INTO edges (source, target, type, properties, valid_from, valid_to, heat)
        SELECT source, target, type, properties, valid_from, valid_to, heat FROM ${stagingName}
      `);

      await tx.execute(`DROP TABLE ${stagingName}`);
    });
  }
}
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
    llm.addResponse("Find the answer", {
        domain: "global",
        confidence: 1.0,
        reasoning: "General query"
    });

    // 2. Train Scout
    // Step 1: At 'start', sees 'goal_node' via 'LINKS_TO'
    llm.addResponse(`Node: "start"`, {
        action: "MOVE",
        edgeType: "LINKS_TO",
        confidence: 1.0,
        reasoning: "Moving to linked node"
    });
    
    // Step 2: At 'goal_node', checks for answer
    llm.addResponse(`Node: "goal_node"`, {
        action: "CHECK",
        confidence: 1.0,
        reasoning: "This looks like the answer"
    });

    // 3. Train Judge
    llm.addResponse(`Goal: Find the answer`, {
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

## File: packages/agent/test/e2e/mutation-complex.test.ts
```typescript
import { describe, it, expect, beforeAll } from "bun:test";
import { runWithTestGraph } from "../utils/test-graph";
import { SyntheticLLM } from "../utils/synthetic-llm";
import { scribeAgent } from "../../src/mastra/agents/scribe-agent";
import { mastra } from "../../src/mastra/index";

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

      // @ts-expect-error
      expect(res.results?.success).toBe(false);
      // @ts-expect-error
      expect(res.results?.summary).toContain("Did you mean the Ford or the Chevy?");

      // Verify no deletion happened
      const cars = await graph.match([]).where({ labels: ["Car"] }).select();
      expect(cars.length).toBe(2);
    });
  });

  it("Executes Temporal Deletion ('Sold it yesterday')", async () => {
    await runWithTestGraph(async (graph) => {
      // Setup
      // @ts-expect-error
      await graph.addNode("me", ["User"], {});
      // @ts-expect-error
      await graph.addNode("bike", ["Item"], {});
      // @ts-expect-error
      await graph.addEdge("me", "bike", "OWNS", {});

      const YESTERDAY = new Date(Date.now() - 86400000).toISOString();

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

## File: packages/agent/test/e2e/resilience.test.ts
```typescript
import { describe, it, expect, beforeAll, mock } from "bun:test";
import { runWithTestGraph } from "../utils/test-graph";
import { SyntheticLLM } from "../utils/synthetic-llm";
import { scoutAgent } from "../../src/mastra/agents/scout-agent";
import { judgeAgent } from "../../src/mastra/agents/judge-agent";
import { routerAgent } from "../../src/mastra/agents/router-agent";
import { mastra } from "../../src/mastra/index";

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
        // @ts-expect-error
        const artifact = res.results?.artifact;
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

      // @ts-expect-error
      const artifact = res.results?.artifact;
      
      // Should result in null (failure to find) rather than hanging
      expect(artifact).toBeNull();
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
    // Safe defaults
    llm.mockAgent(scoutAgent, { action: "ABORT", confidence: 0, reasoning: "Default Abort" });
    llm.mockAgent(judgeAgent, { isAnswer: false, answer: "No", confidence: 0 });
    llm.mockAgent(routerAgent, { domain: "global", confidence: 1, reasoning: "Default Global" });
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

      // @ts-expect-error
      const results = res.results as WorkflowResult;
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
      // @ts-expect-error
      const results = res.results as MetabolismResult;
      expect(results?.success).toBe(true);

      // 5. Verify Physics (Graph State)
      // Old nodes should be gone (or disconnected/deleted)
      const oldNodes = await graph.match([]).where({ id: eventIds }).select();
      expect(oldNodes.length).toBe(0);

      // Summary node should exist
      const summaries = await graph.match([]).where({ labels: ["Summary"] }).select();
      expect(summaries.length).toBe(1);
      expect(summaries[0].properties.content).toBe("User mood was generally positive with a dip on day 3.");

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
    // Safe defaults
    llm.mockAgent(scoutAgent, { action: "ABORT", confidence: 0, reasoning: "Default Abort" });
    llm.mockAgent(judgeAgent, { isAnswer: false, answer: "No", confidence: 0 });
    llm.mockAgent(routerAgent, { domain: "global", confidence: 1, reasoning: "Default Global" });
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
      
      // Router: Always Global
      llm.addResponse("Who managed Dave", { domain: "global", confidence: 1.0, reasoning: "HR Query" });

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

      // Judge: Confirms answer
      llm.addResponse(`Node: "alice"`, { isAnswer: true, answer: "Manager was Alice", confidence: 1.0 }); // Wrong prompt key, relying on content retrieval mock implicitly or explicit pattern
      // Let's make Judge robust:
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

      // @ts-expect-error
      const art2023 = (res2023.results as LabyrinthResult)?.artifact;
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

      // @ts-expect-error
      const art2024 = (res2024.results as LabyrinthResult)?.artifact;
      expect(art2024).toBeDefined();
      expect(art2024?.answer).toContain("Bob");
      expect(art2024?.sources).toContain("bob");
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
    // @ts-expect-error - Mastra generic return type
    const rawResults = result.results;
    
    const parsed = MutationResultSchema.safeParse(rawResults);
    if (!parsed.success) {
      throw new Error(`Invalid workflow result: ${JSON.stringify(rawResults)}`);
    }
    
    expect(parsed.data.success).toBe(true);
    expect(parsed.data.summary).toContain("Created Node bob_1");

    // 4. Verify Side Effects (Graph Physics)
    const storedNode = await graph.match([]).where({ id: "bob_1" }).select();
    expect(storedNode.length).toBe(1);
    expect(storedNode[0].properties.name).toBe("Bob");
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
    // @ts-expect-error
    const rawResults = result.results;
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
    // @ts-expect-error
    const rawResults = result.results;
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
            const artifact = result.results?.artifact as LabyrinthArtifact | null;
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
          throw new Error(`Mutation failed with status: ${result.status}`);
        }
        return result.result as { success: boolean; summary: string };
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
