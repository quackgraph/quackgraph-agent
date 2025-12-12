# Directory Structure
```
packages/
  agent/
    src/
      agent/
        chronos.ts
      mastra/
        workflows/
          metabolism-workflow.ts
          mutation-workflow.ts
    test/
      integration/
        chronos.test.ts
      unit/
        chronos.test.ts
      utils/
        chaos-graph.ts
  quackgraph/
    packages/
      quack-graph/
        src/
          db.ts
          graph.ts
          schema.ts
    test/
      unit/
        graph.test.ts
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
      
      // Force UTC timezone for consistent timestamp handling across all environments
      // This prevents timezone conversion issues when storing/retrieving TIMESTAMPTZ values
      await this.execute("SET TimeZone='UTC'");
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

## File: packages/quackgraph/packages/quack-graph/src/graph.ts
```typescript
import { NativeGraph } from '@quackgraph/native';
import { DuckDBManager } from './db';
import { SchemaManager } from './schema';
import { QueryBuilder } from './query';

class WriteLock {
  private mutex: Promise<void> = Promise.resolve();

  run<T>(fn: () => Promise<T>): Promise<T> {
    // Chain the new operation to the existing promise
    const result = this.mutex.then(() => fn());

    // Update the mutex to wait for the new operation to complete (success or failure)
    // We strictly return void so the mutex remains Promise<void>
    this.mutex = result.then(
      () => { },
      () => { }
    );

    return result;
  }
}

export class QuackGraph {
  db: DuckDBManager;
  schema: SchemaManager;
  native: InstanceType<typeof NativeGraph>;
  private writeLock = new WriteLock();
  private _isInternalWrite = false;

  capabilities = {
    vss: false
  };

  // Context for the current instance (Time Travel)
  context: {
    asOf?: Date;
    topologySnapshot?: string;
  } = {};

  constructor(path: string = ':memory:', options: { asOf?: Date, topologySnapshot?: string } = {}) {
    this.db = new DuckDBManager(path);
    this.schema = new SchemaManager(this.db);
    this.native = new NativeGraph();
    this.context.asOf = options.asOf;
    this.context.topologySnapshot = options.topologySnapshot;
  }

  async init() {
    await this.db.init();

    // Load Extensions
    try {
      await this.db.execute("INSTALL vss; LOAD vss;");
      this.capabilities.vss = true;
    } catch (e) {
      console.warn("QuackGraph: Failed to load 'vss' extension. Vector search will be disabled.", e);
    }

    await this.schema.ensureSchema();

    // If we are in time-travel mode, we might skip hydration or hydrate a snapshot (Advanced).
    // For V1, we always hydrate "Current Active" topology.

    // Check for Topology Snapshot
    let snapshotLoaded = false;
    if (this.context.topologySnapshot) {
      try {
        // Try loading from disk
        this.native.loadSnapshot(this.context.topologySnapshot);
        snapshotLoaded = true;
        // If successful, skip hydration
        // We don't return here because we still need to set up the listener below
      } catch (e) {
        console.warn(`QuackGraph: Failed to load snapshot '${this.context.topologySnapshot}'. Falling back to full hydration.`, e);
      }
    }

    try {
      if (!snapshotLoaded) {
        await this.hydrate();
      }
    } catch (e) {
      console.error("Failed to hydrate graph topology from disk:", e);
      // We don't throw here to allow partial functionality (metadata queries) if needed,
      // but usually this is fatal for graph operations.
      throw e;
    }

    // Setup Reactive Consistency
    // Registered LAST to ensure initialization DDL doesn't trigger premature hydration/reloads
    this.db.onWrite(async () => {
      // If the write didn't originate from our internal methods, it's external (e.g. test SQL, or direct DB manipulation)
      // We must reload the graph to maintain split-brain consistency.
      if (!this._isInternalWrite) {
        await this.reload();
      }
    });
  }

  async reload() {
    await this.writeLock.run(async () => {
      // Clear RAM index
      if (typeof this.native.clear === 'function') {
        this.native.clear();
      } else {
        // Fallback for older native bindings if necessary, though clear() is added now
        // biome-ignore lint/suspicious/noExplicitAny: native method
        (this.native as any).clear?.();
      }
      
      // Re-hydrate
      try {
        await this.hydrate();
      } catch (e) {
        console.warn("QuackGraph: Auto-reload hydration failed", e);
      }
    });
  }

  /**
   * Hydrates the in-memory Rust graph from the persistent DuckDB storage.
   * This is critical for the "Split-Brain" architecture.
   */
  async hydrate() {
    // Zero-Copy Arrow IPC
    // We load ALL edges (active and historical) to support time-travel.
    // We cast valid_from/valid_to to DOUBLE to ensure JS/JSON compatibility (avoiding BigInt issues in fallback)
    try {
      const ipcBuffer = await this.db.queryArrow(
        `SELECT source, target, type, heat,
                date_diff('us', '1970-01-01'::TIMESTAMPTZ, valid_from) as valid_from, 
                date_diff('us', '1970-01-01'::TIMESTAMPTZ, valid_to) as valid_to 
         FROM edges`
      );

      if (ipcBuffer && ipcBuffer.length > 0) {
        // Napi-rs expects a Buffer or equivalent
        // Buffer.from is zero-copy in Node for Uint8Array usually, or cheap copy
        // We cast to any to satisfy the generated TS definitions which might expect Buffer
        const bufferForNapi = Buffer.isBuffer(ipcBuffer)
          ? ipcBuffer
          : Buffer.from(ipcBuffer);

        this.native.loadArrowIpc(bufferForNapi);

        // Reclaim memory after burst hydration
        this.native.compact();
      }
      // biome-ignore lint/suspicious/noExplicitAny: error handling
    } catch (e: any) {
      throw new Error(`Hydration Error: ${e.message}`);
    }
  }

  async close() {
    await this.db.close();
    // Clear RAM index
    if (typeof this.native.clear === 'function') {
      this.native.clear();
    } else {
      // biome-ignore lint/suspicious/noExplicitAny: native method
      (this.native as any).clear?.();
    }
  }

  asOf(date: Date): QuackGraph {
    // Return a shallow copy with new context
    const g = new QuackGraph(this.db.path, { asOf: date });
    // Share the same DB connection and Native index (assuming topology is shared/latest)
    g.db = this.db;
    g.schema = this.schema;
    g.native = this.native;
    g.capabilities = { ...this.capabilities };
    return g;
  }

  // --- Write Operations (Write-Through) ---

  // biome-ignore lint/suspicious/noExplicitAny: generic properties
  async addNode(id: string, labels: string[], props: Record<string, any> = {}, options: { validFrom?: Date, validTo?: Date } = {}) {
    await this.writeLock.run(async () => {
      this._isInternalWrite = true;
      try {
        // 1. Write to Disk (Source of Truth)
        await this.schema.writeNode(id, labels, props, options);
        // 2. Write to RAM (Cache)
        // Note: Rust V1 Index doesn't track node validity history, only existence.
        this.native.addNode(id);
      } finally {
        this._isInternalWrite = false;
      }
    });
  }

  // biome-ignore lint/suspicious/noExplicitAny: generic properties
  async addNodes(nodes: { id: string, labels: string[], properties: Record<string, any>, validFrom?: Date, validTo?: Date }[]) {
    await this.writeLock.run(async () => {
      this._isInternalWrite = true;
      try {
        await this.schema.writeNodesBulk(nodes);
        const ids = nodes.map(n => n.id);
        this.native.addNodes(ids);
      } finally {
        this._isInternalWrite = false;
      }
    });
  }

  // biome-ignore lint/suspicious/noExplicitAny: generic properties
  async addEdge(source: string, target: string, type: string, props: Record<string, any> = {}, options: { validFrom?: Date, validTo?: Date, heat?: number } = {}) {
    await this.writeLock.run(async () => {
      this._isInternalWrite = true;
      try {
        // Generate timestamp in JS to ensure RAM and DB match for deduplication
        const vfDate = options.validFrom || new Date();

        // 1. Write to Disk (Source of Truth) - Pass explicit date to match RAM
        await this.schema.writeEdge(source, target, type, props, { ...options, validFrom: vfDate });
        // 2. Write to RAM (Cache)
        const vf = vfDate.getTime();
        const vt = options.validTo ? options.validTo.getTime() : undefined;
        const heat = options.heat || 0;
        this.native.addEdge(source, target, type, vf, vt, heat);
      } finally {
        this._isInternalWrite = false;
      }
    });
  }

  // biome-ignore lint/suspicious/noExplicitAny: generic properties
  async addEdges(edges: { source: string, target: string, type: string, properties: Record<string, any>, validFrom?: Date, validTo?: Date, heat?: number }[]) {
    await this.writeLock.run(async () => {
      this._isInternalWrite = true;
      try {
        const now = new Date();
        // Normalize validFrom in JS so DB and RAM receive the exact same value
        const edgesWithTime = edges.map(e => ({
            ...e,
            validFrom: e.validFrom || now
        }));

        await this.schema.writeEdgesBulk(edgesWithTime);
        const sources = edgesWithTime.map(e => e.source);
        const targets = edgesWithTime.map(e => e.target);
        const types = edgesWithTime.map(e => e.type);
        // Rust Napi expects parallel arrays
        const validFroms = edgesWithTime.map(e => e.validFrom?.getTime() ?? 0);
        const validTos = edgesWithTime.map(e => e.validTo ? e.validTo.getTime() : Number.MAX_SAFE_INTEGER);
        const heats = edgesWithTime.map(e => e.heat || 0);

        this.native.addEdges(sources, targets, types, validFroms, validTos, heats);
      } finally {
        this._isInternalWrite = false;
      }
    });
  }

  async deleteNode(id: string) {
    await this.writeLock.run(async () => {
      this._isInternalWrite = true;
      try {
        // 1. Write to Disk (Soft Delete)
        await this.schema.deleteNode(id);
        // 2. Write to RAM (Tombstone)
        this.native.removeNode(id);
      } finally {
        this._isInternalWrite = false;
      }
    });
  }

  async deleteEdge(source: string, target: string, type: string) {
    await this.writeLock.run(async () => {
      this._isInternalWrite = true;
      try {
        // 1. Write to Disk (Soft Delete)
        await this.schema.deleteEdge(source, target, type);
        // 2. Write to RAM (Remove)
        this.native.removeEdge(source, target, type);
      } finally {
        this._isInternalWrite = false;
      }
    });
  }

  // --- Pheromones & Schema (Agent) ---

  async updateEdgeHeat(source: string, target: string, type: string, heat: number) {
    await this.writeLock.run(async () => {
      this._isInternalWrite = true;
      try {
        // 1. Write to Disk (In-Place Update for Learning Signal)
        // We only update active edges (valid_to IS NULL)
        await this.db.execute(
          "UPDATE edges SET heat = ? WHERE source = ? AND target = ? AND type = ? AND valid_to IS NULL",
          [heat, source, target, type]
        );
        // 2. Write to RAM (Atomic)
        this.native.updateEdgeHeat(source, target, type, heat);
      } finally {
        this._isInternalWrite = false;
      }
    });
  }

  async getAvailableEdgeTypes(sources: string[]): Promise<string[]> {
    // Fast scan of the CSR index
    // Returns unique edge types outgoing from the source set
    // Used for Labyrinth LOD 0 (Sector Scan)
    const asOfTs = this.context.asOf ? this.context.asOf.getTime() : undefined;
    return this.native.getAvailableEdgeTypes(sources, asOfTs);
  }

  async getSectorStats(sources: string[], allowedEdgeTypes?: string[]): Promise<{ edgeType: string; count: number; avgHeat: number }[]> {
    // Fast scan of the CSR index with aggregation
    // Returns counts and heat for outgoing edge types
    const asOfTs = this.context.asOf ? this.context.asOf.getTime() : undefined;
    return this.native.getSectorStats(sources, asOfTs, allowedEdgeTypes);
  }

  async traverseInterval(sources: string[], edgeType: string | undefined, direction: 'out' | 'in' = 'out', start: Date, end: Date, constraint: 'overlaps' | 'contains' | 'during' | 'meets' = 'overlaps'): Promise<string[]> {
    const s = start.getTime();
    const e = end.getTime();
    // If end < start, return empty
    if (e <= s) return [];
    return this.native.traverseInterval(sources, edgeType, direction, s, e, constraint);
  }

  /**
   * Upsert a node.
   * @param label Primary label to match.
   * @param matchProps Properties to match against (e.g. { email: '...' }).
   * @param setProps Properties to set/update if found or created.
   */
  // biome-ignore lint/suspicious/noExplicitAny: Generic property bag
  async mergeNode(label: string, matchProps: Record<string, any>, setProps: Record<string, any> = {}, options: { validFrom?: Date, validTo?: Date } = {}) {
    return this.writeLock.run(async () => {
      this._isInternalWrite = true;
      try {
        const id = await this.schema.mergeNode(label, matchProps, setProps, options);
        // Update cache
        this.native.addNode(id);
        return id;
      } finally {
        this._isInternalWrite = false;
      }
    });
  }

  // --- Optimization & Maintenance ---

  get optimize() {
    return {
      promoteProperty: async (label: string, property: string, type: string) => {
        await this.schema.promoteNodeProperty(label, property, type);
      },
      saveTopologySnapshot: (path: string) => {
        this.native.saveSnapshot(path);
      }
    };
  }

  // --- Read Operations ---

  match(labels: string[]): QueryBuilder {
    return new QueryBuilder(this, labels);
  }
}
```

## File: packages/quackgraph/packages/quack-graph/src/schema.ts
```typescript
import type { DuckDBManager, DbExecutor } from './db';

const NODES_TABLE = `
CREATE TABLE IF NOT EXISTS nodes (
    row_id UBIGINT NOT NULL, -- Simple auto-increment equivalent logic handled by sequence
    id TEXT NOT NULL,
    labels TEXT[],
    properties JSON,
    embedding DOUBLE[], -- Vector embedding
    valid_from TIMESTAMPTZ DEFAULT current_timestamp,
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
    valid_from TIMESTAMPTZ DEFAULT current_timestamp,
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
      const nextVal = maxId + 1n;
      // Try setval (standard), fallback to ALTER (Postgres/DuckDB variant), fallback to DROP/CREATE
      try {
        await this.db.execute(`SELECT setval('seq_node_id', ${nextVal})`);
      } catch {
        try {
          await this.db.execute(`ALTER SEQUENCE seq_node_id RESTART WITH ${nextVal}`);
        } catch (_e) {
          try {
            await this.db.execute(`DROP SEQUENCE IF EXISTS seq_node_id`);
            await this.db.execute(`CREATE SEQUENCE seq_node_id START ${nextVal}`);
          } catch (finalErr) {
            console.warn("SchemaManager: Could not sync sequence", finalErr);
          }
        }
      }
    }

    // Performance Indexes
    // Note: Partial indexes (WHERE valid_to IS NULL) are not supported in all DuckDB environments/bindings yet.
    // We use standard indexes for now.
    // NOTE: row_id does NOT have a uniqueness constraint (PRIMARY KEY or UNIQUE INDEX) due to a DuckDB bug
    // where updating array columns on tables with such constraints causes "Duplicate key" errors.
    // Since row_id is only used internally for SCD-2 versioning and we query by 'id' instead, this is safe.
    await this.db.execute('CREATE INDEX IF NOT EXISTS idx_nodes_row_id ON nodes (row_id)');
    await this.db.execute('CREATE INDEX IF NOT EXISTS idx_nodes_id ON nodes (id)');
    // idx_nodes_labels removed: Standard B-Tree on LIST column does not help list_contains() queries.
    await this.db.execute('CREATE INDEX IF NOT EXISTS idx_edges_src_tgt_type ON edges (source, target, type)');
  }

  // biome-ignore lint/suspicious/noExplicitAny: generic properties
  async writeNode(id: string, labels: string[], properties: Record<string, any> = {}, options: TemporalOptions = {}) {
    const vf = options.validFrom ? `'${options.validFrom.toISOString()}'` : "(current_timestamp)";
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
    const vf = options.validFrom ? `'${options.validFrom.toISOString()}'` : "(current_timestamp)";
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
        `UPDATE nodes SET valid_to = (current_timestamp) WHERE id = ? AND valid_to IS NULL`,
        [id]
      );
    });
  }

  async deleteEdge(source: string, target: string, type: string) {
    // Soft Delete: Close the validity period
    await this.db.transaction(async (tx: DbExecutor) => {
      await tx.execute(
        `UPDATE edges SET valid_to = (current_timestamp) WHERE source = ? AND target = ? AND type = ? AND valid_to IS NULL`,
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
    const vf = options.validFrom ? `'${options.validFrom.toISOString()}'` : "(current_timestamp)";
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

## File: packages/quackgraph/test/unit/graph.test.ts
```typescript
import { describe, test, expect, beforeEach, afterEach } from 'bun:test';
import { createGraph, cleanupGraph } from '../utils/helpers';
import type { QuackGraph } from '../../packages/quack-graph/src/index';

describe('Unit: QuackGraph Core', () => {
  let g: QuackGraph;
  let path: string;

  beforeEach(async () => {
    const setup = await createGraph('memory');
    g = setup.graph;
    path = setup.path;
  });

  afterEach(async () => {
    await cleanupGraph(path);
  });

  test('should initialize with zero nodes', () => {
    expect(g.native.nodeCount).toBe(0);
    expect(g.native.edgeCount).toBe(0);
  });

  test('should add nodes and increment count', async () => {
    await g.addNode('u:1', ['User'], { name: 'Alice' });
    await g.addNode('u:2', ['User'], { name: 'Bob' });
    
    // Check Rust Index
    expect(g.native.nodeCount).toBe(2);
    
    // Check DuckDB Storage
    const rows = await g.db.query('SELECT * FROM nodes');
    expect(rows.length).toBe(2);
    expect(rows.find(r => r.id === 'u:1').properties).toContain('Alice');
  });

  test('should add edges and support traversal', async () => {
    await g.addNode('a', ['Node']);
    await g.addNode('b', ['Node']);
    await g.addEdge('a', 'b', 'LINK');

    expect(g.native.edgeCount).toBe(1);

    // Simple manual traversal check using native directly
    const neighbors = g.native.traverse(['a'], 'LINK', 'out');
    expect(neighbors).toEqual(['b']);
  });

  test('should be idempotent when adding same edge', async () => {
    await g.addNode('a', ['Node']);
    await g.addNode('b', ['Node']);
    
    await g.addEdge('a', 'b', 'LINK');
    await g.addEdge('a', 'b', 'LINK'); // Duplicate

    expect(g.native.edgeCount).toBe(1);
    const neighbors = g.native.traverse(['a'], 'LINK', 'out');
    expect(neighbors).toEqual(['b']);
  });

  test('should soft delete nodes and stop traversal', async () => {
    await g.addNode('a', ['Node']);
    await g.addNode('b', ['Node']);
    await g.addEdge('a', 'b', 'LINK');

    let neighbors = g.native.traverse(['a'], 'LINK', 'out');
    expect(neighbors).toEqual(['b']);

    await g.deleteNode('b');

    // Rust index should treat it as tombstoned
    neighbors = g.native.traverse(['a'], 'LINK', 'out');
    expect(neighbors).toEqual([]);

    // Check DB soft delete
    const rows = await g.db.query("SELECT * FROM nodes WHERE id = 'b' AND valid_to IS NOT NULL");
    expect(rows.length).toBe(1);
  });
});
```

## File: packages/agent/test/utils/chaos-graph.ts
```typescript
import type { QuackGraph } from '@quackgraph/graph';

/**
 * Simulates data corruption by injecting invalid or unexpected schema data directly into the DB.
 */
export async function corruptNode(graph: QuackGraph, nodeId: string) {
    // We assume properties are stored as JSON string in 'nodes' table.
    // We inject a string that is technically valid JSON but breaks expected schema,
    // or invalid JSON if the DB allows (SQLite/DuckDB often allows loose text).
    
    const badData = '{"corrupted": true, "critical_field": null, "unexpected_array": [1,2,3]}';
    
    // Direct SQL injection to bypass application-layer validation
    await graph.db.execute(
        `UPDATE nodes SET properties = ? WHERE id = ?`,
        [badData, nodeId]
    );
}

/**
 * Simulates a network partition or edge loss.
 * Can be used to test resilience against missing links.
 */
export async function severConnection(graph: QuackGraph, source: string, target: string, type: string) {
    // 1. Soft Delete (Time Travel)
    const now = new Date().toISOString();
    await graph.db.execute(
        `UPDATE edges SET valid_to = ? WHERE source = ? AND target = ? AND type = ? AND valid_to IS NULL`,
        [now, source, target, type]
    );
    
    // 2. Force removal from RAM index if applicable to simulation
    // (Assuming graph.native has a remove method exposed or we rely on reload)
    try {
        // @ts-expect-error - native method might vary
        if (graph.native.removeEdge) {
            // @ts-expect-error
            graph.native.removeEdge(source, target, type);
        }
    } catch (e) {
        console.warn("Could not remove edge from native index manually:", e);
    }
}
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
            const existing = await graph.db.query('SELECT labels FROM nodes WHERE id = ?', [op.match.id]);
            if (existing.length > 0 && existing[0].labels && existing[0].labels.length > 0) {
                label = existing[0].labels[0];
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
                   await graph.db.execute(
                       `UPDATE nodes SET valid_to = ? WHERE id = ? AND valid_to IS NULL`, 
                       [validTo.toISOString(), op.id]
                   );
                   // Update RAM
                   graph.native.removeNode(op.id);
              } else {
                  await graph.deleteNode(op.id);
              }
              summaryLines.push(`Deleted Node ${op.id}`);
              break;
          }
          case 'CLOSE_EDGE': {
              if (validTo) {
                   await graph.db.execute(
                       `UPDATE edges SET valid_to = ? WHERE source = ? AND target = ? AND type = ? AND valid_to IS NULL`, 
                       [validTo.toISOString(), op.source, op.target, op.type]
                   );
                   // Update RAM: remove old edge and re-add with validTo
                   // First, get the edge properties from DB
                   const existingEdge = await graph.db.query(
                       `SELECT valid_from, valid_to, heat FROM edges WHERE source = ? AND target = ? AND type = ?`,
                       [op.source, op.target, op.type]
                   );
                   graph.native.removeEdge(op.source, op.target, op.type);
                   if (existingEdge[0]) {
                       graph.native.addEdge(
                           op.source, 
                           op.target, 
                           op.type, 
                           new Date(existingEdge[0].valid_from).getTime(),
                           new Date(existingEdge[0].valid_to).getTime(),
                           existingEdge[0].heat || 0
                       );
                   }
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

## File: packages/agent/src/mastra/workflows/metabolism-workflow.ts
```typescript
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
    const externalParents = allParents.filter((p: string) => !candidateSet.has(p));

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
    const anchorRows = await this.graph.db.query(
      "SELECT valid_from FROM nodes WHERE id = ?",
      [anchorNodeId]
    );

    if (anchorRows.length === 0) {
      throw new Error(`Anchor node ${anchorNodeId} not found`);
    }

    const sql = `
      WITH Anchor AS (
        SELECT valid_from::TIMESTAMPTZ as t_anchor 
        FROM nodes 
        WHERE id = ?
      ),
      Targets AS (
        SELECT id, valid_from::TIMESTAMPTZ as t_target 
        FROM nodes 
        WHERE list_contains(labels, ?)
      )
      SELECT count(*) as count
      FROM Targets, Anchor
      WHERE t_target >= (t_anchor - (INTERVAL 1 MINUTE * ${Math.floor(windowMinutes)}))
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
