# Directory Structure
```
packages/
  quackgraph/
    packages/
      quack-graph/
        src/
          db.ts
          graph.ts
          schema.ts
        package.json
    tsup.config.ts
package.json
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

    // Setup Reactive Consistency
    this.db.onWrite(async () => {
      // If the write didn't originate from our internal methods, it's external (e.g. test SQL, or direct DB manipulation)
      // We must reload the graph to maintain split-brain consistency.
      if (!this._isInternalWrite) {
        await this.reload();
      }
    });

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
    if (this.context.topologySnapshot) {
      try {
        // Try loading from disk
        this.native.loadSnapshot(this.context.topologySnapshot);
        // If successful, skip hydration
        return;
      } catch (e) {
        console.warn(`QuackGraph: Failed to load snapshot '${this.context.topologySnapshot}'. Falling back to full hydration.`, e);
      }
    }

    try {
      await this.hydrate();
    } catch (e) {
      console.error("Failed to hydrate graph topology from disk:", e);
      // We don't throw here to allow partial functionality (metadata queries) if needed,
      // but usually this is fatal for graph operations.
      throw e;
    }
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
                date_diff('us', '1970-01-01'::TIMESTAMPTZ, valid_from)::DOUBLE as valid_from, 
                date_diff('us', '1970-01-01'::TIMESTAMPTZ, valid_to)::DOUBLE as valid_to 
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
        // 1. Write to Disk
        await this.schema.writeEdge(source, target, type, props, options);
        // 2. Write to RAM
        const vf = options.validFrom ? options.validFrom.getTime() * 1000 : undefined;
        const vt = options.validTo ? options.validTo.getTime() * 1000 : undefined;
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
        await this.schema.writeEdgesBulk(edges);
        const sources = edges.map(e => e.source);
        const targets = edges.map(e => e.target);
        const types = edges.map(e => e.type);
        // Rust Napi expects parallel arrays
        const validFroms = edges.map(e => e.validFrom ? e.validFrom.getTime() * 1000 : 0);
        const validTos = edges.map(e => e.validTo ? e.validTo.getTime() * 1000 : Number.MAX_SAFE_INTEGER);
        const heats = edges.map(e => e.heat || 0);

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
    const asOfTs = this.context.asOf ? this.context.asOf.getTime() * 1000 : undefined;
    return this.native.getAvailableEdgeTypes(sources, asOfTs);
  }

  async getSectorStats(sources: string[], allowedEdgeTypes?: string[]): Promise<{ edgeType: string; count: number; avgHeat: number }[]> {
    // Fast scan of the CSR index with aggregation
    // Returns counts and heat for outgoing edge types
    const asOfTs = this.context.asOf ? this.context.asOf.getTime() * 1000 : undefined;
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
    await this.db.execute(NODES_TABLE);
    await this.db.execute(EDGES_TABLE);

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

## File: packages/quackgraph/packages/quack-graph/package.json
```json
{
  "name": "@quackgraph/graph",
  "version": "0.1.0",
  "main": "src/index.ts",
  "module": "dist/quack-graph.esm.js",
  "types": "src/index.ts",
  "type": "module",
  "exports": {
    ".": {
      "types": "./src/index.ts",
      "import": "./dist/quack-graph.esm.js",
      "require": "./dist/quack-graph.cjs"
    }
  },
  "files": [
    "dist"
  ],
  "scripts": {
    "build": "tsup",
    "build:watch": "tsup --watch",
    "clean": "rm -rf dist"
  },
  "dependencies": {
    "duckdb": "1.1.3",
    "apache-arrow": "^17.0.0",
    "@quackgraph/native": "workspace:*"
  },
  "devDependencies": {
    "typescript": "^5.0.0",
    "@types/node": "^20.0.0"
  }
}
```

## File: packages/quackgraph/tsup.config.ts
```typescript
import { defineConfig } from 'tsup';

export default defineConfig({
  entry: ['packages/quack-graph/src/index.ts'],
  format: ['cjs', 'esm'],
  dts: true,
  splitting: false,
  shims: false,
  sourcemap: true,
  clean: true,
  outDir: 'packages/quack-graph/dist',
  external: ['duckdb-async', 'apache-arrow', '@quackgraph/native'],
  target: 'es2020',
});
```

## File: package.json
```json
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
    "pull:all": "bun run scripts/git-pull.ts",
    "dev:mastra": "bun run --cwd packages/agent dev:mastra"
  },
  "devDependencies": {
    "@biomejs/biome": "latest",
    "@types/bun": "latest",
    "tsup": "^8.5.1",
    "typescript": "^5.0.0"
  }
}
```
