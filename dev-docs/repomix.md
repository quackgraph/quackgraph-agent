# Directory Structure
```
packages/
  quackgraph/
    packages/
      native/
        src/
          lib.rs
      quack-graph/
        src/
          graph.ts
    test/
      integration/
        concurrency.test.ts
        temporal.test.ts
```

# Files

## File: packages/quackgraph/packages/native/src/lib.rs
```rust
#![deny(clippy::all)]

use napi::bindgen_prelude::*;
use napi_derive::napi;
use quack_core::{matcher::{Matcher, PatternEdge}, GraphIndex, Direction, IntervalConstraint};
use arrow::ipc::reader::StreamReader;
use std::io::Cursor;

#[napi]
pub struct NativeGraph {
    inner: GraphIndex,
}

#[napi(object)]
pub struct JsPatternEdge {
    pub src_var: u32,
    pub tgt_var: u32,
    pub edge_type: String,
    pub direction: Option<String>,
}

#[napi(object)]
pub struct JsSectorStat {
    pub edge_type: String,
    pub count: u32,
    pub avg_heat: f64,
}

#[napi]
impl NativeGraph {
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            inner: GraphIndex::new(),
        }
    }

    #[napi]
    pub fn clear(&mut self) {
        self.inner = GraphIndex::new();
    }

    #[napi]
    pub fn add_node(&mut self, id: String) {
        self.inner.get_or_create_node(&id);
    }

    /// Hydrates the graph from an Arrow IPC stream (Buffer).
    /// Zero-copy (mostly) data transfer from DuckDB.
    /// Note: Does not verify duplicates. Caller must call compact() afterwards.
    #[napi]
    pub fn load_arrow_ipc(&mut self, buffer: Buffer) -> napi::Result<()> {
        let cursor = Cursor::new(buffer.as_ref());
        let reader = StreamReader::try_new(cursor, None).map_err(|e| napi::Error::from_reason(e.to_string()))?;

        for batch in reader {
            let batch = batch.map_err(|e| napi::Error::from_reason(e.to_string()))?;
            self.inner.add_arrow_batch(&batch).map_err(napi::Error::from_reason)?;
        }
        Ok(())
    }

    /// Compacts the graph's memory usage.
    /// Call this after hydration to reclaim unused capacity in the adjacency lists.
    /// Also deduplicates edges added via bulk ingestion.
    #[napi]
    pub fn compact(&mut self) {
        self.inner.compact();
    }

    #[napi(js_name = "addNodes")]
    pub fn add_nodes(&mut self, ids: Vec<String>) {
        for id in ids {
            self.inner.get_or_create_node(&id);
        }
    }

    #[napi(js_name = "addEdges")]
    pub fn add_edges(&mut self, 
        sources: Vec<String>, 
        targets: Vec<String>, 
        edge_types: Vec<String>, 
        valid_froms: Vec<f64>, 
        valid_tos: Vec<f64>,
        heats: Vec<u32>
    ) {
        let len = sources.len();
        for i in 0..len {
            let vf_in = valid_froms.get(i).copied().unwrap_or(0.0);
            let vf = (vf_in * 1000.0) as i64;

            let vt_in = valid_tos.get(i).copied().unwrap_or(f64::MAX);
            // If vt_in is very large (e.g. Max Safe Int from JS), treat as MAX_TIME
            let vt = if vt_in > 253402300799000.0 { i64::MAX } else { (vt_in * 1000.0) as i64 };

            let heat = heats.get(i).copied().unwrap_or(0) as u8;
            // Safe check for indices? Napi guarantees vectors passed from JS match arguments.
            // But we trust JS wrapper ensures lengths match.
            if let (Some(s), Some(t), Some(ty)) = (sources.get(i), targets.get(i), edge_types.get(i)) {
                self.inner.add_edge(s, t, ty, Some(vf), Some(vt), Some(heat));
            }
        }
    }

    #[napi]
    pub fn add_edge(&mut self, source: String, target: String, edge_type: String, valid_from: Option<f64>, valid_to: Option<f64>, heat: Option<u32>) {
        // JS timestamps are millis (f64). Convert to micros (i64) for DuckDB compatibility.
        let vf = valid_from.map(|t| (t * 1000.0) as i64);
        let vt = valid_to.map(|t| (t * 1000.0) as i64);
        let h = heat.map(|v| v as u8); // safe cast for V1
        self.inner.add_edge(&source, &target, &edge_type, vf, vt, h);
    }

    #[napi]
    pub fn remove_node(&mut self, id: String) {
        self.inner.remove_node(&id);
    }

    #[napi]
    pub fn remove_edge(&mut self, source: String, target: String, edge_type: String) {
        self.inner.remove_edge(&source, &target, &edge_type);
    }

    /// Updates the 'heat' (pheromone) level of an active edge.
    /// Used for reinforcement learning in the agent loop.
    #[napi(js_name = "updateEdgeHeat")]
    pub fn update_edge_heat(&mut self, source: String, target: String, edge_type: String, heat: u32) {
        self.inner.update_edge_heat(&source, &target, &edge_type, heat as u8);
    }

    /// Returns available edge types from the given source nodes.
    /// Used for "Ghost Earth" Satellite View (LOD 0).
    #[napi(js_name = "getAvailableEdgeTypes")]
    pub fn get_available_edge_types(&self, sources: Vec<String>, as_of: Option<f64>) -> Vec<String> {
        let ts = as_of.map(|t| (t * 1000.0) as i64);
        self.inner.get_available_edge_types(&sources, ts)
    }

    /// Returns aggregated statistics (count, heat) for outgoing edges from the given sources.
    /// More efficient than getAvailableEdgeTypes + traverse loop.
    #[napi(js_name = "getSectorStats")]
    pub fn get_sector_stats(&self, sources: Vec<String>, as_of: Option<f64>, allowed_edge_types: Option<Vec<String>>) -> Vec<JsSectorStat> {
        let ts = as_of.map(|t| (t * 1000.0) as i64);
        let allowed_ref = allowed_edge_types.as_deref();
        let raw_stats = self.inner.get_sector_stats(&sources, ts, allowed_ref);
        
        raw_stats.into_iter().map(|(edge_type, count, avg_heat)| JsSectorStat {
            edge_type,
            count,
            avg_heat
        }).collect()
    }

    /// Performs a single-hop traversal (bfs-step).
    /// Returns unique neighbor IDs.
    #[napi]
    pub fn traverse(&self, sources: Vec<String>, edge_type: Option<String>, direction: Option<String>, as_of: Option<f64>, min_valid_from: Option<f64>) -> Vec<String> {
        let dir = match direction.as_deref() {
            Some("in") | Some("IN") => Direction::Incoming,
            _ => Direction::Outgoing,
        };
        // JavaScript now passes microseconds directly
        let ts = as_of.map(|t| (t * 1000.0) as i64);
        let min_vf = min_valid_from.map(|t| (t * 1000.0) as i64);
        self.inner.traverse(&sources, edge_type.as_deref(), dir, ts, min_vf)
    }

    /// Performs a traversal finding all neighbors connected via edges that overlap 
    /// with the specified time window [start, end).
    /// Timestamps are in milliseconds (JS standard).
    /// Constraint: 'overlaps' (default), 'contains', 'during', 'meets'.
    #[napi]
    pub fn traverse_interval(&self, sources: Vec<String>, edge_type: Option<String>, direction: Option<String>, start: f64, end: f64, constraint: Option<String>) -> Vec<String> {
        let dir = match direction.as_deref() {
            Some("in") | Some("IN") => Direction::Incoming,
            _ => Direction::Outgoing,
        };
        let c = match constraint.as_deref() {
            Some("contains") | Some("CONTAINS") => IntervalConstraint::Contains,
            Some("during") | Some("DURING") => IntervalConstraint::During,
            Some("meets") | Some("MEETS") => IntervalConstraint::Meets,
            _ => IntervalConstraint::Overlaps,
        };
        let s = (start * 1000.0) as i64;
        let e = (end * 1000.0) as i64;
        self.inner.traverse_interval(&sources, edge_type.as_deref(), dir, s, e, c)
    }

    /// Performs a recursive traversal (BFS) with depth bounds.
    /// Returns unique node IDs reachable within [min_depth, max_depth].
    #[napi(js_name = "traverseRecursive")]
    #[allow(clippy::too_many_arguments)]
    pub fn traverse_recursive(&self, sources: Vec<String>, edge_type: Option<String>, direction: Option<String>, min_depth: Option<u32>, max_depth: Option<u32>, as_of: Option<f64>, monotonic: Option<bool>) -> Vec<String> {
        let dir = match direction.as_deref() {
            Some("in") | Some("IN") => Direction::Incoming,
            _ => Direction::Outgoing,
        };
        
        let min = min_depth.unwrap_or(1) as usize;
        let max = max_depth.unwrap_or(1) as usize;
        let ts = as_of.map(|t| (t * 1000.0) as i64);
        let mono = monotonic.unwrap_or(false);
        
        self.inner.traverse_recursive(&sources, edge_type.as_deref(), dir, min, max, ts, mono)
    }

    /// Finds subgraphs matching the given pattern.
    /// `start_ids` maps to variable 0 in the pattern.
    #[napi(js_name = "matchPattern")]
    pub fn match_pattern(&self, start_ids: Vec<String>, pattern: Vec<JsPatternEdge>, as_of: Option<f64>) -> Vec<Vec<String>> {
        let mut core_pattern = Vec::with_capacity(pattern.len());
        for p in pattern {
            if let Some(type_id) = self.inner.get_type_id(&p.edge_type) {
                core_pattern.push(PatternEdge {
                    src_var: p.src_var as usize,
                    tgt_var: p.tgt_var as usize,
                    type_id,
                    direction: match p.direction.as_deref() {
                        Some("in") | Some("IN") => Direction::Incoming,
                        _ => Direction::Outgoing,
                    },
                });
            } else {
                return Vec::new(); // Edge type doesn't exist, no matches possible.
            }
        }

        let start_candidates: Vec<u32> = start_ids.iter()
            .filter_map(|id| self.inner.lookup_id(id))
            .collect();

        if start_candidates.is_empty() {
            return Vec::new();
        }

        let ts = as_of.map(|t| (t * 1000.0) as i64);
        let matcher = Matcher::new(&self.inner, &core_pattern);
        let raw_results = matcher.find_matches(&start_candidates, ts);

        raw_results.into_iter().map(|row| {
            row.into_iter().filter_map(|uid| self.inner.lookup_str(uid).map(|s| s.to_string())).collect()
        }).collect()
    }

    /// Returns the number of nodes in the interned index.
    /// Useful for debugging hydration.
    #[napi(getter)]
    pub fn node_count(&self) -> u32 {
        // We cast to u32 because exposing usize to JS can be finicky depending on napi version,
        // though napi usually handles numbers well. Safe for V1.
        self.inner.node_count() as u32
    }

    #[napi(getter)]
    pub fn edge_count(&self) -> u32 {
        self.inner.edge_count() as u32
    }

    #[napi]
    pub fn save_snapshot(&self, path: String) -> napi::Result<()> {
        self.inner.save_to_file(&path).map_err(napi::Error::from_reason)
    }

    #[napi]
    pub fn load_snapshot(&mut self, path: String) -> napi::Result<()> {
        let loaded = GraphIndex::load_from_file(&path).map_err(napi::Error::from_reason)?;
        self.inner = loaded;
        Ok(())
    }
}

impl Default for NativeGraph {
    fn default() -> Self {
        Self::new()
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
        const vfDate = options.validFrom || new Date(0);

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
        // Normalize validFrom in JS so DB and RAM receive the exact same value
        const edgesWithTime = edges.map(e => ({
            ...e,
            validFrom: e.validFrom || new Date(0)
        }));

        await this.schema.writeEdgesBulk(edgesWithTime);
        const sources = edgesWithTime.map(e => e.source);
        const targets = edgesWithTime.map(e => e.target);
        const types = edgesWithTime.map(e => e.type);
        // Rust Napi expects parallel arrays
        const validFroms = edgesWithTime.map(e => e.validFrom!.getTime());
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

## File: packages/quackgraph/test/integration/concurrency.test.ts
```typescript
import { describe, test, expect, afterEach } from 'bun:test';
import { createGraph, cleanupGraph } from '../utils/helpers';

describe('Integration: Concurrency', () => {
  let path: string;

  afterEach(async () => {
    if (path) await cleanupGraph(path);
  });

  test('should handle concurrent node additions without data loss', async () => {
    const setup = await createGraph('disk', 'int-concurrency');
    const g = setup.graph;
    path = setup.path;

    const count = 100;
    const promises = [];

    // Fire 100 writes "simultaneously"
    for (let i = 0; i < count; i++) {
      promises.push(g.addNode(`node:${i}`, ['Node'], { index: i }));
    }

    await Promise.all(promises);

    expect(g.native.nodeCount).toBe(count);
    
    // Check DB persistence
    const rows = await g.db.query('SELECT count(*) as c FROM nodes WHERE valid_to IS NULL');
    const c = Number(rows[0].c); 
    expect(c).toBe(count);
  });

  test('should handle concurrent edge additions between same nodes', async () => {
    // Tests locking mechanism on adjacency list (if any) or vector resizing safety
    const setup = await createGraph('disk', 'int-concurrency-edges');
    const g = setup.graph;
    path = setup.path;

    await g.addNode('A', ['Node']);
    await g.addNode('B', ['Node']);

    const count = 50;
    const promises = [];

    // Add 50 edges "simultaneously" of DIFFERENT types to avoid idempotency masking the test
    for (let i = 0; i < count; i++) {
      promises.push(g.addEdge('A', 'B', `LINK_${i}`));
    }

    await Promise.all(promises);

    expect(g.native.edgeCount).toBe(count);

    // Verify traversal finds them all
    // Checking one specific link
    const neighbors = g.native.traverse(['A'], 'LINK_42', 'out');
    expect(neighbors).toEqual(['B']);
  });

  test('should deduplicate edges during bulk hydration (Append-Then-Sort)', async () => {
    // This tests the optimized Arrow ingestion strategy
    const setup = await createGraph('disk', 'int-bulk-dedup');
    const g = setup.graph;
    path = setup.path;

    // 1. Manually insert duplicates into DuckDB (bypassing graph API which might check)
    // QuackGraph schema: edges(source, target, type, ...)
    const sql = `
      INSERT INTO edges (source, target, type, valid_from, valid_to) VALUES 
      ('src', 'tgt', 'KNOWS', current_timestamp, NULL),
      ('src', 'tgt', 'KNOWS', current_timestamp, NULL), -- Duplicate
      ('src', 'tgt', 'KNOWS', current_timestamp, NULL)  -- Triplicate
    `;
    await g.db.execute(sql);

    // 2. Hydrate
    // Re-initialize to trigger hydration from disk
    // We use the same file path
    await g.native.loadArrowIpc(
      Buffer.from(await g.db.queryArrow("SELECT source, target, type FROM edges WHERE valid_to IS NULL"))
    );

    // 3. Compact (Triggers Sort & Dedup)
    g.native.compact();

    // 4. Verify
    // Should count as 1 edge in topology
    // Note: If we didn't compact, this might be 3 depending on implementation, but compact enforces uniqueness.
    expect(g.native.edgeCount).toBe(1);
    expect(g.native.traverse(['src'], 'KNOWS', 'out')).toEqual(['tgt']);
  });
});
```

## File: packages/quackgraph/test/integration/temporal.test.ts
```typescript
import { describe, test, expect, afterEach } from 'bun:test';
import { createGraph, cleanupGraph, sleep } from '../utils/helpers';
import { QuackGraph } from '../../packages/quack-graph/src/index';

describe('Integration: Temporal Time-Travel', () => {
  let g: QuackGraph;
  let path: string;

  afterEach(async () => {
    if (path) await cleanupGraph(path);
  });

  test('should retrieve historical property values using asOf', async () => {
    const setup = await createGraph('disk', 'temporal-props');
    g = setup.graph;
    path = setup.path;

    // T0: Create
    await g.addNode('u1', ['User'], { status: 'active' });
    const t0 = new Date();
    await sleep(100); // Ensure clock tick

    // T1: Update
    await g.addNode('u1', ['User'], { status: 'suspended' });
    const t1 = new Date();
    await sleep(100);

    // T2: Update again
    await g.addNode('u1', ['User'], { status: 'banned' });
    const _t2 = new Date();

    // Query Current (T2)
    const current = await g.match(['User']).where({}).select();
    expect(current[0].status).toBe('banned');

    // Query T0 (Should see 'active')
    // Note: strict equality might be tricky with microsecond precision,
    // so we pass a time slightly after T0 or exactly T0.
    // The query logic is: valid_from <= T AND (valid_to > T OR valid_to IS NULL)
    // At T0: valid_from=T0, valid_to=T1.
    // Query at T0: T0 <= T0 (True) AND T1 > T0 (True).
    const q0 = await g.asOf(t0).match(['User']).where({}).select();
    expect(q0[0].status).toBe('active');

    // Query T1 (Should see 'suspended')
    const q1 = await g.asOf(t1).match(['User']).where({}).select();
    expect(q1[0].status).toBe('suspended');
  });

  test('should handle node lifecycle (create -> delete)', async () => {
    const setup = await createGraph('disk', 'temporal-lifecycle');
    g = setup.graph;
    path = setup.path;

    // T0: Empty
    const t0 = new Date();
    await sleep(50);

    // T1: Alive
    await g.addNode('temp', ['Temp']);
    const t1 = new Date();
    await sleep(50);

    // T2: Deleted
    await g.deleteNode('temp');
    const t2 = new Date();

    // Verify
    const resT0 = await g.asOf(t0).match(['Temp']).select();
    expect(resT0.length).toBe(0);

    const resT1 = await g.asOf(t1).match(['Temp']).select();
    expect(resT1.length).toBe(1);
    expect(resT1[0].id).toBe('temp');

    const resT2 = await g.asOf(t2).match(['Temp']).select();
    expect(resT2.length).toBe(0);
  });

  test('should traverse historical topology (Structural Time-Travel)', async () => {
    // Scenario:
    // T0: A -> B
    // T1: Delete A -> B
    // T2: Create A -> C
    // Query at T0: Returns B
    // Query at T2: Returns C

    const setup = await createGraph('disk', 'temporal-topology');
    g = setup.graph;
    path = setup.path;

    await g.addNode('A', ['Node']);
    await g.addNode('B', ['Node']);
    await g.addNode('C', ['Node']);

    // T0: Create Edge
    await g.addEdge('A', 'B', 'LINK');
    await sleep(50);
    const t0 = new Date();
    await sleep(50);

    // T1: Delete Edge
    await g.deleteEdge('A', 'B', 'LINK');
    await sleep(50);

    // T2: Create New Edge
    await g.addEdge('A', 'C', 'LINK');
    await sleep(50);
    const t2 = new Date();

    // To test historical topology, we must re-hydrate from disk to ensure we have the
    // complete temporal edge data, as the live instance's memory might have been
    // modified by hard-deletes (removeEdge).
    const g2 = new QuackGraph(path);
    await g2.init();

    // Check T0 (Historical)
    const resT0 = await g2.asOf(t0).match(['Node']).where({ id: 'A' }).out('LINK').select(n => n.id);
    expect(resT0).toEqual(['B']);

    // Check T2 (Current)
    const resT2 = await g2.asOf(t2).match(['Node']).where({ id: 'A' }).out('LINK').select(n => n.id);
    expect(resT2).toEqual(['C']);
  });
});
```
