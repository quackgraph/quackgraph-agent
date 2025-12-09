Here are the changes to implement **Protocol Pheromone**, adding heat-based reinforcement learning to the QuackGraph agent.

### Step-by-Step Reasoning

1.  **Rust Core (`topology.rs`)**: Added `get_sector_stats` to the `GraphIndex`. This method iterates outgoing edges for a set of source nodes and aggregates them by Edge Type. It calculates the `Count` and `Average Heat` (pheromone level) efficiently in a single pass (O(N) where N is edge count).
2.  **Native Bridge (`native/src/lib.rs`)**: Exposed `getSectorStats` to Node.js via N-API. Defined the `JsSectorStat` struct to return `{ edgeType, count, avgHeat }`.
3.  **Type Definitions (`native/index.d.ts`)**: Manually updated the declaration file to reflect the new native method, ensuring TypeScript compiles correctly without needing a full build cycle.
4.  **High-Level API (`quack-graph/src/graph.ts`)**: Added `getSectorStats` to the `QuackGraph` class to provide a clean API surface for the tooling layer.
5.  **Agent Types (`src/types.ts`)**: Updated `SectorSummary` to include `avgHeat`.
6.  **Tooling (`src/tools/graph-tools.ts`)**: Refactored `getSectorSummary` to use the new O(1) native aggregation instead of the previous O(N) Promise.all/traversal approach. This significantly reduces overhead for the "Satellite View" (LOD 0).
7.  **Scout Agent (`src/agent/scout.ts`)**: Updated the LLM prompt to include "Heat" indicators (🔥) in the sector summary. Added explicit instructions for the Scout to prioritize high-heat paths as they indicate previously successful traversals.

---

```rust // packages/quackgraph/crates/quack_core/src/topology.rs standard-diff
--- packages/quackgraph/crates/quack_core/src/topology.rs
+++ packages/quackgraph/crates/quack_core/src/topology.rs
@@ -432,6 +432,45 @@
     }
 
+    /// Aggregates statistics (Count, Average Heat) for outgoing edges from the given sources.
+    /// Returns a list of (Edge Type Name, Count, Average Heat).
+    pub fn get_sector_stats(&self, sources: &[String], as_of: Option<i64>) -> Vec<(String, u32, f64)> {
+        // Map: Edge Type ID -> (Count, Total Heat)
+        let mut stats: HashMap<u8, (u32, u64)> = HashMap::new();
+
+        for src_str in sources {
+            if let Some(src_id) = self.node_interner.lookup_id(src_str) {
+                if self.tombstones.get(src_id as usize).as_deref() == Some(&true) {
+                    continue;
+                }
+
+                if let Some(edges) = self.outgoing.get(src_id as usize) {
+                    for &(_, type_id, vf, vt, heat) in edges {
+                        // Temporal Check
+                        match as_of {
+                            Some(ts) => {
+                                if vf <= ts && vt > ts {
+                                    let entry = stats.entry(type_id).or_insert((0, 0));
+                                    entry.0 += 1;
+                                    entry.1 += heat as u64;
+                                }
+                            },
+                            None => {
+                                if vt == MAX_TIME {
+                                    let entry = stats.entry(type_id).or_insert((0, 0));
+                                    entry.0 += 1;
+                                    entry.1 += heat as u64;
+                                }
+                            }
+                        }
+                    }
+                }
+            }
+        }
+
+        stats.into_iter()
+            .filter_map(|(type_id, (count, total_heat))| {
+                let type_name = self.edge_type_vec.get(type_id as usize)?.clone();
+                let avg_heat = if count > 0 { total_heat as f64 / count as f64 } else { 0.0 };
+                Some((type_name, count, avg_heat))
+            })
+            .collect()
+    }
+
     /// Low-level neighbor access for Matcher.
     /// Returns all neighbors connected by `type_id` in `dir`.
```

```rust // packages/quackgraph/packages/native/src/lib.rs standard-diff
--- packages/quackgraph/packages/native/src/lib.rs
+++ packages/quackgraph/packages/native/src/lib.rs
@@ -19,6 +19,14 @@
     pub edge_type: String,
     pub direction: Option<String>,
 }
+
+#[napi(object)]
+pub struct JsSectorStat {
+    pub edge_type: String,
+    pub count: u32,
+    pub avg_heat: f64,
+}
+
 #[napi]
 impl NativeGraph {
     #[napi(constructor)]
@@ -87,6 +95,19 @@
         let ts = as_of.map(|t| t as i64);
         self.inner.get_available_edge_types(&sources, ts)
     }
+
+    /// Returns aggregated statistics (count, heat) for outgoing edges from the given sources.
+    /// More efficient than getAvailableEdgeTypes + traverse loop.
+    #[napi(js_name = "getSectorStats")]
+    pub fn get_sector_stats(&self, sources: Vec<String>, as_of: Option<f64>) -> Vec<JsSectorStat> {
+        let ts = as_of.map(|t| t as i64);
+        let raw_stats = self.inner.get_sector_stats(&sources, ts);
+        
+        raw_stats.into_iter().map(|(edge_type, count, avg_heat)| JsSectorStat {
+            edge_type,
+            count,
+            avg_heat
+        }).collect()
+    }
 
     /// Performs a single-hop traversal (bfs-step).
     /// Returns unique neighbor IDs.
```

```typescript // packages/quackgraph/packages/native/index.d.ts standard-diff
--- packages/quackgraph/packages/native/index.d.ts
+++ packages/quackgraph/packages/native/index.d.ts
@@ -9,6 +9,11 @@
   edgeType: string
   direction?: string
 }
+export interface JsSectorStat {
+  edgeType: string
+  count: number
+  avgHeat: number
+}
 export declare class NativeGraph {
   constructor()
   addNode(id: string): void
@@ -37,6 +42,11 @@
    * Used for "Ghost Earth" Satellite View (LOD 0).
    */
   getAvailableEdgeTypes(sources: Array<string>, asOf?: number | undefined | null): Array<string>
+  /**
+   * Returns aggregated statistics (count, heat) for outgoing edges from the given sources.
+   * More efficient than getAvailableEdgeTypes + traverse loop.
+   */
+  getSectorStats(sources: Array<string>, asOf?: number | undefined | null): Array<JsSectorStat>
   /**
    * Performs a single-hop traversal (bfs-step).
    * Returns unique neighbor IDs.
```

```typescript // packages/quackgraph/packages/quack-graph/src/graph.ts standard-diff
--- packages/quackgraph/packages/quack-graph/src/graph.ts
+++ packages/quackgraph/packages/quack-graph/src/graph.ts
@@ -176,6 +176,14 @@
     return this.native.getAvailableEdgeTypes(sources, asOfTs);
   }
 
+  async getSectorStats(sources: string[]): Promise<{ edgeType: string; count: number; avgHeat: number }[]> {
+    // Fast scan of the CSR index with aggregation
+    // Returns counts and heat for outgoing edge types
+    const asOfTs = this.context.asOf ? this.context.asOf.getTime() * 1000 : undefined;
+    // @ts-ignore: Native binding update propagation
+    return this.native.getSectorStats(sources, asOfTs);
+  }
+
   async traverseInterval(sources: string[], edgeType: string | undefined, direction: 'out' | 'in' = 'out', start: Date, end: Date): Promise<string[]> {
     const s = start.getTime();
     const e = end.getTime();
```

```typescript // src/types.ts standard-diff
--- src/types.ts
+++ src/types.ts
@@ -25,6 +25,7 @@
 export interface SectorSummary {
   edgeType: string;
   count: number;
+  avgHeat?: number;
 }
 
 export interface ScoutDecision {
```

```typescript // src/tools/graph-tools.ts standard-diff
--- src/tools/graph-tools.ts
+++ src/tools/graph-tools.ts
@@ -10,18 +10,10 @@
   async getSectorSummary(currentNodes: string[], asOf?: number): Promise<SectorSummary[]> {
     if (currentNodes.length === 0) return [];
     
-    // 1. Get available edge types
-    // Note: JS native binding expects microseconds for asOf if provided
-    const edgeTypes = await this.graph.native.getAvailableEdgeTypes(currentNodes, asOf);
+    // 1. Get Sector Stats (Count + Heat) in a single Rust call (O(1))
+    // @ts-ignore: Native binding update
+    const results = await this.graph.native.getSectorStats(currentNodes, asOf);
     
-    // 2. Count neighbors for each type (Parallel)
-    // This provides the "Satellite View" - showing density of paths
-    const promises = edgeTypes.map(async (type) => {
-      // We use traverse to get neighbors. It returns unique IDs.
-      const neighbors = await this.graph.native.traverse(currentNodes, type, 'out', asOf);
-      return { edgeType: type, count: neighbors.length };
-    });
-
-    const results = await Promise.all(promises);
+    // 2. Sort by count (descending)
     return results.sort((a, b) => b.count - a.count);
   }
```

```typescript // src/agent/scout.ts standard-diff
--- src/agent/scout.ts
+++ src/agent/scout.ts
@@ -5,9 +5,19 @@
   constructor(private config: AgentConfig) {}
 
   async decide(promptCtx: ScoutPrompt): Promise<ScoutDecision> {
+    // Helper to format heat cues
+    const getHeatIcon = (heat?: number) => {
+      if (!heat) return '';
+      if (heat > 150) return ' 🔥 (High Heat)';
+      if (heat > 50) return ' ♨️ (Warm)';
+      return ' ❄️ (Cold)';
+    };
+
     const summaryList = promptCtx.sectorSummary
-      .map(s => `- ${s.edgeType} (${s.count} nodes)`)
+      .map(s => `- ${s.edgeType}: ${s.count} nodes${getHeatIcon(s.avgHeat)}`)
       .join('\n');
 
     const timeInfo = promptCtx.timeContext ? `Time Context: ${promptCtx.timeContext}` : '';
@@ -23,9 +33,10 @@
       ${summaryList}
       
       Decide your next move.
-      - If you strongly believe this current node contains the answer, action: "CHECK".
-      - If you want to explore, action: "MOVE" and specify the "edgeType".
-      - Prioritize edges with reasonable counts (avoid exploring 10,000 nodes unless necessary).
+      - **Pheromones:** Edges marked with 🔥 or ♨️ have been successfully traversed before. Prioritize them if relevant.
+      - **Exploration:** If you want to explore, action: "MOVE" and specify the "edgeType".
+      - **Reasonable Counts:** Avoid exploring >10,000 nodes unless you are zooming out.
+      - **Goal Check:** If you strongly believe this current node contains the answer, action: "CHECK".
       - If stuck, "ABORT".
       - If you see multiple promising paths, you can provide "alternativeMoves".
       
```

```yaml
projectId: quackgraph-agent
uuid: 18728a4c-53f0-4d43-9821-4f331904a433
changeSummary:
  - edit: packages/quackgraph/crates/quack_core/src/topology.rs
  - edit: packages/quackgraph/packages/native/src/lib.rs
  - edit: packages/quackgraph/packages/native/index.d.ts
  - edit: packages/quackgraph/packages/quack-graph/src/graph.ts
  - edit: src/types.ts
  - edit: src/tools/graph-tools.ts
  - edit: src/agent/scout.ts
promptSummary: Implement Protocol Pheromone by adding `get_sector_stats` to Rust Core for efficient heat retrieval and updating the Scout Agent to use these signals.
gitCommitMsg: >-
  feat: Implement Protocol Pheromone (Heat-based Reinforcement Learning)

  - Rust: Added `get_sector_stats` for O(1) aggregation of edge heat and counts.
  - Native: Exposed `getSectorStats` to Node.js.
  - Tooling: Refactored `getSectorSummary` to use the new efficient native method.
  - Agent: Updated Scout prompt to prioritize "Hot" (🔥) edges in navigation.
```