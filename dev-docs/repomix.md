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
  quackgraph/
    crates/
      quack_core/
        src/
          interner.rs
          lib.rs
          matcher.rs
          topology.rs
        Cargo.toml
    packages/
      native/
        src/
          lib.rs
        build.rs
        Cargo.toml
        index.d.ts
        index.js
        package.json
      quack-graph/
        src/
          db.ts
          graph.ts
          index.ts
          query.ts
          schema.ts
        package.json
    biome.json
    Cargo.toml
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

## File: packages/quackgraph/crates/quack_core/src/interner.rs
````rust
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// A bidirectional map between String IDs and u32 internal indices.
/// Used to convert DuckDB UUIDs/Strings into efficient integers for the graph topology.
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct Interner {
    map: HashMap<String, u32>,
    vec: Vec<String>,
}

impl Interner {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
            vec: Vec::new(),
        }
    }

    /// Interns a string: returns existing ID if present, or creates a new one.
    /// O(1) average case.
    pub fn intern(&mut self, name: &str) -> u32 {
        if let Some(&id) = self.map.get(name) {
            return id;
        }
        let id = self.vec.len() as u32;
        let key = name.to_string();
        self.vec.push(key.clone());
        self.map.insert(key, id);
        id
    }

    /// Reverse lookup: u32 -> String.
    /// O(1) worst case.
    pub fn lookup(&self, id: u32) -> Option<&str> {
        self.vec.get(id as usize).map(|s| s.as_str())
    }

    /// Looks up the u32 ID for a given string name.
    /// O(1) average case.
    pub fn lookup_id(&self, name: &str) -> Option<u32> {
        self.map.get(name).copied()
    }

    /// Current number of interned items.
    pub fn len(&self) -> usize {
        self.vec.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vec.is_empty()
    }
}
````

## File: packages/quackgraph/crates/quack_core/src/lib.rs
````rust
pub mod interner;
pub mod topology;
pub mod matcher;

pub use interner::Interner;
pub use topology::{GraphIndex, Direction, IntervalConstraint};
````

## File: packages/quackgraph/crates/quack_core/src/matcher.rs
````rust
use crate::topology::{GraphIndex, Direction};
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct PatternEdge {
    pub src_var: usize,
    pub tgt_var: usize,
    pub type_id: u8,
    pub direction: Direction,
}

/// A simple backtracking solver for subgraph isomorphism.
/// Finds all assignments of graph nodes to pattern variables such that all pattern edges exist.
///
/// Assumptions:
/// 1. Variable 0 is the "start" variable, seeded by `start_candidates`.
/// 2. The pattern is connected: for any variable `i > 0`, there is at least one constraint
///    connecting it to a variable `j < i`.
pub struct Matcher<'a> {
    graph: &'a GraphIndex,
    pattern: &'a [PatternEdge],
    num_vars: usize,
}

impl<'a> Matcher<'a> {
    pub fn new(graph: &'a GraphIndex, pattern: &'a [PatternEdge]) -> Self {
        let mut max_var = 0;
        for e in pattern {
            max_var = max_var.max(e.src_var).max(e.tgt_var);
        }
        Self {
            graph,
            pattern,
            num_vars: max_var + 1,
        }
    }

    pub fn find_matches(&self, start_candidates: &[u32], as_of: Option<i64>) -> Vec<Vec<u32>> {
        let mut results = Vec::new();
        let mut assignment = vec![None; self.num_vars];
        let mut used_nodes = HashSet::new();

        for &start_node in start_candidates {
            if self.graph.is_node_deleted(start_node) {
                continue;
            }

            assignment[0] = Some(start_node);
            used_nodes.insert(start_node);
            
            self.backtrack(1, &mut assignment, &mut used_nodes, &mut results, as_of);
            
            used_nodes.remove(&start_node);
            assignment[0] = None;
        }

        results
    }

    fn backtrack(
        &self,
        current_var: usize,
        assignment: &mut Vec<Option<u32>>,
        used_nodes: &mut HashSet<u32>,
        results: &mut Vec<Vec<u32>>,
        as_of: Option<i64>,
    ) {
        if current_var == self.num_vars {
            results.push(assignment.iter().map(|opt| opt.unwrap()).collect());
            return;
        }

        let mut candidates: Option<Vec<u32>> = None;

        for edge in self.pattern {
            if edge.src_var < current_var && edge.tgt_var == current_var {
                let known_node = assignment[edge.src_var].unwrap();
                let neighbors = self.graph.get_neighbors(known_node, edge.type_id, Direction::Outgoing, as_of);
                candidates = self.intersect(candidates, neighbors);
                if candidates.as_ref().is_some_and(|c| c.is_empty()) { return; }
            }
            else if edge.src_var == current_var && edge.tgt_var < current_var {
                let known_node = assignment[edge.tgt_var].unwrap();
                let neighbors = self.graph.get_neighbors(known_node, edge.type_id, Direction::Incoming, as_of);
                candidates = self.intersect(candidates, neighbors);
                if candidates.as_ref().is_some_and(|c| c.is_empty()) { return; }
            }
        }
        
        if let Some(cands) = candidates {
            for cand in cands {
                if !used_nodes.contains(&cand) {
                    assignment[current_var] = Some(cand);
                    used_nodes.insert(cand);
                    
                    self.backtrack(current_var + 1, assignment, used_nodes, results, as_of);
                    
                    used_nodes.remove(&cand);
                    assignment[current_var] = None;
                }
            }
        }
    }

    fn intersect(&self, current: Option<Vec<u32>>, next: Vec<u32>) -> Option<Vec<u32>> {
        match current {
            None => Some(next),
            Some(curr) => {
                let set: HashSet<_> = next.into_iter().collect();
                Some(curr.into_iter().filter(|id| set.contains(id)).collect())
            }
        }
    }
}
````

## File: packages/quackgraph/crates/quack_core/src/topology.rs
````rust
use crate::interner::Interner;
use bitvec::prelude::*;
use std::collections::{HashMap, VecDeque};
use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use arrow::record_batch::RecordBatch;
use arrow::array::{AsArray, Array, StringArray, LargeStringArray, UInt8Array};
use arrow::datatypes::DataType;
use arrow::compute::cast;

pub const MAX_TIME: i64 = i64::MAX;

/// (Target Node ID, Edge Type ID, Valid From, Valid To, Heat)
type EdgeTuple = (u32, u8, i64, i64, u8);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntervalConstraint {
    Overlaps,
    Contains,
    During,
    Meets,
}

/// The core Graph Index.
/// Stores topology in RAM using integer IDs.
#[derive(Default, Debug, Serialize, Deserialize)]
pub struct GraphIndex {
    node_interner: Interner,
    
    // Mapping edge type strings (e.g. "KNOWS") to u8 for compact storage.
    // Limit: 256 edge types per graph in V1.
    edge_type_map: HashMap<String, u8>,
    edge_type_vec: Vec<String>,

    // Forward Graph: Source Node ID -> List of (Target Node ID, Edge Type ID, Valid From, Valid To, Heat)
    outgoing: Vec<Vec<EdgeTuple>>,
    
    // Reverse Graph: Target Node ID -> List of (Source Node ID, Edge Type ID, Valid From, Valid To, Heat)
    incoming: Vec<Vec<EdgeTuple>>,

    // Bitmask for soft-deleted nodes.
    // true = deleted (tombstone), false = active.
    tombstones: BitVec,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Direction {
    Outgoing,
    Incoming,
}

impl GraphIndex {
    pub fn new() -> Self {
        Self {
            node_interner: Interner::new(),
            edge_type_map: HashMap::new(),
            edge_type_vec: Vec::new(),
            outgoing: Vec::new(),
            incoming: Vec::new(),
            tombstones: BitVec::new(),
        }
    }

    pub fn lookup_id(&self, id: &str) -> Option<u32> {
        self.node_interner.lookup_id(id)
    }

    pub fn lookup_str(&self, id: u32) -> Option<&str> {
        self.node_interner.lookup(id)
    }

    /// Compacts internal vectors to minimize memory usage.
    /// Also sorts and deduplicates adjacency lists (essential after bulk loading).
    /// Should be called after bulk hydration.
    pub fn compact(&mut self) {
        self.outgoing.iter_mut().for_each(|v| {
            v.sort_unstable();
            v.dedup();
            v.shrink_to_fit();
        });
        self.incoming.iter_mut().for_each(|v| {
            v.sort_unstable();
            v.dedup();
            v.shrink_to_fit();
        });
        self.outgoing.shrink_to_fit();
        self.incoming.shrink_to_fit();
        self.edge_type_vec.shrink_to_fit();
    }

    /// Resolves or creates an internal u32 ID for a node string.
    /// Resizes internal storage if necessary.
    pub fn get_or_create_node(&mut self, id: &str) -> u32 {
        let internal_id = self.node_interner.intern(id);
        let idx = internal_id as usize;

        // Ensure vectors are large enough to hold this node
        if idx >= self.outgoing.len() {
            let new_len = idx + 1;
            self.outgoing.resize_with(new_len, Vec::new);
            self.incoming.resize_with(new_len, Vec::new);
            // Resize tombstones, filling new slots with false (active)
            self.tombstones.resize(new_len, false);
        }
        internal_id
    }

    /// Marks a node as deleted (soft delete).
    /// Traversals will skip this node.
    pub fn remove_node(&mut self, id: &str) {
        if let Some(u_id) = self.node_interner.lookup_id(id) {
            let idx = u_id as usize;
            if idx < self.tombstones.len() {
                self.tombstones.set(idx, true);
            }
        }
    }

    pub fn is_node_deleted(&self, id: u32) -> bool {
        self.tombstones.get(id as usize).as_deref() == Some(&true)
    }

    /// Returns the total number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.outgoing.iter().map(|edges| edges.len()).sum()
    }

    /// Resolves or creates a u8 ID for an edge type string.
    /// Panics if more than 255 edge types are used (V1 constraint).
    pub fn get_or_create_type(&mut self, type_name: &str) -> u8 {
        if let Some(&id) = self.edge_type_map.get(type_name) {
            return id;
        }
        let id = self.edge_type_vec.len();
        if id > 255 {
            panic!("QuackGraph V1 Limit: Max 256 unique edge types supported.");
        }
        let id_u8 = id as u8;
        self.edge_type_vec.push(type_name.to_string());
        self.edge_type_map.insert(type_name.to_string(), id_u8);
        id_u8
    }

    pub fn get_type_id(&self, type_name: &str) -> Option<u8> {
        self.edge_type_map.get(type_name).copied()
    }

    /// Adds an edge to the graph. 
    /// Idempotent: Does not add duplicate edges if they already exist.
    /// If timestamps are not provided, defaults to (0, MAX_TIME).
    pub fn add_edge(&mut self, source: &str, target: &str, edge_type: &str, valid_from: Option<i64>, valid_to: Option<i64>, heat: Option<u8>) {
        let vf = valid_from.unwrap_or(0);
        let vt = valid_to.unwrap_or(MAX_TIME);
        let h = heat.unwrap_or(0);
        
        let u_src = self.get_or_create_node(source);
        let u_tgt = self.get_or_create_node(target);
        let u_type = self.get_or_create_type(edge_type);

        // Add to forward index (Idempotent)
        let out_vec = &mut self.outgoing[u_src as usize];
        // We include heat in the uniqueness check, effectively allowing different heat levels to coexist if pushed explicitly
        if !out_vec.contains(&(u_tgt, u_type, vf, vt, h)) {
            out_vec.push((u_tgt, u_type, vf, vt, h));
        }
        
        // Add to reverse index (Idempotent)
        let in_vec = &mut self.incoming[u_tgt as usize];
        if !in_vec.contains(&(u_src, u_type, vf, vt, h)) {
            in_vec.push((u_src, u_type, vf, vt, h));
        }

        // Ensure nodes are not tombstoned if they are being re-added/linked
        if self.tombstones.get(u_src as usize).as_deref() == Some(&true) {
            self.tombstones.set(u_src as usize, false);
        }
        if self.tombstones.get(u_tgt as usize).as_deref() == Some(&true) {
            self.tombstones.set(u_tgt as usize, false);
        }
    }

    /// Removes a specific edge from the graph.
    /// Uses swap_remove for O(1) removal, order is not preserved.
    pub fn remove_edge(&mut self, source: &str, target: &str, edge_type: &str) {
        // We only proceed if all entities exist in our interner/maps
        if let (Some(u_src), Some(u_tgt), Some(u_type)) = (
            self.node_interner.lookup_id(source),
            self.node_interner.lookup_id(target),
            self.edge_type_map.get(edge_type).copied(),
        ) {
            // Note: In V2 Temporal, removing an edge usually means "closing" the validity window.
            // However, this method removes it from RAM entirely (hard delete).
            // Remove from outgoing
            if let Some(edges) = self.outgoing.get_mut(u_src as usize) {
                if let Some(pos) = edges.iter().position(|x| x.0 == u_tgt && x.1 == u_type) { // Ignores validity/heat for deletion
                    edges.swap_remove(pos);
                }
            }
            // Remove from incoming
            if let Some(edges) = self.incoming.get_mut(u_tgt as usize) {
                if let Some(pos) = edges.iter().position(|x| x.0 == u_src && x.1 == u_type) { // Ignores validity/heat for deletion
                    edges.swap_remove(pos);
                }
            }
        }
    }

    /// Ingests an Apache Arrow RecordBatch directly.
    /// Expected Schema: Columns named "source", "target", "type" (case-insensitive or exact).
    pub fn add_arrow_batch(&mut self, batch: &RecordBatch) -> Result<(), String> {
        let schema = batch.schema();
        
        // Resolve column indices by name for robustness (Case-Insensitive)
        let find_col = |name: &str| -> Result<usize, String> {
            schema.fields().iter().position(|f| f.name().eq_ignore_ascii_case(name))
                .ok_or_else(|| format!("Column '{}' not found in Arrow Batch. Available: {:?}", name, schema.fields().iter().map(|f| f.name()).collect::<Vec<_>>()))
        };
        
        let num_rows = batch.num_rows();
        if num_rows == 0 {
            return Ok(());
        }

        // Helper to ensure we have a String/LargeString array, casting Dictionary if needed
        let prepare_col = |col: &std::sync::Arc<dyn Array>, name: &str| -> Result<std::sync::Arc<dyn Array>, String> {
            match col.data_type() {
                DataType::Utf8 | DataType::LargeUtf8 => Ok(col.clone()),
                DataType::Dictionary(_key_type, value_type) => {
                    match value_type.as_ref() {
                        // If we need to support dictionary encoded strings
                        DataType::Utf8 | DataType::LargeUtf8 => {
                            cast(col.as_ref(), value_type.as_ref())
                                .map_err(|e| format!("Cast error for {} column: {}", name, e))
                        },
                        other => {
                            Err(format!("{} column: Dictionary value type {:?} not supported (expected Utf8/LargeUtf8)", name, other))
                        }
                    }
                },
                dt => Err(format!("{} column: Unsupported type {:?}", name, dt)),
            }
        };

        let src_col = prepare_col(batch.column(find_col("source")?), "Source")?;
        let tgt_col = prepare_col(batch.column(find_col("target")?), "Target")?;
        let type_col = prepare_col(batch.column(find_col("type")?), "Type")?;

        // Optional Temporal Columns
        // If missing, we default to (0, MAX_TIME)
        let vf_idx = find_col("valid_from").ok();
        let vt_idx = find_col("valid_to").ok();
        let heat_idx = find_col("heat").ok();

        let vf_col = if let Some(idx) = vf_idx {
            Some(cast(batch.column(idx).as_ref(), &DataType::Int64).map_err(|e| e.to_string())?)
        } else { None };
        let vt_col = if let Some(idx) = vt_idx {
            Some(cast(batch.column(idx).as_ref(), &DataType::Int64).map_err(|e| e.to_string())?)
        } else { None };

        // Heat column (expecting UInt8)
        let heat_col = if let Some(idx) = heat_idx {
             // We try to cast to UInt8 just in case, though usually unnecessary if schema is correct
             Some(cast(batch.column(idx).as_ref(), &DataType::UInt8).map_err(|e| e.to_string())?)
        } else { None };

        // Wrapper to handle different string array types (Utf8 vs LargeUtf8)
        enum StringArrayWrapper<'a> {
            Small(&'a StringArray),
            Large(&'a LargeStringArray),
        }

        impl<'a> StringArrayWrapper<'a> {
            fn value(&self, i: usize) -> &'a str {
                match self {
                    Self::Small(arr) => arr.value(i),
                    Self::Large(arr) => arr.value(i),
                }
            }
        }

        macro_rules! get_wrapper {
            ($col:expr) => {
                match $col.data_type() {
                    DataType::Utf8 => StringArrayWrapper::Small($col.as_string::<i32>()),
                    DataType::LargeUtf8 => StringArrayWrapper::Large($col.as_string::<i64>()),
                    _ => unreachable!("Already validated/casted to Utf8/LargeUtf8"),
                }
            }
        }

        let src_wrapper = get_wrapper!(src_col);
        let tgt_wrapper = get_wrapper!(tgt_col);
        let type_wrapper = get_wrapper!(type_col);

        for i in 0..num_rows {
            let src = src_wrapper.value(i);
            let tgt = tgt_wrapper.value(i);
            let edge_type = type_wrapper.value(i);

            let u_src = self.get_or_create_node(src);
            let u_tgt = self.get_or_create_node(tgt);
            let u_type = self.get_or_create_type(edge_type);

            // Extract timestamps
            let valid_from = if let Some(ref col) = vf_col {
                col.as_any().downcast_ref::<arrow::array::Int64Array>().unwrap().value(i)
            } else { 0 };

            let valid_to = if let Some(ref col) = vt_col {
                let arr = col.as_any().downcast_ref::<arrow::array::Int64Array>().unwrap();
                if arr.is_null(i) {
                    MAX_TIME
                } else {
                    arr.value(i)
                }
            } else { MAX_TIME };

            // Extract Heat
            let heat = if let Some(ref col) = heat_col {
                col.as_any().downcast_ref::<UInt8Array>().unwrap().value(i)
            } else { 0 };

            // Fast Path: Blind push. We rely on compact() to deduplicate later.
            self.outgoing[u_src as usize].push((u_tgt, u_type, valid_from, valid_to, heat));
            self.incoming[u_tgt as usize].push((u_src, u_type, valid_from, valid_to, heat));

            // Ensure nodes are not tombstoned (revival logic)
            if self.tombstones.get(u_src as usize).as_deref() == Some(&true) {
                self.tombstones.set(u_src as usize, false);
            }
            if self.tombstones.get(u_tgt as usize).as_deref() == Some(&true) {
                self.tombstones.set(u_tgt as usize, false);
            }
        }
        Ok(())
    }

    /// Updates the 'heat' (pheromone level) of active edges matching the criteria.
    /// Only affects edges where valid_to is MAX_TIME (active).
    pub fn update_edge_heat(&mut self, source: &str, target: &str, edge_type: &str, new_heat: u8) {
        if let (Some(u_src), Some(u_tgt), Some(u_type)) = (
            self.node_interner.lookup_id(source),
            self.node_interner.lookup_id(target),
            self.edge_type_map.get(edge_type).copied(),
        ) {
            // Update outgoing
            if let Some(edges) = self.outgoing.get_mut(u_src as usize) {
                for edge in edges.iter_mut() {
                    // edge is (target, type, vf, vt, heat)
                    if edge.0 == u_tgt && edge.1 == u_type && edge.3 == MAX_TIME {
                        edge.4 = new_heat;
                    }
                }
            }
            // Update incoming
            if let Some(edges) = self.incoming.get_mut(u_tgt as usize) {
                for edge in edges.iter_mut() {
                    // edge is (source, type, vf, vt, heat)
                    if edge.0 == u_src && edge.1 == u_type && edge.3 == MAX_TIME {
                        edge.4 = new_heat;
                    }
                }
            }
        }
    }

    /// Contextual Schema Pruning.
    /// Returns a list of unique Edge Types (strings) that originate from the given source nodes.
    /// This allows an Agent to know "What moves are available?" without fetching all neighbors.
    pub fn get_available_edge_types(&self, sources: &[String], as_of: Option<i64>) -> Vec<String> {
        let mut type_ids = Vec::new();

        for src_str in sources {
            if let Some(src_id) = self.node_interner.lookup_id(src_str) {
                if self.tombstones.get(src_id as usize).as_deref() == Some(&true) {
                    continue;
                }
                if let Some(edges) = self.outgoing.get(src_id as usize) {
                    for &(_, type_id, vf, vt, _) in edges {
                        // Temporal Check
                        match as_of {
                            Some(ts) => {
                                // Edge is valid if it existed at 'ts' (valid_from <= ts < valid_to)
                                if vf <= ts && vt > ts {
                                    type_ids.push(type_id);
                                }
                            },
                            None => {
                                // Current/Active only
                                if vt == MAX_TIME {
                                    type_ids.push(type_id);
                                }
                            }
                        }
                    }
                }
            }
        }

        type_ids.sort_unstable();
        type_ids.dedup();

        type_ids.into_iter()
            .filter_map(|tid| {
                // We can't easily do O(1) reverse lookup on the map unless we scan or store a reverse vec.
                // Luckily `edge_type_vec` exists!
                self.edge_type_vec.get(tid as usize).cloned()
            })
            .collect()
    }

    /// Aggregates statistics (Count, Average Heat) for outgoing edges from the given sources.
    /// Returns a list of (Edge Type Name, Count, Average Heat).
    pub fn get_sector_stats(&self, sources: &[String], as_of: Option<i64>, allowed_types: Option<&[String]>) -> Vec<(String, u32, f64)> {
        // Map: Edge Type ID -> (Count, Total Heat)
        let mut stats: HashMap<u8, (u32, u64)> = HashMap::new();

        // Resolve allowed types to u8 set for fast lookup
        let allowed_ids: Option<Vec<u8>> = allowed_types.map(|types| {
            types.iter().filter_map(|t| self.edge_type_map.get(t).copied()).collect()
        });

        for src_str in sources {
            if let Some(src_id) = self.node_interner.lookup_id(src_str) {
                if self.tombstones.get(src_id as usize).as_deref() == Some(&true) {
                    continue;
                }

                if let Some(edges) = self.outgoing.get(src_id as usize) {
                    for &(_, type_id, vf, vt, heat) in edges {
                        // Domain Filtering (Push Down)
                        if let Some(ref allowed) = allowed_ids {
                            if !allowed.contains(&type_id) {
                                continue;
                            }
                        }

                        // Temporal Check
                        match as_of {
                            Some(ts) => {
                                if vf <= ts && vt > ts {
                                    let entry = stats.entry(type_id).or_insert((0, 0));
                                    entry.0 += 1;
                                    entry.1 += heat as u64;
                                }
                            },
                            None => {
                                if vt == MAX_TIME {
                                    let entry = stats.entry(type_id).or_insert((0, 0));
                                    entry.0 += 1;
                                    entry.1 += heat as u64;
                                }
                            }
                        }
                    }
                }
            }
        }

        stats.into_iter()
            .filter_map(|(type_id, (count, total_heat))| {
                let type_name = self.edge_type_vec.get(type_id as usize)?.clone();
                let avg_heat = if count > 0 { total_heat as f64 / count as f64 } else { 0.0 };
                Some((type_name, count, avg_heat))
            })
            .collect()
    }

    /// Low-level neighbor access for Matcher.
    /// Returns all neighbors connected by `type_id` in `dir`.
    /// Returns all neighbors connected by `type_id` in `dir`, respecting `as_of`.
    /// Filters out tombstoned neighbors.
    pub fn get_neighbors(&self, node_id: u32, type_id: u8, dir: Direction, as_of: Option<i64>) -> Vec<u32> {
        let adjacency = match dir {
            Direction::Outgoing => &self.outgoing,
            Direction::Incoming => &self.incoming,
        };

        if let Some(edges) = adjacency.get(node_id as usize) {
            edges.iter()
                .filter_map(|&(target, t, vf, vt, _heat)| {
                    // Type match
                    if t != type_id { return None; }
                    // Tombstone check
                    if self.is_node_deleted(target) { return None; }
                    
                    // Temporal Check
                    match as_of {
                        Some(ts) => {
                            if vf <= ts && vt > ts { Some(target) } else { None }
                        },
                        None => {
                            // Current/Active only
                            if vt == MAX_TIME { Some(target) } else { None }
                        }
                    }
                })
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Generic traversal step (Bidirectional).
    /// Given a list of source node IDs (strings), find all neighbors connected by `edge_type`
    /// in the specified `direction`, visible at `as_of`.
    pub fn traverse(&self, sources: &[String], edge_type: Option<&str>, direction: Direction, as_of: Option<i64>, min_valid_from: Option<i64>) -> Vec<String> {
        let type_filter = edge_type.and_then(|t| self.edge_type_map.get(t).copied());
        let min_vf = min_valid_from.unwrap_or(i64::MIN);
        
        let mut result_ids: Vec<u32> = Vec::with_capacity(sources.len() * 2);
        
        let adjacency = match direction {
            Direction::Outgoing => &self.outgoing,
            Direction::Incoming => &self.incoming,
        };

        for src_str in sources {
            // If source node doesn't exist in our index, skip it
            if let Some(src_id) = self.node_interner.lookup_id(src_str) {
                // Check if node is deleted
                if self.tombstones.get(src_id as usize).as_deref() == Some(&true) {
                    continue;
                }

                if let Some(edges) = adjacency.get(src_id as usize) {
                    for &(target, type_id, vf, vt, _heat) in edges {
                        // Apply edge type filter if present
                        if let Some(req_type) = type_filter {
                            if req_type != type_id {
                                continue;
                            }
                        }

                        // Monotonic/Causal Check
                        if vf < min_vf { continue; }

                        // Temporal Check
                        match as_of {
                            Some(ts) => { if !(vf <= ts && vt > ts) { continue; } },
                            None => { if vt != MAX_TIME { continue; } }
                        }
                        // Check if target is deleted
                        if self.tombstones.get(target as usize).as_deref() == Some(&true) {
                            continue;
                        }
                        result_ids.push(target);
                    }
                }
            }
        }

        // Deduplicate results
        result_ids.sort_unstable();
        result_ids.dedup();

        // Convert back to strings
        result_ids
            .into_iter()
            .filter_map(|id| self.node_interner.lookup(id).map(|s| s.to_string()))
            .collect()
    }

    /// Recursive traversal (BFS) with depth bounds.
    /// Returns unique node IDs reachable within [min_depth, max_depth].
    #[allow(clippy::too_many_arguments)]
    pub fn traverse_recursive(
        &self,
        sources: &[String],
        edge_type: Option<&str>,
        direction: Direction,
        min_depth: usize,
        max_depth: usize,
        as_of: Option<i64>,
        monotonic: bool,
    ) -> Vec<String> {
        let type_filter = edge_type.and_then(|t| self.edge_type_map.get(t).copied());
        
        // Track visited nodes to prevent cycles (O(1) access)
        // We assume the interner length is the upper bound of IDs
        let mut visited = bitvec![u8, Lsb0; 0; self.node_interner.len()];
        let mut result_ids: Vec<u32> = Vec::new();
        
        // Queue stores (node_id, current_depth, arrival_time)
        // arrival_time is i64::MIN for start nodes
        let mut queue: VecDeque<(u32, usize, i64)> = VecDeque::new();

        let adjacency = match direction {
            Direction::Outgoing => &self.outgoing,
            Direction::Incoming => &self.incoming,
        };

        // Initialize Queue
        for src_str in sources {
            if let Some(src_id) = self.node_interner.lookup_id(src_str) {
                // Skip soft-deleted nodes
                if self.tombstones.get(src_id as usize).as_deref() == Some(&true) {
                    continue;
                }
                
                // Mark source as visited so we don't loop back to it
                if (src_id as usize) < visited.len() {
                    visited.set(src_id as usize, true);
                }
                
                // If min_depth is 0, include sources in result
                if min_depth == 0 {
                    result_ids.push(src_id);
                }
                
                // Start search
                queue.push_back((src_id, 0, i64::MIN));
            }
        }

        while let Some((curr_id, curr_depth, arrival_time)) = queue.pop_front() {
            if curr_depth >= max_depth {
                continue;
            }
            
            let next_depth = curr_depth + 1;

            if let Some(edges) = adjacency.get(curr_id as usize) {
                for &(target, type_id, vf, vt, _heat) in edges {
                    // Apply edge type filter
                    if let Some(req_type) = type_filter {
                        if req_type != type_id {
                            continue;
                        }
                    }

                    // Monotonic Check
                    if monotonic && vf < arrival_time {
                        continue;
                    }
                    
                    // Temporal Check
                    match as_of {
                        Some(ts) => { if !(vf <= ts && vt > ts) { continue; } },
                        None => { if vt != MAX_TIME { continue; } }
                    }
                    
                    // Check soft delete
                    if self.tombstones.get(target as usize).as_deref() == Some(&true) {
                        continue;
                    }
                    
                    // Check visited and bounds
                    if (target as usize) < visited.len() && !visited[target as usize] {
                        visited.set(target as usize, true);
                        
                        if next_depth >= min_depth {
                            result_ids.push(target);
                        }
                        
                        // Continue BFS only if we haven't hit max depth
                        if next_depth < max_depth {
                            // For monotonic, next arrival time is this edge's valid_from
                            // If not monotonic, we just pass down min (effectively ignoring)
                            let next_arrival = if monotonic { vf } else { i64::MIN };
                            queue.push_back((target, next_depth, next_arrival));
                        }
                    }
                }
            }
        }

        // Sort for deterministic output
        result_ids.sort_unstable();

        result_ids
            .into_iter()
            .filter_map(|id| self.node_interner.lookup(id).map(|s| s.to_string()))
            .collect()
    }

    /// Traverses edges that overlap with the given time interval [start, end).
    /// Overlap logic: edge.valid_from < end && edge.valid_to > start.
    /// Returns unique neighbor IDs.
    pub fn traverse_interval(&self, sources: &[String], edge_type: Option<&str>, direction: Direction, start: i64, end: i64, constraint: IntervalConstraint) -> Vec<String> {
        let type_filter = edge_type.and_then(|t| self.edge_type_map.get(t).copied());
        let mut result_ids: Vec<u32> = Vec::with_capacity(sources.len() * 2);
        
        let adjacency = match direction {
            Direction::Outgoing => &self.outgoing,
            Direction::Incoming => &self.incoming,
        };

        for src_str in sources {
            if let Some(src_id) = self.node_interner.lookup_id(src_str) {
                if self.tombstones.get(src_id as usize).as_deref() == Some(&true) {
                    continue;
                }

                if let Some(edges) = adjacency.get(src_id as usize) {
                    for &(target, type_id, vf, vt, _heat) in edges {
                        if let Some(req_type) = type_filter {
                            if req_type != type_id { continue; }
                        }
                        
                        // Interval Check
                        let is_match = match constraint {
                            IntervalConstraint::Overlaps => vf < end && vt > start,
                            IntervalConstraint::Contains => vf <= start && vt >= end,
                            IntervalConstraint::During => vf >= start && vt <= end,
                            IntervalConstraint::Meets => vt == start || vf == end,
                        };

                        if is_match {
                             if self.tombstones.get(target as usize).as_deref() == Some(&true) {
                                continue;
                            }
                            result_ids.push(target);
                        }
                    }
                }
            }
        }
        
        result_ids.sort_unstable();
        result_ids.dedup();
        
        result_ids
            .into_iter()
            .filter_map(|id| self.node_interner.lookup(id).map(|s| s.to_string()))
            .collect()
    }

    pub fn node_count(&self) -> usize {
        self.node_interner.len()
    }

    /// Serializes the entire graph topology to a binary file.
    pub fn save_to_file(&self, path: &str) -> Result<(), String> {
        let file = File::create(path).map_err(|e| e.to_string())?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, self).map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Deserializes the graph topology from a binary file.
    pub fn load_from_file(path: &str) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| e.to_string())?;
        let reader = BufReader::new(file);
        bincode::deserialize_from(reader).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bulk_add_dedup() {
        let mut graph = GraphIndex::new();
        
        // Simulate batch loading with duplicates
        // A -> B (KNOWS)
        // A -> B (KNOWS)
        // A -> B (LIKES)
        
        let u_a = graph.get_or_create_node("A");
        let u_b = graph.get_or_create_node("B");
        let t_knows = graph.get_or_create_type("KNOWS");
        let t_likes = graph.get_or_create_type("LIKES");

        // Manually push duplicates simulating blind batch add
        graph.outgoing[u_a as usize].push((u_b, t_knows, 0, MAX_TIME, 0));
        graph.outgoing[u_a as usize].push((u_b, t_knows, 0, MAX_TIME, 0)); // Duplicate
        graph.outgoing[u_a as usize].push((u_b, t_likes, 0, MAX_TIME, 0)); // Different type

        // Pre-compact: 3 edges
        assert_eq!(graph.outgoing[u_a as usize].len(), 3);

        // Compact
        graph.compact();

        // Post-compact: 2 edges (KNOWS, LIKES)
        assert_eq!(graph.outgoing[u_a as usize].len(), 2);
        assert!(graph.outgoing[u_a as usize].contains(&(u_b, t_knows, 0, MAX_TIME, 0)));
        assert!(graph.outgoing[u_a as usize].contains(&(u_b, t_likes, 0, MAX_TIME, 0)));
    }
}
````

## File: packages/quackgraph/crates/quack_core/Cargo.toml
````toml
[package]
name = "quack_core"
version = "0.1.0"
edition = "2021"

[dependencies]
thiserror = "1.0"
serde = { version = "1.0", features = ["derive"] }
bitvec = { version = "1.0", features = ["serde"] }
arrow = { version = "53.0.0" }
bincode = "1.3"
````

## File: packages/quackgraph/packages/native/src/lib.rs
````rust
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
        let ts = as_of.map(|t| t as i64);
        self.inner.get_available_edge_types(&sources, ts)
    }

    /// Returns aggregated statistics (count, heat) for outgoing edges from the given sources.
    /// More efficient than getAvailableEdgeTypes + traverse loop.
    #[napi(js_name = "getSectorStats")]
    pub fn get_sector_stats(&self, sources: Vec<String>, as_of: Option<f64>, allowed_edge_types: Option<Vec<String>>) -> Vec<JsSectorStat> {
        let ts = as_of.map(|t| t as i64);
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
        let ts = as_of.map(|t| t as i64);
        let min_vf = min_valid_from.map(|t| t as i64);
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
        let ts = as_of.map(|t| t as i64);
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

        let ts = as_of.map(|t| t as i64);
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
````

## File: packages/quackgraph/packages/native/build.rs
````rust
extern crate napi_build;

fn main() {
  napi_build::setup();
}
````

## File: packages/quackgraph/packages/native/Cargo.toml
````toml
[package]
edition = "2021"
name = "quack_native"
version = "0.0.1"

[lib]
crate-type = ["cdylib"]

[dependencies]
# Napi dependencies for bridging
napi = { version = "2.12.2", default-features = false, features = ["napi4"] }
napi-derive = "2.12.2"

# Our Core Logic
quack_core = { path = "../../crates/quack_core" }
arrow = { version = "53.0.0" }

[build-dependencies]
napi-build = "2.0.1"
````

## File: packages/quackgraph/packages/native/index.d.ts
````typescript
/* tslint:disable */
/* eslint-disable */

/* auto-generated by NAPI-RS */

export interface JsPatternEdge {
  srcVar: number
  tgtVar: number
  edgeType: string
  direction?: string
}
export interface JsSectorStat {
  edgeType: string
  count: number
  avgHeat: number
}
export declare class NativeGraph {
  constructor()
  addNode(id: string): void
  /**
   * Hydrates the graph from an Arrow IPC stream (Buffer).
   * Zero-copy (mostly) data transfer from DuckDB.
   * Note: Does not verify duplicates. Caller must call compact() afterwards.
   */
  loadArrowIpc(buffer: Buffer): void
  /**
   * Compacts the graph's memory usage.
   * Call this after hydration to reclaim unused capacity in the adjacency lists.
   * Also deduplicates edges added via bulk ingestion.
   */
  compact(): void
  addEdge(source: string, target: string, edgeType: string, validFrom?: number | undefined | null, validTo?: number | undefined | null, heat?: number | undefined | null): void
  removeNode(id: string): void
  removeEdge(source: string, target: string, edgeType: string): void
  /**
   * Updates the 'heat' (pheromone) level of an active edge.
   * Used for reinforcement learning in the agent loop.
   */
  updateEdgeHeat(source: string, target: string, edgeType: string, heat: number): void
  /**
   * Returns available edge types from the given source nodes.
   * Used for "Ghost Earth" Satellite View (LOD 0).
   */
  getAvailableEdgeTypes(sources: Array<string>, asOf?: number | undefined | null): Array<string>
  /**
   * Returns aggregated statistics (count, heat) for outgoing edges from the given sources.
   * More efficient than getAvailableEdgeTypes + traverse loop.
   */
  getSectorStats(sources: Array<string>, asOf?: number | undefined | null, allowedEdgeTypes?: Array<string> | undefined | null): Array<JsSectorStat>
  /**
   * Performs a single-hop traversal (bfs-step).
   * Returns unique neighbor IDs.
   */
  traverse(sources: Array<string>, edgeType?: string | undefined | null, direction?: string | undefined | null, asOf?: number | undefined | null, minValidFrom?: number | undefined | null): Array<string>
  /**
   * Performs a traversal finding all neighbors connected via edges that overlap
   * with the specified time window [start, end).
   * Timestamps are in milliseconds (JS standard).
   * Constraint: 'overlaps' (default), 'contains', 'during', 'meets'.
   */
  traverseInterval(sources: Array<string>, edgeType: string | undefined | null, direction: string | undefined | null, start: number, end: number, constraint?: string | undefined | null): Array<string>
  /**
   * Performs a recursive traversal (BFS) with depth bounds.
   * Returns unique node IDs reachable within [min_depth, max_depth].
   */
  traverseRecursive(sources: Array<string>, edgeType?: string | undefined | null, direction?: string | undefined | null, minDepth?: number | undefined | null, maxDepth?: number | undefined | null, asOf?: number | undefined | null, monotonic?: boolean | undefined | null): Array<string>
  /**
   * Finds subgraphs matching the given pattern.
   * `start_ids` maps to variable 0 in the pattern.
   */
  matchPattern(startIds: Array<string>, pattern: Array<JsPatternEdge>, asOf?: number | undefined | null): Array<Array<string>>
  /**
   * Returns the number of nodes in the interned index.
   * Useful for debugging hydration.
   */
  get nodeCount(): number
  get edgeCount(): number
  saveSnapshot(path: string): void
  loadSnapshot(path: string): void
}
````

## File: packages/quackgraph/packages/native/index.js
````javascript
/* tslint:disable */
/* eslint-disable */
/* prettier-ignore */

/* auto-generated by NAPI-RS */

const { existsSync, readFileSync } = require('fs')
const { join } = require('path')

const { platform, arch } = process

let nativeBinding = null
let localFileExisted = false
let loadError = null

function isMusl() {
  // For Node 10
  if (!process.report || typeof process.report.getReport !== 'function') {
    try {
      const lddPath = require('child_process').execSync('which ldd').toString().trim()
      return readFileSync(lddPath, 'utf8').includes('musl')
    } catch (e) {
      return true
    }
  } else {
    const { glibcVersionRuntime } = process.report.getReport().header
    return !glibcVersionRuntime
  }
}

switch (platform) {
  case 'android':
    switch (arch) {
      case 'arm64':
        localFileExisted = existsSync(join(__dirname, 'quack-native.android-arm64.node'))
        try {
          if (localFileExisted) {
            nativeBinding = require('./quack-native.android-arm64.node')
          } else {
            nativeBinding = require('@quackgraph/native-android-arm64')
          }
        } catch (e) {
          loadError = e
        }
        break
      case 'arm':
        localFileExisted = existsSync(join(__dirname, 'quack-native.android-arm-eabi.node'))
        try {
          if (localFileExisted) {
            nativeBinding = require('./quack-native.android-arm-eabi.node')
          } else {
            nativeBinding = require('@quackgraph/native-android-arm-eabi')
          }
        } catch (e) {
          loadError = e
        }
        break
      default:
        throw new Error(`Unsupported architecture on Android ${arch}`)
    }
    break
  case 'win32':
    switch (arch) {
      case 'x64':
        localFileExisted = existsSync(
          join(__dirname, 'quack-native.win32-x64-msvc.node')
        )
        try {
          if (localFileExisted) {
            nativeBinding = require('./quack-native.win32-x64-msvc.node')
          } else {
            nativeBinding = require('@quackgraph/native-win32-x64-msvc')
          }
        } catch (e) {
          loadError = e
        }
        break
      case 'ia32':
        localFileExisted = existsSync(
          join(__dirname, 'quack-native.win32-ia32-msvc.node')
        )
        try {
          if (localFileExisted) {
            nativeBinding = require('./quack-native.win32-ia32-msvc.node')
          } else {
            nativeBinding = require('@quackgraph/native-win32-ia32-msvc')
          }
        } catch (e) {
          loadError = e
        }
        break
      case 'arm64':
        localFileExisted = existsSync(
          join(__dirname, 'quack-native.win32-arm64-msvc.node')
        )
        try {
          if (localFileExisted) {
            nativeBinding = require('./quack-native.win32-arm64-msvc.node')
          } else {
            nativeBinding = require('@quackgraph/native-win32-arm64-msvc')
          }
        } catch (e) {
          loadError = e
        }
        break
      default:
        throw new Error(`Unsupported architecture on Windows: ${arch}`)
    }
    break
  case 'darwin':
    localFileExisted = existsSync(join(__dirname, 'quack-native.darwin-universal.node'))
    try {
      if (localFileExisted) {
        nativeBinding = require('./quack-native.darwin-universal.node')
      } else {
        nativeBinding = require('@quackgraph/native-darwin-universal')
      }
      break
    } catch {}
    switch (arch) {
      case 'x64':
        localFileExisted = existsSync(join(__dirname, 'quack-native.darwin-x64.node'))
        try {
          if (localFileExisted) {
            nativeBinding = require('./quack-native.darwin-x64.node')
          } else {
            nativeBinding = require('@quackgraph/native-darwin-x64')
          }
        } catch (e) {
          loadError = e
        }
        break
      case 'arm64':
        localFileExisted = existsSync(
          join(__dirname, 'quack-native.darwin-arm64.node')
        )
        try {
          if (localFileExisted) {
            nativeBinding = require('./quack-native.darwin-arm64.node')
          } else {
            nativeBinding = require('@quackgraph/native-darwin-arm64')
          }
        } catch (e) {
          loadError = e
        }
        break
      default:
        throw new Error(`Unsupported architecture on macOS: ${arch}`)
    }
    break
  case 'freebsd':
    if (arch !== 'x64') {
      throw new Error(`Unsupported architecture on FreeBSD: ${arch}`)
    }
    localFileExisted = existsSync(join(__dirname, 'quack-native.freebsd-x64.node'))
    try {
      if (localFileExisted) {
        nativeBinding = require('./quack-native.freebsd-x64.node')
      } else {
        nativeBinding = require('@quackgraph/native-freebsd-x64')
      }
    } catch (e) {
      loadError = e
    }
    break
  case 'linux':
    switch (arch) {
      case 'x64':
        if (isMusl()) {
          localFileExisted = existsSync(
            join(__dirname, 'quack-native.linux-x64-musl.node')
          )
          try {
            if (localFileExisted) {
              nativeBinding = require('./quack-native.linux-x64-musl.node')
            } else {
              nativeBinding = require('@quackgraph/native-linux-x64-musl')
            }
          } catch (e) {
            loadError = e
          }
        } else {
          localFileExisted = existsSync(
            join(__dirname, 'quack-native.linux-x64-gnu.node')
          )
          try {
            if (localFileExisted) {
              nativeBinding = require('./quack-native.linux-x64-gnu.node')
            } else {
              nativeBinding = require('@quackgraph/native-linux-x64-gnu')
            }
          } catch (e) {
            loadError = e
          }
        }
        break
      case 'arm64':
        if (isMusl()) {
          localFileExisted = existsSync(
            join(__dirname, 'quack-native.linux-arm64-musl.node')
          )
          try {
            if (localFileExisted) {
              nativeBinding = require('./quack-native.linux-arm64-musl.node')
            } else {
              nativeBinding = require('@quackgraph/native-linux-arm64-musl')
            }
          } catch (e) {
            loadError = e
          }
        } else {
          localFileExisted = existsSync(
            join(__dirname, 'quack-native.linux-arm64-gnu.node')
          )
          try {
            if (localFileExisted) {
              nativeBinding = require('./quack-native.linux-arm64-gnu.node')
            } else {
              nativeBinding = require('@quackgraph/native-linux-arm64-gnu')
            }
          } catch (e) {
            loadError = e
          }
        }
        break
      case 'arm':
        if (isMusl()) {
          localFileExisted = existsSync(
            join(__dirname, 'quack-native.linux-arm-musleabihf.node')
          )
          try {
            if (localFileExisted) {
              nativeBinding = require('./quack-native.linux-arm-musleabihf.node')
            } else {
              nativeBinding = require('@quackgraph/native-linux-arm-musleabihf')
            }
          } catch (e) {
            loadError = e
          }
        } else {
          localFileExisted = existsSync(
            join(__dirname, 'quack-native.linux-arm-gnueabihf.node')
          )
          try {
            if (localFileExisted) {
              nativeBinding = require('./quack-native.linux-arm-gnueabihf.node')
            } else {
              nativeBinding = require('@quackgraph/native-linux-arm-gnueabihf')
            }
          } catch (e) {
            loadError = e
          }
        }
        break
      case 'riscv64':
        if (isMusl()) {
          localFileExisted = existsSync(
            join(__dirname, 'quack-native.linux-riscv64-musl.node')
          )
          try {
            if (localFileExisted) {
              nativeBinding = require('./quack-native.linux-riscv64-musl.node')
            } else {
              nativeBinding = require('@quackgraph/native-linux-riscv64-musl')
            }
          } catch (e) {
            loadError = e
          }
        } else {
          localFileExisted = existsSync(
            join(__dirname, 'quack-native.linux-riscv64-gnu.node')
          )
          try {
            if (localFileExisted) {
              nativeBinding = require('./quack-native.linux-riscv64-gnu.node')
            } else {
              nativeBinding = require('@quackgraph/native-linux-riscv64-gnu')
            }
          } catch (e) {
            loadError = e
          }
        }
        break
      case 's390x':
        localFileExisted = existsSync(
          join(__dirname, 'quack-native.linux-s390x-gnu.node')
        )
        try {
          if (localFileExisted) {
            nativeBinding = require('./quack-native.linux-s390x-gnu.node')
          } else {
            nativeBinding = require('@quackgraph/native-linux-s390x-gnu')
          }
        } catch (e) {
          loadError = e
        }
        break
      default:
        throw new Error(`Unsupported architecture on Linux: ${arch}`)
    }
    break
  default:
    throw new Error(`Unsupported OS: ${platform}, architecture: ${arch}`)
}

if (!nativeBinding) {
  if (loadError) {
    throw loadError
  }
  throw new Error(`Failed to load native binding`)
}

const { NativeGraph } = nativeBinding

module.exports.NativeGraph = NativeGraph
````

## File: packages/quackgraph/packages/native/package.json
````json
{
  "name": "@quackgraph/native",
  "version": "0.0.1",
  "main": "index.js",
  "types": "index.d.ts",
  "napi": {
    "name": "quack-native",
    "triples": {
      "defaults": true,
      "additional": [
        "x86_64-unknown-linux-musl",
        "aarch64-unknown-linux-gnu",
        "i686-pc-windows-msvc",
        "armv7-unknown-linux-gnueabihf",
        "aarch64-apple-darwin",
        "aarch64-linux-android",
        "x86_64-unknown-freebsd",
        "aarch64-unknown-linux-musl",
        "aarch64-pc-windows-msvc",
        "armv7-linux-androideabi"
      ]
    }
  },
  "scripts": {
    "build": "napi build --platform --release",
    "build:debug": "napi build --platform",
    "artifacts": "napi artifacts",
    "test": "node --test"
  },
  "devDependencies": {
    "@napi-rs/cli": "^2.18.0"
  },
  "engines": {
    "node": ">= 10"
  }
}
````

## File: packages/quackgraph/packages/quack-graph/src/db.ts
````typescript
import { Database } from 'duckdb-async';
import { tableFromJSON, tableToIPC } from 'apache-arrow';

// Interface for operations that can be performed within a transaction or globally
export interface DbExecutor {
  // biome-ignore lint/suspicious/noExplicitAny: SQL params are generic
  execute(sql: string, params?: any[]): Promise<void>;
  // biome-ignore lint/suspicious/noExplicitAny: SQL results are generic
  query(sql: string, params?: any[]): Promise<any[]>;
}

export class DuckDBManager implements DbExecutor {
  private db: Database | null = null;
  private _path: string;

  constructor(path: string = ':memory:') {
    this._path = path;
  }

  async init() {
    if (!this.db) {
      this.db = await Database.create(this._path);
    }
  }

  get path(): string {
    return this._path;
  }

  getDb(): Database {
    if (!this.db) {
      throw new Error('Database not initialized. Call init() first.');
    }
    return this.db;
  }

  // biome-ignore lint/suspicious/noExplicitAny: SQL params
  async execute(sql: string, params: any[] = []): Promise<void> {
    const db = this.getDb();
    await db.run(sql, ...params);
  }

  // biome-ignore lint/suspicious/noExplicitAny: SQL results
  async query(sql: string, params: any[] = []): Promise<any[]> {
    const db = this.getDb();
    return await db.all(sql, ...params);
  }

  /**
   * Executes a callback within a transaction using a dedicated connection.
   * This guarantees that all operations inside the callback share the same ACID scope.
   */
  async transaction<T>(callback: (executor: DbExecutor) => Promise<T>): Promise<T> {
    const db = this.getDb();
    const conn = await db.connect();
    
    // Create a transaction-bound executor wrapper
    const txExecutor: DbExecutor = {
      // biome-ignore lint/suspicious/noExplicitAny: SQL params
      execute: async (sql: string, params: any[] = []) => {
        await conn.run(sql, ...params);
      },
      // biome-ignore lint/suspicious/noExplicitAny: SQL results
      query: async (sql: string, params: any[] = []) => {
        return await conn.all(sql, ...params);
      }
    };

    try {
      await conn.run('BEGIN TRANSACTION');
      const result = await callback(txExecutor);
      await conn.run('COMMIT');
      return result;
    } catch (e) {
      try {
        await conn.run('ROLLBACK');
      } catch (rollbackError) {
        console.error('Failed to rollback transaction:', rollbackError);
      }
      throw e;
    } finally {
      // Best effort close
      // biome-ignore lint/suspicious/noExplicitAny: DuckDB connection types are incomplete
      if (conn && typeof (conn as any).close === 'function') {
        // biome-ignore lint/suspicious/noExplicitAny: DuckDB connection types are incomplete
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
      // Hack: Access underlying node-duckdb connection/database
      // duckdb-async instance holds 'db' property which is the native Database
      // biome-ignore lint/suspicious/noExplicitAny: DuckDB internals
      const rawDb = (db as any).db || db;

      if (!rawDb) return reject(new Error("Could not access underlying DuckDB Native instance."));

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
      if (typeof rawDb.arrowIPCAll === 'function') {
        // biome-ignore lint/suspicious/noExplicitAny: internal callback signature
        rawDb.arrowIPCAll(sql, ...params, (err: any, result: any) => {
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
         // Fallback: Create a raw connection
         try {
            const rawConn = rawDb.connect();
            
            // Handle case where rawDb is actually the connection itself (sometimes happens in certain pool configs)
            const target = typeof rawDb.arrowIPCAll === 'function' 
              ? rawDb 
              : (rawConn && typeof rawConn.arrowIPCAll === 'function' ? rawConn : null);

            if (target) {
               // biome-ignore lint/suspicious/noExplicitAny: internal callback signature
               target.arrowIPCAll(sql, ...params, (err: any, result: any) => {
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
````

## File: packages/quackgraph/packages/quack-graph/src/graph.ts
````typescript
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
      () => {},
      () => {}
    );

    return result;
  }
}

export class QuackGraph {
  db: DuckDBManager;
  schema: SchemaManager;
  native: NativeGraph;
  private writeLock = new WriteLock();
  
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
  async addNode(id: string, labels: string[], props: Record<string, any> = {}) {
    await this.writeLock.run(async () => {
      // 1. Write to Disk (Source of Truth)
      await this.schema.writeNode(id, labels, props);
      // 2. Write to RAM (Cache)
      this.native.addNode(id);
    });
  }

  // biome-ignore lint/suspicious/noExplicitAny: generic properties
  async addEdge(source: string, target: string, type: string, props: Record<string, any> = {}) {
    await this.writeLock.run(async () => {
      // 1. Write to Disk
      await this.schema.writeEdge(source, target, type, props);
      // 2. Write to RAM (Current time)
      // We pass undefined for timestamps, so Rust defaults to (0, MAX) which is functionally "Active"
      this.native.addEdge(source, target, type, undefined, undefined);
    });
  }

  async deleteNode(id: string) {
    await this.writeLock.run(async () => {
      // 1. Write to Disk (Soft Delete)
      await this.schema.deleteNode(id);
      // 2. Write to RAM (Tombstone)
      this.native.removeNode(id);
    });
  }

  async deleteEdge(source: string, target: string, type: string) {
    await this.writeLock.run(async () => {
      // 1. Write to Disk (Soft Delete)
      await this.schema.deleteEdge(source, target, type);
      // 2. Write to RAM (Remove)
      this.native.removeEdge(source, target, type);
    });
  }

  // --- Pheromones & Schema (Agent) ---

  async updateEdgeHeat(source: string, target: string, type: string, heat: number) {
    // 1. Write to Disk (In-Place Update for Learning Signal)
    await this.writeLock.run(async () => {
      // We only update active edges (valid_to IS NULL)
      await this.db.execute(
        "UPDATE edges SET heat = ? WHERE source = ? AND target = ? AND type = ? AND valid_to IS NULL", 
        [heat, source, target, type]
      );
    });
    // 2. Write to RAM (Atomic)
    this.native.updateEdgeHeat(source, target, type, heat);
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
  async mergeNode(label: string, matchProps: Record<string, any>, setProps: Record<string, any> = {}) {
    return this.writeLock.run(async () => {
      const id = await this.schema.mergeNode(label, matchProps, setProps);
      // Update cache
      this.native.addNode(id);
      return id;
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
````

## File: packages/quackgraph/packages/quack-graph/src/index.ts
````typescript
export * from './db';
export * from './schema';
export * from './graph';
export * from './query';
````

## File: packages/quackgraph/packages/quack-graph/src/query.ts
````typescript
import type { QuackGraph } from './graph';

type TraversalStep = {
  type: 'out' | 'in';
  edge: string;
  bounds?: { min: number; max: number };
};

export class QueryBuilder {
  private graph: QuackGraph;
  private startLabels: string[];
  private endLabels: string[] = [];

  // Bottom Bun Filters (Initial selection)
  // biome-ignore lint/suspicious/noExplicitAny: Generic filter criteria
  private initialFilters: Record<string, any> = {};
  private vectorSearch: { vector: number[]; limit: number } | null = null;

  // The Meat (Traversal)
  private traversals: TraversalStep[] = [];

  // Top Bun Filters (Final selection)
  // biome-ignore lint/suspicious/noExplicitAny: Generic filter criteria
  private terminalFilters: Record<string, any> = {};

  private aggState = {
    groupBy: [] as string[],
    orderBy: [] as { field: string; dir: 'ASC' | 'DESC' }[],
    limit: undefined as number | undefined,
    offset: undefined as number | undefined,
  };

  constructor(graph: QuackGraph, labels: string[]) {
    this.graph = graph;
    this.startLabels = labels;
  }

  /**
   * Sets depth bounds for the last traversal step.
   * Useful for variable length paths like `(a)-[:KNOWS*1..5]->(b)`.
   * Must be called immediately after .out() or .in().
   * @param min Minimum hops (default: 1)
   * @param max Maximum hops (default: 1)
   */
  depth(min: number, max: number): this {
    if (this.traversals.length === 0) {
      throw new Error("depth() must be called after a traversal step (.out() or .in())");
    }
    const lastIndex = this.traversals.length - 1;
    // biome-ignore lint/style/noNonNullAssertion: length check above ensures array is not empty
    const lastStep = this.traversals[lastIndex]!;
    lastStep.bounds = { min, max };
    return this;
  }

  /**
   * Filter nodes by properties.
   * If called before traversal, applies to Start Nodes.
   * If called after traversal, applies to End Nodes.
   */
  // biome-ignore lint/suspicious/noExplicitAny: Generic filter criteria
  where(criteria: Record<string, any>): this {
    if (this.traversals.length === 0) {
      this.initialFilters = { ...this.initialFilters, ...criteria };
    } else {
      this.terminalFilters = { ...this.terminalFilters, ...criteria };
    }
    return this;
  }

  /**
   * Perform a Vector Similarity Search (HNSW).
   * This effectively sorts the start nodes by distance to the query vector.
   */
  nearText(vector: number[], options: { limit?: number } = {}): this {
    this.vectorSearch = { 
      vector, 
      limit: options.limit || 10 
    };
    return this;
  }

  out(edgeType: string): this {
    this.traversals.push({ type: 'out', edge: edgeType });
    return this;
  }

  in(edgeType: string): this {
    this.traversals.push({ type: 'in', edge: edgeType });
    return this;
  }

  groupBy(field: string): this {
    this.aggState.groupBy.push(field);
    return this;
  }

  orderBy(field: string, dir: 'ASC' | 'DESC' = 'ASC'): this {
    this.aggState.orderBy.push({ field, dir });
    return this;
  }

  limit(n: number): this {
    this.aggState.limit = n;
    return this;
  }

  offset(n: number): this {
    this.aggState.offset = n;
    return this;
  }

  /**
   * Filter the nodes at the end of the traversal by label.
   */
  node(labels: string[]): this {
    this.endLabels = labels;
    return this;
  }

  /**
   * Helper to construct the temporal validity clause
   */
  private getTemporalClause(tableAlias: string = ''): string {
    const prefix = tableAlias ? `${tableAlias}.` : '';
    if (this.graph.context.asOf) {
      // Time Travel: valid_from <= T AND (valid_to > T OR valid_to IS NULL)
      // Use microseconds since epoch for consistency with native layer
      const micros = this.graph.context.asOf.getTime() * 1000;
      // Convert database timestamps to microseconds for comparison
      return `(date_diff('us', '1970-01-01'::TIMESTAMPTZ, ${prefix}valid_from) <= ${micros} AND (date_diff('us', '1970-01-01'::TIMESTAMPTZ, ${prefix}valid_to) > ${micros} OR ${prefix}valid_to IS NULL))`;
    }
    // Default: Current valid records (valid_to is NULL)
    return `${prefix}valid_to IS NULL`;
  }

  /**
   * Executes the query.
   * @param projection Optional SQL projection string (e.g., 'count(*), avg(properties->>age)') or a JS mapper function.
   */
  // biome-ignore lint/suspicious/noExplicitAny: Generic result mapper
  async select<T = any>(projection?: string | ((node: any) => T)): Promise<T[]> {
    const isRawSql = typeof projection === 'string';
    const mapper = typeof projection === 'function' ? projection : undefined;

    // --- Step 1: DuckDB Filter (Bottom Bun) ---
    // Objective: Get a list of "Active" Node IDs to feed into the graph.

    let query = `SELECT id FROM nodes`;
    // biome-ignore lint/suspicious/noExplicitAny: SQL parameters
    const params: any[] = [];
    const conditions: string[] = [];

    // 1.a Temporal Filter
    conditions.push(this.getTemporalClause());

    // 1.b Label Filter
    if (this.startLabels.length > 0) {
      // Check if ANY of the labels match. For V1 we check the first one or intersection.
      conditions.push(`list_contains(labels, ?)`);
      params.push(this.startLabels[0]);
    }

    // 1.c Property Filter
    for (const [key, value] of Object.entries(this.initialFilters)) {
      if (key === 'id') {
        conditions.push(`id = ?`);
        params.push(value);
      } else {
        conditions.push(`json_extract(properties, '$.${key}') = ?::JSON`);
        params.push(JSON.stringify(value));
      }
    }

    // 1.d Vector Search (Order By Distance)
    let orderBy = '';
    let limit = '';
    if (this.vectorSearch) {
      if (!this.graph.capabilities.vss) {
        throw new Error('Vector search requires the DuckDB "vss" extension, which is not available or failed to load.');
      }
      const vectorSize = this.vectorSearch.vector.length;
      orderBy = `ORDER BY array_distance(embedding::DOUBLE[${vectorSize}], ?::DOUBLE[${vectorSize}])`;
      limit = `LIMIT ${this.vectorSearch.limit}`;
      params.push(JSON.stringify(this.vectorSearch.vector));
    }

    if (conditions.length > 0) {
      query += ` WHERE ${conditions.join(' AND ')}`;
    }

    query += ` ${orderBy} ${limit}`;

    const startRows = await this.graph.db.query(query, params);
    let currentIds: string[] = startRows.map(row => row.id);

    if (currentIds.length === 0) return [];

    // --- Step 2: Rust Traversal (The Meat) ---
    // Note: Rust Graph Index is currently "Latest Topology Only". 
    // Time Travel on topology requires checking edge validity during traversal (V2).
    // For V1, we accept that traversal is instant/current, but properties are historical.

    for (const step of this.traversals) {
      const asOfTs = this.graph.context.asOf ? this.graph.context.asOf.getTime() * 1000 : undefined;

      if (currentIds.length === 0) break;
      
      if (step.bounds) {
        currentIds = this.graph.native.traverseRecursive(
          currentIds,
          step.edge,
          step.type,
          step.bounds.min,
          step.bounds.max,
          asOfTs
        );
      } else {
        // step.type is 'out' | 'in'
        currentIds = this.graph.native.traverse(currentIds, step.edge, step.type, asOfTs);
      }
    }

    // Optimization: If traversal resulted in no nodes, stop early.
    if (currentIds.length === 0) return [];

    // --- Step 3: DuckDB Hydration (Top Bun) ---
    // Objective: Fetch full properties for the resulting IDs, applying terminal filters.

    const finalConditions: string[] = [];
    // biome-ignore lint/suspicious/noExplicitAny: SQL parameters
    const finalParams: any[] = [];

    // 3.0 Label Filter (for End Nodes)
    if (this.endLabels.length > 0) {
      finalConditions.push(`list_contains(labels, ?)`);
      finalParams.push(this.endLabels[0]);
    }

    // 3.a IDs match
    // We can't use parameters for IN clause effectively with dynamic length in all drivers.
    // Constructing placeholders.
    const placeholders = currentIds.map(() => '?').join(',');
    finalConditions.push(`id IN (${placeholders})`);
    finalParams.push(...currentIds);

    // 3.b Temporal Validity
    finalConditions.push(this.getTemporalClause());

    // 3.c Terminal Property Filters
    for (const [key, value] of Object.entries(this.terminalFilters)) {
      if (key === 'id') {
        finalConditions.push(`id = ?`);
        finalParams.push(value);
      } else {
        finalConditions.push(`json_extract(properties, '$.${key}') = ?::JSON`);
        finalParams.push(JSON.stringify(value));
      }
    }

    // 3.d Aggregation / Grouping / Ordering
    let selectClause = 'SELECT *';
    if (isRawSql) {
      selectClause = `SELECT ${projection}`;
    }

    let suffix = '';
    if (this.aggState.groupBy.length > 0) {
      suffix += ` GROUP BY ${this.aggState.groupBy.join(', ')}`;
    }
    
    if (this.aggState.orderBy.length > 0) {
      const orders = this.aggState.orderBy.map(o => `${o.field} ${o.dir}`).join(', ');
      suffix += ` ORDER BY ${orders}`;
    }

    if (this.aggState.limit !== undefined) {
      suffix += ` LIMIT ${this.aggState.limit}`;
    }
    if (this.aggState.offset !== undefined) {
      suffix += ` OFFSET ${this.aggState.offset}`;
    }

    const finalSql = `${selectClause} FROM nodes WHERE ${finalConditions.join(' AND ')} ${suffix}`;
    const results = await this.graph.db.query(finalSql, finalParams);

    return results.map(r => {
      if (isRawSql) return r;

      let props = r.properties;
      if (typeof props === 'string') {
        try { props = JSON.parse(props); } catch {}
      }
      const node = {
        id: r.id,
        labels: r.labels,
        ...props
      };
      return mapper ? mapper(node) : node;
    });
  }
}
````

## File: packages/quackgraph/packages/quack-graph/src/schema.ts
````typescript
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
  async writeNode(id: string, labels: string[], properties: Record<string, any> = {}) {
    await this.db.transaction(async (tx: DbExecutor) => {
      // 1. Close existing record (SCD Type 2)
      await tx.execute(
        `UPDATE nodes SET valid_to = (current_timestamp AT TIME ZONE 'UTC') WHERE id = ? AND valid_to IS NULL`,
        [id]
      );
      // 2. Insert new version
      await tx.execute(`
        INSERT INTO nodes (row_id, id, labels, properties, valid_from, valid_to) 
        VALUES (nextval('seq_node_id'), ?, ?::JSON::TEXT[], ?::JSON, (current_timestamp AT TIME ZONE 'UTC'), NULL)
      `, [id, JSON.stringify(labels), JSON.stringify(properties)]);
    });
  }

  // biome-ignore lint/suspicious/noExplicitAny: generic properties
  async writeEdge(source: string, target: string, type: string, properties: Record<string, any> = {}) {
    await this.db.transaction(async (tx: DbExecutor) => {
      // 1. Close existing edge
      await tx.execute(
        `UPDATE edges SET valid_to = (current_timestamp AT TIME ZONE 'UTC') WHERE source = ? AND target = ? AND type = ? AND valid_to IS NULL`,
        [source, target, type]
      );
      // 2. Insert new version
      await tx.execute(`
        INSERT INTO edges (source, target, type, properties, valid_from, valid_to) 
        VALUES (?, ?, ?, ?::JSON, (current_timestamp AT TIME ZONE 'UTC'), NULL)
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
  async mergeNode(label: string, matchProps: Record<string, any>, setProps: Record<string, any>): Promise<string> {
    // 1. Build Search Query
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
        await tx.execute(`UPDATE nodes SET valid_to = (current_timestamp AT TIME ZONE 'UTC') WHERE id = ? AND valid_to IS NULL`, [id]);
      } else {
        // Insert New
        id = matchProps.id || crypto.randomUUID();
        finalProps = { ...matchProps, ...setProps };
        finalLabels = [label];
      }

      // Insert new version (for both Update and Create cases)
      await tx.execute(`
        INSERT INTO nodes (row_id, id, labels, properties, valid_from, valid_to) 
        VALUES (nextval('seq_node_id'), ?, ?::JSON::TEXT[], ?::JSON, (current_timestamp AT TIME ZONE 'UTC'), NULL)
      `, [id, JSON.stringify(finalLabels), JSON.stringify(finalProps)]);

      return id;
    });
  }
}
````

## File: packages/quackgraph/packages/quack-graph/package.json
````json
{
  "name": "@quackgraph/graph",
  "version": "0.1.0",
  "main": "src/index.ts",
  "module": "dist/quack-graph.esm.js",
  "types": "dist/index.d.ts",
  "type": "module",
  "exports": {
    ".": {
      "types": "./dist/index.d.ts",
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
    "duckdb-async": "^1.0.0",
    "apache-arrow": "^17.0.0",
    "@quackgraph/native": "workspace:*"
  },
  "devDependencies": {
    "typescript": "^5.0.0",
    "@types/node": "^20.0.0"
  }
}
````

## File: packages/quackgraph/biome.json
````json
{
  "$schema": "https://biomejs.dev/schemas/2.3.8/schema.json",
  "vcs": {
    "enabled": true,
    "clientKind": "git",
    "useIgnoreFile": true
  },
  "files": {
    "ignoreUnknown": true,
    "includes": [
      "**",
      "!target",
      "!dist",
      "!node_modules",
      "!**/*.node",
      "!repomix.config.json"
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

## File: packages/quackgraph/Cargo.toml
````toml
[workspace]
members = [
    "crates/*",
    "packages/native"
]
resolver = "2"
````

## File: packages/quackgraph/package.json
````json
{
  "name": "quackgraph",
  "module": "index.ts",
  "type": "module",
  "private": true,
  "workspaces": [
    "packages/*"
  ],
  "scripts": {
    "format": "biome format --write . && cargo fmt",
    "clean": "rm -rf node_modules target packages/*/node_modules packages/*/dist packages/native/*.node",
    "postinstall": "bun run --cwd packages/native build",
    "typecheck": "tsc --noEmit && cargo check --workspace",
    "lint": "biome lint . && cargo clippy --workspace -- -D warnings",
    "check": "biome check . && bun run typecheck",
    "dev": "bun test --watch",
    "test": "bun run build && bun test && cargo test --workspace",
    "build": "bun run --cwd packages/native build && tsup",
    "build:native": "bun run --cwd packages/native build",
    "build:ts": "tsup",
    "build:watch": "tsup --watch"
  },
  "devDependencies": {
    "@biomejs/biome": "latest",
    "@types/bun": "latest",
    "tsup": "^8.5.1",
    "typescript": "^5.0.0"
  },
  "peerDependencies": {
    "typescript": "^5"
  }
}
````

## File: packages/quackgraph/tsconfig.json
````json
{
  "compilerOptions": {
    // Environment setup & latest features
    "lib": ["ESNext"],
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
  "include": ["packages/*/src/**/*"]
}
````

## File: packages/quackgraph/tsup.config.ts
````typescript
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
````

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

    // biome-ignore lint/suspicious/noExplicitAny: SQL result
    const result = await this.graph.db.query(sql, [anchorNodeId, targetLabel]);
    // biome-ignore lint/suspicious/noExplicitAny: SQL result row check
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
  edgeType: z.string().describe("The edge type to traverse"),
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

## File: packages/agent/src/mastra/tools/index.ts
````typescript
import { createTool } from '@mastra/core/tools';
import { z } from 'zod';
import { getGraphInstance } from '../../lib/graph-instance';
import { GraphTools } from '../../tools/graph-tools';

// We wrap the existing GraphTools logic to make it available to Mastra agents/workflows

export const sectorScanTool = createTool({
  id: 'sector-scan',
  description: 'Get a summary of available moves (edge types) from the current nodes (LOD 0)',
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
  execute: async ({ context }) => {
    const graph = getGraphInstance();
    const tools = new GraphTools(graph);
    const summary = await tools.getSectorSummary(context.nodeIds, context.asOf, context.allowedEdgeTypes);
    return { summary };
  },
});

export const topologyScanTool = createTool({
  id: 'topology-scan',
  description: 'Get IDs of neighbors reachable via a specific edge type (LOD 1)',
  inputSchema: z.object({
    nodeIds: z.array(z.string()),
    edgeType: z.string(),
    asOf: z.number().optional(),
    minValidFrom: z.number().optional(),
  }),
  outputSchema: z.object({
    neighborIds: z.array(z.string()),
  }),
  execute: async ({ context }) => {
    const graph = getGraphInstance();
    const tools = new GraphTools(graph);
    const neighborIds = await tools.topologyScan(context.nodeIds, context.edgeType, context.asOf, context.minValidFrom);
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
  execute: async ({ context }) => {
    const graph = getGraphInstance();
    const tools = new GraphTools(graph);
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

### 2.1 The Three-Layer Zoom

1.  **LOD 0: Satellite View (The Ghost Layer)**
    *   **Data:** Dynamic Cluster Centroids (Virtual Nodes).
    *   **Action:** The Scout LLM selects a domain. "The user is asking about *Health*. Zoom into the Health Cluster."
    *   **Mechanism:** QuackGraph maintains background community detection. It exposes `GhostID`s representing entire subgraphs.

2.  **LOD 1: Drone View (The Structural Layer)**
    *   **Data:** The "Spine" (Entities & Relationships). No chunks.
    *   **Action:** The Scout navigates the topology. "Path: `(User) --[LOGGED]--> (Symptom: Headache) --[COINCIDES_WITH]--> (Diet: Caffeine)`."
    *   **Mechanism:** Integer-only traversal in Rust.

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
````

## File: packages/agent/src/mastra/agents/judge-agent.ts
````typescript
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
  `,
  model: {
    id: 'groq/llama-3.3-70b-versatile',
  }
});
````

## File: packages/agent/src/mastra/agents/router-agent.ts
````typescript
import { Agent } from '@mastra/core/agent';

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
  }
});
````

## File: packages/agent/src/mastra/index.ts
````typescript
import { Mastra } from '@mastra/core/mastra';
// import { PinoLogger } from '@mastra/loggers';
import { scoutAgent } from './agents/scout-agent';
import { judgeAgent } from './agents/judge-agent';
import { routerAgent } from './agents/router-agent';
import { metabolismWorkflow } from './workflows/metabolism-workflow';

export const mastra = new Mastra({
  agents: { scoutAgent, judgeAgent, routerAgent },
  workflows: { metabolismWorkflow },
  observability: {
    default: {
      enabled: true,

      // exporters: [new DefaultExporter()],
    },
  },
});
````

## File: packages/agent/src/tools/graph-tools.ts
````typescript
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
  async topologyScan(currentNodes: string[], edgeType: string, asOf?: number, _minValidFrom?: number): Promise<string[]> {
    if (currentNodes.length === 0) return [];
    // Native traverse does not support minValidFrom yet
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

## File: packages/agent/src/mastra/agents/scout-agent.ts
````typescript
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

## File: packages/agent/src/index.ts
````typescript
export * from './labyrinth';
export * from './types';
export * from './agent/chronos';
export * from './governance/schema-registry';
export * from './tools/graph-tools';
// Expose Mastra definitions if needed, but facade prefers hiding them
export { mastra } from './mastra';

import type { QuackGraph } from '@quackgraph/graph';
import { Labyrinth } from './labyrinth';
import type { AgentConfig } from './types';
import { mastra } from './mastra';

/**
 * Factory to create a fully wired Labyrinth Agent.
 * Uses default Mastra agents (Scout, Judge, Router) unless overridden.
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



/**
 * Runs the Metabolism (Dreaming) cycle to prune and summarize old nodes.
 */
export async function runMetabolism(targetLabel: string, minAgeDays: number = 30) {
  const workflow = mastra.getWorkflow('metabolismWorkflow');
  if (!workflow) throw new Error("Metabolism workflow not found.");
  const run = await workflow.createRunAsync();
  return run.start({ inputData: { targetLabel, minAgeDays } });
}
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
import type { RouterDecisionSchema, ScoutDecisionSchema, JudgeDecisionSchema } from './agent-schemas';

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
  // biome-ignore lint/suspicious/noExplicitAny: metadata
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

## File: packages/agent/src/labyrinth.ts
````typescript
import { QuackGraph } from '@quackgraph/graph';
import type {
  AgentConfig,
  LabyrinthArtifact,
  CorrelationResult,
  TimeContext,
  DomainConfig,
  MastraAgent
} from './types';
import { randomUUID } from 'node:crypto';
import { trace, type Span } from '@opentelemetry/api';

import { setGraphInstance } from './lib/graph-instance';
import { mastra } from './mastra';

// Tools / Helpers
import { Chronos } from './agent/chronos';
import { GraphTools } from './tools/graph-tools';
import { SchemaRegistry } from './governance/schema-registry';

// Schemas
import {
  RouterDecisionSchema,
  ScoutDecisionSchema,
  JudgeDecisionSchema
} from './agent-schemas';

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

      let foundArtifact: { artifact: LabyrinthArtifact; cursor: Cursor } | null = null;

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
              tracingOptions: { traceId, parentSpanId },
              structuredOutput: {
                schema: RouterDecisionSchema
              }
            });

            // Use structured output
            const decision = res.object;
            if (decision) {
              const valid = availableDomains.find(
                d => d.name.toLowerCase() === decision.domain.toLowerCase()
              );
              if (valid) activeDomain = decision.domain;
              this.logger.debug(`Router selected domain: ${activeDomain}`, { traceId, domain: activeDomain });
            }

            span.setAttribute('labyrinth.active_domain', activeDomain);
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
                      tracingOptions: { traceId, parentSpanId: cursorSpan.spanContext().spanId },
                      structuredOutput: {
                        schema: ScoutDecisionSchema
                      }
                    });

                    decision = res.object;

                    // Trace tool results if any (post-hoc observability)
                    if (res.toolResults && res.toolResults.length > 0) {
                      cursorSpan.setAttribute('scout.tool_calls_count', res.toolResults.length);
                      cursorSpan.addEvent('scout_tool_execution', {
                        results: JSON.stringify(res.toolResults)
                      });
                    }

                  } catch (e) {
                    if (!rootSignal.aborted) this.logger.warn('Scout failed to decide', { traceId, cursorId: cursor.id, error: e });
                    return;
                  }

                  if (!decision) return;

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
                        tracingOptions: { traceId, parentSpanId: cursorSpan.spanContext().spanId },
                        structuredOutput: {
                          schema: JudgeDecisionSchema
                        }
                      });

                      const artifact = res.object;

                      if (
                        artifact &&
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
                  } else if (decision.action === 'MATCH') {
                    // Structural Inference
                    if (!decision.pattern) return; // Should be enforced by Zod

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
                    // Using discriminated union access
                    if (decision.edgeType)
                      moves.push({ edge: decision.edgeType, conf: decision.confidence });
                    if (decision.alternativeMoves) {
                      for (const alt of decision.alternativeMoves) {
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
````
