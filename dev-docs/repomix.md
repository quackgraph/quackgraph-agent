# Directory Structure
```
packages/
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
      agent/
        src/
          agent/
            chronos.ts
            judge.ts
            metabolism.ts
            router.ts
            scout.ts
          governance/
            schema-registry.ts
          tools/
            graph-tools.ts
          index.ts
          labyrinth.ts
          types.ts
        package.json
        tsconfig.json
        tsup.config.ts
      native/
        src/
          lib.rs
        build.rs
        Cargo.toml
        index.d.ts
        index.js
        package.json
        quack-native.linux-x64-gnu.node
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
    README.md
    RFC.README.md
    tsconfig.json
    tsup.config.ts
.gitignore
LICENSE
README.md
relay.config.json
repomix.config.json
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
pub use topology::{GraphIndex, Direction};
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
    pub fn traverse_interval(&self, sources: &[String], edge_type: Option<&str>, direction: Direction, start: i64, end: i64) -> Vec<String> {
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
                        
                        // Interval Overlap Check
                        if vf < end && vt > start {
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

## File: packages/quackgraph/packages/agent/src/agent/chronos.ts
````typescript
import type { QuackGraph } from '@quackgraph/graph';
import type { CorrelationResult, EvolutionResult, SectorSummary, TimeStepDiff } from '../types';
import type { GraphTools } from '../tools/graph-tools';

export class Chronos {
  constructor(private graph: QuackGraph, private tools: GraphTools) {}

  /**
   * Finds events connected to the anchor node that occurred or overlapped
   * with the specified time window.
   */
  async findEventsDuring(
    anchorNodeId: string, 
    windowStart: Date, 
    windowEnd: Date, 
    edgeType?: string,
    direction: 'out' | 'in' = 'out'
  ): Promise<string[]> {
    return await this.graph.traverseInterval(
      [anchorNodeId], 
      edgeType, 
      direction, 
      windowStart, 
      windowEnd
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
    const count = Number(result[0]?.count || 0);
    
    return {
      anchorLabel: 'Unknown', 
      targetLabel,
      windowSizeMinutes,
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
      currentSummaryList.forEach(s => currentSummary.set(s.edgeType, s.count));

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

## File: packages/quackgraph/packages/agent/src/agent/judge.ts
````typescript
import type { AgentConfig, JudgePrompt, LabyrinthArtifact } from '../types';

export class Judge {
  constructor(private config: AgentConfig) {}

  async evaluate(promptCtx: JudgePrompt): Promise<LabyrinthArtifact | null> {
    const timeInfo = promptCtx.timeContext ? `Time Context: ${promptCtx.timeContext}` : '';
    
    const prompt = `
      You are a Judge evaluating data.
      Goal: "${promptCtx.goal}"
      ${timeInfo}
      Data: ${JSON.stringify(promptCtx.nodeContent)}
      
      Does this data answer the goal?
      Return ONLY a JSON object:
      { "isAnswer": true, "answer": "The user is...", "confidence": 0.95 }
    `;

    try {
      const raw = await this.config.llmProvider.generate(prompt);
      const jsonStr = raw.match(/\{[\s\S]*\}/)?.[0] || raw;
      // biome-ignore lint/suspicious/noExplicitAny: Weak schema for LLM response
      const result = JSON.parse(jsonStr) as any;
      
      if (result.isAnswer) {
        return {
          answer: result.answer,
          confidence: result.confidence,
          traceId: '', // Filled by caller
          sources: [] // Filled by caller
        };
      }
      return null;
    } catch (e) {
      return null;
    }
  }
}
````

## File: packages/quackgraph/packages/agent/src/agent/metabolism.ts
````typescript
import { randomUUID } from 'crypto';
import type { QuackGraph } from '@quackgraph/graph';
import type { Judge } from './judge';
import type { JudgePrompt } from '../types';

export class Metabolism {
  constructor(
    private graph: QuackGraph,
    private judge: Judge
  ) {}

  /**
   * Graph Metabolism: Summarize and Prune.
   * Identifies old, dense clusters and summarizes them into a single high-level node.
   */
  async dream(criteria: { minAgeDays: number; targetLabel: string }) {
    // 1. Identify Candidates (Nodes older than X days)
    const sql = `
      SELECT id, properties 
      FROM nodes 
      WHERE list_contains(labels, ?) 
        AND valid_from < (current_timestamp - INTERVAL ${criteria.minAgeDays} DAY)
        AND valid_to IS NULL -- Active nodes only
      LIMIT 50 -- Batch size
    `;

    const candidates = await this.graph.db.query(sql, [criteria.targetLabel]);
    if (candidates.length === 0) return;

    // 2. Synthesize (Judge)
    const judgePrompt: JudgePrompt = {
      goal: `Summarize these ${criteria.targetLabel} logs into a single concise insight.`,
      nodeContent: candidates.map((c) =>
        typeof c.properties === 'string' ? JSON.parse(c.properties) : c.properties
      ),
    };

    const artifact = await this.judge.evaluate(judgePrompt);

    if (!artifact) return; // Judge failed

    // 3. Identification of Anchor (Parent)
    const candidateIds = candidates.map((c) => c.id);
    const potentialParents = await this.graph.native.traverse(
      candidateIds,
      undefined,
      'in',
      undefined
    );

    if (potentialParents.length === 0) return; // Orphaned

    // Use the first parent found as the anchor for the summary
    const anchorId = potentialParents[0];

    // 4. Rewire & Prune
    const summaryId = `summary:${randomUUID()}`;
    const summaryProps = {
      content: artifact.answer,
      source_count: candidates.length,
      generated_at: new Date().toISOString(),
    };

    await this.graph.addNode(summaryId, ['Summary', 'Insight'], summaryProps);
    await this.graph.addEdge(anchorId, summaryId, 'HAS_SUMMARY');

    for (const id of candidateIds) {
      await this.graph.deleteNode(id);
    }
  }
}
````

## File: packages/quackgraph/packages/agent/src/agent/router.ts
````typescript
import type { AgentConfig, DomainConfig, RouterDecision } from '../types';

export class Router {
  constructor(private config: AgentConfig) {}

  /**
   * Semantic Routing: Determines which Domain governs the user's goal.
   * "Ghost Earth" Protocol: This selects the lens through which we view the graph.
   */
  async route(goal: string, domains: DomainConfig[]): Promise<RouterDecision> {
    if (domains.length <= 1) {
      // Trivial case
      return { 
        domain: domains[0]?.name || 'global', 
        confidence: 1.0, 
        reasoning: 'Only one domain available.' 
      };
    }

    const domainDescriptions = domains
      .map(d => `- "${d.name}": ${d.description}`)
      .join('\n');

    const prompt = `
      You are a Semantic Router for a Knowledge Graph.
      Goal: "${goal}"
      
      Available Domains (Lenses):
      ${domainDescriptions}
      
      Task: Select the single most relevant domain to conduct this search.
      If the goal is broad or doesn't fit specific domains, choose 'global'.
      
      Return ONLY a JSON object:
      {
        "domain": "medical",
        "confidence": 0.95,
        "reasoning": "The query mentions symptoms and medication."
      }
    `;

    try {
      const raw = await this.config.llmProvider.generate(prompt);
      const jsonStr = raw.match(/\{[\s\S]*\}/)?.[0] || raw;
      const decision = JSON.parse(jsonStr) as RouterDecision;
      
      // Validate
      const validName = domains.find(d => d.name.toLowerCase() === decision.domain.toLowerCase());
      if (!validName) {
        return { domain: 'global', confidence: 0.0, reasoning: 'LLM returned invalid domain.' };
      }
      
      return decision;
    } catch (e) {
      return { domain: 'global', confidence: 0.0, reasoning: 'Routing failed.' };
    }
  }
}
````

## File: packages/quackgraph/packages/agent/src/agent/scout.ts
````typescript
import type { AgentConfig, ScoutDecision, ScoutPrompt } from '../types';

export class Scout {
  constructor(private config: AgentConfig) {}

  async decide(promptCtx: ScoutPrompt): Promise<ScoutDecision> {
    const getHeatIcon = (heat?: number) => {
      if (!heat) return '';
      if (heat > 150) return ' 🔥 (High Heat)';
      if (heat > 50) return ' ♨️ (Warm)';
      return ' ❄️ (Cold)';
    };

    const summaryList = promptCtx.sectorSummary
      .map(s => `- ${s.edgeType}: ${s.count} nodes${getHeatIcon(s.avgHeat)}`)
      .join('\n');

    const timeInfo = promptCtx.timeContext ? `Time Context: ${promptCtx.timeContext}` : '';

    const prompt = `
      You are a Graph Scout navigating a topology.
      Goal: "${promptCtx.goal}"
      ACTIVE DOMAIN: "${promptCtx.activeDomain}"
      (Note: The graph view is filtered to show only relationships relevant to this domain).

      ${timeInfo}
      Current Node: ${promptCtx.currentNodeId} (Labels: ${promptCtx.currentNodeLabels.join(', ')})
      Path History: ${promptCtx.pathHistory.join(' -> ')}
      
      Satellite View (Available Moves in ${promptCtx.activeDomain}):
      ${summaryList}
      
      Decide your next move.
      - **Pheromones:** Edges marked with 🔥 or ♨️ have been successfully traversed before.
      - **Exploration:** If you want to explore, action: "MOVE" and specify the "edgeType".
      - **Pattern Matching:** If you suspect a specific structure (e.g. A->B->C cycle), action: "MATCH" and provide a "pattern".
        Pattern format: [{ srcVar: 0, tgtVar: 1, edgeType: 'KNOWS' }, { srcVar: 1, tgtVar: 2, edgeType: 'LIKES' }]
        (Variable 0 is current node).
      - **Reasonable Counts:** Avoid exploring >10,000 nodes unless you are zooming out.
      - **Goal Check:** If you strongly believe this current node contains the answer, action: "CHECK".
      - If stuck, "ABORT".
      - If you see multiple promising paths, you can provide "alternativeMoves".
      
      Return ONLY a JSON object:
      { 
        "action": "MOVE", 
        "edgeType": "KNOWS", 
        "confidence": 0.9, 
        "reasoning": "...",
        "alternativeMoves": [
           { "edgeType": "WORKS_WITH", "confidence": 0.5, "reasoning": "..." }
        ]
      }
    `;

    try {
      const raw = await this.config.llmProvider.generate(prompt);
      const jsonStr = raw.match(/\{[\s\S]*\}/)?.[0] || raw;
      return JSON.parse(jsonStr) as ScoutDecision;
    } catch (e) {
      return { action: 'ABORT', confidence: 0, reasoning: 'LLM Parsing Error' };
    }
  }
}
````

## File: packages/quackgraph/packages/agent/src/governance/schema-registry.ts
````typescript
import type { DomainConfig } from '../types';

export class SchemaRegistry {
  private domains = new Map<string, DomainConfig>();

  constructor() {
    // Default 'Global' domain that sees everything (fallback)
    this.register({
      name: 'global',
      description: 'Unrestricted access to the entire topology.',
      allowedEdges: [] // Empty means ALL allowed in our logic, or handled as special case
    });
  }

  register(config: DomainConfig) {
    this.domains.set(config.name.toLowerCase(), config);
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
    if (!domain) return true; // Fallback to permissive
    if (domain.name === 'global') return true;
    if (domain.allowedEdges.length === 0) return true; // Empty whitelist = all allowed? Or none? Usually global handles all.
    
    return domain.allowedEdges.includes(edgeType);
  }
}
````

## File: packages/quackgraph/packages/agent/src/tools/graph-tools.ts
````typescript
import type { QuackGraph } from '@quackgraph/graph';
import type { SectorSummary } from '../types';
import type { JsPatternEdge } from '@quackgraph/native';

export class GraphTools {
  constructor(private graph: QuackGraph) {}

  /**
   * LOD 0: Sector Scan / Satellite View
   * Returns a summary of available moves from the current nodes.
   */
  async getSectorSummary(currentNodes: string[], asOf?: number, allowedEdgeTypes?: string[]): Promise<SectorSummary[]> {
    if (currentNodes.length === 0) return [];
    
    // 1. Get Sector Stats (Count + Heat) in a single Rust call (O(1))
    const results = await this.graph.native.getSectorStats(currentNodes, asOf, allowedEdgeTypes);
    
    // 2. Sort by count (descending)
    return results.sort((a, b) => b.count - a.count);
  }

  /**
   * LOD 1: Topology Scan
   * Returns the IDs of neighbors reachable via a specific edge type.
   */
  async topologyScan(currentNodes: string[], edgeType: string, asOf?: number): Promise<string[]> {
    return this.graph.native.traverse(currentNodes, edgeType, 'out', asOf);
  }

  /**
   * LOD 1.5: Pattern Matching (Structural Inference)
   * Finds subgraphs matching a specific shape.
   */
  async findPattern(startNodes: string[], pattern: Partial<JsPatternEdge>[], asOf?: number): Promise<string[][]> {
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
  async reinforcePath(trace: { source: string; incomingEdge?: string }[]) {
    for (let i = 1; i < trace.length; i++) {
      const prev = trace[i - 1];
      const curr = trace[i];
      if (curr.incomingEdge) {
        // Boost heat by 200 (max is 255 usually, V1 logic)
        await this.graph.updateEdgeHeat(prev.source, curr.source, curr.incomingEdge, 200);
      }
    }
  }
}
````

## File: packages/quackgraph/packages/agent/src/index.ts
````typescript
export * from './labyrinth';
export * from './types';
export * from './agent/scout';
export * from './agent/judge';
export * from './agent/router';
export * from './agent/chronos';
export * from './agent/metabolism';
export * from './governance/schema-registry';
export * from './tools/graph-tools';
````

## File: packages/quackgraph/packages/agent/src/labyrinth.ts
````typescript
import { QuackGraph } from '@quackgraph/graph';
import type { 
  AgentConfig, 
  TraceStep, 
  TraceLog, 
  LabyrinthArtifact, 
  ScoutPrompt,
  JudgePrompt,
  CorrelationResult,
  TimeContext,
  DomainConfig
} from './types';
import { randomUUID } from 'crypto';

import { Scout } from './agent/scout';
import { Judge } from './agent/judge';
import { Chronos } from './agent/chronos';
import { GraphTools } from './tools/graph-tools';
import { Metabolism } from './agent/metabolism';
import { Router } from './agent/router';
import { SchemaRegistry } from './governance/schema-registry';

interface Cursor {
  id: string;
  currentNodeId: string;
  path: string[]; // History of node IDs
  traceHistory: string[]; // Summary of actions for context
  stepCount: number;
  confidence: number;
  parentId?: number; // stepId of the action that led here
  lastEdgeType?: string; // Edge taken to reach currentNodeId
}

export class Labyrinth {
  private traces = new Map<string, TraceLog>();
  
  public scout: Scout;
  public judge: Judge;
  public chronos: Chronos;
  public tools: GraphTools;
  public metabolism: Metabolism;
  public router: Router;
  public registry: SchemaRegistry;

  constructor(
    private graph: QuackGraph, 
    private config: AgentConfig
  ) {
    this.scout = new Scout(config);
    this.judge = new Judge(config);
    this.tools = new GraphTools(graph);
    this.chronos = new Chronos(graph, this.tools);
    this.metabolism = new Metabolism(graph, this.judge);
    
    // Governance
    this.router = new Router(config);
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
    const traceId = randomUUID();
    const startTime = Date.now();
    
    // --- Phase 0: Context Firewall (Routing) ---
    // Before we start walking, decide WHICH domain we are walking in.
    const availableDomains = this.registry.getAllDomains();
    const route = await this.router.route(goal, availableDomains);
    const activeDomain = route.domain;
    
    // Resolve effective timestamp (passed context > graph context > current)
    const effectiveAsOf = timeContext?.asOf || this.graph.context.asOf;
    const asOfTs = effectiveAsOf ? effectiveAsOf.getTime() * 1000 : undefined;

    const timeDesc = effectiveAsOf 
      ? `As of: ${effectiveAsOf.toISOString()}` 
      : (timeContext?.windowStart ? `Window: ${timeContext.windowStart.toISOString()} - ${timeContext.windowEnd?.toISOString()}` : undefined);
    
    const log: TraceLog = {
      traceId,
      goal,
      activeDomain,
      startTime,
      steps: [],
      outcome: 'ABORTED' // Default
    };
    
    this.traces.set(traceId, log);

    let startNodes: string[] = [];

    // Vector Genesis
    if (typeof start === 'object' && 'query' in start) {
      if (!this.config.embeddingProvider) {
        throw new Error("Vector Genesis requires an embeddingProvider in AgentConfig.");
      }
      const vector = await this.config.embeddingProvider.embed(start.query);
      // Get top 3 relevant start nodes
      const matches = await this.graph.match([]).nearText(vector).limit(3).select();
      startNodes = matches.map(m => m.id);
    } else {
      startNodes = [start as string];
    }

    // Initialize Root Cursor
    let cursors: Cursor[] = startNodes.map(nodeId => ({
      id: randomUUID(),
      currentNodeId: nodeId,
      path: [nodeId],
      traceHistory: [`[ROUTER] Selected domain: ${activeDomain}`],
      stepCount: 0,
      confidence: 1.0
    }));

    const maxHops = this.config.maxHops || 10;
    const maxCursors = this.config.maxCursors || 3;
    let globalStepCounter = 0;
    let foundArtifact: LabyrinthArtifact | null = null;

    // Main Loop: While we have active cursors and haven't found the answer
    while (cursors.length > 0 && !foundArtifact) {
      const nextCursors: Cursor[] = [];
      const processingPromises: Promise<void>[] = [];
      
      // We wrap the iteration to allow for "Race" logic (checking foundArtifact)
      
      for (const cursor of cursors) {
        if (foundArtifact) break; // Early exit check

        const task = async () => {
          if (foundArtifact) return; // Double check inside async
          
          // Pruning: Max Depth
          if (cursor.stepCount >= maxHops) return;

          // 1. Context Awareness (LOD 1)
          const nodeMeta = await this.graph.match([])
            .where({ id: cursor.currentNodeId })
            .select(n => ({ id: n.id, labels: n.labels }));
          
          if (nodeMeta.length === 0) return; // Node lost/deleted
          const currentNode = nodeMeta[0];

          // 2. Sector Scan (LOD 0) - Enhanced Satellite View
          
          // CONTEXT FIREWALL: Push filtering down to Rust
          const domainConfig = this.registry.getDomain(activeDomain);
          let allowedEdges: string[] | undefined;
          
          if (domainConfig && domainConfig.name !== 'global' && domainConfig.allowedEdges.length > 0) {
            allowedEdges = domainConfig.allowedEdges;
          }

          const sectorSummary = await this.tools.getSectorSummary([cursor.currentNodeId], asOfTs, allowedEdges);

          // 3. Ask Scout
          const prompt: ScoutPrompt = {
            goal,
            activeDomain,
            currentNodeId: currentNode.id,
            currentNodeLabels: currentNode.labels || [],
            sectorSummary,
            pathHistory: cursor.path,
            timeContext: timeDesc
          };

          const decision = await this.scout.decide(prompt);
          
          // Register Step
          const currentStepId = globalStepCounter++;
          const step: TraceStep = {
            stepId: currentStepId,
            parentId: cursor.parentId,
            cursorId: cursor.id,
            nodeId: cursor.currentNodeId,
            source: cursor.currentNodeId,
            incomingEdge: cursor.lastEdgeType,
            action: decision.action as 'MOVE' | 'CHECK' | 'MATCH',
            decision,
            reasoning: decision.reasoning,
            timestamp: Date.now()
          };
          log.steps.push(step);
          cursor.traceHistory.push(`[${decision.action}] ${decision.reasoning}`);

          // 4. Handle Decision
          if (decision.action === 'CHECK') {
            // LOD 2: Content Retrieval
            const content = await this.tools.contentRetrieval([cursor.currentNodeId]);
            
            // Ask Judge
            const judgePrompt: JudgePrompt = { 
              goal, 
              nodeContent: content,
              timeContext: timeDesc 
            };
            const artifact = await this.judge.evaluate(judgePrompt);
            
            if (artifact && artifact.confidence >= (this.config.confidenceThreshold || 0.7)) {
              // Success found by this cursor!
              artifact.traceId = traceId;
              artifact.sources = [cursor.currentNodeId];
              
              // Set global found flag to stop other cursors
              foundArtifact = { type: 'FOUND', artifact, finalStepId: currentStepId } as any; 
              return;
            } 
            
            // If Judge rejects, this path ends here.
            return;

          } else if (decision.action === 'MATCH' && decision.pattern) {
            // Structural Inference
            const matches = await this.tools.findPattern([cursor.currentNodeId], decision.pattern, asOfTs);
            
            if (matches.length > 0) {
               // Pattern found! Jump to the end of the matched paths.
               // matches is Vec<Vec<string>> (list of paths)
               // We take up to 3 matches to spawn cursors
               const foundPaths = matches.slice(0, 3);
               
               for (const path of foundPaths) {
                  const endNode = path[path.length - 1]; // Jump to end of pattern
                  if (endNode) {
                     nextCursors.push({
                        id: randomUUID(),
                        currentNodeId: endNode,
                        path: [...cursor.path, ...path.slice(1)], // Append path (skipping start which is current)
                        traceHistory: [...cursor.traceHistory, `[MATCH] Pattern Found: ${JSON.stringify(decision.pattern)}`],
                        stepCount: cursor.stepCount + path.length - 1, // Advance step count
                        confidence: cursor.confidence * 0.9, // Slight penalty for jump
                        parentId: currentStepId,
                        lastEdgeType: 'MATCH_JUMP'
                     });
                  }
               }
            }
            return;

          } else if (decision.action === 'MOVE') {
            const moves = [];
            
            // Primary Move
            if (decision.edgeType) {
              moves.push({ edge: decision.edgeType, conf: decision.confidence });
            }
            
            // Alternative Moves (Speculative Execution)
            if (decision.alternativeMoves) {
              for (const alt of decision.alternativeMoves) {
                moves.push({ edge: alt.edgeType, conf: alt.confidence });
              }
            }

            // Process Moves
            for (const move of moves) {
              if (move.conf < (this.config.confidenceThreshold || 0.2)) continue;
              
              // Double check governance (Scout hallucination check)
              if (!this.registry.isEdgeAllowed(activeDomain, move.edge)) {
                // If Scout tries to move on a forbidden edge (despite firewall in prompt), block it.
                continue;
              }

              // LOD 1: Topology Scan
              const nextNodes = await this.tools.topologyScan([cursor.currentNodeId], move.edge, asOfTs);
              
              if (nextNodes.length > 0) {
                // If multiple targets, we branch, taking up to 3 to prevent explosion
                const targets = nextNodes.slice(0, 3); 

                for (const target of targets) {
                  if (move.edge === decision.edgeType && target === nextNodes[0]) {
                     step.target = target;
                  }

                  nextCursors.push({
                    id: randomUUID(),
                    currentNodeId: target,
                    path: [...cursor.path, target],
                    traceHistory: [...cursor.traceHistory],
                    stepCount: cursor.stepCount + 1,
                    confidence: cursor.confidence * move.conf,
                    parentId: currentStepId,
                    lastEdgeType: move.edge
                  });
                }
              }
            }
            return;
          }
        };
        
        processingPromises.push(task());
      }

      await Promise.all(processingPromises);

      // Check results
      // biome-ignore lint/suspicious/noExplicitAny: hack for race result
      const res = foundArtifact as any;
      if (res && res.type === 'FOUND') {
        log.outcome = 'FOUND';
        const artifact = res.artifact as LabyrinthArtifact;
        log.finalArtifact = artifact;
        
        if (res.finalStepId !== undefined) {
          const winningTrace = this.reconstructPath(log.steps, res.finalStepId);
          await this.tools.reinforcePath(winningTrace);
        }
        
        return artifact;
      }

      // Pruning / Management
      // We sort by confidence and take top N
      nextCursors.sort((a, b) => b.confidence - a.confidence);
      cursors = nextCursors.slice(0, maxCursors);
    }

    if (log.outcome !== 'FOUND') {
      log.outcome = 'EXHAUSTED';
    }
    
    return null;
  }

  getTrace(traceId: string): TraceLog | undefined {
    return this.traces.get(traceId);
  }

  private reconstructPath(allSteps: TraceStep[], finalStepId: number): TraceStep[] {
    const path: TraceStep[] = [];
    let currentId: number | undefined = finalStepId;
    
    const stepMap = new Map<number, TraceStep>();
    for (const s of allSteps) stepMap.set(s.stepId, s);

    while (currentId !== undefined) {
      const step = stepMap.get(currentId);
      if (step) {
        path.unshift(step);
        currentId = step.parentId;
      } else {
        break;
      }
    }
    return path;
  }

  // --- Wrapper for Chronos ---
  
  async analyzeCorrelation(anchorNodeId: string, targetLabel: string, windowMinutes: number): Promise<CorrelationResult> {
    return this.chronos.analyzeCorrelation(anchorNodeId, targetLabel, windowMinutes);
  }
}
````

## File: packages/quackgraph/packages/agent/src/types.ts
````typescript
export enum ZoomLevel {
  SECTOR = 0,    // Ghost/Satellite View: Available Moves (Schema)
  TOPOLOGY = 1,  // Drone View: Structural Hops (IDs only)
  CONTENT = 2    // Street View: Full JSON Data
}

export interface AgentConfig {
  llmProvider: {
    generate: (prompt: string) => Promise<string>;
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
  allowedEdges: string[]; // Whitelist of edge types visible to the Scout
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

export interface TraceStep {
  stepId: number;
  parentId?: number;
  cursorId: string;
  incomingEdge?: string;
  nodeId: string; // Source node where decision was made
  source: string;
  target?: string; // Resulting node if MOVE
  action: 'MOVE' | 'CHECK' | 'MATCH';
  decision: ScoutDecision;
  reasoning: string;
  timestamp: number;
}

export interface TraceLog {
  traceId: string;
  goal: string;
  activeDomain: string;
  startTime: number;
  steps: TraceStep[];
  outcome: 'FOUND' | 'EXHAUSTED' | 'ABORTED';
  finalArtifact?: LabyrinthArtifact;
}

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
````

## File: packages/quackgraph/packages/agent/package.json
````json
{
  "name": "@quackgraph/agent",
  "version": "0.1.0",
  "main": "dist/index.js",
  "module": "dist/index.mjs",
  "types": "dist/index.d.ts",
  "exports": {
    ".": {
      "import": "./dist/index.mjs",
      "require": "./dist/index.js",
      "types": "./dist/index.d.ts"
    }
  },
  "scripts": {
    "build": "tsup",
    "dev": "tsup --watch",
    "clean": "rm -rf dist"
  },
  "dependencies": {
    "@quackgraph/graph": "workspace:*",
    "@quackgraph/native": "workspace:*"
  },
  "devDependencies": {
    "typescript": "^5.0.0",
    "tsup": "^8.0.0"
  }
}
````

## File: packages/quackgraph/packages/agent/tsconfig.json
````json
{
  "extends": "../../../tsconfig.json",
  "include": ["src/**/*"],
  "compilerOptions": {
    "outDir": "dist",
    "rootDir": "src"
  }
}
````

## File: packages/quackgraph/packages/agent/tsup.config.ts
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

## File: packages/quackgraph/packages/native/src/lib.rs
````rust
#![deny(clippy::all)]

use napi::bindgen_prelude::*;
use napi_derive::napi;
use quack_core::{matcher::{Matcher, PatternEdge}, GraphIndex, Direction};
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
    #[napi]
    pub fn traverse_interval(&self, sources: Vec<String>, edge_type: Option<String>, direction: Option<String>, start: f64, end: f64) -> Vec<String> {
        let dir = match direction.as_deref() {
            Some("in") | Some("IN") => Direction::Incoming,
            _ => Direction::Outgoing,
        };
        let s = (start * 1000.0) as i64;
        let e = (end * 1000.0) as i64;
        self.inner.traverse_interval(&sources, edge_type.as_deref(), dir, s, e)
    }

    /// Performs a recursive traversal (BFS) with depth bounds.
    /// Returns unique node IDs reachable within [min_depth, max_depth].
    #[napi(js_name = "traverseRecursive")]
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
  getSectorStats(sources: Array<string>, asOf?: number | undefined | null): Array<JsSectorStat>
  /**
   * Performs a single-hop traversal (bfs-step).
   * Returns unique neighbor IDs.
   */
  traverse(sources: Array<string>, edgeType?: string | undefined | null, direction?: string | undefined | null, asOf?: number | undefined | null): Array<string>
  /**
   * Performs a traversal finding all neighbors connected via edges that overlap
   * with the specified time window [start, end).
   * Timestamps are in milliseconds (JS standard).
   */
  traverseInterval(sources: Array<string>, edgeType: string | undefined | null, direction: string | undefined | null, start: number, end: number): Array<string>
  /**
   * Performs a recursive traversal (BFS) with depth bounds.
   * Returns unique node IDs reachable within [min_depth, max_depth].
   */
  traverseRecursive(sources: Array<string>, edgeType?: string | undefined | null, direction?: string | undefined | null, minDepth?: number | undefined | null, maxDepth?: number | undefined | null, asOf?: number | undefined | null): Array<string>
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

  async traverseInterval(sources: string[], edgeType: string | undefined, direction: 'out' | 'in' = 'out', start: Date, end: Date): Promise<string[]> {
    const s = start.getTime();
    const e = end.getTime();
    // If end < start, return empty
    if (e <= s) return [];
    return this.native.traverseInterval(sources, edgeType, direction, s, e);
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
  "name": "quack-graph",
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

## File: packages/quackgraph/README.md
````markdown
# QuackGraph 🦆🕸️

[![npm version](https://img.shields.io/npm/v/quack-graph.svg?style=flat-square)](https://www.npmjs.com/package/quack-graph)
[![Build Status](https://img.shields.io/github/actions/workflow/status/your-repo/quack-graph/ci.yml?style=flat-square)](https://github.com/your-repo/quack-graph/actions)
[![Runtime: Bun](https://img.shields.io/badge/Runtime-Bun%20%2F%20Node-black.svg?style=flat-square)](https://bun.sh)
[![Engine: Rust](https://img.shields.io/badge/Accelerator-Rust%20(CSR)-orange.svg?style=flat-square)](https://www.rust-lang.org/)
[![Storage: DuckDB](https://img.shields.io/badge/Storage-DuckDB-brightgreen.svg?style=flat-square)](https://duckdb.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

> **The Embedded Graph Analytics Engine.**
>
> **Postgres is for records. QuackGraph is for relationships.**
>
> QuackGraph is a **serverless, infrastructure-less** graph index that runs alongside your app. It combines **DuckDB** (Columnar Storage) with a **Rust/Wasm CSR Index** (O(1) Traversal) via **Zero-Copy Apache Arrow**.
>
> No Docker containers. No JVM. Just `npm install` and raw speed.

---

## 📖 Table of Contents

1.  [**Why QuackGraph? (The Pitch)**](#-why-quackgraph-the-pitch)
2.  [**The Architecture: A "Split-Brain" Engine**](#-the-architecture-a-split-brain-engine)
3.  [**Installation**](#-installation)
4.  [**Quick Start (5 Minutes)**](#-quick-start-5-minutes)
5.  [**Core Concepts**](#-core-concepts)
    *   [Schemaless & Gradual Typing](#1-schemaless--gradual-typing)
    *   [GraphRAG (Vector Search)](#2-graphrag-vector-search)
    *   [Temporal Time-Travel](#3-temporal-time-travel)
    *   [Complex Patterns & Recursion](#4-complex-patterns--recursion)
    *   [Declarative Mutations](#5-declarative-mutations)
6.  [**Advanced Usage & Performance Tuning**](#-advanced-usage--performance-tuning)
    *   [Property Promotion](#property-promotion-json--native)
    *   [Topology Snapshots](#topology-snapshots-for-instant-boot)
    *   [Server-Side Aggregations](#server-side-aggregations)
    *   [Cypher Compatibility](#cypher-compatibility)
7.  [**Runtime Targets: Native vs. Edge**](#-runtime-targets-native-vs-edge)
8.  [**Comparison with Alternatives**](#-comparison-with-alternatives)
9.  [**Known Limits & Trade-offs**](#-known-limits--trade-offs)
10. [**Contributing**](#-contributing)
11. [**Roadmap**](#-roadmap)

---

## 💡 Why QuackGraph?

**The "SQLite for Graphs" Moment.**

Enterprises run Neo4j Clusters. Startups and Local-First apps don't have that luxury. You shouldn't need to deploy a heavy Java-based server just to query "friends of friends" or build a RAG pipeline.

QuackGraph is **CQRS in a box**:
1.  **Ingest:** Data lands in **DuckDB**. It's cheap, ACID-compliant, and handles millions of rows on a laptop.
2.  **Index:** We project the topology into a **Rust Compressed Sparse Row (CSR)** structure in RAM.
3.  **Query:** Graph traversals happen in nanoseconds (memory pointers), while heavy aggregations happen in DuckDB (vectorized SQL).

**Use Cases:**
*   **GraphRAG:** Combine Vector Search (HNSW) with Knowledge Graph traversal in a single process.
*   **Fraud Detection:** Detect cycles and rings in transaction logs without network latency.
*   **Local-First SaaS:** Ship complex analytics in Electron apps or Edge workers.

---

## 📐 Architecture: Zero-Copy Hybrid Engine

QuackGraph is not a database replacement; it is a **Read-Optimized View**. It leverages **Apache Arrow** to stream data from Disk to RAM at ~1GB/s.

```ascii
[ Your App (Bun / Node / Wasm) ]
     │
     ▼
[ QuackGraph DX Layer (TypeScript) ]
     │
     ├── Writes ───────────────┐
     │                         ▼
     │                 [ DuckDB Storage ] (Persistent Source of Truth)
     │                 (Parquet / JSON / WAL)
     │                         │
     ├── Reads (Filters) ◄─────┤
     │                         │
     │                 (Arrow IPC Stream for Hydration)
     │                         ▼
     └── Reads (Hops) ◄── [ Rust Index ] (Transient In-Memory Cache)
                          (CSR Topology)
```

1.  **DuckDB is King:** All writes (`addNode`, `addEdge`) go immediately and atomically to DuckDB.
2.  **Rust is a View:** The In-Memory Graph Index is a *read-optimized, transient view* of the data on disk.
3.  **Hydration:** On startup, we stream edges from DuckDB to Rust via Arrow IPC (~1M edges/sec).
4.  **Consistency:** If the process crashes, the RAM index is gone. No data loss occurs because the data is safely in `.duckdb`.

---

## 📦 Installation

Choose your runtime target.

### 🏎️ Native (Backend / CLI)
*Best for: Bun, Node.js, Electron, Tauri.*
Uses `napi-rs` for native C++ performance.

```bash
bun add quack-graph
```

### 🌍 Edge (Serverless / Browser)
*Best for: Cloudflare Workers, Vercel Edge, Local-First Web Apps.*
Uses WebAssembly.

```bash
bun add quack-graph @duckdb/duckdb-wasm apache-arrow
```

---

## ⚡ The API: Graph Topology meets SQL Analytics

Stop writing 50-line `WITH RECURSIVE` SQL queries.
QuackGraph gives you a Fluent TypeScript API for the topology, but lets you drop into raw SQL for the heavy lifting.

**The "Hybrid" Query Pattern:**
1.  **Graph Layer:** Use Rust to traverse hops instantly.
2.  **SQL Layer:** Use DuckDB to aggregate the results.

```typescript
import { QuackGraph } from 'quack-graph';
const g = new QuackGraph('./supply-chain.duckdb');

// Scenario: "Find all downstream products affected by a bad Lithium batch,
// and calculate the total inventory value at risk."

const results = await g
  // 1. Start: DuckDB Index Scan
  .match(['Material'])
  .where({ batch: 'BAD-BATCH-001' })

  // 2. Traversal: Rust In-Memory CSR (Nanoseconds)
  // Find everything this material flows into, up to 10 hops deep
  .out('PART_OF').depth(1, 10)

  // 3. Filter: Apply logic to the found nodes
  .node(['Product'])
  .where({ status: 'active' })

  // 4. Analytics: Push aggregation down to DuckDB (Zero Data Transfer)
  // We can write raw SQL inside .select()!
  .select(`
    id,
    properties->>'name' as product_name,
    (properties->>'price')::FLOAT * (properties->>'stock')::INT as value_at_risk
  `);

console.table(results);
/*
┌────────────┬──────────────┬───────────────┐
│ id         │ product_name │ value_at_risk │
├────────────┼──────────────┼───────────────┤
│ prod:ev_1  │ Tesla Model3 │ 1500000       │
│ prod:bat_x │ PowerWall    │ 45000         │
└────────────┴──────────────┴───────────────┘
*/
```

---

## 🧠 Core Concepts

### 1. Schemaless & Gradual Typing
Start with `any`. Harden with `Zod`. QuackGraph stores properties as a `JSON` column in DuckDB, allowing instant iteration. When you need safety, bind a Schema.

```typescript
import { z } from 'zod';
const UserSchema = z.object({ name: z.string(), role: z.enum(['Admin', 'User']) });

const g = new QuackGraph('db.duckdb').withSchemas({ User: UserSchema });
// TypeScript now provides strict autocomplete and runtime validation
```

### 2. GraphRAG (Vector Search)
Build **Local-First AI** apps. QuackGraph bundles `duckdb_vss` (HNSW Indexing). Your graph *is* your vector store.

```typescript
// Find documents similar to [Query], then find who wrote them
const authors = await g
  .nearText(['Document'], queryVector, { limit: 10 }) // HNSW Search
  .in('AUTHORED_BY')                                  // Graph Hop
  .node(['User'])
  .select(u => u.name);
```

### 3. Temporal Time-Travel
The database is **Append-Only**. We never overwrite data; we version it. This gives you Git-like history for your data.

```typescript
// Oops, someone deleted the edges? Query the graph as it existed 10 minutes ago.
const snapshot = g.asOf(new Date(Date.now() - 10 * 60 * 1000));
const count = await snapshot.match(['User']).count();
```

### 4. Complex Patterns & Recursion
Match Neo4j's expressiveness with fluent ergonomics.

**Variable-Length Paths (Recursive):**
```typescript
// Find friends of friends (1 to 5 hops away)
const network = await g.match(['User'])
  .where({ id: 'Alice' })
  .out('KNOWS').depth(1, 5)
  .select(u => u.name);
```

**Pattern Matching (Isomorphism):**
```typescript
// Find a "Triangle" (A knows B, B knows C, C knows A)
const triangles = await g.match(['User']).as('a')
  .out('KNOWS').as('b')
  .out('KNOWS').as('c')
  .matchEdge('c', 'a', 'KNOWS') // Close the loop
  .return(row => ({
    a: row.a.name,
    b: row.b.name,
    c: row.c.name
  }));
```

### 5. Declarative Mutations (Upserts)
Don't write race-condition-prone check-then-insert code. We provide atomic `MERGE` semantics equivalent to Neo4j.

```typescript
// Idempotent Ingestion
const userId = await g.mergeNode('User', { email: 'alice@corp.com' })
  .match({ email: 'alice@corp.com' })   // Look up by unique key
  .set({ last_seen: new Date() })       // Update if exists
  .run();
```

### 6. Batch Ingestion
For high-throughput scenarios, use batch operations to minimize transaction overhead.

```typescript
// Insert 10,000 nodes in one transaction
await g.addNodes([
  { id: 'u:1', labels: ['User'], properties: { name: 'Alice' } },
  { id: 'u:2', labels: ['User'], properties: { name: 'Bob' } }
]);

// Insert 50,000 edges
await g.addEdges([
  { source: 'u:1', target: 'u:2', type: 'KNOWS', properties: { since: 2022 } }
]);
```

---

## 🛠️ Advanced Usage & Performance Tuning

### Property Promotion (JSON -> Native)
Filtering inside large JSON blobs is slower than native columns. QuackGraph can materialize hot fields for you.

```typescript
// Background migration: pulls 'age' out of the JSON blob into a native INTEGER column for 50x faster reads.
await g.optimize.promoteProperty('User', 'age', 'INTEGER');
```

### Topology Snapshots (for Instant Boot)
The "Hydration" phase can be slow for huge graphs. You can snapshot the in-memory Rust index to disk.

```typescript
// Save the RAM index to disk
await g.optimize.saveTopologySnapshot('./topology.snapshot');

// On next boot, load the snapshot instead of re-reading from DuckDB
const g = new QuackGraph('./data.duckdb', { topologySnapshot: './topology.snapshot' });
```

### Server-Side Aggregations
Don't pull data back to JS just to count it. Push the math to DuckDB.

```typescript
// "MATCH (u:User) RETURN u.city, count(u) as pop"
const stats = await g.match(['User'])
  .groupBy(u => u.city)
  .count()
  .as('pop')
  .run();
```

### Cypher Compatibility
For easy migration and interoperability, you can run raw Cypher queries.

```typescript
// (Roadmap v1.0)
const results = await g.query(`
  MATCH (u:User {name: 'Alice'})-[:MENTORS]->(mentee:User)
  WHERE mentee.age < 30
  RETURN mentee.name
`);```

---

## 🎯 Runtime Targets: Native vs. Edge

| Feature | **Native (Bun/Node)** | **Edge (Wasm)** |
| :--- | :--- | :--- |
| **Engine** | Rust (Napi-rs) | Rust (Wasm) |
| **Performance** | 🚀 **Highest** | 🐇 Fast |
| **Cold Start** | ~50ms | ~400ms (Wasm boot) |
| **Max Memory** | System RAM | ~128MB (CF Workers) |
| **Best For** | Backends, CLI, Desktop | Serverless, Browser, Local-First |

---

## 🆚 Comparison with Alternatives

| Feature | QuackGraph 🦆 | Neo4j / TigerGraph | Raw SQL (Postgres/DuckDB) |
| :--- | :--- | :--- | :--- |
| **Deployment** | **`npm install`** | Docker / K8s Cluster | Docker / RDS |
| **Architecture** | **Embedded Library** | Standalone Server | Database Engine |
| **Latency** | **Nanoseconds (In-Proc)** | Milliseconds (Network) | Microseconds (IO) |
| **Vector RAG**| **Native (HNSW)** | Plugin Required | Extension (pgvector) |
| **Traversal** | **O(1) RAM Pointers** | O(1) RAM Pointers | O(log n) Index Joins |
| **Cost** | **$0 / Compute Only** | $$ License / Cloud | $ Instance Cost |

---

## ⚠️ Known Limits & Trade-offs

1.  **Memory Wall (Edge):**
    *   On Cloudflare Workers (128MB limit), the Graph Index can hold **~200k edges** before OOM.
    *   *Workaround:* Use integer IDs (`1001` vs `"user_uuid_v4"`) to save ~60% RAM.
2.  **Concurrency:**
    *   DuckDB is **Single-Writer**. This is not for high-concurrency OLTP (e.g., a Banking Ledger).
    *   It is designed for **Read-Heavy / Analytic** workloads (RAG, Recommendations, Dashboards).
3.  **Deep Pattern Matching:**
    *   While we support basic isomorphism (triangles, rings), extremely large subgraph queries (>10 node patterns) are computationally expensive in any engine. We optimize for "OLTP-style" pattern matching (small local patterns) rather than whole-graph analytics.

---

## 🤝 Contributing

We are building the standard library for Graph Data in TypeScript.
This project is a Bun Workspace monorepo.

1.  **Install:** `bun install`
2.  **Build Native:** `cd packages/native && bun build`
3.  **Run Tests:** `bun test`

All contributions are welcome. Please open an issue to discuss your ideas.

---

## 🗓️ Roadmap

*   ✅ **v0.1:** Core Engine (Native + Wasm).
*   🟡 **v0.5:** **Recursion & Patterns.** Rust-side VF2 solver and Recursive DFS.
*   ⚪️ **v1.0:** **Auto-Columnarization.** Background job that detects hot JSON fields and promotes them to native DuckDB columns.
*   ⚪️ **v1.1:** **Cypher Parser.** `g.query('MATCH (n)-[:KNOWS]->(m) RETURN m')` for easy migration.
*   ⚪️ **v1.2:** **Replication.** `g.sync('s3://bucket/graph')` for multi-device sync.

---

## 📄 License

**MIT**
````

## File: packages/quackgraph/RFC.README.md
````markdown
# RCC.README.md 🏗️

> **Project:** QuackGraph (Core Engine)
> **Stack:** Bun (Runtime) + Rust (Compute) + DuckDB (Storage)
> **Architecture:** "Split-Brain" (In-Memory CSR + On-Disk Columnar)
> **License:** MIT

---

## 1. The Core Philosophy (Engineering Constraints)

To maintain performance and the "Embedded" promise, we strictly adhere to these constraints:

1.  **NO Garbage Collection in the Hot Path:** The traversal index must live in Rust `Vec<T>` (Native) or Wasm Linear Memory. We never store topology in JS Objects (`{ id: 'a', neighbors: [...] }`) to avoid V8 GC pauses.
2.  **NO Random Disk I/O:** Topology lives in RAM. Disk is only for sequential columnar scans (DuckDB).
3.  **NO Serialization Overhead:** We do not serialize JSON between DuckDB and Rust. We use **Apache Arrow** (IPC) pointers for Zero-Copy transfer.
4.  **DuckDB is the Source of Truth:** If the process crashes, RAM is lost. On restart, we Hydrate RAM from DuckDB. Rust is a *Transient Cache*.
5.  **Append-Only Storage:** We never `UPDATE` or `DELETE` rows in DuckDB. We insert new versions with `valid_from` timestamps.

---

## 2. Monorepo Structure

We use a **Bun Workspace** combined with a **Cargo Workspace**.

```text
/quack-graph
├── /packages
│   ├── /quack-graph        # Public TS API (The entry point)
│   ├── /native             # Napi-rs bindings (Node/Bun glue)
│   └── /wasm               # Wasm-bindgen bindings (Browser/Edge glue)
├── /crates
│   └── /quack_core         # Shared Rust Logic (The "Brain")
│       ├── /src/topology.rs  # CSR Index
│       └── /src/interner.rs  # String <-> u32
├── /benchmarks             # Performance testing suite
├── Cargo.toml              # Rust Workspace
├── package.json            # Bun Workspace
└── bun.lockb
```

---

## 3. The Rust Core Spec (`/crates/quack_core`)

This code must compile to both **CDYLIB** (Native) and **WASM32-UNKNOWN-UNKNOWN** (Edge).

### 3.1 The String Interner
Since DuckDB uses `TEXT` IDs (UUIDs), but fast traversal requires `u32` integers, we map them.

*   **Struct:** `BiMap` (Bidirectional Map).
*   **Forward:** `HashMap<String, u32>` (O(1) lookup).
*   **Reverse:** `Vec<String>` (Index lookup).
*   **Edge constraint:** On Cloudflare, `HashMap` overhead is significant.
    *   *Optimization V2:* Use a Double-Array Trie or enforce integer IDs for large graphs.

### 3.2 The Topology (Mutable CSR)
We use a hybrid Adjacency List that acts like a Compressed Sparse Row (CSR).

```rust
pub struct GraphIndex {
    // Forward Graph: Source u32 -> [(Target u32, Type u8)]
    // We use Vec<Vec<>> for O(1) appends during hydration.
    // Ideally, we compact this to flat Vec<u32> (CSR) after hydration.
    outgoing: Vec<Vec<(u32, u8)>>, 
    
    // Reverse Graph: Target u32 -> [(Source u32, Type u8)]
    // Required for .in() traversals
    incoming: Vec<Vec<(u32, u8)>>,
    
    // Bitmask for soft-deleted nodes (to avoid checking DuckDB for every hop)
    tombstones: BitVec,
}

### 3.3 The Graph Solver (Pattern Matching)
To match Neo4j's isomorphism capabilities (e.g., finding triangles or specific shapes), we implement a **Subgraph Isomorphism Solver** in Rust.
*   **Algorithm:** VF2 or Backtracking DFS with state pruning.
*   **Input:** A query graph (small topology of what we look for).
*   **Execution:**
    1.  Candidate Selection: Identify potential start nodes based on labels/properties (filtered by DuckDB).
    2.  Matching: Rust engine expands candidates, checking structural constraints.
    3.  Output: A set of matching path tuples `[(NodeA, NodeB, NodeC), ...]`.

### 3.4 Recursive Engine
To support `MATCH (n)-[:KNOWS*1..5]->(m)`, the CSR index must support depth-bounded traversals.
*   **Function:** `traverse_recursive(starts, type, min_depth, max_depth)`.
*   **Visited Set:** Essential to prevent cycles in infinite recursions.
*   **Memory:** Using a bitset for `visited` is efficient given we intern everything to `u32`.
```

---

## 4. The Storage Spec (DuckDB)

We treat DuckDB as a **Log-Structured Merge Tree (LSM)** style store.

### 4.1 Schema Definition

```sql
-- NODES
CREATE TABLE nodes (
    row_id UBIGINT PRIMARY KEY, -- Internal sequence for fast joins
    id TEXT NOT NULL,           -- Public ID
    labels TEXT[],              -- Multi-label support
    properties JSON,            -- Schemaless payload
    embedding FLOAT[1536],      -- Vector (HNSW)
    
    -- TEMPORAL COLUMNS
    valid_from TIMESTAMP DEFAULT current_timestamp,
    valid_to TIMESTAMP DEFAULT NULL -- NULL means 'Active'
);

-- EDGES
CREATE TABLE edges (
    source TEXT NOT NULL,
    target TEXT NOT NULL,
    type TEXT NOT NULL,
    properties JSON,
    valid_from TIMESTAMP DEFAULT current_timestamp,
    valid_to TIMESTAMP DEFAULT NULL
);
```

### 4.2 The Hydration Flow (Critical Path)
Startup time is the #1 KPI.

1.  **TS Layer:** Calls `duckdb.stream("SELECT source, target, type FROM edges WHERE valid_to IS NULL")`.
2.  **TS Layer:** Receives **Apache Arrow RecordBatch** (C++ memory pointer).
3.  **Bridge:** Passes the pointer to Rust via Napi/Wasm.
4.  **Rust Layer:**
    *   Reads `source` column (String View). Interns to `u32`.
    *   Reads `target` column (String View). Interns to `u32`.
    *   Updates `GraphIndex`.
5.  **Target Speed:** 1 Million Edges / second processing rate.

---

## 5. The Query Planner (`/packages/quack-graph`)

The TypeScript layer compiles the Fluent API into an **Execution Plan (AST V2)**.

**User Query:**
```typescript
g.match(['User']).as('a')
 .out('KNOWS').as('b')
 .out('KNOWS').as('c')
 .matchEdge('c', 'a', 'KNOWS') // Cycle
 .return('a', 'b', 'c')
```

**Compilation Pipeline (The "Solver" Model):**

1.  **Symbolic AST:** We track aliases (`a`, `b`) and their relationships.
2.  **Hybrid Optimization:**
    *   **Filter Pushdown:** DuckDB narrows the candidate sets for `a`, `b`, and `c` based on properties.
    *   **Pattern Extraction:** The topological constraints (`a->b`, `b->c`, `c->a`) are extracted into a "Pattern Query" for Rust.
3.  **Execution (Iterative Solver):**
    *   **Step 1 (Candidates):** DuckDB fetches IDs for start nodes.
    *   **Step 2 (Rust Solver):** The Rust engine runs VF2/Backtracking on the in-memory graph to find valid tuples `(id_a, id_b, id_c)`.
    *   **Step 3 (Projection):** The resulting tuples are joined back with DuckDB to fetch properties (`RETURN a.name, c.age`).

**Aggregations & Grouping:**
Aggregations (`count`, `avg`, `collect`) are pushed down to DuckDB's SQL engine on the final result set.

---

## 6. The Native Bridge (`/packages/native`)

We use `napi-rs` to expose Rust to Bun/Node.

```rust
// packages/native/src/lib.rs
use napi_derive::napi;
use quack_core::GraphIndex;

#[napi]
pub struct NativeGraph {
    inner: GraphIndex
}

#[napi]
impl NativeGraph {
    #[napi(constructor)]
    pub fn new() -> Self { ... }

    // Fast Bulk Load via Arrow
    #[napi]
    pub fn load_arrow_batch(&mut self, buffer_ptr: BigInt) {
        // Unsafe pointer magic to read Arrow batch from DuckDB
    }

    #[napi]
    pub fn traverse(&self, start_ids: Vec<String>, edge_type: String) -> Vec<String> {
        // Delegates to quack_core
    }
}
```

---

## 7. Development Workflow

### Prerequisites
1.  **Bun:** `curl -fsSL https://bun.sh/install | bash`
2.  **Rust:** `rustup update`
3.  **LLVM/Clang:** Required for building DuckDB extensions (if compiling from source).

### Setup

```bash
# 1. Install JS dependencies
bun install

# 2. Build the Rust Core & Bindings
# This runs cargo build inside /packages/native and /packages/wasm
bun run build:all

# 3. Run the Test Suite
# Uses Bun's native test runner (extremely fast)
bun test
```

### Running Benchmarks
We use a dedicated benchmark script to track regression in "Hydration" and "Traversal" speeds.

```bash
bun run bench
# Output:
# [Ingest] 100k nodes: 85ms
# [Hop] 3-depth traversal: 4ms
```

---

## 8. Cross-Platform Strategy

### Native (Backend)
*   **Tool:** `napi-rs`.
*   **Output:** `.node` binary file.
*   **Architecture:** We ship pre-built binaries for `linux-x64-gnu`, `linux-x64-musl`, `darwin-x64`, `darwin-arm64`, `win32-x64`.

### Edge (Wasm)
*   **Tool:** `wasm-pack`.
*   **Output:** `.wasm` file + JS glue.
*   **Constraint:** Wasm is single-threaded (mostly) and 32-bit address space (4GB limit).
*   **Storage:** On Edge, DuckDB uses `HTTPFS` to read Parquet from S3/R2, or `OPFS` in the browser.

---

## 9. Debugging & Profiling

### Rust Panics
Rust panics will crash the Bun process. To debug:
```bash
export RUST_BACKTRACE=1
bun test
```

### Memory Leaks
If `GraphIndex` grows indefinitely:
1.  Check `interner.rs`. Are we removing strings when nodes are deleted? (Current design: No, we tombstone. Strings leak until restart).
2.  Check Napi `External` references. Are we properly dropping Rust structs when JS objects are GC'd?

---

## 10. Future Proofing (Roadmap Specs)

### v0.5: Topology Snapshots
*   **Problem:** Hydration takes too long for 10M+ edges.
*   **Spec:** Implement `GraphIndex::serialize()` using `bincode` or `rkyv` (Zero-Copy deserialization framework).
*   **Flow:** Save `graph.bin` alongside `db.duckdb`. On boot, `mmap` `graph.bin` directly into memory.

### v0.8: Declarative Mutations (Merge)
*   **Problem:** "Check-then-Act" logic in JS is slow and race-condition prone.
*   **Spec:** Implement `MERGE` logic.
    *   Locking: Optimistic concurrency control or single-threaded writer queue.
    *   Logic: `INSERT ON CONFLICT DO UPDATE` generated in DuckDB.

### v1.0: Cypher Parser
*   **Problem:** DSL lock-in.
*   **Spec:** Use a PEG parser in Rust to parse Cypher strings into our internal AST.
*   **Goal:** `g.query("MATCH (n)-[:KNOWS]->(m) RETURN m")`.

### v1.0: Replication
*   **Problem:** Local-only limits usage.
*   **Spec:** Simple S3 sync.
*   **Command:** `g.sync.push('s3://bucket/latest')`.
*   **Logic:** Upload the `.duckdb` file and the `.bin` topology snapshot. Clients pull and hot-reload.
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

## File: .gitignore
````
# relay state
/.relay/
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
//"test-docs/unit.test-plan.md"
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
      "packages/quackgraph/src"
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
