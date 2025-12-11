# Directory Structure
```
packages/
  agent/
    src/
      agent/
        chronos.ts
    test/
      e2e/
        mutation.test.ts
      utils/
        result-helper.ts
        synthetic-llm.ts
  quackgraph/
    crates/
      quack_core/
        src/
          topology.rs
    packages/
      quack-graph/
        src/
          schema.ts
```

# Files

## File: packages/quackgraph/crates/quack_core/src/topology.rs
```rust
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
                let arr = col.as_any().downcast_ref::<UInt8Array>().unwrap();
                if arr.is_null(i) {
                    0
                } else {
                    arr.value(i)
                }
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
                            // Edge contains the window [start, end)
                            IntervalConstraint::Contains => vf <= start && vt >= end,
                            // Edge is contained by the window [start, end)
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
      const nextVal = maxId + 1n;
      // Try setval (standard), fallback to ALTER (Postgres/DuckDB variant), fallback to DROP/CREATE
      try {
        await this.db.execute(`SELECT setval('seq_node_id', ${nextVal})`);
      } catch {
        try {
          await this.db.execute(`ALTER SEQUENCE seq_node_id RESTART WITH ${nextVal}`);
        } catch (e) {
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

## File: packages/agent/test/utils/result-helper.ts
```typescript
export function getWorkflowResult(res: any): any {
  if (res.status === 'failed') {
    throw new Error(`Workflow failed: ${res.error?.message || 'Unknown error'}`);
  }
  
  // Prioritize "results" (plural) as seen in some Mastra versions/mocks
  if (res.results) return res.results;
  
  // Check "result" (singular)
  if (res.result) return res.result;
  
  // Fallback: Check if the object itself looks like a payload (has artifact or success)
  // or if it's just the wrapper but missing the specific keys we know.
  // We return res to handle cases where the payload is the root object.
  return res;
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

## File: packages/agent/test/utils/synthetic-llm.ts
```typescript
import { mock } from "bun:test";
import type { Agent } from "@mastra/core/agent";

/**
 * A deterministic LLM simulator for testing Mastra Agents.
 * Allows mapping prompt keywords to specific JSON responses.
 */
export class SyntheticLLM {
  private responses: Map<string, object> = new Map();
  
  // A "God Object" default that satisfies Scout, Judge, Router, and Scribe schemas
  // to prevent Zod validation errors during test fallbacks.
  private globalDefault: object = { 
    // Scout (Action Union)
    action: "ABORT",
    alternativeMoves: [], // Satisfy MoveAction optional but good to have empty
    
    // Router
    domain: "global",
    
    // Judge
    isAnswer: false,
    answer: "Synthetic Fallback",
    
    // Scribe
    operations: [],
    requiresClarification: undefined,

    // Common
    confidence: 0.0,
    reasoning: "No matching synthetic response configured (Fallback)." 
  };

  /**
   * Register a response trigger.
   * @param keyword If the prompt contains this string, the response will be returned.
   * @param response The JSON object to return.
   */
  addResponse(keyword: string, response: object) {
    this.responses.set(keyword, response);
  }

  setDefault(response: object) {
    this.globalDefault = response;
  }

  /**
   * Hijacks the `generate` method of a Mastra agent to return synthetic data.
   * @param agent The agent to mock
   * @param agentDefault Optional default response specific to this agent
   */
  // biome-ignore lint/suspicious/noExplicitAny: Mocking internal agent types
  mockAgent(agent: Agent<any, any, any>, agentDefault?: object) {
    // @ts-expect-error - Overwriting the generate method for testing
    // biome-ignore lint/suspicious/noExplicitAny: Mocking internal agent types
    agent.generate = mock(async (prompt: string, _options?: any) => {
      // 1. Check for keyword matches
      for (const [key, val] of this.responses) {
        if (prompt.includes(key)) {
          // Return a structured response that mimics Mastra's expected output
          return {
            text: JSON.stringify(val),
            object: val,
            usage: { promptTokens: 10, completionTokens: 10, totalTokens: 20 },
          };
        }
      }

      // 2. Fallback
      // Use agent-specific default if provided, otherwise global default
      const fallback = agentDefault || this.globalDefault;

      // Log warning for debugging
      // console.warn(`[SyntheticLLM] No match for prompt: "${prompt.slice(0, 50)}...". Using default.`);

      return {
        text: JSON.stringify(fallback),
        object: fallback,
        usage: { promptTokens: 1, completionTokens: 1, totalTokens: 2 },
      };
    });

    return agent;
  }
}
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
    // @ts-expect-error - Mastra generic return type
    const rawResults = result.results || result;
    
    const parsed = MutationResultSchema.safeParse(rawResults);
    if (!parsed.success) {
      throw new Error(`Invalid workflow result: ${JSON.stringify(rawResults)}`);
    }
    
    expect(parsed.data.success).toBe(true);
    expect(parsed.data.summary).toContain("Created Node bob_1");
    
    const rawResults = getWorkflowResult(result);
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
