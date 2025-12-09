Use `search-replace` for the failing file `packages/quackgraph/crates/quack_core/src/topology.rs` to ensure the patch applies correctly. The rest of the files seem fine with `standard-diff` but I will be careful with context.

### Step 1: Update Rust Core (`quack_core`)

I will use `search-replace` to safely inject the enum and update the method signature.

```rust // packages/quackgraph/crates/quack_core/src/topology.rs search-replace
<<<<<<< SEARCH
/// (Target Node ID, Edge Type ID, Valid From, Valid To, Heat)
type EdgeTuple = (u32, u8, i64, i64, u8);

/// The core Graph Index.
=======
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
>>>>>>> REPLACE
<<<<<<< SEARCH
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
=======
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
>>>>>>> REPLACE
```

Update `lib.rs` using `search-replace` as well to be safe.

```rust // packages/quackgraph/crates/quack_core/src/lib.rs search-replace
<<<<<<< SEARCH
pub use interner::Interner;
pub use topology::{GraphIndex, Direction};
=======
pub use interner::Interner;
pub use topology::{GraphIndex, Direction, IntervalConstraint};
>>>>>>> REPLACE
```

### Step 2: Update Native Bindings

```rust // packages/quackgraph/packages/native/src/lib.rs search-replace
<<<<<<< SEARCH
use quack_core::{matcher::{Matcher, PatternEdge}, GraphIndex, Direction};
=======
use quack_core::{matcher::{Matcher, PatternEdge}, GraphIndex, Direction, IntervalConstraint};
>>>>>>> REPLACE
<<<<<<< SEARCH
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
=======
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
>>>>>>> REPLACE
```

```typescript // packages/quackgraph/packages/native/index.d.ts search-replace
<<<<<<< SEARCH
   * Performs a traversal finding all neighbors connected via edges that overlap
   * with the specified time window [start, end).
   * Timestamps are in milliseconds (JS standard).
   */
  traverseInterval(sources: Array<string>, edgeType: string | undefined | null, direction: string | undefined | null, start: number, end: number): Array<string>
=======
   * Performs a traversal finding all neighbors connected via edges that overlap
   * with the specified time window [start, end).
   * Timestamps are in milliseconds (JS standard).
   * Constraint: 'overlaps' (default), 'contains', 'during', 'meets'.
   */
  traverseInterval(sources: Array<string>, edgeType: string | undefined | null, direction: string | undefined | null, start: number, end: number, constraint?: string | undefined | null): Array<string>
>>>>>>> REPLACE
```

### Step 3: Update Agent & Tools

**Types**

```typescript // packages/quackgraph/packages/agent/src/types.ts search-replace
<<<<<<< SEARCH
export interface AgentConfig {
  llmProvider: {
    generate: (prompt: string) => Promise<string>;
  };
=======
export interface AgentConfig {
  llmProvider: {
    generate: (prompt: string, signal?: AbortSignal) => Promise<string>;
  };
>>>>>>> REPLACE
```

**Router**

```typescript // packages/quackgraph/packages/agent/src/agent/router.ts search-replace
<<<<<<< SEARCH
  async route(goal: string, domains: DomainConfig[]): Promise<RouterDecision> {
=======
  async route(goal: string, domains: DomainConfig[], signal?: AbortSignal): Promise<RouterDecision> {
>>>>>>> REPLACE
<<<<<<< SEARCH
    try {
      const raw = await this.config.llmProvider.generate(prompt);
      const jsonStr = raw.match(/\{[\s\S]*\}/)?.[0] || raw;
=======
    try {
      const raw = await this.config.llmProvider.generate(prompt, signal);
      const jsonStr = raw.match(/\{[\s\S]*\}/)?.[0] || raw;
>>>>>>> REPLACE
```

**Scout**

```typescript // packages/quackgraph/packages/agent/src/agent/scout.ts search-replace
<<<<<<< SEARCH
  async decide(promptCtx: ScoutPrompt): Promise<ScoutDecision> {
=======
  async decide(promptCtx: ScoutPrompt, signal?: AbortSignal): Promise<ScoutDecision> {
>>>>>>> REPLACE
<<<<<<< SEARCH
    try {
      const raw = await this.config.llmProvider.generate(prompt);
      const jsonStr = raw.match(/\{[\s\S]*\}/)?.[0] || raw;
=======
    try {
      const raw = await this.config.llmProvider.generate(prompt, signal);
      const jsonStr = raw.match(/\{[\s\S]*\}/)?.[0] || raw;
>>>>>>> REPLACE
```

**Judge**

```typescript // packages/quackgraph/packages/agent/src/agent/judge.ts search-replace
<<<<<<< SEARCH
  async evaluate(promptCtx: JudgePrompt): Promise<LabyrinthArtifact | null> {
=======
  async evaluate(promptCtx: JudgePrompt, signal?: AbortSignal): Promise<LabyrinthArtifact | null> {
>>>>>>> REPLACE
<<<<<<< SEARCH
    try {
      const raw = await this.config.llmProvider.generate(prompt);
      const jsonStr = raw.match(/\{[\s\S]*\}/)?.[0] || raw;
=======
    try {
      const raw = await this.config.llmProvider.generate(prompt, signal);
      const jsonStr = raw.match(/\{[\s\S]*\}/)?.[0] || raw;
>>>>>>> REPLACE
```

**Labyrinth (Main Logic)**

```typescript // packages/quackgraph/packages/agent/src/labyrinth.ts search-replace
<<<<<<< SEARCH
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
=======
    goal: string,
    timeContext?: TimeContext
  ): Promise<LabyrinthArtifact | null> {
    const rootController = new AbortController();
    const rootSignal = rootController.signal;

    const traceId = randomUUID();
    const startTime = Date.now();
    
    // --- Phase 0: Context Firewall (Routing) ---
    // Before we start walking, decide WHICH domain we are walking in.
    const availableDomains = this.registry.getAllDomains();
    const route = await this.router.route(goal, availableDomains, rootSignal);
    const activeDomain = route.domain;
>>>>>>> REPLACE
<<<<<<< SEARCH
    // Main Loop: While we have active cursors and haven't found the answer
    while (cursors.length > 0 && !foundArtifact) {
      const nextCursors: Cursor[] = [];
      const processingPromises: Promise<void>[] = [];
      
      // We wrap the iteration to allow for "Race" logic (checking foundArtifact)
      
      for (const cursor of cursors) {
        if (foundArtifact) break; // Early exit check

        const task = async () => {
          if (foundArtifact) return; // Double check inside async
=======
    // Speculative Execution: We use a local controller for the batch to kill peers if winner found
    try {
      // Main Loop: While we have active cursors and haven't found the answer
      while (cursors.length > 0 && !foundArtifact && !rootSignal.aborted) {
      const nextCursors: Cursor[] = [];
      const processingPromises: Promise<void>[] = [];
      
      // We wrap the iteration to allow for "Race" logic (checking foundArtifact)
      
      for (const cursor of cursors) {
        if (foundArtifact) break; // Early exit check

        const task = async () => {
          if (foundArtifact) return; // Double check inside async
>>>>>>> REPLACE
<<<<<<< SEARCH
          const decision = await this.scout.decide(prompt);
          
          // Register Step
=======
          const decision = await this.scout.decide(prompt, rootSignal);
          
          // Register Step
>>>>>>> REPLACE
<<<<<<< SEARCH
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
=======
            // Ask Judge
            const judgePrompt: JudgePrompt = { 
              goal, 
              nodeContent: content,
              timeContext: timeDesc 
            };
            const artifact = await this.judge.evaluate(judgePrompt, rootSignal);
            
            if (artifact && artifact.confidence >= (this.config.confidenceThreshold || 0.7)) {
              // Success found by this cursor!
              artifact.traceId = traceId;
              artifact.sources = [cursor.currentNodeId];
              
              // Set global found flag to stop other cursors
              foundArtifact = { type: 'FOUND', artifact, finalStepId: currentStepId } as any; 
              rootController.abort(); // Kill other pending scouts in this batch
              return;
            } 
>>>>>>> REPLACE
<<<<<<< SEARCH
      // Pruning / Management
      // We sort by confidence and take top N
      nextCursors.sort((a, b) => b.confidence - a.confidence);
      cursors = nextCursors.slice(0, maxCursors);
    }

    if (log.outcome !== 'FOUND') {
=======
      // Pruning / Management
      // We sort by confidence and take top N
      nextCursors.sort((a, b) => b.confidence - a.confidence);
      cursors = nextCursors.slice(0, maxCursors);
      }
    } finally {
      // Ensure we clean up if something throws
      if (!foundArtifact && !rootSignal.aborted) {
          rootController.abort();
      }
    }

    if (log.outcome !== 'FOUND') {
>>>>>>> REPLACE
```

**Metabolism (Optimized Dreaming)**

```typescript // packages/quackgraph/packages/agent/src/agent/metabolism.ts search-replace
<<<<<<< SEARCH
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
=======
    // 1. Identify Candidates (Nodes older than X days)
    // We process in batches to allow iterative cleanup
    const sql = `
      SELECT id, properties 
      FROM nodes 
      WHERE list_contains(labels, ?) 
        AND valid_from < (current_timestamp - INTERVAL ${criteria.minAgeDays} DAY)
        AND valid_to IS NULL -- Active nodes only
      LIMIT 100 -- Batch size
    `;

    const candidates = await this.graph.db.query(sql, [criteria.targetLabel]);
    if (candidates.length === 0) return;

    // 2. Synthesize (Judge)
    const judgePrompt: JudgePrompt = {
      goal: `Metabolism/Dreaming: Summarize these ${candidates.length} ${criteria.targetLabel} logs into a single concise insight node. Focus on patterns and key events.`,
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
      undefined,
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
      period_end: new Date().toISOString()
    };

    await this.graph.addNode(summaryId, ['Summary', 'Insight'], summaryProps);
    await this.graph.addEdge(anchorId, summaryId, 'HAS_SUMMARY');

    // Soft delete raw nodes
    for (const id of candidateIds) {
      await this.graph.deleteNode(id);
    }
  }
>>>>>>> REPLACE
```

**GraphTools**

```typescript // packages/quackgraph/packages/agent/src/tools/graph-tools.ts search-replace
<<<<<<< SEARCH
  async topologyScan(currentNodes: string[], edgeType: string, asOf?: number): Promise<string[]> {
    return this.graph.native.traverse(currentNodes, edgeType, 'out', asOf);
  }

  /**
   * LOD 1.5: Pattern Matching (Structural Inference)
=======
  async topologyScan(currentNodes: string[], edgeType: string, asOf?: number): Promise<string[]> {
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
   * LOD 1.5: Pattern Matching (Structural Inference)
>>>>>>> REPLACE
```

**Chronos**

```typescript // packages/quackgraph/packages/agent/src/agent/chronos.ts search-replace
<<<<<<< SEARCH
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
=======
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
>>>>>>> REPLACE
```

**Graph Facade**

```typescript // packages/quackgraph/packages/quack-graph/src/graph.ts search-replace
<<<<<<< SEARCH
  async traverseInterval(sources: string[], edgeType: string | undefined, direction: 'out' | 'in' = 'out', start: Date, end: Date): Promise<string[]> {
    const s = start.getTime();
    const e = end.getTime();
    // If end < start, return empty
    if (e <= s) return [];
    return this.native.traverseInterval(sources, edgeType, direction, s, e);
  }
=======
  async traverseInterval(sources: string[], edgeType: string | undefined, direction: 'out' | 'in' = 'out', start: Date, end: Date, constraint: 'overlaps' | 'contains' | 'during' | 'meets' = 'overlaps'): Promise<string[]> {
    const s = start.getTime();
    const e = end.getTime();
    // If end < start, return empty
    if (e <= s) return [];
    return this.native.traverseInterval(sources, edgeType, direction, s, e, constraint);
  }
>>>>>>> REPLACE
```

```yaml
projectId: quackgraph-agent
uuid: 44b7d03a-0e6d-495c-9c22-b5f7e7f7b1d2
changeSummary:
  - edit: packages/quackgraph/crates/quack_core/src/topology.rs
  - edit: packages/quackgraph/crates/quack_core/src/lib.rs
  - edit: packages/quackgraph/packages/native/src/lib.rs
  - edit: packages/quackgraph/packages/native/index.d.ts
  - edit: packages/quackgraph/packages/agent/src/types.ts
  - edit: packages/quackgraph/packages/agent/src/agent/router.ts
  - edit: packages/quackgraph/packages/agent/src/agent/scout.ts
  - edit: packages/quackgraph/packages/agent/src/agent/judge.ts
  - edit: packages/quackgraph/packages/agent/src/labyrinth.ts
  - edit: packages/quackgraph/packages/agent/src/agent/metabolism.ts
  - edit: packages/quackgraph/packages/agent/src/tools/graph-tools.ts
  - edit: packages/quackgraph/packages/agent/src/agent/chronos.ts
  - edit: packages/quackgraph/packages/quack-graph/src/graph.ts
promptSummary: Retry implementing "Ghost Earth" architecture with `search-replace` strategy for failing files, adding temporal interval constraints and speculative execution.
gitCommitMsg: >-
  feat: Implement Ghost Earth architecture (Retry).

  - Added IntervalConstraint (Overlaps, Contains, During, Meets) to Rust Core.
  - Implemented AbortController for speculative execution in Labyrinth Agent.
  - Enhanced Metabolism to support batch summarization.
  - Exposed temporal constraints in Native bindings and GraphTools.
```