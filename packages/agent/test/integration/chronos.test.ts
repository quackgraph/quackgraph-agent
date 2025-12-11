import { describe, it, expect, beforeEach, afterEach } from "bun:test";
import { Chronos } from "../../src/agent/chronos";
import { GraphTools } from "../../src/tools/graph-tools";
import { createTestGraph } from "../utils/test-graph";
import type { QuackGraph } from "@quackgraph/graph";

describe("Integration: Chronos (Temporal Physics)", () => {
  let graph: QuackGraph;
  let tools: GraphTools;
  let chronos: Chronos;

  beforeEach(async () => {
    graph = await createTestGraph();
    tools = new GraphTools(graph);
    chronos = new Chronos(graph, tools);
  });

  afterEach(async () => {
    // @ts-ignore
    if (typeof graph.close === 'function') await graph.close();
  });

  it("analyzeCorrelation: detects causal overlap", async () => {
    const NOW = new Date("2025-01-01T12:00:00Z");
    const ONE_HOUR = 60 * 60 * 1000;

    // 1. Anchor Node (e.g., "Migraine") at T=0
    // @ts-ignore
    await graph.addNodes([{ id: "migraine_1", labels: ["Symptom"], properties: {}, validFrom: NOW }]);

    // 2. Target Node (e.g., "Coffee") at T=-30min (Inside window)
    const tInside = new Date(NOW.getTime() - (30 * 60 * 1000));
    // @ts-ignore
    await graph.addNodes([{ id: "coffee_1", labels: ["Food"], properties: { name: "Espresso" }, validFrom: tInside }]);

    // 3. Target Node (e.g., "Coffee") at T=-2hours (Outside window)
    const tOutside = new Date(NOW.getTime() - (2 * ONE_HOUR));
    // @ts-ignore
    await graph.addNodes([{ id: "coffee_2", labels: ["Food"], properties: { name: "Latte" }, validFrom: tOutside }]);

    // Analyze 60 minute window
    const result = await chronos.analyzeCorrelation("migraine_1", "Food", 60);

    expect(result.targetLabel).toBe("Food");
    expect(result.sampleSize).toBe(1); // Only coffee_1
    expect(result.correlationScore).toBe(1.0); // Simple boolean presence
  });

  it("evolutionaryDiff: tracks topology changes", async () => {
    const t1 = new Date("2024-01-01");
    const t2 = new Date("2024-02-01");
    const t3 = new Date("2024-03-01");

    // T1: Anchor exists, connected to A
    // @ts-ignore
    await graph.addNodes([{ id: "anchor", labels: ["Entity"], properties: {}, validFrom: t1 }]);
    // @ts-ignore
    await graph.addNodes([{ id: "A", labels: ["Child"], properties: {}, validFrom: t1 }]);
    // @ts-ignore
    await graph.addEdges([{ source: "anchor", target: "A", type: "KNOWS", properties: {}, validFrom: t1 }]);

    // T2: Add B
    // @ts-ignore
    await graph.addNodes([{ id: "B", labels: ["Child"], properties: {}, validFrom: t2 }]);
    // @ts-ignore
    await graph.addEdges([{ source: "anchor", target: "B", type: "KNOWS", properties: {}, validFrom: t2 }]);

    // T3: Remove A (Close edge)
    // Direct DB manipulation to simulate edge closing if API is limited
    // @ts-ignore
    await graph.db.execute(
        `UPDATE edges SET valid_to = ? WHERE source = ? AND target = ?`, 
        [t3.toISOString(), "anchor", "A"]
    );

    const result = await chronos.evolutionaryDiff("anchor", [t1, t2, t3]);

    expect(result.timeline.length).toBe(3);

    // Snapshot T1: KNOWS (1)
    // Initial state diff vs empty
    const t1Added = result.timeline[0].addedEdges.find(e => e.edgeType === "KNOWS");
    expect(t1Added?.count).toBe(1);

    // Snapshot T2: KNOWS (2) -> A + B
    // Diff vs T1: KNOWS Persisted (count 2)
    // Note: Because edgeType is the same, Chronos groups them. 
    // It sees KNOWS in T1 (count 1) and KNOWS in T2 (count 2).
    // It classifies this as "Persisted" because the type exists in both.
    const t2Step = result.timeline[1];
    const knowsPersistedT2 = t2Step.persistedEdges.find(e => e.edgeType === "KNOWS");
    expect(knowsPersistedT2?.count).toBe(2); 

    // Snapshot T3: KNOWS (1) -> B only
    // Diff vs T2: KNOWS Persisted (count 1)
    const t3Step = result.timeline[2];
    const knowsPersistedT3 = t3Step.persistedEdges.find(e => e.edgeType === "KNOWS");
    expect(knowsPersistedT3?.count).toBe(1);
    
    // Density Calculation check
    // T2 (2) -> T3 (1). Change = (1 - 2) / 2 = -0.5 (-50%)
    expect(t3Step.densityChange).toBe(-50);
  });
});