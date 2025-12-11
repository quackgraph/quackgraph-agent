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