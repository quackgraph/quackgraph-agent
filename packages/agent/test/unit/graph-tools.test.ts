import { describe, it, expect } from "bun:test";
import { GraphTools } from "../../src/tools/graph-tools";
import { runWithTestGraph } from "../utils/test-graph";

describe("Unit: GraphTools (Physics Layer)", () => {
  it("getNavigationalMap: handles cycles gracefully (Infinite Loop Protection)", async () => {
    await runWithTestGraph(async (graph) => {
      const tools = new GraphTools(graph);

      // Create a cycle: A <-> B
      // @ts-expect-error - dynamic graph
      await graph.addNode("A", ["Entity"], { name: "A" });
      // @ts-expect-error
      await graph.addNode("B", ["Entity"], { name: "B" });

      // @ts-expect-error
      await graph.addEdge("A", "B", "LOOP", {});
      // @ts-expect-error
      await graph.addEdge("B", "A", "LOOP", {});

      // Recursion depth 3
      const { map, truncated } = await tools.getNavigationalMap("A", 3);

      // Should show A -> B -> A -> B
      // The logic clamps at max depth, preventing infinite recursion
      expect(map).toContain("[ROOT] A");
      expect(map).toContain("(B)");
      
      // We expect some repetition due to depth=3, but it must not crash or hang
      const occurrencesOfB = map.match(/\(B\)/g)?.length || 0;
      expect(occurrencesOfB).toBeGreaterThanOrEqual(1);
      
      // Should not have exploded the line count limit immediately
      expect(truncated).toBe(false);
    });
  });

  it("reinforcePath: increments edge heat (Pheromones)", async () => {
    await runWithTestGraph(async (graph) => {
      const tools = new GraphTools(graph);

      // A -> B
      // @ts-expect-error
      await graph.addNode("start", ["Start"], {});
      // @ts-expect-error
      await graph.addNode("end", ["End"], {});
      // @ts-expect-error
      await graph.addEdge("start", "end", "PATH", { weight: 1 });

      // Initial state: heat is 0 (or default)
      const initialSummary = await tools.getSectorSummary(["start"]);
      const initialHeat = initialSummary.find(s => s.edgeType === "PATH")?.avgHeat || 0;

      // Reinforce
      await tools.reinforcePath(["start", "end"], [undefined, "PATH"], 1.0);

      // Check heat increase
      const newSummary = await tools.getSectorSummary(["start"]);
      const newHeat = newSummary.find(s => s.edgeType === "PATH")?.avgHeat || 0;

      // Note: In-memory mock might not implement the full u8 heat decay math, 
      // but we expect the command to have been issued and state updated if supported.
      // If the mock supports it:
      expect(newHeat).toBeGreaterThan(initialHeat);
    });
  });

  it("getSectorSummary: strictly enforces allowedEdgeTypes (Governance)", async () => {
    await runWithTestGraph(async (graph) => {
      const tools = new GraphTools(graph);

      // Root connected to Safe and Unsafe
      // @ts-expect-error
      await graph.addNode("root", ["Root"], {});
      // @ts-expect-error
      await graph.addNode("safe", ["Child"], {});
      // @ts-expect-error
      await graph.addNode("unsafe", ["Child"], {});

      // @ts-expect-error
      await graph.addEdge("root", "safe", "SAFE_LINK", {});
      // @ts-expect-error
      await graph.addEdge("root", "unsafe", "FORBIDDEN_LINK", {});

      // 1. Unrestricted
      const all = await tools.getSectorSummary(["root"]);
      expect(all.length).toBe(2);

      // 2. Restricted
      const restricted = await tools.getSectorSummary(["root"], undefined, ["SAFE_LINK"]);
      expect(restricted.length).toBe(1);
      expect(restricted[0].edgeType).toBe("SAFE_LINK");
      
      const forbidden = restricted.find(r => r.edgeType === "FORBIDDEN_LINK");
      expect(forbidden).toBeUndefined();
    });
  });
});