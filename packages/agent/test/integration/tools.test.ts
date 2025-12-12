import { describe, it, expect, beforeEach, afterEach } from "bun:test";
import { GraphTools } from "@quackgraph/graph";
import { createTestGraph } from "../utils/test-graph";
import type { QuackGraph } from "@quackgraph/graph";

describe("Integration: Graph Tools (Physics Layer)", () => {
  let graph: QuackGraph;
  let tools: GraphTools;

  beforeEach(async () => {
    graph = await createTestGraph();
    tools = new GraphTools(graph);
    
    // Seed Data
    // Root Node
    // @ts-expect-error - Dynamic graph method
    await graph.addNode("root", ["Entity"], { name: "Root" });
    
    // Branch A (Hot)
    // @ts-expect-error
    await graph.addNode("a1", ["Entity"], { name: "A1" });
    // @ts-expect-error
    await graph.addEdge("root", "a1", "LINK", { weight: 1 });
    
    // Branch B (Cold)
    // @ts-expect-error
    await graph.addNode("b1", ["Entity"], { name: "B1" });
    // @ts-expect-error
    await graph.addEdge("root", "b1", "LINK", { weight: 1 });
    
    // Temporal Node (Future)
    // using addNodes to ensure validFrom property support if addNode signature varies
    // @ts-expect-error
    await graph.addNodes([{
      id: "future",
      labels: ["Entity"],
      properties: { name: "Future" },
      validFrom: new Date("2030-01-01")
    }]);
    
    // @ts-expect-error
    await graph.addEdges([{
      source: "root",
      target: "future",
      type: "FUTURE_LINK",
      properties: {},
      validFrom: new Date("2030-01-01")
    }]);
  });

  afterEach(async () => {
    // Teardown if supported by graph instance
    // @ts-expect-error
    if (typeof graph.close === 'function') await graph.close();
  });

  it("LOD 0: getSectorSummary aggregates edge types", async () => {
    // Add more edges to test aggregation
    // @ts-expect-error
    await graph.addNode("a2", ["Entity"], {});
    // @ts-expect-error
    await graph.addEdge("root", "a2", "LINK", {});
    
    // @ts-expect-error
    await graph.addNode("c1", ["Entity"], {});
    // @ts-expect-error
    await graph.addEdge("root", "c1", "OTHER", {});

    const summary = await tools.getSectorSummary(["root"]);
    
    const linkStats = summary.find(s => s.edgeType === "LINK");
    const otherStats = summary.find(s => s.edgeType === "OTHER");
    
    expect(linkStats).toBeDefined();
    // 2 initial LINKs (a1, b1) + 1 new (a2) = 3
    expect(linkStats?.count).toBeGreaterThanOrEqual(3);
    expect(otherStats?.count).toBe(1);
  });

  it("LOD 0: getSectorSummary respects time travel (asOf)", async () => {
    const past = new Date("2020-01-01").getTime();
    const summary = await tools.getSectorSummary(["root"], past);
    
    // The "future" edge shouldn't exist in 2020
    const futureStats = summary.find(s => s.edgeType === "FUTURE_LINK");
    expect(futureStats).toBeUndefined();
  });

  it("LOD 1: topologyScan traverses edges", async () => {
    const neighbors = await tools.topologyScan(["root"], "LINK");
    expect(neighbors).toContain("a1");
    expect(neighbors).toContain("b1");
    expect(neighbors).not.toContain("future"); // Wrong type
  });

  it("LOD 1.5: getNavigationalMap generates ASCII tree", async () => {
    const { map, truncated } = await tools.getNavigationalMap("root", 2);
    
    // console.log("Debug Map Output:\n", map);
    
    expect(map).toContain("[ROOT] root");
    expect(map).toContain("├── [LINK]──> (a1)"); // Order depends on heat/count
    expect(map).not.toContain("future"); // Should be hidden by default "now"
    
    expect(truncated).toBe(false);
  });

  it("LOD 1.5: getNavigationalMap respects asOf", async () => {
    const futureTime = new Date("2035-01-01").getTime();
    const { map } = await tools.getNavigationalMap("root", 1, futureTime);
    
    expect(map).toContain("future");
    expect(map).toContain("[FUTURE_LINK]");
  });
});