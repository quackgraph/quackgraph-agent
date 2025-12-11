import { describe, it, expect, beforeEach } from "bun:test";
import { sectorScanTool, topologyScanTool } from "../../src/mastra/tools";
import { getSchemaRegistry } from "../../src/governance/schema-registry";
import { runWithTestGraph } from "../utils/test-graph";

describe("Integration: Tool Governance Enforcement", () => {
  beforeEach(() => {
    const registry = getSchemaRegistry();
    registry.register({
      name: "Classified",
      description: "Top Secret",
      allowedEdges: ["PUBLIC_INFO"]
    });
  });

  it("sectorScanTool: blinds agent to unauthorized edges", async () => {
    await runWithTestGraph(async (graph) => {
      // Setup: Node with Public and Secret links
      // @ts-expect-error
      await graph.addNode("doc_1", ["Document"], {});
      // @ts-expect-error
      await graph.addEdge("doc_1", "public_ref", "PUBLIC_INFO", {});
      // @ts-expect-error
      await graph.addEdge("doc_1", "secret_ref", "SECRET_SOURCE", {});

      // Execute Tool as "Classified" Domain
      const result = await sectorScanTool.execute({
        context: { nodeIds: ["doc_1"] },
        // @ts-expect-error
        runtimeContext: { get: (k) => k === 'domain' ? 'Classified' : undefined }
      });

      // Should see PUBLIC_INFO
      expect(result.summary.find(s => s.edgeType === "PUBLIC_INFO")).toBeDefined();
      
      // Should NOT see SECRET_SOURCE
      expect(result.summary.find(s => s.edgeType === "SECRET_SOURCE")).toBeUndefined();
    });
  });

  it("topologyScanTool: prevents traversing hidden paths", async () => {
    await runWithTestGraph(async (graph) => {
      // A ->(SECRET)-> B
      // @ts-expect-error
      await graph.addNode("A", ["Entity"], {});
      // @ts-expect-error
      await graph.addNode("B", ["Entity"], {});
      // @ts-expect-error
      await graph.addEdge("A", "B", "SECRET_SOURCE", {});

      // Attempt to traverse explicitly
      const result = await topologyScanTool.execute({
        context: { nodeIds: ["A"], edgeType: "SECRET_SOURCE" },
        // @ts-expect-error
        runtimeContext: { get: (k) => k === 'domain' ? 'Classified' : undefined }
      });

      // Should return empty, effectively invisible
      expect(result.neighborIds).toEqual([]);
    });
  });
});