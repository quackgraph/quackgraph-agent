import { describe, it, expect, beforeEach } from "bun:test";
import { sectorScanTool, topologyScanTool } from "../../src/mastra/tools";
import { getSchemaRegistry } from "../../src/governance/schema-registry";
import { runWithTestGraph } from "../utils/test-graph";

describe("Integration: Governance Enforcement", () => {
  beforeEach(() => {
    // Reset registry to ensure clean state
    const registry = getSchemaRegistry();
    // Register test domains
    registry.register({
      name: "Medical",
      description: "Health data only",
      allowedEdges: ["TREATED_WITH", "HAS_SYMPTOM"]
    });
    registry.register({
      name: "Financial",
      description: "Money data only",
      allowedEdges: ["BOUGHT", "OWES"]
    });
  });

  it("sectorScanTool: blinds agent to unauthorized edges", async () => {
    await runWithTestGraph(async (graph) => {
      // Setup: A node with mixed sensitive data
      // @ts-expect-error
      await graph.addNode("patient_zero", ["Person"], {});
      
      // Medical Edge
      // @ts-expect-error
      await graph.addEdge("patient_zero", "flu", "HAS_SYMPTOM", {});
      
      // Financial Edge
      // @ts-expect-error
      await graph.addEdge("patient_zero", "hospital", "OWES", { amount: 5000 });

      // 1. Run as "Medical" Agent
      // Mimic Mastra tool execution context
      const medResult = await sectorScanTool.execute({
        context: { nodeIds: ["patient_zero"] },
        // @ts-expect-error - Mocking runtime context
        runtimeContext: { get: (key: string) => key === 'domain' ? 'Medical' : undefined }
      });

      const medSummary = medResult.summary;
      expect(medSummary.find(s => s.edgeType === "HAS_SYMPTOM")).toBeDefined();
      expect(medSummary.find(s => s.edgeType === "OWES")).toBeUndefined(); // BLOCKED

      // 2. Run as "Financial" Agent
      const finResult = await sectorScanTool.execute({
        context: { nodeIds: ["patient_zero"] },
        // @ts-expect-error
        runtimeContext: { get: (key: string) => key === 'domain' ? 'Financial' : undefined }
      });

      const finSummary = finResult.summary;
      expect(finSummary.find(s => s.edgeType === "OWES")).toBeDefined();
      expect(finSummary.find(s => s.edgeType === "HAS_SYMPTOM")).toBeUndefined(); // BLOCKED
    });
  });

  it("topologyScanTool: prevents traversal of unauthorized edges", async () => {
    await runWithTestGraph(async (graph) => {
      // @ts-expect-error
      await graph.addNode("A", ["Entity"], {});
      // @ts-expect-error
      await graph.addNode("B", ["Entity"], {});
      // @ts-expect-error
      await graph.addEdge("A", "B", "SECRET_LINK", {});

      const registry = getSchemaRegistry();
      registry.register({
        name: "Public",
        description: "Public info",
        allowedEdges: ["PUBLIC_LINK"]
      });

      // Attempt to traverse SECRET_LINK while in Public domain
      const result = await topologyScanTool.execute({
        context: { nodeIds: ["A"], edgeType: "SECRET_LINK" },
        // @ts-expect-error
        runtimeContext: { get: (key: string) => key === 'domain' ? 'Public' : undefined }
      });

      // Should return empty, effectively invisible
      expect(result.neighborIds).toEqual([]);
    });
  });
});