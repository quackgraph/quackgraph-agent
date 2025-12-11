import { describe, it, expect, beforeAll } from "bun:test";
import { runWithTestGraph } from "../utils/test-graph";
import { SyntheticLLM } from "../utils/synthetic-llm";
import { scribeAgent } from "../../src/mastra/agents/scribe-agent";
import { mastra } from "../../src/mastra/index";
import { getWorkflowResult } from "../utils/result-helper";

describe("E2E: Scribe (Complex Mutations)", () => {
  let llm: SyntheticLLM;

  beforeAll(() => {
    llm = new SyntheticLLM();
    llm.mockAgent(scribeAgent, { operations: [], reasoning: "Default", requiresClarification: undefined });
  });

  it("Halts on Ambiguity ('Delete the blue car')", async () => {
    await runWithTestGraph(async (graph) => {
      // Setup: Two blue cars
      // @ts-expect-error
      await graph.addNode("ford", ["Car"], { color: "blue" });
      // @ts-expect-error
      await graph.addNode("chevy", ["Car"], { color: "blue" });

      // Train Scribe to be confused
      llm.addResponse("Delete the blue car", {
        reasoning: "Ambiguous target.",
        operations: [],
        requiresClarification: "Did you mean the Ford or the Chevy?"
      });

      const run = await mastra.getWorkflow("mutationWorkflow").createRunAsync();
      const res = await run.start({
        inputData: { query: "Delete the blue car" }
      });

      // @ts-expect-error
      if (res.status === "failed") throw new Error(`Workflow failed: ${res.error?.message}`);
      
      const payload = getWorkflowResult(res);

      // @ts-expect-error
      expect(payload?.success).toBe(false);
      // @ts-expect-error
      expect(payload?.summary).toContain("Did you mean the Ford or the Chevy?");

      // Verify no deletion happened
      const cars = await graph.match([]).where({ labels: ["Car"] }).select();
      expect(cars.length).toBe(2);
    });
  });

  it("Executes Temporal Deletion ('Sold it yesterday')", async () => {
    await runWithTestGraph(async (graph) => {
      // Setup
      // @ts-expect-error
      await graph.addNode("me", ["User"], {});
      // @ts-expect-error
      await graph.addNode("bike", ["Item"], {});
      // @ts-expect-error
      await graph.addEdge("me", "bike", "OWNS", {});

      const YESTERDAY = new Date(Date.now() - 86400000).toISOString();

      // Train Scribe
      llm.addResponse("I sold the bike yesterday", {
        reasoning: "Ownership ended.",
        operations: [
          {
            op: "CLOSE_EDGE",
            source: "me",
            target: "bike",
            type: "OWNS",
            validTo: YESTERDAY
          }
        ]
      });

      const run = await mastra.getWorkflow("mutationWorkflow").createRunAsync();
      const res = await run.start({ inputData: { query: "I sold the bike yesterday" } });

      // @ts-expect-error
      if (res.status === "failed") throw new Error(`Workflow failed: ${res.error?.message}`);
      
      // Verify Physics: Edge should not exist in "Present" view
      // traverse() defaults to now()
      const currentItems = await graph.native.traverse(["me"], "OWNS", "out");
      expect(currentItems).not.toContain("bike");

      // Verify it exists in the past (Time Travel)
      // Check 2 days ago
      const twoDaysAgo = Date.now() - (2 * 86400000);
      const pastItems = await graph.native.traverse(["me"], "OWNS", "out", twoDaysAgo);
      // Depending on how strict the test graph implementation is, this should be true if supported
      // For in-memory QuackGraph, basic time travel is supported.
      expect(pastItems).toContain("bike");
    });
  });
});