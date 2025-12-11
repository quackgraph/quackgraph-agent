import { describe, it, expect, beforeAll } from "bun:test";
import { runWithTestGraph } from "../utils/test-graph";
import { SyntheticLLM } from "../utils/synthetic-llm";
import { judgeAgent } from "../../src/mastra/agents/judge-agent";
import { mastra } from "../../src/mastra/index";
import { generateTimeSeries } from "../utils/generators";

describe("E2E: Metabolism (The Dreaming Graph)", () => {
  let llm: SyntheticLLM;

  beforeAll(() => {
    llm = new SyntheticLLM();
    llm.mockAgent(judgeAgent);
  });

  it("Digests raw logs into a summary node", async () => {
    await runWithTestGraph(async (graph) => {
      // 1. Generate 10 days of "Mood" logs
      // @ts-expect-error
      await graph.addNode("user_alice", ["User"], { name: "Alice" });
      const { eventIds } = await generateTimeSeries(graph, "user_alice", 10, 24 * 60, 10 * 24 * 60);

      // 2. Train the Brain (Judge)
      llm.addResponse("Summarize these", {
        isAnswer: true,
        answer: "User mood was generally positive with a dip on day 3.",
        confidence: 1.0
      });

      // 3. Run Metabolism Workflow
      const run = await mastra.getWorkflow("metabolismWorkflow").createRunAsync();
      const res = await run.start({
        inputData: {
          minAgeDays: 0, // Process everything immediately for test
          targetLabel: "Event" // Matching generator label
        }
      });

      // 4. Verify Success
      // @ts-expect-error
      expect(res.results.success).toBe(true);

      // 5. Verify Physics (Graph State)
      // Old nodes should be gone (or disconnected/deleted)
      const oldNodes = await graph.match([]).where({ id: eventIds }).select();
      expect(oldNodes.length).toBe(0);

      // Summary node should exist
      const summaries = await graph.match([]).where({ labels: ["Summary"] }).select();
      expect(summaries.length).toBe(1);
      expect(summaries[0].properties.content).toBe("User mood was generally positive with a dip on day 3.");

      // Check linkage: user_alice -> HAS_SUMMARY -> SummaryNode
      // We need to verify user_alice is connected to the new summary
      const summaryId = summaries[0].id;
      const neighbors = await graph.native.traverse(["user_alice"], "HAS_SUMMARY", "out");
      expect(neighbors).toContain(summaryId);
    });
  });
});