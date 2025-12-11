import { describe, it, expect, beforeAll, mock } from "bun:test";
import { runWithTestGraph } from "../utils/test-graph";
import { SyntheticLLM } from "../utils/synthetic-llm";
import { scoutAgent } from "../../src/mastra/agents/scout-agent";
import { judgeAgent } from "../../src/mastra/agents/judge-agent";
import { routerAgent } from "../../src/mastra/agents/router-agent";
import { mastra } from "../../src/mastra/index";

describe("E2E: Chaos Monkey (Resilience)", () => {
  let llm: SyntheticLLM;

  beforeAll(() => {
    llm = new SyntheticLLM();
    // Safe defaults
    llm.mockAgent(scoutAgent, { action: "ABORT", confidence: 0, reasoning: "Default Abort" });
    llm.mockAgent(judgeAgent, { isAnswer: false, answer: "No", confidence: 0 });
    llm.mockAgent(routerAgent, { domain: "global", confidence: 1, reasoning: "Default Global" });
  });

  it("handles Brain Damage (Malformed JSON from Scout)", async () => {
    await runWithTestGraph(async (graph) => {
      // @ts-expect-error
      await graph.addNode("start", ["Entity"], {});
      // @ts-expect-error
      await graph.addNode("end", ["Entity"], {});
      // @ts-expect-error
      await graph.addEdge("start", "end", "LINK", {});

      // 1. Sabotage the Scout Agent for this specific run
      // We override the generate method to throw garbage
      const originalGenerate = scoutAgent.generate;
      
      // @ts-expect-error - hijacking
      scoutAgent.generate = mock(async () => {
        return {
          text: "{ NOT VALID JSON ",
          object: null, // Simulate parser failure or raw text return
          usage: { totalTokens: 0 }
        };
      });

      try {
        const run = await mastra.getWorkflow("labyrinthWorkflow").createRunAsync();
        const res = await run.start({
          inputData: {
            goal: "Garbage In",
            start: "start",
            maxHops: 2
          }
        });

        // The workflow should complete, but find nothing because the thread was killed
        // @ts-expect-error
        const artifact = res.results?.artifact;
        expect(artifact).toBeNull(); // No winner found

      } finally {
        // Restore sanity
        scoutAgent.generate = originalGenerate;
      }
    });
  });

  it("handles Exhaustion (Max Hops Reached)", async () => {
    await runWithTestGraph(async (graph) => {
      // Infinite Chain: 1 -> 2 -> 3 ...
      // @ts-expect-error
      await graph.addNode("1", ["Entity"], {});
      // @ts-expect-error
      await graph.addNode("2", ["Entity"], {});
      // @ts-expect-error
      await graph.addEdge("1", "2", "NEXT", {});
      // @ts-expect-error
      await graph.addEdge("2", "3", "NEXT", {}); // Ghost edge to 3

      // Train Scout to always move NEXT
      llm.setDefault({
        action: "MOVE",
        edgeType: "NEXT",
        confidence: 0.9,
        reasoning: "Forever onward"
      });

      // Judge never satisfied
      llm.addResponse("search", { isAnswer: false, confidence: 0 });

      // Router
      llm.addResponse("search", { domain: "global" });

      const run = await mastra.getWorkflow("labyrinthWorkflow").createRunAsync();
      const res = await run.start({
        inputData: {
          goal: "search",
          start: "1",
          maxHops: 1 // Strict limit
        }
      });

      // @ts-expect-error
      const artifact = res.results?.artifact;
      
      // Should result in null (failure to find) rather than hanging
      expect(artifact).toBeNull();
    });
  });
});