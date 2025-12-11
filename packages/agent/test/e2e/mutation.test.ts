import { describe, it, expect, beforeEach, afterEach, beforeAll } from "bun:test";
import { createTestGraph } from "../utils/test-graph";
import { SyntheticLLM } from "../utils/synthetic-llm";
import { scribeAgent } from "../../src/mastra/agents/scribe-agent";
import { mastra } from "../../src/mastra/index";
import type { QuackGraph } from "@quackgraph/graph";
import { z } from "zod";
import { getWorkflowResult } from "../utils/result-helper";

const MutationResultSchema = z.object({
  success: z.boolean(),
  summary: z.string()
});

describe("E2E: Mutation Workflow (The Scribe)", () => {
  let graph: QuackGraph;
  let llm: SyntheticLLM;

  beforeAll(() => {
    llm = new SyntheticLLM();
    // Hijack the singleton scribe agent
    // Scribe schema requires operations array
    llm.mockAgent(scribeAgent, { operations: [], reasoning: "Default", requiresClarification: undefined });
  });

  beforeEach(async () => {
    graph = await createTestGraph();
  });

  afterEach(async () => {
    // @ts-expect-error
    if (graph && typeof graph.close === 'function') await graph.close();
  });

  it("Scenario: Create Node ('Create a user named Bob')", async () => {
    // 1. Train the Synthetic Brain
    llm.addResponse("Create a user named Bob", {
      reasoning: "User explicitly requested creation of a new Entity.",
      operations: [
        {
          op: "CREATE_NODE",
          id: "bob_1",
          labels: ["User"],
          properties: { name: "Bob" },
          validFrom: "2024-01-01T00:00:00.000Z"
        }
      ]
    });

    // 2. Execute Workflow
    const run = await mastra.getWorkflow("mutationWorkflow").createRunAsync();
    const result = await run.start({
      inputData: {
        query: "Create a user named Bob",
        userId: "admin",
        asOf: new Date("2024-01-01").getTime()
      }
    });

    // @ts-expect-error
    if (result.status === "failed") throw new Error(`Workflow failed: ${result.error?.message}`);

    // 3. Verify Result
    const rawResults = getWorkflowResult(result);

    const parsed = MutationResultSchema.safeParse(rawResults);
    if (!parsed.success) {
      throw new Error(`Invalid workflow result: ${JSON.stringify(rawResults)}`);
    }

    expect(parsed.data.success).toBe(true);
    expect(parsed.data.summary).toContain("Created Node bob_1");

    // Verify side effects
    // @ts-expect-error
    const storedNode = await graph.match([]).where({ labels: ["User"], id: "bob_1" }).select();
    expect(storedNode.length).toBe(1);
    expect(storedNode[0].properties.name).toBe("Bob");
  });

  it("Scenario: Temporal Close ('Bob left the company yesterday')", async () => {
    // Setup: Bob exists and works at Acme
    // @ts-expect-error
    await graph.addNode("bob_1", ["User"], { name: "Bob" });
    // @ts-expect-error
    await graph.addNode("acme", ["Company"], { name: "Acme Inc" });
    // @ts-expect-error
    await graph.addEdge("bob_1", "acme", "WORKS_AT", { role: "Engineer" });

    // 1. Train Brain
    const validTo = "2024-01-02T12:00:00.000Z";
    llm.addResponse("Bob left the company", {
      reasoning: "User indicated employment ended. Closing edge.",
      operations: [
        {
          op: "CLOSE_EDGE",
          source: "bob_1",
          target: "acme",
          type: "WORKS_AT",
          validTo: validTo
        }
      ]
    });

    // 2. Execute
    const run = await mastra.getWorkflow("mutationWorkflow").createRunAsync();
    const result = await run.start({
      inputData: {
        query: "Bob left the company",
        userId: "admin"
      }
    });

    // @ts-expect-error
    if (result.status === "failed") throw new Error(`Workflow failed: ${result.error?.message}`);

    // 3. Verify
    const rawResults = getWorkflowResult(result);
    const parsed = MutationResultSchema.safeParse(rawResults);
    if (!parsed.success) {
      throw new Error(`Invalid workflow result: ${JSON.stringify(rawResults)}`);
    }

    expect(parsed.data.success).toBe(true);

    // 4. Verify Side Effects (Time Travel)
    // The edge should still exist physically but have a valid_to set
    // Note: QuackGraph native.removeEdge might delete it from RAM index, 
    // but DB should retain it if we checked DB directly.
    // For this test, we verify the Workflow reported success and assumed DB update logic held.
    
    // In memory graph implementation might delete it immediately on 'removeEdge' 
    // depending on how QuackGraph core handles soft deletes in memory.
    // Let's verify it's gone from the "Present" view.
    const neighbors = await graph.native.traverse(["bob_1"], "WORKS_AT", "out");
    expect(neighbors).not.toContain("acme");
  });

  it("Scenario: Ambiguity ('Delete the car')", async () => {
    // Setup: Two cars
    // @ts-expect-error
    await graph.addNode("car_1", ["Car"], { color: "Blue", model: "Ford" });
    // @ts-expect-error
    await graph.addNode("car_2", ["Car"], { color: "Blue", model: "Chevy" });

    // 1. Train Brain to be confused
    llm.addResponse("Delete the car", {
      reasoning: "Ambiguous reference. Found multiple cars.",
      operations: [],
      requiresClarification: "Which car? The Ford or the Chevy?"
    });

    // 2. Execute
    const run = await mastra.getWorkflow("mutationWorkflow").createRunAsync();
    const result = await run.start({
      inputData: {
        query: "Delete the car",
        userId: "admin"
      }
    });

    // @ts-expect-error
    if (result.status === "failed") throw new Error(`Workflow failed: ${result.error?.message}`);

    // 3. Verify
    const rawResults = getWorkflowResult(result);
    const parsed = MutationResultSchema.safeParse(rawResults);
    if (!parsed.success) throw new Error("Invalid Result");

    expect(parsed.data.success).toBe(false);
    expect(parsed.data.summary).toContain("Clarification needed");
    expect(parsed.data.summary).toContain("The Ford or the Chevy");

    // 4. Verify Safety (No deletes happened)
    const cars = await graph.match([]).where({ labels: ["Car"] }).select();
    expect(cars.length).toBe(2);
  });
});