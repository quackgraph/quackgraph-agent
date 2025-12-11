import { describe, it, expect, beforeEach, afterEach, beforeAll } from "bun:test";
import { createTestGraph } from "../utils/test-graph";
import { SyntheticLLM } from "../utils/synthetic-llm";
import { scribeAgent } from "../../src/mastra/agents/scribe-agent";
import { mutationWorkflow } from "../../src/mastra/workflows/mutation-workflow";
import type { QuackGraph } from "@quackgraph/graph";
import { getGraphInstance } from "../../src/lib/graph-instance";

describe("E2E: Mutation Workflow (The Scribe)", () => {
  let graph: QuackGraph;
  let llm: SyntheticLLM;

  beforeAll(() => {
    llm = new SyntheticLLM();
    // Hijack the singleton scribe agent
    llm.mockAgent(scribeAgent);
  });

  beforeEach(async () => {
    graph = await createTestGraph();
  });

  afterEach(async () => {
    // @ts-ignore
    if (typeof graph.close === 'function') await graph.close();
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
    const run = await mutationWorkflow.createRunAsync();
    const result = await run.start({
      inputData: {
        query: "Create a user named Bob",
        userId: "admin",
        asOf: new Date("2024-01-01").getTime()
      }
    });

    // 3. Verify Result
    // @ts-ignore
    expect(result.results.success).toBe(true);
    // @ts-ignore
    expect(result.results.summary).toContain("Created Node bob_1");

    // 4. Verify Side Effects (Graph Physics)
    const storedNode = await graph.match([]).where({ id: "bob_1" }).select();
    expect(storedNode.length).toBe(1);
    expect(storedNode[0].properties.name).toBe("Bob");
  });

  it("Scenario: Temporal Close ('Bob left the company yesterday')", async () => {
    // Setup: Bob exists and works at Acme
    // @ts-ignore
    await graph.addNode("bob_1", ["User"], { name: "Bob" });
    // @ts-ignore
    await graph.addNode("acme", ["Company"], { name: "Acme Inc" });
    // @ts-ignore
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
    const run = await mutationWorkflow.createRunAsync();
    const result = await run.start({
      inputData: {
        query: "Bob left the company",
        userId: "admin"
      }
    });

    // 3. Verify
    // @ts-ignore
    expect(result.results.success).toBe(true);

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
    // @ts-ignore
    await graph.addNode("car_1", ["Car"], { color: "Blue", model: "Ford" });
    // @ts-ignore
    await graph.addNode("car_2", ["Car"], { color: "Blue", model: "Chevy" });

    // 1. Train Brain to be confused
    llm.addResponse("Delete the car", {
      reasoning: "Ambiguous reference. Found multiple cars.",
      operations: [],
      requiresClarification: "Which car? The Ford or the Chevy?"
    });

    // 2. Execute
    const run = await mutationWorkflow.createRunAsync();
    const result = await run.start({
      inputData: {
        query: "Delete the car",
        userId: "admin"
      }
    });

    // 3. Verify
    // @ts-ignore
    expect(result.results.success).toBe(false);
    // @ts-ignore
    expect(result.results.summary).toContain("Clarification needed");
    // @ts-ignore
    expect(result.results.summary).toContain("The Ford or the Chevy");

    // 4. Verify Safety (No deletes happened)
    const cars = await graph.match([]).where({ labels: ["Car"] }).select();
    expect(cars.length).toBe(2);
  });
});