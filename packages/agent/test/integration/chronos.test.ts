import { describe, it, expect } from "bun:test";
import { Chronos, GraphTools } from "@quackgraph/graph";
import { runWithTestGraph } from "../utils/test-graph";
import { generateTimeSeries } from "../utils/generators";

describe("Integration: Chronos (Temporal Physics)", () => {
  
  it("traverseInterval: strictly enforces interval algebra", async () => {
    await runWithTestGraph(async (graph) => {
      const tools = new GraphTools(graph);
      const chronos = new Chronos(graph, tools);

      const BASE_TIME = new Date("2025-01-01T12:00:00Z").getTime();
      const ONE_HOUR = 60 * 60 * 1000;

      // 1. Setup Anchor Node (The Meeting)
      // Duration: 12:00 -> 13:00
      // @ts-expect-error
      await graph.addNode("meeting", ["Event"], { name: "Meeting" });
      
      // We represent interval edges by having a target node that represents the interval context,
      // OR we just test traverseInterval on nodes that have validFrom.
      // QuackGraph's traverseInterval usually checks edge validity or target node validity.
      // Let's assume we are checking TARGET NODE validFrom/validTo relative to the window.

      // Target A: Inside (12:15)
      // @ts-expect-error
      await graph.addNode("note_inside", ["Note"], {}, {
        validFrom: new Date(BASE_TIME + 15 * 60 * 1000), // 12:15
        validTo: new Date(BASE_TIME + 45 * 60 * 1000)    // 12:45 (Must end before window end for strictly DURING)
      });
      
      // Target B: Before (11:00)
      // @ts-expect-error
      await graph.addNode("note_before", ["Note"], {}, { validFrom: new Date(BASE_TIME - ONE_HOUR) }); // 11:00

      // Target C: After (14:00)
      // @ts-expect-error
      await graph.addNode("note_after", ["Note"], {}, { validFrom: new Date(BASE_TIME + 2 * ONE_HOUR) }); // 14:00

      // Connect them all
      // @ts-expect-error
      await graph.addEdge("meeting", "note_inside", "HAS_NOTE", {}, { 
        validFrom: new Date(BASE_TIME + 15 * 60 * 1000),
        validTo: new Date(BASE_TIME + 45 * 60 * 1000) // Must end within window for DURING
      });
      // @ts-expect-error
      await graph.addEdge("meeting", "note_before", "HAS_NOTE", {}, { validFrom: new Date(BASE_TIME - ONE_HOUR) });
      // @ts-expect-error
      await graph.addEdge("meeting", "note_after", "HAS_NOTE", {}, { validFrom: new Date(BASE_TIME + 2 * ONE_HOUR) });

      // Define Window: 12:00 -> 13:00
      const wStart = new Date(BASE_TIME);
      const wEnd = new Date(BASE_TIME + ONE_HOUR);

      // Test: CONTAINS / DURING (Strictly inside)
      // Note: Implementation of 'contains' might vary, usually means Window contains Node.
      const inside = await chronos.findEventsDuring("meeting", wStart, wEnd, 'during');
      
      // Depending on implementation, 'during' might mean the event is during the window.
      // note_inside (12:15) is DURING [12:00, 13:00].
      expect(inside).toContain("note_inside");
      expect(inside).not.toContain("note_before");
      expect(inside).not.toContain("note_after");

      // Test: OVERLAPS (Any intersection)
      // If we had an event spanning 11:30 -> 12:30, it should appear.
      // note_before (11:00 point) does not overlap 12:00-13:00 if it's a point event.
      const overlaps = await chronos.findEventsDuring("meeting", wStart, wEnd, 'overlaps');
      expect(overlaps).toContain("note_inside");
    });
  });

  it("evolutionaryDiff: handles out-of-order writes (Non-linear insertion)", async () => {
    await runWithTestGraph(async (graph) => {
      const tools = new GraphTools(graph);
      const chronos = new Chronos(graph, tools);

      const t1 = new Date("2024-01-01T00:00:00Z");
      const t2 = new Date("2024-02-01T00:00:00Z");
      const t3 = new Date("2024-03-01T00:00:00Z");

      // @ts-expect-error
      await graph.addNode("anchor", ["Entity"], {});
      // @ts-expect-error
      await graph.addNode("target", ["Entity"], {});

      // 1. Write T1 state (First) - Edge valid from T1 to just after T1
      await graph.addEdge("anchor", "target", "LINK", {}, { 
        validFrom: t1, 
        validTo: new Date(t1.getTime() + 1000) // Closed 1 second after T1
      });

      // 2. Write T3 state (Second) - Jump to future
      // We simulate a change in T3. Let's say we add a NEW edge type.
      await graph.addEdge("anchor", "target", "FUTURE_LINK", {}, { validFrom: t3 });

      // Now request the diff in order: T1 -> T2 -> T3
      const result = await chronos.evolutionaryDiff("anchor", [t1, t2, t3]);
      
      // T1 Snapshot: LINK exists
      const snap1 = result.timeline[0];
      expect(snap1.addedEdges.find(e => e.edgeType === "LINK")).toBeDefined();

      // T2 Snapshot: LINK should be REMOVED (because we backfilled the close)
      const snap2 = result.timeline[1];
      expect(snap2.removedEdges.find(e => e.edgeType === "LINK")).toBeDefined();
      
      // T3 Snapshot: FUTURE_LINK should be ADDED
      const snap3 = result.timeline[2];
      expect(snap3.addedEdges.find(e => e.edgeType === "FUTURE_LINK")).toBeDefined();
    });
  });

  it("analyzeCorrelation: detects patterns in high-noise environments", async () => {
    await runWithTestGraph(async (graph) => {
      const tools = new GraphTools(graph);
      const chronos = new Chronos(graph, tools);
      const windowMinutes = 60;

      // 1. Generate Noise (Background events)
      // @ts-expect-error
      await graph.addNode("root", ["System"], {});
      // Generate 50 events over the last 50 hours
      await generateTimeSeries(graph, "root", 50, 60, 50 * 60);

      // 2. Inject Correlation
      // Anchor: "Failure" at T=0 (Now)
      const now = new Date();
      // @ts-expect-error
      await graph.addNode("failure", ["Failure"], {}, { validFrom: now });

      // Target: "CPU_Spike" at T=-30m (Inside window)
      const tInside = new Date(now.getTime() - 30 * 60 * 1000);
      // @ts-expect-error
      await graph.addNode("cpu_spike", ["Metric"], { val: 99 }, { validFrom: tInside });
      
      // Target: "CPU_Spike" at T=-90m (Outside window)
      const tOutside = new Date(now.getTime() - 90 * 60 * 1000);
      // @ts-expect-error
      await graph.addNode("cpu_spike_old", ["Metric"], { val: 80 }, { validFrom: tOutside });

      const res = await chronos.analyzeCorrelation("failure", "Metric", windowMinutes);

      expect(res.sampleSize).toBe(1); // Should only catch the one inside the window
      expect(res.correlationScore).toBe(1.0);
    });
  });
});