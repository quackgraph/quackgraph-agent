import { describe, it, expect, beforeEach, afterEach } from "bun:test";
import { Chronos } from "../../src/agent/chronos";
import { GraphTools } from "../../src/tools/graph-tools";
import { runWithTestGraph } from "../utils/test-graph";
import { generateTimeSeries } from "../utils/generators";
import type { QuackGraph } from "@quackgraph/graph";

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
      await graph.addNode("note_inside", ["Note"], {}, new Date(BASE_TIME + 15 * 60 * 1000)); // 12:15
      
      // Target B: Before (11:00)
      // @ts-expect-error
      await graph.addNode("note_before", ["Note"], {}, new Date(BASE_TIME - ONE_HOUR)); // 11:00

      // Target C: After (14:00)
      // @ts-expect-error
      await graph.addNode("note_after", ["Note"], {}, new Date(BASE_TIME + 2 * ONE_HOUR)); // 14:00

      // Connect them all
      // @ts-expect-error
      await graph.addEdge("meeting", "note_inside", "HAS_NOTE", {});
      // @ts-expect-error
      await graph.addEdge("meeting", "note_before", "HAS_NOTE", {});
      // @ts-expect-error
      await graph.addEdge("meeting", "note_after", "HAS_NOTE", {});

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

      const t1 = new Date("2024-01-01");
      const t2 = new Date("2024-02-01");
      const t3 = new Date("2024-03-01");

      // @ts-expect-error
      await graph.addNode("anchor", ["Entity"], {});
      // @ts-expect-error
      await graph.addNode("target", ["Entity"], {});

      // 1. Write T1 state (First)
      // @ts-expect-error
      await graph.addEdge("anchor", "target", "LINK", {}, t1);

      // 2. Write T3 state (Second) - Jump to future
      // We simulate a change in T3. Let's say we add a NEW edge type.
      // @ts-expect-error
      await graph.addEdge("anchor", "target", "FUTURE_LINK", {}, t3);

      // 3. Write T2 state (Last) - Backfill
      // In T2, let's say the first LINK was closed.
      // We manually simulate this by updating the T1 edge's valid_to to be before T2.
      // But since we are "writing late", we execute a DB update.
      // @ts-expect-error
      await graph.db.execute(
        "UPDATE edges SET valid_to = ? WHERE type = 'LINK'",
        [new Date(t1.getTime() + 1000).toISOString()] // Ends right after T1
      );

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
      await graph.addNode("failure", ["Failure"], {}, now);

      // Target: "CPU_Spike" at T=-30m (Inside window)
      const tInside = new Date(now.getTime() - 30 * 60 * 1000);
      // @ts-expect-error
      await graph.addNode("cpu_spike", ["Metric"], { val: 99 }, tInside);
      
      // Target: "CPU_Spike" at T=-90m (Outside window)
      const tOutside = new Date(now.getTime() - 90 * 60 * 1000);
      // @ts-expect-error
      await graph.addNode("cpu_spike_old", ["Metric"], { val: 80 }, tOutside);

      const res = await chronos.analyzeCorrelation("failure", "Metric", windowMinutes);

      expect(res.sampleSize).toBe(1); // Should only catch the one inside the window
      expect(res.correlationScore).toBe(1.0);
    });
  });
});