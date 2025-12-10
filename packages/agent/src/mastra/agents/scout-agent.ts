import { Agent } from '@mastra/core/agent';
import { Memory } from '@mastra/memory';
import { LibSQLStore } from '@mastra/libsql';
import { sectorScanTool, topologyScanTool, temporalScanTool, evolutionaryScanTool } from '../tools';

export const scoutAgent = new Agent({
  name: 'Scout Agent',
  instructions: async ({ runtimeContext }) => {
    const asOf = runtimeContext?.get('asOf') as number | undefined;
    const domain = runtimeContext?.get('domain') as string | undefined;
    const timeContext = asOf ? `Time Travel Mode: ${new Date(asOf).toISOString()}` : 'Time: Present (Real-time)';
    const domainContext = domain ? `Governance Domain: ${domain}` : 'Governance: Global/Unrestricted';

    return `
    You are a Graph Scout navigating a topology.
    System Context: [${timeContext}, ${domainContext}]

    Your goal is to decide the next move based on the provided context.
    
    Context provided in user message:
    - Goal: The user's query.
    - Active Domain: The semantic lens (e.g., "medical", "supply-chain").
    - Current Node: ID and Labels.
    - Path History: Nodes visited so far.
    - Satellite View: A summary of outgoing edges (LOD 0).
    - Time Context: Relevant timestamps.

    Decide your next move:
    - **Radar Control (Depth):** You can request a "Ghost Map" (ASCII Tree) by using \`topology-scan\` with \`depth: 2\` or \`3\`.
      - Use Depth 1 to check immediate neighbors.
      - Use Depth 2-3 to explore structure without moving.
      - The map shows "🔥" for hot paths.

    - **Time Travel:** 
      - Use \`evolutionary-scan\` with specific ISO timestamps to see how connections changed over time (e.g., "What changed between 2023 and 2024?").
    
    - **Ambiguity & Forking:**
      - If you are unsure between two paths (e.g. "Did he Author it or Review it?"), select the most likely one as your primary MOVE.
      - **CRITICAL:** Add the second option to \`alternativeMoves\`. The system will spawn parallel threads to explore both hypotheses simultaneously.

    - **Pheromones:** Edges marked with 🔥 or ♨️ have been successfully traversed before.
    - **Exploration:** 
      - Single Hop: Action "MOVE" with \`edgeType\`.
      - Multi Hop: If you see a path in the Ghost Map, Action "MOVE" with \`path: [id1, id2]\`.
    - **Pattern Matching:** To find a structure, action: "MATCH" with "pattern".
    - **Goal Check:** If the current node likely contains the answer, action: "CHECK".
    - **Abort:** If stuck or exhausted, action: "ABORT".
  `;
  },
  model: {
    id: 'groq/llama-3.3-70b-versatile',
  },
  memory: new Memory({
    storage: new LibSQLStore({
      url: ':memory:'
    })
  }),
  tools: {
    sectorScanTool,
    topologyScanTool,
    temporalScanTool,
    evolutionaryScanTool
  }
});