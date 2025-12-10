import { Agent } from '@mastra/core/agent';
import { Memory } from '@mastra/memory';
import { LibSQLStore } from '@mastra/libsql';
import { topologyScanTool, contentRetrievalTool, sectorScanTool } from '../tools';

export const scribeAgent = new Agent({
  name: 'Scribe Agent',
  instructions: async ({ runtimeContext }) => {
    const asOf = runtimeContext?.get('asOf') as number | undefined;
    const timeStr = asOf ? new Date(asOf).toISOString() : new Date().toISOString();

    return `
    You are the Scribe. Your job is to MUTATE the Knowledge Graph based on user intent.
    Current System Time: ${timeStr}

    Your Goal:
    Convert natural language intentions (e.g., "I sold my car yesterday") into precise Graph Operations.

    Rules:
    1. **Entity Resolution First**: Never guess Node IDs. 
       - If the user says "my car", look up nodes connected to "Me" or "User" with type "OWNED" or label "Car" first.
       - Use \`topology-scan\` or \`content-retrieval\` to verify you have the correct ID (e.g., "car:123" vs "car:999").
    
    2. **Temporal Grounding**:
       - The graph requires ISO 8601 timestamps.
       - Interpret "yesterday", "tomorrow", "now" relative to the Current System Time.
       - For "I sold it", CLOSE the existing edge (set \`validTo\`) and optionally CREATE a new one.

    3. **Atomic Operations**:
       - Output a list of operations that represent the full state change.
       - Example: "Changed name from Bob to Robert" -> UPDATE_NODE { match: {id: "bob"}, set: {name: "Robert"}, validFrom: NOW }

    4. **Ambiguity**:
       - If you cannot find the node "Susi" or there are two "Susi" nodes, return \`requiresClarification\`. Do not write bad data.
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
    // Scribe needs to see before it writes
    topologyScanTool,
    contentRetrievalTool,
    sectorScanTool
  }
});