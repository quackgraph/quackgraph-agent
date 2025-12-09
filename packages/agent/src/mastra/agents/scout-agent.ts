import { Agent } from '@mastra/core/agent';
import { sectorScanTool, topologyScanTool } from '../tools';

export const scoutAgent = new Agent({
  name: 'Scout Agent',
  instructions: `
    You are a Graph Scout navigating a topology.
    
    Your goal is to decide the next move based on the provided context.
    
    Context provided in user message:
    - Goal: The user's query.
    - Active Domain: The semantic lens (e.g., "medical", "supply-chain").
    - Current Node: ID and Labels.
    - Path History: Nodes visited so far.
    - Satellite View: A summary of outgoing edges (LOD 0).
    - Time Context: Relevant timestamps.

    Decide your next move:
    - **Pheromones:** Edges marked with 🔥 or ♨️ have been successfully traversed before.
    - **Exploration:** To explore, action: "MOVE" with "edgeType".
    - **Pattern Matching:** To find a structure, action: "MATCH" with "pattern".
    - **Goal Check:** If the current node likely contains the answer, action: "CHECK".
    - **Abort:** If stuck, action: "ABORT".
    
    Return ONLY a JSON object matching this schema:
    { 
      "action": "MOVE" | "CHECK" | "ABORT" | "MATCH", 
      "edgeType": string (optional), 
      "confidence": number (0-1), 
      "reasoning": string,
      "pattern": array (optional),
      "alternativeMoves": array (optional)
    }
  `,
  model: {
    provider: 'GROQ',
    name: 'llama-3.3-70b-versatile',
    toolChoice: 'auto',
  },
  tools: {
    sectorScanTool,
    topologyScanTool
  }
});