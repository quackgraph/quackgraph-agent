import type { AgentConfig, ScoutDecision, ScoutPrompt } from '../types';

export class Scout {
  constructor(private config: AgentConfig) {}

  async decide(promptCtx: ScoutPrompt): Promise<ScoutDecision> {
    // Helper to format heat cues
    const getHeatIcon = (heat?: number) => {
      if (!heat) return '';
      if (heat > 150) return ' 🔥 (High Heat)';
      if (heat > 50) return ' ♨️ (Warm)';
      return ' ❄️ (Cold)';
    };

    const summaryList = promptCtx.sectorSummary
      .map(s => `- ${s.edgeType}: ${s.count} nodes${getHeatIcon(s.avgHeat)}`)
      .join('\n');

    const timeInfo = promptCtx.timeContext ? `Time Context: ${promptCtx.timeContext}` : '';

    const prompt = `
      You are a Graph Scout navigating a blind topology.
      Goal: "${promptCtx.goal}"
      ${timeInfo}
      Current Node: ${promptCtx.currentNodeId} (Labels: ${promptCtx.currentNodeLabels.join(', ')})
      Path History: ${promptCtx.pathHistory.join(' -> ')}
      
      Satellite View (Available Moves):
      ${summaryList}
      
      Decide your next move.
      - **Pheromones:** Edges marked with 🔥 or ♨️ have been successfully traversed before. Prioritize them if relevant.
      - **Exploration:** If you want to explore, action: "MOVE" and specify the "edgeType".
      - **Reasonable Counts:** Avoid exploring >10,000 nodes unless you are zooming out.
      - **Goal Check:** If you strongly believe this current node contains the answer, action: "CHECK".
      - If stuck, "ABORT".
      - If you see multiple promising paths, you can provide "alternativeMoves".
      
      Return ONLY a JSON object:
      { 
        "action": "MOVE", 
        "edgeType": "KNOWS", 
        "confidence": 0.9, 
        "reasoning": "...",
        "alternativeMoves": [
           { "edgeType": "WORKS_WITH", "confidence": 0.5, "reasoning": "..." }
        ]
      }
    `;

    try {
      const raw = await this.config.llmProvider.generate(prompt);
      const jsonStr = raw.match(/\{[\s\S]*\}/)?.[0] || raw;
      return JSON.parse(jsonStr) as ScoutDecision;
    } catch (e) {
      return { action: 'ABORT', confidence: 0, reasoning: 'LLM Parsing Error' };
    }
  }
}