import type { AgentConfig, JudgePrompt, LabyrinthArtifact } from '../types';

export class Judge {
  constructor(private config: AgentConfig) {}

  async evaluate(promptCtx: JudgePrompt, signal?: AbortSignal): Promise<LabyrinthArtifact | null> {
    const timeInfo = promptCtx.timeContext ? `Time Context: ${promptCtx.timeContext}` : '';
    
    const prompt = `
      You are a Judge evaluating data.
      Goal: "${promptCtx.goal}"
      ${timeInfo}
      Data: ${JSON.stringify(promptCtx.nodeContent)}
      
      Does this data answer the goal?
      Return ONLY a JSON object:
      { "isAnswer": true, "answer": "The user is...", "confidence": 0.95 }
    `;

    try {
      const raw = await this.config.llmProvider.generate(prompt, signal);
      const jsonStr = raw.match(/\{[\s\S]*\}/)?.[0] || raw;
      // biome-ignore lint/suspicious/noExplicitAny: Weak schema for LLM response
      const result = JSON.parse(jsonStr) as any;
      
      if (result.isAnswer) {
        return {
          answer: result.answer,
          confidence: result.confidence,
          traceId: '', // Filled by caller
          sources: [] // Filled by caller
        };
      }
      return null;
    } catch (e) {
      return null;
    }
  }
}