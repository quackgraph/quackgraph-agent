import type { AgentConfig, DomainConfig, RouterDecision } from '../types';

export class Router {
  constructor(private config: AgentConfig) {}

  /**
   * Semantic Routing: Determines which Domain governs the user's goal.
   * "Ghost Earth" Protocol: This selects the lens through which we view the graph.
   */
  async route(goal: string, domains: DomainConfig[], signal?: AbortSignal): Promise<RouterDecision> {
    if (domains.length <= 1) {
      // Trivial case
      return { 
        domain: domains[0]?.name || 'global', 
        confidence: 1.0, 
        reasoning: 'Only one domain available.' 
      };
    }

    const domainDescriptions = domains
      .map(d => `- "${d.name}": ${d.description}`)
      .join('\n');

    const prompt = `
      You are a Semantic Router for a Knowledge Graph.
      Goal: "${goal}"
      
      Available Domains (Lenses):
      ${domainDescriptions}
      
      Task: Select the single most relevant domain to conduct this search.
      If the goal is broad or doesn't fit specific domains, choose 'global'.
      
      Return ONLY a JSON object:
      {
        "domain": "medical",
        "confidence": 0.95,
        "reasoning": "The query mentions symptoms and medication."
      }
    `;

    try {
      const raw = await this.config.llmProvider.generate(prompt, signal);
      const jsonStr = raw.match(/\{[\s\S]*\}/)?.[0] || raw;
      const decision = JSON.parse(jsonStr) as RouterDecision;
      
      // Validate
      const validName = domains.find(d => d.name.toLowerCase() === decision.domain.toLowerCase());
      if (!validName) {
        return { domain: 'global', confidence: 0.0, reasoning: 'LLM returned invalid domain.' };
      }
      
      return decision;
    } catch (e) {
      return { domain: 'global', confidence: 0.0, reasoning: 'Routing failed.' };
    }
  }
}