import { Agent } from '@mastra/core/agent';
import { Memory } from '@mastra/memory';
import { LibSQLStore } from '@mastra/libsql';

export const routerAgent = new Agent({
  name: 'Router Agent',
  instructions: async ({ runtimeContext }) => {
    const forcedDomain = runtimeContext?.get('domain') as string | undefined;
    const domainHint = forcedDomain ? `\nHint: The system has pre-selected '${forcedDomain}', verify if this applies.` : '';

    return `
    You are a Semantic Router for a Knowledge Graph.
    
    Task: Select the single most relevant domain (lens) to conduct the search based on the user's goal.
    
    Input provided:
    - Goal: User query.
    - Available Domains: List of domains and descriptions.${domainHint}
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
});