import { Agent } from '@mastra/core/agent';
import { Memory } from '@mastra/memory';
import { LibSQLStore } from '@mastra/libsql';

export const routerAgent = new Agent({
  name: 'Router Agent',
  instructions: `
    You are a Semantic Router for a Knowledge Graph.
    
    Task: Select the single most relevant domain (lens) to conduct the search based on the user's goal.
    
    Input provided:
    - Goal: User query.
    - Available Domains: List of domains and descriptions.
  `,
  model: {
    id: 'groq/llama-3.3-70b-versatile',
  },
  memory: new Memory({
    storage: new LibSQLStore({
      url: ':memory:'
    })
  }),
});