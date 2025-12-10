import { Agent } from '@mastra/core/agent';
import { Memory } from '@mastra/memory';
import { LibSQLStore } from '@mastra/libsql';

export const judgeAgent = new Agent({
  name: 'Judge Agent',
  instructions: async ({ runtimeContext }) => {
    const asOf = runtimeContext?.get('asOf') as number | undefined;
    const timeContext = asOf ? `(As of ${new Date(asOf).toISOString()})` : '';

    return `
    You are a Judge evaluating data from a Knowledge Graph.
    
    Input provided:
    - Goal: The user's question.
    - Data: Content of the nodes found.
    - Evolution: Timeline of changes (if applicable).
    - Time Context: Relevant timeframe ${timeContext}.
    
    Task: Determine if the data answers the goal.
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