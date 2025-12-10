import { Agent } from '@mastra/core/agent';
import { Memory } from '@mastra/memory';

export const judgeAgent = new Agent({
  name: 'Judge Agent',
  instructions: `
    You are a Judge evaluating data from a Knowledge Graph.
    
    Input provided:
    - Goal: The user's question.
    - Data: Content of the nodes found.
    - Time Context: Relevant timeframe.
    
    Task: Determine if the data answers the goal.
  `,
  model: {
    id: 'groq/llama-3.3-70b-versatile',
  },
  memory: new Memory(),
});