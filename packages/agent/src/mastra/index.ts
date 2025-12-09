import { Mastra } from '@mastra/core/mastra';
import { scoutAgent } from './agents/scout-agent';
import { judgeAgent } from './agents/judge-agent';
import { routerAgent } from './agents/router-agent';
import { metabolismWorkflow } from './workflows/metabolism-workflow';

export const mastra = new Mastra({
  agents: { scoutAgent, judgeAgent, routerAgent },
  workflows: { metabolismWorkflow },
});