import { Mastra } from '@mastra/core/mastra';
import { DefaultExporter } from '@mastra/core/ai-tracing';
// import { PinoLogger } from '@mastra/loggers';
import { scoutAgent } from './agents/scout-agent';
import { judgeAgent } from './agents/judge-agent';
import { routerAgent } from './agents/router-agent';
import { metabolismWorkflow } from './workflows/metabolism-workflow';

export const mastra = new Mastra({
  agents: { scoutAgent, judgeAgent, routerAgent },
  workflows: { metabolismWorkflow },
  observability: {
    default: {
      enabled: true,
      sampling: { type: 'always' },
      exporters: [new DefaultExporter()],
    },
  },
});