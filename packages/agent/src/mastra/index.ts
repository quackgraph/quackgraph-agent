import { Mastra } from '@mastra/core/mastra';
import { LibSQLStore } from '@mastra/libsql';
// import { PinoLogger } from '@mastra/loggers';
import { scoutAgent } from './agents/scout-agent';
import { judgeAgent } from './agents/judge-agent';
import { routerAgent } from './agents/router-agent';
import { metabolismWorkflow } from './workflows/metabolism-workflow';
import { labyrinthWorkflow } from './workflows/labyrinth-workflow';

export const mastra = new Mastra({
  agents: { scoutAgent, judgeAgent, routerAgent },
  workflows: { metabolismWorkflow, labyrinthWorkflow },
  storage: new LibSQLStore({
    url: ':memory:',
  }),
  observability: {
    default: {
      enabled: true,

      // exporters: [new DefaultExporter()],
    },
  },
});