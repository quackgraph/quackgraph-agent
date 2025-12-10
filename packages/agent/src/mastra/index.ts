import { Mastra } from '@mastra/core/mastra';
import { LibSQLStore } from '@mastra/libsql';
import { DefaultExporter, SensitiveDataFilter } from '@mastra/core/ai-tracing';
import { scoutAgent } from './agents/scout-agent';
import { judgeAgent } from './agents/judge-agent';
import { routerAgent } from './agents/router-agent';
import { scribeAgent } from './agents/scribe-agent';
import { metabolismWorkflow } from './workflows/metabolism-workflow';
import { labyrinthWorkflow } from './workflows/labyrinth-workflow';
import { mutationWorkflow } from './workflows/mutation-workflow';

export const mastra = new Mastra({
  agents: { scoutAgent, judgeAgent, routerAgent, scribeAgent },
  workflows: { metabolismWorkflow, labyrinthWorkflow, mutationWorkflow },
  storage: new LibSQLStore({
    url: ':memory:',
  }),
  observability: {
    configs: {
      default: {
        serviceName: 'quackgraph-agent',
        processors: [new SensitiveDataFilter()],
        exporters: [new DefaultExporter()],
      },
    },
  },
  server: {
    port: 4111,
    cors: {
      origin: '*',
      allowMethods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
      allowHeaders: ['Content-Type', 'Authorization', 'x-quack-as-of', 'x-quack-domain'],
    },
    middleware: [
      async (context, next) => {
        const asOfHeader = context.req.header('x-quack-as-of');
        const domainHeader = context.req.header('x-quack-domain');
        const runtimeContext = context.get('runtimeContext');

        if (runtimeContext) {
          if (asOfHeader) {
            const val = parseInt(asOfHeader, 10);
            if (!Number.isNaN(val)) runtimeContext.set('asOf', val);
          }
          if (domainHeader) {
            runtimeContext.set('domain', domainHeader);
          }
        }
        await next();
      },
    ],
  },
});