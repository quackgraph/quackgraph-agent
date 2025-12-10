// Core Facade
export { Labyrinth } from './labyrinth';

// Types & Schemas
export * from './types';
export * from './agent-schemas';

// Utilities
export * from './agent/chronos';
export * from './governance/schema-registry';

// Mastra Internals (Exposed for advanced configuration)
export { mastra } from './mastra';
export { labyrinthWorkflow } from './mastra/workflows/labyrinth-workflow';

// Factory
import type { QuackGraph } from '@quackgraph/graph';
import { Labyrinth } from './labyrinth';
import type { AgentConfig } from './types';
import { mastra } from './mastra';

/**
 * Factory to create a fully wired Labyrinth Agent.
 * Checks for required Mastra agents (Scout, Judge, Router) before instantiation.
 */
export function createAgent(graph: QuackGraph, config: AgentConfig) {
  const scout = mastra.getAgent('scoutAgent');
  const judge = mastra.getAgent('judgeAgent');
  const router = mastra.getAgent('routerAgent');

  if (!scout || !judge || !router) {
    const missing = [];
    if (!scout) missing.push('scoutAgent');
    if (!judge) missing.push('judgeAgent');
    if (!router) missing.push('routerAgent');
    throw new Error(
      `Failed to create QuackGraph Agent. Required Mastra agents are missing: ${missing.join(', ')}. ` +
      `Ensure you have imported 'mastra' from this package and registered these agents.`
    );
  }

  return new Labyrinth(
    graph,
    { scout, judge, router },
    config
  );
}