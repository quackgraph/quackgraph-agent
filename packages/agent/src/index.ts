// Core Facade
export { Labyrinth } from './labyrinth';
export { createAgent } from './index'; // Self-reference for factory below

// Types & Schemas
export * from './types';
export * from './agent-schemas';

// Utilities
export * from './agent/chronos';
export * from './governance/schema-registry';
export * from './tools/graph-tools';

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
    throw new Error('Required Mastra agents not found. Ensure scoutAgent, judgeAgent, and routerAgent are registered.');
  }

  return new Labyrinth(
    graph,
    { scout, judge, router },
    config
  );
}