export * from './labyrinth';
export * from './types';
export * from './agent/chronos';
export * from './governance/schema-registry';
export * from './tools/graph-tools';
// Expose Mastra definitions if needed, but facade prefers hiding them
export { mastra } from './mastra';

import type { QuackGraph } from '@quackgraph/graph';
import { Labyrinth } from './labyrinth';
import type { AgentConfig } from './types';
import { mastra } from './mastra';

/**
 * Factory to create a fully wired Labyrinth Agent.
 * Uses default Mastra agents (Scout, Judge, Router) unless overridden.
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



/**
 * Runs the Metabolism (Dreaming) cycle to prune and summarize old nodes.
 */
export async function runMetabolism(targetLabel: string, minAgeDays: number = 30) {
  const workflow = mastra.getWorkflow('metabolismWorkflow');
  if (!workflow) throw new Error("Metabolism workflow not found.");
  const run = await workflow.createRunAsync();
  return run.start({ inputData: { targetLabel, minAgeDays } });
}