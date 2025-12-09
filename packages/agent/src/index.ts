export * from './labyrinth';
export * from './types';
export * from './agent/chronos';
export * from './governance/schema-registry';
export * from './tools/graph-tools';
// Expose Mastra definitions if needed, but facade prefers hiding them
export { mastra } from './mastra';

import { QuackGraph } from '@quackgraph/graph';
import { Labyrinth } from './labyrinth';
import { AgentConfig } from './types';
import { mastra } from './mastra';

/**
 * Factory to create a fully wired Labyrinth Agent.
 * Uses default Mastra agents (Scout, Judge, Router) unless overridden.
 */
export function createAgent(graph: QuackGraph, config: AgentConfig) {
  return new Labyrinth(
    graph,
    {
      scout: mastra.getAgent('scoutAgent'),
      judge: mastra.getAgent('judgeAgent'),
      router: mastra.getAgent('routerAgent'),
    },
    config
  );
}

/**
 * Runs the Metabolism (Dreaming) cycle to prune and summarize old nodes.
 */
export async function runMetabolism(targetLabel: string, minAgeDays: number = 30) {
    const workflow = mastra.getWorkflow('metabolismWorkflow');
    if (!workflow) throw new Error("Metabolism workflow not found.");
    return await workflow.execute({ triggerData: { targetLabel, minAgeDays } });
}