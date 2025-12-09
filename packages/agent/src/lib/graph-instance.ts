import type { QuackGraph } from '@quackgraph/graph';

let graphInstance: QuackGraph | null = null;

export function setGraphInstance(graph: QuackGraph) {
  graphInstance = graph;
}

export function getGraphInstance(): QuackGraph {
  if (!graphInstance) {
    throw new Error('Graph instance not initialized. Call setGraphInstance() first.');
  }
  return graphInstance;
}