import { AsyncLocalStorage } from 'node:async_hooks';
import type { QuackGraph } from '@quackgraph/graph';

const graphStorage = new AsyncLocalStorage<QuackGraph>();
let globalGraphInstance: QuackGraph | null = null;

export function setGraphInstance(graph: QuackGraph) {
  globalGraphInstance = graph;
}

export function runWithGraph<T>(graph: QuackGraph, callback: () => T): T {
  return graphStorage.run(graph, callback);
}

export function enterGraphContext(graph: QuackGraph) {
  graphStorage.enterWith(graph);
}

export function getGraphInstance(): QuackGraph {
  const storeGraph = graphStorage.getStore();
  if (storeGraph) return storeGraph;

  if (!globalGraphInstance) {
    throw new Error('Graph instance not initialized. Call setGraphInstance() or run within runWithGraph context.');
  }
  return globalGraphInstance;
}