import { QuackGraph } from '@quackgraph/graph';
import { setGraphInstance, enterGraphContext, runWithGraph } from '../../src/lib/graph-instance';

/**
 * Factory for creating ephemeral, in-memory graph instances.
 * Ensures tests are isolated and idempotent.
 */
export async function createTestGraph(): Promise<QuackGraph> {
  // Initialize QuackGraph in-memory
  // Note: We assume the QuackGraph constructor supports a config object for storage.
  // If the core package implementation differs, this needs adjustment.
  const graph = new QuackGraph({
    storage: ':memory:',
    dbUrl: ':memory:', // Dual support depending on core version
  });

  // If there's an async init method, call it
  // @ts-expect-error - Checking for potential init method
  if (typeof graph.initialize === 'function') {
    // @ts-expect-error
    await graph.initialize();
  }

  // 1. Try to set isolation context for this async flow (Best effort for legacy tests)
  enterGraphContext(graph);

  // 2. Set as global instance for the test context (fallback)
  setGraphInstance(graph);

  return graph;
}

/**
 * Isolated execution wrapper for new tests.
 * Ensures graph is closed after use and strictly isolated via AsyncLocalStorage.
 */
export async function runWithTestGraph<T>(callback: (graph: QuackGraph) => Promise<T>): Promise<T> {
  const graph = new QuackGraph({
    storage: ':memory:',
    dbUrl: ':memory:',
  });

  // @ts-expect-error - Checking for potential init method
  if (typeof graph.initialize === 'function') {
    // @ts-expect-error
    await graph.initialize();
  }

  try {
    return await runWithGraph(graph, async () => {
      return await callback(graph);
    });
  } finally {
    // @ts-expect-error
    if (typeof graph.close === 'function') {
      // @ts-expect-error
      await graph.close();
    }
  }
}