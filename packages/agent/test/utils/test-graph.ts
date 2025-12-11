import { QuackGraph } from '@quackgraph/graph';
import { setGraphInstance } from '../../src/lib/graph-instance';

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

  // Set as global instance for the test context (so tools can pick it up)
  setGraphInstance(graph);

  return graph;
}