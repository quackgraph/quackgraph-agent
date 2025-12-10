import type { QuackGraph } from '@quackgraph/graph';
import type {
  AgentConfig,
  LabyrinthArtifact,
  TimeContext,
  DomainConfig,
  MastraAgent
} from './types';
import { trace, type Span } from '@opentelemetry/api';

// Core Dependencies
import { setGraphInstance } from './lib/graph-instance';
import { mastra } from './mastra';
import { schemaRegistry } from './governance/schema-registry';

/**
 * The QuackGraph Agent Facade.
 * 
 * A Native Mastra implementation.
 * This class acts as a thin client that orchestrates the `labyrinth-workflow` 
 * and injects the RuntimeContext (Time Travel & Governance).
 */
export class Labyrinth {
  private logger = mastra.getLogger();
  private tracer = trace.getTracer('quackgraph-agent');

  constructor(
    graph: QuackGraph,
    _agents: {
      scout: MastraAgent;
      judge: MastraAgent;
      router: MastraAgent;
    },
    private config: AgentConfig
  ) {
    // Bridge Pattern: Inject the graph instance into the global scope
    // so Mastra Tools can access it without passing it through every step.
    setGraphInstance(graph);
  }

  /**
   * Registers a semantic domain (LOD 0 governance).
   * Direct proxy to the singleton registry used by tools.
   */
  registerDomain(config: DomainConfig) {
    schemaRegistry.register(config);
  }

  /**
   * Main Entry Point: Finds a path through the Labyrinth.
   * 
   * @param start - Starting Node ID or natural language query
   * @param goal - The question to answer
   * @param timeContext - "Time Travel" parameters (asOf, window)
   */
  async findPath(
    start: string | { query: string },
    goal: string,
    timeContext?: TimeContext
  ): Promise<LabyrinthArtifact | null> {
    return this.tracer.startActiveSpan('labyrinth.findPath', async (span: Span) => {
        try {
            const workflow = mastra.getWorkflow('labyrinthWorkflow');
            if (!workflow) throw new Error("Labyrinth Workflow not registered in Mastra.");

            // 1. Prepare Input Data & Configuration
            const inputData = {
                goal,
                start,
                maxHops: this.config.maxHops,
                maxCursors: this.config.maxCursors,
                confidenceThreshold: this.config.confidenceThreshold,
                timeContext: timeContext ? {
                    asOf: timeContext.asOf instanceof Date ? timeContext.asOf.getTime() : timeContext.asOf,
                    windowStart: timeContext.windowStart?.toISOString(),
                    windowEnd: timeContext.windowEnd?.toISOString()
                } : undefined
            };

            // 2. Execute Workflow
            const run = await workflow.createRunAsync();
            
            // The workflow steps are responsible for extracting timeContext from input
            // and passing it to agents via runtimeContext injection in the 'speculative-traversal' step.
            const result = await run.start({ inputData });
            
            // 3. Extract Result
            // @ts-expect-error - Result payload typing
            const artifact = result.results?.artifact as LabyrinthArtifact | null;
            
            if (artifact) {
              span.setAttribute('labyrinth.confidence', artifact.confidence);
              span.setAttribute('labyrinth.traceId', artifact.traceId);
            }

            return artifact;

        } catch (e) {
            this.logger.error("Labyrinth traversal failed", { error: e });
            span.recordException(e as Error);
            throw e;
        } finally {
            span.end();
        }
    });
  }
}