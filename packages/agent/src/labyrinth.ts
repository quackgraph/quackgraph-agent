import type { QuackGraph } from '@quackgraph/graph';
import type {
  AgentConfig,
  LabyrinthArtifact,
  CorrelationResult,
  TimeContext,
  DomainConfig,
  MastraAgent
} from './types';
import { trace, type Span } from '@opentelemetry/api';

import { setGraphInstance } from './lib/graph-instance';
import { mastra } from './mastra';

// Tools / Helpers
import { Chronos } from './agent/chronos';
import { GraphTools } from './tools/graph-tools';
import { SchemaRegistry } from './governance/schema-registry';

export class Labyrinth {
  public chronos: Chronos;
  public tools: GraphTools;
  public registry: SchemaRegistry;
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
    setGraphInstance(graph);
    this.tools = new GraphTools(graph);
    this.chronos = new Chronos(graph, this.tools);
    this.registry = new SchemaRegistry();
  }

  registerDomain(config: DomainConfig) {
    this.registry.register(config);
  }

  async findPath(
    start: string | { query: string },
    goal: string,
    timeContext?: TimeContext
  ): Promise<LabyrinthArtifact | null> {
    return this.tracer.startActiveSpan('labyrinth.facade', async (span: Span) => {
        try {
            const workflow = mastra.getWorkflow('labyrinthWorkflow');
            if (!workflow) throw new Error("Labyrinth Workflow not registered");

            const run = await workflow.createRunAsync();
            
            // Map input
            const inputData = {
                goal,
                start,
                domain: undefined, // Let workflow route it
                maxHops: this.config.maxHops,
                maxCursors: this.config.maxCursors,
                confidenceThreshold: this.config.confidenceThreshold,
                timeContext: timeContext ? {
                    asOf: timeContext.asOf?.getTime(),
                    windowStart: timeContext.windowStart?.toISOString(),
                    windowEnd: timeContext.windowEnd?.toISOString()
                } : undefined
            };

            // Pass runtime context trigger data if needed, mostly handled via inputData for now
            // but we ensure the run is started cleanly.
            const result = await run.start({ inputData });
            
            // @ts-expect-error - Result payload typing
            return result.results?.artifact || null;
            
        } catch (e) {
            this.logger.error("Labyrinth workflow failed", { error: e });
            throw e;
        } finally {
            span.end();
        }
    });
  }

  async analyzeCorrelation(
    anchorNodeId: string,
    targetLabel: string,
    windowMinutes: number
  ): Promise<CorrelationResult> {
    return this.chronos.analyzeCorrelation(anchorNodeId, targetLabel, windowMinutes);
  }
}