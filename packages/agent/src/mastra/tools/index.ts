import { createTool } from '@mastra/core/tools';
import { z } from 'zod';
import { getGraphInstance } from '../../lib/graph-instance';
import { GraphTools } from '../../tools/graph-tools';
import { getSchemaRegistry } from '../../governance/schema-registry';
import { Chronos } from '../../agent/chronos';

// We wrap the existing GraphTools logic to make it available to Mastra agents/workflows

export const sectorScanTool = createTool({
  id: 'sector-scan',
  description: 'Get a summary of available moves (edge types) from the current nodes (LOD 0). Context aware: filters by active domain.',
  inputSchema: z.object({
    nodeIds: z.array(z.string()),
    asOf: z.number().optional(),
    allowedEdgeTypes: z.array(z.string()).optional(),
  }),
  outputSchema: z.object({
    summary: z.array(z.object({
      edgeType: z.string(),
      count: z.number(),
      avgHeat: z.number().optional(),
    })),
  }),
  execute: async ({ context, runtimeContext }) => {
    const graph = getGraphInstance();
    const tools = new GraphTools(graph);

    // 1. Resolve Context
    // @ts-expect-error - Runtime context typing overlap
    const ctxAsOf = (runtimeContext?.asOf || runtimeContext?.get?.('asOf')) as number | undefined;
    // @ts-expect-error - Runtime context typing overlap
    const ctxDomain = (runtimeContext?.domain || runtimeContext?.get?.('domain')) as string | undefined;
    
    // Prioritize tool input (if agent explicitly sets it), fallback to runtime context
    const asOf = context.asOf ?? ctxAsOf;

    // 2. Resolve Governance
    const registry = getSchemaRegistry();
    const domainEdges = ctxDomain ? registry.getValidEdges(ctxDomain) : undefined;
    const effectiveAllowed = context.allowedEdgeTypes ?? domainEdges;

    const summary = await tools.getSectorSummary(context.nodeIds, asOf, effectiveAllowed);
    return { summary };
  },
});

export const topologyScanTool = createTool({
  id: 'topology-scan',
  description: 'Get IDs of neighbors reachable via a specific edge type (LOD 1)',
  inputSchema: z.object({
    nodeIds: z.array(z.string()),
    edgeType: z.string().optional(),
    asOf: z.number().optional(),
    minValidFrom: z.number().optional(),
    depth: z.number().min(1).max(4).optional(),
  }),
  outputSchema: z.object({
    neighborIds: z.array(z.string()).optional(),
    map: z.string().optional(),
    truncated: z.boolean().optional(),
  }),
  execute: async ({ context, runtimeContext }) => {
    const graph = getGraphInstance();
    const tools = new GraphTools(graph);
    
    // Resolve Context
    // @ts-expect-error - Runtime context typing overlap
    const ctxAsOf = (runtimeContext?.asOf || runtimeContext?.get?.('asOf')) as number | undefined;
    // @ts-expect-error - Runtime context typing overlap
    const ctxDomain = (runtimeContext?.domain || runtimeContext?.get?.('domain')) as string | undefined;

    const asOf = context.asOf ?? ctxAsOf;
    
    // Enforce Domain Governance if implicit
    if (ctxDomain && context.edgeType) {
      const registry = getSchemaRegistry();
      if (!registry.isEdgeAllowed(ctxDomain, context.edgeType)) {
        return { neighborIds: [] }; // Silently block restricted edges
      }
    }

    if (context.depth && context.depth > 1) {
      // Ghost Map Mode (LOD 1.5)
      const maps = [];
      let truncated = false;
      for (const id of context.nodeIds) {
        // Note: NavigationalMap internal logic might need asOf update in future, currently uses standard scan
        const res = await tools.getNavigationalMap(id, context.depth, asOf);
        maps.push(res.map);
        if (res.truncated) truncated = true;
      }
      return { map: maps.join('\n\n'), truncated };
    }

    // Implicit map mode if no edgeType is provided, defaulting to depth 1 map
    if (!context.edgeType) {
        const maps = [];
        for (const id of context.nodeIds) {
            const res = await tools.getNavigationalMap(id, 1, asOf);
            maps.push(res.map);
        }
        return { map: maps.join('\n\n') };
    }

    const neighborIds = await tools.topologyScan(context.nodeIds, context.edgeType, asOf, context.minValidFrom);
    return { neighborIds };
  },
});

export const temporalScanTool = createTool({
  id: 'temporal-scan',
  description: 'Find neighbors connected via edges overlapping a specific time window',
  inputSchema: z.object({
    nodeIds: z.array(z.string()),
    windowStart: z.string().describe('ISO Date String'),
    windowEnd: z.string().describe('ISO Date String'),
    edgeType: z.string().optional(),
    constraint: z.enum(['overlaps', 'contains', 'during', 'meets']).optional().default('overlaps'),
  }),
  outputSchema: z.object({
    neighborIds: z.array(z.string()),
  }),
  execute: async ({ context, runtimeContext }) => {
    const graph = getGraphInstance();
    const tools = new GraphTools(graph);
    
    // Enforce Governance
    // @ts-expect-error - Runtime context typing overlap
    const ctxDomain = (runtimeContext?.domain || runtimeContext?.get?.('domain')) as string | undefined;

    if (ctxDomain && context.edgeType) {
       const registry = getSchemaRegistry();
       if (!registry.isEdgeAllowed(ctxDomain, context.edgeType)) {
         return { neighborIds: [] };
       }
    }
    
    const s = new Date(context.windowStart).getTime();
    const e = new Date(context.windowEnd).getTime();
    const neighborIds = await tools.temporalScan(context.nodeIds, s, e, context.edgeType, context.constraint);
    return { neighborIds };
  },
});

export const contentRetrievalTool = createTool({
  id: 'content-retrieval',
  description: 'Retrieve full content for nodes, including virtual spine expansion (LOD 2)',
  inputSchema: z.object({
    nodeIds: z.array(z.string()),
  }),
  outputSchema: z.object({
    content: z.array(z.record(z.any())),
  }),
  execute: async ({ context }) => {
    const graph = getGraphInstance();
    const tools = new GraphTools(graph);
    const content = await tools.contentRetrieval(context.nodeIds);
    return { content };
  },
});

export const evolutionaryScanTool = createTool({
  id: 'evolutionary-scan',
  description: 'Analyze how the topology around a node changed over specific timepoints (LOD 4 - Time). Useful for trend analysis.',
  inputSchema: z.object({
    nodeId: z.string(),
    timestamps: z.array(z.string()).describe('ISO Date Strings to compare (e.g. ["2020-01-01", "2021-01-01"])'),
  }),
  outputSchema: z.object({
    timeline: z.array(z.object({
      timestamp: z.string(),
      added: z.array(z.string()),
      removed: z.array(z.string()),
      persisted: z.array(z.string()),
      densityChange: z.string()
    })),
    summary: z.string()
  }),
  execute: async ({ context }) => {
    const graph = getGraphInstance();
    const tools = new GraphTools(graph);
    const chronos = new Chronos(graph, tools);

    const dates = context.timestamps.map(t => new Date(t));
    const result = await chronos.evolutionaryDiff(context.nodeId, dates);

    // Format for LLM consumption
    const timeline = result.timeline.map(t => ({
      timestamp: t.timestamp.toISOString(),
      added: t.addedEdges.map(e => `${e.edgeType} (${e.count})`),
      removed: t.removedEdges.map(e => `${e.edgeType} (${e.count})`),
      persisted: t.persistedEdges.map(e => `${e.edgeType} (${e.count})`),
      densityChange: `${t.densityChange.toFixed(1)}%`
    }));

    const summary = `Evolution of ${context.nodeId} across ${dates.length} points. Net density change: ${timeline[timeline.length - 1]?.densityChange || '0%'}.`;

    return { timeline, summary };
  }
});