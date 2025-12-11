import { randomUUID } from 'node:crypto';
import type { QuackGraph } from '@quackgraph/graph';

/**
 * Generates a star topology (Cluster).
 * Center node connected to N leaf nodes.
 */
export async function generateCluster(
  graph: QuackGraph,
  size: number,
  centerLabel: string = 'ClusterCenter',
  leafLabel: string = 'ClusterLeaf',
  edgeType: string = 'LINKED_TO'
) {
  const centerId = `center_${randomUUID().slice(0, 8)}`;
  await graph.addNode(centerId, [centerLabel], { name: `Center ${centerId}` });

  const leafIds = [];
  const edges = [];

  for (let i = 0; i < size; i++) {
    const leafId = `leaf_${randomUUID().slice(0, 8)}`;
    leafIds.push({
        id: leafId,
        labels: [leafLabel],
        properties: { index: i, generated: true }
    });

    edges.push({
      source: centerId,
      target: leafId,
      type: edgeType,
      properties: { weight: Math.random() }
    });
  }
  
  // Batch add for performance in tests
  await graph.addNodes(leafIds);
  await graph.addEdges(edges);

  return { centerId, leafIds: leafIds.map(l => l.id) };
}

/**
 * Generates a linear sequence of events (Time Series).
 * Root node connected to N event nodes, each with incrementing timestamps.
 */
export async function generateTimeSeries(
  graph: QuackGraph,
  rootId: string,
  count: number,
  intervalMinutes: number = 60,
  startBufferMinutes: number = 0 // How long ago to start relative to NOW
) {
    const now = Date.now();
    // Start time calculated backwards if buffer is positive
    const startTime = now - (startBufferMinutes * 60 * 1000);
    
    const events = [];
    const edges = [];
    const generatedIds = [];

    for(let i=0; i<count; i++) {
        const eventId = `event_${randomUUID().slice(0, 8)}`;
        const time = new Date(startTime + (i * intervalMinutes * 60 * 1000));
        
        events.push({
            id: eventId,
            labels: ['Event', 'Log'],
            properties: { sequence: i, timestamp: time.toISOString(), val: Math.random() },
            validFrom: time
        });

        edges.push({
            source: rootId,
            target: eventId,
            type: 'HAS_EVENT',
            properties: {},
            validFrom: time
        });
        
        generatedIds.push(eventId);
    }

    await graph.addNodes(events);
    await graph.addEdges(edges);

    return { eventIds: generatedIds };
}