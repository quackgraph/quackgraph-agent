import type { QuackGraph } from '@quackgraph/graph';

/**
 * Simulates data corruption by injecting invalid or unexpected schema data directly into the DB.
 */
export async function corruptNode(graph: QuackGraph, nodeId: string) {
    // We assume properties are stored as JSON string in 'nodes' table.
    // We inject a string that is technically valid JSON but breaks expected schema,
    // or invalid JSON if the DB allows (SQLite/DuckDB often allows loose text).
    
    const badData = '{"corrupted": true, "critical_field": null, "unexpected_array": [1,2,3]}';
    
    // Direct SQL injection to bypass application-layer validation
    await graph.db.execute(
        `UPDATE nodes SET properties = ? WHERE id = ?`,
        [badData, nodeId]
    );
}

/**
 * Simulates a network partition or edge loss.
 * Can be used to test resilience against missing links.
 */
export async function severConnection(graph: QuackGraph, source: string, target: string, type: string) {
    // 1. Soft Delete (Time Travel)
    const now = new Date().toISOString();
    await graph.db.execute(
        `UPDATE edges SET valid_to = ? WHERE source = ? AND target = ? AND type = ? AND valid_to IS NULL`,
        [now, source, target, type]
    );
    
    // 2. Force removal from RAM index if applicable to simulation
    // (Assuming graph.native has a remove method exposed or we rely on reload)
    try {
        // @ts-expect-error - native method might vary
        if (graph.native.removeEdge) {
            // @ts-expect-error
            graph.native.removeEdge(source, target, type);
        }
    } catch (e) {
        console.warn("Could not remove edge from native index manually:", e);
    }
}