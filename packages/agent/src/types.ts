export enum ZoomLevel {
  SECTOR = 0,    // Ghost/Satellite View: Available Moves (Schema)
  TOPOLOGY = 1,  // Drone View: Structural Hops (IDs only)
  CONTENT = 2    // Street View: Full JSON Data
}

// Type alias for Mastra Agent - imports the actual Agent type from @mastra/core
import type { Agent, ToolsInput } from '@mastra/core/agent';
import type { Metric } from '@mastra/core/eval';
import type { z } from 'zod';
import type { RouterDecisionSchema, ScoutDecisionSchema } from './agent-schemas';
import type { SectorSummary, CorrelationResult, EvolutionResult, TimeStepDiff } from '@quackgraph/graph';

// Re-export as an alias for cleaner internal usage
export type MastraAgent = Agent<string, ToolsInput, Record<string, Metric>>;

// Labyrinth Runtime Context (Injected via Mastra)
export interface LabyrinthContext {
  // Temporal: The "Now" for the graph traversal (Unix Timestamp in seconds or Date)
  // If undefined, defaults to real-time.
  asOf?: number | Date;

  // Governance: The semantic lens restricting traversal (e.g., "Medical", "Financial")
  domain?: string;

  // Traceability: The distributed trace ID for this execution
  traceId?: string;

  // Threading: The specific cursor/thread ID for parallel speculation
  threadId?: string;
}

export interface AgentConfig {
  llmProvider: {
    generate: (prompt: string, signal?: AbortSignal) => Promise<string>;
  };
  maxHops?: number;
  // Max number of concurrent exploration threads
  maxCursors?: number;
  // Minimum confidence to pursue a path (0.0 - 1.0)
  confidenceThreshold?: number;
  // Vector Genesis: Optional embedding provider for start-from-text
  embeddingProvider?: {
    embed: (text: string) => Promise<number[]>;
  };
}

// --- Governance & Router ---

export interface DomainConfig {
  name: string;
  description: string;
  allowedEdges: string[]; // Whitelist of edge types (empty = all unless excluded)
  excludedEdges?: string[]; // Blacklist of edge types (overrides allowed)
  // If true, traversal enforces Monotonic Time (Next Event >= Current Event)
  isCausal?: boolean;
}

export interface RouterPrompt {
  goal: string;
  availableDomains: DomainConfig[];
}

export type RouterDecision = z.infer<typeof RouterDecisionSchema>;

// --- Scout ---

export interface TimeContext {
  asOf?: Date;
  windowStart?: Date;
  windowEnd?: Date;
}

export type ScoutDecision = z.infer<typeof ScoutDecisionSchema>;

export interface ScoutPrompt {
  goal: string;
  activeDomain: string; // The semantic domain grounding this search
  currentNodeId: string;
  currentNodeLabels: string[];
  sectorSummary: SectorSummary[];
  pathHistory: string[];
  timeContext?: string;
}

export interface JudgePrompt {
  goal: string;
  // biome-ignore lint/suspicious/noExplicitAny: generic content
  nodeContent: Record<string, any>[];
  timeContext?: string;
}

// --- Traces ---

export interface LabyrinthArtifact {
  answer: string;
  confidence: number;
  traceId: string;
  sources: string[];
  metadata?: LabyrinthMetadata;
}

export interface LabyrinthMetadata {
  duration_ms: number;
  tokens_used: number;
  governance: {
    query: string;
    selected_domain: string;
    rejected_domains: string[];
    reasoning: string;
  };
  execution: ThreadTrace[];
  judgment?: {
    verdict: string;
    confidence: number;
  };
}

export interface ThreadTrace {
  thread_id: string;
  status: 'COMPLETED' | 'KILLED' | 'ACTIVE';
  steps: {
    step: number;
    node_id: string;
    ghost_view?: string; // Snapshot of what the agent saw
    action: string;
    reasoning: string;
  }[];
}

// Temporal Logic Types
export interface TemporalWindow {
  anchorNodeId: string;
  windowStart: number; // Unix timestamp
  windowEnd: number;   // Unix timestamp
}
export type { SectorSummary, CorrelationResult, EvolutionResult, TimeStepDiff };

export interface StepEvent {
  step: number;
  node_id: string;
  ghost_view?: string;
  action: string;
  reasoning: string;
}

export interface LabyrinthCursor {
  id: string;
  currentNodeId: string;
  path: string[];
  pathEdges: (string | undefined)[];
  stepHistory: StepEvent[];
  stepCount: number;
  confidence: number;
  lastEdgeType?: string;
  lastTimestamp?: number;
}