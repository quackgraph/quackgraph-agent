export enum ZoomLevel {
  SECTOR = 0,    // Ghost/Satellite View: Available Moves (Schema)
  TOPOLOGY = 1,  // Drone View: Structural Hops (IDs only)
  CONTENT = 2    // Street View: Full JSON Data
}

// Type alias for Mastra Agent - imports the actual Agent type from @mastra/core
import type { Agent, ToolsInput } from '@mastra/core/agent';
import type { Metric } from '@mastra/core/eval';
import { z } from 'zod';
import { RouterDecisionSchema, ScoutDecisionSchema, JudgeDecisionSchema } from './agent-schemas';

// Re-export as an alias for cleaner internal usage
export type MastraAgent = Agent<string, ToolsInput, Record<string, Metric>>;

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

export interface SectorSummary {
  edgeType: string;
  count: number;
  avgHeat?: number;
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
  nodeContent: Record<string, any>[];
  timeContext?: string;
}

// --- Traces ---

export interface LabyrinthArtifact {
  answer: string;
  confidence: number;
  traceId: string;
  sources: string[];
  metadata?: Record<string, any>;
}

// Temporal Logic Types
export interface TemporalWindow {
  anchorNodeId: string;
  windowStart: number; // Unix timestamp
  windowEnd: number;   // Unix timestamp
}

export interface CorrelationResult {
  anchorLabel: string;
  targetLabel: string;
  windowSizeMinutes: number;
  correlationScore: number; // 0.0 - 1.0
  sampleSize: number;
  description: string;
}

export interface EvolutionResult {
  anchorNodeId: string;
  timeline: TimeStepDiff[];
}

export interface TimeStepDiff {
  timestamp: Date;
  // Comparison vs previous step (or baseline)
  addedEdges: SectorSummary[];
  removedEdges: SectorSummary[];
  persistedEdges: SectorSummary[];
  densityChange: number; // percentage
}