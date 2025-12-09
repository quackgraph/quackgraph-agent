export enum ZoomLevel {
  SECTOR = 0,    // Ghost/Satellite View: Available Moves (Schema)
  TOPOLOGY = 1,  // Drone View: Structural Hops (IDs only)
  CONTENT = 2    // Street View: Full JSON Data
}

// Type alias for Mastra Agent - imports the actual Agent type from @mastra/core
import type { Agent, ToolsInput } from '@mastra/core/agent';
import type { Metric } from '@mastra/core/eval';

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

export interface RouterDecision {
  domain: string;
  confidence: number;
  reasoning: string;
}

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

export interface ScoutDecision {
  action: 'MOVE' | 'CHECK' | 'ABORT' | 'MATCH';
  edgeType?: string; // Required if action is MOVE
  pattern?: { srcVar: number; tgtVar: number; edgeType: string; direction?: string }[]; // Required if action is MATCH
  targetLabels?: string[]; // Optional filter for the move
  confidence: number;
  reasoning: string;
  alternativeMoves?: {
    edgeType: string;
    confidence: number;
    reasoning: string;
  }[];
}

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

export interface TraceStep {
  stepId: number;
  parentId?: number;
  cursorId: string;
  incomingEdge?: string;
  nodeId: string; // Source node where decision was made
  source: string;
  target?: string; // Resulting node if MOVE
  action: 'MOVE' | 'CHECK' | 'MATCH';
  decision: ScoutDecision;
  reasoning: string;
  timestamp: number;
}

export interface TraceLog {
  traceId: string;
  goal: string;
  activeDomain: string;
  startTime: number;
  steps: TraceStep[];
  outcome: 'FOUND' | 'EXHAUSTED' | 'ABORTED';
  finalArtifact?: LabyrinthArtifact;
}

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