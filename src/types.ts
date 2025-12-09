export enum ZoomLevel {
  SECTOR = 0,    // Ghost/Satellite View: Available Moves (Schema)
  TOPOLOGY = 1,  // Drone View: Structural Hops (IDs only)
  CONTENT = 2    // Street View: Full JSON Data
}

export interface AgentConfig {
  llmProvider: {
    generate: (prompt: string) => Promise<string>;
  };
  maxHops?: number;
  // Max number of concurrent exploration threads
  maxCursors?: number;
  // Minimum confidence to pursue a path (0.0 - 1.0)
  confidenceThreshold?: number;
}

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
  action: 'MOVE' | 'CHECK' | 'ABORT';
  edgeType?: string; // Required if action is MOVE
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
  currentNodeId: string;
  currentNodeLabels: string[];
  sectorSummary: SectorSummary[]; // Changed from availableEdgeTypes
  pathHistory: string[]; // Summarized path history
  timeContext?: string; // Human readable time context
}

export interface JudgePrompt {
  goal: string;
  nodeContent: Record<string, any>[];
  timeContext?: string;
}

export interface TraceStep {
  stepId: number;
  parentId?: number;
  cursorId: string;
  incomingEdge?: string;
  nodeId: string; // Source node where decision was made
  source: string;
  target?: string; // Resulting node if MOVE
  action: 'MOVE' | 'CHECK';
  decision: ScoutDecision;
  reasoning: string;
  timestamp: number;
}

export interface TraceLog {
  traceId: string;
  goal: string;
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