import { z } from 'zod';

export const RouterDecisionSchema = z.object({
  domain: z.string(),
  confidence: z.number().min(0).max(1),
  reasoning: z.string(),
});

export const JudgeDecisionSchema = z.object({
  isAnswer: z.boolean(),
  answer: z.string(),
  confidence: z.number().min(0).max(1),
});

// Discriminated Union for Scout Actions
const MoveAction = z.object({
  action: z.literal('MOVE'),
  edgeType: z.string().optional().describe("The edge type to traverse (Single Hop)"),
  path: z.array(z.string()).optional().describe("Sequence of node IDs to traverse (Multi Hop)"),
  confidence: z.number().min(0).max(1),
  reasoning: z.string(),
  alternativeMoves: z.array(z.object({
    edgeType: z.string(),
    confidence: z.number(),
    reasoning: z.string()
  })).optional()
});

const CheckAction = z.object({
  action: z.literal('CHECK'),
  confidence: z.number().min(0).max(1),
  reasoning: z.string(),
});

const MatchAction = z.object({
  action: z.literal('MATCH'),
  pattern: z.array(z.object({
    srcVar: z.number(),
    tgtVar: z.number(),
    edgeType: z.string(),
    direction: z.string().optional()
  })),
  confidence: z.number().min(0).max(1),
  reasoning: z.string(),
});

const AbortAction = z.object({
  action: z.literal('ABORT'),
  confidence: z.number().min(0).max(1),
  reasoning: z.string(),
});

export const ScoutDecisionSchema = z.discriminatedUnion('action', [
  MoveAction,
  CheckAction,
  MatchAction,
  AbortAction
]);

// --- Scribe Agent Schemas (Mutations) ---

const CreateNodeOp = z.object({
  op: z.literal('CREATE_NODE'),
  id: z.string().optional().describe('Optional custom ID. If omitted, system generates UUID.'),
  labels: z.array(z.string()),
  properties: z.record(z.any()),
  validFrom: z.string().optional().describe('ISO Date string. If omitted, defaults to NOW.'),
  validTo: z.string().optional().describe('ISO Date string. If omitted, node is active indefinitely.')
});

const UpdateNodeOp = z.object({
  op: z.literal('UPDATE_NODE'),
  // We use a match query (usually ID) to find the node
  match: z.object({
    id: z.string().describe('The distinct ID of the node to update.')
  }),
  set: z.record(z.any()),
  validFrom: z.string().optional().describe('ISO Date string. The effective start time of this update.')
});

const DeleteNodeOp = z.object({
  op: z.literal('DELETE_NODE'),
  id: z.string(),
  validTo: z.string().optional().describe('ISO Date string. When the node ceased to exist/be valid.')
});

const CreateEdgeOp = z.object({
  op: z.literal('CREATE_EDGE'),
  source: z.string().describe('Source Node ID'),
  target: z.string().describe('Target Node ID'),
  type: z.string().describe('Edge Type (e.g. KNOWS, BOUGHT)'),
  properties: z.record(z.any()).optional(),
  validFrom: z.string().optional().describe('ISO Date string. When this relationship started.'),
  validTo: z.string().optional().describe('ISO Date string. When this relationship ended (if applicable).')
});

const CloseEdgeOp = z.object({
  op: z.literal('CLOSE_EDGE'),
  source: z.string(),
  target: z.string(),
  type: z.string(),
  validTo: z.string().describe('ISO Date string. When this relationship ended.')
});

export const GraphMutationSchema = z.discriminatedUnion('op', [
  CreateNodeOp,
  UpdateNodeOp,
  DeleteNodeOp,
  CreateEdgeOp,
  CloseEdgeOp
]);

export const ScribeDecisionSchema = z.object({
  reasoning: z.string().describe('Explanation of why these mutations are required.'),
  operations: z.array(GraphMutationSchema),
  requiresClarification: z.string().optional().describe('If the user intent is ambiguous, ask a question instead of mutating.')
});