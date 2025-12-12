interface WorkflowResult {
  status: 'success' | 'failed' | 'suspended';
  error?: { message: string };
  results?: unknown;
  result?: unknown;
  payload?: unknown;
}

export function getWorkflowResult(res: WorkflowResult): unknown {
  if (res.status === 'failed') {
    throw new Error(`Workflow failed: ${res.error?.message || 'Unknown error'}`);
  }
  
  // Prioritize "results" (plural) as seen in some Mastra versions
  if (res.results) return res.results;
  
  // Check "result" (singular)
  if (res.result) return res.result;
  
  // Check if the payload is wrapped in a "payload" property (common in some testing harnesses)
  if (res.payload) return res.payload;

  // Fallback: Check if the object itself looks like a payload (has artifact or success)
  // or if it's just the wrapper but missing the specific keys we know.
  return res;
}