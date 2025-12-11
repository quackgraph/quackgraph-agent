export function getWorkflowResult(res: any): any {
  if (res.status === 'failed') {
    throw new Error(`Workflow failed: ${res.error?.message || 'Unknown error'}`);
  }
  
  // Prioritize "results" (plural) as seen in some Mastra versions/mocks
  if (res.results) return res.results;
  
  // Check "result" (singular)
  if (res.result) return res.result;
  
  // Fallback: Check if the object itself looks like a payload (has artifact or success)
  // or if it's just the wrapper but missing the specific keys we know.
  // We return res to handle cases where the payload is the root object.
  return res;
}