import { mock } from "bun:test";
import type { Agent } from "@mastra/core/agent";

/**
 * A deterministic LLM simulator for testing Mastra Agents.
 * Allows mapping prompt keywords to specific JSON responses.
 */
export class SyntheticLLM {
  private responses: Map<string, object> = new Map();
  
  // A "God Object" default that satisfies Scout, Judge, Router, and Scribe schemas
  // to prevent Zod validation errors during test fallbacks.
  private globalDefault: object = { 
    // Scout (Action Union)
    action: "ABORT",
    alternativeMoves: [], 
    targetNodeId: "fallback_target", // Ensure targetNodeId exists for types that might check it

    // Router
    domain: "global",

    // Judge
    isAnswer: false,
    answer: "Synthetic Fallback",

    // Scribe
    operations: [],
    requiresClarification: undefined,

    // Common
    confidence: 0.0,
    reasoning: "No matching synthetic response configured (Fallback)." 
  };

  /**
   * Register a response trigger.
   * @param keyword If the prompt contains this string, the response will be returned.
   * @param response The JSON object to return.
   */
  addResponse(keyword: string, response: object) {
    this.responses.set(keyword, response);
  }

  setDefault(response: object) {
    this.globalDefault = response;
  }

  /**
   * Hijacks the `generate` method of a Mastra agent to return synthetic data.
   * @param agent The agent to mock
   * @param agentDefault Optional default response specific to this agent (deprecated - use setDefault instead)
   */
  // biome-ignore lint/suspicious/noExplicitAny: Mocking internal agent types
  mockAgent(agent: Agent<any, any, any>, agentDefault?: object) {
    // @ts-expect-error - Overwriting the generate method for testing
    // biome-ignore lint/suspicious/noExplicitAny: Mocking internal agent types
    agent.generate = mock(async (prompt: string, _options?: any) => {
      // 1. Check for keyword matches
      for (const [key, val] of this.responses) {
        if (prompt.includes(key)) {
          // Return a structured response that mimics Mastra's expected output
          return {
            text: JSON.stringify(val),
            object: val,
            usage: { promptTokens: 10, completionTokens: 10, totalTokens: 20 },
          };
        }
      }

      // 2. Fallback
      // Always look up globalDefault dynamically to allow setDefault() to work after mockAgent()
      const fallback = agentDefault || this.globalDefault;

      // Log warning for debugging
      if (process.env.DEBUG_SYNTHETIC_LLM) {
        console.warn(`[SyntheticLLM] No match for prompt: "${prompt.slice(0, 80)}..."`);
        console.warn(`[SyntheticLLM] Using fallback:`, JSON.stringify(fallback).slice(0, 150));
      }

      return {
        text: JSON.stringify(fallback),
        object: fallback,
        usage: { promptTokens: 1, completionTokens: 1, totalTokens: 2 },
      };
    });

    return agent;
  }
}