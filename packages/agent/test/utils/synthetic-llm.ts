import { type Mock, mock } from "bun:test";
import type { Agent } from "@mastra/core/agent";

/**
 * A deterministic LLM simulator for testing Mastra Agents.
 * Allows mapping prompt keywords to specific JSON responses.
 */
export class SyntheticLLM {
  private responses: Map<string, object> = new Map();
  private defaultResponse: object = { error: "No matching synthetic response configured" };

  /**
   * Register a response trigger.
   * @param keyword If the prompt contains this string, the response will be returned.
   * @param response The JSON object to return.
   */
  addResponse(keyword: string, response: object) {
    this.responses.set(keyword, response);
  }

  setDefault(response: object) {
    this.defaultResponse = response;
  }

  /**
   * Hijacks the `generate` method of a Mastra agent to return synthetic data.
   */
  // biome-ignore lint/suspicious/noExplicitAny: Mocking internal agent types
  mockAgent(agent: Agent<any, any, any>) {
    // @ts-ignore - Overwriting the generate method for testing
    agent.generate = mock(async (prompt: string, _options?: any) => {
      // 1. Check for keyword matches
      for (const [key, val] of this.responses) {
        if (prompt.includes(key)) {
          return {
            text: JSON.stringify(val),
            object: val,
            usage: { promptTokens: 10, completionTokens: 10, totalTokens: 20 },
          };
        }
      }

      // 2. Fallback
      return {
        text: JSON.stringify(this.defaultResponse),
        object: this.defaultResponse,
        usage: { promptTokens: 1, completionTokens: 1, totalTokens: 2 },
      };
    });

    return agent;
  }
}