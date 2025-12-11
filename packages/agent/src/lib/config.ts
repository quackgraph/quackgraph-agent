import { z } from 'zod';

const envSchema = z.object({
  // Server Config
  MASTRA_PORT: z.coerce.number().default(4111),
  LOG_LEVEL: z.enum(['debug', 'info', 'warn', 'error']).default('info'),

  // Agent Model Configuration (Granular)
  // Format: provider/model-name (e.g., 'groq/llama-3.3-70b-versatile', 'openai/gpt-4')
  AGENT_SCOUT_MODEL: z.string().default('groq/llama-3.3-70b-versatile'),
  AGENT_JUDGE_MODEL: z.string().default('groq/llama-3.3-70b-versatile'),
  AGENT_ROUTER_MODEL: z.string().default('groq/llama-3.3-70b-versatile'),
  AGENT_SCRIBE_MODEL: z.string().default('groq/llama-3.3-70b-versatile'),

  // API Keys (Validated for existence if required by selected models)
  GROQ_API_KEY: z.string().optional(),
  OPENAI_API_KEY: z.string().optional(),
  ANTHROPIC_API_KEY: z.string().optional(),
});

// Validate process.env
// Note: In Bun, process.env is automatically populated from .env files
const parsed = envSchema.parse(process.env);

export const config = {
  server: {
    port: parsed.MASTRA_PORT,
    logLevel: parsed.LOG_LEVEL,
  },
  agents: {
    scout: {
      model: { id: parsed.AGENT_SCOUT_MODEL as `${string}/${string}` },
    },
    judge: {
      model: { id: parsed.AGENT_JUDGE_MODEL as `${string}/${string}` },
    },
    router: {
      model: { id: parsed.AGENT_ROUTER_MODEL as `${string}/${string}` },
    },
    scribe: {
      model: { id: parsed.AGENT_SCRIBE_MODEL as `${string}/${string}` },
    },
  },
};