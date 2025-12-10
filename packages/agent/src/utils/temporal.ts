/**
 * Simple heuristic parser for relative time strings.
 * Used to ground natural language ("yesterday") into absolute ISO timestamps for the Graph.
 * 
 * In a production system, this would be replaced by a robust library like `chrono-node`.
 */
export function resolveRelativeTime(input: string, referenceDate: Date = new Date()): Date | null {
  const lower = input.toLowerCase().trim();
  const now = referenceDate.getTime();
  const ONE_MINUTE = 60 * 1000;
  const ONE_HOUR = 60 * ONE_MINUTE;
  const ONE_DAY = 24 * ONE_HOUR;

  // 1. Direct keywords
  if (lower === 'now' || lower === 'today') return new Date(now);
  if (lower === 'yesterday') return new Date(now - ONE_DAY);
  if (lower === 'tomorrow') return new Date(now + ONE_DAY);

  // 2. "X [unit] ago"
  const agoMatch = lower.match(/^(\d+)\s+(day|days|hour|hours|minute|minutes|week|weeks)\s+ago$/);
  if (agoMatch) {
    const amount = parseInt(agoMatch[1] || '0', 10);
    const unit = agoMatch[2] || '';
    if (unit.startsWith('day')) return new Date(now - amount * ONE_DAY);
    if (unit.startsWith('hour')) return new Date(now - amount * ONE_HOUR);
    if (unit.startsWith('minute')) return new Date(now - amount * ONE_MINUTE);
    if (unit.startsWith('week')) return new Date(now - amount * 7 * ONE_DAY);
  }

  // 3. "in X [unit]"
  const inMatch = lower.match(/^in\s+(\d+)\s+(day|days|hour|hours|minute|minutes|week|weeks)$/);
  if (inMatch) {
    const amount = parseInt(inMatch[1] || '0', 10);
    const unit = inMatch[2] || '';
    if (unit.startsWith('day')) return new Date(now + amount * ONE_DAY);
    if (unit.startsWith('hour')) return new Date(now + amount * ONE_HOUR);
    if (unit.startsWith('minute')) return new Date(now + amount * ONE_MINUTE);
    if (unit.startsWith('week')) return new Date(now + amount * 7 * ONE_DAY);
  }

  // 4. Fallback: Try native Date parse (e.g. "2023-01-01", "Oct 5 2024")
  const parsed = Date.parse(input);
  if (!Number.isNaN(parsed)) {
    return new Date(parsed);
  }

  return null;
}