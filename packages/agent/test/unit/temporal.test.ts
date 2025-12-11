import { describe, it, expect } from "bun:test";
import { resolveRelativeTime } from "../../src/utils/temporal";

describe("Temporal Logic (resolveRelativeTime)", () => {
  // Fixed reference time: 2025-01-01T12:00:00Z
  // Timestamp: 1735732800000
  const refDate = new Date("2025-01-01T12:00:00Z");
  const ONE_DAY = 24 * 60 * 60 * 1000;
  const ONE_HOUR = 60 * 60 * 1000;

  it("resolves exact keywords", () => {
    expect(resolveRelativeTime("now", refDate)?.getTime()).toBe(refDate.getTime());
    expect(resolveRelativeTime("today", refDate)?.getTime()).toBe(refDate.getTime());
    
    const yesterday = resolveRelativeTime("yesterday", refDate);
    expect(yesterday?.getTime()).toBe(refDate.getTime() - ONE_DAY);

    const tomorrow = resolveRelativeTime("tomorrow", refDate);
    expect(tomorrow?.getTime()).toBe(refDate.getTime() + ONE_DAY);
  });

  it("resolves 'X time ago' patterns", () => {
    const twoDaysAgo = resolveRelativeTime("2 days ago", refDate);
    expect(twoDaysAgo?.getTime()).toBe(refDate.getTime() - (2 * ONE_DAY));

    const fiveHoursAgo = resolveRelativeTime("5 hours ago", refDate);
    expect(fiveHoursAgo?.getTime()).toBe(refDate.getTime() - (5 * ONE_HOUR));
  });

  it("resolves 'in X time' patterns", () => {
    const inThreeWeeks = resolveRelativeTime("in 3 weeks", refDate);
    // 3 weeks = 21 days
    expect(inThreeWeeks?.getTime()).toBe(refDate.getTime() + (21 * ONE_DAY));
  });

  it("resolves absolute dates", () => {
    const iso = "2023-10-05T00:00:00.000Z";
    const result = resolveRelativeTime(iso, refDate);
    expect(result?.toISOString()).toBe(iso);
  });

  it("returns null for garbage input", () => {
    expect(resolveRelativeTime("not a date", refDate)).toBeNull();
  });
});