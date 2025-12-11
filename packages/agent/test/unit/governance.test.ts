import { describe, it, expect, beforeEach } from "bun:test";
import { SchemaRegistry } from "../../src/governance/schema-registry";

describe("Governance (SchemaRegistry)", () => {
  let registry: SchemaRegistry;

  beforeEach(() => {
    registry = new SchemaRegistry();
  });

  it("registers and retrieves domains", () => {
    registry.register({
      name: "Medical",
      description: "Health data",
      allowedEdges: ["TREATED_WITH", "HAS_SYMPTOM"]
    });

    const domain = registry.getDomain("Medical");
    expect(domain).toBeDefined();
    expect(domain?.allowedEdges).toContain("TREATED_WITH");
  });

  it("handles case-insensitive domain names", () => {
    registry.register({
      name: "Finance",
      description: "Money",
      allowedEdges: []
    });
    expect(registry.getDomain("finance")).toBeDefined();
    expect(registry.getDomain("FINANCE")).toBeDefined();
  });

  it("enforces whitelists (allowedEdges)", () => {
    registry.register({
      name: "Strict",
      description: "Only A and B",
      allowedEdges: ["A", "B"]
    });

    expect(registry.isEdgeAllowed("Strict", "A")).toBe(true);
    expect(registry.isEdgeAllowed("Strict", "B")).toBe(true);
    expect(registry.isEdgeAllowed("Strict", "C")).toBe(false); // blocked
  });

  it("enforces blacklists (excludedEdges)", () => {
    registry.register({
      name: "OpenButSafe",
      description: "Everything except DANGER",
      allowedEdges: [], // All allowed by default
      excludedEdges: ["DANGER"]
    });

    expect(registry.isEdgeAllowed("OpenButSafe", "SAFE")).toBe(true);
    expect(registry.isEdgeAllowed("OpenButSafe", "DANGER")).toBe(false); // blocked
  });

  it("defaults to permissive if domain not found", () => {
    // If a domain doesn't exist, we usually default to global/permissive 
    // or the registry returns true as per implementation
    expect(registry.isEdgeAllowed("UnknownDomain", "ANYTHING")).toBe(true);
  });

  it("getValidEdges returns undefined for unrestricted domains", () => {
    registry.register({
      name: "Global",
      description: "All access",
      allowedEdges: []
    });
    // Implementation details: empty allowedEdges usually means 'undefined' (all) in return 
    // or empty array depending on implementation. 
    // Checking src: `if (domain.allowedEdges.length === 0 ...) return undefined;`
    expect(registry.getValidEdges("Global")).toBeUndefined();
  });

  it("getValidEdges returns specific list for restricted domains", () => {
    registry.register({
      name: "Restricted",
      description: "Restricted",
      allowedEdges: ["A"]
    });
    expect(registry.getValidEdges("Restricted")).toEqual(["A"]);
  });
});