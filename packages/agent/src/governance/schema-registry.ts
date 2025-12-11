import type { DomainConfig } from '../types';

export class SchemaRegistry {
  private domains = new Map<string, DomainConfig>();

  constructor() {
    // Default 'Global' domain that sees everything (fallback)
    this.register({
      name: 'global',
      description: 'Unrestricted access to the entire topology.',
      allowedEdges: [], // Empty means ALL allowed
      excludedEdges: []
    });
  }

  register(config: DomainConfig) {
    this.domains.set(config.name.toLowerCase(), config);
  }

  loadFromConfig(configs: DomainConfig[]) {
    configs.forEach(c => { this.register(c); });
  }

  getDomain(name: string): DomainConfig | undefined {
    return this.domains.get(name.toLowerCase());
  }

  getAllDomains(): DomainConfig[] {
    return Array.from(this.domains.values());
  }

  /**
   * Returns true if the edge type is allowed within the domain.
   * If domain is 'global' or not found, it defaults to true (permissive) 
   * unless strict mode is desired.
   */
  isEdgeAllowed(domainName: string, edgeType: string): boolean {
    const domain = this.domains.get(domainName.toLowerCase());
    if (!domain) return true;
    
    const target = edgeType.toLowerCase();

    // 1. Check Exclusion (Blacklist)
    if (domain.excludedEdges?.some(e => e.toLowerCase() === target)) return false;

    // 2. Check Inclusion (Whitelist)
    if (domain.allowedEdges.length > 0) {
      return domain.allowedEdges.some(e => e.toLowerCase() === target);
    }

    // 3. Default Permissive
    return true;
  }

  getValidEdges(domainName: string): string[] | undefined {
    const domain = this.domains.get(domainName.toLowerCase());
    if (!domain || (domain.allowedEdges.length === 0 && (!domain.excludedEdges || domain.excludedEdges.length === 0))) {
      return undefined; // All allowed
    }
    return domain.allowedEdges;
  }

  /**
   * Returns true if the domain requires causal (monotonic) time traversal.
   */
  isDomainCausal(domainName: string): boolean {
    const domain = this.domains.get(domainName.toLowerCase());
    return !!domain?.isCausal;
  }
}

export const schemaRegistry = new SchemaRegistry();
export const getSchemaRegistry = () => schemaRegistry;