import type { DomainConfig } from '../types';

export class SchemaRegistry {
  private domains = new Map<string, DomainConfig>();

  constructor() {
    // Default 'Global' domain that sees everything (fallback)
    this.register({
      name: 'global',
      description: 'Unrestricted access to the entire topology.',
      allowedEdges: [] // Empty means ALL allowed in our logic, or handled as special case
    });
  }

  register(config: DomainConfig) {
    this.domains.set(config.name.toLowerCase(), config);
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
    if (!domain) return true; // Fallback to permissive
    if (domain.name === 'global') return true;
    if (domain.allowedEdges.length === 0) return true; // Empty whitelist = all allowed? Or none? Usually global handles all.
    
    return domain.allowedEdges.includes(edgeType);
  }
}