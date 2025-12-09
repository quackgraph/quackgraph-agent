# Monorepo Structure - Federated Repositories

This monorepo uses a **federated repository structure** where:

- **`quackgraph-agent`** (this repo) owns the high-level Agent logic
- **`packages/quackgraph`** is a nested Git repository containing the Core engine

## Repository Structure

```
quackgraph-agent/               # GitHub: quackgraph/quackgraph-agent
├── packages/
│   ├── agent/                  # Agent logic (owned by parent repo)
│   │   └── src/
│   └── quackgraph/             # GitHub: quackgraph/quackgraph
│       └── packages/
│           ├── native/         # Rust N-API bindings
│           └── quack-graph/    # Core graph TypeScript library
├── scripts/
│   └── git-sync.ts             # Federated push automation
├── package.json                # Root workspace config
└── tsconfig.json
```

## Workspace Configuration

The root `package.json` defines a Bun workspace that spans both repos:

```json
{
  "workspaces": [
    "packages/agent",
    "packages/quackgraph/packages/*"
  ]
}
```

This allows seamless dependency resolution:
- `@quackgraph/agent` → `packages/agent`
- `@quackgraph/graph` → `packages/quackgraph/packages/quack-graph`
- `@quackgraph/native` → `packages/quackgraph/packages/native`

## Git Workflow

### Synchronized Push

To push changes to both repositories atomically:

```bash
bun run push:all "your commit message"
```

This runs `scripts/git-sync.ts` which:
1. Checks if `packages/quackgraph` has uncommitted changes
2. Commits and pushes the inner repo first
3. Commits and pushes the parent repo

### Manual Push (Fine-Grained Control)

```bash
# Push inner repo only
cd packages/quackgraph
git add -A && git commit -m "message" && git push

# Push outer repo only (after inner)
cd ../..
git add -A && git commit -m "message" && git push
```

## Development Commands

| Command | Description |
|---------|-------------|
| `bun install` | Install all dependencies across workspaces |
| `bun run build` | Build core + agent packages |
| `bun run build:core` | Build only the quackgraph core |
| `bun run build:agent` | Build only the agent package |
| `bun run test` | Run tests |
| `bun run push:all` | Synchronized git push to both repos |
| `bun run clean` | Clean all build artifacts |

## Dependency Flow

```
@quackgraph/agent
    ├── depends on → @quackgraph/graph
    └── depends on → @quackgraph/native

@quackgraph/graph
    └── depends on → @quackgraph/native
```

The Agent extends and orchestrates the Core, not the other way around.
