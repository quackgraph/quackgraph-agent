#!/usr/bin/env bun
/**
 * Git Pull Script - Federated Pull for Nested Repositories
 * 
 * Usage:
 *   bun run scripts/git-pull.ts
 *   bun run pull:all
 */

import { $ } from "bun";

const INNER_REPO_PATH = "packages/quackgraph";
const ROOT_DIR = import.meta.dir.replace("/scripts", "");

async function pullRepo(cwd: string, repoName: string): Promise<void> {
    console.log(`\n⬇️ [${repoName}] Pulling changes...`);
    try {
        await $`git -C ${cwd} pull`.quiet();
        console.log(`   ✅ Successfully pulled ${repoName}`);
    } catch (error) {
        console.error(`   ❌ Failed to pull ${repoName}:`, error);
        throw error;
    }
}

async function pullAll(): Promise<void> {
    console.log("🔄 Git Pull - Federated Repository Update");
    console.log("=========================================");

    // Pull parent first
    console.log("\n\n🔷 Step 1: Processing parent repository (quackgraph-agent)...");
    await pullRepo(ROOT_DIR, "quackgraph-agent");

    // Pull inner repo
    console.log("\n\n🔷 Step 2: Processing inner repository (quackgraph core)...");
    const innerRepoPath = `${ROOT_DIR}/${INNER_REPO_PATH}`;
    await pullRepo(innerRepoPath, "quackgraph");

    console.log("\n\n=========================================");
    console.log("✅ Git pull completed successfully!");
    console.log("=========================================\n");
}

pullAll().catch((error) => {
    console.error("\n❌ Pull failed:", error);
    process.exit(1);
});
