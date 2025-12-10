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

async function pullRepo(cwd: string, repoName: string, repoUrl?: string): Promise<void> {
    console.log(`\n⬇️ [${repoName}] Processing...`);

    // Check if directory exists and has .git
    const fs = await import("node:fs/promises");
    const hasGit = await fs.exists(`${cwd}/.git`).catch(() => false);

    if (!hasGit && repoUrl) {
        console.log(`   ✨ Repository not found. Cloning from ${repoUrl}...`);
        try {
            // Ensure parent dir exists
            await $`mkdir -p ${cwd}`;
            // Remove the empty dir if it exists so clone works (or clone into it if empty)
            // Safest is to remove checking uniqueness or just run git clone
            // If cwd exists but is empty, git clone <url> <dir> works.

            await $`git clone ${repoUrl} ${cwd}`;
            console.log(`   ✅ Successfully cloned ${repoName}`);
            return;
        } catch (error) {
            console.error(`   ❌ Failed to clone ${repoName}:`, error);
            throw error;
        }
    }

    console.log(`   ⬇️ Pulling changes...`);
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
    const innerRepoUrl = "https://github.com/quackgraph/quackgraph.git";

    // Custom logic to ensure 'agent' branch
    await pullRepo(innerRepoPath, "quackgraph", innerRepoUrl);
    // Force checkout agent branch if not already
    try {
        await $`git -C ${innerRepoPath} checkout agent`.quiet();
        await $`git -C ${innerRepoPath} pull origin agent`.quiet();
    } catch (e) {
        console.warn("   ⚠️ Could not checkout/pull agent branch explicitly:", e);
    }

    console.log("\n\n=========================================");
    console.log("✅ Git pull completed successfully!");
    console.log("=========================================\n");
}

pullAll().catch((error) => {
    console.error("\n❌ Pull failed:", error);
    process.exit(1);
});
