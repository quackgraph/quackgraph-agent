#!/usr/bin/env bun
/**
 * Git Sync Script - Federated Push for Nested Repositories
 * 
 * This script handles the synchronization of the nested git structure:
 * - quackgraph-agent (parent) -> Contains packages/agent
 * - packages/quackgraph (nested repo) -> The core engine
 * 
 * Usage:
 *   bun run scripts/git-sync.ts [message]
 *   bun run push:all
 * 
 * The script will:
 * 1. Check if the inner repo (packages/quackgraph) has changes
 * 2. Commit and push the inner repo if needed
 * 3. Update the parent repo with any changes (including submodule pointer if configured)
 * 4. Push the parent repo
 */

import { $ } from "bun";

const INNER_REPO_PATH = "packages/quackgraph";
const ROOT_DIR = import.meta.dir.replace("/scripts", "");

interface GitStatus {
    isDirty: boolean;
    isAhead: boolean;
    branch: string;
}

async function getGitStatus(cwd: string): Promise<GitStatus> {
    try {
        // Check for uncommitted changes
        const statusResult = await $`git -C ${cwd} status --porcelain`.text();
        const isDirty = statusResult.trim().length > 0;

        // Get current branch
        const branchResult = await $`git -C ${cwd} rev-parse --abbrev-ref HEAD`.text();
        const branch = branchResult.trim();

        // Check if ahead of remote
        let isAhead = false;
        try {
            const aheadResult = await $`git -C ${cwd} rev-list --count @{upstream}..HEAD 2>/dev/null`.text();
            isAhead = parseInt(aheadResult.trim(), 10) > 0;
        } catch {
            // No upstream configured, assume not ahead
            isAhead = false;
        }

        return { isDirty, isAhead, branch };
    } catch (error) {
        console.error(`Error getting git status for ${cwd}:`, error);
        throw error;
    }
}

async function commitAndPush(cwd: string, message: string, repoName: string): Promise<boolean> {
    const status = await getGitStatus(cwd);

    console.log(`\n📦 [${repoName}] Status:`);
    console.log(`   Branch: ${status.branch}`);
    console.log(`   Dirty: ${status.isDirty}`);
    console.log(`   Ahead of remote: ${status.isAhead}`);

    if (!status.isDirty && !status.isAhead) {
        console.log(`   ✅ Nothing to push for ${repoName}`);
        return false;
    }

    if (status.isDirty) {
        console.log(`\n   📝 Staging and committing changes in ${repoName}...`);
        await $`git -C ${cwd} add -A`.quiet();
        await $`git -C ${cwd} commit -m ${message}`.quiet();
        console.log(`   ✅ Committed: "${message}"`);
    }

    console.log(`\n   🚀 Pushing ${repoName} to remote...`);
    try {
        await $`git -C ${cwd} push`.quiet();
        console.log(`   ✅ Successfully pushed ${repoName}`);
        return true;
    } catch (error) {
        console.error(`   ❌ Failed to push ${repoName}:`, error);
        throw error;
    }
}

async function syncRepos(): Promise<void> {
    // Get commit message from args or use default
    const args = process.argv.slice(2);
    const commitMessage = args.join(" ") || `sync: ${new Date().toISOString()}`;

    console.log("🔄 Git Sync - Federated Repository Push");
    console.log("=========================================");
    console.log(`📝 Commit message: "${commitMessage}"`);

    const innerRepoPath = `${ROOT_DIR}/${INNER_REPO_PATH}`;

    // Step 1: Handle inner repository (quackgraph core)
    console.log("\n\n🔷 Step 1: Processing inner repository (quackgraph core)...");
    try {
        await commitAndPush(innerRepoPath, commitMessage, "quackgraph");
    } catch (error) {
        console.error("❌ Failed to sync inner repository");
        throw error;
    }

    // Step 2: Handle parent repository (quackgraph-agent)
    console.log("\n\n🔷 Step 2: Processing parent repository (quackgraph-agent)...");
    try {
        await commitAndPush(ROOT_DIR, commitMessage, "quackgraph-agent");
    } catch (error) {
        console.error("❌ Failed to sync parent repository");
        throw error;
    }

    console.log("\n\n=========================================");
    console.log("✅ Git sync completed successfully!");
    console.log("=========================================\n");
}

// Run the sync
syncRepos().catch((error) => {
    console.error("\n❌ Sync failed:", error);
    process.exit(1);
});
