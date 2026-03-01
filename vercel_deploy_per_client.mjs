// factory/vercel_deploy_per_client.mjs
import fs from "fs";
import path from "path";

const ROOT = process.cwd();            // running inside app_src
const REPO_ROOT = path.resolve(ROOT, "..");

function readJSON(p) {
  return JSON.parse(fs.readFileSync(p, "utf8"));
}
function ensureDir(p) {
  fs.mkdirSync(p, { recursive: true });
}
function sh(cmd) {
  const { execSync } = require("child_process");
  return execSync(cmd, { stdio: "pipe", encoding: "utf8" }).trim();
}
function shInherit(cmd) {
  const { execSync } = require("child_process");
  execSync(cmd, { stdio: "inherit" });
}

async function main() {
  const token = process.env.VERCEL_TOKEN;
  const orgId = process.env.VERCEL_ORG_ID; // teamId or user orgId
  if (!token) throw new Error("Missing VERCEL_TOKEN");
  if (!orgId) throw new Error("Missing VERCEL_ORG_ID");

  const payloadPath = path.join(REPO_ROOT, "client_payload", "payload.json");
  const payload = readJSON(payloadPath);

  const slug = payload?.project?.slug;
  if (!slug) throw new Error("payload.project.slug missing");

  // Vercel project name rules: keep it safe
  const projectName = String(slug)
    .toLowerCase()
    .replace(/[^a-z0-9-]/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "");

  if (!projectName) throw new Error("Invalid projectName derived from slug");

  console.log(`✅ Client slug: ${slug}`);
  console.log(`✅ Vercel project: ${projectName}`);

  // Ensure .vercel folder exists
  ensureDir(path.join(ROOT, ".vercel"));

  // 1) Try to find existing project by name
  // Vercel API: GET /v9/projects/:name?teamId=...
  let projectId = "";
  try {
    const raw = sh(
      `curl -sS -H "Authorization: Bearer ${token}" "https://api.vercel.com/v9/projects/${projectName}?teamId=${orgId}"`
    );
    const obj = JSON.parse(raw);
    if (obj && obj.id) projectId = obj.id;
  } catch (e) {
    // ignore and create below
  }

  // 2) If not found, create it
  if (!projectId) {
    console.log("🆕 Project not found. Creating new Vercel project...");
    const createBody = JSON.stringify({
      name: projectName,
      framework: "nextjs",
    });

    const raw = sh(
      `curl -sS -X POST -H "Authorization: Bearer ${token}" -H "Content-Type: application/json" ` +
      `"https://api.vercel.com/v9/projects?teamId=${orgId}" -d '${createBody}'`
    );

    const obj = JSON.parse(raw);
    if (!obj?.id) {
      throw new Error(`Failed to create project: ${raw.slice(0, 500)}`);
    }
    projectId = obj.id;
    console.log(`✅ Created projectId: ${projectId}`);
  } else {
    console.log(`✅ Found existing projectId: ${projectId}`);
  }

  // 3) Write .vercel/project.json (link)
  const projectJsonPath = path.join(ROOT, ".vercel", "project.json");
  fs.writeFileSync(
    projectJsonPath,
    JSON.stringify({ orgId, projectId }, null, 2),
    "utf8"
  );
  console.log(`🔗 Linked: ${projectJsonPath}`);

  // 4) Deploy (preview by default). Add --prod if you want production deploys.
  console.log("🚀 Deploying to Vercel...");
  // Pull not strictly required once project.json exists, but it helps env sync
  shInherit(`vercel pull --yes --environment=preview --token=${token}`);
  const url = sh(`vercel deploy --yes --token=${token}`);
  console.log(`✅ Deploy URL: ${url}`);

  // 5) Export to GitHub Actions env
  const envFile = process.env.GITHUB_ENV;
  if (envFile) {
    fs.appendFileSync(envFile, `DEPLOY_URL=${url}\n`);
    fs.appendFileSync(envFile, `CLIENT_SLUG=${projectName}\n`);
  }
}

main().catch((e) => {
  console.error("❌ vercel_deploy_per_client failed:", e?.stack || e);
  process.exit(1);
});
