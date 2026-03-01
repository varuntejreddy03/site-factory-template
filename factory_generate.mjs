// factory_generate.mjs (ESM)
// Node 18+ required (GitHub Actions ubuntu-latest has it)

import fs from "fs";
import path from "path";
import crypto from "crypto";

const ROOT = process.cwd();

function readJSON(p) {
  return JSON.parse(fs.readFileSync(p, "utf8"));
}
function readText(p, fallback = "") {
  return fs.existsSync(p) ? fs.readFileSync(p, "utf8") : fallback;
}
function ensureDir(p) {
  fs.mkdirSync(p, { recursive: true });
}
function isUrl(s) {
  try { new URL(s); return true; } catch { return false; }
}
function safeJoin(base, rel) {
  const full = path.resolve(base, rel);
  if (!full.startsWith(path.resolve(base))) throw new Error(`Unsafe path: ${rel}`);
  return full;
}
function sha1(s) {
  return crypto.createHash("sha1").update(s).digest("hex").slice(0, 10);
}

function detectAppBase(appSrcDir) {
  // create-next-app with --src-dir uses app_src/src/app
  const withSrc = path.join(appSrcDir, "src");
  return fs.existsSync(withSrc) ? withSrc : appSrcDir;
}

function stripCodeFences(s) {
  // If model returns ```json ...```, extract inner content
  const m = s.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
  return m ? m[1] : s;
}

async function fetchWithTimeout(url, opts = {}, ms = 60_000) {
  const controller = new AbortController();
  const t = setTimeout(() => controller.abort(), ms);
  try {
    const res = await fetch(url, { ...opts, signal: controller.signal });
    return res;
  } finally {
    clearTimeout(t);
  }
}

async function downloadToFile(url, outPath) {
  const res = await fetchWithTimeout(url, {}, 60_000);
  if (!res.ok) throw new Error(`Download failed ${res.status} ${res.statusText}: ${url}`);
  const buf = Buffer.from(await res.arrayBuffer());
  ensureDir(path.dirname(outPath));
  fs.writeFileSync(outPath, buf);
}

function guessExtFromUrl(urlStr) {
  try {
    const u = new URL(urlStr);
    const ext = path.extname(u.pathname);
    if (ext && ext.length <= 6) return ext;
  } catch {}
  return "";
}

function normalizeModelOutputToFileMap(text) {
  const cleaned = stripCodeFences(text).trim();

  // Expecting: { "files": { "path": "content" } }
  let obj;
  try {
    obj = JSON.parse(cleaned);
  } catch (e) {
    throw new Error(
      "AI output was not valid JSON. Ask the model to return JSON only.\n" +
      "Raw (first 400 chars):\n" + cleaned.slice(0, 400)
    );
  }

  if (!obj || typeof obj !== "object" || !obj.files || typeof obj.files !== "object") {
    throw new Error(`AI JSON must have shape { "files": { "path": "content" } }`);
  }
  return obj.files;
}

/**
 * Providers
 * - OpenRouter: OpenAI-compatible
 * - Groq: OpenAI-compatible
 * - Gemini: different endpoint
 */

async function callOpenAICompatible({ endpoint, apiKey, model, messages, temperature = 0.4 }) {
  const res = await fetchWithTimeout(
    endpoint,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model,
        messages,
        temperature,
        response_format: { type: "json_object" }, // best-effort for JSON
      }),
    },
    120_000
  );

  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(
      `AI error (${res.status}): ${JSON.stringify(data).slice(0, 600)}`
    );
  }
  const text = data?.choices?.[0]?.message?.content;
  if (!text) throw new Error("AI returned empty content");
  return text;
}

async function callGemini({ apiKey, model, messages }) {
  // Convert messages -> Gemini contents
  // We'll merge system+user into one "user" prompt for simplicity.
  const joined = messages
    .map((m) => {
      const role = m.role === "system" ? "SYSTEM" : m.role.toUpperCase();
      return `${role}:\n${m.content}`;
    })
    .join("\n\n---\n\n");

  const url =
    `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent?key=${encodeURIComponent(apiKey)}`;

  const res = await fetchWithTimeout(
    url,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        contents: [
          { role: "user", parts: [{ text: joined }] }
        ],
        generationConfig: {
          temperature: 0.4,
        },
      }),
    },
    120_000
  );

  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(`Gemini error (${res.status}): ${JSON.stringify(data).slice(0, 600)}`);
  }

  const text =
    data?.candidates?.[0]?.content?.parts
      ?.map((p) => p.text || "")
      .join("") || "";

  if (!text.trim()) throw new Error("Gemini returned empty content");
  return text;
}

async function callAIWithFallback({ messages }) {
  const order = (process.env.AI_PROVIDERS || "openrouter,groq,gemini")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);

  const errors = [];

  for (const provider of order) {
    try {
      if (provider === "openrouter") {
        const key = process.env.OPENROUTER_API_KEY;
        const model = process.env.OPENROUTER_MODEL || "anthropic/claude-3.5-sonnet";
        if (!key) throw new Error("Missing OPENROUTER_API_KEY");
        return await callOpenAICompatible({
          endpoint: "https://openrouter.ai/api/v1/chat/completions",
          apiKey: key,
          model,
          messages,
        });
      }

      if (provider === "groq") {
        const key = process.env.GROQ_API_KEY;
        const model = process.env.GROQ_MODEL || "llama-3.1-70b-versatile";
        if (!key) throw new Error("Missing GROQ_API_KEY");
        return await callOpenAICompatible({
          endpoint: "https://api.groq.com/openai/v1/chat/completions",
          apiKey: key,
          model,
          messages,
        });
      }

      if (provider === "gemini") {
        const key = process.env.GEMINI_API_KEY;
        const model = process.env.GEMINI_MODEL || "gemini-1.5-pro";
        if (!key) throw new Error("Missing GEMINI_API_KEY");
        return await callGemini({ apiKey: key, model, messages });
      }

      throw new Error(`Unknown provider: ${provider}`);
    } catch (e) {
      errors.push({ provider, error: String(e?.message || e) });
      console.warn(`⚠️ Provider failed: ${provider} -> ${e?.message || e}`);
    }
  }

  throw new Error(
    "All AI providers failed:\n" +
      errors.map((x) => `- ${x.provider}: ${x.error}`).join("\n")
  );
}

async function main() {
  // Paths
  const payloadPath = path.join(ROOT, "client_payload", "payload.json");
  const promptPath = path.join(ROOT, "factory", "prompt_system.txt");
  const schemaPath = path.join(ROOT, "factory", "site_schema.json");

  if (!fs.existsSync(payloadPath)) throw new Error(`Missing ${payloadPath}`);

  const payload = readJSON(payloadPath);
  const promptSystem = readText(promptPath, "");
  const schema = fs.existsSync(schemaPath) ? readJSON(schemaPath) : null;

  // Next.js scaffold path
  const appSrcDir = path.join(ROOT, "app_src");
  if (!fs.existsSync(appSrcDir)) {
    throw new Error("Missing app_src. Scaffold Next.js first (create-next-app).");
  }

  const appBase = detectAppBase(appSrcDir);      // app_src/src OR app_src
  const appDir = path.join(appBase, "app");      // app_src/src/app
  const componentsDir = path.join(appBase, "components");
  const publicAssetsDir = path.join(appSrcDir, "public", "assets");

  ensureDir(appDir);
  ensureDir(componentsDir);
  ensureDir(publicAssetsDir);

  // Download assets
  const assets = payload?.assets || {};
  const logoUrl = assets?.logoUrl || "";
  const galleryUrls = Array.isArray(assets?.galleryUrls) ? assets.galleryUrls : [];

  let logoPathHint = "/public/assets/logo";
  if (logoUrl && isUrl(logoUrl)) {
    const ext = guessExtFromUrl(logoUrl) || ".png";
    const out = path.join(publicAssetsDir, `logo${ext}`);
    console.log(`⬇️ Downloading logo: ${logoUrl} -> ${out}`);
    await downloadToFile(logoUrl, out);
    logoPathHint = `/public/assets/logo${ext}`;
  }

  const gallerySaved = [];
  for (const u of galleryUrls) {
    if (!u || !isUrl(u)) continue;
    const ext = guessExtFromUrl(u) || ".jpg";
    const name = `gallery-${sha1(u)}${ext}`;
    const out = path.join(publicAssetsDir, "gallery", name);
    console.log(`⬇️ Downloading gallery: ${u} -> ${out}`);
    await downloadToFile(u, out);
    gallerySaved.push(`/public/assets/gallery/${name}`);
  }

  // Build the AI prompt (force JSON file map)
  const system = [
    "You are a code generator for a Next.js App Router project.",
    "Return ONLY valid JSON. No markdown. No backticks.",
    "JSON format: { \"files\": { \"relative/path/from/app_src\": \"file content as string\" } }",
    "Only write these files (no others):",
    "- src/app/layout.tsx OR app/layout.tsx (depending on project)",
    "- src/app/page.tsx OR app/page.tsx",
    "- src/components/Sections.tsx OR components/Sections.tsx (optional but allowed)",
    "Do NOT modify package.json, next.config.*, tsconfig.*, eslint config.",
    "Use Tailwind classes (already installed).",
    "SEO basics in layout.tsx (metadata title/description, favicon ok).",
  ].join("\n");

  const user = [
    "Generation rules (prompt_system.txt):",
    promptSystem || "(none)",
    "",
    "Optional schema (site_schema.json):",
    schema ? JSON.stringify(schema, null, 2) : "(none)",
    "",
    "Client payload (payload.json):",
    JSON.stringify(payload, null, 2),
    "",
    "Assets available:",
    `- logo path: ${logoPathHint}`,
    `- gallery paths: ${gallerySaved.length ? gallerySaved.join(", ") : "(none downloaded)"}`,
    "",
    "Important:",
    "- All text content must be derived from payload.rawText",
    "- Create sections: Hero, About, Services, Gallery, Contact, Footer",
    "- Gallery should render from the saved gallery paths if present",
  ].join("\n");

  const messages = [
    { role: "system", content: system },
    { role: "user", content: user },
  ];

  console.log("🤖 Calling AI with fallback providers...");
  const aiText = await callAIWithFallback({ messages });

  const files = normalizeModelOutputToFileMap(aiText);

  // Write files into app_src
  // We accept both src/... and non-src paths from model, but enforce within app_src.
  for (const [rel, content] of Object.entries(files)) {
    if (typeof content !== "string") continue;

    // Enforce the allowed file set
    const allowed = [
      "app/page.tsx",
      "app/layout.tsx",
      "components/Sections.tsx",
      "src/app/page.tsx",
      "src/app/layout.tsx",
      "src/components/Sections.tsx",
    ];
    if (!allowed.includes(rel)) {
      console.warn(`⛔ Skipping disallowed path from AI: ${rel}`);
      continue;
    }

    const outPath = safeJoin(appSrcDir, rel);
    ensureDir(path.dirname(outPath));
    fs.writeFileSync(outPath, content, "utf8");
    console.log(`✅ Wrote ${rel}`);
  }

  // If model wrote to app/... but project uses src/app, copy into correct place (safety)
  const usesSrc = fs.existsSync(path.join(appSrcDir, "src"));
  if (usesSrc) {
    const a = path.join(appSrcDir, "app");
    const sa = path.join(appSrcDir, "src", "app");
    if (fs.existsSync(a) && !fs.existsSync(sa)) ensureDir(sa);

    // If AI wrote app/page.tsx but src/app exists, mirror it.
    const appPage = path.join(appSrcDir, "app", "page.tsx");
    const srcAppPage = path.join(appSrcDir, "src", "app", "page.tsx");
    if (fs.existsSync(appPage)) {
      ensureDir(path.dirname(srcAppPage));
      fs.copyFileSync(appPage, srcAppPage);
      console.log("🔁 Mirrored app/page.tsx -> src/app/page.tsx");
    }

    const appLayout = path.join(appSrcDir, "app", "layout.tsx");
    const srcAppLayout = path.join(appSrcDir, "src", "app", "layout.tsx");
    if (fs.existsSync(appLayout)) {
      ensureDir(path.dirname(srcAppLayout));
      fs.copyFileSync(appLayout, srcAppLayout);
      console.log("🔁 Mirrored app/layout.tsx -> src/app/layout.tsx");
    }

    const comp = path.join(appSrcDir, "components", "Sections.tsx");
    const srcComp = path.join(appSrcDir, "src", "components", "Sections.tsx");
    if (fs.existsSync(comp)) {
      ensureDir(path.dirname(srcComp));
      fs.copyFileSync(comp, srcComp);
      console.log("🔁 Mirrored components/Sections.tsx -> src/components/Sections.tsx");
    }
  }

  console.log("🎉 Generation complete.");
}

main().catch((e) => {
  console.error("❌ factory_generate.mjs failed:", e?.stack || e);
  process.exit(1);
});
