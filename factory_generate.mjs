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
  const withSrc = path.join(appSrcDir, "src");
  return fs.existsSync(withSrc) ? withSrc : appSrcDir;
}

function stripCodeFences(s) {
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

// ---------------------------------------------------------------------------
// AI Providers
// ---------------------------------------------------------------------------

async function callOpenAICompatible({
  endpoint,
  apiKey,
  model,
  messages,
  temperature = 0.4,
  max_tokens = 8000,
}) {
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
        max_tokens,
      }),
    },
    120_000
  );

  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(`AI error (${res.status}): ${JSON.stringify(data).slice(0, 600)}`);
  }

  const text = data?.choices?.[0]?.message?.content;
  if (!text || !String(text).trim()) throw new Error("AI returned empty content");
  return String(text).trim();
}

async function callGemini({
  apiKey,
  model,
  messages,
  temperature = 0.4,
  maxOutputTokens = 8000,
}) {
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
        contents: [{ role: "user", parts: [{ text: joined }] }],
        generationConfig: {
          temperature,
          maxOutputTokens,
          responseMimeType: "application/json",
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
    data?.candidates?.[0]?.content?.parts?.map((p) => p.text || "").join("") || "";

  if (!text.trim()) throw new Error("Gemini returned empty content");
  return text.trim();
}

// FIX #3: Removed forcedMessages wrapper — main() system prompt already enforces JSON-only.
// FIX #2: Default token limit raised from 1200 → 8000.
async function callAIWithFallback({ messages }) {
  const order = (process.env.AI_PROVIDERS || "gemini,groq,openrouter")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);

  const maxTokens = Number(process.env.AI_MAX_TOKENS || 8000);
  const temperature = Number(process.env.AI_TEMPERATURE || 0.4);

  const errors = [];

  for (const provider of order) {
    try {
      if (provider === "gemini") {
        const key = process.env.GEMINI_API_KEY;
        // FIX #4: Updated to a stable Gemini model string — override via GEMINI_MODEL env if needed.
        const model = process.env.GEMINI_MODEL || "gemini-2.0-flash";
        if (!key) throw new Error("Missing GEMINI_API_KEY");

        return await callGemini({
          apiKey: key,
          model,
          messages,
          temperature,
          maxOutputTokens: maxTokens,
        });
      }

      if (provider === "groq") {
        const key = process.env.GROQ_API_KEY;
        const model = process.env.GROQ_MODEL || "llama-3.1-8b-instant";
        if (!key) throw new Error("Missing GROQ_API_KEY");

        return await callOpenAICompatible({
          endpoint: "https://api.groq.com/openai/v1/chat/completions",
          apiKey: key,
          model,
          messages,
          temperature,
          max_tokens: maxTokens,
        });
      }

      if (provider === "openrouter") {
        const key = process.env.OPENROUTER_API_KEY;
        const model = process.env.OPENROUTER_MODEL || "openai/gpt-4o-mini";
        if (!key) throw new Error("Missing OPENROUTER_API_KEY");

        return await callOpenAICompatible({
          endpoint: "https://openrouter.ai/api/v1/chat/completions",
          apiKey: key,
          model,
          messages,
          temperature,
          max_tokens: maxTokens,
        });
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

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  const payloadPath = path.join(ROOT, "client_payload", "payload.json");
  const promptPath = path.join(ROOT, "factory", "prompt_system.txt");
  const schemaPath = path.join(ROOT, "factory", "site_schema.json");

  if (!fs.existsSync(payloadPath)) throw new Error(`Missing ${payloadPath}`);

  // FIX #1: Removed duplicate top-level payload parse. Only loaded here inside main().
  const payload = readJSON(payloadPath);
  console.log("Loaded payload:", payload.project?.slug);

  const promptSystem = readText(promptPath, "");
  const schema = fs.existsSync(schemaPath) ? readJSON(schemaPath) : null;

  const appSrcDir = path.join(ROOT, "app_src");
  if (!fs.existsSync(appSrcDir)) {
    throw new Error("Missing app_src. Scaffold Next.js first (create-next-app).");
  }

  const appBase = detectAppBase(appSrcDir);
  const appDir = path.join(appBase, "app");
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
    try {
      await downloadToFile(logoUrl, out);
      logoPathHint = `/public/assets/logo${ext}`;
    } catch (e) {
      console.warn(`⚠️ Logo download failed, skipping: ${e.message}`);
    }
  }

  const gallerySaved = [];
  for (const u of galleryUrls) {
    if (!u || !isUrl(u)) continue;
    const ext = guessExtFromUrl(u) || ".jpg";
    const name = `gallery-${sha1(u)}${ext}`;
    const out = path.join(publicAssetsDir, "gallery", name);
    console.log(`⬇️ Downloading gallery: ${u} -> ${out}`);
    // FIX #6: Wrapped each gallery download in try/catch so one failure doesn't abort all.
    try {
      await downloadToFile(u, out);
      gallerySaved.push(`/public/assets/gallery/${name}`);
    } catch (e) {
      console.warn(`⚠️ Skipping gallery image ${u}: ${e.message}`);
    }
  }

  // Build AI prompt — system message enforces JSON-only output.
  const system = [
    "You are a senior frontend engineer generating a premium Next.js App Router page.",
    "Return ONLY valid JSON. No markdown. No backticks. No explanations.",
    "JSON format MUST be: { \"files\": { \"relative/path/from/app_src\": \"file content\" } }",
    "",
    "Allowed output files ONLY (write ONLY these keys in files):",
    "- src/app/layout.tsx OR app/layout.tsx",
    "- src/app/page.tsx OR app/page.tsx",
    "- src/components/Sections.tsx OR components/Sections.tsx (optional)",
    "",
    "Project rules:",
    "- Tailwind is installed. Use Tailwind classes only.",
    "- Do NOT modify package.json, next.config.*, tsconfig.*, eslint configs.",
    "- Must compile with `npm run build`.",
    "",
    "UI/UX rules (premium):",
    "- Mobile-first, clean spacing, modern typography.",
    "- Use cards, subtle borders, good hierarchy.",
    "- Add WhatsApp CTA button (using phone from payload if present).",
    "- Sections: Hero, About, Services, Gallery (if images), Contact, Footer.",
    "- Put services in a responsive grid.",
    "- In contact section include clickable phone/mail links.",
    "- If Instagram link exists show a button.",
    "",
    "SEO rules:",
    "- Set metadata title + description in layout.tsx derived from business name + services.",
    "- Use semantic HTML headings and sections.",
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

  const allowed = [
    "app/page.tsx",
    "app/layout.tsx",
    "components/Sections.tsx",
    "src/app/page.tsx",
    "src/app/layout.tsx",
    "src/components/Sections.tsx",
  ];

  let writtenCount = 0;

  for (const [rel, content] of Object.entries(files)) {
    if (typeof content !== "string") continue;

    if (!allowed.includes(rel)) {
      console.warn(`⛔ Skipping disallowed path from AI: ${rel}`);
      continue;
    }

    const outPath = safeJoin(appSrcDir, rel);
    ensureDir(path.dirname(outPath));
    fs.writeFileSync(outPath, content, "utf8");
    console.log(`✅ Wrote ${rel}`);
    writtenCount++;
  }

  // FIX #5: Validate that the AI actually wrote at least one valid file.
  if (writtenCount === 0) {
    throw new Error(
      "AI wrote no valid files. Check token limit, model output, or allowed path list.\n" +
      "AI returned keys: " + Object.keys(files).join(", ")
    );
  }

  // Mirror app/* -> src/app/* if project uses src layout
  const usesSrc = fs.existsSync(path.join(appSrcDir, "src"));
  if (usesSrc) {
    const mirrorPairs = [
      ["app/page.tsx",           "src/app/page.tsx"],
      ["app/layout.tsx",         "src/app/layout.tsx"],
      ["components/Sections.tsx","src/components/Sections.tsx"],
    ];

    for (const [from, to] of mirrorPairs) {
      const src = path.join(appSrcDir, from);
      const dest = path.join(appSrcDir, to);
      if (fs.existsSync(src)) {
        ensureDir(path.dirname(dest));
        fs.copyFileSync(src, dest);
        console.log(`🔁 Mirrored ${from} -> ${to}`);
      }
    }
  }

  console.log("🎉 Generation complete.");
}

main().catch((e) => {
  console.error("❌ factory_generate.mjs failed:", e?.stack || e);
  process.exit(1);
});
