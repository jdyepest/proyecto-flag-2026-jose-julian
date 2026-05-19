/**
 * api.js — Cliente HTTP para el backend SciText-ES
 *
 * Estrategia de fallback:
 *   1. Intenta llamar al backend real (misma origin por defecto).
 *   2. Si el servidor no está disponible (Failed to fetch, CORS, timeout),
 *      genera los datos localmente con el mock de navegador.
 *   3. El resto de la app (app.js) recibe exactamente el mismo formato
 *      sin importar si los datos vienen del servidor o del mock local.
 */

function _defaultApiBase() {
  // In prod (EC2), the frontend is served by the backend on the same origin.
  // Using window.location.origin prevents accidental calls to the user's localhost,
  // which would trigger the browser mock fallback.
  try {
    const overridden = window?.SCITEXT_API_BASE;
    if (typeof overridden === "string" && overridden.trim()) {
      return overridden.trim().replace(/\/$/, "");
    }
    const origin = window?.location?.origin;
    if (origin && origin !== "null") return origin;
  } catch {
    // Ignore.
  }
  return "http://localhost:5000";
}

const API_BASE = _defaultApiBase();
const ANALYZE_TIMEOUT_MS = 600000; // Gemini/LLM/descargas iniciales pueden tardar > 8s
// Comparación ahora es totalmente estática (sin endpoint).

/* ================================================================
   Mock de navegador (replica la lógica del backend en Python)
   ================================================================ */

const _MOCK = (() => {

  // ── Configuración de modelos ──────────────────────────────────
  const MODELS = {
    encoder: { delay: 500,  task1F1: 0.83, task2F1: 0.78, cost: 0.001 },
    llm:     { delay: 2000, task1F1: 0.79, task2F1: 0.82, cost: 0.008 },
    api:     { delay: 1500, task1F1: 0.87, task2F1: 0.91, cost: 0.035 },
  };

  // ── Resultados fijos para comparación (fuente: artifacts/eval_results/*.json) ──
  // encoder_roberta.json, openrouter_meta-llama_llama-3.1-8b-instruct.json, gemini_gemini-2.5-flash-lite.json
  const FIXED_COMPARE_METRICS = {
    task1_metrics: {
      encoder: { f1: 0.3959, precision: 0.389,  recall: 0.3367, latency: 20.4958 },
      llm:     { f1: 0.6168, precision: 0.6689, recall: 0.699,  latency: 12.632  },
      api:     { f1: 0.9195, precision: 0.9466, recall: 0.975,  latency: 5.0452  },
    },
    task2_metrics: {
      encoder: { f1: 0.3769, precision: 0.4021, recall: 0.3825, latency: 25.5897 },
      llm:     { f1: 0.6371, precision: 0.5696, recall: 0.6316, latency: 8.2272  },
      api:     { f1: 0.9366, precision: 0.9711, recall: 0.9517, latency: 6.3713  },
    },
    cost_per_doc: {
      encoder: 0.001,
      llm: 0.008,
      api: 0.035,
    },
    total_time: {
      encoder: 21514.57,
      llm: 11751.07,
      api: 5310.46,
    },
  };

  // ── Keywords por categoría retórica ──────────────────────────
  const KEYWORDS = {
    INTRO: ["introducción","este trabajo","en este artículo","el presente","motivación","objetivo","propósito","presente trabajo","este estudio"],
    BACK:  ["antecedentes","trabajos previos","revisión","estado del arte","investigaciones anteriores","han propuesto","fue propuesto","en la literatura","según","de acuerdo con"],
    METH:  ["metodología","método","procedimiento","experimento","implementación","arquitectura","entrenamiento","corpus","dataset","conjunto de datos","evaluamos","utilizamos","se utilizó","fine-tuning","hiperparámetros","configuración"],
    RES:   ["resultado","tabla","figura","obtuvo","obtuvimos","rendimiento","accuracy","f1","precisión","recall","métricas","muestra","se observa","se puede ver","en la tabla","en la figura"],
    DISC:  ["discusión","análisis","interpretamos","esto sugiere","esto indica","podemos inferir","se debe a","explicar","probable","parece","comparado con","en comparación"],
    CONTR: ["contribución","aporte","propuesta","novedad","innovación","original","nueva","nuevo","presentamos","proponemos","a diferencia de","primer trabajo"],
    LIM:   ["limitación","limitaciones","trabajo futuro","futuras investigaciones","no se consideró","fuera del alcance","restricción","sesgo"],
    CONC:  ["conclusión","conclusiones","en conclusión","en resumen","resumiendo","concluimos","hemos mostrado","hemos demostrado","finalmente","en definitiva"],
  };

  const ORDER = ["INTRO","BACK","METH","RES","DISC","CONTR","LIM","CONC"];

  // ── PRNG determinista (xorshift) ─────────────────────────────
  function seededRng(seed) {
    let s = (seed >>> 0) || 1;
    return () => {
      s ^= s << 13; s ^= s >> 17; s ^= s << 5;
      return ((s >>> 0) / 0xFFFFFFFF);
    };
  }

  function strHash(str) {
    let h = 0x811c9dc5;
    for (let i = 0; i < Math.min(str.length, 120); i++) {
      h ^= str.charCodeAt(i);
      h = (h * 0x01000193) >>> 0;
    }
    return h;
  }

  function jitter(rng, base, spread) {
    return Math.min(0.99, Math.max(0.10, base + (rng() * 2 - 1) * spread));
  }

  // ── Dividir texto en párrafos ─────────────────────────────────
  function splitParagraphs(text) {
    return text.split(/\n\s*\n/)
      .map(p => p.trim())
      .filter(p => p.length > 20);
  }

  // ── Puntaje por keywords ──────────────────────────────────────
  function keywordScores(para) {
    const lower = para.toLowerCase();
    const scores = {};
    for (const [lbl, kws] of Object.entries(KEYWORDS)) {
      scores[lbl] = kws.filter(k => lower.includes(k)).length;
    }
    return scores;
  }

  // ── Label + confianza para un párrafo ────────────────────────
  function labelParagraph(para, idx, total, model, rng) {
    const scores = keywordScores(para);
    const maxScore = Math.max(...Object.values(scores));
    let label, baseConf;

    if (idx === 0 && maxScore < 2) {
      label = "INTRO"; baseConf = 0.82;
    } else if (idx === total - 1 && maxScore < 2) {
      label = "CONC"; baseConf = 0.80;
    } else if (maxScore > 0) {
      label = Object.entries(scores).sort((a,b) => b[1]-a[1])[0][0];
      baseConf = 0.72 + Math.min(maxScore * 0.05, 0.22);
    } else {
      const ratio = idx / Math.max(total - 1, 1);
      label = ORDER[Math.round(ratio * (ORDER.length - 1))];
      baseConf = 0.62 + rng() * 0.10;
    }

    const noise = model === "encoder" ? (rng()*0.12-0.06)
                : model === "llm"     ? (rng()*0.09-0.04)
                :                       (rng()*0.06-0.02);

    return { label, confidence: Math.round(jitter(rng, baseConf + noise, 0) * 100) / 100 };
  }

  // ── Highlight: primera frase de ≥6 palabras ──────────────────
  function findHighlight(text) {
    const sentences = text.split(/[.;]/);
    for (const s of sentences) {
      const words = s.trim().split(/\s+/);
      if (words.length >= 6) return words.slice(0, 10).join(" ");
    }
    return text.slice(0, 80).trim();
  }

  // ── Generar UUID simple ───────────────────────────────────────
  function uuid() {
    return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, c => {
      const r = Math.random() * 16 | 0;
      return (c === "x" ? r : (r & 0x3 | 0x8)).toString(16);
    });
  }

  // ── sleep ─────────────────────────────────────────────────────
  function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

  // ── Mock /api/analyze ─────────────────────────────────────────
  async function mockAnalyze(text, model, tasks) {
    const cfg = MODELS[model] || MODELS.encoder;
    await sleep(cfg.delay);

    const rng   = seededRng(strHash(text.slice(0,100)) ^ strHash(model));
    const paras = splitParagraphs(text);
    const total = paras.length || 1;
    const id    = uuid();

    // Tarea 1 ─ segmentación
    const segments = paras.map((para, i) => {
      const { label, confidence } = labelParagraph(para, i, total, model, rng);
      return { paragraph_index: i, text: para, label, confidence };
    });

    const totalWords  = paras.reduce((s, p) => s + p.split(/\s+/).length, 0);
    const avgConf     = Math.round(segments.reduce((s,g) => s + g.confidence, 0) / total * 1000) / 1000;
    const timeSec     = Math.round((cfg.delay / 1000 + rng() * 0.4) * 100) / 100;

    const segmentation = {
      segments,
      stats: { total_paragraphs: total, total_words: totalWords, avg_confidence: avgConf, time_seconds: timeSec },
    };

    // Tarea 2 ─ contribuciones
    const HIGH = new Set(["CONTR","METH","RES"]);
    const MED  = new Set(["DISC","CONC","INTRO"]);

    const fragments = segments.map(seg => {
      const baseProb = HIGH.has(seg.label) ? 0.80 : MED.has(seg.label) ? 0.40 : 0.20;
      const probNoise = model === "encoder" ? rng()*0.15-0.10
                      : model === "llm"     ? rng()*0.13-0.05
                      :                       rng()*0.10;
      const isContrib = rng() < baseProb + probNoise;

      let baseC = isContrib
        ? 0.78 + rng()*0.18 + (model === "api" ? 0.04 : model === "encoder" ? -0.03 : 0)
        : 0.30 + rng()*0.30;
      baseC = Math.min(0.99, Math.max(0.10, Math.round(baseC * 100) / 100));

      const highlight = isContrib ? findHighlight(seg.text) : "";

      return {
        paragraph_index: seg.paragraph_index,
        text:           seg.text,
        is_contribution: isContrib,
        contribution_type: null,
        confidence:     baseC,
        highlight,
        source_label:   seg.label,
      };
    });

    const positives = fragments.filter(f => f.is_contribution);
    const avgCP = positives.length
      ? Math.round(positives.reduce((s,f) => s + f.confidence, 0) / positives.length * 1000) / 1000
      : 0;

    const contributions = {
      fragments,
      stats: { total_fragments: fragments.length, positive: positives.length, negative: fragments.length - positives.length, avg_confidence_positive: avgCP },
    };

    return {
      id,
      model,
      model_name: { encoder:"Encoder (SciBERT)", llm:"Llama 3.1 8B Instruct (OpenRouter)", api:"API Comercial" }[model],
      segmentation: tasks.includes("segmentation") ? segmentation : null,
      contributions: tasks.includes("contributions") ? contributions : null,
    };
  }

  // ── Mock /api/compare ─────────────────────────────────────────
  // Usa valores fijos de artifacts/eval_results para evitar variación en cada carga.
  async function mockCompare(analysisId) {
    return {
      analysis_id: analysisId,
      ...FIXED_COMPARE_METRICS,
    };
  }

  return { mockAnalyze, mockCompare };
})();

/* ================================================================
   Funciones públicas con fallback automático
   ================================================================ */

function _isNetworkLikeError(err) {
  // fetch() failures are usually TypeError (DNS/refused/CORS/etc).
  // No hacemos fallback en timeout (AbortError) porque escondería fallos/latencia real del backend.
  return err instanceof TypeError;
}

/**
 * POST /api/analyze — con fallback a mock de navegador.
 */
async function apiAnalyze(text, model, tasks) {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), ANALYZE_TIMEOUT_MS);

    const resp = await fetch(`${API_BASE}/api/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, model, tasks, encoder_variant: "scibert" }),
      signal: controller.signal,
    });
    clearTimeout(timer);

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || err.error || `Error del servidor (${resp.status})`);
    }
    return resp.json();

  } catch (fetchErr) {
    // Solo hacemos fallback cuando el backend no es alcanzable (red/timeout).
    // Si el backend respondió con error (p.ej. Ollama/Gemini falló) queremos que se vea el fallo real.
    if (fetchErr?.name === "AbortError") {
      throw new Error(`Timeout llamando al backend (${Math.round(ANALYZE_TIMEOUT_MS/1000)}s). Reintenta o aumenta el límite.`);
    }
    if (_isNetworkLikeError(fetchErr)) {
      console.warn("[SciText-ES] Backend no disponible, usando mock local:", fetchErr.message);
      return _MOCK.mockAnalyze(text, model, tasks);
    }
    throw fetchErr;
  }
}

/**
 * Comparación totalmente estática (sin endpoint).
 */
async function apiCompare(analysisId) {
  return _MOCK.mockCompare(analysisId);
}
