/**
 * app.js — Lógica principal y navegación
 *
 * Estado de la aplicación almacenado en sessionStorage:
 *   scitext_analysis  → resultado completo del POST /api/analyze
 */

const STATE_KEY = "scitext_analysis";
const SIDE_COMPARE_KEY = "scitext_side_compare";
const MAX_TOTAL_COMPARE_MODELS = 3;

/* ------------------------------------------------------------------ */
/* Helpers de estado                                                    */
/* ------------------------------------------------------------------ */

function saveAnalysis(data) {
  sessionStorage.setItem(STATE_KEY, JSON.stringify(data));
}

function loadAnalysis() {
  const raw = sessionStorage.getItem(STATE_KEY);
  if (!raw) return null;
  try { return JSON.parse(raw); } catch { return null; }
}

function clearAnalysis() {
  sessionStorage.removeItem(STATE_KEY);
  sessionStorage.removeItem(SIDE_COMPARE_KEY);
}

function saveSideCompare(data) {
  sessionStorage.setItem(SIDE_COMPARE_KEY, JSON.stringify(data));
}

function loadSideCompare() {
  const raw = sessionStorage.getItem(SIDE_COMPARE_KEY);
  if (!raw) return null;
  try { return JSON.parse(raw); } catch { return null; }
}

function clearSideCompare() {
  sessionStorage.removeItem(SIDE_COMPARE_KEY);
}

/* ------------------------------------------------------------------ */
/* Helpers de UI                                                        */
/* ------------------------------------------------------------------ */

function showLoading(msg = "Analizando documento…") {
  const el = document.getElementById("loading-overlay");
  if (!el) return;
  const txt = el.querySelector(".loading-text");
  if (txt) txt.textContent = msg;
  el.classList.add("visible");
}

function hideLoading() {
  const el = document.getElementById("loading-overlay");
  if (el) el.classList.remove("visible");
}

function showError(msg, containerId = "error-container") {
  const el = document.getElementById(containerId);
  if (!el) { console.error(msg); return; }
  el.innerHTML = `<div class="error-banner">⚠️ ${msg}</div>`;
  el.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function clearError(containerId = "error-container") {
  const el = document.getElementById(containerId);
  if (el) el.innerHTML = "";
}

/* ------------------------------------------------------------------ */
/* Helpers de render                                                    */
/* ------------------------------------------------------------------ */

const LABEL_COLORS = {
  INTRO: "#1565C0", BACK: "#6A1B9A", METH: "#00838F",
  RES:   "#2E7D32", DISC: "#E65100", CONTR: "#C62828",
  LIM:   "#4E342E", CONC: "#283593",
};

const LABEL_NAMES = {
  INTRO: "Introducción", BACK: "Antecedentes", METH: "Metodología",
  RES:   "Resultados",   DISC: "Discusión",   CONTR: "Contribución",
  LIM:   "Limitaciones", CONC: "Conclusión",
};

function confidenceClass(val) {
  if (val >= 0.80) return "conf-high";
  if (val >= 0.60) return "conf-med";
  return "conf-low";
}

function confPct(val) {
  return Math.round(val * 100) + "%";
}

function renderConfBar(confidence, width = 180) {
  const cls = confidenceClass(confidence);
  const pct = confPct(confidence);
  return `
    <div class="confidence-bar" style="max-width:${width}px">
      <div class="confidence-bar-track">
        <div class="confidence-bar-fill ${cls}" style="width:${pct}"></div>
      </div>
      <span class="confidence-value">${pct}</span>
    </div>`;
}

function labelBadge(label) {
  return `<span class="badge badge-label badge-${label.toLowerCase()}">${label}</span>`;
}

function modelName(model) {
  const names = {
    encoder: "Encoder (BETO/RoBERTa)",
    llm: "Llama 3.3 70B Instruct (OpenRouter)",
    api: "API Comercial",
  };
  return names[model] || model;
}

function modelIcon(model) {
  const icons = { encoder: "⚡", llm: "🧠", api: "☁️" };
  return icons[model] || "•";
}

function modelShortName(model) {
  const names = { encoder: "Encoder", llm: "Llama 70B", api: "API" };
  return names[model] || model;
}

/* ------------------------------------------------------------------ */
/* Vista 1 — Entrada                                                    */
/* ------------------------------------------------------------------ */

function initInputView() {
  const textarea   = document.getElementById("text-input");
  const counter    = document.getElementById("char-counter");
  const form       = document.getElementById("analyze-form");
  const modelCards = document.querySelectorAll(".model-card");

  if (!textarea || !form) return;

  // Contador de caracteres
  textarea.addEventListener("input", () => {
    const len = textarea.value.length;
    counter.textContent = `${len} caracteres`;
    counter.className = "char-counter" +
      (len < 250 ? " warning" : " ok");
  });

  // Selección de tarjeta de modelo
  modelCards.forEach(card => {
    card.addEventListener("click", () => {
      modelCards.forEach(c => c.classList.remove("selected"));
      card.classList.add("selected");
      card.querySelector("input[type='radio']").checked = true;
    });
  });

  // Envío del formulario
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    clearError();

    const text = textarea.value.trim();
    if (text.length < 50) {
      showError("El texto debe tener al menos 50 caracteres.");
      return;
    }

    const model = form.querySelector("input[name='model']:checked")?.value || "encoder";
    const tasks = [];
    if (form.querySelector("#task-seg")?.checked)   tasks.push("segmentation");
    if (form.querySelector("#task-cont")?.checked)  tasks.push("contributions");

    if (tasks.length === 0) {
      showError("Selecciona al menos una tarea.");
      return;
    }

    showLoading("Analizando documento…");
    try {
      const result = await apiAnalyze(text, model, tasks);
      result.input_text = text;
      result.tasks = tasks;
      saveAnalysis(result);
      clearSideCompare();
      if (tasks.includes("segmentation")) {
        window.location.href = "segmentation.html";
      } else if (tasks.includes("contributions")) {
        window.location.href = "contributions.html";
      } else {
        window.location.href = "comparison.html";
      }
    } catch (err) {
      hideLoading();
      showError(err.message);
    }
  });
}

/* ------------------------------------------------------------------ */
/* Vista 2 — Segmentación retórica                                      */
/* ------------------------------------------------------------------ */

function initSegmentationView() {
  const analysis = loadAnalysis();

  if (analysis && !analysis.segmentation && analysis.contributions) {
    window.location.href = "contributions.html";
    return;
  }

  if (!analysis || !analysis.segmentation) {
    document.getElementById("segments-container").innerHTML = `
      <div class="empty-state">
        <div class="empty-state-icon">📄</div>
        <h3>Sin datos de segmentación</h3>
        <p><a href="index.html">Realiza un nuevo análisis</a></p>
      </div>`;
    return;
  }

  const { segments, stats } = analysis.segmentation;

  // Contar por categoría
  const counts = {};
  segments.forEach(s => { counts[s.label] = (counts[s.label] || 0) + 1; });

  // Sidebar: leyenda
  const legendEl = document.getElementById("legend-list");
  if (legendEl) {
    legendEl.innerHTML = Object.entries(LABEL_COLORS).map(([lbl, color]) => `
      <li class="legend-item">
        <div class="legend-dot-label">
          <span class="legend-dot" style="background:${color}"></span>
          <span>${LABEL_NAMES[lbl]}</span>
        </div>
        <span class="legend-count">${counts[lbl] || 0}</span>
      </li>`).join("");
  }

  // Sidebar: estadísticas
  const statsEl = document.getElementById("sidebar-stats");
  if (statsEl) {
    statsEl.innerHTML = `
      <div class="stat-row"><span class="stat-label">Párrafos</span><span class="stat-value">${stats.total_paragraphs}</span></div>
      <div class="stat-row"><span class="stat-label">Palabras</span><span class="stat-value">${stats.total_words.toLocaleString("es")}</span></div>
      <div class="stat-row"><span class="stat-label">Confianza media</span><span class="stat-value">${confPct(stats.avg_confidence)}</span></div>
      <div class="stat-row"><span class="stat-label">Tiempo</span><span class="stat-value">${stats.time_seconds}s</span></div>
    `;
  }

  // Sidebar: badge del modelo
  const modelEl = document.getElementById("model-badge");
  if (modelEl) {
    modelEl.innerHTML = `
      <span class="text-secondary text-small">Modelo usado</span>
      <strong>${modelName(analysis.model)}</strong>`;
  }

  // Segmentos
  const container = document.getElementById("segments-container");
  container.innerHTML = segments.map((seg, i) => `
    <div class="segment-card seg-${seg.label}">
      <div class="segment-header">
        <div class="segment-meta">
          ${labelBadge(seg.label)}
          <span class="segment-index">Párrafo ${i + 1}</span>
        </div>
        <div class="segment-confidence">
          ${renderConfBar(seg.confidence, 200)}
        </div>
      </div>
      <div class="segment-text">${escHtml(seg.text)}</div>
    </div>`).join("");

  // Botón siguiente
  const btnNext = document.getElementById("btn-contributions");
  if (btnNext) {
    const hasContributions = Boolean(analysis.contributions);
    btnNext.textContent = hasContributions ? "Ver contribuciones →" : "Comparar modelos →";
    btnNext.addEventListener("click", () => {
      if (hasContributions) {
        window.location.href = "contributions.html";
      } else {
        window.location.href = "comparison.html";
      }
    });
  }
}

/* ------------------------------------------------------------------ */
/* Vista 3 — Contribuciones                                             */
/* ------------------------------------------------------------------ */

function initContributionsView() {
  const analysis = loadAnalysis();

  if (!analysis || !analysis.contributions) {
    document.getElementById("fragments-container").innerHTML = `
      <div class="empty-state">
        <div class="empty-state-icon">🔍</div>
        <h3>Sin datos de contribuciones</h3>
        <p><a href="index.html">Realiza un nuevo análisis</a></p>
      </div>`;
    return;
  }

  const { fragments, stats } = analysis.contributions;

  // Tarjetas de resumen
  const sc = document.getElementById("summary-cards");
  if (sc) {
    sc.innerHTML = `
      <div class="summary-card neutral">
        <div class="summary-card-value">${stats.total_fragments}</div>
        <div class="summary-card-label">Fragmentos analizados</div>
      </div>
      <div class="summary-card positive">
        <div class="summary-card-value">${stats.positive}</div>
        <div class="summary-card-label">Contribuciones encontradas</div>
      </div>
      <div class="summary-card neutral">
        <div class="summary-card-value">${confPct(stats.avg_confidence_positive)}</div>
        <div class="summary-card-label">Confianza media</div>
      </div>
      <div class="summary-card neutral">
        <div class="summary-card-value">${stats.negative}</div>
        <div class="summary-card-label">Sin contribución</div>
      </div>`;
  }

  // Lista de fragmentos
  const container = document.getElementById("fragments-container");
  container.innerHTML = fragments.map((frag, i) => {
    const isPos = frag.is_contribution;
    const bodyText = isPos && frag.highlight
      ? highlightText(frag.text, frag.highlight)
      : escHtml(frag.text);

    return `
      <div class="fragment-card ${isPos ? "" : "no-contribution"}">
        <div class="fragment-header">
          <div class="fragment-header-left">
            <span class="${isPos ? "badge-contribution" : "badge-no-contribution"}">
              ${isPos ? "✓ Contribución detectada" : "— Sin contribución"}
            </span>
            ${isPos && frag.contribution_type ? `<span class="badge-type">${frag.contribution_type}</span>` : ""}
            <span class="para-index">Párrafo ${frag.paragraph_index + 1}</span>
          </div>
          <div class="fragment-header-right">
            <div class="fragment-conf-wrapper">
              ${renderConfBar(frag.confidence, 160)}
            </div>
          </div>
        </div>
        <div class="fragment-body">${bodyText}</div>
        <div class="fragment-footer">
          <span class="footer-label">Sección:</span>
          ${labelBadge(frag.source_label)}
        </div>
      </div>`;
  }).join("");

  // Botón exportar
  const btnExport = document.getElementById("btn-export");
  if (btnExport) {
    btnExport.addEventListener("click", () => exportJson(analysis, "scitext_analysis.json"));
  }
}

/* ------------------------------------------------------------------ */
/* Vista 4 — Comparación                                               */
/* ------------------------------------------------------------------ */

async function initComparisonView() {
  const analysis = loadAnalysis();
  const analysisId = analysis?.id || "static";

  showLoading("Cargando métricas comparativas…");
  try {
    const metrics = await apiCompare(analysisId);
    hideLoading();
    renderComparison(metrics);
  } catch (err) {
    hideLoading();
    showError(err.message, "comparison-error");
  }

  initSideBySideCompare(analysis);

  // Exportar reporte
  const btnExport = document.getElementById("btn-export-report");
  if (btnExport) {
    btnExport.addEventListener("click", () => {
      try {
        if (!analysis) {
          showError("No hay análisis cargado para exportar.", "comparison-error");
          return;
        }
        const persisted = loadSideCompare();
        const compared = (
          persisted?.baseAnalysisId === (analysis.id || null) && Array.isArray(persisted.comparedAnalyses)
        ) ? persisted.comparedAnalyses : [];
        const report = buildParagraphExportReport(analysis, compared);
        exportJson(report, "scitext_paragraph_report.json");
      } catch (e) {
        showError(e.message, "comparison-error");
      }
    });
  }
}

function renderComparison(metrics) {
  const { task1_metrics, task2_metrics } = metrics;
  const models = ["encoder", "llm", "api"];
  const modelIcons = { encoder: "⚡", llm: "🧠", api: "☁️" };
  const modelColors = { encoder: "#1565C0", llm: "#7B1FA2", api: "#E65100" };
  const qualitativeCostScale = { encoder: 1, llm: 2, api: 3 };
  const qualitativeCostLabel = { encoder: "Low", llm: "Medium", api: "High" };

  // Tabla helper
  function renderTable(tableMetrics, containerId) {
    const el = document.getElementById(containerId);
    if (!el) return;

    const metricKeys = ["f1", "precision", "recall"];
    const metricLabels = { f1: "F1-Score", precision: "Precisión", recall: "Recall" };

    // Encontrar el mejor valor por métrica
    const bestPerMetric = {};
    metricKeys.forEach(k => {
      const vals = models.map(m => tableMetrics[m][k]);
      bestPerMetric[k] = Math.max(...vals);
    });

    el.innerHTML = `
      <table class="metrics-table">
        <thead>
          <tr>
            <th>Modelo</th>
            ${metricKeys.map(k => `<th>${metricLabels[k]}</th>`).join("")}
          </tr>
        </thead>
        <tbody>
          ${models.map(m => `
            <tr>
              <td>
                <div class="model-cell">
                  <span class="model-dot" style="background:${modelColors[m]}"></span>
                  ${modelIcons[m]} ${m.charAt(0).toUpperCase() + m.slice(1)}
                </div>
              </td>
              ${metricKeys.map(k => {
                const val = tableMetrics[m][k];
                const isBest = Math.abs(val - bestPerMetric[k]) < 0.001;
                const display = (val * 100).toFixed(1) + "%";
                return `<td class="metric-value ${isBest ? "best-value" : ""}">
                  ${display}${isBest ? '<span class="best-star">★</span>' : ""}
                </td>`;
              }).join("")}
            </tr>`).join("")}
        </tbody>
      </table>`;
  }

  renderTable(task1_metrics, "table-task1");
  renderTable(task2_metrics, "table-task2");
  renderBarChart(
    "chart-cost",
    qualitativeCostScale,
    models,
    modelColors,
    m => qualitativeCostLabel[m],
    "Qualitative"
  );

  // Trade-offs
  renderTradeoffs();
}

function renderBarChart(containerId, values, models, colors, labelFn, unit) {
  const el = document.getElementById(containerId);
  if (!el) return;

  const max = Math.max(...models.map(m => values[m]));
  const modelNames = { encoder: "Encoder", llm: "Llama 3.3 70B Instruct", api: "API" };
  const modelIcons = { encoder: "⚡", llm: "🧠", api: "☁️" };

  el.innerHTML = models.map(m => {
    const pct = Math.round((values[m] / max) * 100);
    return `
      <div class="bar-row">
        <div class="bar-label">${modelIcons[m]} ${modelNames[m]}</div>
        <div class="bar-track">
          <div class="bar-fill bar-${m}" style="width:${pct}%"></div>
        </div>
        <div class="bar-value">${labelFn(m)}</div>
      </div>`;
  }).join("");
}

function renderTradeoffs() {
  const el = document.getElementById("tradeoff-cards");
  if (!el) return;

  const cards = [
    {
      model: "encoder", icon: "⚡", name: "Encoder",
      color: "#1565C0",
      pros: ["Costo: Low", "Sin dependencia de API externa", "Reproducible y auditable", "Buen equilibrio general"],
      cons: ["Menor F1 en Tarea 2 vs API", "Limitado a idiomas entrenados", "Requiere GPU para escala"],
    },
    {
      model: "llm", icon: "🧠", name: "Llama 3.3 70B Instruct (OpenRouter)",
      color: "#7B1FA2",
      pros: ["Calidad alta sin GPU local", "Integración rápida vía API", "Buen recall en Tarea 2", "Escala sin infraestructura propia"],
      cons: ["Rate limits en modelo free", "Dependencia de proveedor externo", "Costo: Medium"],
    },
    {
      model: "api", icon: "☁️", name: "API Comercial",
      color: "#E65100",
      pros: ["Mayor F1 general", "Sin setup de infraestructura", "Modelos actualizados", "Mejor en textos cortos"],
      cons: ["Costo: High", "Dependencia de proveedor externo", "Sin control sobre el modelo"],
    },
  ];

  el.innerHTML = cards.map(c => `
    <div class="tradeoff-card">
      <div class="tradeoff-card-title" style="color:${c.color}">
        ${c.icon} ${c.name}
      </div>
      <ul class="tradeoff-pros">
        ${c.pros.map(p => `<li>${p}</li>`).join("")}
      </ul>
      <div class="tradeoff-divider"></div>
      <ul class="tradeoff-cons">
        ${c.cons.map(p => `<li>${p}</li>`).join("")}
      </ul>
    </div>`).join("");
}

function inferAnalysisTasks(analysis) {
  if (Array.isArray(analysis?.tasks) && analysis.tasks.length) {
    return analysis.tasks.slice();
  }
  const tasks = [];
  if (analysis?.segmentation) tasks.push("segmentation");
  if (analysis?.contributions) tasks.push("contributions");
  return tasks.length ? tasks : ["segmentation", "contributions"];
}

function splitParagraphsFromText(text) {
  return String(text || "")
    .split(/\n\s*\n/)
    .map((p) => p.trim())
    .filter((p) => p.length > 0);
}

function getAnalysisParagraphTexts(analysis) {
  const fromSeg = analysis?.segmentation?.segments?.map((s) => s.text).filter(Boolean) || [];
  if (fromSeg.length) return fromSeg;
  const fromCont = analysis?.contributions?.fragments?.map((f) => f.text).filter(Boolean) || [];
  if (fromCont.length) return fromCont;
  return splitParagraphsFromText(analysis?.input_text || "");
}

function topSegmentationLabels(segments, maxItems = 3) {
  const counts = {};
  (segments || []).forEach((s) => {
    counts[s.label] = (counts[s.label] || 0) + 1;
  });
  return Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, maxItems)
    .map(([label, count]) => `${label} (${count})`);
}

function renderModelSummaryCard(analysis, subtitle, tasks) {
  const segStats = analysis?.segmentation?.stats || null;
  const contStats = analysis?.contributions?.stats || null;
  const segTop = topSegmentationLabels(analysis?.segmentation?.segments || []).join(" · ");
  const showTask1 = tasks.includes("segmentation");
  const showTask2 = tasks.includes("contributions");

  return `
    <div class="side-compare-model-card ${analysis.model}">
      <div class="side-compare-model-title">${modelIcon(analysis.model)} ${modelName(analysis.model)}</div>
      <div class="side-compare-model-subtitle">${subtitle}</div>
      <div class="side-compare-stat-grid">
        ${showTask1 ? `
          <div class="side-compare-stat">
            <div class="side-compare-stat-label">T1 Párrafos</div>
            <div class="side-compare-stat-value">${segStats ? segStats.total_paragraphs : "N/A"}</div>
          </div>
          <div class="side-compare-stat">
            <div class="side-compare-stat-label">T1 Conf. media</div>
            <div class="side-compare-stat-value">${segStats ? confPct(segStats.avg_confidence) : "N/A"}</div>
          </div>
        ` : ""}
        ${showTask2 ? `
          <div class="side-compare-stat">
            <div class="side-compare-stat-label">T2 Positivas</div>
            <div class="side-compare-stat-value">${contStats ? `${contStats.positive}/${contStats.total_fragments}` : "N/A"}</div>
          </div>
          <div class="side-compare-stat">
            <div class="side-compare-stat-label">T2 Conf. media</div>
            <div class="side-compare-stat-value">${contStats ? confPct(contStats.avg_confidence_positive) : "N/A"}</div>
          </div>
        ` : ""}
      </div>
      ${showTask1 ? `<p class="side-compare-hint mt-1">${segTop ? `Top etiquetas T1: ${segTop}` : "Sin segmentación para resumir."}</p>` : ""}
    </div>
  `;
}

function predictionForParagraph(analysis, idx) {
  const seg = analysis?.segmentation?.segments?.[idx] || null;
  const cont = analysis?.contributions?.fragments?.[idx] || null;
  return {
    label: seg?.label || null,
    labelConfidence: seg?.confidence ?? null,
    isContribution: typeof cont?.is_contribution === "boolean" ? cont.is_contribution : null,
    contributionConfidence: cont?.confidence ?? null,
  };
}

function buildParagraphExportReport(baseAnalysis, comparedAnalyses) {
  const tasks = inferAnalysisTasks(baseAnalysis);
  const list = Array.isArray(comparedAnalyses) ? comparedAnalyses : [];
  const models = [baseAnalysis, ...list];
  const paragraphs = getAnalysisParagraphTexts(baseAnalysis);

  return {
    generated_at: new Date().toISOString(),
    source_analysis_id: baseAnalysis?.id || null,
    source_model: baseAnalysis?.model || null,
    tasks,
    models: models.map((m) => ({
      model: m.model,
      model_name: modelName(m.model),
    })),
    paragraphs: paragraphs.map((text, idx) => {
      const predictions = models.map((m) => {
        const pred = predictionForParagraph(m, idx);
        const entry = {
          model: m.model,
          model_name: modelName(m.model),
        };
        if (tasks.includes("segmentation")) {
          entry.task1_label = pred.label;
          entry.task1_confidence = pred.labelConfidence;
        }
        if (tasks.includes("contributions")) {
          entry.task2_is_contribution = pred.isContribution;
          entry.task2_confidence = pred.contributionConfidence;
        }
        return entry;
      });
      return {
        paragraph_index: idx + 1,
        text,
        predictions,
      };
    }),
  };
}

function renderPredictionCard(modelAnalysis, idx, tasks) {
  const pred = predictionForParagraph(modelAnalysis, idx);
  const showTask1 = tasks.includes("segmentation");
  const showTask2 = tasks.includes("contributions");

  const task1Chip = showTask1
    ? (pred.label
      ? `${labelBadge(pred.label)}`
      : `<span class="side-compare-hint">Sin etiqueta</span>`)
    : "";
  const task1Conf = showTask1 && pred.labelConfidence != null
    ? `<span class="text-small text-secondary">${confPct(pred.labelConfidence)}</span>`
    : "";

  let task2Chip = "";
  let task2Conf = "";
  if (showTask2) {
    if (pred.isContribution == null) {
      task2Chip = `<span class="side-compare-hint">Sin predicción</span>`;
    } else {
      task2Chip = `<span class="side-compare-contrib ${pred.isContribution ? "yes" : "no"}">${pred.isContribution ? "Contribución" : "No contribución"}</span>`;
    }
    if (pred.contributionConfidence != null) {
      task2Conf = `<span class="text-small text-secondary">${confPct(pred.contributionConfidence)}</span>`;
    }
  }

  return `
    <div class="side-compare-prediction-card">
      <div class="side-compare-prediction-title">${modelIcon(modelAnalysis.model)} ${modelShortName(modelAnalysis.model)}</div>
      ${showTask1 ? `<div class="side-compare-chip-row">${task1Chip}${task1Conf}</div>` : ""}
      ${showTask2 ? `<div class="side-compare-chip-row mt-1">${task2Chip}${task2Conf}</div>` : ""}
    </div>
  `;
}

function renderParagraphCompareList(baseAnalysis, comparedAnalyses) {
  const tasks = inferAnalysisTasks(baseAnalysis);
  const models = [baseAnalysis, ...comparedAnalyses];
  const paragraphs = getAnalysisParagraphTexts(baseAnalysis);
  if (!paragraphs.length) {
    return `
      <div class="empty-state">
        <div class="empty-state-icon">📄</div>
        <h3>Sin párrafos para comparar</h3>
        <p>Ejecuta de nuevo el análisis con texto válido.</p>
      </div>
    `;
  }

  return `
    <div class="metrics-section">
      <h2>Comparación por párrafo</h2>
      <div class="side-compare-paragraphs">
        ${paragraphs.map((text, idx) => `
          <div class="side-compare-paragraph-card">
            <div class="side-compare-para-header">
              <div class="side-compare-para-meta">
                <div class="side-compare-para-index">Párrafo ${idx + 1}</div>
                <div class="side-compare-para-text">${escHtml(text)}</div>
              </div>
              <button class="btn btn-secondary btn-toggle-para" data-idx="${idx}" data-open="0">Ver más</button>
            </div>
            <div class="side-compare-prediction-grid">
              ${models.map((m) => renderPredictionCard(m, idx, tasks)).join("")}
            </div>
            <div id="side-para-details-${idx}" class="side-compare-para-details hidden">
              <div class="side-compare-para-fulltext">${escHtml(text)}</div>
            </div>
          </div>
        `).join("")}
      </div>
    </div>
  `;
}

function bindParagraphToggleHandlers() {
  document.querySelectorAll(".btn-toggle-para").forEach((btn) => {
    btn.addEventListener("click", () => {
      const idx = btn.getAttribute("data-idx");
      const details = document.getElementById(`side-para-details-${idx}`);
      if (!details) return;
      const isOpen = btn.getAttribute("data-open") === "1";
      if (isOpen) {
        details.classList.add("hidden");
        btn.setAttribute("data-open", "0");
        btn.textContent = "Ver más";
      } else {
        details.classList.remove("hidden");
        btn.setAttribute("data-open", "1");
        btn.textContent = "Ver menos";
      }
    });
  });
}

function renderSideBySideContent(baseAnalysis, comparedAnalyses) {
  const contentEl = document.getElementById("side-compare-content");
  if (!contentEl) return;
  const list = Array.isArray(comparedAnalyses) ? comparedAnalyses : [];
  const tasks = inferAnalysisTasks(baseAnalysis);

  if (!list.length) {
    contentEl.innerHTML = `
      <div class="side-compare-model-grid">
        ${renderModelSummaryCard(baseAnalysis, "Modelo base (resultado actual)", tasks)}
        <div class="side-compare-model-card">
          <div class="side-compare-model-title">Comparación pendiente</div>
          <p class="side-compare-hint">Selecciona otro modelo y pulsa <strong>Agregar modelo</strong>.</p>
        </div>
      </div>
      ${renderParagraphCompareList(baseAnalysis, [])}
    `;
    bindParagraphToggleHandlers();
    return;
  }

  contentEl.innerHTML = `
    <div class="side-compare-model-grid">
      ${renderModelSummaryCard(baseAnalysis, "Modelo base (resultado actual)", tasks)}
      ${list.map((m) => renderModelSummaryCard(m, "Modelo agregado para comparación", tasks)).join("")}
    </div>
    ${renderParagraphCompareList(baseAnalysis, list)}
  `;
  bindParagraphToggleHandlers();
}

function buildAvailableModelOptions(baseModel, comparedAnalyses) {
  const used = new Set([baseModel, ...comparedAnalyses.map((a) => a.model)]);
  return ["encoder", "llm", "api"].filter((m) => !used.has(m));
}

function renderModelSelect(selectEl, options) {
  if (!options.length) {
    selectEl.innerHTML = `<option value="">No hay más modelos disponibles</option>`;
    selectEl.disabled = true;
    return;
  }
  selectEl.disabled = false;
  selectEl.innerHTML = options.map((m) => `<option value="${m}">${modelIcon(m)} ${modelName(m)}</option>`).join("");
}

function initSideBySideCompare(analysis) {
  const selectEl = document.getElementById("compare-model-select");
  const buttonEl = document.getElementById("btn-run-side-compare");
  const contentEl = document.getElementById("side-compare-content");
  if (!selectEl || !buttonEl || !contentEl) return;

  if (!analysis?.input_text) {
    buttonEl.disabled = true;
    selectEl.disabled = true;
    contentEl.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-icon">ℹ️</div>
        <h3>No hay texto fuente para comparar</h3>
        <p>Ejecuta un análisis nuevo desde Entrada para habilitar el side by side.</p>
      </div>
    `;
    return;
  }

  let comparedAnalyses = [];

  const persisted = loadSideCompare();
  if (persisted?.baseAnalysisId === (analysis.id || null) && Array.isArray(persisted.comparedAnalyses)) {
    comparedAnalyses = persisted.comparedAnalyses.filter((a) => a && a.model && a.model !== analysis.model);
  } else {
    clearSideCompare();
  }

  if (comparedAnalyses.length > (MAX_TOTAL_COMPARE_MODELS - 1)) {
    comparedAnalyses = comparedAnalyses.slice(0, MAX_TOTAL_COMPARE_MODELS - 1);
  }

  let options = buildAvailableModelOptions(analysis.model, comparedAnalyses);
  renderModelSelect(selectEl, options);
  buttonEl.disabled = !options.length;
  renderSideBySideContent(analysis, comparedAnalyses);

  buttonEl.addEventListener("click", async () => {
    clearError("side-compare-error");

    const otherModel = selectEl.value;
    if (!otherModel) {
      showError("Selecciona un modelo para comparar.", "side-compare-error");
      return;
    }

    const tasks = inferAnalysisTasks(analysis);
    showLoading(`Ejecutando ${modelShortName(otherModel)} para comparación…`);
    try {
      const compared = await apiAnalyze(analysis.input_text, otherModel, tasks);
      compared.input_text = analysis.input_text;
      compared.tasks = tasks;
      comparedAnalyses.push(compared);
      if (comparedAnalyses.length > (MAX_TOTAL_COMPARE_MODELS - 1)) {
        comparedAnalyses = comparedAnalyses.slice(0, MAX_TOTAL_COMPARE_MODELS - 1);
      }
      saveSideCompare({
        baseAnalysisId: analysis.id || null,
        comparedAnalyses,
      });
      options = buildAvailableModelOptions(analysis.model, comparedAnalyses);
      renderModelSelect(selectEl, options);
      buttonEl.disabled = !options.length;
      renderSideBySideContent(analysis, comparedAnalyses);
    } catch (err) {
      showError(err.message, "side-compare-error");
    } finally {
      hideLoading();
    }
  });
}

/* ------------------------------------------------------------------ */
/* Utilidades                                                           */
/* ------------------------------------------------------------------ */

function escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function highlightText(fullText, highlight) {
  if (!highlight) return escHtml(fullText);
  const escaped = escHtml(fullText);
  const escapedHighlight = escHtml(highlight);
  // Buscar la subcadena (case-insensitive) y envolverla con <mark>
  const idx = escaped.toLowerCase().indexOf(escapedHighlight.toLowerCase());
  if (idx === -1) return escaped;
  return (
    escaped.slice(0, idx) +
    `<mark>${escaped.slice(idx, idx + escapedHighlight.length)}</mark>` +
    escaped.slice(idx + escapedHighlight.length)
  );
}

function exportJson(data, filename) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href     = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

/* ------------------------------------------------------------------ */
/* Auto-init según página                                               */
/* ------------------------------------------------------------------ */

document.addEventListener("DOMContentLoaded", () => {
  const page = window.location.pathname.split("/").pop() || "index.html";

  if (page === "index.html" || page === "")        initInputView();
  else if (page === "segmentation.html")           initSegmentationView();
  else if (page === "contributions.html")          initContributionsView();
  else if (page === "comparison.html")             initComparisonView();
});
