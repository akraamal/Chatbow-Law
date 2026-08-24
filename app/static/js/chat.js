/**
 * ADLI Morocco — logique du chat.
 * Chaque message passe par /api/chat, qui appelle le vrai chatbot RAG
 * côté serveur (aucune donnée factice).
 */
(function () {
  "use strict";

  const messagesEl = document.getElementById("chat-messages");
  const formEl = document.getElementById("chat-form");
  const inputEl = document.getElementById("chat-input");
  const subjectEl = document.getElementById("chat-subject");
  const docPreviewEl = document.getElementById("doc-preview");
  const docTabOriginal = document.getElementById("doc-tab-original");
  const docTabAnalysis = document.getElementById("doc-tab-ai");
  const downloadBtn = document.getElementById("download-btn");
  const langToggle = document.getElementById("lang-toggle");
  const newAnalysisBtn = document.getElementById("new-analysis-btn");
  const exportBtn = document.getElementById("export-convo-btn");
  const headerSearch = document.getElementById("header-search");
  const attachBtn = document.getElementById("attach-file-btn");
  const attachInput = document.getElementById("attach-file");
  const sendBtn = formEl.querySelector("button[type=submit]");

  let history = [];
  let lastSources = [];
  let currentDocView = "original"; // "original" | "analysis"
  let attachFile = null;      // PDF sélectionné sur le disque (téléchargement)
  let attachTask = null;      // task_id en cours d'analyse
  let docResult = null;       // résultat JSON du document attaché
  let interruptMode = false;  // bouton d'envoi transformé en bouton d'interruption

  // Conversation persistée côté serveur (SQLite) : identifiant courant
  let conversationId = (() => {
    try { return localStorage.getItem("adli_current_conversation"); }
    catch (e) { return null; }
  })();
  const GREETING_HTML = messagesEl.innerHTML;   // bulle d'accueil d'origine
  const DEFAULT_SUBJECT = subjectEl.textContent;

  // Aperçu PDF paginé (« Document Original » en mode document attaché)
  let pdfDoc = null;        // instance PDF.js chargée (cache après premier chargement)
  let pdfPageNum = 1;        // page actuellement affichée
  let pdfRenderTask = null;  // tâche de rendu en cours, annulable

  function ensurePdfLib() {
    return window.pdfjsLib || null;
  }

  function setInterruptMode(on) {
    interruptMode = on;
    const ico = sendBtn.querySelector(".material-symbols-outlined");
    ico.textContent = on ? "stop" : "send";
    sendBtn.classList.toggle("interrupt-mode", on);
    sendBtn.title = on ? "Interrompre l'analyse" : "Envoyer";
  }

  function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function timeNow() {
    return new Date().toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit" });
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  function renderAnswerText(text) {
    const escaped = escapeHtml(text);
    return escaped.replace(/\[Source (\d+)\]/g, '<span class="source-badge" data-source-index="$1">Source $1</span>');
  }

  function addUserMessage(text) {
    const wrapper = document.createElement("div");
    wrapper.className = "flex justify-end";
    wrapper.innerHTML = `
      <div class="max-w-[80%] brutal-border bg-white p-4">
        <p class="font-body-md text-primary">${escapeHtml(text)}</p>
        <span class="block text-[10px] text-right mt-2 text-outline">${timeNow()} • Utilisateur</span>
      </div>`;
    messagesEl.appendChild(wrapper);
    scrollToBottom();
  }

  function addTypingIndicator() {
    const wrapper = document.createElement("div");
    wrapper.className = "flex justify-start";
    wrapper.id = "typing-indicator";
    wrapper.innerHTML = `
      <div class="max-w-[85%] brutal-border bg-[#D9DCD9] p-4 relative">
        <div class="absolute -left-[2px] top-0 bottom-0 w-1 bg-primary"></div>
        <div class="typing-dots"><span></span><span></span><span></span></div>
      </div>`;
    messagesEl.appendChild(wrapper);
    scrollToBottom();
  }

  function removeTypingIndicator() {
    const el = document.getElementById("typing-indicator");
    if (el) el.remove();
  }

  function sourceCardHtml(source, index) {
    const bo = source.bo_number || source.doc_id || "?";
    const isArabic = source.lang === "ar";
    const textDir = isArabic ? ' dir="rtl"' : "";
    const textClass = isArabic ? "font-arabic-body text-right" : "font-body-md";
    const snippet = (source.text || "").slice(0, 220);
    const isInstrument = !!source.instrument_id;
    const badge = isInstrument
      ? `INSTRUMENT JURIDIQUE — Source ${index}`
      : `EXTRAIT LÉGAL — Source ${index}`;
    const meta = isInstrument
      ? `BO n°${escapeHtml(String(bo))} — ${escapeHtml(String(source.type || "Instrument"))}${source.reference ? ` n°${escapeHtml(String(source.reference))}` : ""} · ${escapeHtml(String(source.n_articles || "?"))} article(s) · importance ${escapeHtml(String(source.importance ?? 0))}/100`
      : `BO n°${escapeHtml(String(bo))} — article ${escapeHtml(String(source.article_number || "?"))} · pertinence ${(source.score || 0).toFixed(2)}`;
    return `
      <div class="bg-white border border-primary p-3 mb-3" data-source-card="${index}">
        <div class="flex justify-between items-start mb-2">
          <span class="text-[10px] font-bold text-primary bg-primary-fixed px-2 py-0.5">
            ${badge}
          </span>
          <span class="material-symbols-outlined text-primary text-sm cursor-pointer copy-source" data-copy-index="${index}">content_copy</span>
        </div>
        <p class="${textClass} mb-1"${textDir}>${escapeHtml(snippet)}${source.text && source.text.length > 220 ? "…" : ""}</p>
        <p class="text-[10px] text-outline">${meta}</p>
      </div>`;
  }

  function addBotMessage(answerText, sources) {
    const wrapper = document.createElement("div");
    wrapper.className = "flex justify-start";

    const sourcesHtml = (sources || []).map((s, i) => sourceCardHtml(s, i + 1)).join("");
    const actionsHtml = (sources && sources.length)
      ? `<div class="mt-4 flex flex-wrap gap-2">
           <button type="button" class="px-3 py-1 bg-white border border-primary text-[10px] font-bold text-primary rounded-[7px] hover:bg-primary-fixed transition-colors quick-action" data-quick="Cite la jurisprudence relative à cet article.">VÉRIFIER JURISPRUDENCE</button>
           <button type="button" class="px-3 py-1 bg-white border border-primary text-[10px] font-bold text-primary rounded-[7px] hover:bg-primary-fixed transition-colors" id="download-current-btn">TÉLÉCHARGER LE TEXTE COMPLET</button>
         </div>`
      : "";

    wrapper.innerHTML = `
      <div class="max-w-[85%] brutal-border bg-[#D9DCD9] p-4 relative">
        <div class="absolute -left-[2px] top-0 bottom-0 w-1 bg-primary"></div>
        <p class="font-body-md text-on-surface mb-3">${renderAnswerText(answerText)}</p>
        ${sourcesHtml}
        ${actionsHtml}
        <span class="block text-[10px] mt-2 text-outline">${timeNow()} • ADLI AI</span>
      </div>`;
    messagesEl.appendChild(wrapper);
    scrollToBottom();

    const dlBtn = wrapper.querySelector("#download-current-btn");
    if (dlBtn && sources && sources[0]) {
      dlBtn.addEventListener("click", () => {
        window.open(`/download/${encodeURIComponent(sources[0].doc_id)}`, "_blank");
      });
    }
    wrapper.querySelectorAll(".quick-action").forEach((btn) => {
      btn.addEventListener("click", () => {
        inputEl.value = btn.dataset.quick;
        inputEl.focus();
      });
    });
  }

  function addErrorMessage(message) {
    const wrapper = document.createElement("div");
    wrapper.className = "flex justify-start";
    wrapper.innerHTML = `
      <div class="max-w-[85%] brutal-border bg-white p-4 relative" style="border-color:#93000a">
        <p class="font-body-md" style="color:#93000a">${escapeHtml(message)}</p>
      </div>`;
    messagesEl.appendChild(wrapper);
    scrollToBottom();
  }

  function updateAnalysisPanel(sources) {
    lastSources = sources || [];
    if (docResult) {
      renderDocPanel();
      return;
    }
    if (!lastSources.length) return;

    downloadBtn.disabled = false;
    downloadBtn.onclick = () => {
      window.open(`/download/${encodeURIComponent(lastSources[0].doc_id)}`, "_blank");
    };
    renderDocPanel();
  }

  function renderDocPanel() {
    if (attachFile && !docResult) {
      // Upload en cours (pipeline en arrière-plan) : montrer le PDF si
      // l'onglet « original » est actif, plutôt que rien.
      if (currentDocView === "original") {
        renderAttachedOriginalPanel();
        return;
      }
      docPreviewEl.innerHTML =
        '<p class="text-xs text-outline text-center mt-12">Analyse en cours…</p>';
      return;
    }
    if (docResult) {
      if (currentDocView === "original") {
        renderAttachedOriginalPanel();
      } else {
        renderAttachedDocPanel();
      }
      return;
    }
    if (!lastSources.length) {
      docPreviewEl.innerHTML = `
        <p class="text-xs text-outline text-center mt-12">
          Pose une question pour afficher ici l'extrait du texte officiel correspondant.
        </p>`;
      return;
    }

    if (currentDocView === "original") {
      const top = lastSources[0];
      const isArabic = top.lang === "ar";
      docPreviewEl.innerHTML = `
        <div class="border-b-2 border-primary/30 pb-4 mb-6 text-center">
          <div class="w-12 h-12 mx-auto mb-2 opacity-20">
            <span class="material-symbols-outlined text-4xl">account_balance</span>
          </div>
          <h4 class="font-arabic-body text-lg font-bold" dir="rtl">الجريدة الرسمية</h4>
          <p class="text-[8px] uppercase tracking-widest mt-1">Royaume du Maroc — BO n°${escapeHtml(String(top.bo_number || "?"))}</p>
        </div>
        <div class="${isArabic ? 'font-arabic-body text-right' : 'font-body-md'} text-sm whitespace-pre-wrap"${isArabic ? ' dir="rtl"' : ""}>
          ${escapeHtml(top.text || "")}
        </div>
        <div class="text-center mt-6">
          <p class="text-[8px] text-outline">Article ${escapeHtml(String(top.article_number || "?"))}</p>
        </div>`;
    } else {
      docPreviewEl.innerHTML = lastSources.map((s, i) => `
        <div class="mb-4 pb-4 border-b border-outline-variant">
          <p class="text-[10px] font-bold text-primary">SOURCE ${i + 1} — pertinence ${(s.score || 0).toFixed(3)}</p>
          <p class="text-xs text-outline">BO n°${escapeHtml(String(s.bo_number || "?"))} · article ${escapeHtml(String(s.article_number || "?"))} · ${s.lang === "ar" ? "arabe" : "français"}</p>
        </div>`).join("");
    }
  }

  docTabOriginal.addEventListener("click", () => {
    currentDocView = "original";
    docTabOriginal.classList.add("bg-primary", "text-white");
    docTabOriginal.classList.remove("text-primary");
    docTabAnalysis.classList.remove("bg-primary", "text-white");
    docTabAnalysis.classList.add("text-primary");
    renderDocPanel();
  });
  docTabAnalysis.addEventListener("click", () => {
    currentDocView = "analysis";
    docTabAnalysis.classList.add("bg-primary", "text-white");
    docTabAnalysis.classList.remove("text-primary");
    docTabOriginal.classList.remove("bg-primary", "text-white");
    docTabOriginal.classList.add("text-primary");
    renderDocPanel();
  });

  async function sendQuery(query) {
    if (attachTask) {
      inputEl.value = "";
      addErrorMessage("Attends la fin de l'analyse du document en cours avant de poser une question.");
      return;
    }
    addUserMessage(query);
    inputEl.value = "";
    inputEl.style.height = "auto";
    addTypingIndicator();

    if (docResult && history.length === 0) {
      subjectEl.textContent = `Document : ${attachFile ? attachFile.name : "document analysé"}`;
    } else if (!docResult && history.length === 0) {
      subjectEl.textContent = `Sujet : ${query.slice(0, 70)}${query.length > 70 ? "…" : ""}`;
    }

    let data;
    try {
      const response = await fetch(docResult ? "/chat" : "/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // En mode document attaché, /chat gère son propre historique serveur.
        // Sinon : conversation_id suffit — l'historique de reformulation est
        // reconstruit côté serveur depuis SQLite (client plus source de vérité).
        body: JSON.stringify(docResult
          ? { question: query, doc_id: docResult.doc_id }
          : { query, conversation_id: conversationId, lang: uiLang }),
      });
      data = await response.json();
      if (!response.ok) {
        removeTypingIndicator();
        addErrorMessage(data.error || "Une erreur est survenue.");
        return;
      }
      if (!docResult && data.conversation_id) {
        conversationId = data.conversation_id;
        try { localStorage.setItem("adli_current_conversation", conversationId); }
        catch (e) { /* stockage indisponible : la persistance ne survira juste pas */ }
      }
    } catch (err) {
      removeTypingIndicator();
      addErrorMessage("Impossible de contacter le serveur. Vérifie ta connexion et réessaie.");
      return;
    }

    removeTypingIndicator();
    addBotMessage(data.answer, data.sources);
    updateAnalysisPanel(data.sources);
    history.push({ question: query, answer: data.answer });
    if (history.length > 8) history.splice(0, history.length - 8);
  }

  formEl.addEventListener("submit", (e) => {
    e.preventDefault();
    if (interruptMode) {
      cancelAttach();
      return;
    }
    const query = inputEl.value.trim();
    if (!query) return;
    sendQuery(query);
  });

  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      formEl.requestSubmit();
    }
  });

  inputEl.addEventListener("input", function () {
    this.style.height = "auto";
    this.style.height = this.scrollHeight + "px";
  });

  document.querySelectorAll(".quick-suggestion").forEach((btn) => {
    btn.addEventListener("click", () => {
      inputEl.value = btn.dataset.suggestion;
      inputEl.focus();
      inputEl.dispatchEvent(new Event("input"));
    });
  });

  function exportConversation() {
    const parts = ["# Conversation ADLI MOROCCO", ""];
    document.querySelectorAll("#chat-messages > .flex").forEach((wrapper) => {
      const bubble = wrapper.querySelector(".brutal-border");
      if (!bubble) return;
      const role = wrapper.classList.contains("justify-end")
        ? "**Utilisateur**"
        : "**ADLI AI**";
      const text = bubble.textContent.replace(/\s+/g, " ").trim();
      if (!text) return;
      parts.push(`${role} : ${text}`, "");
    });
    if (parts.length <= 2) {
      addErrorMessage("Aucune conversation à exporter pour le moment.");
      return;
    }
    const blob = new Blob([parts.join("\n")], { type: "text/markdown;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `conversation-adli-${new Date().toISOString().slice(0, 10)}.md`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  if (newAnalysisBtn) {
    newAnalysisBtn.addEventListener("click", () => {
      window.location.href = "/analyzer";
    });
  }

  if (exportBtn) {
    exportBtn.addEventListener("click", exportConversation);
  }

  if (headerSearch) {
    headerSearch.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        const q = headerSearch.value.trim();
        if (!q) return;
        headerSearch.value = "";
        inputEl.value = q;
        sendQuery(q);
      }
    });
  }

  // ── Pièce jointe : analyse d'un PDF du Bulletin Officiel ─────────────

  // Aperçu PDF paginé (onglet « Document Original », mode document attaché)
  async function ensurePdfLoaded() {
    const lib = ensurePdfLib();
    if (!lib) throw new Error("PDF.js indisponible");
    if (pdfDoc || !attachFile) return;
    const arrayBuffer = await attachFile.arrayBuffer();
    pdfDoc = await lib.getDocument({ data: arrayBuffer }).promise;
    pdfPageNum = 1;
  }

  async function _renderPdfToCanvas(num, canvasId, indicatorId, prevBtnId,
                                     nextBtnId, computeScale) {
    if (!pdfDoc) return;
    if (pdfRenderTask) {
      pdfRenderTask.cancel(); // pas de rendus concurrents sur le même canvas
    }
    const page = await pdfDoc.getPage(num);
    const canvas = document.getElementById(canvasId);
    if (!canvas) return; // l'utilisateur a changé de vue entre-temps

    const ctx = canvas.getContext("2d");
    const unscaledViewport = page.getViewport({ scale: 1 });
    const scale = computeScale(unscaledViewport);
    const viewport = page.getViewport({ scale });
    canvas.width = viewport.width;
    canvas.height = viewport.height;

    pdfRenderTask = page.render({ canvasContext: ctx, viewport });
    try {
      await pdfRenderTask.promise;
    } catch (err) {
      if (err && err.name !== "RenderingCancelledException") throw err;
      return; // rendu annulé volontairement
    }

    const indicator = document.getElementById(indicatorId);
    if (indicator) indicator.textContent = "Page " + num + " / " + pdfDoc.numPages;
    const prevBtn = document.getElementById(prevBtnId);
    if (prevBtn) prevBtn.disabled = num <= 1;
    const nextBtn = document.getElementById(nextBtnId);
    if (nextBtn) nextBtn.disabled = num >= pdfDoc.numPages;
  }

  function renderPdfPage(num) {
    const containerWidth = docPreviewEl.parentElement.clientWidth - 32;
    return _renderPdfToCanvas(
      num, "pdf-canvas", "pdf-page-indicator", "pdf-prev-btn", "pdf-next-btn",
      (vp) => Math.min(containerWidth / vp.width, 2),
    );
  }

  function renderPdfPageFullscreen(num) {
    const maxW = window.innerWidth * 0.9;
    const maxH = (window.innerHeight - 140) * 0.95; // place pour croix + contrôles
    return _renderPdfToCanvas(
      num, "pdf-canvas-fullscreen", "pdf-fs-page-indicator",
      "pdf-fs-prev-btn", "pdf-fs-next-btn",
      (vp) => Math.min(maxW / vp.width, maxH / vp.height, 4),
    );
  }

  // ── Modale plein écran (position:fixed custom, pas la Fullscreen API) ──
  let pdfFullscreenOpen = false;
  let _pdfKeyHandler = null;
  let _pdfResizeHandler = null;

  function _pdfBackdropClickHandler(e) {
    if (e.target.id === "pdf-fullscreen-modal") closePdfFullscreen();
  }

  function openPdfFullscreen() {
    if (!pdfDoc) return; // PDF pas encore chargé — bouton normalement non cliquable avant
    const modal = document.getElementById("pdf-fullscreen-modal");
    modal.style.display = "flex";
    pdfFullscreenOpen = true;
    document.body.style.overflow = "hidden"; // bloque le scroll derrière la modale

    document.getElementById("pdf-fs-prev-btn").onclick = () => {
      if (pdfPageNum > 1) { pdfPageNum -= 1; renderPdfPageFullscreen(pdfPageNum); }
    };
    document.getElementById("pdf-fs-next-btn").onclick = () => {
      if (pdfDoc && pdfPageNum < pdfDoc.numPages) { pdfPageNum += 1; renderPdfPageFullscreen(pdfPageNum); }
    };
    document.getElementById("pdf-fullscreen-close").onclick = closePdfFullscreen;

    _pdfKeyHandler = (e) => {
      if (e.key === "Escape") {
        closePdfFullscreen();
      } else if (e.key === "ArrowRight") {
        if (pdfDoc && pdfPageNum < pdfDoc.numPages) { pdfPageNum += 1; renderPdfPageFullscreen(pdfPageNum); }
      } else if (e.key === "ArrowLeft") {
        if (pdfPageNum > 1) { pdfPageNum -= 1; renderPdfPageFullscreen(pdfPageNum); }
      }
    };
    document.addEventListener("keydown", _pdfKeyHandler);
    modal.addEventListener("click", _pdfBackdropClickHandler);

    // Re-rendu à la bonne échelle au redimensionnement (débouncé)
    let resizeTimer = null;
    _pdfResizeHandler = () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => renderPdfPageFullscreen(pdfPageNum), 150);
    };
    window.addEventListener("resize", _pdfResizeHandler);

    document.getElementById("pdf-fullscreen-close").focus();
    renderPdfPageFullscreen(pdfPageNum);
  }

  function closePdfFullscreen() {
    const modal = document.getElementById("pdf-fullscreen-modal");
    modal.style.display = "none";
    pdfFullscreenOpen = false;
    document.body.style.overflow = "";

    // Toujours retirer les écouteurs — sinon ils s'accumulent à chaque
    // ouverture et la navigation clavier sauterait plusieurs pages d'un coup.
    if (_pdfKeyHandler) {
      document.removeEventListener("keydown", _pdfKeyHandler);
      _pdfKeyHandler = null;
    }
    modal.removeEventListener("click", _pdfBackdropClickHandler);
    if (_pdfResizeHandler) {
      window.removeEventListener("resize", _pdfResizeHandler);
      _pdfResizeHandler = null;
    }

    // Le mini-aperçu reflète la page consultée en plein écran.
    const expandBtn = document.getElementById("pdf-expand-btn");
    if (expandBtn) expandBtn.focus();
    if (document.getElementById("pdf-canvas")) {
      renderPdfPage(pdfPageNum);
    }
  }

  async function renderAttachedOriginalPanel() {
    docPreviewEl.innerHTML =
      '<p class="text-xs text-outline text-center mt-12">Chargement de l\'aperçu…</p>';

    try {
      await ensurePdfLoaded();
    } catch (err) {
      docPreviewEl.innerHTML =
        '<p class="text-xs text-center" style="display:block;background:#ffdad6;border:2px solid #ba1a1a;color:#93000a;padding:10px 14px">Impossible de charger l\'aperçu du PDF.</p>';
      return;
    }

    docPreviewEl.innerHTML =
      '<div class="flex flex-col items-center gap-3">' +
        '<canvas id="pdf-canvas" class="shadow max-w-full"></canvas>' +
        '<div class="flex items-center gap-3">' +
          '<button id="pdf-prev-btn" class="px-3 py-1 text-sm font-bold text-primary border border-primary/30 rounded-md hover:bg-primary-fixed disabled:opacity-40" title="Page précédente">&#x2039;</button>' +
          '<span id="pdf-page-indicator" class="text-xs text-outline"></span>' +
          '<button id="pdf-next-btn" class="px-3 py-1 text-sm font-bold text-primary border border-primary/30 rounded-md hover:bg-primary-fixed disabled:opacity-40" title="Page suivante">&#x203A;</button>' +
          '<button id="pdf-expand-btn" class="px-2 py-1 text-primary border border-primary/30 rounded-md hover:bg-primary-fixed" title="Plein écran">' +
            '<span class="material-symbols-outlined text-sm">open_in_full</span></button>' +
        '</div>' +
      '</div>';

    document.getElementById("pdf-prev-btn").addEventListener("click", () => {
      if (pdfPageNum > 1) { pdfPageNum -= 1; renderPdfPage(pdfPageNum); }
    });
    document.getElementById("pdf-next-btn").addEventListener("click", () => {
      if (pdfDoc && pdfPageNum < pdfDoc.numPages) { pdfPageNum += 1; renderPdfPage(pdfPageNum); }
    });
    document.getElementById("pdf-expand-btn").addEventListener("click", openPdfFullscreen);

    try {
      await renderPdfPage(pdfPageNum);
    } catch (err) {
      docPreviewEl.innerHTML =
        '<p class="text-xs text-center" style="display:block;background:#ffdad6;border:2px solid #ba1a1a;color:#93000a;padding:10px 14px">Impossible d\'afficher cette page.</p>';
    }
  }

  function renderAttachedDocPanel() {
    const d = docResult;
    const allArts = (d.instruments || []).flatMap(i => i.articles || []);
    const artsHtml = allArts.slice(0, 3).map((a) => `
      <div class="mb-3">
        <p class="text-[10px] font-bold text-primary mb-1">Article ${escapeHtml(a.number || "?")}</p>
        <p class="font-body-md text-xs whitespace-pre-wrap">${escapeHtml((a.text || "").slice(0, 220))}${(a.text || "").length > 220 ? "…" : ""}</p>
      </div>`).join("");
    docPreviewEl.innerHTML = `
      <div class="border-b-2 border-primary/30 pb-4 mb-4 text-center">
        <h4 class="font-arabic-body text-lg font-bold" dir="rtl">الجريدة الرسمية</h4>
        <p class="text-[8px] uppercase tracking-widest mt-1">ROYAUME DU MAROC — BO n°${escapeHtml(String(d.bo_number || "?"))} · ${escapeHtml(d.date_publication || "")}</p>
      </div>
      <div class="flex gap-2 mb-4">
        <span class="text-[10px] font-bold text-primary bg-primary-fixed px-2 py-1">${d.n_instruments} instruments</span>
        <span class="text-[10px] font-bold text-primary bg-primary-fixed px-2 py-1">${d.n_articles} articles</span>
      </div>
      <p class="text-[9px] text-outline uppercase tracking-widest mb-2">Extraits</p>
      ${artsHtml}`;
  }

  attachBtn.addEventListener("click", () => {
    attachInput.click();
  });

  attachInput.addEventListener("change", () => {
    const file = attachInput.files[0];
    attachInput.value = "";
    if (!file) return;
    if (file.size > 50 * 1024 * 1024) {
      addErrorMessage("Le fichier dépasse la limite de 50 Mo.");
      return;
    }
    attachDocument(file);
  });

  function setDocStatus(text) {
    const el = document.getElementById("doc-status");
    if (el) el.querySelector("p").textContent = text;
  }

  function addDocStatus(text) {
    const wrapper = document.createElement("div");
    wrapper.className = "flex justify-start";
    wrapper.id = "doc-status";
    wrapper.innerHTML = `
      <div class="max-w-[85%] brutal-border bg-white p-4 relative">
        <p class="font-body-md text-on-surface-variant">${escapeHtml(text)}</p>
      </div>`;
    messagesEl.appendChild(wrapper);
    scrollToBottom();
  }

  async function cancelAttach() {
    if (!attachTask || attachTask === "pending") return;
    const tid = attachTask;
    addDocStatus("⚠ Interruption demandée — arrêt de l'analyse…");
    try {
      await fetch(`/cancel/${tid}`, { method: "POST" });
    } catch (err) {
      setInterruptMode(false);
      addErrorMessage("Erreur lors de l'interruption.");
    }
  }

  async function attachDocument(file) {
    if (attachTask || docResult) {
      addErrorMessage("Un document est déjà en cours d'analyse. Clique sur « Nouvelle analyse » pour en joindre un autre.");
      return;
    }
    if (!/\.pdf$/i.test(file.name)) {
      addErrorMessage("Seuls les fichiers PDF du Bulletin Officiel sont acceptés.");
      return;
    }
    attachFile = file;
    attachTask = "pending";
    addUserMessage(`📎 ${file.name}`);
    addTypingIndicator();

    const form = new FormData();
    form.append("file", file);
    try {
      const resp = await fetch("/upload", { method: "POST", body: form });
      const data = await resp.json();
      if (!resp.ok || data.error) {
        throw new Error(data.error || "Échec de l'envoi du fichier.");
      }
      removeTypingIndicator();
      addDocStatus(`⏳ Analyse de « ${file.name} » en cours…`);
      attachTask = data.task_id;
      setInterruptMode(true);
      watchTaskProgress(data.task_id, file);
    } catch (err) {
      removeTypingIndicator();
      addErrorMessage(`Impossible de lancer l'analyse : ${err.message || err}`);
      attachFile = null;
      attachTask = null;
      setInterruptMode(false);
    }
  }

  function watchTaskProgress(tid, file) {
    const es = new EventSource(`/stream/${tid}`);

    es.onmessage = (e) => {
      setDocStatus(`⏳ Analyse de « ${file.name} » en cours… ${e.data}`);
    };

    es.addEventListener("done", async () => {
      es.close();
      try {
        const resp = await fetch(`/result/${tid}`);
        const data = await resp.json();
        if (!resp.ok || data.error) {
          setDocStatus(`❌ ${data.error || "Analyse échouée."}`);
          attachFile = null;
          attachTask = null;
          setInterruptMode(false);
          return;
        }
        docResult = data;
        attachTask = null;
        setInterruptMode(false);
        setDocStatus(`✅ « ${file.name} » analysé${data.bo_number ? ` — BO n° ${data.bo_number}` : ""}. Pose une question sur ce document.`);
        subjectEl.textContent = `Document : ${file.name}`;
        downloadBtn.disabled = false;
        downloadBtn.onclick = () => {
          const a = document.createElement("a");
          a.href = URL.createObjectURL(file);
          a.download = file.name || "document.pdf";
          a.click();
          URL.revokeObjectURL(a.href);
        };
        renderDocPanel();
      } catch (err) {
        setDocStatus("❌ Erreur lors de la récupération des résultats.");
        attachFile = null;
        attachTask = null;
        setInterruptMode(false);
      }
    });

    es.addEventListener("error", (e) => {
      es.close();
      if (e.data) {
        setDocStatus(`❌ ${e.data}`);
      } else {
        setDocStatus("❌ Connexion au pipeline perdue.");
      }
      attachFile = null;
      attachTask = null;
      setInterruptMode(false);
    });
  }

  const I18N = {
    fr: {
      newAnalysis: "Nouvelle analyse", placeholder: "Posez votre question juridique...",
      exporter: "Exporter", langToggle: "ARABIC",
    },
    ar: {
      newAnalysis: "تحليل جديد", placeholder: "اطرح سؤالك القانوني...",
      exporter: "تصدير", langToggle: "FRANÇAIS",
    },
  };
  let uiLang = "fr";
  langToggle.addEventListener("click", () => {
    uiLang = uiLang === "fr" ? "ar" : "fr";
    document.documentElement.dir = uiLang === "ar" ? "rtl" : "ltr";
    document.querySelectorAll("[data-i18n]").forEach((el) => {
      const key = el.dataset.i18n;
      if (I18N[uiLang][key]) el.textContent = I18N[uiLang][key];
    });
    if (I18N[uiLang].placeholder) inputEl.placeholder = I18N[uiLang].placeholder;
  });

  // ── Chat historique : conversations persistées (SQLite côté serveur) ──

  const histPanelEl = document.getElementById("chat-history-panel");
  const histListEl = document.getElementById("chat-history-list");
  const histNavBtn = document.getElementById("chat-history-nav-btn");
  const histCloseBtn = document.getElementById("chat-history-close");
  const newConvoBtn = document.getElementById("new-conversation-btn");

  function fmtRelative(ts) {
    const s = Math.max(0, Math.floor(Date.now() / 1000 - ts));
    if (s < 60) return "à l'instant";
    if (s < 3600) return "il y a " + Math.floor(s / 60) + " min";
    if (s < 86400) return "il y a " + Math.floor(s / 3600) + " h";
    return "il y a " + Math.floor(s / 86400) + " j";
  }

  function openChatHistory() {
    histPanelEl.style.display = "flex";
    loadChatHistoryList();
  }

  function closeChatHistory() {
    histPanelEl.style.display = "none";
  }

  async function loadChatHistoryList() {
    histListEl.innerHTML =
      '<p class="text-xs text-outline p-2">Chargement…</p>';
    try {
      const resp = await fetch("/api/chat/conversations");
      const data = await resp.json();
      const convos = data.conversations || [];
      if (!convos.length) {
        histListEl.innerHTML =
          '<p class="text-xs text-outline p-2">Aucune conversation enregistrée.</p>';
        return;
      }
      histListEl.innerHTML = convos.map((c) => `
        <div class="border border-primary p-3 mb-2 bg-white cursor-pointer hover:bg-primary-fixed" data-conv="${escapeHtml(c.id)}">
          <div class="flex justify-between items-start gap-2">
            <div style="min-width:0">
              <p class="text-xs font-bold text-primary truncate">${escapeHtml(c.title || "Sans titre")}</p>
              <p class="text-[10px] text-outline">${fmtRelative(c.updated_at)}</p>
            </div>
            <span class="material-symbols-outlined text-outline text-sm cursor-pointer delete-conv" data-id="${escapeHtml(c.id)}" title="Supprimer">delete</span>
          </div>
        </div>`).join("");
      histListEl.querySelectorAll("[data-conv]").forEach((el) => {
        el.addEventListener("click", (e) => {
          if (e.target.closest(".delete-conv")) return;
          selectConversation(el.dataset.conv);
        });
      });
      histListEl.querySelectorAll(".delete-conv").forEach((el) => {
        el.addEventListener("click", async (e) => {
          e.stopPropagation();
          const id = el.dataset.id;
          if (!confirm("Supprimer cette conversation ?")) return;
          await fetch(`/api/chat/conversations/${encodeURIComponent(id)}`,
            { method: "DELETE" });
          if (conversationId === id) startNewConversation();
          loadChatHistoryList();
        });
      });
    } catch (err) {
      histListEl.innerHTML =
        '<p class="text-xs text-outline p-2">Impossible de charger l\'historique.</p>';
    }
  }

  function selectConversation(id) {
    conversationId = id;
    try { localStorage.setItem("adli_current_conversation", id); } catch (e) { /* non bloquant */ }
    closeChatHistory();
    loadConversationMessages(id);
  }

  function startNewConversation() {
    conversationId = null;
    try { localStorage.removeItem("adli_current_conversation"); } catch (e) { /* non bloquant */ }
    history = [];
    lastSources = [];
    messagesEl.innerHTML = GREETING_HTML;
    subjectEl.textContent = DEFAULT_SUBJECT;
    scrollToBottom();
  }

  async function loadConversationMessages(convId) {
    if (!convId) return;
    // Verrouille la saisie PENDANT le fetch : sans ça, un message envoyé
    // avant la résolution serait effacé de l'écran par le wipe ci-dessous
    // (et history.length===0 ferait écraser le sujet à tort).
    inputEl.disabled = true;
    sendBtn.disabled = true;
    const placeholder = inputEl.placeholder;
    inputEl.placeholder = "Chargement de la conversation…";
    try {
      const resp = await fetch(
        "/api/chat/history/" + encodeURIComponent(convId));
      const data = await resp.json();
      const msgs = data.messages || [];
      if (!msgs.length) return;   // identifiant inconnu : page vierge, sans erreur
      messagesEl.innerHTML = GREETING_HTML;
      let pendingQ = null;
      for (const m of msgs) {
        if (m.role === "user") {
          addUserMessage(m.content);
          pendingQ = m.content;
        } else {
          addBotMessage(m.content, m.sources);   // sources re-rendues en cartes
          if (pendingQ !== null) {
            history.push({ question: pendingQ, answer: m.content });
            pendingQ = null;
          }
        }
      }
      if (history.length > 8) history.splice(0, history.length - 8);
      const firstUser = msgs.find((m) => m.role === "user");
      if (firstUser) {
        subjectEl.textContent = "Sujet : " + firstUser.content.slice(0, 70)
          + (firstUser.content.length > 70 ? "…" : "");
      }
      scrollToBottom();
    } catch (err) {
      console.warn("Historique de chat indisponible :", err);
    } finally {
      // Toujours réactiver, même sur fetch en échec ou réponse non-OK.
      inputEl.disabled = false;
      sendBtn.disabled = false;
      inputEl.placeholder = placeholder;
    }
  }

  if (histNavBtn) {
    histNavBtn.addEventListener("click", openChatHistory);
    // Accessibilité clavier (élément sans href : role="button" tabindex="0")
    histNavBtn.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        openChatHistory();
      }
    });
  }
  if (histCloseBtn) histCloseBtn.addEventListener("click", closeChatHistory);
  if (newConvoBtn) {
    newConvoBtn.addEventListener("click", startNewConversation);
  }

  // Réhydratation au chargement : la conversation courante réapparaît.
  loadConversationMessages(conversationId);

  renderDocPanel();
})();
