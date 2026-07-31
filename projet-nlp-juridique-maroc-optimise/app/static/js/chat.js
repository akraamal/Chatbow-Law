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

  let history = [];
  let lastSources = [];
  let currentDocView = "original"; // "original" | "analysis"

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
    return `
      <div class="bg-white border border-primary p-3 mb-3" data-source-card="${index}">
        <div class="flex justify-between items-start mb-2">
          <span class="text-[10px] font-bold text-primary bg-primary-fixed px-2 py-0.5">
            EXTRAIT LÉGAL — Source ${index}
          </span>
          <span class="material-symbols-outlined text-primary text-sm cursor-pointer copy-source" data-copy-index="${index}">content_copy</span>
        </div>
        <p class="${textClass} mb-1"${textDir}>${escapeHtml(snippet)}${source.text && source.text.length > 220 ? "…" : ""}</p>
        <p class="text-[10px] text-outline">BO n°${escapeHtml(String(bo))} — article ${escapeHtml(String(source.article_number || "?"))} · pertinence ${(source.score || 0).toFixed(2)}</p>
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
    if (!lastSources.length) return;

    downloadBtn.disabled = false;
    downloadBtn.onclick = () => {
      window.open(`/download/${encodeURIComponent(lastSources[0].doc_id)}`, "_blank");
    };
    renderDocPanel();
  }

  function renderDocPanel() {
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
    addUserMessage(query);
    inputEl.value = "";
    inputEl.style.height = "auto";
    addTypingIndicator();

    if (history.length === 0) {
      subjectEl.textContent = `Sujet : ${query.slice(0, 70)}${query.length > 70 ? "…" : ""}`;
    }

    let data;
    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // Historique plafonné aux 8 derniers tours (coût de la reformulation
        // et surface d'injection) ; lang transmet le toggle FR/AR réellement.
        body: JSON.stringify({ query, history: history.slice(-8), lang: uiLang }),
      });
      data = await response.json();
      if (!response.ok) {
        removeTypingIndicator();
        addErrorMessage(data.error || "Une erreur est survenue.");
        return;
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

  renderDocPanel();
})();
