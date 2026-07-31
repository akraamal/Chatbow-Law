"""
app.py
-------
Interface Streamlit pour le chatbot RAG juridique marocain.
Design system exact depuis le projet Stitch SGG Legal Chatbot UI.
Usage : streamlit run app.py
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

# Attempt to import fitz for PDF processing, with a graceful fallback warning
try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).resolve().parent))

# API key — MUST be set via environment variable for security
if not os.environ.get("GROQ_API_KEY"):
    import warnings
    warnings.warn(
        "GROQ_API_KEY non définie. Le chatbot RAG ne fonctionnera pas. "
        "Définis-la via : set GROQ_API_KEY=votre_cle (Windows) "
        "ou export GROQ_API_KEY=votre_cle (Linux/Mac). "
        "Clé gratuite sur https://console.groq.com/keys"
    )

# Fallback imports in case the src modules are not fully set up yet
try:
    from src.rag.chatbot import LegalRAGChatbot
    from src.rag.prompt_builder import SYSTEM_INSTRUCTION, build_user_prompt
    from src.rag.llm_client import LLMClient
    MODULES_LOADED = True
except ImportError:
    MODULES_LOADED = False
    SYSTEM_INSTRUCTION = "Vous êtes un assistant juridique."


# --- 1. PAGE CONFIGURATION & COLORS ---
st.set_page_config(
    page_title="ADLI MOROCCO — Portail Légal",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

P = {
    "primary": "#2c537f",
    "on-primary": "#ffffff",
    "primary-container": "#466b99",
    "on-primary-container": "#dfeaff",
    "teal-accent": "#178AB4",
    "surface-container-lowest": "#ffffff",
    "surface-container-low": "#e8f6ff",
    "on-surface": "#001e2b",
    "on-surface-variant": "#43474f",
    "panel-header": "#D9DCD9",
    "background": "#f4faff",
    "outline-variant": "#c3c6d0",
    "tertiary": "#005774",
}

# --- 2. SESSION STATE MANAGEMENT ---
default_states = {
    "history": [],
    "bot": None,
    "bot_error": None,
    "lang_filter": None,
    "query_used": "",
    "uploaded_text": None,
    "uploaded_name": None,
    "use_arabic": False,
    "uploaded_docs": [],
}

for key, default in default_states.items():
    if key not in st.session_state:
        st.session_state[key] = default

def _arabic(fr: str, ar: str) -> str:
    return ar if st.session_state.use_arabic else fr

# --- 3. UI CLEANUP & CSS ---
st.markdown(
    f"""
    <style>
        .stApp {{ background-color: {P["background"]}; color: {P["on-surface"]}; }}
        section[data-testid="stSidebar"] {{
            background-color: {P["surface-container-low"]} !important;
            border-right: 2px solid {P["primary"]} !important;
        }}
        .user-msg {{
            padding: 1rem; border: 2px solid {P["primary"]}; margin-bottom: 1rem;
            max-width: 80%; margin-left: auto; text-align: right;
        }}
        .bot-msg {{
            padding: 1rem; background-color: {P["panel-header"]};
            border-left: 4px solid {P["primary"]}; margin-bottom: 1rem; max-width: 85%;
        }}
        .citation-card {{ background: #fff; border: 1px solid {P["primary"]}; margin-top: 0.5rem; padding: 0.5rem; font-size: 0.85rem; }}
        hr {{ border-color: {P["outline-variant"]} !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- 4. HELPER FUNCTIONS ---
def _init_bot():
    if st.session_state.bot is not None:
        return st.session_state.bot
    if not MODULES_LOADED:
        st.session_state.bot_error = "Modules src.rag introuvables."
        return None
    try:
        st.session_state.bot = LegalRAGChatbot()
        st.session_state.bot_error = None
    except Exception as exc:
        st.session_state.bot_error = str(exc)
    return st.session_state.bot

def process_uploaded_file(file):
    """Handles text extraction for PDF and TXT files securely."""
    try:
        if file.name.endswith(".txt"):
            return file.read().decode("utf-8")
        elif file.name.endswith(".pdf"):
            if not FITZ_AVAILABLE:
                return "Erreur: La bibliothèque PyMuPDF (fitz) n'est pas installée."
            with fitz.open(stream=file.read(), filetype="pdf") as doc:
                text = "\n".join(str(page.get_text()) for page in doc)
            return text
        return ""
    except Exception as e:
        return f"Erreur de lecture: {e}"


# --- 5. SIDEBAR (UPLOAD & SETTINGS) ---
with st.sidebar:
    st.markdown(f"### ⚖️ {_arabic('Portail Légal', 'البوابة القانونية')}")
    st.markdown("---")
    
    if st.button(_arabic("＋ Nouvelle Analyse", "＋ تحليل جديد"), use_container_width=True, type="primary"):
        st.session_state.history = []
        st.session_state.query_used = ""
        st.rerun()

    st.session_state.use_arabic = st.toggle(
        _arabic("Passer en Arabe", "التبديل إلى الفرنسية"),
        value=st.session_state.use_arabic,
    )
    
    st.markdown("### 📄 " + _arabic("Gestion Documentaire", "إدارة المستندات"))
    
    # Upload File Function
    uploaded_file = st.file_uploader(
        _arabic("Déposer un document (PDF/TXT)", "تحميل مستند (PDF/TXT)"),
        type=["pdf", "txt"],
    )
    
    if uploaded_file is not None:
        exists = any(d["name"] == uploaded_file.name for d in st.session_state.uploaded_docs)
        if not exists:
            extracted_text = process_uploaded_file(uploaded_file)
            st.session_state.uploaded_docs.append({
                "name": uploaded_file.name,
                "text": extracted_text,
            })
            st.session_state.uploaded_text = extracted_text
            st.session_state.uploaded_name = uploaded_file.name
            st.success(_arabic("Document ajouté avec succès!", "تمت إضافة المستند بنجاح!"))
            st.rerun()

    if st.session_state.uploaded_docs:
        for i, doc in enumerate(st.session_state.uploaded_docs):
            active = doc["name"] == st.session_state.uploaded_name
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"{'🔵' if active else '⚪'} {doc['name'][:20]}")
            with col2:
                if not active and st.button("Utiliser", key=f"use_{i}"):
                    st.session_state.uploaded_text = doc["text"]
                    st.session_state.uploaded_name = doc["name"]
                    st.rerun()
                elif active and st.button("✕", key=f"del_{i}"):
                    st.session_state.uploaded_docs.pop(i)
                    st.session_state.uploaded_text = None
                    st.session_state.uploaded_name = None
                    st.rerun()

# --- 6. MAIN CHAT INTERFACE ---
chat_col, meta_col = st.columns([2.5, 1.5])

with chat_col:
    st.subheader(_arabic("Conversation Légale", "محادثة قانونية"))
    
    # Display Chat History
    for turn in st.session_state.history:
        st.markdown(f"<div class='user-msg'>{turn['question']}</div>", unsafe_allow_html=True)
        
        sources_html = ""
        if turn.get("sources"):
            for src in turn["sources"]:
                sources_html += f"""
                <div class='citation-card'>
                    <strong>{src.get('doc_id', '?')}</strong> - Art. {src.get('article_number', '?')}
                    <br><small>{src.get('text', '')[:150]}...</small>
                </div>"""
                
        st.markdown(f"<div class='bot-msg'>{turn['answer']}{sources_html}</div>", unsafe_allow_html=True)

    # User Input Handling
    user_query = st.chat_input(_arabic("Posez votre question juridique...", "اطرح سؤالك القانوني..."))
    
    if user_query:
        query = user_query.strip()
        has_doc = bool(st.session_state.uploaded_text)
        
        # Document Context Route
        if has_doc and MODULES_LOADED:
            try:
                llm = LLMClient()
                doc_context = st.session_state.uploaded_text or ""
                context_excerpt = (doc_context[:80000] if isinstance(doc_context, str) else str(doc_context))

                history_block = ""
                if st.session_state.history:
                    turns = []
                    for t in st.session_state.history[-4:]:
                        turns.append(f"Question : {t['question']}\nRéponse : {t['answer']}")
                    history_block = "Historique de la conversation (du plus ancien au plus récent) :\n\n" + "\n\n".join(turns) + "\n\n"

                user_prompt = (
                    f"Contexte du document ({st.session_state.uploaded_name}) :\n\n"
                    f"{context_excerpt}\n\n"
                    f"{history_block}"
                    f"Nouvelle question : {query}\n\n"
                    f"Rappel : réponds uniquement à partir du texte ci-dessus. Cite le numéro de décret exact et la date textuelle entre parenthèses."
                )
                answer_text = llm.generate(SYSTEM_INSTRUCTION, user_prompt)
                
                result = {
                    "answer": answer_text,
                    "sources": [{"doc_id": st.session_state.uploaded_name, "article_number": "—", "text": "Extrait du document local", "score": 1.0}]
                }
            except Exception as exc:
                result = {"answer": f"Erreur: {exc}", "sources": []}
                
        # RAG Route
        elif not has_doc and MODULES_LOADED and (bot := _init_bot()):
            try:
                result = bot.answer(query=query, history=st.session_state.history, top_k=5, lang=st.session_state.lang_filter)
            except Exception as exc:
                result = {"answer": f"Erreur RAG: {exc}", "sources": []}
        
        # Fallback / Demo Route
        else:
            msg = st.session_state.bot_error or "Mode Démo - Veuillez charger un document ou connecter la base RAG."
            result = {"answer": f"**[Mode Démo]** {msg}", "sources": []}

        st.session_state.history.append({
            "question": query,
            "answer": result["answer"],
            "sources": result.get("sources", []),
        })
        st.rerun()

# --- 7. METADATA & PREVIEW (RIGHT COLUMN) ---
with meta_col:
    view_tabs = st.tabs([
        _arabic("Document Original", "المستند الأصلي"),
        _arabic("Analyse IA", "تحليل الذكاء الاصطناعي")
    ])

    with view_tabs[0]:
        if st.session_state.uploaded_text:
            st.text_area("Aperçu du document", st.session_state.uploaded_text[:3000], height=400, disabled=True)
        else:
            st.info(_arabic("Aucun document chargé.", "لم يتم تحميل أي مستند."))

    with view_tabs[1]:
        if st.session_state.history and st.session_state.history[-1].get("sources"):
            for src in st.session_state.history[-1]["sources"]:
                st.write(f"**{src.get('doc_id', '?')}** - Score: {src.get('score', 0)}")
        else:
            st.info(_arabic("Posez une question pour voir l'analyse des sources.", "اطرح سؤالاً لرؤية تحليل المصادر."))