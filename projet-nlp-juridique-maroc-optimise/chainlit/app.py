"""
app.py
-------
Interface Chainlit pour le chatbot RAG juridique marocain.
Usage : chainlit run app.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import chainlit as cl

# Attempt to import fitz for PDF processing
try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).resolve().parent))

# API Key Check
if not os.environ.get("GROQ_API_KEY"):
    print("WARNING: GROQ_API_KEY non définie.")

# Fallback imports
try:
    from src.rag.chatbot import LegalRAGChatbot
    from src.rag.prompt_builder import SYSTEM_INSTRUCTION, build_user_prompt
    from src.rag.llm_client import LLMClient
    MODULES_LOADED = True
except ImportError:
    MODULES_LOADED = False
    SYSTEM_INSTRUCTION = "Vous êtes un assistant juridique."

# --- HELPER FUNCTIONS ---

def process_uploaded_file(file_bytes: bytes, file_name: str) -> str:
    """Handles text extraction for PDF and TXT files securely."""
    try:
        if file_name.endswith(".txt"):
            return file_bytes.decode("utf-8")
        elif file_name.endswith(".pdf"):
            if not FITZ_AVAILABLE:
                return "Erreur: La bibliothèque PyMuPDF (fitz) n'est pas installée."
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                text = "\n".join(str(page.get_text()) for page in doc)
            return text
        return ""
    except Exception as e:
        return f"Erreur de lecture: {e}"

def _arabic(fr: str, ar: str, use_arabic: bool) -> str:
    return ar if use_arabic else fr

# --- CHAINLIT EVENTS ---

@cl.on_chat_start
async def start():
    """Initializes the session when a user opens the app."""
    cl.user_session.set("history", [])
    cl.user_session.set("uploaded_text", None)
    cl.user_session.set("uploaded_name", None)
    cl.user_session.set("bot", None)
    cl.user_session.set("bot_error", None)
    
    # Send the welcome message
    await cl.Message(
        content="Bonjour ! Je suis votre assistant juridique marocain. \n\nVous pouvez me poser une question sur la loi, ou **télécharger un document (PDF/TXT)** directement dans le chat pour que je l'analyse.",
        author="ADLI Morocco"
    ).send()

@cl.on_settings_update
async def settings_update(settings):
    """Handles the language toggle from the UI settings."""
    cl.user_session.set("use_arabic", settings.get("use_arabic", False))

@cl.on_message
async def main(message: cl.Message):
    """Main logic triggered every time the user sends a message or a file."""
    
    use_arabic = cl.user_session.get("use_arabic", False)
    
    # 1. HANDLE FILE UPLOADS (Chainlit handles this natively and beautifully)
    if message.elements:
        for element in message.elements:
            if isinstance(element, cl.File):
                # Extract text
                extracted_text = process_uploaded_file(element.content, element.name)
                
                cl.user_session.set("uploaded_text", extracted_text)
                cl.user_session.set("uploaded_name", element.name)
                
                # Notify the user that the document is loaded
                await cl.Message(
                    content=f"✅ Document `{element.name}` chargé avec succès. Posez-moi une question dessus.",
                    author="Système"
                ).send()
                
                # If the user ONLY uploaded a file without asking a question, stop here.
                if not message.content.strip():
                    return

    query = message.content.strip()
    if not query:
        return

    has_doc = bool(cl.user_session.get("uploaded_text"))
    
    # 2. SHOW THINKING INDICATOR
    msg = cl.Message(content="", author="ADLI Morocco")
    await msg.send()
    
    # 3. PROCESS THE QUERY
    result = {"answer": "", "sources": []}
    
    # Route A: Document Context
    if has_doc and MODULES_LOADED:
        try:
            llm = LLMClient()
            doc_context = cl.user_session.get("uploaded_text") or ""
            context_excerpt = (doc_context[:80000] if isinstance(doc_context, str) else str(doc_context))

            history_block = ""
            history = cl.user_session.get("history")
            if history:
                turns = [f"Question : {t['question']}\nRéponse : {t['answer']}" for t in history[-4:]]
                history_block = "Historique :\n\n" + "\n\n".join(turns) + "\n\n"

            user_prompt = (
                f"Contexte du document ({cl.user_session.get('uploaded_name')}) :\n\n"
                f"{context_excerpt}\n\n{history_block}"
                f"Nouvelle question : {query}\n\n"
                f"Rappel : réponds uniquement à partir du texte ci-dessus. Cite le numéro de décret exact et la date textuelle entre parenthèses."
            )
            
            # Chainlit's step feature allows us to stream the response token by token
            answer_text = ""
            async for chunk in llm.generate_stream(SYSTEM_INSTRUCTION, user_prompt):
                answer_text += chunk
                await msg.stream_token(chunk)
                
            result["answer"] = answer_text
            result["sources"] = [{"doc_id": cl.user_session.get("uploaded_name"), "article_number": "—", "text": "Extrait du document local", "score": 1.0}]
            
        except Exception as exc:
            result["answer"] = f"❌ Erreur: {exc}"
            
    # Route B: RAG Vector Database
    elif not has_doc and MODULES_LOADED:
        bot = cl.user_session.get("bot")
        if not bot:
            try:
                bot = LegalRAGChatbot()
                cl.user_session.set("bot", bot)
            except Exception as exc:
                cl.user_session.set("bot_error", str(exc))
                bot = None
                
        if bot:
            try:
                # Note: If your RAG supports streaming, implement it here. 
                # Otherwise, we just send the final response.
                res = bot.answer(query=query, history=cl.user_session.get("history"), top_k=5)
                result = res
                
                # Stream the final result to make it look smooth
                await msg.stream_token(result.get("answer", ""))
                
            except Exception as exc:
                result["answer"] = f"❌ Erreur RAG: {exc}"
        else:
            result["answer"] = f"**[Erreur]** Impossible d'initialiser la base RAG."

    # Route C: Demo / Fallback
    else:
        error_msg = cl.user_session.get("bot_error") or "Veuillez charger un document ou configurer la base RAG."
        fallback_text = f"**[Mode Démo]** {error_msg}"
        await msg.stream_token(fallback_text)
        result["answer"] = fallback_text

    # 4. ADD CITATIONS (Native Chainlit Feature)
    if result.get("sources"):
        sources_elements = []
        for src in result["sources"]:
            # Creates a beautiful, expandable citation block inside the chat
            sources_elements.append(
                cl.Text(name=f"📜 {src.get('doc_id', '?')} - Art. {src.get('article_number', '?')}", 
                        content=src.get('text', '')[:500], 
                        display="inline")
            )
        
        # Append elements to the message before finalizing
        msg.elements = sources_elements

    # 5. FINALIZE MESSAGE & SAVE HISTORY
    await msg.update()
    
    cl.user_session.get("history").append({
        "question": query,
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
    })