# chatbot.py
import os
from dotenv import load_dotenv
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

# ── Configuración — igual que el colab original ────────────────────
GCP_PROJECT_ID         = os.getenv("GCP_PROJECT_ID", "asistente-ambiental")
VERTEXAI_LOCATION      = "us-central1"
VERTEXAI_EMB_MODEL     = "text-multilingual-embedding-002"
VERTEXAI_MODEL_NAME    = "gemini-2.5-flash"
PINECONE_INDEX_NAME    = "reporte-iga"
PINECONE_IGA_NAMESPACE = "iga"
DOC_ID_DEFAULT         = "iga:3b0f4eb7cfe21c807aa05a56d281f56bd9dafbddd98718cda442006cce94ed17"

def chat_con_iga(pregunta: str, doc_id: str = None, historial: list = None) -> dict:
    """
    Chat RAG con el IGA.
    Usa la misma lógica del colab original:
    - VertexAI embeddings para buscar en Pinecone
    - Gemini para generar la respuesta
    - similarity_search_with_score igual que el sanity check original
    """
    if not doc_id:
        doc_id = DOC_ID_DEFAULT
    if historial is None:
        historial = []

    # ── Embeddings — igual que el colab ───────────────────────────
    emb = VertexAIEmbeddings(
        model_name=VERTEXAI_EMB_MODEL,
        project=GCP_PROJECT_ID,
        location=VERTEXAI_LOCATION
    )

    # ── Vector store — igual que iga_store del colab ───────────────
    iga_store = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=emb,
        namespace=PINECONE_IGA_NAMESPACE
    )

    # ── Búsqueda por similitud — igual que el sanity check ─────────
    docs = iga_store.similarity_search_with_score(
        f"query: {pregunta}",
        k=5,
        filter={"doc_id": doc_id}
    )

    # ── Construir contexto con referencias de página ───────────────
    contexto = ""
    fuentes  = []
    for i, (doc, score) in enumerate(docs):
        texto  = doc.page_content.replace("passage: ", "")
        pagina = doc.metadata.get("page", "?")
        idx    = doc.metadata.get("chunk_idx", "?")
        contexto += f"[{i+1}] (p.{pagina}): {texto}\n\n"
        fuentes.append({
            "ref":       i+1,
            "pagina":    pagina,
            "chunk_idx": idx,
            "score":     round(float(score), 4)
        })

    # ── Gemini genera la respuesta — igual que VERTEXAI_MODEL_NAME ──
    llm = ChatVertexAI(
        model_name=VERTEXAI_MODEL_NAME,
        project=GCP_PROJECT_ID,
        location=VERTEXAI_LOCATION,
        temperature=0.2,
        max_output_tokens=1024
    )

    # ── Construir mensajes con historial ───────────────────────────
    mensajes = []
    for msg in historial[-6:]:
        if msg["role"] == "user":
            mensajes.append(HumanMessage(content=msg["content"]))
        else:
            mensajes.append(AIMessage(content=msg["content"]))

    # Mensaje actual con contexto del IGA
    mensajes.append(HumanMessage(
        content=f"Contexto del IGA:\n{contexto}\n\nPregunta: {pregunta}"
    ))

    respuesta = llm.invoke(mensajes)

    return {
        "respuesta": respuesta.content,
        "fuentes":   fuentes
    }