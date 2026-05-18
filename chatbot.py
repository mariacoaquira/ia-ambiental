# chatbot.py
import os
from dotenv import load_dotenv
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage, AIMessage
from pinecone import Pinecone as PineconeClient

load_dotenv()

GCP_PROJECT_ID         = os.getenv("GCP_PROJECT_ID", "asistente-ambiental")
VERTEXAI_LOCATION      = "us-central1"
VERTEXAI_EMB_MODEL     = "text-multilingual-embedding-002"
VERTEXAI_MODEL_NAME    = "gemini-2.5-flash"
PINECONE_INDEX_NAME    = "reporte-iga"
PINECONE_IGA_NAMESPACE = "iga"
DOC_ID_DEFAULT         = "iga:3b0f4eb7cfe21c807aa05a56d281f56bd9dafbddd98718cda442006cce94ed17"

def chat_con_iga(pregunta: str, doc_id: str = None, historial: list = None) -> dict:
    if not doc_id:
        doc_id = DOC_ID_DEFAULT
    if historial is None:
        historial = []

    # ── Embeddings con VertexAI ────────────────────────────────────
    emb = VertexAIEmbeddings(
        model_name=VERTEXAI_EMB_MODEL,
        project=GCP_PROJECT_ID,
        location=VERTEXAI_LOCATION
    )
    vector = emb.embed_query(f"query: {pregunta}")

    # ── Búsqueda directa en Pinecone — igual que extractor.py ──────
    pc  = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
    idx = pc.Index(PINECONE_INDEX_NAME)
    res = idx.query(
        vector=vector,
        top_k=8,
        namespace=PINECONE_IGA_NAMESPACE,
        filter={"doc_id": {"$eq": doc_id}},
        include_metadata=True
    )

    # ── Construir contexto con referencias ─────────────────────────
    contexto = ""
    fuentes  = []
    for i, match in enumerate(res.matches):
        texto  = match.metadata.get("text", "")
        pagina = match.metadata.get("page", match.metadata.get("chunk_idx", "?"))
        idx_chunk = match.metadata.get("chunk_idx", "?")
        contexto += f"[{i+1}] (p.{pagina}): {texto}\n\n"
        fuentes.append({"ref": i+1, "pagina": pagina, "chunk_idx": idx_chunk})

    # ── Gemini genera la respuesta ─────────────────────────────────
    llm = ChatVertexAI(
        model_name=VERTEXAI_MODEL_NAME,
        project=GCP_PROJECT_ID,
        location=VERTEXAI_LOCATION,
        temperature=0.2,
        max_output_tokens=4096
    )

    mensajes = []
    for msg in historial[-6:]:
        if msg["role"] == "user":
            mensajes.append(HumanMessage(content=msg["content"]))
        else:
            mensajes.append(AIMessage(content=msg["content"]))

    mensajes.append(HumanMessage(
        content=f"""Eres un experto en instrumentos de gestión ambiental (IGA) peruanos.
    Tienes acceso a fragmentos del siguiente documento oficial.

    FRAGMENTOS DEL IGA:
    {contexto}

    PREGUNTA: {pregunta}

    Responde de forma completa y detallada basándote en los fragmentos.
    Cita las referencias [N] cuando uses información de un fragmento específico.
    Si la información no está en los fragmentos, indícalo claramente."""
    ))

    respuesta = llm.invoke(mensajes)

    return {
        "respuesta": respuesta.content,
        "fuentes":   fuentes
    }