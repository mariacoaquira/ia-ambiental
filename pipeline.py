# pipeline.py
import os, json, math, re, unicodedata, hashlib, datetime, uuid
from typing import List, Dict, Any, Optional, Iterable
from dotenv import load_dotenv
from pypdf import PdfReader, PdfWriter
from google.cloud import documentai, storage
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from pinecone import Pinecone as PineconeClient

load_dotenv()
creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if creds_path:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
    
# ── Configuración ─────────────────────────────────────────────────
GCP_PROJECT_ID              = os.getenv("GCP_PROJECT_ID")
GCS_BUCKET_IGA              = "asistente-ambiental"
DOCUMENT_AI_LOCATION        = "us"
DOCUMENT_AI_PROCESSOR_ID    = "72298e5ca0163c73"
DOCUMENT_AI_PROCESSOR_VERSION = "pretrained-layout-parser-v1.0-2024-06-03"
PINECONE_INDEX_NAME         = "reporte-iga"
PINECONE_IGA_NAMESPACE      = "iga"
VERTEXAI_EMBEDDING_MODEL    = "text-multilingual-embedding-002"
VERTEXAI_LOCATION           = "us-central1"
DOC_AI_PAGE_LIMIT           = 900
DOCUMENT_AI_TIMEOUT_SECONDS = 1800
DOC_VERSION                 = "v1"

# ── Clientes GCP ──────────────────────────────────────────────────
storage_client    = storage.Client()
documentai_client = documentai.DocumentProcessorServiceClient()
processor_name    = documentai_client.processor_version_path(
    GCP_PROJECT_ID, DOCUMENT_AI_LOCATION,
    DOCUMENT_AI_PROCESSOR_ID, DOCUMENT_AI_PROCESSOR_VERSION
)

# ── Pega aquí todas tus funciones del notebook ────────────────────
def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00A0"," ").replace("\u00AD","")
    s = re.sub(r"(\S)-\n(\S)", r"\1\2", s)
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def clean_text_line(s: str) -> str:
    s = normalize_text(s)
    return s

def is_noise(s: str) -> bool:
    if not s: return True
    if re.fullmatch(r"\d{1,3}", s): return True
    if re.fullmatch(r"0{3,}\d+", s): return True
    return False

def file_sha256(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

"""##### **Funciones de Document AI / GCS**"""

import json
import math
from pypdf import PdfReader, PdfWriter
from typing import List, Dict, Any, Optional

def process_document(gcs_file_path: str, gcs_bucket_name: str, output_prefix: Optional[str] = None) -> list[dict]:

    file_base = gcs_file_path.rsplit('.', 1)[0]
    gcs_input_uri = f"gs://{gcs_bucket_name}/input/{gcs_file_path}"

    if output_prefix is not None:
        norm = output_prefix.strip().strip("/")
        if not norm:
            raise ValueError("output_prefix no puede ser vacío ni solo '/'")
        out_prefix = f"{norm}/"
    else:
        out_prefix = f"output/json/{file_base}/"

    gcs_output_uri = f"gs://{gcs_bucket_name}/{out_prefix}"

    print(f"Iniciando DocAI para:  {gcs_input_uri}")
    print(f"DocAI output: gs://{gcs_bucket_name}/{out_prefix}")

    bucket = storage_client.bucket(gcs_bucket_name)

    existing = [b for b in bucket.list_blobs(prefix=out_prefix) if b.name.endswith(".json")]
    if existing:
        existing = sorted(existing, key=lambda x: x.name)
        print(f"Reutilizando JSONs previos: {len(existing)} desde {gcs_output_uri}")
        return [json.loads(b.download_as_text()) for b in existing]

    gcs_document = documentai.GcsDocument(gcs_uri=gcs_input_uri, mime_type="application/pdf")
    input_config = documentai.BatchDocumentsInputConfig(
        gcs_documents=documentai.GcsDocuments(documents=[gcs_document])
    )
    output_config = documentai.DocumentOutputConfig(
        gcs_output_config=documentai.DocumentOutputConfig.GcsOutputConfig(gcs_uri=gcs_output_uri)
    )

    req = documentai.BatchProcessRequest(
        name=processor_name,
        input_documents=input_config,
        document_output_config=output_config
    )

    operation = documentai_client.batch_process_documents(req)
    operation.result(DOCUMENT_AI_TIMEOUT_SECONDS)

    blobs = sorted(
        [b for b in bucket.list_blobs(prefix=out_prefix) if b.name.endswith(".json")],
        key=lambda x: x.name,
    )

    if not blobs:
        raise FileNotFoundError(f"No se encontró JSON de salida en {gcs_output_uri}")

    outs = [json.loads(b.download_as_text()) for b in blobs]
    print(f"JSONs de DocAI descargados: {len(outs)} desde {gcs_output_uri}")
    return outs

def process_large_pdf(local_file_path: str, gcs_bucket_name: str) -> dict:

    print(f"Iniciando procesamiento de {local_file_path}...")
    reader = PdfReader(local_file_path)
    total_pages = len(reader.pages)
    print(f"El documento tiene {total_pages} páginas.")

    original_sha256 = file_sha256(local_file_path)
    print(f"SHA256 del PDF original: {original_sha256}")

    parts_info = []
    bucket = storage_client.bucket(gcs_bucket_name)

    if total_pages <= DOC_AI_PAGE_LIMIT:
        print("Documento bajo el límite. Procesando directamente.")
        gcs_file_name = os.path.basename(local_file_path)
        bucket.blob(f"input/{gcs_file_name}").upload_from_filename(local_file_path, content_type="application/pdf")
        print(f"Archivo subido a GCS: gs://{gcs_bucket_name}/input/{gcs_file_name}")

        output_prefix = f"output/json/by_sha/{original_sha256}"
        try:
            jsons = process_document(gcs_file_name, gcs_bucket_name, output_prefix=output_prefix)
        except Exception as e:
            print(f"Se ha producido un error obteniendo JSONs: {e}")
            jsons = []

        parts_info.append({"gcs_file": gcs_file_name, "page_offset": 0, "jsons": jsons, "part_sha256": original_sha256})

    else:
        num_parts = math.ceil(total_pages / DOC_AI_PAGE_LIMIT)
        print(f"Documento excede el límite. Se generarán {num_parts} partes de hasta {DOC_AI_PAGE_LIMIT} páginas.")
        file_base_name = os.path.splitext(local_file_path)[0]

        for i in range(num_parts):
            writer = PdfWriter()
            start_page = i * DOC_AI_PAGE_LIMIT
            end_page = min((i + 1) * DOC_AI_PAGE_LIMIT, total_pages)
            part_local_name = f"{file_base_name}_part_{i+1}.pdf"
            gcs_file_name = os.path.basename(part_local_name)

            print(f"Creando parte {i+1} (páginas {start_page+1}-{end_page})...")
            for j in range(start_page, end_page):
                writer.add_page(reader.pages[j])
            with open(part_local_name, "wb") as f:
                writer.write(f)

            part_sha256 = file_sha256(part_local_name)

            bucket.blob(f"input/{gcs_file_name}").upload_from_filename(part_local_name, content_type="application/pdf")
            os.remove(part_local_name)
            print(f"Parte {i+1} subida a GCS: gs://{gcs_bucket_name}/input/{gcs_file_name}")

            output_prefix = f"output/json/by_sha/{original_sha256}/part_{i+1}"

            try:
                jsons = process_document(gcs_file_name, gcs_bucket_name, output_prefix=output_prefix)
            except Exception as e:
                print(f"Se ha producido un error obteniendo JSONs: {e}")
                jsons = []

            parts_info.append({"gcs_file": gcs_file_name, "page_offset": start_page, "jsons": jsons, "part_sha256": part_sha256})

    return {"total_pages": total_pages, "parts": parts_info, "original_sha256": original_sha256}

"""##### Funciones de extracción de layout y Serialización a JSONL"""

from typing import Iterable
import datetime

def _extract_texts_from_block(block: Dict[str, Any]) -> List[str]:
    texts = []
    tb = block.get("textBlock")
    if tb:
        t = tb.get("text")
        if isinstance(t, str) and t.strip():
            texts.append(normalize_text(t))
        for sub in tb.get("blocks", []) or []:
            texts.extend(_extract_texts_from_block(sub))
    for sub in block.get("blocks", []) or []:
        texts.extend(_extract_texts_from_block(sub))
    return texts

def _cells_from_row(row: Dict[str, Any]) -> List[str]:
    row_cells = []
    for cell in row.get("cells", []) or []:
        cell_texts: List[str] = []
        for sub in cell.get("blocks", []) or []:
            cell_texts.extend(_extract_texts_from_block(sub))
        raw = " ".join(t for t in cell_texts if t)
        row_cells.append(normalize_text(raw))
    return row_cells

def _yield_table_rows(tb: dict, page_start: int, source_block_id: str, iga_id: str):
    for j, row in enumerate(tb.get("headerRows", []) or []):
        cells = _cells_from_row(row)
        if any(cells):
            yield {
                "iga_id": iga_id, "page": page_start,
                "kind": "table_header_row", "text": None, "cells": cells,
                "block_id": f"{source_block_id}-h{j:03d}",
            }
    for j, row in enumerate(tb.get("bodyRows", []) or []):
        cells = _cells_from_row(row)
        if any(cells):
            yield {
                "iga_id": iga_id, "page": page_start,
                "kind": "table_body_row", "text": None, "cells": cells,
                "block_id": f"{source_block_id}-r{j:03d}",
            }

def iter_layout_records(blocks: Iterable[dict], iga_id: str):
    for b in blocks or []:
        page_start = int(b.get("pageSpan", {}).get("pageStart", 0)) or None
        block_id = b.get("blockId", "") or "blk"

        if "textBlock" in b:
            tb = b["textBlock"]
            t = tb.get("text")
            if isinstance(t, str) and t.strip():
                yield {
                    "iga_id": iga_id, "page": page_start,
                    "kind": "text", "text": normalize_text(t), "cells": None,
                    "block_id": block_id,
                }

            for sub in tb.get("blocks", []) or []:
                if "tableBlock" in sub:
                    sub_id = sub.get("blockId", block_id)
                    yield from _yield_table_rows(sub["tableBlock"], page_start, sub_id, iga_id)
                if "textBlock" in sub or "blocks" in sub:
                    yield from iter_layout_records([sub], iga_id)

        if "tableBlock" in b:
            yield from _yield_table_rows(b["tableBlock"], page_start, block_id, iga_id)

        for sub in b.get("blocks", []) or []:
            yield from iter_layout_records([sub], iga_id)

def generate_clean_stream_from_jsonl(jsonl_path: str, out_path: str):
    out_lines = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            page = rec.get("page") or "?"
            blk  = rec.get("block_id") or "?"
            kind = rec.get("kind","")
            out_lines.append(f"[PAG={page}][BLK={blk}]")
            if kind.startswith("table_"):
                cells = rec.get("cells") or []
                norm_cells = [normalize_text((c or "")) for c in cells]
                out_lines.append("| " + " | ".join(norm_cells) + " |")
                out_lines.append("")
            else:
                t = clean_text_line(rec.get("text") or "")
                if not is_noise(t):
                    out_lines.append(t)
                    out_lines.append("")

    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(out_lines)).strip() + "\n"
    with open(out_path, "w", encoding="utf-8") as w:
        w.write(cleaned)
    print(f"Stream limpio escrito en {out_path}")

def generate_raw_jsonl(payload: dict, iga_id: str, local_pdf_path: str, jsonl_path: str, bucket_name: str, processor_id: str, processor_version: str):

    file_name = os.path.basename(local_pdf_path)
    total_pages = int(payload["total_pages"])
    run_ts = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    is_single = (
        len(payload.get("parts", [])) == 1
        and int(payload["parts"][0]["page_offset"]) == 0
        and payload["parts"][0]["gcs_file"] == file_name
    )

    gcs_uri_original = f"gs://{bucket_name}/input/{file_name}" if is_single else None

    original_sha256 = payload.get("original_sha256")

    if is_single:
        print(f"PDF no partido en gcs_uri_original: {gcs_uri_original}")

    total = 0

    with open(jsonl_path, "w", encoding="utf-8") as fj:

        for part_idx, part in enumerate(payload["parts"]):
            offset = int(part["page_offset"])
            part_uri = f"gs://{bucket_name}/input/{part['gcs_file']}"
            part_sha256 = part.get("part_sha256")

            for layout_json in part["jsons"]:
                blocks = layout_json.get("documentLayout", {}).get("blocks", []) or []
                for rec in iter_layout_records(blocks, iga_id):
                    abs_page = (rec.get("page") or 0) + offset if rec.get("page") else None
                    out = {
                        "iga_id": iga_id,
                        "page": abs_page,
                        "kind": rec.get("kind"),
                        "text": rec.get("text"),
                        "cells": rec.get("cells"),
                        "block_id": rec.get("block_id"),
                        "source": {
                            "file_name": file_name,
                            "gcs_uri_original": gcs_uri_original,
                            "gcs_part_uri": part_uri,
                            "part_index": part_idx,
                            "page_offset": offset,
                            "num_pages_total": total_pages,
                            "original_sha256": original_sha256,
                            "part_sha256": part_sha256,
                        },
                        "extraction": {
                            "processor": f"{processor_id}@{processor_version}",
                            "run_ts": run_ts
                        }
                    }
                    fj.write(json.dumps(out, ensure_ascii=False) + "\n")
                    total += 1

    print(f"Partes: {len(payload['parts'])} | Páginas totales: {total_pages} | Registros: {total}")


# ── Función principal del pipeline ───────────────────────────────
def procesar_iga(local_pdf_path: str) -> str:
    """Ejecuta Hito 1 y Hito 2. Devuelve el doc_id."""
    iga_id     = f"iga-{uuid.uuid4()}"
    jsonl_path = "iga_raw.jsonl"

    print(f"[1/3] Procesando con Document AI: {local_pdf_path}")
    payload = process_large_pdf(local_pdf_path, GCS_BUCKET_IGA)
    generate_raw_jsonl(
        payload=payload,
        iga_id=iga_id,
        local_pdf_path=local_pdf_path,
        jsonl_path=jsonl_path,
        bucket_name=GCS_BUCKET_IGA,
        processor_id=DOCUMENT_AI_PROCESSOR_ID,
        processor_version=DOCUMENT_AI_PROCESSOR_VERSION,
    )

    print("[2/3] Generando embeddings y subiendo a Pinecone...")
    original_sha256 = payload["original_sha256"]
    doc_id          = f"iga:{original_sha256}"

    with open(jsonl_path, "r", encoding="utf-8") as f:
        lineas = [json.loads(l) for l in f]

    partes = []
    for rec in lineas:
        if rec.get("text"):
            partes.append(rec["text"])
        elif rec.get("cells"):
            partes.append(" | ".join(c for c in rec["cells"] if c))

    iga_stream = " ".join(partes)

    base_doc = Document(
        page_content=iga_stream,
        metadata={
            "doc_id":          doc_id,
            "doc_family":      "iga",
            "doc_version":     DOC_VERSION,
            "source":          os.path.basename(local_pdf_path),
            "original_sha256": original_sha256,
        },
    )

    splitter = RecursiveCharacterTextSplitter(
        separators=[r"\n{2,}", r"\n", r"\. ", " ", ""],
        is_separator_regex=True,
        chunk_size=2000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents([base_doc])

    prefixed = [
        Document(
            page_content="passage: " + c.page_content,
            metadata={**c.metadata, "chunk_idx": i, "text": c.page_content},
        )
        for i, c in enumerate(chunks)
    ]

    emb = VertexAIEmbeddings(
        model_name=VERTEXAI_EMBEDDING_MODEL,
        project=GCP_PROJECT_ID,
        location=VERTEXAI_LOCATION,
    )
    pc  = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
    idx = pc.Index(PINECONE_INDEX_NAME)

    BATCH = 16
    for i in range(0, len(prefixed), BATCH):
        batch = prefixed[i:i+BATCH]
        vectors = emb.embed_documents([d.page_content for d in batch])
        upsert_data = [
            {
                "id":       f"{doc_id}:{DOC_VERSION}:{i+j}",
                "values":   vectors[j],
                "metadata": batch[j].metadata
            }
            for j, _ in enumerate(batch)
        ]
        idx.upsert(vectors=upsert_data, namespace=PINECONE_IGA_NAMESPACE)
    
    print(f"[3/3] Listo. DOC_ID: {doc_id} | Chunks: {len(prefixed)}")
    return doc_id

if __name__ == "__main__":
    import sys
    pdf = sys.argv[1] if len(sys.argv) > 1 else "pdfs/IGA.pdf"
    doc_id = procesar_iga(pdf)
    print(f"\nDOC_ID para usar en extractor: {doc_id}")