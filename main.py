# main.py
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json, os, uuid, shutil

app = FastAPI(title="Asistente Ambiental API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Rutas ──────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(__file__)
OBLIGACIONES_F = os.path.join(BASE_DIR, "obligaciones_iga.json")
UPLOADS_DIR    = os.path.join(BASE_DIR, "uploads")
JOBS_DIR       = os.path.join(BASE_DIR, "jobs")
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(JOBS_DIR,    exist_ok=True)

# ── Jobs en memoria (en producción usarías Redis o DB) ─────────────
jobs = {}

# ── Endpoints existentes ───────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "mensaje": "Asistente Ambiental API activa"}

@app.get("/api/obligaciones")
def get_obligaciones():
    with open(OBLIGACIONES_F, encoding="utf-8") as f:
        return json.load(f)

@app.get("/api/obligaciones/{categoria}")
def get_categoria(categoria: str):
    with open(OBLIGACIONES_F, encoding="utf-8") as f:
        data = json.load(f)
    obligs = data.get("por_categoria", {}).get(categoria, [])
    return {"categoria": categoria, "total": len(obligs), "obligaciones": obligs}

@app.get("/api/unidades")
def get_unidades():
    return {
        "empresa": "Sociedad Minera Cerro Verde S.A.A.",
        "ruc": "20170072465",
        "unidades": [
            {
                "id": "uf-001",
                "nombre": "U.P. La Querendosa",
                "codigo": "UF22-001",
                "sector": "Minería",
                "subsector": "Metálica",
                "actividad": "Exploración",
                "departamento": "Arequipa",
                "igas": [
                    {
                        "id": "iga-001",
                        "tipo": "DIA",
                        "nombre": "DIA Proyecto Exploración Geológica La Querendosa",
                        "resolucion": "Res. 142-2023-MEM/DGAAM",
                        "fecha_aprobacion": "2023-03-15",
                        "estado": "Activo",
                        "doc_id": "iga:3b0f4eb7cfe21c807aa05a56d281f56bd9dafbddd98718cda442006cce94ed17"
                    }
                ]
            },
            {
                "id": "uf-002",
                "nombre": "U.P. Cerro Verde",
                "codigo": "UF22-002",
                "sector": "Minería",
                "subsector": "Metálica",
                "actividad": "Explotación",
                "departamento": "Arequipa",
                "igas": []
            },
            {
                "id": "uf-003",
                "nombre": "U.P. Pampa Norte",
                "codigo": "UF23-001",
                "sector": "Minería",
                "subsector": "Metálica",
                "actividad": "Exploración",
                "departamento": "Moquegua",
                "igas": []
            }
        ]
    }

@app.get("/api/stats")
def get_stats():
    with open(OBLIGACIONES_F, encoding="utf-8") as f:
        data = json.load(f)
    total = data.get("total", 0)
    categorias = len(data.get("por_categoria", {}))
    return {
        "total_obligaciones": total,
        "total_categorias": categorias,
        "total_unidades": 3,
        "total_igas": 1,
        "pendientes": total,
        "cumplidas": 0
    }

# ── NUEVO: Procesamiento de IGA ────────────────────────────────────

def procesar_iga_background(job_id: str, pdf_path: str, tipo_iga: str, nombre_iga: str):
    """Corre en background — pipeline + extractor."""
    try:
        jobs[job_id] = {"status": "procesando", "paso": "Document AI", "progreso": 10}

        # Importar pipeline y extractor
        from pipeline import procesar_iga
        from extractor import extraer_obligaciones

        jobs[job_id] = {"status": "procesando", "paso": "Subiendo a GCS y procesando OCR", "progreso": 20}
        doc_id = procesar_iga(pdf_path)

        jobs[job_id] = {"status": "procesando", "paso": "Extrayendo obligaciones con Claude", "progreso": 60}
        resultado = extraer_obligaciones(doc_id)

        # Guardar resultado
        output_path = os.path.join(JOBS_DIR, f"{job_id}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "doc_id":   doc_id,
                "tipo_iga": tipo_iga,
                "nombre":   nombre_iga,
                "resultado": resultado
            }, f, ensure_ascii=False, indent=2)

        jobs[job_id] = {
            "status":    "completado",
            "paso":      "Listo",
            "progreso":  100,
            "doc_id":    doc_id,
            "total":     resultado.get("total", 0),
            "output":    f"/api/jobs/{job_id}"
        }

        # Limpiar PDF temporal
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

    except Exception as e:
        jobs[job_id] = {"status": "error", "mensaje": str(e), "progreso": 0}


@app.post("/api/iga/procesar")
async def procesar_iga_endpoint(
    background_tasks: BackgroundTasks,
    archivo: UploadFile = File(...),
    tipo_iga: str = Form("DIA"),
    nombre_iga: str = Form(""),
    unidad_id: str = Form("")
):
    """Recibe el PDF y lanza el procesamiento en background."""

    # Validar que sea PDF
    if not archivo.filename.endswith(".pdf"):
        return JSONResponse(status_code=400, content={"error": "Solo se aceptan archivos PDF"})

    # Guardar PDF temporalmente
    job_id  = str(uuid.uuid4())
    pdf_path = os.path.join(UPLOADS_DIR, f"{job_id}.pdf")

    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(archivo.file, f)

    # Inicializar job
    jobs[job_id] = {"status": "iniciando", "paso": "Recibiendo archivo", "progreso": 5}

    # Lanzar en background
    background_tasks.add_task(
        procesar_iga_background,
        job_id, pdf_path, tipo_iga, nombre_iga
    )

    return {"job_id": job_id, "mensaje": "Procesamiento iniciado"}


@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str):
    """Consulta el estado de un job de procesamiento."""
    if job_id not in jobs:
        # Buscar en disco si ya terminó
        output_path = os.path.join(JOBS_DIR, f"{job_id}.json")
        if os.path.exists(output_path):
            with open(output_path, encoding="utf-8") as f:
                data = json.load(f)
            return {"status": "completado", "progreso": 100, **data}
        return JSONResponse(status_code=404, content={"error": "Job no encontrado"})
    return jobs[job_id]


@app.get("/api/jobs/{job_id}/obligaciones")
def get_job_obligaciones(job_id: str):
    """Devuelve las obligaciones extraídas de un job completado."""
    output_path = os.path.join(JOBS_DIR, f"{job_id}.json")
    if not os.path.exists(output_path):
        return JSONResponse(status_code=404, content={"error": "Job no encontrado o en proceso"})
    with open(output_path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("resultado", {})

@app.post("/api/chat")
async def chat_endpoint(request: dict):
    """Chat RAG con el IGA usando Gemini + Pinecone."""
    from chatbot import chat_con_iga
    return chat_con_iga(
        pregunta  = request.get("pregunta", ""),
        doc_id    = request.get("doc_id"),
        historial = request.get("historial", [])
    )