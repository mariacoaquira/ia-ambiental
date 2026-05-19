# extractor.py
import os, json, time, re
from dotenv import load_dotenv
import anthropic
from pinecone import Pinecone as PineconeClient
from langchain_google_vertexai import VertexAIEmbeddings

load_dotenv()
creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if creds_path:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path

# ── Configuración ─────────────────────────────────────────────────
PINECONE_API_KEY       = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME    = "reporte-iga"
PINECONE_IGA_NAMESPACE = "iga"
GCP_PROJECT_ID         = os.getenv("GCP_PROJECT_ID")
VERTEXAI_LOCATION      = "us-central1"
VERTEXAI_EMB_MODEL     = "text-multilingual-embedding-002"
ANTHROPIC_API_KEY      = os.getenv("ANTHROPIC_API_KEY")
CACHE_DIR              = "/tmp/cache_iga"
SCORE_MINIMO           = 0.70   # umbral mínimo de relevancia
TOP_K_PINECONE         = 8      # chunks a recuperar por categoría en Fase 1

# ── Clientes (lazy) ───────────────────────────────────────────────
def _get_clientes():
    cliente_claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    pc             = PineconeClient(api_key=PINECONE_API_KEY)
    idx_client     = pc.Index(PINECONE_INDEX_NAME)
    emb            = VertexAIEmbeddings(
        model_name=VERTEXAI_EMB_MODEL,
        project=GCP_PROJECT_ID,
        location=VERTEXAI_LOCATION,
    )
    return cliente_claude, idx_client, emb

# ── Categorías ────────────────────────────────────────────────────
CATEGORIAS_IGA = {
    "manejo_infraestructura_suelo": (
        "habilitación construcción rehabilitación accesos plataformas "
        "perforación pozas sedimentación lodo material removido taludes "
        "cunetas canales geomembrana erosión hídrica eólica nivelación "
        "relleno revegetación cierre desmantelamiento obturación sondajes "
        "especies nativas suelo orgánico infraestructura equipos remoción"
    ),
    "control_agua_aire_ruido": (
        "calidad agua efluentes descarga cuerpos agua superficial LMP "
        "recirculación aguas residuales tratamiento sistema bombeo bocamina "
        "emisiones gases combustión material particulado polvo PM10 ruido "
        "vibraciones mantenimiento vehículos maquinaria silenciadores "
        "riego periódico secano calidad aire monitoreo parámetros"
    ),
    "residuos_sustancias_derrames": (
        "residuos sólidos peligrosos domésticos industriales EPS-RS DIGESA "
        "almacenamiento temporal cilindros colores segregación sustancias "
        "químicas combustibles aceites insumos geomembrana bandeja contención "
        "baño químico disposición final derrames hidrocarburos paños absorbentes "
        "Landfarm suelo contaminado contingencias emergencias plan respuesta"
    ),
    "flora_fauna_arqueologia_social": (
        "flora fauna cactáceas rescate reubicación especies nativas hábitat "
        "perturbado ahuyentamiento velocidad vehículos cobertura vegetal "
        "biodiversidad recursos arqueológicos CIRAs patrimonio cultural "
        "Ministerio Cultura hallazgo participación ciudadana relaciones "
        "comunitarias empleo local talleres información comunicaciones"
    ),
    "seguridad_salud_capacitacion": (
        "equipos protección personal EPP cascos lentes zapatos guantes "
        "protectores auditivos chalecos reflectores seguridad salud "
        "ocupacional capacitación señalización procedimientos evacuación "
        "incendio sismo emergencias plan respuesta implementar sistemas"
    ),
    "monitoreo_seguimiento_reporte": (
        "monitoreo seguimiento post cierre estabilidad taludes superficies "
        "intervenidas cronograma frecuencia parámetros puntos control "
        "laboratorio acreditado reporte ambiental OEFA MINEM DGAAM ANA "
        "registro bitácora informe trimestral semestral anual verificación"
    ),
}

# ── Extraer obligaciones de una categoría ─────────────────────────
def _extraer_categoria(categoria: str, chunks: list, cliente_claude) -> list:
    """Llama a Claude para extraer obligaciones de UNA categoría con sus chunks asignados."""
    if not chunks:
        return []

    contexto = "\n\n---\n\n".join([
        f"[PÁGINA {c['pagina']}] {c['texto']}" if c['pagina'] else c['texto']
        for c in chunks
    ])

    prompt = f"""Eres un auditor ambiental senior del OEFA en Perú.
Analiza los siguientes fragmentos de un Instrumento de Gestión Ambiental (IGA)
y extrae TODAS las obligaciones fiscalizables de la categoría: "{categoria}".

REGLAS ESTRICTAS:
1. VERBO EN INFINITIVO: Cada obligación DEBE iniciar con verbo operativo.
   Ejemplos: "Implementar", "Regar", "Disponer", "Monitorear", "Instalar".
2. DETALLE MÁXIMO: Incluye parámetros exactos, frecuencias y ubicaciones del texto.
3. UNA MEDIDA = UNA OBLIGACIÓN: Si aplica en áreas distintas con diferente frecuencia,
   crea una fila por variante.
4. SIN REFERENCIAS EXTERNAS: No escribas "ver ítem 7.4" — extrae el dato directamente.
5. SOLO LO QUE ESTÁ EN EL TEXTO: No inferir ni inventar.
6. SOLO categoría "{categoria}": Si un fragmento contiene obligaciones de otra
   categoría, ignóralas completamente.
7. Si no hay obligaciones de esta categoría, devuelve [].
8. AGRUPA medidas del mismo párrafo que tienen igual responsable, etapa y frecuencia
   en UNA sola obligación descriptiva.
9. Captura el número de página del prefijo [PÁGINA X] y asígnalo al campo "pagina".
10. Responde ÚNICAMENTE con array JSON válido, sin markdown ni texto adicional.

EJEMPLO DE EXTRACCIÓN PERFECTA:
[
  {{
    "descripcion": "Regar las vías de acceso no pavimentadas con camión cisterna para suprimir la dispersión de material particulado (PM10).",
    "plan": "Plan de Manejo Ambiental",
    "etapa": "Construcción",
    "frecuencia": "Diaria",
    "componente": "Vías de acceso y frentes de trabajo",
    "evidencia_cumplimiento": "Bitácora de riego con fecha, hora y sector",
    "responsable": "Jefe de Operaciones",
    "a_quien_reporta": "OEFA",
    "normativa": null,
    "parametros": "PM10",
    "pagina": 15
  }}
]

VALORES ESTANDARIZADOS:
Etapa: "Construcción" | "Operación" | "Cierre" | "Post-cierre" | "Todas las etapas"
Frecuencia: "Permanente" | "Diaria" | "Semanal" | "Quincenal" | "Mensual" |
  "Trimestral" | "Semestral" | "Anual" | "Puntual - Inicio de operaciones" |
  "Puntual - Al cierre de componente" | "Eventual - Ante derrame o emergencia" |
  "Según cronograma aprobado"
Autoridad: "OEFA" | "MINEM" | "DGAAM" | "ANA" | "GORE" | "DIGESA" | "No especificado"

FRAGMENTOS DEL DOCUMENTO (con número de página):
{contexto}

Array JSON:"""

    for intento in range(3):
        try:
            response = cliente_claude.messages.create(
                model="claude-haiku-4-5",
                max_tokens=8192,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            text = response.content[0].text.strip()
            text = re.sub(r'```json|```', '', text).strip()
            inicio = text.find("[")
            fin    = text.rfind("]")
            if inicio != -1 and fin != -1:
                text = text[inicio:fin+1]

            obligs = json.loads(text)
            return obligs if isinstance(obligs, list) else []

        except json.JSONDecodeError:
            print(f"    [intento {intento+1}/3] JSON malformado, reintentando...")
            time.sleep(5)
        except Exception as e:
            msg = str(e)
            if 'overloaded' in msg.lower() or '529' in msg:
                print(f"    [Claude sobrecargado] Esperando 30s...")
                time.sleep(30)
            else:
                print(f"    [error]: {msg[:80]}")
                break

    print(f"    [fallo definitivo en '{categoria}']")
    return []

# ── Función principal ─────────────────────────────────────────────
def extraer_obligaciones(doc_id: str) -> dict:
    print(f"\nExtrayendo obligaciones para: {doc_id}")
    print("=" * 60)
    resultado = {"doc_id": doc_id, "total": 0, "por_categoria": {}}

    # ── Caché por doc_id ──────────────────────────────────────────
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = f"{CACHE_DIR}/{doc_id.replace(':','_').replace('/','_')}.json"
    print(f"  [caché] Buscando: {cache_path}")  # ← AGREGA
    if os.path.exists(cache_path):
        print(f"  [caché] Encontrado — devolviendo resultado previo")  # ← AGREGA
    if os.path.exists(cache_path):
        print(f"  [caché] Usando resultado previo para {doc_id}")
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    cliente_claude, idx_client, emb = _get_clientes()

    # ── FASE 1: Recolección y deduplicación ───────────────────────
    print("  [Fase 1] Recuperando chunks de Pinecone...")
    pool = {}  # chunk_id → {texto, pagina, scores: {categoria: score}}

    for categoria, query in CATEGORIAS_IGA.items():
        vector = emb.embed_query(f"query: {query}")
        res = idx_client.query(
            vector=vector,
            top_k=TOP_K_PINECONE,
            namespace=PINECONE_IGA_NAMESPACE,
            filter={"doc_id": {"$eq": doc_id}},
            include_metadata=True,
        )
        for m in res.matches:
            texto = m.metadata.get("text", "").strip()
            if not texto:
                continue
            if m.id not in pool:
                pool[m.id] = {
                    "texto":  texto,
                    "pagina": m.metadata.get("page", None),
                    "scores": {}
                }
            pool[m.id]["scores"][categoria] = m.score

    print(f"  [Fase 1] {len(pool)} chunks únicos recuperados")

    if not pool:
        return resultado

    # ── Asignación exclusiva: cada chunk → categoría de mayor score ─
    chunks_por_categoria = {cat: [] for cat in CATEGORIAS_IGA}

    for chunk_id, chunk in pool.items():
        if not chunk["scores"]:
            continue
        mejor_categoria = max(chunk["scores"], key=chunk["scores"].get)
        mejor_score     = chunk["scores"][mejor_categoria]

        if mejor_score >= SCORE_MINIMO:
            chunks_por_categoria[mejor_categoria].append(chunk)

    print("\n  Distribución de chunks por categoría:")
    for cat, chunks in chunks_por_categoria.items():
        if chunks:
            scores = [f"{c['scores'][cat]:.3f}" for c in chunks]
            print(f"    {cat}: {len(chunks)} chunks | scores: {', '.join(scores)}")
        else:
            print(f"    {cat}: 0 chunks")

    # ── FASE 2: Extracción por categoría ─────────────────────────
    print("\n  [Fase 2] Extrayendo obligaciones por categoría...")
    for categoria in CATEGORIAS_IGA:
        chunks = chunks_por_categoria[categoria]
        print(f"  Procesando: {categoria} ({len(chunks)} chunks)...", end=" ", flush=True)

        obligs = _extraer_categoria(categoria, chunks, cliente_claude)
        resultado["por_categoria"][categoria] = obligs
        resultado["total"] += len(obligs)
        print(f"→ {len(obligs)} obligaciones")

    print(f"\nTOTAL: {resultado['total']} obligaciones extraídas")

    # ── Paso 3: Deduplicación final con Claude ────────────────────
    print("\n  [Fase 3] Deduplicando obligaciones con Claude...")
    todas_obligs = []
    for cat, obligs in resultado["por_categoria"].items():
        for o in obligs:
            o["_categoria"] = cat
            todas_obligs.append(o)

    if len(todas_obligs) > 5:
        prompt_dedup = f"""Eres un auditor ambiental senior del OEFA.
Revisa esta lista de obligaciones extraídas de un IGA y elimina duplicados semánticos.
Dos obligaciones son duplicadas si describen la MISMA acción aunque usen palabras distintas.
Conserva la versión más completa y detallada de cada obligación.
NO elimines obligaciones distintas aunque sean similares.
Responde ÚNICAMENTE con el array JSON filtrado, sin markdown.

OBLIGACIONES:
{json.dumps(todas_obligs, ensure_ascii=False)}

Array JSON sin duplicados:"""

        try:
            response = cliente_claude.messages.create(
                model="claude-haiku-4-5",
                max_tokens=8192,
                temperature=0,
                messages=[{"role": "user", "content": prompt_dedup}]
            )
            text = response.content[0].text.strip()
            text = re.sub(r'```json|```', '', text).strip()
            inicio = text.find("[")
            fin    = text.rfind("]")
            if inicio != -1 and fin != -1:
                text = text[inicio:fin+1]
            obligs_dedup = json.loads(text)

            # Reconstruir por_categoria
            resultado["por_categoria"] = {cat: [] for cat in CATEGORIAS_IGA}
            resultado["total"] = 0
            for o in obligs_dedup:
                cat = o.pop("_categoria", None)
                if cat and cat in resultado["por_categoria"]:
                    resultado["por_categoria"][cat].append(o)
                    resultado["total"] += 1
            print(f"  [Fase 3] {len(todas_obligs)} → {resultado['total']} obligaciones tras deduplicación")
        except Exception as e:
            print(f"  [Fase 3] Error en deduplicación: {e} — conservando resultado original")
            # Limpiar _categoria si falló
            for cat, obligs in resultado["por_categoria"].items():
                for o in obligs:
                    o.pop("_categoria", None)

    # ── Guardar caché ─────────────────────────────────────────────
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)

    return resultado


# ── EJECUCIÓN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    DOC_ID = "iga:3b0f4eb7cfe21c807aa05a56d281f56bd9dafbddd98718cda442006cce94ed17"

    resultado = extraer_obligaciones(DOC_ID)

    with open("obligaciones_iga.json", "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)

    print("\nGuardado en: obligaciones_iga.json")
