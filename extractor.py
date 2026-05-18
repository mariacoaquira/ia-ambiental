# extractor.py
import os, json, time, re
from datetime import datetime, timedelta
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
os.makedirs(CACHE_DIR, exist_ok=True)

# ── Clientes ──────────────────────────────────────────────────────
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

# ── Categorías (las mismas que tienes en ia-ambiental.py) ─────────
CATEGORIAS_IGA = {
    "manejo_accesos_plataformas_pozas": (
        "habilitación construcción rehabilitación accesos plataformas "
        "perforación pozas sedimentación lodo material removido "
        "erosión hídrica eólica geomembrana taludes cunetas canales"
    ),
    "control_agua_efluentes": (
        "calidad agua efluentes descarga cuerpos agua superficial "
        "recirculación aguas residuales industriales LMP pozas "
        "infiltración bocamina tratamiento sistema bombeo"
    ),
    "control_emisiones_ruido_polvo": (
        "emisiones gases combustión material particulado polvo PM10 "
        "ruido vibraciones mantenimiento vehículos maquinaria "
        "silenciadores riego periódico secano calidad aire"
    ),
    "manejo_residuos_sustancias": (
        "residuos sólidos peligrosos domésticos industriales EPS-RS "
        "DIGESA almacenamiento temporal cilindros colores segregación "
        "sustancias químicas combustibles aceites insumos perforación "
        "geomembrana bandeja contención baño químico disposición final"
    ),
    "control_derrames_emergencias": (
        "derrames hidrocarburos paños absorbentes microfibras "
        "contingencias emergencias incendio sismo evacuación "
        "Landfarm suelo contaminado plan respuesta procedimientos"
    ),
    "proteccion_flora_fauna": (
        "flora fauna cactáceas rescate reubicación especies nativas "
        "hábitat perturbado ruido ahuyentamiento silenciadores "
        "velocidad vehículos cobertura vegetal biodiversidad"
    ),
    "proteccion_arqueologica_social": (
        "recursos arqueológicos CIRAs patrimonio cultural Ministerio "
        "Cultura hallazgo comunicaciones participación ciudadana "
        "relaciones comunitarias empleo local talleres información"
    ),
    "equipos_seguridad_personal": (
        "equipos protección personal EPP cascos lentes zapatos "
        "guantes protectores auditivos chalecos reflectores "
        "seguridad salud ocupacional capacitación señalización "
        "implementar sistemas comprometidos IGA infraestructura"
    ),
    "cierre_rehabilitacion_revegetacion": (
        "cierre plataformas pozas lodo accesos rehabilitación "
        "relleno rasgado recubrimiento nivelación revegetación "
        "especies nativas suelo orgánico obturación sondajes "
        "desmantelamiento remoción infraestructura equipos"
    ),
    "monitoreo_seguimiento": (
        "monitoreo seguimiento post cierre estabilidad taludes "
        "superficies intervenidas cronograma frecuencia parámetros "
        "puntos control laboratorio acreditado reporte ambiental"
    ),
}

# ── Función principal ─────────────────────────────────────────────
def extraer_obligaciones(doc_id: str) -> dict:
    print(f"\nExtrayendo obligaciones para: {doc_id}")
    print("=" * 60)
    resultado = {"doc_id": doc_id, "total": 0, "por_categoria": {}}

    # 1. Recuperar chunks de todas las categorías
    print("  Recuperando chunks relevantes...")
    chunks_vistos = set()
    chunks_unicos = []

    for categoria, query in CATEGORIAS_IGA.items():
        cliente_claude, idx_client, emb = _get_clientes()
        vector = emb.embed_query(f"query: {query}")
        res = idx_client.query(
            vector=vector,
            top_k=8,
            namespace=PINECONE_IGA_NAMESPACE,
            filter={"doc_id": {"$eq": doc_id}},
            include_metadata=True,
        )
        for m in res.matches:
            chunk_id = m.id
            texto = m.metadata.get("text", "").strip()
            pagina = m.metadata.get("page", None)
            if chunk_id not in chunks_vistos and texto:
                chunks_vistos.add(chunk_id)
                # Incluir página en el texto para que Claude la capture
                prefix = f"[PÁGINA {pagina}] " if pagina else ""
                chunks_unicos.append(f"{prefix}{texto}")

    print(f"  → {len(chunks_unicos)} chunks únicos recuperados")

    if not chunks_unicos:
        return resultado

    # 2. Una sola extracción con todos los chunks únicos
    categorias_desc = "\n".join([f'  - "{c}"' for c in CATEGORIAS_IGA.keys()])
    contexto = "\n\n---\n\n".join(chunks_unicos)

    prompt = f"""Eres un auditor ambiental senior del OEFA en Perú.
Analiza los siguientes fragmentos de un Instrumento de Gestión Ambiental (IGA)
y extrae TODAS las obligaciones fiscalizables, clasificándolas en las siguientes categorías:
{categorias_desc}

REGLAS ESTRICTAS:
1. VERBO EN INFINITIVO: Cada obligación DEBE iniciar con verbo operativo.
   Ejemplos: "Implementar", "Regar", "Disponer", "Monitorear", "Instalar".
2. DETALLE MÁXIMO: Incluye parámetros exactos, frecuencias y ubicaciones del texto.
3. UNA MEDIDA = UNA OBLIGACIÓN: Si aplica en áreas distintas con diferente frecuencia,
   crea una fila por variante.
4. SIN REFERENCIAS EXTERNAS: No escribas "ver ítem 7.4" — extrae el dato directamente.
5. SOLO LO QUE ESTÁ EN EL TEXTO: No inferir ni inventar.
6. NO DUPLIQUES: Cada obligación aparece en UNA sola categoría — la más específica.
7. AGRUPA medidas del mismo párrafo que tienen igual responsable, etapa y frecuencia
   en UNA sola obligación descriptiva.
8. Si una categoría no tiene obligaciones en los fragmentos, devuelve [] para esa categoría.
9. Captura el número de página indicado como [PÁGINA X] al inicio de cada fragmento
   y asígnalo al campo "pagina" de cada obligación extraída de ese fragmento.
10. Responde ÚNICAMENTE con objeto JSON válido, sin markdown ni texto adicional.

EJEMPLO DE EXTRACCIÓN PERFECTA:
{{
  "manejo_accesos_plataformas_pozas": [
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
  ],
  "control_agua_efluentes": []
}}

VALORES ESTANDARIZADOS:
Etapa: "Construcción" | "Operación" | "Cierre" | "Post-cierre" | "Todas las etapas"
Frecuencia: "Permanente" | "Diaria" | "Semanal" | "Quincenal" | "Mensual" |
  "Trimestral" | "Semestral" | "Anual" | "Puntual - Inicio de operaciones" |
  "Puntual - Al cierre de componente" | "Eventual - Ante derrame o emergencia" |
  "Según cronograma aprobado"
Autoridad: "OEFA" | "MINEM" | "DGAAM" | "ANA" | "GORE" | "DIGESA" | "No especificado"

FRAGMENTOS DEL DOCUMENTO (con número de página):
{{contexto}}

JSON:"""

    cliente_claude, _, _ = _get_clientes()
    # Cache por doc_id para no reprocesar
    cache_path = f"{CACHE_DIR}/{doc_id.replace(':','_').replace('/','_')}.json"
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(cache_path):
        print(f"  [caché] Usando resultado previo para {doc_id}")
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    cliente_claude, _, _ = _get_clientes()
    for intento in range(3):
        try:
            response = cliente_claude.messages.create(
                model="claude-haiku-4-5",
                max_tokens=8192,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            text = response.content[0].text.strip()
            # Limpiar markdown
            text = re.sub(r'```json|```', '', text).strip()
            inicio = text.find("{")
            fin    = text.rfind("}")
            if inicio != -1 and fin != -1:
                text = text[inicio:fin+1]

            data = json.loads(text)

            # Rellenar todas las categorías
            for cat in CATEGORIAS_IGA:
                obligs = data.get(cat, [])
                obligs = obligs if isinstance(obligs, list) else []
                resultado["por_categoria"][cat] = obligs
                resultado["total"] += len(obligs)
                print(f"  {cat}: {len(obligs)} obligaciones")

            print(f"\nTOTAL: {resultado['total']} obligaciones extraídas")
            # Guardar en caché
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(resultado, f, ensure_ascii=False, indent=2)
            return resultado

        except json.JSONDecodeError:
            print(f"  [intento {intento+1}/3] JSON malformado, reintentando...")
            time.sleep(5)
        except Exception as e:
            msg = str(e)
            if 'overloaded' in msg.lower():
                print(f"  [Claude sobrecargado] Esperando 30s...")
                time.sleep(30)
            else:
                print(f"  [error]: {msg[:80]}")
                break

    return resultado


# ── EJECUCIÓN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    # Si el documento ya está en Pinecone, pon el DOC_ID directamente
    DOC_ID = "iga:3b0f4eb7cfe21c807aa05a56d281f56bd9dafbddd98718cda442006cce94ed17"

    resultado = extraer_obligaciones(DOC_ID)

    with open("obligaciones_iga.json", "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)

    print("\nGuardado en: obligaciones_iga.json")