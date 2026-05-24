import re
import unicodedata
import logging

logger = logging.getLogger(__name__)


# =========================================================
# NORMALIZACIÓN DE TEXTO
# =========================================================

def normalize_text(text: str) -> str:
    """
    Convierte a minúsculas y elimina tildes/diacríticos.
    Permite comparar 'batería' == 'bateria' sin distinción.
    """
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


# =========================================================
# MODELOS SOPORTADOS
# Ordenados de más específico a más general para evitar
# que "s25" matchee antes que "s25 ultra".
# =========================================================

MODELOS_SAMSUNG: dict[str, str] = {
    # S25 ULTRA primero (más específico)
    "s25 ultra":          "GALAXY S25 ULTRA",
    "galaxy s25 ultra":   "GALAXY S25 ULTRA",

    # S25 después (más general)
    "s25":                "GALAXY S25",
    "galaxy s25":         "GALAXY S25",

    # A55
    "a55":                "GALAXY A55",
    "galaxy a55":         "GALAXY A55",

    # TAB S10 ULTRA primero (más específico)
    "tab s10 ultra":      "GALAXY TAB S10 ULTRA",
    "tablet s10 ultra":   "GALAXY TAB S10 ULTRA",
    "tablet s 10 ultra":  "GALAXY TAB S10 ULTRA",

    # TAB S10 después (más general)
    "tab s10":            "GALAXY TAB S10 ULTRA",
    "tablet s10":         "GALAXY TAB S10 ULTRA",

    # WATCH
    "watch7":             "GALAXY WATCH7",
    "galaxy watch7":      "GALAXY WATCH7",

    # FOLD
    "z fold6":            "GALAXY Z FOLD6",
    "fold6":              "GALAXY Z FOLD6",
    "galaxy z fold6":     "GALAXY Z FOLD6",
}

# Ordenadas por longitud desc para que el match greedy sea correcto
_MODELOS_SORTED = sorted(
    MODELOS_SAMSUNG.items(),
    key=lambda x: len(x[0]),
    reverse=True
)


# =========================================================
# KEYWORDS DEL DOMINIO (normalizadas sin tilde)
# =========================================================

KEYWORDS_DOMINIO: list[str] = [
    # Marca y línea
    "samsung", "galaxy", "tab", "tablet",

    # Modelos
    "s25", "a55", "watch7", "fold6",

    # Funcionalidades y componentes
    "bateria",       # normalizado (sin tilde)
    "pantalla",
    "wifi",
    "bluetooth",
    "camara",        # normalizado
    "android",
    "one ui",
    "actualizacion", # normalizado
    "configurar",
    "reiniciar",
    "memoria",
    "almacenamiento",
    "carga",
    "aplicacion",    # normalizado
    "notificacion",  # normalizado
    "modo oscuro",
    "microfono",     # normalizado
    "altavoz",
    "procesador",
    "ram",
    "nfc",
    "huella",
]


# =========================================================
# EXTRAER MODELO
# Itera en orden de especificidad (más largo primero).
# =========================================================

def extract_model_from_query(query: str) -> str | None:
    query_norm = normalize_text(query)

    for keyword, nombre in _MODELOS_SORTED:
        keyword_norm = normalize_text(keyword)
        if keyword_norm in query_norm:
            logger.debug("Modelo detectado: %s → %s", keyword, nombre)
            return nombre

    return None


# =========================================================
# VALIDAR SI ES SAMSUNG
# =========================================================

def is_samsung_related(query: str) -> bool:
    query_norm = normalize_text(query)
    return any(kw in query_norm for kw in KEYWORDS_DOMINIO)


# =========================================================
# FORMATEAR PLAN
# =========================================================

def format_plan_summary(plan: list[dict]) -> str:
    n = len(plan)
    lines = [f"Plan de ejecución ({n} pasos):"]
    for step in plan:
        lines.append(f"  {step['paso']}. {step['accion']}")
    return "\n".join(lines)


# =========================================================
# LIMPIAR RESPUESTA LLM
# =========================================================

def clean_llm_response(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text
