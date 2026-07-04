import re
import unicodedata
import logging

logger = logging.getLogger(__name__)



# NORMALIZACIÓN DE TEXTO

def normalize_text(text: str) -> str:
    """
    Normaliza texto para facilitar comparaciones:
    - Minúsculas
    - Elimina tildes
    - Reemplaza separadores comunes
    - Elimina puntuación
    - Colapsa espacios
    """
    text = text.lower()

    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        c for c in text
        if not unicodedata.combining(c)
    )

    # Separadores comunes
    text = (
        text.replace("-", " ")
            .replace("_", " ")
            .replace("/", " ")
    )

    # Eliminar puntuación (manteniendo espacios)
    text = re.sub(r"[^\w\s+]", " ", text)

    # Espacios múltiples
    text = re.sub(r"\s+", " ", text)

    return text.strip()

MODELOS_SAMSUNG = {

    # ---------- S25 Ultra ----------
    "s25 ultra":          "GALAXY S25 ULTRA",
    "s 25 ultra":         "GALAXY S25 ULTRA",
    "galaxy s25 ultra":   "GALAXY S25 ULTRA",
    "galaxy s 25 ultra":  "GALAXY S25 ULTRA",
    "samsung s25 ultra":  "GALAXY S25 ULTRA",

    # ---------- S25 ----------
    "s25":               "GALAXY S25",
    "s 25":              "GALAXY S25",
    "galaxy s25":        "GALAXY S25",
    "galaxy s 25":       "GALAXY S25",
    "samsung s25":       "GALAXY S25",

    # ---------- A55 ----------
    "a55":               "GALAXY A55",
    "a 55":              "GALAXY A55",
    "galaxy a55":        "GALAXY A55",
    "galaxy a 55":       "GALAXY A55",

    # ---------- Tab ----------
    "tab s10 ultra":     "GALAXY TAB S10 ULTRA",
    "tab s 10 ultra":    "GALAXY TAB S10 ULTRA",
    "tablet s10 ultra":  "GALAXY TAB S10 ULTRA",

    "tab s10":           "GALAXY TAB S10",
    "tab s 10":          "GALAXY TAB S10",

    # ---------- Fold ----------
    "fold6":             "GALAXY Z FOLD6",
    "fold 6":            "GALAXY Z FOLD6",
    "z fold6":           "GALAXY Z FOLD6",
    "z fold 6":          "GALAXY Z FOLD6",

    # ---------- Flip ----------
    "flip6":             "GALAXY Z FLIP6",
    "flip 6":            "GALAXY Z FLIP6",
    "z flip6":           "GALAXY Z FLIP6",
    "z flip 6":          "GALAXY Z FLIP6",

    # ---------- Watch ----------
    "watch7":            "GALAXY WATCH7",
    "watch 7":           "GALAXY WATCH7",
    "galaxy watch7":     "GALAXY WATCH7",
}

# Ordenadas por longitud desc para que el match greedy sea correcto
_MODELOS_SORTED = sorted(
     (
        (normalize_text(alias), canonical)
        for alias, canonical in MODELOS_SAMSUNG.items()
    ),
    key=lambda x: len(x[0]),
    reverse=True,
)


# KEYWORDS DEL DOMINIO (normalizadas sin tilde)

KEYWORDS_DOMINIO: list[str] = [
    "oneui",

    "one ui",

    "galaxy ai",

    "bixby",

    "knox",

    "dex",

    "smart switch",

    "quick share",

    "good lock",

    "wireless powershare",

    "always on display",

    "circle to search",

    "s pen",

    "modo recuperacion",

    "download mode",

    "recovery",

    "factory reset",

    "galeria",

    "galaxy store",

    "play store",

]


def extract_model_from_query(query: str) -> str | None:
    """
    Detecta el modelo Samsung mencionado en la consulta.
    Prioriza los modelos más específicos.
    """

    query_norm = normalize_text(query)

    for keyword, nombre in _MODELOS_SORTED:

        keyword_norm = normalize_text(keyword)

        pattern = rf"\b{re.escape(keyword_norm)}\b"

        if re.search(pattern, query_norm):

            logger.debug(
                "Modelo detectado: '%s' -> %s",
                keyword,
                nombre,
            )

            return nombre

    logger.debug("No se detectó ningún modelo.")

    return None



def is_samsung_related(query: str) -> bool:
    """
    Determina si la consulta pertenece al dominio Samsung.
    """

    if extract_model_from_query(query):
        return True

    query_norm = normalize_text(query)

    return any(
        kw in query_norm
        for kw in KEYWORDS_DOMINIO
    )



def format_plan_summary(plan: list[dict]) -> str:
    n = len(plan)
    lines = [f"Plan de ejecución ({n} pasos):"]
    for step in plan:
        lines.append(f"  {step['paso']}. {step['accion']}")
    return "\n".join(lines)


# LIMPIAR RESPUESTA LLM

def clean_llm_response(text: str) -> str:

    text = text.strip()

    text = re.sub(r"\n{3,}", "\n\n", text)

    text = re.sub(r"[ \t]+", " ", text)

    text = re.sub(r" +\n", "\n", text)

    return text
