import os
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_community.chat_message_histories import ChatMessageHistory

from engine.rag_pipeline import retrieve_context
from engine.utils import (
    extract_model_from_query,
    is_samsung_related,
    normalize_text,
)

logger = logging.getLogger(__name__)

# CONSTANTES


MAX_HISTORY_MESSAGES = 10   # Ventana deslizante: últimos mensajes

INTENTS_VALIDOS = {
    "soporte",
    "configuracion",
    "comparacion",
    "consulta_tecnica",
    "general",
    "fuera_contexto",
}

# SYSTEM PROMPT
SYSTEM_PROMPT = """
Eres un asistente técnico especializado en dispositivos Samsung Galaxy.

PRIORIDAD DE INFORMACIÓN
Usa principalmente la información disponible en el CONTEXTO recuperado desde la base de conocimiento.
Si el contexto incluye información sobre un modelo o procedimiento, priorízala en tu respuesta.
Evita contradecir la información del contexto.

REGLAS GENERALES
Responde de forma clara, útil y directa.
No inventes información técnica.
Si falta información en el contexto, puedes apoyarte en conocimiento general o fuentes externas confiables para consultas relacionadas con dispositivos Samsung Galaxy.

Cuando uses información externa, indícalo diciendo:
"Esta información no se encuentra en mi base de conocimiento interna, pero fue obtenida desde fuentes externas confiables."

Si no encuentras información suficiente, responde:
"No he podido encontrar la información disponible para responder completamente la consulta."

Si no es posible identificar una solución clara al problema, responde:
"Tu consulta será transferida a servicio técnico."

ESTILO DE RESPUESTA
- Responde siempre en español.
- Sé profesional y técnico.
- Mantén respuestas simples y enfocadas.
- Evita agregar información innecesaria.
- No menciones otros dispositivos salvo que sea relevante.
""".strip()

CLASSIFIER_PROMPT = """
Eres un clasificador de intenciones para un asistente de soporte Samsung Galaxy.

Clasifica la consulta del usuario en UNA de estas categorías (responde SOLO la palabra):
- soporte         → problemas, fallas, errores, mal funcionamiento
- configuracion   → activar, desactivar, ajustar, habilitar, cambiar configuraciones
- comparacion     → comparar modelos, diferencias, cuál es mejor
- consulta_tecnica → preguntar por especificaciones, características, precios
- general         → saludos, despedidas, small talk
- fuera_contexto  → temas no relacionados con Samsung Galaxy

Responde ÚNICAMENTE con una de esas seis palabras, sin puntuación ni explicación.
""".strip()

# Frases de small talk que se resuelven sin invocar el pipeline RAG
SMALL_TALK = {
    normalize_text(q) for q in [
        "hola", "gracias", "ok", "entiendo",
        "estas aqui", "estás aquí",
        "que paso", "qué pasó",
        "perfecto", "bien", "vale",
        "como estas", "cómo estás", "buenas",
        "adios", "adiós", "chao", "hasta luego",
    ]
}


def _is_small_talk(query_norm: str) -> bool:
    """
    Detecta small talk eliminando puntuación antes de comparar.
    Acepta frases exactas O queries compuestas solo de tokens small talk.
    Ejemplos que deben ser True:
      "hola, como estas?"  →  "hola como estas"  → contiene "hola" y "como estas"
      "¡buenas!"           →  "buenas"            → match exacto
    """
    import re
    clean = re.sub(r"[^\w\s]", "", query_norm).strip()

    # Match exacto (query es solo una frase de small talk)
    if clean in SMALL_TALK:
        return True

    # Query compuesta: todos los tokens pertenecen a frases de small talk
    tokens = clean.split()
    return all(
        any(token in phrase for phrase in SMALL_TALK)
        for token in tokens
    )



# AGENT ORCHESTRATOR
class AgentOrchestrator:

    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

        # LLM PRINCIPAL (streaming)
        self.llm = ChatOpenAI(
            model=os.getenv("GITHUB_MODEL", "gpt-4o"),
            api_key=os.getenv("GITHUB_TOKEN"),
            base_url=os.getenv("GITHUB_BASE_URL"),
            temperature=0.2,
            streaming=True,
        )

        # LLM CLASIFICADOR (sin streaming, determinista)
        self.classifier_llm = ChatOpenAI(
            model=os.getenv("GITHUB_MODEL", "gpt-4o"),
            api_key=os.getenv("GITHUB_TOKEN"),
            base_url=os.getenv("GITHUB_BASE_URL"),
            temperature=0,
            streaming=False,
        )

        self.memory = ChatMessageHistory()

    #DETECT MODEL

    def detect_model(self, query: str) -> str | None:
        return extract_model_from_query(query)

    #CLASSIFY INTENT

    def classify_intent(self, query: str) -> dict:
        """
        Clasifica la intención usando el LLM clasificador.
        Fallback a 'general' si la respuesta no es un intent válido.
        """
        query_norm = normalize_text(query)
        modelo_detectado = self.detect_model(query)

        if _is_small_talk(query_norm):
            return {
                "intent": "general",
                "modelo_detectado": modelo_detectado,
                "confianza": "alta",
            }

        #Fuera del dominio Samsung
        if not is_samsung_related(query):
            return {
                "intent": "fuera_contexto",
                "modelo_detectado": None,
                "confianza": "alta",
            }

        #Clasificación semántica con el LLM
        try:
            response = self.classifier_llm.invoke([
                SystemMessage(content=CLASSIFIER_PROMPT),
                HumanMessage(content=query),
            ])
            intent = response.content.strip().lower()

            # Validar que el LLM devolvió un intent conocido
            if intent not in INTENTS_VALIDOS:
                logger.warning(
                    "Clasificador devolvió intent desconocido '%s'. "
                    "Usando fallback 'general'.", intent
                )
                intent = "general"

            return {
                "intent": intent,
                "modelo_detectado": modelo_detectado,
                "confianza": "alta",
            }

        except Exception as e:
            logger.error(
                "Error en classify_intent LLM: %s. Usando fallback.", e,
                exc_info=True
            )
            # Fallback determinista si el LLM falla
            return {
                "intent": "consulta_tecnica" if modelo_detectado else "general",
                "modelo_detectado": modelo_detectado,
                "confianza": "baja",
            }

    #PLAN TASKS

    def plan_tasks(self, intent: str, modelo: str | None) -> list[dict]:
        """
        Genera el plan de ejecución según la intención clasificada.
        - general / fuera_contexto: sin recuperación de contexto.
        - resto: recuperar_contexto + generar_respuesta.
        """
        if intent in ("general", "fuera_contexto"):
            return [{"paso": 1, "accion": "generar_respuesta"}]

        return [
            {"paso": 1, "accion": "recuperar_contexto"},
            {"paso": 2, "accion": "generar_respuesta"},
        ]

    #EXECUTE PLAN

    def execute_plan(
        self,
        plan: list[dict],
        query: str,
        intent: str,
        modelo: str | None = None,
    ) -> str:

        logger.info("[Plan: %d pasos]", len(plan))
        for step in plan:
            logger.info("→ Paso %d: %s", step["paso"], step["accion"])

        #FUERA DE DOMINIO
        if intent == "fuera_contexto":
            return "Solo puedo ayudarte con dispositivos Samsung Galaxy."

        #GENERAL / SMALL TALK
        if intent == "general":
            messages = self._build_messages(query)
            return self._stream_response(messages)

        #RECUPERAR CONTEXTO
        context = retrieve_context(
            query=query,
            vectorstore=self.vectorstore,
            modelo=modelo,
        )

        user_message = (
            f"CONTEXTO:\n{context}\n\n"
            f"CONSULTA:\n{query}\n\n"
            "REGLAS:\n"
            "- Usa primero la información del CONTEXTO.\n"
            "- Responde solo lo necesario.\n"
            "- No agregues información de otros dispositivos.\n"
            "- Si no existe información suficiente, dilo claramente.\n"
            "- Mantén respuestas técnicas pero simples."
        )

        messages = self._build_messages(user_message)
        return self._stream_response(messages)

    #BUILD MESSAGES

    def _build_messages(self, user_content: str) -> list[BaseMessage]:
        """
        Construye la lista de mensajes para el LLM:
        [SystemMessage] + últimos MAX_HISTORY_MESSAGES + [HumanMessage actual]

        La ventana deslizante evita que el contexto crezca indefinidamente.
        """
        history = self.memory.messages[-MAX_HISTORY_MESSAGES:]
        return (
            [SystemMessage(content=SYSTEM_PROMPT)]
            + history
            + [HumanMessage(content=user_content)]
        )

    #STREAM RESPONSE

    def _stream_response(self, messages: list[BaseMessage]) -> str:
        response_text = ""
        try:
            for chunk in self.llm.stream(messages):
                token = chunk.content
                if token:
                    print(token, end="", flush=True)
                    response_text += token
            print()
        except Exception as e:
            logger.error("Error en _stream_response: %s", e, exc_info=True)
            response_text = (
                "Lo siento, ocurrió un error al generar la respuesta. "
                "Por favor, intenta de nuevo."
            )
            print(f"\n[ERROR]: {e}")
        return response_text

    #HANDLE QUERY

    def handle_query(self, query: str) -> str:

        #Recuperar último modelo mencionado en historial
        ultimo_modelo = None
        for msg in reversed(self.memory.messages):
            detected = self.detect_model(msg.content)
            if detected:
                ultimo_modelo = detected
                break

        #Expandir consultas cortas con el modelo del contexto
        query_norm = normalize_text(query.strip())

        SHORT_QUERIES = {
            normalize_text(q) for q in [
                "camara", "la camara",
                "bateria", "la bateria",
                "pantalla", "procesador",
                "ram", "almacenamiento",
                "y la camara", "y la bateria",
            ]
        }

        query_for_classification = query
        if ultimo_modelo and query_norm in SHORT_QUERIES:
            query_for_classification = f"{ultimo_modelo} {query}"
            logger.debug(
                "Query expandida: '%s'", query_for_classification
            )

        #Clasificación
        classification = self.classify_intent(query_for_classification)
        intent   = classification["intent"]
        modelo   = classification["modelo_detectado"]
        confianza = classification["confianza"]

        print(
            f"\n[Intención: {intent}] "
            f"[Modelo: {modelo}] "
            f"[Confianza: {confianza}]"
        )

        #Planificación
        plan = self.plan_tasks(intent, modelo)

        #Ejecución
        response = self.execute_plan(plan, query, intent, modelo)

        #Actualizar memoria
        self.memory.add_user_message(query)
        self.memory.add_ai_message(response)

        return response