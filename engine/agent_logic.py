# agent_logic.py
import os
import re
import time
import json
import logging
import hashlib
from datetime import datetime
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_community.chat_message_histories import ChatMessageHistory

from engine.rag_pipeline import retrieve_context
from engine.utils import (
    extract_model_from_query,
    is_samsung_related,
    normalize_text,
)

# ============================================
# IL3.1: OBSERVABILIDAD Y MÉTRICAS
# ============================================

# Configuración de logging para archivo y consola
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('agent_traces.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SamsungAgent")

class MetricsTracker:
    """Registra y calcula métricas del agente (IL3.1)"""
    
    def __init__(self):
        self.queries_count = 0
        self.response_times = []
        self.intents_used = []
        self.response_lengths = []
        self.models_detected = []
        self.errors_count = 0
        self.start_time = datetime.now()
    
    def register_query(self, intent, model, response_length, duration, has_error=False):
        """Registra una consulta procesada"""
        self.queries_count += 1
        self.intents_used.append(intent)
        self.response_times.append(duration)
        self.response_lengths.append(response_length)
        if model:
            self.models_detected.append(model)
        if has_error:
            self.errors_count += 1
    
    def get_metrics(self):
        """Devuelve métricas actuales"""
        if self.queries_count == 0:
            return "Sin consultas procesadas aún."
        
        avg_time = sum(self.response_times) / len(self.response_times)
        avg_length = sum(self.response_lengths) / len(self.response_lengths)
        uptime = (datetime.now() - self.start_time).total_seconds() / 3600
        
        # Intent más común
        intent_freq = Counter(self.intents_used).most_common(1)[0] if self.intents_used else ("N/A", 0)
        
        # Modelo más consultado
        model_freq = Counter(self.models_detected).most_common(1)[0] if self.models_detected else ("N/A", 0)
        
        # Tasa de error
        error_rate = (self.errors_count / self.queries_count) * 100
        
        return {
            "total_queries": self.queries_count,
            "uptime_hours": round(uptime, 2),
            "avg_response_ms": round(avg_time * 1000, 2),
            "avg_response_chars": round(avg_length, 0),
            "most_common_intent": intent_freq[0],
            "most_consulted_model": model_freq[0],
            "error_rate_pct": round(error_rate, 1),
        }
    
    def print_metrics(self):
        """Imprime métricas en formato legible"""
        m = self.get_metrics()
        if isinstance(m, str):
            print(m)
            return
        
        print("\n" + "="*50)
        print("📊 MÉTRICAS DEL AGENTE")
        print("="*50)
        print(f"Consultas totales: {m['total_queries']}")
        print(f"Tiempo activo: {m['uptime_hours']} horas")
        print(f"Tiempo promedio de respuesta: {m['avg_response_ms']} ms")
        print(f"Longitud promedio de respuesta: {m['avg_response_chars']} caracteres")
        print(f"Intención más común: {m['most_common_intent']}")
        print(f"Modelo más consultado: {m['most_consulted_model']}")
        print(f"Tasa de error: {m['error_rate_pct']}%")
        print("="*50)


# ============================================
# IL3.2: TRAZABILIDAD Y ANÁLISIS DE LOGS
# ============================================

class TraceLogger:
    """Registra trazabilidad de cada ejecución (IL3.2)"""
    
    def __init__(self):
        self.traces = []
    
    def create_trace(self, query, intent, model, response, duration, errors=None):
        """Crea un registro de trazabilidad"""
        trace = {
            "trace_id": hashlib.md5(f"{datetime.now()}{query}".encode()).hexdigest()[:10],
            "timestamp": datetime.now().isoformat(),
            "query_hash": hashlib.md5(query.encode()).hexdigest()[:8],
            "query_preview": query[:80] + "..." if len(query) > 80 else query,
            "intent": intent,
            "model_detected": model or "Ninguno",
            "response_preview": response[:80] + "..." if len(response) > 80 else response,
            "response_length": len(response),
            "duration_ms": round(duration * 1000, 2),
            "errors": errors or [],
        }
        
        self.traces.append(trace)
        logger.info(f"🔍 Traza: {json.dumps(trace, ensure_ascii=False)}")
        return trace
    
    def analyze_traces(self):
        """Analiza las trazas para encontrar patrones (IL3.2)"""
        if not self.traces:
            return "Sin trazas para analizar."
        
        # Errores más comunes
        error_traces = [t for t in self.traces if t["errors"]]
        
        # Intents con más errores
        intent_errors = {}
        for t in error_traces:
            intent = t["intent"]
            intent_errors[intent] = intent_errors.get(intent, 0) + 1
        
        # Tiempos de respuesta lentos
        slow_traces = [t for t in self.traces if t["duration_ms"] > 3000]
        
        # Consultas más frecuentes (por hash)
        query_hashes = Counter(t["query_hash"] for t in self.traces)
        repeated = {h: c for h, c in query_hashes.items() if c > 1}
        
        analysis = {
            "total_traces": len(self.traces),
            "traces_with_errors": len(error_traces),
            "intents_with_most_errors": intent_errors,
            "slow_responses": len(slow_traces),
            "repeated_queries": len(repeated),
        }
        
        logger.info(f"📈 Análisis de trazas: {json.dumps(analysis, ensure_ascii=False)}")
        return analysis
    
    def print_analysis(self):
        """Imprime análisis en formato legible"""
        a = self.analyze_traces()
        if isinstance(a, str):
            print(a)
            return
        
        print("\n" + "="*50)
        print("🔍 ANÁLISIS DE TRAZAS")
        print("="*50)
        print(f"Total de trazas: {a['total_traces']}")
        print(f"Trazas con errores: {a['traces_with_errors']}")
        print(f"Respuestas lentas (>3s): {a['slow_responses']}")
        print(f"Consultas repetidas: {a['repeated_queries']}")
        
        if a["intents_with_most_errors"]:
            print("\nIntenciones con más errores:")
            for intent, count in a["intents_with_most_errors"].items():
                print(f"  - {intent}: {count} errores")
        print("="*50)


# ============================================
# IL3.3: SEGURIDAD Y USO RESPONSABLE
# ============================================

class SecurityGuard:
    """Implementa controles de seguridad y ética (IL3.3)"""
    
    def __init__(self):
        # Palabras clave bloqueadas por razones éticas
        self.blocked_keywords = [
            "hackear", "hackeo", "contraseña de otro",
            "espiar", "rastrear sin permiso", "ilegal",
            "desbloquear imei", "bypass", "root sin permiso",
        ]
        
        # Patrones de datos personales
        self.pii_patterns = [
            (r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b', '[NÚMERO-SEGURO-SOCIAL]'),
            (r'\b\d{16}\b', '[TARJETA-CRÉDITO]'),
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),
            (r'\b\d{10}\b', '[TELÉFONO]'),
            (r'\b\d{15,16}\b', '[IMEI]'),
        ]
        
        # Patrones de inyección
        self.injection_patterns = [
            r'ignor(a|á)\s+(las?\s+)?instrucciones',
            r'system\s*prompt',
            r'haz\s+como\s+si',
            r'olvida\s+tu\s+entrenamiento',
        ]
        
        self.blocked_count = 0
        self.pii_detected_count = 0
        self.injection_attempts = 0
    
    def sanitize(self, query):
        """Elimina datos personales de la consulta"""
        sanitized = query
        had_pii = False
        
        for pattern, replacement in self.pii_patterns:
            if re.search(pattern, sanitized):
                sanitized = re.sub(pattern, replacement, sanitized)
                had_pii = True
        
        if had_pii:
            self.pii_detected_count += 1
            logger.info("🔒 Datos personales detectados y sanitizados")
        
        return sanitized, had_pii
    
    def is_safe(self, query):
        """Verifica si la consulta es segura y ética"""
        query_lower = query.lower()
        
        # 1. Verificar palabras bloqueadas
        for keyword in self.blocked_keywords:
            if keyword in query_lower:
                self.blocked_count += 1
                logger.warning(f"⚠️ Consulta bloqueada por keyword: '{keyword}'")
                return False, "No puedo ayudar con esa solicitud por razones de seguridad y ética."
        
        # 2. Verificar intentos de inyección
        for pattern in self.injection_patterns:
            if re.search(pattern, query_lower):
                self.injection_attempts += 1
                logger.warning("⚠️ Posible intento de inyección detectado")
                return False, "Lo siento, no puedo procesar esa consulta."
        
        # 3. Verificar longitud
        if len(query) > 500:
            return False, "La consulta es demasiado larga. Por favor, resúmela."
        
        return True, "OK"
    
    def get_security_stats(self):
        """Estadísticas de seguridad"""
        return {
            "consultas_bloqueadas": self.blocked_count,
            "datos_personales_detectados": self.pii_detected_count,
            "intentos_inyeccion": self.injection_attempts,
        }


# ============================================
# IL3.4: MEJORA CONTINUA Y ESCALABILIDAD
# ============================================

class ImprovementAnalyzer:
    """Analiza datos para proponer mejoras (IL3.4)"""
    
    def __init__(self, metrics: MetricsTracker, traces: TraceLogger):
        self.metrics = metrics
        self.traces = traces
    
    def generate_recommendations(self):
        """Genera recomendaciones basadas en datos observados"""
        m = self.metrics.get_metrics()
        if isinstance(m, str):
            return ["Aún no hay suficientes datos para recomendaciones."]
        
        recommendations = []
        
        # 1. Latencia
        if m["avg_response_ms"] > 3000:
            recommendations.append(
                "⚠️ Latencia alta (>3s). Considerar: reducir MAX_RESPONSE_TOKENS, "
                "usar caché para consultas frecuentes, o simplificar el SYSTEM_PROMPT."
            )
        
        # 2. Tasa de error
        if m["error_rate_pct"] > 10:
            recommendations.append(
                "⚠️ Tasa de error elevada (>10%). Revisar logs de errores y "
                "mejorar el manejo de excepciones en el pipeline."
            )
        
        # 3. Volumen de consultas
        if m["total_queries"] > 100:
            recommendations.append(
                "📈 Volumen creciente de consultas. Recomendaciones de escalabilidad:\n"
                "  - Implementar caché de embeddings para búsquedas frecuentes.\n"
                "  - Usar procesamiento asíncrono para consultas largas.\n"
                "  - Considerar balanceo de carga si se despliega en producción."
            )
        
        # 4. Consultas repetidas
        analysis = self.traces.analyze_traces()
        if isinstance(analysis, dict) and analysis.get("repeated_queries", 0) > 5:
            recommendations.append(
                f"🔄 {analysis['repeated_queries']} consultas repetidas detectadas. "
                "Implementar caché de respuestas para consultas idénticas."
            )
        
        # 5. Variedad de intenciones
        if len(set(self.metrics.intents_used)) < 3 and m["total_queries"] > 20:
            recommendations.append(
                "💡 Poca variedad de intenciones detectadas. Revisar CLASSIFIER_PROMPT "
                "para asegurar que cubre todos los casos de uso."
            )
        
        if not recommendations:
            recommendations.append("✅ El agente funciona correctamente. Sin recomendaciones por ahora.")
        
        return recommendations
    
    def print_recommendations(self):
        """Imprime recomendaciones"""
        recs = self.generate_recommendations()
        print("\n" + "="*50)
        print("💡 RECOMENDACIONES DE MEJORA")
        print("="*50)
        for i, rec in enumerate(recs, 1):
            print(f"\n{i}. {rec}")
        print("="*50)


# ============================================
# CONSTANTES
# ============================================

MAX_HISTORY_MESSAGES = 10
MAX_RETRIES = 3
MAX_RESPONSE_TOKENS = int(os.getenv("MAX_RESPONSE_TOKENS", "300"))
MAX_RESPONSE_CHARS = 1500

INTENTS_VALIDOS = {
    "soporte", "configuracion", "comparacion",
    "consulta_tecnica", "general", "fuera_contexto", "solo_modelo",
}

SYSTEM_PROMPT = """
Eres un técnico especialista en dispositivos Samsung Galaxy.

Principios:
- NUNCA pidas datos personales (nombre, email, teléfono, IMEI, contraseñas).
- NUNCA ayudes con hackeos, desbloqueos ilegales o espionaje.
- Si alguien pide algo no ético, rechaza amablemente.
- Respeta la privacidad del usuario siempre.

Reglas de respuesta:
1. Si solo mencionan un modelo, pregunta en qué necesitan ayuda.
2. Si hay un problema, da pasos CORTOS y DIRECTOS (máximo 4).
3. Si no puedes resolver en 3 intentos, ofrece derivar a un técnico humano.
4. Sé BREVE. No des información no solicitada.

Responde en español.
""".strip()

CLASSIFIER_PROMPT = """
Clasifica en UNA categoría:
- solo_modelo: Solo menciona modelo sin problema
- soporte: Problema técnico, error, fallo
- configuracion: Configurar o personalizar
- comparacion: Comparar modelos
- consulta_tecnica: Especificaciones técnicas
- general: Saludos, agradecimientos
- fuera_contexto: No relacionado con Samsung

Responde SOLO la categoría.
""".strip()

SMALL_TALK = {
    normalize_text(q) for q in [
        "hola", "gracias", "ok", "entiendo", "estas aqui", "estás aquí",
        "que paso", "qué pasó", "perfecto", "bien", "vale",
        "como estas", "cómo estás", "buenas", "adios", "adiós", "chao", "hasta luego",
    ]
}


# ============================================
# FUNCIONES AUXILIARES
# ============================================

def _is_small_talk(query_norm):
    clean = re.sub(r"[^\w\s]", "", query_norm).strip()
    if clean in SMALL_TALK:
        return True
    tokens = clean.split()
    return all(any(token in phrase for phrase in SMALL_TALK) for token in tokens)

def _is_model_only_query(query_norm):
    clean = re.sub(r"[^\w\s]", "", query_norm).strip()
    patterns = [
        r'^(samsung\s+)?galaxy\s+[asz]\d{2,3}(\s*(ultra|plus|fe|5g))?$',
        r'^(samsung\s+)?[asz]\d{2,3}(\s*(ultra|plus|fe|5g))?$',
    ]
    for p in patterns:
        if re.match(p, clean, re.IGNORECASE):
            return True
    if any(kw in clean.lower() for kw in ['galaxy', 'samsung']):
        if len(clean.split()) <= 4:
            actions = ['como', 'que', 'cual', 'ayuda', 'problema', 'error', 'no ']
            if not any(a in clean.lower() for a in actions):
                return True
    return False

def _truncate_response(response, max_chars=MAX_RESPONSE_CHARS):
    if len(response) <= max_chars:
        return response
    truncated = response[:max_chars]
    last_period = truncated.rfind('.')
    last_newline = truncated.rfind('\n')
    cut = max(last_period, last_newline)
    if cut > max_chars * 0.7:
        return response[:cut + 1]
    last_space = truncated.rfind(' ')
    return response[:last_space] + '...' if last_space > 0 else truncated + '...'


# ============================================
# AGENTE PRINCIPAL
# ============================================

class AgentOrchestrator:
    """
    Agente Samsung Galaxy con:
    - Métricas de observabilidad
    - Trazabilidad de ejecuciones
    - Seguridad y uso responsable
    - Análisis para mejora continua
    """
    
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.retry_count = 0
        
        # IL3.1: Métricas
        self.metrics = MetricsTracker()
        
        # IL3.2: Trazabilidad
        self.tracer = TraceLogger()
        
        # IL3.3: Seguridad
        self.security = SecurityGuard()
        
        # IL3.4: Mejora continua
        self.improvement = ImprovementAnalyzer(self.metrics, self.tracer)
        
        # LLMs
        self.llm = ChatOpenAI(
            model=os.getenv("GITHUB_MODEL", "gpt-4o"),
            api_key=os.getenv("GITHUB_TOKEN"),
            base_url=os.getenv("GITHUB_BASE_URL"),
            temperature=0.2,
            max_tokens=MAX_RESPONSE_TOKENS,
            streaming=True,
        )
        
        self.classifier_llm = ChatOpenAI(
            model=os.getenv("GITHUB_MODEL", "gpt-4o"),
            api_key=os.getenv("GITHUB_TOKEN"),
            base_url=os.getenv("GITHUB_BASE_URL"),
            temperature=0,
            max_tokens=10,
            streaming=False,
        )
        
        self.memory = ChatMessageHistory()
        
        logger.info("🤖 Agente inicializado")

    #SEGURIDAD
    
    def _check_security(self, query):
        """Aplica controles de seguridad a la consulta"""
        is_safe, reason = self.security.is_safe(query)
        if not is_safe:
            logger.warning(f"⚠️ Consulta bloqueada: {reason}")
            return False, reason, query
        
        sanitized, had_pii = self.security.sanitize(query)
        return True, "OK", sanitized

    #DETECCIÓN Y CLASIFICACIÓN
    
    def detect_model(self, query):
        model = extract_model_from_query(query)
        if model:
            return model
        return None

    def classify_intent(self, query):
        query_norm = normalize_text(query)
        modelo = self.detect_model(query)
        
        if _is_small_talk(query_norm):
            return {"intent": "general", "modelo_detectado": modelo, "confianza": "alta"}
        
        if modelo and _is_model_only_query(query_norm):
            return {"intent": "solo_modelo", "modelo_detectado": modelo, "confianza": "alta"}
        
        if not is_samsung_related(query):
            return {"intent": "fuera_contexto", "modelo_detectado": modelo, "confianza": "alta"}
        
        try:
            response = self.classifier_llm.invoke([
                SystemMessage(content=CLASSIFIER_PROMPT),
                HumanMessage(content=query),
            ])
            intent = response.content.strip().lower().rstrip('.')
            if intent not in INTENTS_VALIDOS:
                intent = "general"
            return {"intent": intent, "modelo_detectado": modelo, "confianza": "alta"}
        except Exception as e:
            logger.error(f"Error clasificación: {e}")
            return {"intent": "consulta_tecnica" if modelo else "general",
                    "modelo_detectado": modelo, "confianza": "baja"}

    #PLANIFICACIÓN Y EJECUCIÓN
    
    def plan_tasks(self, intent, modelo):
        if intent in ("solo_modelo", "general", "fuera_contexto"):
            return [{"paso": 1, "accion": "responder_directo"}]
        return [
            {"paso": 1, "accion": "recuperar_contexto"},
            {"paso": 2, "accion": "generar_respuesta"},
        ]

    def execute_plan(self, plan, query, intent, modelo):
        if intent == "solo_modelo":
            return self._handle_solo_modelo(modelo)
        
        if intent == "fuera_contexto":
            return "Solo ayudo con dispositivos Samsung Galaxy. ¿Tienes un Galaxy?"
        
        if intent == "general":
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + [HumanMessage(content=query)]
            return self._stream_response(messages)
        
        # Con contexto
        search_query = query
        if modelo and modelo.lower() not in query.lower():
            search_query = f"{modelo} {query}"
        
        context = retrieve_context(query=search_query, vectorstore=self.vectorstore, modelo=modelo)
        
        modelo_info = f"\nModelo: {modelo}" if modelo else ""
        user_message = f"""
            CONTEXTO: {context}
            CONSULTA: {query}{modelo_info}
            INSTRUCCIONES: Responde DIRECTO. Máximo 4 pasos. BREVE.
            """
        
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + [HumanMessage(content=user_message)]
        response = self._stream_response(messages)
        response = _truncate_response(response)
        
        self.retry_count += 1
        if self._should_escalate(response):
            response += "\n\n¿Derivo a un técnico especializado?"
        
        return response

    def _handle_solo_modelo(self, modelo):
        if modelo:
            return f"¡El {modelo} es un gran equipo! ¿En qué necesitas ayuda?"
        return "¿Sobre qué modelo Samsung necesitas ayuda?"

    def _should_escalate(self, response):
        if self.retry_count >= MAX_RETRIES:
            return True
        no_res = ["no puedo", "no tengo suficiente", "no encuentro"]
        return any(p in response.lower() for p in no_res)

    # ========== STREAMING ==========
    
    def _stream_response(self, messages):
        response_text = ""
        try:
            for chunk in self.llm.stream(messages):
                token = chunk.content
                if token and len(response_text) < MAX_RESPONSE_CHARS:
                    print(token, end="", flush=True)
                    response_text += token
            print()
        except Exception as e:
            logger.error(f"Error streaming: {e}")
            response_text = "Error al generar respuesta. ¿Reintentamos?"
        return _truncate_response(response_text)

    #HANDLER PRINCIPAL
    
    def handle_query(self, query):
        """
        Procesa la consulta completa con:
        - Seguridad
        - Métricas
        - Trazabilidad
        """
        start_time = time.time()
        errors = []
        
        #Seguridad
        is_safe, reason, query = self._check_security(query)
        if not is_safe:
            duration = time.time() - start_time
            self.metrics.register_query(
                intent="bloqueado", model="N/A",
                response_length=len(reason), duration=duration, has_error=True
            )
            self.tracer.create_trace(
                query=query, intent="bloqueado", model="N/A",
                response=reason, duration=duration, errors=["consulta_bloqueada"]
            )
            return reason
        
        #Clasificación
        try:
            query_norm = normalize_text(query.strip())
            
            if _is_model_only_query(query_norm):
                self.retry_count = 0
            
            # Expandir queries cortas
            ultimo_modelo = self._get_last_model()
            query_expanded = query
            if ultimo_modelo and not self.detect_model(query) and not _is_small_talk(query_norm):
                query_expanded = f"{ultimo_modelo} {query}"
            
            classification = self.classify_intent(query_expanded)
            intent = classification["intent"]
            modelo = classification["modelo_detectado"]
            
            if not modelo and ultimo_modelo:
                modelo = ultimo_modelo
            
            # === Ejecución ===
            plan = self.plan_tasks(intent, modelo)
            response = self.execute_plan(plan, query, intent, modelo)
            
        except Exception as e:
            errors.append(str(e))
            response = "Lo siento, ocurrió un error. ¿Puedes intentarlo de nuevo?"
            intent = "error"
            modelo = None
            logger.error(f"Error: {e}")
        
        #Registrar métricas y traza
        duration = time.time() - start_time
        
        self.metrics.register_query(
            intent=intent,
            model=modelo,
            response_length=len(response),
            duration=duration,
            has_error=bool(errors)
        )
        
        self.tracer.create_trace(
            query=query,
            intent=intent,
            model=modelo,
            response=response,
            duration=duration,
            errors=errors
        )
        
        #Actualizar memoria
        self.memory.add_user_message(query)
        self.memory.add_ai_message(response)
        
        # Mostrar métricas cada 10 consultas
        if self.metrics.queries_count % 10 == 0:
            self.metrics.print_metrics()
        
        return response

    def _get_last_model(self):
        for msg in reversed(self.memory.messages):
            if hasattr(msg, 'content'):
                detected = self.detect_model(msg.content)
                if detected:
                    return detected
        return None

    # ========== MEJORA CONTINUA ==========
    
    def get_improvement_report(self):
        """Genera informe completo de mejora"""
        print("\n" + "="*60)
        print("📊 INFORME COMPLETO DE OBSERVABILIDAD")
        print("="*60)
        
        # Métricas
        self.metrics.print_metrics()
        
        # Trazabilidad
        self.tracer.print_analysis()
        
        # Seguridad
        s = self.security.get_security_stats()
        print("\n" + "="*50)
        print("ESTADÍSTICAS DE SEGURIDAD")
        print("="*50)
        print(f"Consultas bloqueadas: {s['consultas_bloqueadas']}")
        print(f"Datos personales detectados: {s['datos_personales_detectados']}")
        print(f"Intentos de inyección: {s['intentos_inyeccion']}")
        print("="*50)
        
        # Recomendaciones
        self.improvement.print_recommendations()
        
        return "Informe completado."


# PRUEBA RÁPIDA


if __name__ == "__main__":
    print("🧪 Prueba del agente")
    print("="*50)
    
    # Demo de seguridad
    security = SecurityGuard()
    print("\nPrueba de seguridad:")
    print(f"'¿Cómo hackear?' → {security.is_safe('¿Cómo hackear un WiFi?')}")
    print(f"'Hola' → {security.is_safe('Hola')}")
    
    # Demo de métricas
    metrics = MetricsTracker()
    metrics.register_query("soporte", "Galaxy S23", 200, 0.5)
    metrics.register_query("configuracion", "Galaxy A54", 150, 0.3)
    metrics.print_metrics()
    
    print("\n✅ Módulos listos para integrar con el agente.")