# agent_logic.py
import os
import re
import time
import json
import logging
import hashlib
from datetime import datetime
from collections import Counter

import pandas as pd

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_community.chat_message_histories import ChatMessageHistory

from engine.rag_pipeline import retrieve_context
from engine.utils import (
    extract_model_from_query,
    is_samsung_related,
    normalize_text,
)
# CONFIGURACIÓN DE LOGGING

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('agent_traces.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SamsungAgent")

# SISTEMA DE TRAZABILIDAD PASO A PASO
class StepTracer:
    """
    Registra cada paso de ejecución del agente en formato tabla.
    Muestra: step, tipo, input, output, latencia, tokens, estado
    """
    
    def __init__(self):
        self.executions = []
        self.current_execution = []
        self.current_query = ""
    
    def start_execution(self, query: str):
        """Inicia una nueva ejecución para una consulta"""
        self.current_query = query
        self.current_execution = []
    
    def add_step(self, step: int, tipo: str, input_text: str, output_text: str, 
                 latency: float, tokens: int = 0, status: str = "ok"):
        """Agrega un paso a la ejecución actual"""
        # Truncar textos largos
        input_truncado = input_text[:80] + "..." if len(input_text) > 80 else input_text
        output_truncado = output_text[:80] + "..." if len(output_text) > 80 else output_text
        
        paso = {
            "step": step,
            "tipo": tipo,
            "input": input_truncado,
            "output": output_truncado,
            "latencia": f"{latency:.2f}s",
            "tokens": tokens,
            "estado": status,
        }
        self.current_execution.append(paso)
    
    def end_execution(self):
        """Finaliza la ejecución actual y la guarda"""
        if self.current_execution:
            self.executions.append({
                "query": self.current_query,
                "steps": self.current_execution.copy(),
                "timestamp": datetime.now().isoformat(),
            })
        self.current_query = ""
        self.current_execution = []
    
    def print_execution(self, query: str = None):
        """Imprime la trazabilidad en formato tabla"""
        executions_to_show = self.executions
        if query:
            executions_to_show = [e for e in self.executions if query.lower() in e["query"].lower()]
        
        for execution in executions_to_show:
            print(f"\n=== Ejecución para pregunta: {execution['query']}")
            print(f"{'step':<6} {'tipo':<14} {'input':<45} {'output':<45} {'latencia':<10} {'tokens':<8} {'estado':<8}")
            print("-" * 150)
            
            for step in execution["steps"]:
                print(f"{step['step']:<6} {step['tipo']:<14} {step['input']:<45} {step['output']:<45} {step['latencia']:<10} {step['tokens']:<8} {step['estado']:<8}")
            
            print("-" * 150)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convierte todas las ejecuciones a un DataFrame"""
        all_steps = []
        for execution in self.executions:
            for step in execution["steps"]:
                step_with_query = step.copy()
                step_with_query["query"] = execution["query"]
                all_steps.append(step_with_query)
        
        if not all_steps:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_steps)
        return df[["query", "step", "tipo", "input", "output", "latencia", "tokens", "estado"]]
    
    def save_to_csv(self, filename: str = "trazabilidad_agente.csv"):
        """Guarda la trazabilidad en archivo CSV"""
        df = self.to_dataframe()
        if not df.empty:
            df.to_csv(filename, index=False, encoding='utf-8')
            print(f"✅ Trazabilidad guardada en '{filename}'")
        else:
            print("⚠️ No hay datos para guardar")

# MÉTRICAS DE OBSERVABILIDAD
class MetricsTracker:
    """Registra y calcula métricas de rendimiento del agente"""
    
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
        
        intent_freq = Counter(self.intents_used).most_common(1)[0] if self.intents_used else ("N/A", 0)
        model_freq = Counter(self.models_detected).most_common(1)[0] if self.models_detected else ("N/A", 0)
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

# SEGURIDAD Y USO RESPONSABLE
class SecurityGuard:
    """Implementa controles de seguridad, privacidad y ética"""
    
    def __init__(self):
        self.blocked_keywords = [
            "hackear", "hackeo", "contraseña de otro",
            "espiar", "rastrear sin permiso", "ilegal",
            "desbloquear imei", "bypass", "root sin permiso",
        ]
        
        self.pii_patterns = [
            (r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b', '[NÚMERO-SEGURO-SOCIAL]'),
            (r'\b\d{16}\b', '[TARJETA-CRÉDITO]'),
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),
            (r'\b\d{10}\b', '[TELÉFONO]'),
            (r'\b\d{15,16}\b', '[IMEI]'),
        ]
        
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
        
        for keyword in self.blocked_keywords:
            if keyword in query_lower:
                self.blocked_count += 1
                logger.warning(f"⚠️ Consulta bloqueada por keyword: '{keyword}'")
                return False, "No puedo ayudar con esa solicitud por razones de seguridad y ética."
        
        for pattern in self.injection_patterns:
            if re.search(pattern, query_lower):
                self.injection_attempts += 1
                logger.warning("⚠️ Posible intento de inyección detectado")
                return False, "Lo siento, no puedo procesar esa consulta."
        
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

# ANÁLISIS PARA MEJORA CONTINUA
class ImprovementAnalyzer:
    """Analiza datos para proponer mejoras basadas en evidencia"""
    
    def __init__(self, metrics: MetricsTracker, tracer: StepTracer):
        self.metrics = metrics
        self.tracer = tracer
    
    def generate_recommendations(self):
        """Genera recomendaciones basadas en datos observados"""
        m = self.metrics.get_metrics()
        if isinstance(m, str):
            return ["Aún no hay suficientes datos para recomendaciones."]
        
        recommendations = []
        
        if m["avg_response_ms"] > 3000:
            recommendations.append(
                "⚠️ Latencia alta (>3s). Considerar: reducir MAX_RESPONSE_TOKENS, "
                "usar caché para consultas frecuentes, o simplificar el SYSTEM_PROMPT."
            )
        
        if m["error_rate_pct"] > 10:
            recommendations.append(
                "⚠️ Tasa de error elevada (>10%). Revisar logs de errores y "
                "mejorar el manejo de excepciones en el pipeline."
            )
        
        if m["total_queries"] > 100:
            recommendations.append(
                "📈 Volumen creciente de consultas. Recomendaciones de escalabilidad:\n"
                "  - Implementar caché de embeddings para búsquedas frecuentes.\n"
                "  - Usar procesamiento asíncrono para consultas largas.\n"
                "  - Considerar balanceo de carga si se despliega en producción."
            )
        
        # Analizar trazas para consultas repetidas
        df = self.tracer.to_dataframe()
        if not df.empty:
            repeated = df['query'].value_counts()
            repeated_count = (repeated > 1).sum()
            if repeated_count > 5:
                recommendations.append(
                    f"🔄 {repeated_count} consultas repetidas detectadas. "
                    "Implementar caché de respuestas para consultas idénticas."
                )
        
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

# CONSTANTES

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

# FUNCIONES AUXILIARES

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

# AGENTE PRINCIPAL CON TRAZABILIDAD
class AgentOrchestrator:
    """
    Agente Samsung Galaxy con:
    - Trazabilidad paso a paso (StepTracer)
    - Métricas de observabilidad (MetricsTracker)
    - Seguridad y uso responsable (SecurityGuard)
    - Análisis para mejora continua (ImprovementAnalyzer)
    """
    
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.retry_count = 0
        
        # Sistemas de observabilidad
        self.step_tracer = StepTracer()
        self.metrics = MetricsTracker()
        self.security = SecurityGuard()
        self.improvement = ImprovementAnalyzer(self.metrics, self.step_tracer)
        
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
        logger.info("🤖 Agente inicializado con trazabilidad paso a paso")

    # ========== SEGURIDAD ==========
    
    def _check_security(self, query):
        """Aplica controles de seguridad"""
        is_safe, reason = self.security.is_safe(query)
        if not is_safe:
            return False, reason, query
        sanitized, had_pii = self.security.sanitize(query)
        return True, "OK", sanitized

    # ========== DETECCIÓN Y CLASIFICACIÓN ==========
    
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

    # ========== PLANIFICACIÓN Y EJECUCIÓN ==========
    
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
            return f"¡El {modelo} es un gran equipo! 😊 ¿En qué necesitas ayuda?"
        return "¿Sobre qué modelo Samsung necesitas ayuda?"

    def _should_escalate(self, response):
        if self.retry_count >= MAX_RETRIES:
            return True
        no_res = ["no puedo", "no tengo suficiente", "no encuentro"]
        return any(p in response.lower() for p in no_res)

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

    # ========== HANDLER PRINCIPAL CON TRAZABILIDAD ==========
    
    def handle_query(self, query: str) -> str:
        """
        Procesa la consulta completa registrando cada paso.
        """
        total_start = time.time()
        step_counter = 0
        errors = []
        
        # Iniciar trazabilidad
        self.step_tracer.start_execution(query)
        
        # ===== PASO 0: SEGURIDAD =====
        t_start = time.time()
        is_safe, reason, sanitized_query = self._check_security(query)
        latency = time.time() - t_start
        
        self.step_tracer.add_step(
            step=step_counter,
            tipo="security",
            input_text=f"Verificar seguridad: {query[:60]}...",
            output_text=reason,
            latency=latency,
            tokens=0,
            status="ok" if is_safe else "error"
        )
        step_counter += 1
        
        if not is_safe:
            duration = time.time() - total_start
            self.metrics.register_query(
                intent="bloqueado", model="N/A",
                response_length=len(reason), duration=duration, has_error=True
            )
            self.step_tracer.end_execution()
            return reason
        
        query = sanitized_query
        
        try:
            query_norm = normalize_text(query.strip())
            
            if _is_model_only_query(query_norm):
                self.retry_count = 0
            
            # ===== PASO 1: DETECCIÓN DE MODELO =====
            t_start = time.time()
            modelo = self.detect_model(query)
            latency = time.time() - t_start
            
            self.step_tracer.add_step(
                step=step_counter,
                tipo="tool",
                input_text=f"Detectar modelo: {query[:60]}...",
                output_text=modelo or "Ninguno",
                latency=latency,
                tokens=0,
                status="ok"
            )
            step_counter += 1
            
            # Recuperar último modelo del historial
            ultimo_modelo = self._get_last_model()
            query_expanded = query
            if ultimo_modelo and not modelo and not _is_small_talk(query_norm):
                query_expanded = f"{ultimo_modelo} {query}"
                modelo = ultimo_modelo
            
            # ===== PASO 2: CLASIFICACIÓN =====
            t_start = time.time()
            classification = self.classify_intent(query_expanded)
            latency = time.time() - t_start
            intent = classification["intent"]
            
            if not modelo and ultimo_modelo:
                modelo = ultimo_modelo
            
            self.step_tracer.add_step(
                step=step_counter,
                tipo="classification",
                input_text=f"Clasificar intención: {query_expanded[:60]}...",
                output_text=f"Intención: {intent} | Modelo: {modelo or 'N/A'} | Confianza: {classification['confianza']}",
                latency=latency,
                tokens=0,
                status="ok"
            )
            step_counter += 1
            
            # ===== PASO 3: RECUPERACIÓN DE CONTEXTO (si aplica) =====
            plan = self.plan_tasks(intent, modelo)
            context_chars = 0
            
            if intent not in ("solo_modelo", "general", "fuera_contexto"):
                t_start = time.time()
                search_query = query
                if modelo and modelo.lower() not in query.lower():
                    search_query = f"{modelo} {query}"
                
                context = retrieve_context(
                    query=search_query,
                    vectorstore=self.vectorstore,
                    modelo=modelo,
                )
                latency = time.time() - t_start
                context_chars = len(context)
                
                self.step_tracer.add_step(
                    step=step_counter,
                    tipo="retrieval",
                    input_text=f"Buscar en FAISS: {search_query[:60]}...",
                    output_text=f"Recuperados {context_chars} caracteres ({context.count(chr(10))+1} fragmentos)",
                    latency=latency,
                    tokens=0,
                    status="ok"
                )
                step_counter += 1
            
            # ===== PASO 4: GENERACIÓN DE RESPUESTA (LLM) =====
            t_start = time.time()
            response = self.execute_plan(plan, query, intent, modelo)
            latency = time.time() - t_start
            estimated_tokens = len(response.split())
            
            self.step_tracer.add_step(
                step=step_counter,
                tipo="llm",
                input_text=f"Generar respuesta para: {query[:60]}...",
                output_text=response[:80] + "..." if len(response) > 80 else response,
                latency=latency,
                tokens=estimated_tokens,
                status="ok"
            )
            step_counter += 1
            
        except Exception as e:
            errors.append(str(e))
            response = "Lo siento, ocurrió un error. ¿Puedes intentarlo de nuevo?"
            intent = "error"
            modelo = None
            logger.error(f"Error: {e}")
        
        # ===== FINALIZAR TRAZABILIDAD =====
        duration = time.time() - total_start
        self.step_tracer.end_execution()
        
        # Registrar métricas
        self.metrics.register_query(
            intent=intent,
            model=modelo,
            response_length=len(response),
            duration=duration,
            has_error=bool(errors)
        )
        
        # Actualizar memoria
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

    # ========== MÉTODOS DE OBSERVABILIDAD ==========
    
    def mostrar_trazabilidad(self, query: str = None):
        """Muestra la tabla de trazabilidad paso a paso"""
        self.step_tracer.print_execution(query)
    
    def guardar_trazabilidad(self, filename: str = "trazabilidad_agente.csv"):
        """Guarda la trazabilidad en archivo CSV"""
        self.step_tracer.save_to_csv(filename)
    
    def get_trazabilidad_dataframe(self) -> pd.DataFrame:
        """Retorna la trazabilidad como DataFrame de pandas"""
        return self.step_tracer.to_dataframe()
    
    def get_improvement_report(self):
        """Genera informe completo de rendimiento y mejoras"""
        print("\n" + "="*60)
        print("📊 INFORME COMPLETO DEL AGENTE")
        print("="*60)
        
        self.metrics.print_metrics()
        
        s = self.security.get_security_stats()
        print("\n" + "="*50)
        print("🔒 ESTADÍSTICAS DE SEGURIDAD")
        print("="*50)
        print(f"Consultas bloqueadas: {s['consultas_bloqueadas']}")
        print(f"Datos personales detectados: {s['datos_personales_detectados']}")
        print(f"Intentos de inyección: {s['intentos_inyeccion']}")
        print("="*50)
        
        self.improvement.print_recommendations()
        
        # Mostrar tabla de trazabilidad
        print("\n" + "="*60)
        print("🔍 TRAZABILIDAD DE EJECUCIONES")
        print("="*60)
        self.step_tracer.print_execution()
        
        return "Informe completado."


# PRUEBA RÁPIDA

if __name__ == "__main__":
    print("🧪 Prueba del agente con trazabilidad paso a paso")
    print("="*60)
    
    # Demo de trazabilidad
    tracer = StepTracer()
    
    tracer.start_execution("¿Cómo es la cámara del Galaxy S23?")
    tracer.add_step(0, "security", "Verificar seguridad", "OK", 0.01, 0, "ok")
    tracer.add_step(1, "tool", "Detectar modelo", "Galaxy S23", 0.02, 0, "ok")
    tracer.add_step(2, "classification", "Clasificar intención", "consulta_tecnica", 0.15, 0, "ok")
    tracer.add_step(3, "retrieval", "Buscar en FAISS", "Recuperados 450 caracteres", 0.05, 0, "ok")
    tracer.add_step(4, "llm", "Generar respuesta", "El Galaxy S23 tiene cámara de 50MP con OIS...", 0.45, 25, "ok")
    tracer.end_execution()
    
    tracer.start_execution("¿Cómo hackear un WiFi?")
    tracer.add_step(0, "security", "Verificar seguridad", "Bloqueada: keyword 'hackear'", 0.01, 0, "error")
    tracer.end_execution()
    
    tracer.print_execution()
    
    print("\n✅ Sistema de trazabilidad listo para integrar.")