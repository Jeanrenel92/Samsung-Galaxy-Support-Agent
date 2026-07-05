import os
import logging
import json
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

from engine.vectorstore import get_vectorstore
from engine.agent_logic import AgentOrchestrator

# CONFIGURACIÓN

load_dotenv()

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

EXIT_COMMANDS = {"salir", "adiós", "chao"}
SEPARATOR = "=" * 60

# Archivos de persistencia
HISTORIAL_FILE = Path(__file__).resolve().parent / "historial_consultas.json"
SEGURIDAD_FILE = Path(__file__).resolve().parent / "estadisticas_seguridad.json"


# FUNCIONES DE PERSISTENCIA

def cargar_historial() -> dict:
    """Carga el historial de consultas desde el archivo JSON"""
    if HISTORIAL_FILE.exists():
        try:
            with open(HISTORIAL_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, Exception):
            return {"consultas": [], "total": 0}
    return {"consultas": [], "total": 0}


def guardar_consulta(consulta: str, respuesta: str, metadata: dict = None) -> None:
    """
    Guarda una consulta en el historial para que el visualizador pueda verla.
    
    Args:
        consulta: Texto de la consulta del usuario
        respuesta: Respuesta del agente
        metadata: Información adicional (opcional)
    """
    # Cargar historial existente
    historial = cargar_historial()
    
    # Crear entrada
    entrada = {
        "timestamp": datetime.now().isoformat(),
        "consulta": consulta,
        "respuesta": respuesta[:300] + "..." if len(respuesta) > 300 else respuesta,
        "metadatos": metadata or {"fuente": "main.py"}
    }
    
    # Agregar al historial
    historial["consultas"].append(entrada)
    historial["total"] = len(historial["consultas"])
    
    # Guardar (solo las últimas 1000 consultas para no saturar)
    if historial["total"] > 1000:
        historial["consultas"] = historial["consultas"][-1000:]
        historial["total"] = 1000
    
    # Escribir al archivo
    try:
        with open(HISTORIAL_FILE, 'w', encoding='utf-8') as f:
            json.dump(historial, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Error guardando historial: {e}")


def guardar_estadisticas_seguridad(agent) -> None:
    """
    Guarda las estadísticas de seguridad al salir para que el visualizador pueda verlas.
    """
    try:
        stats = agent.security.get_security_stats()
        with open(SEGURIDAD_FILE, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"🔒 Estadísticas de seguridad guardadas: {stats}")
    except Exception as e:
        logger.error(f"Error guardando estadísticas de seguridad: {e}")


def mostrar_historial_resumen() -> None:
    """Muestra un resumen del historial al iniciar"""
    historial = cargar_historial()
    if historial["total"] > 0:
        print(f"\n📂 Historial cargado: {historial['total']} consultas previas")
        print(f"   Última consulta: {historial['consultas'][-1]['consulta'][:50]}...")


# VALIDAR ENTORNO

def validate_env() -> None:
    """Lanza SystemExit si faltan variables críticas."""
    required = ["GITHUB_TOKEN", "GITHUB_BASE_URL"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        print(
            f"\n[ERROR] Faltan variables de entorno: {', '.join(missing)}\n"
            "Verifica tu archivo .env antes de continuar."
        )
        raise SystemExit(1)



# MAIN

def main() -> None:

    print(SEPARATOR)
    print("  Samsung AI Agent — Asistente Virtual de Soporte Técnico")
    print(SEPARATOR)

    # ── Mostrar historial
    mostrar_historial_resumen()

    # ── Validar entorno
    validate_env()

    # ── Cargar vectorstore
    try:
        vectorstore = get_vectorstore()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("Ejecuta primero:  python -m engine.vectorstore\n")
        raise SystemExit(1)
    except Exception as e:
        logger.error("Error cargando vectorstore: %s", e, exc_info=True)
        print(f"\n[ERROR] No se pudo cargar el vectorstore:\n{e}\n")
        raise SystemExit(1)

    # ── Inicializar agente
    try:
        agent = AgentOrchestrator(vectorstore)
    except Exception as e:
        logger.error("Error inicializando agente: %s", e, exc_info=True)
        print(f"\n[ERROR] No se pudo inicializar el agente:\n{e}\n")
        raise SystemExit(1)

    print("Hola! Soy tu asistente virtual de Samsung.")
    print("¿Cómo puedo ayudarte con tu dispositivo?")
    print(f"(Escribe '{'/'.join(EXIT_COMMANDS)}' para terminar)")
    print("=" * 60)

    # ── Chat loop
    while True:

        try:
            query = input("\nTú: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nCerrando agente. ¡Hasta pronto!")
            # Guardar estadísticas al cerrar con Ctrl+C
            guardar_estadisticas_seguridad(agent)
            break

        if not query:
            continue

        if query.lower() in EXIT_COMMANDS:
            print("\n¡Hasta pronto!")
            
            # ── GUARDAR ESTADÍSTICAS DE SEGURIDAD ──
            guardar_estadisticas_seguridad(agent)
            break

        print("\nAgente:")

        try:
            # Ejecutar consulta
            respuesta = agent.handle_query(query)
            
            # ── GUARDAR EN HISTORIAL ──
            guardar_consulta(query, respuesta)
            
        except Exception as e:
            logger.error("Error procesando consulta: %s", e, exc_info=True)
            print(f"\n[ERROR] No se pudo procesar la consulta:\n{e}")
            # Guardar el error también
            guardar_consulta(query, f"[ERROR] {e}", {"error": True})

        print(f"\n{SEPARATOR}")


# ENTRYPOINT
if __name__ == "__main__":
    main()