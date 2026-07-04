import os
import logging

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

#MAIN

def main() -> None:

    print(SEPARATOR)
    print("  Samsung AI Agent — Asistente Virtual de Soporte Técnico")
    print(SEPARATOR)

    # ── Validar entorno
    validate_env()

    # ── Cargar vectorstore (usa get_vectorstore: carga si existe, genera si no, sin duplicar embeddings)
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

    #Inicializar agente
    try:
        agent = AgentOrchestrator(vectorstore)
    except Exception as e:
        logger.error("Error inicializando agente: %s", e, exc_info=True)
        print(f"\n[ERROR] No se pudo inicializar el agente:\n{e}\n")
        raise SystemExit(1)

    print("\n[Agente inicializado correctamente.]")
    print("Hola! Soy tu asistente virtual de Samsung.\n ¿Cómo puedo ayudarte con tu dispositivo?")
    #print(f"(Escribe '{'/'.join(EXIT_COMMANDS)}' para terminar)\n")

    #Chat loop
    while True:

        try:
            query = input("Tú: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nCerrando agente. ¡Hasta pronto!")
            break

        if not query:
            continue

        if query.lower() in EXIT_COMMANDS:
            print("\n¡Hasta pronto!")
            break

        print("\nAgente:\n")
        try:
            agent.handle_query(query)
        except Exception as e:
            logger.error("Error procesando consulta: %s", e, exc_info=True)
            print(f"\n[ERROR] No se pudo procesar la consulta:\n{e}")

        print(f"\n{SEPARATOR}\n")



# ENTRYPOINT
if __name__ == "__main__":
    main()
