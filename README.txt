#Samsung Galaxy Support Agent

Agente conversacional de soporte técnico especializado en dispositivos Samsung Galaxy, construido con **LangChain**, **FAISS** y **GPT-4o** mediante un pipeline **RAG** (Retrieval-Augmented Generation).

---

## 📐 Arquitectura General

```
Usuario
  │
  ▼
handle_query()          ← AgentOrchestrator (agent_logic.py)
  │
  ├─ detect_model()     ← utils.py  (extrae modelo Samsung de la query)
  ├─ is_samsung_related() ← utils.py (valida que sea del dominio)
  ├─ classify_intent()  ← agent_logic.py (soporte / configuracion / comparacion / general / fuera_contexto)
  │
  ├─ plan_tasks()       ← [paso1: recuperar_contexto, paso2: generar_respuesta]
  │
  ├─ retrieve_context() ← rag_pipeline.py → FAISS similarity_search (k=3, filtro por modelo)
  │
  ├─ ChatOpenAI stream  ← SystemPrompt + History + Context + Query → GPT-4o
  │
  └─ memory.add_*()     ← ChatMessageHistory actualiza historial
```

---

## 📁Estructura del Proyecto

```
proyecto/
├── engine/
│   ├── agent_logic.py      # Orquestador principal del agente
│   ├── rag_pipeline.py     # Recuperación semántica de contexto
│   ├── vectorstore.py      # Pipeline de indexación (build)
│   └── utils.py            # Modelos soportados, validación de dominio
├── data/
│   ├── galaxy_s25_ultra.txt
│   ├── galaxy_s25.txt
│   ├── galaxy_a55.txt
│   ├── galaxy_tab_s10_ultra.txt
│   ├── galaxy_watch7.txt
│   └── galaxy_z_fold6.txt
├── vectorstore/            # Generado automáticamente por build_vectorstore
├── main.py                 # Punto de entrada CLI
├── .env                    # Variables de entorno (no subir al repo)
└── requirements.txt
```

---

## ⚙️ Requisitos

- Python 3.10+
- Acceso a la API de GitHub Models (GPT-4o) o Azure OpenAI

---

## 🚀 Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/Jeanrenel92/AGENT-LLM
cd samsung-support-agent

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Instalar dependencias
pip install -r requirements.txt
```

---

## 🔑 Variables de Entorno

Crear archivo `.env` en la raíz del proyecto:

```env
GITHUB_TOKEN=tu_token_aqui
GITHUB_BASE_URL=https://models.inference.ai.azure.com
GITHUB_MODEL=gpt-4o
```

> Para obtener un token de GitHub Models: https://github.com/marketplace/models

---

## 🗄️ Construir el Vectorstore

Ejecutar **una sola vez** (o cuando se actualicen los documentos en `/data`):

```bash
python -m engine.vectorstore
```

Esto:
1. Carga todos los `.txt` de la carpeta `/data`
2. Genera chunks de 800 tokens con overlap de 150
3. Crea embeddings con `text-embedding-3-small`
4. Guarda el índice FAISS en `/vectorstore`

---

## 💬Ejecutar el Agente

```bash
python main.py
```

El agente inicia una sesión interactiva en terminal.

**Ejemplo de sesión:**

```
Tu consulta: ¿Cuánta RAM tiene el S25 Ultra?
[Intención: consulta_tecnica] [Modelo: GALAXY S25 ULTRA] [Confianza: alta]
→ Paso 1: recuperar_contexto
→ Paso 2: generar_respuesta
El Samsung Galaxy S25 Ultra cuenta con 12 GB de RAM...

Tu consulta: ¿y la batería?
[Intención: consulta_tecnica] [Modelo: GALAXY S25 ULTRA] [Confianza: alta]
→ Contexto expandido desde memoria: "GALAXY S25 ULTRA y la batería"
...
```

---

## 🧠 Modelos Samsung Soportados

| Alias en consulta | Modelo normalizado |
|---|---|
| s25 ultra, galaxy s25 ultra | GALAXY S25 ULTRA |
| s25, galaxy s25 | GALAXY S25 |
| a55, galaxy a55 | GALAXY A55 |
| tab s10 ultra, tab s10 | GALAXY TAB S10 ULTRA |
| watch7, galaxy watch7 | GALAXY WATCH7 |
| z fold6, fold6 | GALAXY Z FOLD6 |

---

## 🔁 Intenciones Detectadas

| Intención | Descripción | Ejemplo |
|---|---|---|
| `soporte` | Problemas técnicos o fallas | "mi S25 no carga" |
| `configuracion` | Ajustes y configuración | "cómo activar NFC en el A55" |
| `comparacion` | Comparación entre modelos | "diferencia S25 vs A55" |
| `consulta_tecnica` | Consulta de especificaciones | "procesador del Fold6" |
| `general` | Small talk o saludo | "hola", "gracias" |
| `fuera_contexto` | Fuera del dominio Samsung | "cómo programar en Python" |

---

## 📦 requirements.txt

```
langchain
langchain-openai
langchain-community
faiss-cpu
openai
python-dotenv
tiktoken
```

---

## 🧪 Pruebas Básicas

```bash
# Verificar que el vectorstore existe
python -c "from engine.rag_pipeline import retrieve_context; print('OK')"

# Test de clasificación de intención
python -c "
from engine.agent_logic import AgentOrchestrator
a = AgentOrchestrator(None)
print(a.classify_intent('mi Galaxy S25 Ultra se calienta mucho'))
"
# Salida esperada: {'intent': 'soporte', 'modelo_detectado': 'GALAXY S25 ULTRA', 'confianza': 'alta'}
```

---

## 📚 Referencias

- LangChain AI. (2024). *LangChain documentation*. https://python.langchain.com/docs/
- OpenAI. (2024). *API Reference*. https://platform.openai.com/docs/
---

## 👥 Autor

Jred. Ingeniería de Soluciones con IA