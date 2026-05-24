import os
import logging

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()

logger = logging.getLogger(__name__)


# PATHS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FOLDER = os.path.join(BASE_DIR, "data")
VECTORSTORE_PATH = os.path.join(BASE_DIR, "vectorstore")

BATCH_SIZE = 50
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


# VALIDAR CREDENCIALES
def _validate_env() -> None:
    """Lanza ValueError si faltan variables de entorno críticas."""
    required = ["GITHUB_TOKEN", "GITHUB_BASE_URL"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise ValueError(
            f"Faltan variables de entorno: {', '.join(missing)}. "
            "Verifica tu archivo .env"
        )



# LOAD DOCUMENTS
def load_documents() -> list[Document]:
    """Carga archivos .txt desde DATA_FOLDER como Documents."""
    documents = []

    if not os.path.exists(DATA_FOLDER):
        logger.error("La carpeta de datos no existe: %s", DATA_FOLDER)
        return []

    for file in os.listdir(DATA_FOLDER):
        if not file.endswith(".txt"):
            continue

        path = os.path.join(DATA_FOLDER, file)
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            if not content:
                logger.warning("Archivo vacío ignorado: %s", file)
                continue

            modelo = (
                file
                .replace(".txt", "")
                .replace("_", " ")
                .upper()
            )

            content = f"MODELO: {modelo}\n\n{content}"

            doc = Document(
                page_content=content,
                metadata={"source": file, "modelo": modelo}
            )
            documents.append(doc)
            logger.info("Documento cargado: %s", file)

        except Exception as e:
            logger.error("Error cargando %s: %s", file, e, exc_info=True)

    logger.info("Total documentos cargados: %d", len(documents))
    return documents



# SPLIT DOCUMENTS
def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    logger.info("Chunks creados: %d", len(chunks))
    return chunks



# EMBEDDINGS
def create_embeddings() -> OpenAIEmbeddings:
    _validate_env()
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("GITHUB_TOKEN"),
        base_url=os.getenv("GITHUB_BASE_URL")
    )



# CREATE VECTORSTORE
def create_vectorstore(
    chunks: list[Document],
    embeddings: OpenAIEmbeddings
) -> FAISS:
    """
    Genera el índice FAISS por lotes y lo guarda en disco.
    Retorna el objeto vectorstore para uso inmediato.
    """
    logger.info("Generando embeddings por lotes (batch=%d)...", BATCH_SIZE)

    vectorstore = None

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i: i + BATCH_SIZE]
        logger.info("Procesando lote %d – %d", i, i + len(batch))

        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)

    if vectorstore is None:
        raise RuntimeError("No se generó ningún vectorstore: chunks vacíos.")

    vectorstore.save_local(VECTORSTORE_PATH)
    logger.info("Vectorstore guardado en: %s", VECTORSTORE_PATH)
    return vectorstore


# LOAD VECTORSTORE (reutilizar sin regenerar)
def load_vectorstore(embeddings: OpenAIEmbeddings) -> FAISS:
    """
    Carga el vectorstore desde disco si existe.
    Lanza FileNotFoundError si no ha sido construido aún.
    """
    index_file = os.path.join(VECTORSTORE_PATH, "index.faiss")
    if not os.path.exists(index_file):
        raise FileNotFoundError(
            f"Vectorstore no encontrado en '{VECTORSTORE_PATH}'. "
            "Ejecuta primero: python -m engine.vectorstore"
        )
    logger.info("Cargando vectorstore existente desde disco...")
    return FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )



# GET VECTORSTORE (carga o crea según corresponda)
def get_vectorstore(force_rebuild: bool = False) -> FAISS:
    """
    Punto de entrada principal para obtener el vectorstore.
    - Si ya existe en disco y force_rebuild=False → carga.
    - Si no existe o force_rebuild=True → genera desde cero.
    """
    embeddings = create_embeddings()
    index_file = os.path.join(VECTORSTORE_PATH, "index.faiss")

    if os.path.exists(index_file) and not force_rebuild:
        return load_vectorstore(embeddings)

    logger.info("Construyendo vectorstore desde documentos...")
    docs = load_documents()
    if not docs:
        raise RuntimeError(
            f"No se encontraron documentos .txt en '{DATA_FOLDER}'"
        )
    chunks = split_documents(docs)
    return create_vectorstore(chunks, embeddings)



# MAIN
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    try:
        _validate_env()
        get_vectorstore(force_rebuild=True)
        logger.info("Pipeline finalizado correctamente.")
    except Exception as e:
        logger.error("Error en pipeline: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    main()
