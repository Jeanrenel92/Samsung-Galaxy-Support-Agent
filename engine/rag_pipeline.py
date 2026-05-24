import logging

logger = logging.getLogger(__name__)


# RETRIEVE CONTEXT


def retrieve_context(
    query: str,
    vectorstore,
    modelo: str | None = None,
    k: int = 3
) -> str:
    """
    Recupera chunks relevantes desde el vectorstore FAISS.

    Estrategia:
    1. Si hay modelo detectado, intenta búsqueda filtrada.
    2. Si el filtro no retorna resultados, hace fallback a
       búsqueda global (sin filtro) para no perder contexto.
    3. Si tampoco hay resultados globales, retorna mensaje claro.
    """
    try:
        docs = []
        used_fallback = False

        #BÚSQUEDA FILTRADA POR MODELO
        if modelo:
            filtro = {"modelo": modelo}
            docs = vectorstore.similarity_search(
                query=query,
                k=k,
                filter=filtro
            )
            logger.debug(
                "Chunks con filtro '%s': %d", modelo, len(docs)
            )

        #FALLBACK: búsqueda global si el filtro no devuelve nada
        if not docs:
            if modelo:
                logger.warning(
                    "Sin resultados con filtro '%s'. "
                    "Reintentando búsqueda global.", modelo
                )
                used_fallback = True
            docs = vectorstore.similarity_search(
                query=query,
                k=k
            )
            logger.debug("Chunks globales: %d", len(docs))

        #SIN RESULTADOS EN NINGÚN CASO
        if not docs:
            return "No se encontró información relevante en la base de conocimiento."

        #DEBUG CHUNKS
        for i, doc in enumerate(docs):
            logger.debug(
                "\n--- Chunk %d ---\n%s", i + 1, doc.page_content[:400]
            )

        #BUILD CONTEXT
        context = "\n\n".join(doc.page_content for doc in docs)

        # Si se usó fallback global, advertir al LLM para evitar
        # que confunda el modelo del contexto con el solicitado.
        if used_fallback and modelo:
            context = (
                f"ADVERTENCIA: No se encontró información específica para {modelo}. "
                f"El siguiente contexto es de carácter general y puede referirse "
                f"a otros modelos Samsung.\n\n{context}"
            )

        return context

    except Exception as e:
        logger.error("Error en retrieve_context: %s", e, exc_info=True)
        return "Error recuperando contexto desde la base de conocimiento."