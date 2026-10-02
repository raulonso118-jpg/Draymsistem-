
import logging
from duckduckgo_search import DDGS
from config import DraymConfig

logger = logging.getLogger("WebSearch")

def search_web(query: str, max_results: int = DraymConfig.MAX_SEARCH_RESULTS) -> list:
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title"),
                    "snippet": r.get("body"),
                    "link": r.get("href")
                })
        return results
    except Exception as e:
        logger.error(f"Error en búsqueda web: {e}")
        return [{"error": f"No se pudo consultar la red: {str(e)}"}]
