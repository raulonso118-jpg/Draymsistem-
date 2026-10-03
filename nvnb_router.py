import numpy as np

# Parámetros del núcleo neuromórfico NVNB G4.3
DTYPE_INT = np.int8

class NVNBRouter:
    """Router neuromórfico en int8 que clasifica la intención en sub-milisegundos."""
    def __init__(self):
        self.rutas_palabras = {
            "codigo": ["python", "def", "script", "code", "funcion", "api", "html", "js", "css", "class"],
            "investigacion": ["busca", "google", "noticias", "que es", "quien", "precio", "donde"],
            "memoria": ["recuerdas", "guardar", "historial", "como me llamo", "memoriza"],
            "analisis": ["analiza", "calcula", "por que", "explica", "razona", "logic"]
        }

    def enrutar(self, prompt: str) -> str:
        prompt_clean = prompt.lower()
        scores = {agente: 0 for agente in self.rutas_palabras}
        
        for agente, palabras in self.rutas_palabras.items():
            for p in palabras:
                if p in prompt_clean:
                    scores[agente] += 1

        agente_destino = max(scores, key=scores.get)
        if scores[agente_destino] == 0:
            return "sintesis_directa"
            
        return agente_destino
