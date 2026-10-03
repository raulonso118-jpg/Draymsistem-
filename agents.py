from typing import Dict, Any
from nbo_core import predict_nbo, text_to_vector

class BaseAgent:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

class MemoryAgent(BaseAgent):
    def __init__(self):
        super().__init__("Agente_Memoria", "Gestión Contextual")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        payload = input_data.get("payload_original", input_data)
        learned = payload.get("learned_context", {})
        return {"conceptos_cargados": len(learned)}

class AnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__("Agente_Analisis", "Evaluación Perceptronal")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        prompt = input_data.get("prompt", "")
        vector_in = text_to_vector(prompt, dim=4)
        activaciones = predict_nbo(vector_in)
        return {
            "vector_entrada": vector_in,
            "activaciones_nbo": [round(x, 4) for x in activaciones]
        }

class WebResearchAgent(BaseAgent):
    def __init__(self):
        super().__init__("Agente_Investigacion", "Contexto Web")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        payload = input_data.get("payload_original", {})
        web_context = payload.get("web_context", [])
        return {"fuentes": web_context, "hay_web": len(web_context) > 0}

class CodeAgent(BaseAgent):
    def __init__(self):
        super().__init__("Agente_Codigo", "Sintaxis e Ingeniería")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        prompt = input_data.get("prompt", "")
        palabras_clave = ["python", "code", "script", "función", "def ", "html", "js", "css"]
        es_codigo = any(kw in prompt.lower() for kw in palabras_clave)
        return {"requiere_codigo": es_codigo}

class SynthesisAgent(BaseAgent):
    def __init__(self):
        super().__init__("Agente_Sintesis", "Consolidación de Respuesta")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        prompt = input_data.get("prompt", "")
        analisis = input_data.get("analisis", {})
        codigo = input_data.get("codigo", {})
        investigacion = input_data.get("investigacion", {})

        salida_red = analisis.get("activaciones_nbo", [0.0, 0.0])
        partes = []

        if codigo.get("requiere_codigo"):
            partes.append("```python\n# [Bloque generado bajo patrón perceptrónico]\n```")

        if investigacion.get("hay_web"):
            partes.append("\n[Información relevante incorporada]:")
            for item in investigacion.get("fuentes", []):
                partes.append(f"- {item.get('title')}: {item.get('snippet')}")

        partes.append(
            f"\n[Procesamiento NBO-a completado con éxito]"
            f"\nActivación relu_211 en capa final: {salida_red}"
        )

        return {
            "estado": "exito",
            "texto_sintetizado": "\n".join(partes)
    }
    
