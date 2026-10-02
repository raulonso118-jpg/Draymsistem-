from typing import Dict, Any, List

class BaseAgent:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Cada agente debe implementar su propia lógica de procesamiento.")

class AnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__("Agente_Analisis", "Inferencia y Lógica Principal")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        prompt = input_data.get("user_prompt", "")
        return {"analisis": f"Procesamiento lógico y descomposición de consulta: '{prompt}'"}

class CodeAgent(BaseAgent):
    def __init__(self):
        super().__init__("Agente_Codigo", "Sintaxis, Desarrollo e Ingeniería")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        prompt = input_data.get("user_prompt", "")
        is_code = any(kw in prompt.lower() for kw in ["python", "code", "script", "función", "api", "html", "css", "js", "def "])
        return {"requiere_codigo": is_code, "contexto_tecnico": "Modo de asistencia técnica activado" if is_code else "Consulta estándar"}

class MemoryAgent(BaseAgent):
    def __init__(self):
        super().__init__("Agente_Memoria", "Gestión de Contexto y Persistencia")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        learned = input_data.get("learned_context", {})
        return {"conceptos_previos_cargados": len(learned)}

class WebResearchAgent(BaseAgent):
    def __init__(self):
        super().__init__("Agente_Investigacion", "Análisis de Datos Web Externe")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        web_context = input_data.get("web_context", [])
        return {"fuentes_analizadas": len(web_context), "hay_informacion_web": len(web_context) > 0}

class SynthesisAgent(BaseAgent):
    def __init__(self):
        super().__init__("Agente_Sintesis", "Generación y Consolidación de Respuesta")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return {"estado_sintesis": "Respuesta lista para compilación final"}
