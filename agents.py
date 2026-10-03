from typing import Dict, Any, List

class BaseAgent:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Cada agente debe implementar su propia lógica de procesamiento.")

class MemoryAgent(BaseAgent):
    def __init__(self):
        super().__init__("Agente_Memoria", "Gestión de Contexto y Persistencia")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # Extrae el historial/contexto previo si existe en el payload original
        payload = input_data if "payload_original" not in input_data else input_data.get("payload_original", {})
        learned = payload.get("learned_context", {})
        return {
            "conceptos_previos_cargados": len(learned),
            "memoria_activa": learned
        }

class AnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__("Agente_Analisis", "Inferencia y Lógica Principal")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        prompt = input_data.get("prompt", "")
        return {
            "analisis": f"Análisis de intención completado para: '{prompt}'",
            "longitud_prompt": len(prompt)
        }

class WebResearchAgent(BaseAgent):
    def __init__(self):
        super().__init__("Agente_Investigacion", "Análisis de Datos Web Externe")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        payload = input_data.get("payload_original", {})
        web_context = payload.get("web_context", [])
        return {
            "fuentes_analizadas": len(web_context),
            "contexto_web": web_context,
            "hay_informacion_web": len(web_context) > 0
        }

class CodeAgent(BaseAgent):
    def __init__(self):
        super().__init__("Agente_Codigo", "Sintaxis, Desarrollo e Ingeniería")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        prompt = input_data.get("prompt", "")
        palabras_clave = ["python", "code", "script", "función", "api", "html", "css", "js", "def "]
        is_code = any(kw in prompt.lower() for kw in palabras_clave)
        return {
            "requiere_codigo": is_code,
            "contexto_tecnico": "Modo técnico activado" if is_code else "Consulta estándar"
        }

class SynthesisAgent(BaseAgent):
    def __init__(self):
        super().__init__("Agente_Sintesis", "Generación y Consolidación de Respuesta")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        prompt = input_data.get("prompt", "")
        analisis = input_data.get("analisis", {})
        codigo = input_data.get("codigo", {})
        investigacion = input_data.get("investigacion", {})

        # Si estás integrando una API externa (OpenAI, Gemini, Ollama, etc.)
        # aquí es donde le pasas la instrucción (system prompt) al modelo.

        # Construcción de respuesta consolidada del pipeline
        partes_respuesta = []
        
        # 1. Indicador técnico si detectó código
        if codigo.get("requiere_codigo"):
            partes_respuesta.append("[Agente Código]: Se detectó una solicitud de desarrollo.")

        # 2. Información web si la hay
        if investigacion.get("hay_informacion_web"):
            fuentes = investigacion.get("contexto_web", [])
            partes_respuesta.append("\n[Agente Investigación]: Se halló la siguiente información relevante:")
            for item in fuentes:
                partes_respuesta.append(f"- {item.get('title', 'Fuente')}: {item.get('snippet', '')}")

        # 3. Respuesta base generada
        partes_respuesta.append(f"\n[Respuesta Procesada]: En relación a tu consulta '{prompt}', el sistema ha coordinado el análisis y la validación de contexto con éxito.")

        texto_final = "\n".join(partes_respuesta)

        return {
            "estado_sintesis": "Respuesta consolidada",
            "texto_sintetizado": texto_final
        }
