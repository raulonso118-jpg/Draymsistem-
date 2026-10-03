from typing import Dict, Any
from config import DraymConfig
from orchestrator import AgentOrchestrator

class GenerativeCoreEngine:
    def __init__(self, engine_id: str = DraymConfig.ENGINE_ID):
        self.engine_id = engine_id
        self.system_name = DraymConfig.SYSTEM_NAME
        self.orchestrator = AgentOrchestrator()

    def generate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        prompt = payload.get("user_prompt", "")
        web_context = payload.get("web_context", [])
        
        # Procesar flujo a través de los 5 agentes
        pipeline_results = self.orchestrator.run_pipeline(payload)

        # Construcción de la respuesta integrada
        contexto_web_texto = ""
        if web_context:
            contexto_web_texto = "\n\n[Información encontrada en la red]:\n" + "\n".join(
                [f"- {item.get('title')}: {item.get('snippet')}" for item in web_context if "error" not in item]
            )

        resultado_texto = f"[{self.system_name}] Procesado con éxito el prompt: '{prompt}'."
        if contexto_web_texto:
            resultado_texto += contexto_web_texto

        return {
            "engine": self.system_name,
            "engine_id": self.engine_id,
            "status": "success",
            "result": resultado_texto,
            "pipeline_execution": pipeline_results
        }
