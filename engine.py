from typing import Dict, Any
from config import DraymConfig
from orchestrator import Orchestrator as AgentOrchestrator

class GenerativeCoreEngine:
    def __init__(self, engine_id: str = DraymConfig.ENGINE_ID):
        self.engine_id = engine_id
        self.system_name = DraymConfig.SYSTEM_NAME
        self.orchestrator = AgentOrchestrator()

    def generate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # Ejecución del pipeline síncrono - CORREGIDO
        user_prompt = payload.get("user_prompt", "")
        respuesta_generada = self.orchestrator.procesar(user_prompt)

        return {
            "engine": self.system_name,
            "engine_id": self.engine_id,
            "status": "success",
            "result": respuesta_generada,
            "pipeline_execution": {
                "prompt": user_prompt,
                "web_context": payload.get("web_context", []),
                "learned_context": payload.get("learned_context", {})
            }
        }
