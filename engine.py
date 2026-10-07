from typing import Dict, Any
from config import DraymConfig
from orchestrator import Orchestrator as AgentOrchestrator


class GenerativeCoreEngine:
    def __init__(self, engine_id: str = DraymConfig.ENGINE_ID):
        self.engine_id = engine_id
        self.system_name = DraymConfig.SYSTEM_NAME
        self.orchestrator = AgentOrchestrator()

    def generate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # Compatibilidad con el diseño híbrido experimental.
        # Se acepta tanto un prompt directo como un payload completo.
        user_prompt = payload.get("user_prompt", "")
        if not user_prompt and isinstance(payload.get("prompt"), str):
            user_prompt = payload["prompt"]

        if not user_prompt:
            raise ValueError("No se recibió un prompt de entrada para generar una respuesta.")

        respuesta_generada = self.orchestrator.procesar(user_prompt)

        return {
            "engine": self.system_name,
            "engine_id": self.engine_id,
            "status": "success",
            "result": respuesta_generada,
            "pipeline_execution": {
                "prompt": user_prompt,
                "web_context": payload.get("web_context", []),
                "learned_context": payload.get("learned_context", {}),
                "extra_context": payload.get("extra_context", {})
            }
        }

    def run_pipeline(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Alias de compatibilidad para evitar roturas con llamadas antiguas.
        Mantiene compatibilidad con el diseño experimental y con frontends previos.
        """
        return self.generate(payload)
