from typing import Dict, Any
from config import DraymConfig
from orchestrator import Orchestrator as AgentOrchestrator

class GenerativeCoreEngine:
    def __init__(self, engine_id: str = DraymConfig.ENGINE_ID):
        self.engine_id = engine_id
        self.system_name = DraymConfig.SYSTEM_NAME
        self.orchestrator = AgentOrchestrator()

    def generate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # Ejecución del pipeline síncrono
        pipeline_results = self.orchestrator.run_pipeline(payload)

        # Extraer respuesta del perceptrón / agente de síntesis
        resultado_texto = pipeline_results.get(
            "respuesta_generada",
            f"[{self.system_name}] Sin respuesta del agente de síntesis."
        )

        return {
            "engine": self.system_name,
            "engine_id": self.engine_id,
            "status": "success",
            "result": resultado_texto,
            "pipeline_execution": pipeline_results
        }
