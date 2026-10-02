from typing import Dict, Any
from agents import AnalysisAgent, CodeAgent, MemoryAgent, WebResearchAgent, SynthesisAgent

class AgentOrchestrator:
    def __init__(self):
        self.agente_analisis = AnalysisAgent()
        self.agente_codigo = CodeAgent()
        self.agente_memoria = MemoryAgent()
        self.agente_investigacion = WebResearchAgent()
        self.agente_sintesis = SynthesisAgent()

    def run_pipeline(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # Ejecución coordinada de los 5 agentes
        res_memoria = self.agente_memoria.process(payload)
        res_analisis = self.agente_analisis.process(payload)
        res_codigo = self.agente_codigo.process(payload)
        res_investigacion = self.agente_investigacion.process(payload)
        res_sintesis = self.agente_sintesis.process(payload)

        return {
            "agente_memoria": res_memoria,
            "agente_analisis": res_analisis,
            "agente_codigo": res_codigo,
            "agente_investigacion": res_investigacion,
            "agente_sintesis": res_sintesis
        }
