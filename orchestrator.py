from typing import Dict, Any
from agents import AnalysisAgent, CodeAgent, MemoryAgent, WebResearchAgent, SynthesisAgent
from nvnb_router import NVNBRouter

class AgentOrchestrator:
    def __init__(self):
        self.router_nvnb = NVNBRouter()
        self.agente_analisis = AnalysisAgent()
        self.agente_codigo = CodeAgent()
        self.agente_memoria = MemoryAgent()
        self.agente_investigacion = WebResearchAgent()
        self.agente_sintesis = SynthesisAgent()

    def run_pipeline(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        prompt = payload.get("user_prompt", "")
        
        # Enrutamiento en int8
        ruta = self.router_nvnb.enrutar(prompt)

        contexto = {
            "payload_original": payload,
            "prompt": prompt,
            "ruta_nvnb": ruta
        }

        # Ejecución secuencial interna en memoria
        contexto["memoria"] = self.agente_memoria.process(payload)
        contexto["analisis"] = self.agente_analisis.process(contexto)
        contexto["investigacion"] = self.agente_investigacion.process(contexto)
        contexto["codigo"] = self.agente_codigo.process(contexto)

        resultado_final = self.agente_sintesis.process(contexto)

        return {
            "ruta_evaluada": ruta,
            "agente_sintesis": resultado_final,
            "respuesta_generada": resultado_final.get("texto_sintetizado", "")
        }
        
