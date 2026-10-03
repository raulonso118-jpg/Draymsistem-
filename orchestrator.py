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
        # Contexto acumulativo para que los agentes compartan información
        contexto_acumulado = {
            "payload_original": payload,
            "prompt": payload.get("user_prompt", "")
        }

        # 1. Recuperar memoria/historial
        contexto_acumulado["memoria"] = self.agente_memoria.process(payload)
        
        # 2. Análisis del prompt
        contexto_acumulado["analisis"] = self.agente_analisis.process(contexto_acumulado)

        # 3. Búsqueda web / Investigación
        contexto_acumulado["investigacion"] = self.agente_investigacion.process(contexto_acumulado)

        # 4. Generación o revisión de código
        contexto_acumulado["codigo"] = self.agente_codigo.process(contexto_acumulado)

        # 5. Síntesis final (Recibe TODO el trabajo previo para generar la respuesta)
        resultado_final = self.agente_sintesis.process(contexto_acumulado)

        return {
            "agente_memoria": contexto_acumulado["memoria"],
            "agente_analisis": contexto_acumulado["analisis"],
            "agente_investigacion": contexto_acumulado["investigacion"],
            "agente_codigo": contexto_acumulado["codigo"],
            "agente_sintesis": resultado_final,
            "respuesta_generada": resultado_final.get("texto_sintetizado", "") # Salida limpia
        }
