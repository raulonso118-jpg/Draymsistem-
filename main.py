import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional

from agentes import Agente1Planner, Agente2Coder, Agente3Simulator, Agente4Memory, Agente5Tactical
from nvnb_core import NVNB_Orquestador

app = FastAPI(
    title="Plataforma de Inteligencia Híbrida NVNB G4.3 & Enjambre NBO-a",
    description="Sistema multi-agente con motor neuromórfico de generación.",
    version="2.0.0"
)

planner = Agente1Planner()
coder = Agente2Coder()
simulator = Agente3Simulator()
memory = Agente4Memory(db_path="memoria_orquestador.db")
tactical = Agente5Tactical()

nvnb_ia = NVNB_Orquestador()

class SolicitudChat(BaseModel):
    mensaje: str
    aprender: bool = True

class SolicitudRefactorizacion(BaseModel):
    id_tarea: str
    nombre_modulo: str
    codigo_original: str
    instrucciones: str
    soporte_hardware: Optional[Dict[str, str]] = None

@app.get("/")
def estado_sistema():
    return {
        "sistema": "NVNB G4.3 & NBO-a Multi-Agent System",
        "estado": "OPERATIVO",
        "agentes": [planner.nombre, coder.nombre, simulator.nombre, memory.nombre, tactical.nombre],
        "motor_ia_generativa": "NVNB Spintronic Neuromorphic Core G4.3"
    }

@app.post("/chat")
def chatear_con_nvnb(solicitud: SolicitudChat):
    respuesta = nvnb_ia.procesar(solicitud.mensaje)
    
    if solicitud.aprender:
        nvnb_ia.aprendizaje_dirigido(solicitud.mensaje, "maestro")
        if not respuesta.startswith("[Silencio") and not respuesta.startswith("[SISTEMA"):
            limpia_resp = respuesta.split("]", 1)[-1].strip()
            nvnb_ia.aprendizaje_dirigido(f"{solicitud.mensaje} {limpia_resp}", "maestro")
        nvnb_ia.guardar_todo()

    memory.registrar_historial("usuario", solicitud.mensaje)
    memory.registrar_historial("ia_nvnb", respuesta)

    return {
        "entrada": solicitud.mensaje,
        "respuesta_nvnb": respuesta,
        "memoria_preservada": solicitud.aprender
    }

@app.post("/procesar_codigo")
def refactorizar_codigo(solicitud: SolicitudRefactorizacion) -> Dict[str, Any]:
    diagnostico_hw = None
    if solicitud.soporte_hardware:
        etapa = solicitud.soporte_hardware.get("etapa", "")
        sintoma = solicitud.soporte_hardware.get("sintoma", "")
        diagnostico_hw = tactical.diagnosticar_etapa(etapa, sintoma)

    plan = planner.planificar_y_validar(
        id_tarea=solicitud.id_tarea,
        codigo_bruto=solicitud.codigo_original,
        instrucciones=solicitud.instrucciones
    )

    if not plan["ast_valido"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Error sintáctico AST: {plan['error_ast']}"
        )

    codigo_refactorizado = coder.generar_codigo_limpio(
        instrucciones=solicitud.instrucciones,
        codigo_base=solicitud.codigo_original
    )

    resultado_simulacion = simulator.simular_y_evaluar(
        id_tarea=solicitud.id_tarea,
        codigo_python=codigo_refactorizado
    )

    if resultado_simulacion["aprobado_para_despliegue"]:
        memory.guardar_modulo_aprobado(
            id_tarea=solicitud.id_tarea,
            nombre_modulo=solicitud.nombre_modulo,
            codigo=codigo_refactorizado,
            score_nbo=resultado_simulacion["vector_evaluacion_nbo"]
        )

    return {
        "id_tarea": solicitud.id_tarea,
        "nombre_modulo": solicitud.nombre_modulo,
        "estado": "VALIDADO_Y_GUARDADO" if resultado_simulacion["aprobado_para_despliegue"] else "REVISAR",
        "evaluacion_nbo": resultado_simulacion["vector_evaluacion_nbo"],
        "diagnostico_hardware": diagnostico_hw,
        "codigo_resultado": codigo_refactorizado
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
