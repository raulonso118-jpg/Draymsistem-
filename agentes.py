import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Importación de agentes y orquestador desde agentes.py
from agentes import NVNBNode, Agente3Simulator, Agente4Memory, SwarmOrchestrator

app = FastAPI(
    title="Servidor FastAPI Copilot - NVNB G4.3",
    description="Motor neuromórfico híbrido con aprendizaje autónomo progresivo",
    version="4.3.0"
)

# Inicialización de componentes globales del sistema
motor_nvnb = NVNBNode()
memoria_agente = Agente4Memory()
simulator_agente = Agente3Simulator(timeout_sec=3.0)
orquestador = SwarmOrchestrator(motor_nvnb, memoria_agente, simulator_agente)


# Esquemas de entrada (Pydantic)
class CodigoRequest(BaseModel):
    id_tarea: str
    codigo_original: str
    instrucciones: str


# Endpoints de la API
@app.get("/")
async def root():
    return {
        "sistema": "NVNB G4.3 Swarm Copilot",
        "estado": "ONLINE",
        "modo": "Híbrido Spintrónico + Aprendizaje Autónomo"
    }


@app.post("/procesar_codigo")
async def procesar_codigo(request: CodigoRequest):
    """
    Endpoint principal: Procesa instrucciones de código enviándolas primero al motor
    local NVNB. Si el conocimiento local es bajo, delega al LLM maestro y aprende la solución.
    """
    try:
        resultado = orquestador.procesar_solicitud(
            id_tarea=request.id_tarea,
            codigo_original=request.codigo_original,
            instruccion=request.instrucciones
        )

        # Registrar traza completa de ejecución en la base de datos SQLite
        memoria_agente.registrar_ejecucion_tarea({
            "id_tarea": request.id_tarea,
            "origen_ejecucion": resultado["origen_ejecucion"],
            "instrucciones": request.instrucciones,
            "codigo_original": request.codigo_original,
            "codigo_resultado": resultado["codigo_resultado"],
            "evaluacion_nbo": resultado["evaluacion_nbo"],
            "sintaxis_valida": resultado["sintaxis_valida"]
        })

        return resultado

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en el flujo de procesamiento: {str(e)}"
        )


@app.get("/estadisticas_autonomia", tags=["Monitoreo de Aprendizaje"])
async def obtener_estadisticas_autonomia():
    """
    Endpoint de monitoreo: Muestra la tasa de autonomía (%) alcanzada por el motor local 
    frente a la dependencia de llamadas al LLM Maestro.
    """
    try:
        stats = memoria_agente.obtener_estadisticas_autonomia()
        return {
            "estatus": "exito",
            "metricas_aprendizaje": stats
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al recuperar estadísticas de aprendizaje: {str(e)}"
        )
