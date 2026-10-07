from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

from config import DraymConfig
from vision_core import process_base64_image
from web_search import search_web
from memory import MemoryManager
from engine import GenerativeCoreEngine

app = FastAPI(
    title=DraymConfig.SYSTEM_NAME, 
    version=DraymConfig.VERSION
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambiar por dominios específicos en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = GenerativeCoreEngine()
memory = MemoryManager()

class ImageRequest(BaseModel):
    imagen: str
    tono_r: Optional[float] = Field(default=100.0, ge=0.0, le=250.0)
    tono_g: Optional[float] = Field(default=100.0, ge=0.0, le=250.0)
    tono_b: Optional[float] = Field(default=100.0, ge=0.0, le=250.0)
    brillo: Optional[float] = Field(default=50.0, ge=0.0, le=100.0)
    contraste: Optional[float] = Field(default=50.0, ge=0.0, le=100.0)
    suavizado: Optional[float] = Field(default=0.0, ge=0.0, le=100.0)

class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    use_web_search: Optional[bool] = False
    context: Optional[Dict[str, Any]] = None

@app.get("/")
async def root():
    return {
        "status": "online",
        "system": DraymConfig.SYSTEM_NAME,
        "version": DraymConfig.VERSION
    }

@app.post("/api/procesar_rostro")
async def api_procesar_rostro(req: ImageRequest):
    try:
        # Ejecuta el procesamiento de imagen en un hilo secundario para no bloquear la API
        resultado = await run_in_threadpool(process_base64_image, req.imagen, req.model_dump())
        return {"estatus": "ok", "imagen_procesada": resultado}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando imagen: {str(e)}")

@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    try:
        web_data = []
        if req.use_web_search:
            web_data = await run_in_threadpool(search_web, req.prompt)

        payload = {
            "user_prompt": req.prompt,
            "web_context": web_data,
            "learned_context": memory.data.get("learned_concepts", {}),
            "extra_context": req.context or {}
        }

        response = await run_in_threadpool(engine.generate, payload)
        
        # Guarda el historial/memoria
        key_name = f"last_query_{req.prompt[:15]}"
        memory.save_concept(key_name, response.get("result"))

        return {
            "status": "success",
            "response": response,
            "sources": web_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en motor de chat: {str(e)}")
