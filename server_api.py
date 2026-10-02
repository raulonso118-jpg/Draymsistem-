from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional

from config import DraymConfig
from vision_core import process_base64_image
from web_search import search_web
from memory import MemoryManager
from engine import GenerativeCoreEngine

app = FastAPI(title=DraymConfig.SYSTEM_NAME, version=DraymConfig.VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = GenerativeCoreEngine()
memory = MemoryManager()

class ImageRequest(BaseModel):
    imagen: str
    tono_r: Optional[float] = 100
    tono_g: Optional[float] = 100
    tono_b: Optional[float] = 100
    brillo: Optional[float] = 50
    contraste: Optional[float] = 50
    suavizado: Optional[float] = 0

class ChatRequest(BaseModel):
    prompt: str
    use_web_search: Optional[bool] = False
    context: Optional[Dict[str, Any]] = None

@app.get("/")
def root():
    return {
        "status": "online",
        "system": DraymConfig.SYSTEM_NAME,
        "version": DraymConfig.VERSION
    }

@app.post("/api/procesar_rostro")
def api_procesar_rostro(req: ImageRequest):
    try:
        resultado = process_base64_image(req.imagen, req.dict())
        return {"estatus": "ok", "imagen_procesada": resultado}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
def api_chat(req: ChatRequest):
    try:
        web_data = []
        if req.use_web_search:
            web_data = search_web(req.prompt)

        payload = {
            "user_prompt": req.prompt,
            "web_context": web_data,
            "learned_context": memory.data["learned_concepts"],
            "extra_context": req.context or {}
        }

        response = engine.generate(payload)
        memory.save_concept(f"last_query_{req.prompt[:15]}", response.get("result"))

        return {
            "status": "success",
            "response": response,
            "sources": web_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
