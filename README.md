# 🚀 Draymsistem - Plataforma NVNB G4.3 & Enjambre NBO-a

Sistema multi-agente con motor neuromórfico spintrónico para refactorización inteligente de código y diagnóstico de hardware.

## ✨ Características

- 5 Agentes especializados: Planner, Coder, Simulator, Memory, Tactical
- Motor neuromórfico NVNB G4.3
- API FastAPI para chat y refactorización
- Persistencia SQLite para memoria de módulos
- Validación AST y sandbox seguro

## Requisitos

- Python 3.11+
- pip

## Instalación local

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
python main.py
```

La app corre en http://localhost:8000

## Deploy en Render

- Build Command: `bash build.sh`
- Start Command: `python main.py`
- Runtime: Python 3.11

## Archivos importantes

- `main.py`: servidor FastAPI
- `agentes.py`: agentes del enjambre
- `nvnb_core.py`: motor NVNB
- `requirements.txt`: dependencias compatibles con Render
- `build.sh`: instalación segura para Render
- `render.yaml`: configuración de deploy
