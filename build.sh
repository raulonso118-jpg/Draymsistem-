#!/usr/bin/env bash
# Script de build compatible con Render
# Limpia caché de pip para evitar conflictos
pip cache purge

# Instala únicamente wheels precompilados (sin compilar código C/Rust)
pip install --no-cache-dir --only-binary=:all: -r requirements.txt

# Si falta algo, intenta instalación estándar
pip install --no-build-isolation -r requirements.txt 2>/dev/null || true

# Verifica que FastAPI esté instalado
python -c "import fastapi; print(f'✓ FastAPI {fastapi.__version__} OK')"
python -c "import uvicorn; print(f'✓ Uvicorn OK')"
python -c "import pydantic; print(f'✓ Pydantic OK')"
python -c "import numpy; print(f'✓ NumPy OK')"
