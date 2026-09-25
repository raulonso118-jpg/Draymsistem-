# Dockerfile para ejecutar Draymsistem en cualquier plataforma
FROM python:3.11-slim

# Establece directorio de trabajo
WORKDIR /app

# Copia requirements
COPY requirements.txt .

# Instala dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copia código
COPY *.py .

# Expone puerto
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/').read()" || exit 1

# Comando de inicio
CMD ["python", "main.py"]
