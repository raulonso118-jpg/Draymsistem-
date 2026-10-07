"""
ARCHIVO DE DIAGNÓSTICO COMPLETO
================================
Este archivo realiza pruebas en cadena para identificar exactamente dónde falla el sistema.
Ejecuta: python diagnostico.py
"""

import sys
import traceback
from typing import Dict, Any, Tuple

print("=" * 80)
print("INICIANDO DIAGNÓSTICO COMPLETO DE DRAYMSYSTEM")
print("=" * 80)

# TEST 1: Cargar módulos básicos
print("\n[TEST 1] Importando módulos base...")
try:
    from config import DraymConfig
    print("✓ config.py cargado correctamente")
except Exception as e:
    print(f"✗ ERROR en config.py: {e}")
    traceback.print_exc()
    sys.exit(1)

# TEST 2: Verificar archivo de memoria
print("\n[TEST 2] Verificando archivo de memoria...")
try:
    import os
    if not os.path.exists(DraymConfig.MEMORY_FILE):
        print(f"⚠ Archivo de memoria no existe: {DraymConfig.MEMORY_FILE}")
        print("  Creando archivo vacío...")
        import json
        with open(DraymConfig.MEMORY_FILE, "w") as f:
            json.dump({"projects": {}, "learned_concepts": {}}, f)
        print(f"✓ Archivo creado en: {DraymConfig.MEMORY_FILE}")
    else:
        print(f"✓ Archivo de memoria existe: {DraymConfig.MEMORY_FILE}")
except Exception as e:
    print(f"✗ ERROR con archivo de memoria: {e}")
    traceback.print_exc()

# TEST 3: Cargar Memory Manager
print("\n[TEST 3] Cargando MemoryManager...")
try:
    from memory import MemoryManager
    memory = MemoryManager()
    print("✓ MemoryManager inicializado correctamente")
except Exception as e:
    print(f"✗ ERROR en memory.py: {e}")
    traceback.print_exc()
    sys.exit(1)

# TEST 4: Cargar BaseAgente
print("\n[TEST 4] Verificando BaseAgente...")
try:
    from agents import BaseAgente
    print("✓ BaseAgente disponible en agents.py")
except ImportError:
    print("⚠ BaseAgente no en agents.py, será creado dinámicamente en orchestrator.py")
    print("✓ Esto es normal, orchestrator.py lo define si falta")

# TEST 5: Cargar Orchestrator (el punto crítico)
print("\n[TEST 5] Cargando Orchestrator (PUNTO CRÍTICO)...")
try:
    from orchestrator import Orchestrator
    print("✓ Orchestrator importado correctamente")
    
    print("  Inicializando Orchestrator...")
    orchestrator = Orchestrator()
    print("✓ Orchestrator inicializado correctamente")
    
    print(f"  Agentes registrados: {list(orchestrator.agentes_registrados.keys())}")
    if orchestrator.agentes_registrados:
        print(f"  ✓ Se encontraron {len(orchestrator.agentes_registrados)} agentes")
    else:
        print(f"  ⚠ ADVERTENCIA: No se encontraron agentes automáticos")
    
except Exception as e:
    print(f"✗ ERROR CRÍTICO en orchestrator.py: {e}")
    traceback.print_exc()
    print("\n>>> Este es el problema. El Orchestrator está fallando.")
    sys.exit(1)

# TEST 6: Cargar Engine
print("\n[TEST 6] Cargando GenerativeCoreEngine...")
try:
    from engine import GenerativeCoreEngine
    engine = GenerativeCoreEngine()
    print("✓ Engine inicializado correctamente")
except Exception as e:
    print(f"✗ ERROR en engine.py: {e}")
    traceback.print_exc()
    sys.exit(1)

# TEST 7: Prueba simple de chat
print("\n[TEST 7] Prueba de chat simple...")
try:
    test_prompt = "hola"
    print(f"  Enviando prompt: '{test_prompt}'")
    
    response = engine.generate({
        "user_prompt": test_prompt,
        "web_context": [],
        "learned_context": {},
        "extra_context": {}
    })
    
    print("✓ Respuesta recibida:")
    print(f"  Status: {response.get('status')}")
    print(f"  Result: {response.get('result')[:100]}...")
    
except Exception as e:
    print(f"✗ ERROR durante la ejecución del chat: {e}")
    traceback.print_exc()
    print("\n>>> El problema está en engine.generate() -> orchestrator.procesar()")
    sys.exit(1)

# TEST 8: Prueba con diferentes tipos de prompts
print("\n[TEST 8] Prueba de múltiples prompts...")
test_cases = [
    "genera un código en python",
    "revisa este código",
    "dime la documentación",
    "analiza la seguridad"
]

for prompt in test_cases:
    try:
        response = engine.generate({
            "user_prompt": prompt,
            "web_context": [],
            "learned_context": {},
            "extra_context": {}
        })
        print(f"✓ '{prompt}' -> OK")
    except Exception as e:
        print(f"✗ '{prompt}' -> ERROR: {str(e)[:50]}...")

print("\n" + "=" * 80)
print("DIAGNÓSTICO COMPLETADO")
print("=" * 80)
print("\nRESUMEN:")
print("- Si llegaste aquí y todo está OK, el sistema está funcionando correctamente")
print("- Si hubo errores, revisa los mensajes de ERROR arriba")
print("- El problema más probable está en:")
print("  1. orchestrator.py - _cargar_agentes_dinamicamente()")
print("  2. Falta de archivos agente_*.py")
print("  3. Errores de importación en los agentes")
print("  4. Archivo de memoria corrompido o sin permisos de escritura")
