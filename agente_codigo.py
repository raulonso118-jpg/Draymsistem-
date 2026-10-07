import re
from typing import Any, Dict, List, Optional

from agents import BaseAgente


class CodeMasterAgent(BaseAgente):
    """
    Agente especializado en generación, revisión, diagnóstico y mejora de código.
    Mantiene la arquitectura experimental y agrega una capacidad profesional de ingeniería.
    """

    INTENT_TAG: str = "code_master"
    EJEMPLOS_ENTRENAMIENTO: List[str] = [
        "genera un codigo en python para una api rest",
        "revisa este codigo y dime si tiene errores",
        "analiza la calidad de este script",
        "haz una mejora de rendimiento en este codigo",
        "documenta esta funcion",
        "escribe un ejemplo en javascript con fetch",
        "detecta vulnerabilidades y anti patrones",
        "refactoriza este bloque para hacerlo mas limpio"
    ]

    def ejecutar(self, prompt: str, contexto_global: Dict[str, Any]) -> str:
        prompt_norm = (prompt or "").strip()
        if not prompt_norm:
            return "[CodeMaster]: No hay un prompt de código para procesar."

        code_block = self._extract_code_block(prompt_norm)
        language = self._detect_language(prompt_norm)

        if self._is_review_request(prompt_norm):
            return self._review_code(code_block or prompt_norm, language)

        if self._is_refactor_request(prompt_norm):
            return self._refactor_code(code_block or prompt_norm, language)

        if self._is_generation_request(prompt_norm):
            return self._generate_code(prompt_norm, language)

        if self._is_quality_request(prompt_norm):
            return self._quality_assessment(code_block or prompt_norm, language)

        return self._default_response(prompt_norm, language)

    def _extract_code_block(self, prompt: str) -> Optional[str]:
        match = re.search(r"```(?:\w+)?\s*(.*?)```", prompt, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _detect_language(self, prompt: str) -> str:
        lower = prompt.lower()
        if "javascript" in lower or "js" in lower:
            return "javascript"
        if "typescript" in lower or "ts" in lower:
            return "typescript"
        if "sql" in lower or "query" in lower:
            return "sql"
        if "html" in lower or "css" in lower:
            return "web"
        if "bash" in lower or "shell" in lower or "terminal" in lower:
            return "bash"
        return "python"

    def _is_generation_request(self, prompt: str) -> bool:
        lower = prompt.lower()
        return any(token in lower for token in ["genera", "generar", "crea", "crear", "escribe", "escribir", "implementa", "implementacion"]) and not self._is_review_request(prompt)

    def _is_review_request(self, prompt: str) -> bool:
        lower = prompt.lower()
        return any(token in lower for token in ["revisa", "revisar", "analiza", "analizar", "diagnostica", "diagnosticar", "error", "bug", "problema", "corrige", "corregir"]) or "```" in prompt

    def _is_refactor_request(self, prompt: str) -> bool:
        lower = prompt.lower()
        return any(token in lower for token in ["refactoriza", "refactorizar", "optimiza", "optimizar", "mejora", "mejorar", "limpia", "limpiar"]) 

    def _is_quality_request(self, prompt: str) -> bool:
        lower = prompt.lower()
        return any(token in lower for token in ["calidad", "quality", "seguridad", "security", "anti pattern", "patron", "mantenibilidad", "legibilidad", "rendimiento", "performance"]) 

    def _generate_code(self, prompt: str, language: str) -> str:
        description = prompt.strip()
        if language == "javascript":
            sample = """// Generado por CodeMaster
async function fetchData(url) {
  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (error) {
    console.error('Error fetching data:', error);
    return null;
  }
}

(async () => {
  const data = await fetchData('/api/ejemplo');
  console.log(data);
})();
"""
        elif language == "sql":
            sample = """SELECT
    u.id,
    u.nombre,
    COUNT(o.id) AS total_pedidos
FROM usuarios u
LEFT JOIN pedidos o ON o.usuario_id = u.id
GROUP BY u.id, u.nombre
ORDER BY total_pedidos DESC;
"""
        elif language == "web":
            sample = """<!DOCTYPE html>
<html>
  <body>
    <button id="btn">Ejecutar</button>
    <script>
      document.getElementById('btn').addEventListener('click', () => {
        console.log('Acción ejecutada');
      });
    </script>
  </body>
</html>
"""
        else:
            sample = """# Generado por CodeMaster
from fastapi import FastAPI

app = FastAPI()

@app.get('/health')
def health_check():
    return {'status': 'ok'}

@app.get('/items/{item_id}')
def read_item(item_id: int):
    return {'item_id': item_id, 'status': 'success'}
"""

        return (
            f"[CodeMaster - Generación de código]\n"
            f"Solicitud: {description}\n\n"
            f"Lenguaje sugerido: {language}\n\n"
            f"```{language}\n{sample}\n```\n\n"
            "Este bloque es una base funcional y puedes adaptarlo a tu caso concreto."
        )

    def _review_code(self, code: str, language: str) -> str:
        if not code.strip():
            return "[CodeMaster]: No se proporcionó código para revisar."

        issues = []
        lower = code.lower()

        if "print(" in lower and "logging" not in lower:
            issues.append("- Se recomienda usar logging en lugar de prints para depuración en producción.")
        if "except:" in lower and "traceback" not in lower:
            issues.append("- Se detectó manejo genérico de excepciones; conviene registrar el traceback y no ocultar el error.")
        if "eval(" in lower:
            issues.append("- Riesgo de seguridad: 'eval' puede ejecutar código arbitrario.")
        if "open(" in lower and "with open" not in lower:
            issues.append("- Se recomienda usar context managers para manejar archivos.")
        if "pass" in lower and "todo" in lower:
            issues.append("- Hay un placeholder con actividad pendiente; conviene completar o documentar la intención.")
        if not issues:
            issues.append("- No se detectaron señales obvias de riesgo en este bloque; aún así conviene revisar lógica y cobertura.")

        return (
            f"[CodeMaster - Diagnóstico de código]\n"
            f"Lenguaje: {language}\n\n"
            "Observaciones:\n"
            + "\n".join(issues) + 
            "\n\n"
            "Sugerencia:\n"
            "- Separar responsabilidades\n"
            "- Validar entradas\n"
            "- Revisar manejo de errores\n"
            "- Medir complejidad y rendimiento\n"
            "\n"
            f"Código recibido:\n```{language}\n{code[:800]}\n```"
        )

    def _refactor_code(self, code: str, language: str) -> str:
        if not code.strip():
            return "[CodeMaster]: No hay código para refactorizar."

        refactor_example = """# Refactor sugerido
# 1. Extraer responsabilidades
# 2.Usar nombres claros
# 3. Manejar errores explicitamente
# 4. Reducir duplicación

from typing import Optional


def procesar_dato(dato: Optional[str]) -> str:
    if not dato:
        return 'vacío'
    return dato.strip().lower()
"""

        return (
            f"[CodeMaster - Refactorización recomendada]\n"
            f"Lenguaje: {language}\n\n"
            "Principios aplicados:\n"
            "- Extraer funciones\n"
            "- Reducir duplicación\n"
            "- Mejorar naming\n"
            "- Manejo explícito de null y errores\n\n"
            f"Ejemplo de mejora:\n```{language}\n{refactor_example}\n```"
        )

    def _quality_assessment(self, code: str, language: str) -> str:
        if not code.strip():
            return "[CodeMaster]: No tiene código para evaluar."

        score = 82
        if "pass" in code.lower():
            score -= 5
        if "except:" in code.lower():
            score -= 4
        if "print(" in code.lower():
            score -= 3

        return (
            f"[CodeMaster - Evaluación de calidad]\n"
            f"Lenguaje: {language}\n"
            f"Puntuación estimada: {max(60, min(99, score))}/100\n\n"
            "Métricas sugeridas:\n"
            "- Legibilidad: buena\n"
            "- Mantenibilidad: aceptable\n"
            "- Complejidad: media\n"
            "- Riesgos: revisar errores y validación\n\n"
            "Recomendaciones:\n"
            "- Añadir validación de entradas\n"
            "- Reducir lógica duplicada\n"
            "- Mejorar nombres de variables\n"
            "- Añadir tests mínimos"
        )

    def _default_response(self, prompt: str, language: str) -> str:
        return (
            f"[CodeMaster - Asistente de ingeniería]\n"
            f"He detectado una solicitud relacionada con desarrollo de software.\n"
            f"Lenguaje probable: {language}\n\n"
            "Puedo ayudarte con:\n"
            "- Generación de código\n"
            "- Diagnóstico de errores\n"
            "- Revisión de calidad\n"
            "- Refactorización\n"
            "- Seguridad y rendimiento\n"
            "- Documentación y estructura\n\n"
            f"Tu consulta:\n{prompt[:400]}"
        )
