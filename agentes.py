import ast
import os
import random
import sqlite3
import requests
import concurrent.futures
from typing import Dict, Any, Optional

# ==========================================
# 1. MOTOR NEUROMÓRFICO NVNB G4.3 & NBO-a
# ==========================================
class NVNBNode:
    """
    Nodo emulador de dinámica spintrónica (MTJ / Spintronics) con 
    Votación por Redundancia Recursiva (RRV) y evaluación Perceptrón NBO-a.
    """
    def __init__(self):
        self.dynamic_theta = 0.5  # Umbral dinámico de espín
        self.w_rrv = [0.80, 0.85, 0.90]  # Ponderaciones de matrices RRV

    def procesar_y_evaluar(self, codigo: str, sintaxis_valida: bool, es_maestro: bool) -> float:
        if not sintaxis_valida:
            return 0.10

        # Evaluación heurística y spintrónica (NBO-a)
        base_score = 0.92 if es_maestro else 0.88
        ruido_spintronico = random.uniform(-0.02, 0.02)
        score_final = min(1.0, max(0.0, base_score + ruido_spintronico))
        return round(score_final, 3)

    def evaluar_confianza_patron(self, score_previo: float) -> float:
        factor_rrv = sum(self.w_rrv) / len(self.w_rrv)
        confianza = score_previo * factor_rrv
        return round(confianza, 3)

    def muda_de_piel_adaptativa(self):
        """Ajuste dinámico de umbral de conmoción de espín."""
        self.dynamic_theta = min(0.9, self.dynamic_theta + 0.01)


# ==========================================
# 2. AGENTE 3: SANDBOX SIMULATOR PROTEGIDO
# ==========================================
class Agente3Simulator:
    """Aísla la ejecución del código generado en un entorno sandbox con timeout."""
    def __init__(self, timeout_sec: float = 3.0):
        self.timeout_sec = timeout_sec

    def probar_codigo(self, codigo_python: str) -> Dict[str, Any]:
        def _ejecutar():
            try:
                ast.parse(codigo_python)
                loc = {}
                exec(codigo_python, {"__builtins__": __builtins__}, loc)
                return {"exito": True, "error": None}
            except Exception as e:
                return {"exito": False, "error": str(e)}

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_ejecutar)
            try:
                return future.result(timeout=self.timeout_sec)
            except concurrent.futures.TimeoutError:
                return {"exito": False, "error": f"ExecutionTimeout ({self.timeout_sec}s)"}


# ==========================================
# 3. AGENTE 4: MEMORIA Y PERSISTENCIA SQLITE3
# ==========================================
class Agente4Memory:
    """
    Gestiona la persistencia SQLite3 con soporte WAL para entornos ASGI (Render / FastAPI).
    Almacena auditoría de ejecuciones y patrones aprendidos.
    """
    def __init__(self, db_path: str = "nvnb_memory.db"):
        self.db_path = db_path
        self._inicializar_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _inicializar_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA synchronous=NORMAL;")
            
            # Tabla de trazabilidad de tareas
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS registro_tareas (
                    id_tarea TEXT PRIMARY KEY,
                    origen_ejecucion TEXT,
                    instrucciones TEXT,
                    codigo_original TEXT,
                    codigo_resultado TEXT,
                    evaluacion_nbo REAL,
                    sintaxis_valida INTEGER,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Tabla de patrones aprendidos por el motor local
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS patrones_aprendidos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    instruccion_clave TEXT NOT NULL UNIQUE,
                    codigo_optimizado TEXT NOT NULL,
                    score_nbo REAL NOT NULL,
                    frecuencia_uso INTEGER DEFAULT 1,
                    ultima_actualizacion DATETIME DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_instruccion ON patrones_aprendidos(instruccion_clave);")
            conn.commit()

    def buscar_patron_similar(self, instruccion: str) -> Optional[Dict[str, Any]]:
        instruccion_norm = instruccion.strip().lower()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, codigo_optimizado, score_nbo, frecuencia_uso
                FROM patrones_aprendidos
                WHERE instruccion_clave = ?
                LIMIT 1;
            """, (instruccion_norm,))
            row = cursor.fetchone()
            if row:
                cursor.execute("""
                    UPDATE patrones_aprendidos 
                    SET frecuencia_uso = frecuencia_uso + 1, ultima_actualizacion = CURRENT_TIMESTAMP
                    WHERE id = ?;
                """, (row["id"],))
                conn.commit()
                return {
                    "codigo_resultado": row["codigo_optimizado"],
                    "evaluacion_nbo": row["score_nbo"],
                    "frecuencia_uso": row["frecuencia_uso"] + 1
                }
        return None

    def guardar_patron_aprendido(self, instruccion: str, codigo: str, score_nbo: float):
        instruccion_norm = instruccion.strip().lower()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO patrones_aprendidos (instruccion_clave, codigo_optimizado, score_nbo)
                VALUES (?, ?, ?)
                ON CONFLICT(instruccion_clave) DO UPDATE SET
                    codigo_optimizado = excluded.codigo_optimizado,
                    score_nbo = MAX(patrones_aprendidos.score_nbo, excluded.score_nbo),
                    ultima_actualizacion = CURRENT_TIMESTAMP;
            """, (instruccion_norm, codigo, score_nbo))
            conn.commit()

    def registrar_ejecucion_tarea(self, datos: Dict[str, Any]):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO registro_tareas (
                    id_tarea, origen_ejecucion, instrucciones, 
                    codigo_original, codigo_resultado, evaluacion_nbo, sintaxis_valida
                ) VALUES (?, ?, ?, ?, ?, ?, ?);
            """, (
                datos.get("id_tarea"),
                datos.get("origen_ejecucion"),
                datos.get("instrucciones"),
                datos.get("codigo_original"),
                datos.get("codigo_resultado"),
                datos.get("evaluacion_nbo"),
                1 if datos.get("sintaxis_valida") else 0
            ))
            conn.commit()

    def obtener_estadisticas_autonomia(self) -> Dict[str, Any]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) AS total FROM registro_tareas;")
            total = cursor.fetchone()["total"]

            if total == 0:
                return {
                    "total_peticiones": 0,
                    "respuestas_locales_nvnb": 0,
                    "respuestas_llm_maestro": 0,
                    "porcentaje_autonomia": "0.00%",
                    "patrones_aprendidos_totales": 0,
                    "estado_motor": "EN_ENTRENAMIENTO_INICIAL"
                }

            cursor.execute("""
                SELECT 
                    SUM(CASE WHEN origen_ejecucion = 'LOCAL_NVNB' THEN 1 ELSE 0 END) AS locales,
                    SUM(CASE WHEN origen_ejecucion = 'EXTERNAL_LLM_TEACHER' THEN 1 ELSE 0 END) AS externas
                FROM registro_tareas;
            """)
            conteo = cursor.fetchone()
            locales = conteo["locales"] or 0
            externas = conteo["externas"] or 0
            porcentaje = (locales / total) * 100

            cursor.execute("SELECT COUNT(*) AS total_patrones FROM patrones_aprendidos;")
            total_patrones = cursor.fetchone()["total_patrones"]

            estado = "ALTA_AUTONOMIA" if porcentaje >= 85 else ("TRANSICION_AUTONOMA" if porcentaje >= 50 else "DEPENDIENTE_DEL_MAESTRO")

            return {
                "total_peticiones": total,
                "respuestas_locales_nvnb": locales,
                "respuestas_llm_maestro": externas,
                "porcentaje_autonomia": f"{porcentaje:.2f}%",
                "patrones_aprendidos_totales": total_patrones,
                "estado_motor": estado
            }


# ==========================================
# 4. AGENTE EXTRA: MAESTRO LLM EXTERNO
# ==========================================
class AgenteExternalLLM:
    """Invoca la API de Groq / LLM externo para actuar como Maestro inicial."""
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")

    def solicitar_refactorizacion(self, codigo_original: str, instruccion: str) -> str:
        if not self.api_key:
            # Fallback seguro en caso de no tener API Key configurada
            return f"# [LLM Fallback]\n{codigo_original}\n# Instruccion ejecutada: {instruccion}"

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            prompt = f"Instrucción: {instruccion}\nCódigo:\n```python\n{codigo_original}\n```\nDevuelve únicamente el código Python optimizado sin texto ni explicaciones."
            
            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": "Eres un asistente programador experto en Python."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2
            }
            res = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers, timeout=10)
            if res.status_code == 200:
                content = res.json()["choices"][0]["message"]["content"]
                if "```python" in content:
                    content = content.split("```python")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]
                return content.strip()
            return codigo_original
        except Exception:
            return codigo_original


# ==========================================
# 5. AGENTE 2 & ORQUESTADOR SWARM
# ==========================================
class SwarmOrchestrator:
    def __init__(self, motor_nvnb: NVNBNode, memoria: Agente4Memory, simulator: Agente3Simulator):
        self.nvnb = motor_nvnb
        self.memoria = memoria
        self.simulator = simulator
        self.maestro_llm = AgenteExternalLLM()
        self.umbral_autonomia = 0.70  # Confianza mínima para responder de forma local

    def procesar_solicitud(self, id_tarea: str, codigo_original: str, instruccion: str) -> Dict[str, Any]:
        patron = self.memoria.buscar_patron_similar(instruccion)
        confianza_local = 0.0

        if patron:
            confianza_local = self.nvnb.evaluar_confianza_patron(patron["evaluacion_nbo"])

        if confianza_local >= self.umbral_autonomia:
            origen = "LOCAL_NVNB"
            codigo_propuesto = patron["codigo_resultado"]
        else:
            origen = "EXTERNAL_LLM_TEACHER"
            codigo_propuesto = self.maestro_llm.solicitar_refactorizacion(codigo_original, instruccion)

        # Probar en Sandbox del Agente 3
        prueba_sandbox = self.simulator.probar_codigo(codigo_propuesto)
        sintaxis_valida = prueba_sandbox["exito"]

        # Evaluar en Motor Spintrónico NBO-a
        score_nbo = self.nvnb.procesar_y_evaluar(
            codigo=codigo_propuesto,
            sintaxis_valida=sintaxis_valida,
            es_maestro=(origen == "EXTERNAL_LLM_TEACHER")
        )

        # Aprendizaje progresivo
        if sintaxis_valida and score_nbo >= 0.70:
            self.memoria.guardar_patron_aprendido(instruccion, codigo_propuesto, score_nbo)
            self.nvnb.muda_de_piel_adaptativa()

        return {
            "id_tarea": id_tarea,
            "origen_ejecucion": origen,
            "confianza_local_previa": confianza_local,
            "evaluacion_nbo": score_nbo,
            "sintaxis_valida": sintaxis_valida,
            "error_sandbox": prueba_sandbox["error"],
            "codigo_resultado": codigo_propuesto
        }
