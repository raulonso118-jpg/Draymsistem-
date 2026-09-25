import sys
import io
import json
import ast
import traceback
import sqlite3
import time
import os
from typing import Dict, Any, List, Optional

# =====================================================================
# AGENTE 1: PLANNER & AST VERIFIER
# =====================================================================
class Agente1Planner:
    def __init__(self):
        self.nombre = "Agente1_Planner_AST"

    def planificar_y_validar(self, id_tarea: str, codigo_bruto: str, instrucciones: str) -> Dict[str, Any]:
        es_valido = True
        error_msg = ""
        try:
            ast.parse(codigo_bruto)
        except SyntaxError as e:
            es_valido = False
            error_msg = f"Sintaxis inválida en línea {e.lineno}: {e.msg}"

        return {
            "agente": self.nombre,
            "id_tarea": id_tarea,
            "ast_valido": es_valido,
            "error_ast": error_msg,
            "plan": f"Refactorizar módulo '{id_tarea}' garantizando preservación total de interfaz gráfica y lógica."
        }

# =====================================================================
# AGENTE 2: CODER (CLEAN CODE ENGINE)
# =====================================================================
class Agente2Coder:
    def __init__(self):
        self.nombre = "Agente2_Coder_Clean"

    def generar_codigo_limpio(self, instrucciones: str, codigo_base: str) -> str:
        codigo_limpio = codigo_base.replace("\r\n", "\n").strip()
        return codigo_limpio

# =====================================================================
# INTEGRACIÓN RUNTIME NBO-a PARA AGENTE 3
# =====================================================================
W1 = [[0.009827, -1.673184, -0.479324, 0.022902, -1.203937, -0.275713, -0.499627, -0.060934], 
      [0.397949, -0.967759, 0.255584, 0.495538, 1.210088, -0.242116, -0.22057, -0.46945], 
      [0.676937, -0.038244, -0.667513, 0.41029, 0.771993, -0.633136, 0.531934, 1.039185], 
      [1.463662, -1.001581, -0.618165, 0.681863, 1.606629, -0.392373, 0.043346, -1.428108]]
b1 = [-0.128037, 0.001638, -0.001312, -0.007377, 0.118432, 0.000127, 0.017772, -0.048603]
W2 = [[0.486183, 0.418387], [0.316011, -0.630265], [-0.262076, 0.518192], [0.360306, 0.137607], 
      [-0.302119, -0.714048], [-0.167102, -0.211522], [-0.187807, 0.061643], [-0.076087, 0.2535]]
b2 = [0.151399, -0.178274]

def relu_211(x: float) -> float:
    return x / (1.0 + 0.01 * x) if x > 0 else 0.01 * x

def nbo_a_predict(inputs: List[float]) -> List[float]:
    h = [relu_211(sum(inputs[i] * W1[i][j] for i in range(len(inputs))) + b1[j]) for j in range(len(b1))]
    o = [relu_211(sum(h[j] * W2[j][k] for j in range(len(h))) + b2[k]) for k in range(len(b2))]
    return o

# =====================================================================
# AGENTE 3: SIMULATOR & NBO-a EVALUATOR
# =====================================================================
class Agente3Simulator:
    def __init__(self):
        self.nombre = "Agente3_Simulator_NBO"

    def _ejecutar_en_sandbox(self, codigo_python: str) -> Dict[str, Any]:
        buffer_salida = io.StringIO()
        entorno_local = {}
        sys_stdout_original = sys.stdout
        sys.stdout = buffer_salida
        ejecucion_exitosa = False
        error_msg = ""

        try:
            code_compiled = compile(codigo_python, filename="<agente3_sandbox>", mode="exec")
            exec(code_compiled, {}, entorno_local)
            ejecucion_exitosa = True
        except Exception:
            error_msg = traceback.format_exc()
        finally:
            sys.stdout = sys_stdout_original

        return {
            "exito": ejecucion_exitosa,
            "stdout": buffer_salida.getvalue().strip(),
            "error": error_msg,
            "elementos_creados": list(entorno_local.keys())
        }

    def simular_y_evaluar(self, id_tarea: str, codigo_python: str) -> Dict[str, Any]:
        res_sandbox = self._ejecutar_en_sandbox(codigo_python)
        lineas = len(codigo_python.splitlines())
        elementos = len(res_sandbox["elementos_creados"])
        
        vector_entrada = [
            1.0 if res_sandbox["exito"] else 0.0,
            min(lineas / 100.0, 1.0),
            min(elementos / 10.0, 1.0),
            1.0 if len(res_sandbox["stdout"]) > 0 else 0.5
        ]
        
        score_nbo = nbo_a_predict(vector_entrada)
        es_estable = score_nbo[0] > score_nbo[1] or res_sandbox["exito"]

        return {
            "agente": self.nombre,
            "id_tarea": id_tarea,
            "estado": "SIMULACION_EXITOSA" if es_estable else "ERROR_SIMULACION",
            "prueba_sandbox": res_sandbox,
            "vector_evaluacion_nbo": [round(x, 4) for x in score_nbo],
            "aprobado_para_despliegue": es_estable
        }

# =====================================================================
# AGENTE 4: MEMORY CORE (SQLITE3)
# =====================================================================
class Agente4Memory:
    def __init__(self, db_path: str = "memoria_orquestador.db"):
        self.nombre = "Agente4_Memory_Core"
        self.db_path = db_path
        self._inicializar_tabla()

    def _conectar(self):
        return sqlite3.connect(self.db_path)

    def _inicializar_tabla(self):
        with self._conectar() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS modulos_codigo (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    id_tarea TEXT UNIQUE,
                    nombre_modulo TEXT,
                    codigo TEXT,
                    score_nbo TEXT,
                    timestamp REAL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS historial_contexto (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rol TEXT,
                    contenido TEXT,
                    timestamp REAL
                )
            """)
            conn.commit()

    def guardar_modulo_aprobado(self, id_tarea: str, nombre_modulo: str, codigo: str, score_nbo: List[float]) -> Dict[str, Any]:
        score_json = json.dumps(score_nbo)
        ts = time.time()
        try:
            with self._conectar() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO modulos_codigo (id_tarea, nombre_modulo, codigo, score_nbo, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (id_tarea, nombre_modulo, codigo, score_json, ts))
                conn.commit()
            return {"agente": self.nombre, "estado": "PERSISTENCIA_EXITOSA", "id_tarea": id_tarea}
        except Exception as e:
            return {"agente": self.nombre, "estado": "ERROR_PERSISTENCIA", "detalle": str(e)}

    def registrar_historial(self, rol: str, contenido: str):
        with self._conectar() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO historial_contexto (rol, contenido, timestamp) VALUES (?, ?, ?)",
                           (rol, contenido, time.time()))
            conn.commit()

# =====================================================================
# AGENTE 5: MITHRA TACTICAL (HEURÍSTICA Y HARDWARE)
# =====================================================================
class Agente5Tactical:
    def __init__(self):
        self.nombre = "Agente5_Mithra_Tactical"
        self.base_conocimiento = {
            "FUENTES_CONMUTADAS": {
                "sin_voltaje_standby": [
                    "Verificar fusible de entrada y MOV.",
                    "Comprobar puente rectificador.",
                    "Medir voltaje en capacitor principal (310V/160V DC).",
                    "Revisar resistencia de arranque del PWM."
                ]
            },
            "LINEA_BLANCA_INVERTER": {
                "error_ipm_sobrecorriente": [
                    "Medir resistencia entre fases U, V, W del compresor.",
                    "Verificar aislamiento a masa de cada fase.",
                    "Comprobar alimentación del IPM (15V driver / Bus DC)."
                ]
            }
        }

    def diagnosticar_etapa(self, etapa: str, sintoma: str) -> Dict[str, Any]:
        etapa_key = etapa.upper().replace(" ", "_")
        sintoma_key = sintoma.lower().replace(" ", "_")
        if etapa_key in self.base_conocimiento and sintoma_key in self.base_conocimiento[etapa_key]:
            return {
                "agente": self.nombre,
                "estado": "DIAGNOSTICO_ENCONTRADO",
                "procedimiento": self.base_conocimiento[etapa_key][sintoma_key]
            }
        return {
            "agente": self.nombre,
            "estado": "DIAGNOSTICO_GENERICO",
            "procedimiento": ["Inspección visual de componentes", "Prueba de ESR en capacitores", "Inyección de voltaje secundario"]
        }
