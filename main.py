import sys
import io
import json
import ast
import traceback
import sqlite3
import time
import os
import numpy as np
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Plataforma NVNB G4.3 & Enjambre NBO-a",
    description="Sistema multi-agente con motor neuromórfico spintrónico.",
    version="2.0.0"
)

# RUTA DE ALMACENAMIENTO TEMPORAL PERMITIDA EN RENDER
TMP_DIR = "/tmp"

# =====================================================================
# AGENTES DEL ENJAMBRE
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
            "plan": f"Refactorizar módulo '{id_tarea}' garantizando estabilidad."
        }

class Agente2Coder:
    def __init__(self):
        self.nombre = "Agente2_Coder_Clean"

    def generar_codigo_limpio(self, instrucciones: str, codigo_base: str) -> str:
        return codigo_base.replace("\r\n", "\n").strip()

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
            code_compiled = compile(codigo_python, filename="<sandbox>", mode="exec")
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

class Agente4Memory:
    def __init__(self, db_path: str = os.path.join(TMP_DIR, "memoria_orquestador.db")):
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

# =====================================================================
# MOTOR NEUROMÓRFICO NVNB G4.3 (RUTAS /tmp Y KILL-SWITCH DESACTIVADO)
# =====================================================================
DTYPE_INT = np.int8
NUM_MTJ_RRV = 3
BASE_THRESHOLD_THETA = 1.0
CALIBRATION_FACTOR = 1e-10

def noise_simulation(H_potential):
    return np.random.normal(0, CALIBRATION_FACTOR, H_potential.shape)

def activation_and_spin_conmutation(H_potential, threshold):
    H_noisy = H_potential + noise_simulation(H_potential)
    H_sign = np.sign(H_noisy).astype(DTYPE_INT)
    spike_condition = np.abs(H_noisy) >= threshold
    return np.where(spike_condition, H_sign, 0).astype(DTYPE_INT)

def recursive_redundancy_voting(outputs_list):
    outputs_sum = np.sum(np.array(outputs_list), axis=0)
    return np.sign(outputs_sum).astype(DTYPE_INT)

class NVNB_Node:
    def __init__(self, nombre, depth=1000):
        self.nombre = nombre
        self.depth = depth
        # Se guardan en /tmp para evitar errores de permisos de lectura/escritura
        self.file_weights = os.path.join(TMP_DIR, f"nvnb_{nombre}_mtj.npy")
        self.file_mapa = os.path.join(TMP_DIR, f"nvnb_{nombre}_mapa.json")
        
        self.mapa_vocal, self.mapa_inverso = self._cargar_memoria()
        self.W_rrv = self._cargar_pesos()
        self.dynamic_theta = np.full((1, depth), BASE_THRESHOLD_THETA, dtype=np.float32)

    def _cargar_memoria(self):
        if os.path.exists(self.file_mapa):
            try:
                with open(self.file_mapa, 'r', encoding='utf-8') as f:
                    mapa = json.load(f)
                return mapa, {int(v): k for k, v in mapa.items()}
            except: pass
        return {"<NULL>": 0}, {0: "<NULL>"}

    def _cargar_pesos(self):
        if os.path.exists(self.file_weights):
            try:
                pesos = np.load(self.file_weights)
                if pesos.shape == (NUM_MTJ_RRV, self.depth, self.depth):
                    return [pesos[i] for i in range(NUM_MTJ_RRV)]
            except: pass
        return [np.zeros((self.depth, self.depth), dtype=DTYPE_INT) for _ in range(NUM_MTJ_RRV)]

    def muda_de_piel(self):
        actividad = np.sum(np.abs(self.W_rrv[0]), axis=1)
        actividad[0] = 1e9
        num_eliminar = int(self.depth * 0.2)
        indices_debiles = np.argsort(actividad)[:num_eliminar]
        
        for idx in indices_debiles:
            palabra = self.mapa_inverso.get(idx)
            if palabra:
                del self.mapa_vocal[palabra]
                del self.mapa_inverso[idx]
                for w in self.W_rrv:
                    w[idx, :] = 0
                    w[:, idx] = 0

    def aprendizaje_sinaptico(self, idx_pre, idx_act):
        for W in self.W_rrv:
            if W[idx_pre, idx_act] < 1:
                W[idx_pre, idx_act] += 1

    def inferir_siguiente(self, palabra, palabras_usadas=None, temperatura=0.7):
        # Kill Signal FIJO en 1.0 para evitar bloqueos no deseados
        kill_signal = 1.0

        idx = self.mapa_vocal.get(palabra.lower(), 0)
        if idx == 0:
            return 0, None, "SILENCIO"

        X = np.zeros((1, self.depth), dtype=DTYPE_INT)
        X[0, idx] = 1
        active_indices = np.flatnonzero(X)
        X_active = X[:, active_indices]

        all_outputs = []
        for W in self.W_rrv:
            W_active = W[active_indices, :]
            H_pot = X_active.dot(W_active) * kill_signal
            spike = activation_and_spin_conmutation(H_pot, self.dynamic_theta)
            all_outputs.append(spike)

        final_spikes = recursive_redundancy_voting(all_outputs)[0]

        if palabras_usadas:
            for p_usada in palabras_usadas:
                idx_u = self.mapa_vocal.get(p_usada, 0)
                if idx_u != 0:
                    final_spikes[idx_u] = 0

        spiked_indices = np.where(final_spikes > 0)[0]
        if len(spiked_indices) == 0:
            return 0, None, "SIN_CONMUTACION"

        potenciales = final_spikes[spiked_indices].astype(np.float32) / max(temperatura, 0.1)
        exp_p = np.exp(potenciales - np.max(potenciales))
        probs = exp_p / np.sum(exp_p)
        
        max_idx = np.random.choice(spiked_indices, p=probs)
        self.dynamic_theta[0, max_idx] += 0.02

        return final_spikes[max_idx], self.mapa_inverso.get(max_idx), "EXITO"

    def guardar(self):
        for W in self.W_rrv:
            mask = (np.random.rand(*W.shape) < 0.05) & (W > 0)
            W[mask] -= 1

        try:
            np.save(self.file_weights, np.array(self.W_rrv, dtype=DTYPE_INT))
            with open(self.file_mapa, 'w', encoding='utf-8') as f:
                json.dump(self.mapa_vocal, f, ensure_ascii=False, indent=4)
        except Exception:
            pass

class NVNB_Orquestador:
    def __init__(self):
        # Orquestador Maestro Directo
        self.maestro = NVNB_Node("maestro_g43", depth=2000)

    def aprendizaje_dirigido(self, frase):
        tokens = frase.lower().replace(".", " . ").replace(",", " , ").split()
        for palabra in tokens:
            if palabra not in self.maestro.mapa_vocal:
                if len(self.maestro.mapa_vocal) >= self.maestro.depth:
                    self.maestro.muda_de_piel()
                idx = next(idx for idx in range(self.maestro.depth) if idx not in self.maestro.mapa_inverso)
                self.maestro.mapa_vocal[palabra] = idx
                self.maestro.mapa_inverso[idx] = palabra
        
        for i in range(len(tokens)):
            idx_act = self.maestro.mapa_vocal.get(tokens[i], 0)
            if i > 0:
                idx_pre1 = self.maestro.mapa_vocal.get(tokens[i-1], 0)
                self.maestro.aprendizaje_sinaptico(idx_pre1, idx_act)
            if i > 1:
                idx_pre2 = self.maestro.mapa_vocal.get(tokens[i-2], 0)
                self.maestro.aprendizaje_sinaptico(idx_pre2, idx_act)

    def procesar(self, entrada, max_tokens=16):
        tokens = entrada.lower().replace(".", " . ").split()
        if not tokens: return "..."
        semilla = tokens[-1]

        intensidad_inicial, primera_palabra, estado = self.maestro.inferir_siguiente(semilla)

        if intensidad_inicial == 0 or not primera_palabra:
            return "[NVNB Maestro: Aprendiendo contexto...]"

        respuesta_tokens = []
        palabras_usadas = set()
        palabra_actual = primera_palabra

        for _ in range(max_tokens):
            if palabra_actual == ".": break
            respuesta_tokens.append(palabra_actual)
            palabras_usadas.add(palabra_actual)
            _, siguiente, _ = self.maestro.inferir_siguiente(palabra_actual, palabras_usadas)
            if not siguiente: break
            palabra_actual = siguiente

        return f"[NVNB-Maestro] {' '.join(respuesta_tokens)}"

    def guardar_todo(self):
        self.maestro.guardar()

# =====================================================================
# INICIALIZACIÓN DE COMPONENTES
# =====================================================================
planner = Agente1Planner()
coder = Agente2Coder()
simulator = Agente3Simulator()
memory = Agente4Memory()
tactical = Agente5Tactical()
nvnb_ia = NVNB_Orquestador()

class SolicitudChat(BaseModel):
    mensaje: str
    aprender: bool = True

class SolicitudRefactorizacion(BaseModel):
    id_tarea: str
    nombre_modulo: str
    codigo_original: str
    instrucciones: str
    soporte_hardware: Optional[Dict[str, str]] = None

@app.get("/")
def estado_sistema():
    return {
        "sistema": "NVNB G4.3 & NBO-a Multi-Agent System",
        "estado": "OPERATIVO",
        "agentes": [planner.nombre, coder.nombre, simulator.nombre, memory.nombre, tactical.nombre],
        "motor_ia_generativa": "NVNB Spintronic Neuromorphic Core G4.3 (Maestro)"
    }

@app.post("/chat")
def chatear_con_nvnb(solicitud: SolicitudChat):
    respuesta = nvnb_ia.procesar(solicitud.mensaje)
    
    if solicitud.aprender:
        nvnb_ia.aprendizaje_dirigido(solicitud.mensaje)
        if not respuesta.startswith("[NVNB Maestro:"):
            limpia_resp = respuesta.split("]", 1)[-1].strip()
            nvnb_ia.aprendizaje_dirigido(f"{solicitud.mensaje} {limpia_resp}")
        nvnb_ia.guardar_todo()

    memory.registrar_historial("usuario", solicitud.mensaje)
    memory.registrar_historial("ia_nvnb", respuesta)

    return {
        "entrada": solicitud.mensaje,
        "respuesta_nvnb": respuesta,
        "memoria_preservada": solicitud.aprender
    }

@app.post("/procesar_codigo")
def refactorizar_codigo(solicitud: SolicitudRefactorizacion) -> Dict[str, Any]:
    diagnostico_hw = None
    if solicitud.soporte_hardware:
        etapa = solicitud.soporte_hardware.get("etapa", "")
        sintoma = solicitud.soporte_hardware.get("sintoma", "")
        diagnostico_hw = tactical.diagnosticar_etapa(etapa, sintoma)

    plan = planner.planificar_y_validar(
        id_tarea=solicitud.id_tarea,
        codigo_bruto=solicitud.codigo_original,
        instrucciones=solicitud.instrucciones
    )

    if not plan["ast_valido"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Error sintáctico AST: {plan['error_ast']}"
        )

    codigo_refactorizado = coder.generar_codigo_limpio(
        instrucciones=solicitud.instrucciones,
        codigo_base=solicitud.codigo_original
    )

    resultado_simulacion = simulator.simular_y_evaluar(
        id_tarea=solicitud.id_tarea,
        codigo_python=codigo_refactorizado
    )

    if resultado_simulacion["aprobado_para_despliegue"]:
        memory.guardar_modulo_aprobado(
            id_tarea=solicitud.id_tarea,
            nombre_modulo=solicitud.nombre_modulo,
            codigo=codigo_refactorizado,
            score_nbo=resultado_simulacion["vector_evaluacion_nbo"]
        )

    return {
        "id_tarea": solicitud.id_tarea,
        "nombre_modulo": solicitud.nombre_modulo,
        "estado": "VALIDADO_Y_GUARDADO" if resultado_simulacion["aprobado_para_despliegue"] else "REVISAR",
        "evaluacion_nbo": resultado_simulacion["vector_evaluacion_nbo"],
        "diagnostico_hardware": diagnostico_hw,
        "codigo_resultado": codigo_refactorizado
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
