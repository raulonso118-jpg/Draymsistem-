import numpy as np
import json
import os

DTYPE_INT = np.int8
NUM_MTJ_RRV = 3
BASE_THRESHOLD_THETA = 1.0
CALIBRATION_FACTOR = 1e-10
TOKEN_HASH_BASE = np.sin(211)

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

def security_kill_signal(sync_data=None):
    if sync_data is not None and len(sync_data) > 1:
        token = sync_data[0]
        context_hash = np.mean(np.sin(token))
        if np.isclose(context_hash, TOKEN_HASH_BASE, atol=1e-5):
            payload = sync_data[1:]
            payload_symbols = np.sign(payload)
            if np.isclose(np.sum(payload_symbols), 0.0, atol=1e-5):
                return 0.0
    return 1.0

class NVNB_Node:
    def __init__(self, nombre, depth=1000):
        self.nombre = nombre
        self.depth = depth
        self.file_weights = f"nvnb_{nombre}_mtj.npy"
        self.file_mapa = f"nvnb_{nombre}_mapa.json"
        
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

    def inferir_siguiente(self, palabra, palabras_usadas=None, sync_input=None, temperatura=0.7):
        kill_signal = security_kill_signal(sync_input)
        if kill_signal == 0.0:
            return 0, None, "BLOQUEADO_CCC"

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

        np.save(self.file_weights, np.array(self.W_rrv, dtype=DTYPE_INT))
        with open(self.file_mapa, 'w', encoding='utf-8') as f:
            json.dump(self.mapa_vocal, f, ensure_ascii=False, indent=4)

class NVNB_Orquestador:
    def __init__(self):
        self.maestro = NVNB_Node("maestro_g43", depth=2000)
        self.nodos = {
            "tecnico": NVNB_Node("tecnico_g43", depth=1500),
            "filosofico": NVNB_Node("filosofico_g43", depth=1500)
        }

    def aprendizaje_dirigido(self, frase, destino="maestro"):
        obj = self.maestro if destino == "maestro" else self.nodos.get(destino)
        if not obj: return
        
        tokens = frase.lower().replace(".", " . ").replace(",", " , ").split()
        for palabra in tokens:
            if palabra not in obj.mapa_vocal:
                if len(obj.mapa_vocal) >= obj.depth:
                    obj.muda_de_piel()
                idx = next(idx for idx in range(obj.depth) if idx not in obj.mapa_inverso)
                obj.mapa_vocal[palabra] = idx
                obj.mapa_inverso[idx] = palabra
        
        for i in range(len(tokens)):
            idx_act = obj.mapa_vocal.get(tokens[i], 0)
            if i > 0:
                idx_pre1 = obj.mapa_vocal.get(tokens[i-1], 0)
                obj.aprendizaje_sinaptico(idx_pre1, idx_act)
            if i > 1:
                idx_pre2 = obj.mapa_vocal.get(tokens[i-2], 0)
                obj.aprendizaje_sinaptico(idx_pre2, idx_act)

    def procesar(self, entrada, max_tokens=16, sync_input=None):
        tokens = entrada.lower().replace(".", " . ").split()
        if not tokens: return "..."
        semilla = tokens[-1]

        res = {}
        nodos_totales = {"MAESTRO": self.maestro, **{k.upper(): v for k, v in self.nodos.items()}}
        
        for n_nombre, nodo in nodos_totales.items():
            intensidad, palabra, est = nodo.inferir_siguiente(semilla, sync_input=sync_input)
            res[n_nombre] = (intensidad, palabra, nodo, est)

        ganador = max(res, key=lambda k: res[k][0])
        intensidad_inicial, primera_palabra, nodo_activo, estado = res[ganador]

        if estado == "BLOQUEADO_CCC":
            return "[SISTEMA BLOQUEADO: Kill Signal Activo]"
        if intensidad_inicial == 0 or not primera_palabra:
            return "[Silencio NVNB]"

        respuesta_tokens = []
        palabras_usadas = set()
        palabra_actual = primera_palabra

        for _ in range(max_tokens):
            if palabra_actual == ".": break
            respuesta_tokens.append(palabra_actual)
            palabras_usadas.add(palabra_actual)
            _, siguiente, _ = nodo_activo.inferir_siguiente(palabra_actual, palabras_usadas, sync_input)
            if not siguiente: break
            palabra_actual = siguiente

        return f"[{ganador}-NVNB] {' '.join(respuesta_tokens)}"

    def guardar_todo(self):
        self.maestro.guardar()
        for n in self.nodos.values(): n.guardar()
