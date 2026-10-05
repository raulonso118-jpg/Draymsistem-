import numpy as np
import math
import json
import os
import glob
import importlib
from typing import Dict, Any, List, Tuple

# Importar protocolo base desde agents.py
try:
    from agents import BaseAgente
except ImportError:
    from abc import ABC, abstractmethod
    class BaseAgente(ABC):
        INTENT_TAG: str = "generico"
        EJEMPLOS_ENTRENAMIENTO: List[str] = []
        @abstractmethod
        def ejecutar(self, prompt: str, contexto_global: Dict[str, Any]) -> str:
            pass

# =================================================================
# 1. PUENTE DE MEMORIA COMPARTIDA
# =================================================================

class SharedMemoryBridge:
    def __init__(self, depth=2000):
        self.depth = depth
        self.word2idx: Dict[str, int] = {"<NULL>": 0}
        self.idx2word: Dict[int, str] = {0: "<NULL>"}
        self.next_idx = 1

    def get_or_register(self, word: str) -> int:
        word_clean = word.lower().strip()
        if word_clean not in self.word2idx:
            if self.next_idx >= self.depth:
                return 0
            self.word2idx[word_clean] = self.next_idx
            self.idx2word[self.next_idx] = word_clean
            self.next_idx += 1
        return self.word2idx[word_clean]

    def text_to_spikes_int8(self, text: str) -> np.ndarray:
        tokens = text.lower().split()
        X = np.zeros((1, self.depth), dtype=np.int8)
        for token in tokens:
            idx = self.word2idx.get(token, 0)
            if idx != 0:
                X[0, idx] = 1
        return X

    def text_to_bow_fp32(self, text: str) -> np.ndarray:
        tokens = set(text.lower().split())
        vector = np.zeros((1, self.depth), dtype=np.float32)
        for token in tokens:
            idx = self.word2idx.get(token, 0)
            if 0 < idx < self.depth:
                vector[0, idx] = 1.0
        return vector

    def export_state(self) -> Dict[str, Any]:
        return {
            "depth": self.depth,
            "word2idx": self.word2idx,
            "next_idx": self.next_idx
        }

    def import_state(self, data: Dict[str, Any]):
        self.depth = data["depth"]
        self.word2idx = data["word2idx"]
        self.idx2word = {int(v): k for k, v in self.word2idx.items()}
        self.next_idx = data["next_idx"]


# =================================================================
# 2. CAPA 1: FRONT-ROUTER INT8
# =================================================================

class NVNB_FrontRouter:
    def __init__(self, memory_bridge: SharedMemoryBridge, decay_rate: float = 0.30):
        self.bridge = memory_bridge
        self.depth = memory_bridge.depth
        self.W_mtj = [np.zeros((self.depth, self.depth), dtype=np.int8) for _ in range(3)]
        self.threshold = 1.0
        self.decay_rate = decay_rate
        self.inference_counter = 0

        self.MAX_SPIN = 127
        self.MIN_SPIN = 0
        self.CONSOLIDATION_THRESHOLD = 10

    def fast_route_eval(self, text: str) -> Tuple[bool, str, int]:
        X = self.bridge.text_to_spikes_int8(text)
        active_indices = np.flatnonzero(X)
        if len(active_indices) == 0:
            return False, "DESCONOCIDO", 0

        outputs = []
        for W in self.W_mtj:
            H_pot = X[:, active_indices].dot(W[active_indices, :])
            spike = np.where(np.abs(H_pot) >= self.threshold, np.sign(H_pot), 0).astype(np.int8)
            outputs.append(spike)

        final_spikes = np.sign(np.sum(np.array(outputs), axis=0))[0]
        spiked_indices = np.where(final_spikes > 0)[0]

        if len(spiked_indices) == 0:
            return False, "AMBIGUO", 0

        best_idx = int(spiked_indices[np.argmax(final_spikes[spiked_indices])])
        intensity = int(final_spikes[best_idx])
        target_word = self.bridge.idx2word.get(best_idx, "DESCONOCIDO")

        return True, target_word, intensity

    def reinforce_spin(self, pre_word: str, target_tag: str):
        idx_pre = self.bridge.word2idx.get(pre_word.lower(), 0)
        idx_target = self.bridge.get_or_register(target_tag.lower())

        if idx_pre != 0 and idx_target != 0:
            for W in self.W_mtj:
                current_spin = W[idx_pre, idx_target]
                if current_spin < self.MAX_SPIN:
                    W[idx_pre, idx_target] = min(self.MAX_SPIN, current_spin + 2)

    def apply_proportional_decay(self, force: bool = False):
        self.inference_counter += 1
        if force or (self.inference_counter % 3 == 0):
            for W in self.W_mtj:
                volatile_mask = (W > self.MIN_SPIN) & (W < self.CONSOLIDATION_THRESHOLD)
                random_mask = np.random.rand(*W.shape) < self.decay_rate
                decay_condition = volatile_mask & random_mask
                W[decay_condition] -= 1


# =================================================================
# 3. CAPA 2: DEEP-ORCHESTRATOR NBO-a (FP32)
# =================================================================

def relu_211_vec(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, x / (1.0 + 0.01 * x), 0.01 * x)

def relu_211_derivative(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, 1.0 / ((1.0 + 0.01 * x) ** 2), 0.01)

def softmax_vec(x: np.ndarray) -> np.ndarray:
    x_clamped = np.clip(x - np.max(x, axis=-1, keepdims=True), -500.0, 500.0)
    exp_x = np.exp(x_clamped)
    sum_exp = np.sum(exp_x, axis=-1, keepdims=True)
    return exp_x / np.where(sum_exp > 0, sum_exp, 1.0)

class NBO_DeepOrchestrator:
    def __init__(self, memory_bridge: SharedMemoryBridge, hidden_dim: int, output_intents: List[str]):
        self.bridge = memory_bridge
        self.input_dim = memory_bridge.depth
        self.hidden_dim = hidden_dim
        self.intents = output_intents
        self.intent2idx = {intent: idx for idx, intent in enumerate(output_intents)}

        scale1 = math.sqrt(2.0 / self.input_dim)
        self.W1 = np.random.normal(0, scale1, (self.input_dim, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros((1, hidden_dim), dtype=np.float32)

        scale2 = math.sqrt(2.0 / hidden_dim)
        self.W2 = np.random.normal(0, scale2, (hidden_dim, len(output_intents))).astype(np.float32)
        self.b2 = np.zeros((1, len(output_intents)), dtype=np.float32)

    def evaluate_deep(self, text: str) -> Tuple[str, float]:
        x_vec = self.bridge.text_to_bow_fp32(text)
        z1 = np.dot(x_vec, self.W1) + self.b1
        a1 = relu_211_vec(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        probs = softmax_vec(z2[0])

        max_idx = int(np.argmax(probs))
        return self.intents[max_idx], float(probs[max_idx])

    def train_step(self, text: str, target_intent: str, learning_rate: float = 0.01) -> float:
        if target_intent not in self.intent2idx:
            return 0.0

        target_idx = self.intent2idx[target_intent]
        y_true = np.zeros((1, len(self.intents)), dtype=np.float32)
        y_true[0, target_idx] = 1.0

        x_vec = self.bridge.text_to_bow_fp32(text)
        z1 = np.dot(x_vec, self.W1) + self.b1
        a1 = relu_211_vec(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        probs = softmax_vec(z2)

        loss = -np.sum(y_true * np.log(probs + 1e-8))

        dz2 = probs - y_true
        dW2 = np.dot(a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * relu_211_derivative(z1)
        dW1 = np.dot(x_vec.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

        return float(loss)

    def fit(self, dataset: List[Tuple[str, str]], epochs: int = 50, lr: float = 0.05):
        for epoch in range(epochs):
            total_loss = 0.0
            for text, target in dataset:
                for word in text.split():
                    self.bridge.get_or_register(word)
                total_loss += self.train_step(text, target, learning_rate=lr)


# =================================================================
# 4. NÚCLEO HÍBRIDO Y PERSISTENCIA
# =================================================================

class HybridCoreSystem:
    def __init__(self, depth=2000, confidence_threshold=0.20, intents_l2: List[str] = None):
        self.depth = depth
        self.memory = SharedMemoryBridge(depth=depth)
        self.router_l1 = NVNB_FrontRouter(self.memory, decay_rate=0.30)
        self.intents_l2 = intents_l2 if intents_l2 else ["desconocido"]
        self.orchestrator_l2 = NBO_DeepOrchestrator(self.memory, hidden_dim=32, output_intents=self.intents_l2)
        self.confidence_threshold = confidence_threshold

    def train_orchestrator(self, dataset: List[Tuple[str, str]], epochs: int = 50, lr: float = 0.05):
        if dataset:
            self.orchestrator_l2.fit(dataset, epochs=epochs, lr=lr)

    def process_query_eval(self, user_input: str) -> Tuple[str, float, str]:
        tokens = user_input.split()
        for word in tokens:
            self.memory.get_or_register(word)

        has_conmutated, tag_l1, intensity = self.router_l1.fast_route_eval(user_input)
        if has_conmutated:
            self.router_l1.apply_proportional_decay()
            return tag_l1, float(intensity), "Capa 1 - NVNB INT8"

        predicted_tag, confidence = self.orchestrator_l2.evaluate_deep(user_input)

        if confidence >= self.confidence_threshold and tokens:
            for word in tokens:
                self.router_l1.reinforce_spin(word, predicted_tag)

        self.router_l1.apply_proportional_decay()
        return predicted_tag, confidence, "Capa 2 - NBO-a FP32"

    def save_state(self, filepath_prefix: str = "hybrid_core"):
        metadata = {
            "depth": self.depth,
            "confidence_threshold": self.confidence_threshold,
            "intents_l2": self.intents_l2,
            "decay_rate": self.router_l1.decay_rate,
            "memory": self.memory.export_state()
        }
        with open(f"{filepath_prefix}_meta.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        np.savez_compressed(
            f"{filepath_prefix}_weights.npz",
            w_mtj_0=self.router_l1.W_mtj[0],
            w_mtj_1=self.router_l1.W_mtj[1],
            w_mtj_2=self.router_l1.W_mtj[2],
            l2_w1=self.orchestrator_l2.W1,
            l2_b1=self.orchestrator_l2.b1,
            l2_w2=self.orchestrator_l2.W2,
            l2_b2=self.orchestrator_l2.b2
        )

    def load_state(self, filepath_prefix: str = "hybrid_core") -> bool:
        meta_filename = f"{filepath_prefix}_meta.json"
        weights_filename = f"{filepath_prefix}_weights.npz"

        if not os.path.exists(meta_filename) or not os.path.exists(weights_filename):
            return False

        try:
            with open(meta_filename, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            self.depth = metadata["depth"]
            self.confidence_threshold = metadata["confidence_threshold"]
            self.intents_l2 = metadata["intents_l2"]
            self.memory.import_state(metadata["memory"])

            weights = np.load(weights_filename)
            self.router_l1.W_mtj[0] = weights["w_mtj_0"]
            self.router_l1.W_mtj[1] = weights["w_mtj_1"]
            self.router_l1.W_mtj[2] = weights["w_mtj_2"]

            self.orchestrator_l2 = NBO_DeepOrchestrator(self.memory, hidden_dim=32, output_intents=self.intents_l2)
            self.orchestrator_l2.W1 = weights["l2_w1"]
            self.orchestrator_l2.b1 = weights["l2_b1"]
            self.orchestrator_l2.W2 = weights["l2_w2"]
            self.orchestrator_l2.b2 = weights["l2_b2"]
            return True
        except Exception as e:
            print(f"[Error al cargar estado]: {e}")
            return False


# =================================================================
# 5. ORQUESTADOR PRINCIPAL
# =================================================================

class Orchestrator:
    def __init__(self, ruta_raiz: str = ".", depth: int = 1000):
        self.ruta_raiz = ruta_raiz
        self.contexto_global: Dict[str, Any] = {
            "historial": [],
            "variables_sesion": {},
            "ultimo_agente": None
        }
        self.agentes_registrados: Dict[str, BaseAgente] = {}
        self.dataset_auto_entrenamiento: List[Tuple[str, str]] = []

        # 1. Escanear y cargar dinámicamente archivos 'agente_*.py'
        self._cargar_agentes_dinamicamente()

        intenciones = list(self.agentes_registrados.keys())
        if not intenciones:
            intenciones = ["desconocido"]

        # 2. Inicializar Núcleo Híbrido
        self.core = HybridCoreSystem(depth=depth, confidence_threshold=0.20, intents_l2=intenciones)

        # 3. Intentar cargar estado persistente o entrenar Capa 2
        if not self.core.load_state():
            if self.dataset_auto_entrenamiento:
                self.core.train_orchestrator(self.dataset_auto_entrenamiento, epochs=40, lr=0.08)

    def _cargar_agentes_dinamicamente(self):
        patron = os.path.join(self.ruta_raiz, "agente_*.py")
        archivos_agentes = glob.glob(patron)

        for ruta_archivo in archivos_agentes:
            nombre_modulo = os.path.basename(ruta_archivo)[:-3]
            try:
                modulo = importlib.import_module(nombre_modulo)
                importlib.reload(modulo)

                for atributo in dir(modulo):
                    obj = getattr(modulo, atributo)
                    if isinstance(obj, type) and issubclass(obj, BaseAgente) and obj is not BaseAgente:
                        instancia = obj()
                        tag = instancia.INTENT_TAG
                        self.agentes_registrados[tag] = instancia

                        for ejemplo in instancia.EJEMPLOS_ENTRENAMIENTO:
                            self.dataset_auto_entrenamiento.append((ejemplo, tag))
            except Exception as e:
                print(f"[Error cargando módulo {nombre_modulo}]: {e}")

    def procesar(self, prompt: str) -> str:
        tag_destino, certeza, origen = self.core.process_query_eval(prompt)

        if tag_destino in self.agentes_registrados:
            agente = self.agentes_registrados[tag_destino]
            try:
                respuesta = agente.ejecutar(prompt, self.contexto_global["variables_sesion"])
                self.contexto_global["ultimo_agente"] = tag_destino
                self.contexto_global["historial"].append({
                    "agente": tag_destino,
                    "prompt": prompt,
                    "respuesta": respuesta
                })
                return f"[{origen} -> {tag_destino.upper()}]\nRespuesta: {respuesta}"
            except Exception as e:
                return f"[Error en Agente {tag_destino}]: {str(e)}"
        else:
            return f"[{origen}] No hay un agente registrado para la intención '{tag_destino}'."
