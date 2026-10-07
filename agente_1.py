import math
import random
import json
import os
import glob
import importlib
import urllib.request
import urllib.parse
import re
import sys
from typing import Dict, Any, List, Tuple

# =================================================================
# 0. PROTOCOLO BASE Y ENTORNO
# =================================================================

RUTA_LOCAL = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
if RUTA_LOCAL not in sys.path:
    sys.path.insert(0, RUTA_LOCAL)

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
# 1. MATEMÁTICAS NATIVAS NBO-a (relu_211, Derivada y Softmax)
# =================================================================

def relu_211(x: float) -> float:
    """Función de activación nativa NBO-a."""
    return x / (1.0 + 0.01 * x) if x > 0 else 0.01 * x

def relu_211_derivative(x: float) -> float:
    """Derivada exacta de relu_211 para Backpropagation."""
    if x > 0:
        denom = 1.0 + 0.01 * x
        return 1.0 / (denom * denom)
    return 0.01

def softmax(x_list: List[float]) -> List[float]:
    """Softmax nativo numéricamente estable contra Overflow/Underflow."""
    if not x_list:
        return []
    max_x = max(x_list)
    exp_x = [math.exp(max(-500.0, min(500.0, i - max_x))) for i in x_list]
    sum_exp = sum(exp_x)
    if sum_exp == 0.0:
        return [1.0 / len(x_list)] * len(x_list)
    return [e / sum_exp for e in exp_x]

# =================================================================
# 2. TOKENIZADOR SUBPALABRA / BPE NATIVO
# =================================================================

class NativeSubwordTokenizer:
    """
    Tokenizador subpalabra nativo para alimentar al Transformer 
    sin depender de librerías externas.
    """
    def __init__(self, vocab_size: int = 500):
        self.special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        self.vocab: List[str] = list(self.special_tokens)
        self.token2idx: Dict[str, int] = {t: i for i, t in enumerate(self.vocab)}
        self.idx2token: Dict[int, str] = {i: t for i, t in enumerate(self.vocab)}
        self.vocab_size = vocab_size

    def build_vocab(self, corpus: List[str]):
        words = []
        for text in corpus:
            clean = text.lower().replace("\n", " ")
            clean = re.sub(r'[^a-záéíóúñ0-9\s]', '', clean)
            words.extend(clean.split())

        char_counts: Dict[str, int] = {}
        for w in words:
            for char in w:
                char_counts[char] = char_counts.get(char, 0) + 1

        for char, _ in sorted(char_counts.items(), key=lambda x: x[1], reverse=True):
            if char not in self.token2idx and len(self.vocab) < self.vocab_size:
                idx = len(self.vocab)
                self.vocab.append(char)
                self.token2idx[char] = idx
                self.idx2token[idx] = char

        for w in set(words):
            if w not in self.token2idx and len(self.vocab) < self.vocab_size:
                idx = len(self.vocab)
                self.vocab.append(w)
                self.token2idx[w] = idx
                self.idx2token[idx] = w

    def encode(self, text: str) -> List[int]:
        clean = text.lower().replace("\n", " ")
        clean = re.sub(r'[^a-záéíóúñ0-9\s]', '', clean)
        words = clean.split()
        tokens = [self.token2idx["<BOS>"]]

        for w in words:
            if w in self.token2idx:
                tokens.append(self.token2idx[w])
            else:
                for char in w:
                    tokens.append(self.token2idx.get(char, self.token2idx["<UNK>"]))

        tokens.append(self.token2idx["<EOS>"])
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        words = []
        for tid in token_ids:
            tok = self.idx2token.get(tid, "")
            if tok in ["<PAD>", "<BOS>", "<EOS>"]:
                continue
            words.append(tok)
        return " ".join(words)

# =================================================================
# 3. TRANSFORMER AUTORREGRESIVO NATIVO (Sistema 2)
# =================================================================

class NativeTransformerLM:
    """
    Arquitectura Transformer Autorregresiva nativa (Self-Attention + Positional Feed-Forward).
    Genera secuencias token a token para razonamiento o síntesis abierta.
    """
    def __init__(self, vocab_size: int, d_model: int = 32, max_seq_len: int = 64):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        random.seed(42)
        scale = math.sqrt(2.0 / d_model)

        # Token Embeddings & Positional Embeddings
        self.token_emb = [[random.gauss(0, scale) for _ in range(d_model)] for _ in range(vocab_size)]
        self.pos_emb = self._init_sinusoidal_pos_emb(max_seq_len, d_model)

        # Proyecciones Self-Attention (Query, Key, Value, Output)
        self.W_q = [[random.gauss(0, scale) for _ in range(d_model)] for _ in range(d_model)]
        self.W_k = [[random.gauss(0, scale) for _ in range(d_model)] for _ in range(d_model)]
        self.W_v = [[random.gauss(0, scale) for _ in range(d_model)] for _ in range(d_model)]
        self.W_o = [[random.gauss(0, scale) for _ in range(d_model)] for _ in range(d_model)]

        # Feed-Forward Network con activación relu_211
        self.FF1 = [[random.gauss(0, scale) for _ in range(d_model * 2)] for _ in range(d_model)]
        self.FF2 = [[random.gauss(0, scale) for _ in range(d_model)] for _ in range(d_model * 2)]

        # Capa de Proyección de Vocabulario Final (LM Head)
        self.W_head = [[random.gauss(0, scale) for _ in range(vocab_size)] for _ in range(d_model)]

    def _init_sinusoidal_pos_emb(self, max_len: int, d_model: int) -> List[List[float]]:
        pe = [[0.0] * d_model for _ in range(max_len)]
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                div_term = math.exp(i * -(math.log(10000.0) / d_model))
                pe[pos][i] = math.sin(pos * div_term)
                if i + 1 < d_model:
                    pe[pos][i + 1] = math.cos(pos * div_term)
        return pe

    def _mat_mul(self, A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
        rows_A, cols_A = len(A), len(A[0])
        rows_B, cols_B = len(B), len(B[0])
        result = [[0.0] * cols_B for _ in range(rows_A)]
        for i in range(rows_A):
            for j in range(cols_B):
                result[i][j] = sum(A[i][k] * B[k][j] for k in range(cols_A))
        return result

    def forward(self, token_ids: List[int]) -> List[List[float]]:
        seq_len = min(len(token_ids), self.max_seq_len)
        X = []
        for pos in range(seq_len):
            tid = token_ids[pos] if token_ids[pos] < self.vocab_size else 1
            emb = [self.token_emb[tid][d] + self.pos_emb[pos][d] for d in range(self.d_model)]
            X.append(emb)

        # Proyecciones Q, K, V
        Q = self._mat_mul(X, self.W_q)
        K = self._mat_mul(X, self.W_k)
        V = self._mat_mul(X, self.W_v)

        # Self-Attention Escalada con Máscara Autorregresiva
        scores = [[0.0] * seq_len for _ in range(seq_len)]
        scale = math.sqrt(self.d_model)
        for i in range(seq_len):
            for j in range(seq_len):
                if j > i:  # Máscara causal (no ver tokens futuros)
                    scores[i][j] = -1e9
                else:
                    dot = sum(Q[i][d] * K[j][d] for d in range(self.d_model))
                    scores[i][j] = dot / scale

        attn_weights = [softmax(row) for row in scores]
        context = self._mat_mul(attn_weights, V)
        attn_out = self._mat_mul(context, self.W_o)

        # Conexión Residual 1
        X_res1 = [[X[i][d] + attn_out[i][d] for d in range(self.d_model)] for i in range(seq_len)]

        # Feed-Forward con activación relu_211
        ff1_out = self._mat_mul(X_res1, self.FF1)
        ff1_act = [[relu_211(val) for val in row] for row in ff1_out]
        ff2_out = self._mat_mul(ff1_act, self.FF2)

        # Conexión Residual 2
        X_res2 = [[X_res1[i][d] + ff2_out[i][d] for d in range(self.d_model)] for i in range(seq_len)]

        # Proyección final a Logits de Vocabulario
        logits = self._mat_mul(X_res2, self.W_head)
        return logits

    def generate(self, tokenizer: NativeSubwordTokenizer, prompt: str, max_new_tokens: int = 15, temperature: float = 0.7) -> str:
        input_ids = tokenizer.encode(prompt)
        generated_ids = list(input_ids[:-1])  # Remover <EOS> inicial para continuar

        for _ in range(max_new_tokens):
            if len(generated_ids) >= self.max_seq_len:
                break
            logits = self.forward(generated_ids)
            last_logits = logits[-1]

            # Aplicar Temperatura al Softmax
            scaled_logits = [l / max(0.01, temperature) for l in last_logits]
            probs = softmax(scaled_logits)

            # Muestreo estocástico proporcional
            r = random.random()
            acc = 0.0
            next_id = 0
            for idx, p in enumerate(probs):
                acc += p
                if r <= acc:
                    next_id = idx
                    break

            if next_id == tokenizer.token2idx["<EOS>"]:
                break
            generated_ids.append(next_id)

        return tokenizer.decode(generated_ids)

# =================================================================
# 4. MODELO PERCEPTRÓN NBO-a (Sistema 1)
# =================================================================

class NativeNLP:
    IGNORE_CHARS = set('!?.,¿¡:;()-"\'')
    ACCENT_MAP = str.maketrans({'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ü': 'u', 'ñ': 'n'})

    @staticmethod
    def normalize_text(text: str) -> str:
        text_clean = text.lower().translate(NativeNLP.ACCENT_MAP)
        return "".join([c for c in text_clean if c not in NativeNLP.IGNORE_CHARS])

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return NativeNLP.normalize_text(text).split()

    @staticmethod
    def create_vocabulary(intents: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        vocab_set, classes_set = set(), set()
        for intent in intents:
            classes_set.add(intent["tag"])
            for pattern in intent["patterns"]:
                vocab_set.update(NativeNLP.tokenize(pattern))
        return sorted(list(vocab_set)), sorted(list(classes_set))

    @staticmethod
    def bag_of_words(text: str, vocabulary: List[str]) -> List[float]:
        tokens = set(NativeNLP.tokenize(text))
        return [1.0 if word in tokens else 0.0 for word in vocabulary]

class NBOPerceptronModel:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        random.seed(42)
        scale1 = math.sqrt(2.0 / input_dim) if input_dim > 0 else 0.1
        scale2 = math.sqrt(2.0 / hidden_dim) if hidden_dim > 0 else 0.1

        self.W1 = [[random.gauss(0, scale1) for _ in range(hidden_dim)] for _ in range(input_dim)]
        self.b1 = [0.0] * hidden_dim
        self.W2 = [[random.gauss(0, scale2) for _ in range(output_dim)] for _ in range(hidden_dim)]
        self.b2 = [0.0] * output_dim

        self.v_W1 = [[0.0] * hidden_dim for _ in range(input_dim)]
        self.v_b1 = [0.0] * hidden_dim
        self.v_W2 = [[0.0] * output_dim for _ in range(hidden_dim)]
        self.v_b2 = [0.0] * output_dim

    def forward(self, x_vec: List[float]) -> Tuple[List[float], List[float], List[float]]:
        z1, a1 = [], []
        for j in range(self.hidden_dim):
            val = sum(x_vec[i] * self.W1[i][j] for i in range(len(x_vec))) + self.b1[j]
            z1.append(val)
            a1.append(relu_211(val))

        z2 = []
        for k in range(self.output_dim):
            val = sum(a1[j] * self.W2[j][k] for j in range(len(a1))) + self.b2[k]
            z2.append(val)

        probs = softmax(z2)
        return a1, z1, probs

    def train_step(self, x_vec: List[float], target_idx: int, lr: float = 0.05, beta: float = 0.9):
        a1, z1, probs = self.forward(x_vec)
        d_z2 = list(probs)
        d_z2[target_idx] -= 1.0

        d_a1 = [sum(d_z2[k] * self.W2[j][k] for k in range(self.output_dim)) for j in range(self.hidden_dim)]
        d_z1 = [d_a1[j] * relu_211_derivative(z1[j]) for j in range(self.hidden_dim)]

        for j in range(self.hidden_dim):
            for k in range(self.output_dim):
                grad = d_z2[k] * a1[j]
                self.v_W2[j][k] = beta * self.v_W2[j][k] + (1 - beta) * grad
                self.W2[j][k] -= lr * self.v_W2[j][k]

        for k in range(self.output_dim):
            grad = d_z2[k]
            self.v_b2[k] = beta * self.v_b2[k] + (1 - beta) * grad
            self.b2[k] -= lr * self.v_b2[k]

        for i in range(self.input_dim):
            if x_vec[i] != 0:
                for j in range(self.hidden_dim):
                    grad = d_z1[j] * x_vec[i]
                    self.v_W1[i][j] = beta * self.v_W1[i][j] + (1 - beta) * grad
                    self.W1[i][j] -= lr * self.v_W1[i][j]

        for j in range(self.hidden_dim):
            grad = d_z1[j]
            self.v_b1[j] = beta * self.v_b1[j] + (1 - beta) * grad
            self.b1[j] -= lr * self.v_b1[j]

    def fit(self, training_data: List[Tuple[List[float], int]], epochs: int = 150, lr_initial: float = 0.08):
        for epoch in range(epochs):
            current_lr = lr_initial / (1.0 + 0.005 * epoch)
            for x_vec, target_idx in training_data:
                self.train_step(x_vec, target_idx, lr=current_lr)

# =================================================================
# 5. DATASET E INICIALIZACIÓN DE AMBOS SISTEMAS
# =================================================================

INTENTS_DATA = [
    {
        "tag": "saludo",
        "patterns": ["hola", "buenos dias", "buenas tardes", "que tal", "escuchas", "hola draym"],
        "responses": ["Hola. El sistema DraymSystem está en línea, conectado y listo."]
    },
    {
        "tag": "despedida",
        "patterns": ["adios", "hasta luego", "nos vemos", "salir", "chao"],
        "responses": ["Hasta luego. Proceso finalizado correctamente."]
    },
    {
        "tag": "creador",
        "patterns": ["quien es raul", "quien te creo", "tu creador", "quien soy yo", "raul berny"],
        "responses": ["Mi creador es Raúl Berny Alonso Morales, desarrollador e ingeniero de DraymSystem."]
    },
    {
        "tag": "identidad",
        "patterns": ["quien eres", "cual es tu nombre", "tu eres draym", "que eres"],
        "responses": ["Soy DraymSystem, una arquitectura neuronal híbrida acoplada a un Transformer NBO-a."]
    }
]

VOCABULARY, CLASSES = NativeNLP.create_vocabulary(INTENTS_DATA)
MODEL_NBO = NBOPerceptronModel(len(VOCABULARY), 16, len(CLASSES))

DATASET = []
for intent in INTENTS_DATA:
    target_idx = CLASSES.index(intent["tag"])
    for pattern in intent["patterns"]:
        vec = NativeNLP.bag_of_words(pattern, VOCABULARY)
        DATASET.append((vec, target_idx))

MODEL_NBO.fit(DATASET, epochs=150, lr_initial=0.08)

# Inicializar Tokenizador BPE y Transformer Autorregresivo
TOKENIZER = NativeSubwordTokenizer(vocab_size=250)
corpus_textos = [p for intent in INTENTS_DATA for p in intent["patterns"]] + [r for intent in INTENTS_DATA for r in intent["responses"]]
TOKENIZER.build_vocab(corpus_textos)
TRANSFORMER_LLM = NativeTransformerLM(vocab_size=len(TOKENIZER.vocab), d_model=32, max_seq_len=64)

# =================================================================
# 6. ARQUITECTURA DE MICRO-AGENTES
# =================================================================

class BaseAgentInternal:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

class MemoryAgent(BaseAgentInternal):
    def __init__(self):
        super().__init__("Agente_Memoria", "Gestión Contextual")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return {"conceptos_cargados": len(VOCABULARY), "tokens_transformer": len(TOKENIZER.vocab)}

class AnalysisAgent(BaseAgentInternal):
    def __init__(self):
        super().__init__("Agente_Analisis", "Evaluación Híbrida NBO-a + Transformer")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        prompt = input_data.get("prompt", "")
        prompt_norm = NativeNLP.normalize_text(prompt)

        bow_vector = NativeNLP.bag_of_words(prompt, VOCABULARY)
        _, _, probabilities = MODEL_NBO.forward(bow_vector)

        max_idx = max(range(len(probabilities)), key=lambda i: probabilities[i])
        predicted_tag = CLASSES[max_idx]
        confidence = probabilities[max_idx]

        # Respaldo directo por coincidencias de patrón
        matched_tag = None
        for intent in INTENTS_DATA:
            for pattern in intent["patterns"]:
                if NativeNLP.normalize_text(pattern) in prompt_norm:
                    matched_tag = intent["tag"]
                    break
            if matched_tag:
                break

        final_tag = matched_tag if matched_tag and confidence < 0.35 else predicted_tag

        return {
            "predicted_tag": final_tag,
            "confidence": round(confidence, 4),
            "probabilities": [round(p, 4) for p in probabilities]
        }

class CodeAgent(BaseAgentInternal):
    def __init__(self):
        super().__init__("Agente_Codigo", "Sintaxis e Ingeniería")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        prompt_norm = NativeNLP.normalize_text(input_data.get("prompt", ""))
        palabras_clave = ["python", "code", "script", "funcion", "def ", "html", "js"]
        return {"requiere_codigo": any(kw in prompt_norm for kw in palabras_clave)}

class SynthesisAgent(BaseAgentInternal):
    def __init__(self):
        super().__init__("Agente_Sintesis", "Consolidación de Respuesta")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        prompt = input_data.get("prompt", "").strip()
        analisis = input_data.get("analisis", {})
        codigo = input_data.get("codigo", {})

        tag = analisis.get("predicted_tag")
        confidence = analisis.get("confidence", 0.0)
        partes = []

        # ENRUTAMIENTO HÍBRIDO ESPECULATIVO (System 1 vs System 2)
        if confidence >= 0.45:
            # Sistema 1: Respuesta rápida por plantilla directa NBO-a
            for intent in INTENTS_DATA:
                if intent["tag"] == tag:
                    partes.append(intent["responses"][0])
                    break
            origen_respuesta = "Sistema 1 (Reflejo NBO-a)"
        else:
            # Sistema 2: Delegación al Transformer Autorregresivo
            generacion_llm = TRANSFORMER_LLM.generate(TOKENIZER, prompt, max_new_tokens=12, temperature=0.6)
            if generacion_llm.strip():
                partes.append(f"DraymSystem LLM: {generacion_llm}")
            else:
                partes.append(f"Procesando consulta abierta: '{prompt}'.")
            origen_respuesta = "Sistema 2 (Transformer Autorregresivo)"

        if codigo.get("requiere_codigo"):
            partes.append("\n```python\n# [Bloque generado nativamente bajo patrón NBO-a]\n```")

        partes.append(f"\n\n[{origen_respuesta} | relu_211]: Confianza NBO {confidence} -> Tag: '{tag}'")

        return {
            "estado": "exito",
            "texto_sintetizado": "\n".join(partes)
        }

# =================================================================
# 7. INTEGRACIÓN CON EL ORQUESTADOR PRINCIPAL (agents.py)
# =================================================================

class AgenteDraymMultitarea(BaseAgente):
    """
    Agente registrado en la raíz (agente_*.py) para comunicarse con el Orchestrator.
    """
    INTENT_TAG: str = "draym_core"
    EJEMPLOS_ENTRENAMIENTO: List[str] = [
        "hola draym",
        "quien te creo",
        "quien es raul berny",
        "explicame un concepto",
        "escribe un codigo en python"
    ]

    def __init__(self):
        self.memory = MemoryAgent()
        self.analysis = AnalysisAgent()
        self.code = CodeAgent()
        self.synthesis = SynthesisAgent()

    def ejecutar(self, prompt: str, contexto_global: Dict[str, Any]) -> str:
        payload = {
            "prompt": prompt,
            "payload_original": contexto_global
        }

        payload["memoria"] = self.memory.process(payload)
        payload["analisis"] = self.analysis.process(payload)
        payload["codigo"] = self.code.process(payload)

        resultado = self.synthesis.process(payload)
        return resultado["texto_sintetizado"]
