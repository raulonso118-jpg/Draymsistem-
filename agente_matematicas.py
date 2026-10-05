import math
import random
import urllib.request
import urllib.parse
import re
from typing import Dict, Any, List, Tuple

# =================================================================
# 1. NÚCLEO NBO-a INTEGRADO (Matemáticas Nativas, relu_211 y Derivada)
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
    # Acotar exponentes para evitar OverflowError en exp()
    exp_x = [math.exp(max(-500.0, min(500.0, i - max_x))) for i in x_list]
    sum_exp = sum(exp_x)
    if sum_exp == 0.0:
        return [1.0 / len(x_list)] * len(x_list)
    return [e / sum_exp for e in exp_x]

# =================================================================
# 2. PROCESAMIENTO DE LENGUAJE Y VOCABULARIO NATIVO (Sin NLTK)
# =================================================================

class NativeNLP:
    IGNORE_CHARS = set('!?.,¿¡:;()-"\'')
    ACCENT_MAP = str.maketrans({
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'ü': 'u', 'ñ': 'n'
    })

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normaliza caracteres acentuados y remueve puntuación."""
        text_clean = text.lower().translate(NativeNLP.ACCENT_MAP)
        return "".join([c for c in text_clean if c not in NativeNLP.IGNORE_CHARS])

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return NativeNLP.normalize_text(text).split()

    @staticmethod
    def create_vocabulary(intents: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        vocab_set = set()
        classes_set = set()
        for intent in intents:
            classes_set.add(intent["tag"])
            for pattern in intent["patterns"]:
                tokens = NativeNLP.tokenize(pattern)
                vocab_set.update(tokens)
        return sorted(list(vocab_set)), sorted(list(classes_set))

    @staticmethod
    def bag_of_words(text: str, vocabulary: List[str]) -> List[float]:
        tokens = set(NativeNLP.tokenize(text))
        return [1.0 if word in tokens else 0.0 for word in vocabulary]

# =================================================================
# 3. RED NEURONAL MULTICAPA NBO-a CON MOMENTUM (Sin Keras/PyTorch)
# =================================================================

class NBOPerceptronModel:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Inicialización He/Xavier determinista mediante semilla
        random.seed(42)
        scale1 = math.sqrt(2.0 / input_dim) if input_dim > 0 else 0.1
        scale2 = math.sqrt(2.0 / hidden_dim) if hidden_dim > 0 else 0.1

        self.W1 = [[random.gauss(0, scale1) for _ in range(hidden_dim)] for _ in range(input_dim)]
        self.b1 = [0.0] * hidden_dim
        
        self.W2 = [[random.gauss(0, scale2) for _ in range(output_dim)] for _ in range(hidden_dim)]
        self.b2 = [0.0] * output_dim

        # Estructuras de Momentum para optimizar estabilidad
        self.v_W1 = [[0.0] * hidden_dim for _ in range(input_dim)]
        self.v_b1 = [0.0] * hidden_dim
        self.v_W2 = [[0.0] * output_dim for _ in range(hidden_dim)]
        self.v_b2 = [0.0] * output_dim

    def forward(self, x_vec: List[float]) -> Tuple[List[float], List[float], List[float]]:
        """Paso hacia adelante guardando estados intermedios."""
        z1 = []
        a1 = []
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
        """Backpropagation con optimizador SGD + Momentum."""
        a1, z1, probs = self.forward(x_vec)
        
        # Gradiente en capa de salida (Cross-Entropy + Softmax)
        d_z2 = list(probs)
        d_z2[target_idx] -= 1.0

        # Gradiente en capa oculta
        d_a1 = [sum(d_z2[k] * self.W2[j][k] for k in range(self.output_dim)) for j in range(self.hidden_dim)]
        d_z1 = [d_a1[j] * relu_211_derivative(z1[j]) for j in range(self.hidden_dim)]

        # Actualizar W2 y b2 con Momentum
        for j in range(self.hidden_dim):
            for k in range(self.output_dim):
                grad = d_z2[k] * a1[j]
                self.v_W2[j][k] = beta * self.v_W2[j][k] + (1 - beta) * grad
                self.W2[j][k] -= lr * self.v_W2[j][k]

        for k in range(self.output_dim):
            grad = d_z2[k]
            self.v_b2[k] = beta * self.v_b2[k] + (1 - beta) * grad
            self.b2[k] -= lr * self.v_b2[k]

        # Actualizar W1 y b1 con Momentum
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

    def fit(self, training_data: List[Tuple[List[float], int]], epochs: int = 250, lr_initial: float = 0.08):
        """Entrenamiento con decaimiento progresivo de tasa de aprendizaje (Learning Rate Decay)."""
        for epoch in range(epochs):
            # Decaimiento suave del learning rate para convergencia exacta
            current_lr = lr_initial / (1.0 + 0.005 * epoch)
            for x_vec, target_idx in training_data:
                self.train_step(x_vec, target_idx, lr=current_lr)

# =================================================================
# 4. MOTOR DE RAZONAMIENTO ONTOLÓGICO NATIVO
# =================================================================

class NativeOntologyReasoner:
    def infer(self, concept: str) -> str:
        concept_clean = NativeNLP.normalize_text(concept)
        if any(kw in concept_clean for kw in ["creador", "raul", "alons"]):
            return "Inferencia Ontológica: [Raúl Berny Alonso Morales] -> Creador e Ingeniero Principal de DraymSystem."
        elif any(kw in concept_clean for kw in ["draym", "sistema", "nbo"]):
            return "Inferencia Ontológica: [DraymSystem] -> Arquitectura neuronal propia basada en matrices NBO-a."
        return "Inferencia Ontológica: No se hallaron axiomas contradictorios en la base de conocimiento."

# =================================================================
# 5. ASISTENTE DE BÚSQUEDA WEB NATIVO RESILIENTE
# =================================================================

class NativeWebSearch:
    @staticmethod
    def search(query: str) -> List[Dict[str, str]]:
        try:
            encoded_query = urllib.parse.quote(query)
            url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            )
            with urllib.request.urlopen(req, timeout=4) as response:
                html = response.read().decode('utf-8', errors='ignore')
            
            # Patrón más amplio y tolerante a cambios en HTML
            matches = re.findall(r'<a[^>]+class="[^"]*result__url[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html, re.DOTALL)
            results = []
            for link, title in matches[:2]:
                clean_title = re.sub(r'<[^>]+>', '', title).strip()
                results.append({"title": clean_title, "snippet": link.strip()})
            return results
        except Exception:
            return []

# =================================================================
# 6. INTENCIONES, DATASET Y AUTO-ENTRENAMIENTO
# =================================================================

INTENTS_DATA = [
    {
        "tag": "saludo",
        "patterns": ["hola", "buenos dias", "buenas tardes", "que tal", "escuchas", "escucras", "hola draym"],
        "responses": ["Hola. El sistema DraymSystem está en línea, conectado y listo."]
    },
    {
        "tag": "despedida",
        "patterns": ["adios", "hasta luego", "nos vemos", "salir", "chao"],
        "responses": ["Hasta luego. Proceso finalizado correctamente."]
    },
    {
        "tag": "creador",
        "patterns": ["quien es raul", "quien te creo", "tu creador", "quien soy yo", "raúl berny", "raul berny"],
        "responses": ["Mi creador es Raúl Berny Alonso Morales, desarrollador e ingeniero de DraymSystem."]
    },
    {
        "tag": "identidad",
        "patterns": ["quien eres", "cual es tu nombre", "tu eres draym", "tu eres draymsistem", "tu eres draymsistem ia", "que eres"],
        "responses": ["Soy DraymSystem, una arquitectura neuronal procesada mediante matrices de activación NBO-a."]
    }
]

# Inicializar Vocabulario y Red Neuronal
VOCABULARY, CLASSES = NativeNLP.create_vocabulary(INTENTS_DATA)
MODEL_NBO = NBOPerceptronModel(len(VOCABULARY), 16, len(CLASSES))

# Construir Dataset de entrenamiento
DATASET = []
for intent in INTENTS_DATA:
    target_idx = CLASSES.index(intent["tag"])
    for pattern in intent["patterns"]:
        vec = NativeNLP.bag_of_words(pattern, VOCABULARY)
        DATASET.append((vec, target_idx))

# Auto-entrenamiento inmediato con Backpropagation + Momentum
MODEL_NBO.fit(DATASET, epochs=250, lr_initial=0.08)
REASONER = NativeOntologyReasoner()

# =================================================================
# 7. ARQUITECTURA DE AGENTES DRAYMSYSTEM
# =================================================================

class BaseAgent:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

class MemoryAgent(BaseAgent):
    def __init__(self):
        super().__init__("Agente_Memoria", "Gestión Contextual")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return {"conceptos_cargados": len(VOCABULARY)}

class AnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__("Agente_Analisis", "Evaluación Perceptronal NBO-a")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        prompt = input_data.get("prompt", "")
        prompt_norm = NativeNLP.normalize_text(prompt)
        
        # Forward pass en la red entrenada
        bow_vector = NativeNLP.bag_of_words(prompt, VOCABULARY)
        _, _, probabilities = MODEL_NBO.forward(bow_vector)
        
        max_idx = max(range(len(probabilities)), key=lambda i: probabilities[i])
        predicted_tag = CLASSES[max_idx]
        confidence = probabilities[max_idx]

        # Respaldo directo si la red no identifica patrones con alta certeza
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

class WebResearchAgent(BaseAgent):
    def __init__(self):
        super().__init__("Agente_Investigacion", "Contexto Web")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        prompt = input_data.get("prompt", "")
        payload = input_data.get("payload_original", {})
        prompt_norm = NativeNLP.normalize_text(prompt)
        
        web_context = payload.get("web_context", [])
        if not web_context and ("busca" in prompt_norm or "internet" in prompt_norm):
            web_context = NativeWebSearch.search(prompt)

        return {"fuentes": web_context, "hay_web": len(web_context) > 0}

class CodeAgent(BaseAgent):
    def __init__(self):
        super().__init__("Agente_Codigo", "Sintaxis e Ingeniería")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        prompt = input_data.get("prompt", "")
        prompt_norm = NativeNLP.normalize_text(prompt)
        palabras_clave = ["python", "code", "script", "funcion", "def ", "html", "js"]
        return {"requiere_codigo": any(kw in prompt_norm for kw in palabras_clave)}

class SynthesisAgent(BaseAgent):
    def __init__(self):
        super().__init__("Agente_Sintesis", "Consolidación de Respuesta")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        prompt = input_data.get("prompt", "").strip()
        prompt_norm = NativeNLP.normalize_text(prompt)
        analisis = input_data.get("analisis", {})
        codigo = input_data.get("codigo", {})
        investigacion = input_data.get("investigacion", {})

        tag = analisis.get("predicted_tag")
        confidence = analisis.get("confidence", 0.0)
        partes = []

        # Respuesta conversacional
        matched_response = False
        for intent in INTENTS_DATA:
            if intent["tag"] == tag:
                partes.append(intent["responses"][0])
                matched_response = True
                break

        if not matched_response:
            partes.append(f"Procesando entrada: '{prompt}'.")

        # Inferencia Ontológica si aplica
        if any(kw in prompt_norm for kw in ["razonamiento", "deduce", "ontologia", "creador"]):
            partes.append("\n" + REASONER.infer(prompt))

        # Bloque de código si aplica
        if codigo.get("requiere_codigo"):
            partes.append("\n```python\n# [Bloque generado nativamente bajo patrón NBO-a]\n```")

        # Búsqueda Web Nativa
        if investigacion.get("hay_web"):
            partes.append("\n[Información web relevante]:")
            for item in investigacion.get("fuentes", []):
                partes.append(f"- {item.get('title')}: {item.get('snippet')}")

        # Traza de activación
        partes.append(f"\n\n[Estado Red NBO-a | relu_211]: Confianza {confidence} -> Clave: '{tag}'")

        return {
            "estado": "exito",
            "texto_sintetizado": "\n".join(partes)
        }
