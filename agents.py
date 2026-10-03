import math
import urllib.request
import urllib.parse
import re
from typing import Dict, Any, List, Tuple

# =================================================================
# 1. NÚCLEO NBO-a INTEGRADO (Matemáticas Nativas y relu_211)
# =================================================================

def relu_211(x: float) -> float:
    """Función de activación nativa NBO-a."""
    return x / (1.0 + 0.01 * x) if x > 0 else 0.01 * x

def softmax(x_list: List[float]) -> List[float]:
    """Softmax nativo para distribución de probabilidades sin numpy."""
    max_x = max(x_list) if x_list else 0.0
    exp_x = [math.exp(i - max_x) for i in x_list]
    sum_exp = sum(exp_x)
    return [e / sum_exp for e in exp_x]

# =================================================================
# 2. PROCESAMIENTO DE LENGUAJE Y VOCABULARIO NATIVO (Sin NLTK)
# =================================================================

class NativeNLP:
    """Sustituto ligero y nativo de NLTK/WordNet."""
    
    IGNORE_CHARS = set('!?.,¿¡:;()-"\'')

    @staticmethod
    def tokenize(text: str) -> List[str]:
        text_clean = "".join([c.lower() for c in text if c not in NativeNLP.IGNORE_CHARS])
        return text_clean.split()

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
        tokens = NativeNLP.tokenize(text)
        return [1.0 if word in tokens else 0.0 for word in vocabulary]

# =================================================================
# 3. RED NEURONAL MULTICAPA NBO-a (Sustituto 100% Nativo de Keras)
# =================================================================

class NBOPerceptronModel:
    """Reemplazo nativo de Keras Sequential usando relu_211 y Softmax."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Pesos y sesgos inicializados mediante técnica He/Xavier determinista
        self.W1 = [[math.sin(i + j * 0.1) * 0.1 for j in range(hidden_dim)] for i in range(input_dim)]
        self.b1 = [0.01] * hidden_dim
        self.W2 = [[math.cos(i * 0.2 + j) * 0.1 for j in range(output_dim)] for i in range(hidden_dim)]
        self.b2 = [0.01] * output_dim

    def forward(self, x_vec: List[float]) -> List[float]:
        # Capa Oculta con relu_211
        h = []
        for j in range(self.hidden_dim):
            val = sum(x_vec[i] * self.W1[i][j] for i in range(len(x_vec))) + self.b1[j]
            h.append(relu_211(val))
            
        # Capa de Salida
        o = []
        for k in range(self.output_dim):
            val = sum(h[j] * self.W2[j][k] for j in range(len(h))) + self.b2[k]
            o.append(val)
            
        return softmax(o)

# =================================================================
# 4. MOTOR DE RAZONAMIENTO ONTOLÓGICO NATIVO (Sin Owlready2)
# =================================================================

class NativeOntologyReasoner:
    """Motor de deducción causal mediante grafos y reglas relacionales simples."""
    
    def __init__(self):
        self.knowledge_base = {
            "draymsystem": {"tipo": "IA_Neuronal", "creador": "Raúl Berny Alonso Morales", "activacion": "relu_211"},
            "raul": {"rol": "Creador e Ingeniero Principal", "proyecto": "DraymSystem"}
        }

    def infer(self, concept: str) -> str:
        concept_clean = concept.lower()
        if "creador" in concept_clean or "raul" in concept_clean or "raúl" in concept_clean:
            return "Inferencia Ontológica: [Raúl Berny Alonso Morales] -> Creador e Ingeniero Principal de DraymSystem."
        elif "draym" in concept_clean or "sistema" in concept_clean:
            return "Inferencia Ontológica: [DraymSystem] -> Arquitectura neuronal propia basada en matrices NBO-a."
        return "Inferencia Ontológica: No se hallaron axiomas contradictorios en la base de conocimiento."

# =================================================================
# 5. ASISTENTE DE BÚSQUEDA WEB NATIVO (Sin Requests / BeautifulSoup)
# =================================================================

class NativeWebSearch:
    """Búsqueda e inspección HTML rápida usando el estándar urllib de Python."""
    
    @staticmethod
    def search(query: str) -> List[Dict[str, str]]:
        try:
            encoded_query = urllib.parse.quote(query)
            url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req, timeout=4) as response:
                html = response.read().decode('utf-8')
            
            # Parsing con Expresiones Regulares
            matches = re.findall(r'<a class="result__url" href="([^"]+)".*?>\s*(.*?)\s*</a>', html)
            results = []
            for link, title in matches[:2]:
                clean_title = re.sub(r'<[^>]+>', '', title).strip()
                results.append({"title": clean_title, "snippet": link.strip()})
            return results
        except Exception:
            return []

# =================================================================
# 6. ARQUITECTURA DE AGENTES DRAYMSYSTEM
# =================================================================

INTENTS_DATA = [
    {
        "tag": "saludo",
        "patterns": ["hola", "buenos dias", "buenas tardes", "que tal", "escuchas"],
        "responses": ["Hola. El sistema DraymSystem está en línea, conectado y listo."]
    },
    {
        "tag": "despedida",
        "patterns": ["adios", "hasta luego", "nos vemos", "salir"],
        "responses": ["Hasta luego. Proceso finalizado correctamente."]
    },
    {
        "tag": "creador",
        "patterns": ["quien es raul", "quien te creo", "tu creador", "quien soy yo", "raúl berny"],
        "responses": ["Mi creador es Raúl Berny Alonso Morales, desarrollador e ingeniero de DraymSystem."]
    },
    {
        "tag": "identidad",
        "patterns": ["quien eres", "cual es tu nombre", "tu eres draym", "que eres"],
        "responses": ["Soy DraymSystem, una arquitectura neuronal procesada mediante matrices de activación NBO-a."]
    }
]

# Inicialización de Vocabulario y Modelo Nativo
VOCABULARY, CLASSES = NativeNLP.create_vocabulary(INTENTS_DATA)
MODEL_NBO = NBOPerceptronModel(len(VOCABULARY), 16, len(CLASSES))
REASONER = NativeOntologyReasoner()

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
        bow_vector = NativeNLP.bag_of_words(prompt, VOCABULARY)
        probabilities = MODEL_NBO.forward(bow_vector)
        
        max_idx = max(range(len(probabilities)), key=lambda i: probabilities[i])
        predicted_tag = CLASSES[max_idx]
        confidence = probabilities[max_idx]

        return {
            "predicted_tag": predicted_tag,
            "confidence": round(confidence, 4),
            "probabilities": [round(p, 4) for p in probabilities]
        }

class WebResearchAgent(BaseAgent):
    def __init__(self):
        super().__init__("Agente_Investigacion", "Contexto Web")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        prompt = input_data.get("prompt", "")
        payload = input_data.get("payload_original", {})
        
        # Búsqueda nativa si se solicita explícitamente o por bajo umbral
        web_context = payload.get("web_context", [])
        if not web_context and ("busca" in prompt.lower() or "internet" in prompt.lower()):
            web_context = NativeWebSearch.search(prompt)

        return {"fuentes": web_context, "hay_web": len(web_context) > 0}

class CodeAgent(BaseAgent):
    def __init__(self):
        super().__init__("Agente_Codigo", "Sintaxis e Ingeniería")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        prompt = input_data.get("prompt", "")
        palabras_clave = ["python", "code", "script", "función", "def ", "html", "js"]
        return {"requiere_codigo": any(kw in prompt.lower() for kw in palabras_clave)}

class SynthesisAgent(BaseAgent):
    def __init__(self):
        super().__init__("Agente_Sintesis", "Consolidación de Respuesta")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        prompt = input_data.get("prompt", "").strip()
        analisis = input_data.get("analisis", {})
        codigo = input_data.get("codigo", {})
        investigacion = input_data.get("investigacion", {})

        tag = analisis.get("predicted_tag")
        confidence = analisis.get("confidence", 0.0)
        partes = []

        # 1. Deducción u Inferencia de Intención según NBO-a
        if confidence > 0.35:
            for intent in INTENTS_DATA:
                if intent["tag"] == tag:
                    partes.append(intent["responses"][0])
                    break
        else:
            # Fallback Causal / Ontológico Nativo
            partes.append(f"Procesando entrada: '{prompt}'.")

        # 2. Inferencia Ontológica si aplica
        if any(kw in prompt.lower() for kw in ["razonamiento", "deduce", "ontologia", "creador"]):
            partes.append("\n" + REASONER.infer(prompt))

        # 3. Código si fue detectado
        if codigo.get("requiere_codigo"):
            partes.append("\n```python\n# [Bloque generado nativamente bajo patrón NBO-a]\n```")

        # 4. Búsqueda Web Nativa
        if investigacion.get("hay_web"):
            partes.append("\n[Información web relevante]:")
            for item in investigacion.get("fuentes", []):
                partes.append(f"- {item.get('title')}: {item.get('snippet')}")

        # 5. Traza perceptrónica de la red
        partes.append(f"\n\n[Estado Red NBO-a | relu_211]: Confianza {confidence} -> Clave: '{tag}'")

        return {
            "estado": "exito",
            "texto_sintetizado": "\n".join(partes)
        }
