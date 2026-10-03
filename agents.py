from typing import Dict, Any

# =================================================================
# NÚCLEO NBO-a INTEGRADO (Inferencia Matemática Directa)
# =================================================================
W1 = [
    [0.009827, -1.673184, -0.479324, 0.022902, -1.203937, -0.275713, -0.499627, -0.060934],
    [0.397949, -0.967759, 0.255584, 0.495538, 1.210088, -0.242116, -0.22057, -0.46945],
    [0.676937, -0.038244, -0.667513, 0.41029, 0.771993, -0.633136, 0.531934, 1.039185],
    [1.463662, -1.001581, -0.618165, 0.681863, 1.606629, -0.392373, 0.043346, -1.428108]
]
b1 = [-0.128037, 0.001638, -0.001312, -0.007377, 0.118432, 0.000127, 0.017772, -0.048603]
W2 = [
    [0.486183, 0.418387], [0.316011, -0.630265], [-0.262076, 0.518192], [0.360306, 0.137607],
    [-0.302119, -0.714048], [-0.167102, -0.211522], [-0.187807, 0.061643], [-0.076087, 0.2535]
]
b2 = [0.151399, -0.178274]

def relu_211(x: float) -> float:
    return x / (1.0 + 0.01 * x) if x > 0 else 0.01 * x

def predict_nbo(inputs: list) -> list:
    h = [relu_211(sum(inputs[i] * W1[i][j] for i in range(len(inputs))) + b1[j]) for j in range(len(b1))]
    o = [relu_211(sum(h[j] * W2[j][k] for j in range(len(h))) + b2[k]) for k in range(len(b2))]
    return o

def text_to_vector(text: str, dim: int = 4) -> list:
    vector = [0.0] * dim
    for i, char in enumerate(text[:dim]):
        vector[i] = ord(char) / 255.0
    return vector

# =================================================================
# ARQUITECTURA DE AGENTES NBO-a
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
        payload = input_data.get("payload_original", input_data)
        learned = payload.get("learned_context", {})
        return {"conceptos_cargados": len(learned)}

class AnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__("Agente_Analisis", "Evaluación Perceptronal")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        prompt = input_data.get("prompt", "")
        vector_in = text_to_vector(prompt, dim=4)
        activaciones = predict_nbo(vector_in)
        return {
            "vector_entrada": vector_in,
            "activaciones_nbo": [round(x, 4) for x in activaciones]
        }

class WebResearchAgent(BaseAgent):
    def __init__(self):
        super().__init__("Agente_Investigacion", "Contexto Web")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        payload = input_data.get("payload_original", {})
        web_context = payload.get("web_context", [])
        return {"fuentes": web_context, "hay_web": len(web_context) > 0}

class CodeAgent(BaseAgent):
    def __init__(self):
        super().__init__("Agente_Codigo", "Sintaxis e Ingeniería")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        prompt = input_data.get("prompt", "")
        palabras_clave = ["python", "code", "script", "función", "def ", "html", "js", "css"]
        es_codigo = any(kw in prompt.lower() for kw in palabras_clave)
        return {"requiere_codigo": es_codigo}

class SynthesisAgent(BaseAgent):
    def __init__(self):
        super().__init__("Agente_Sintesis", "Consolidación de Respuesta")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        prompt = input_data.get("prompt", "").strip()
        analisis = input_data.get("analisis", {})
        codigo = input_data.get("codigo", {})
        investigacion = input_data.get("investigacion", {})

        salida_red = analisis.get("activaciones_nbo", [0.0, 0.0])
        partes = []

        # Respuesta conversacional según el prompt
        prompt_lower = prompt.lower()
        if any(saludo in prompt_lower for saludo in ["hola", "buenas", "saludos", "escuchas"]):
            partes.append("Hola. El sistema DraymSystem está en línea, conectado y listo.")
        elif "quien eres" in prompt_lower or "tu nombre" in prompt_lower or "tu eres" in prompt_lower:
            partes.append("Soy DraymSystem, una arquitectura neuronal procesada mediante matrices de activación NBO-a.")
        else:
            partes.append(f"Procesando entrada: '{prompt}'.")

        # Bloque de código si se solicita
        if codigo.get("requiere_codigo"):
            partes.append("\n```python\n# [Bloque generado bajo patrón perceptrónico NBO-a]\n```")

        # Fuentes web si están activas
        if investigacion.get("hay_web"):
            partes.append("\n[Información web relevante]:")
            for item in investigacion.get("fuentes", []):
                partes.append(f"- {item.get('title')}: {item.get('snippet')}")

        # Activación matemática de la red
        partes.append(f"\n\n[Estado Red NBO-a | relu_211]: {salida_red}")

        return {
            "estado": "exito",
            "texto_sintetizado": "\n".join(partes)
        }
