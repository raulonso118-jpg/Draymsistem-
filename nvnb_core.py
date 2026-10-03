import math

# Pesos y sesgos entrenados NBO-a
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
    """Activación NBO-a sin operaciones exponenciales pesadas."""
    return x / (1.0 + 0.01 * x) if x > 0 else 0.01 * x

def predict_nbo(inputs: list) -> list:
    """Inferencia perceptrón de capa oculta y salida."""
    h = [relu_211(sum(inputs[i] * W1[i][j] for i in range(len(inputs))) + b1[j]) for j in range(len(b1))]
    o = [relu_211(sum(h[j] * W2[j][k] for j in range(len(h))) + b2[k]) for k in range(len(b2))]
    return o

def text_to_vector(text: str, dim: int = 4) -> list:
    """Mapeo de caracteres a vector flotante normalizado."""
    vector = [0.0] * dim
    for i, char in enumerate(text[:dim]):
        vector[i] = ord(char) / 255.0
    return vector
