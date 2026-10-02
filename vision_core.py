import io
import base64
import numpy as np
from PIL import Image, ImageFilter
from config import DraymConfig

def relu_mod_211(S: float) -> float:
    """Activación modificada basada en la constante 211 (NBO-a)."""
    if S <= 0:
        return 0.0
    y1 = S * 0.5
    bound = DraymConfig.PERCEPTRON_211_BOUND
    return y1 * (bound / (bound + y1))

def procesar_matriz_rostro(img_pil: Image.Image, tono_r: float, tono_g: float, tono_b: float, brillo: float, contraste: float, suavizado: float) -> Image.Image:
    arr = np.array(img_pil, dtype=np.float32)

    factor_brillo = relu_mod_211(brillo) / 50.0
    factor_contraste = relu_mod_211(contraste) / 50.0

    arr = (arr - 128.0) * factor_contraste + 128.0 + (factor_brillo * 25.5)

    if tono_r != 100 or tono_g != 100 or tono_b != 100:
        arr[:, :, 0] *= (tono_r / 100.0)
        arr[:, :, 1] *= (tono_g / 100.0)
        arr[:, :, 2] *= (tono_b / 100.0)

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img_resultado = Image.fromarray(arr)

    if suavizado > 0:
        radio_blur = (suavizado / 100.0) * 3.0
        img_resultado = img_resultado.filter(ImageFilter.GaussianBlur(radius=radio_blur))

    return img_resultado

def process_base64_image(data_b64: str, params: dict) -> str:
    if "," in data_b64:
        data_b64 = data_b64.split(",")[1]
    
    img_bytes = base64.b64decode(data_b64)
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

    img_editada = procesar_matriz_rostro(
        img,
        float(params.get('tono_r', 100)),
        float(params.get('tono_g', 100)),
        float(params.get('tono_b', 100)),
        float(params.get('brillo', 50)),
        float(params.get('contraste', 50)),
        float(params.get('suavizado', 0))
    )

    buffered = io.BytesIO()
    img_editada.save(buffered, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode('utf-8')
