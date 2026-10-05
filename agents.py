"""
agents.py - Protocolo base para agentes independientes.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseAgente(ABC):
    """
    Clase base abstracta de la que deben heredar todos los agentes independientes.
    Cada agente puede definirse en un archivo 'agente_*.py' en la raíz.
    """
    INTENT_TAG: str = "generico"
    EJEMPLOS_ENTRENAMIENTO: List[str] = []

    @abstractmethod
    def ejecutar(self, prompt: str, contexto_global: Dict[str, Any]) -> str:
        """
        Ejecuta la lógica del agente utilizando el contexto compartido.
        """
        pass
