from orchestrator import AgentOrchestrator  # O la clase orquestadora que tengas

class GenerativeCoreEngine:
    def __init__(self):
        self.orchestrator = AgentOrchestrator()

    def generate(self, payload: dict) -> dict:
        user_prompt = payload.get("user_prompt", "")
        web_context = payload.get("web_context", [])
        learned_context = payload.get("learned_context", {})

        # Aquí tus agentes procesan el texto en lugar de repetirlo
        respuesta_procesada = self.orchestrator.run(
            prompt=user_prompt,
            context=web_context
        )

        return {
            "result": respuesta_procesada
        }
